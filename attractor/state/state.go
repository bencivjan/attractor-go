// Package state provides thread-safe execution context, checkpointing, and
// artifact storage for the Attractor pipeline engine.
//
// Context is the primary data-sharing mechanism between pipeline stages.
// Checkpoint enables crash recovery by serialising execution state to disk.
// ArtifactStore provides named, typed storage for large stage outputs that
// should not be inlined into the context.
package state

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// StageStatus
// ---------------------------------------------------------------------------

// StageStatus represents the result of executing a node handler.
type StageStatus string

const (
	// StatusSuccess indicates the stage completed without error.
	StatusSuccess StageStatus = "success"
	// StatusPartialSuccess indicates the stage completed but with degraded results.
	StatusPartialSuccess StageStatus = "partial_success"
	// StatusRetry indicates the stage should be retried.
	StatusRetry StageStatus = "retry"
	// StatusFail indicates the stage failed permanently.
	StatusFail StageStatus = "fail"
	// StatusSkipped indicates the stage was not executed.
	StatusSkipped StageStatus = "skipped"
)

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

// Outcome is the result of executing a node handler. It carries the status,
// routing hints, and optional context mutations that the engine uses to
// determine the next pipeline step.
type Outcome struct {
	Status           StageStatus    `json:"status"`
	PreferredLabel   string         `json:"preferred_label,omitempty"`
	SuggestedNextIDs []string       `json:"suggested_next_ids,omitempty"`
	ContextUpdates   map[string]any `json:"context_updates,omitempty"`
	Notes            string         `json:"notes,omitempty"`
	FailureReason    string         `json:"failure_reason,omitempty"`
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

// Context is a thread-safe key-value store shared across pipeline stages.
// All reads and writes are serialised via a read-write mutex so that
// parallel branches can safely share a single Context instance (or work
// with a Clone).
type Context struct {
	mu     sync.RWMutex
	values map[string]any
	logs   []string
}

// NewContext returns an initialised, empty Context.
func NewContext() *Context {
	return &Context{
		values: make(map[string]any),
	}
}

// Set stores a value under the given key.
func (c *Context) Set(key string, value any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values[key] = value
}

// Get retrieves a value by key. Returns nil when the key is absent.
func (c *Context) Get(key string) any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.values[key]
}

// GetString returns the string value for key or def if the key is missing
// or not a string.
func (c *Context) GetString(key string, def string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.values[key]
	if !ok {
		return def
	}
	s, ok := v.(string)
	if !ok {
		return def
	}
	return s
}

// GetInt returns the int value for key or def if the key is missing or not
// an int. It also recognises float64 (common after JSON round-tripping) and
// json.Number, converting them losslessly when possible.
func (c *Context) GetInt(key string, def int) int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.values[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		// Only convert when the float represents a whole number.
		if n == float64(int(n)) {
			return int(n)
		}
		return def
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return def
		}
		return int(i)
	default:
		return def
	}
}

// AppendLog adds a timestamped entry to the context log.
func (c *Context) AppendLog(entry string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.logs = append(c.logs, entry)
}

// Snapshot returns a shallow, serialisable copy of all context values.
// The caller must not mutate the returned map.
func (c *Context) Snapshot() map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	snap := make(map[string]any, len(c.values))
	for k, v := range c.values {
		snap[k] = v
	}
	return snap
}

// Clone returns a deep copy of the Context suitable for use in a parallel
// branch. Values are deep-copied via a JSON round-trip so that mutations
// in the clone do not affect the original.
func (c *Context) Clone() *Context {
	c.mu.RLock()
	defer c.mu.RUnlock()

	cloned := &Context{
		values: make(map[string]any, len(c.values)),
		logs:   make([]string, len(c.logs)),
	}

	// Deep copy values through JSON to break shared references.
	if len(c.values) > 0 {
		data, err := json.Marshal(c.values)
		if err == nil {
			_ = json.Unmarshal(data, &cloned.values)
		} else {
			// Fallback: shallow copy if marshalling fails.
			for k, v := range c.values {
				cloned.values[k] = v
			}
		}
	}

	copy(cloned.logs, c.logs)
	return cloned
}

// Clear removes all mutable context values and logs, resetting the context
// to an empty state. Used by loop_restart to reset run state (Section 3.2).
func (c *Context) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values = make(map[string]any)
	c.logs = nil
}

// ApplyUpdates merges the given key-value pairs into the context. Existing
// keys are overwritten.
func (c *Context) ApplyUpdates(updates map[string]any) {
	if len(updates) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, v := range updates {
		c.values[k] = v
	}
}

// Logs returns a copy of the accumulated log entries.
func (c *Context) Logs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]string, len(c.logs))
	copy(out, c.logs)
	return out
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

// Checkpoint is a serialisable snapshot of pipeline execution state that
// enables crash recovery. It captures which node is currently executing,
// which nodes have already completed, retry counts, and a copy of the
// context values and logs.
type Checkpoint struct {
	Timestamp      time.Time      `json:"timestamp"`
	CurrentNode    string         `json:"current_node"`
	CompletedNodes []string       `json:"completed_nodes"`
	NodeRetries    map[string]int `json:"node_retries"`
	ContextValues  map[string]any `json:"context"`
	Logs           []string       `json:"logs"`
}

// Save serialises the checkpoint to a JSON file at the given path. Parent
// directories are created as needed. The file is written atomically by
// first writing to a temporary file and then renaming.
func (cp *Checkpoint) Save(path string) error {
	if path == "" {
		return fmt.Errorf("checkpoint save: path must not be empty")
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("checkpoint save: create directory: %w", err)
	}

	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return fmt.Errorf("checkpoint save: marshal: %w", err)
	}

	// Write to a temporary file in the same directory, then rename for
	// atomic replacement.
	tmp, err := os.CreateTemp(dir, ".checkpoint-*.tmp")
	if err != nil {
		return fmt.Errorf("checkpoint save: create temp file: %w", err)
	}
	tmpName := tmp.Name()

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("checkpoint save: write: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("checkpoint save: close: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("checkpoint save: rename: %w", err)
	}
	return nil
}

// LoadCheckpoint reads and deserialises a checkpoint from a JSON file.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load checkpoint: read: %w", err)
	}
	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, fmt.Errorf("load checkpoint: unmarshal: %w", err)
	}
	if cp.NodeRetries == nil {
		cp.NodeRetries = make(map[string]int)
	}
	if cp.ContextValues == nil {
		cp.ContextValues = make(map[string]any)
	}
	return &cp, nil
}

// ---------------------------------------------------------------------------
// ArtifactStore
// ---------------------------------------------------------------------------

// artifactEntry is the internal representation of a stored artifact.
type artifactEntry struct {
	info ArtifactInfo
	data any
}

// ArtifactInfo describes a stored artifact without exposing its payload.
type ArtifactInfo struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	SizeBytes    int       `json:"size_bytes"`
	StoredAt     time.Time `json:"stored_at"`
	IsFileBacked bool      `json:"is_file_backed"`
}

// ArtifactStore provides named, typed storage for large stage outputs.
// Artifacts are keyed by a unique ID and may optionally be backed by a
// file on disk (when baseDir is non-empty and the data is a byte slice or
// string large enough to justify spilling).
type ArtifactStore struct {
	mu        sync.RWMutex
	artifacts map[string]*artifactEntry
	baseDir   string
}

// fileSizeThreshold is the minimum payload size (in bytes) before an
// artifact is automatically persisted to disk (Section 5.5).
const fileSizeThreshold = 100 * 1024 // 100 KB

// NewArtifactStore creates an ArtifactStore. When baseDir is non-empty,
// large artifacts are automatically persisted to that directory.
func NewArtifactStore(baseDir string) *ArtifactStore {
	return &ArtifactStore{
		artifacts: make(map[string]*artifactEntry),
		baseDir:   baseDir,
	}
}

// Store adds or replaces an artifact. The data can be any serialisable
// value. When the store has a baseDir configured and the data is a []byte
// or string exceeding fileSizeThreshold, it is written to disk and only
// the file reference is kept in memory.
func (s *ArtifactStore) Store(id, name string, data any) (*ArtifactInfo, error) {
	if id == "" {
		return nil, fmt.Errorf("artifact store: id must not be empty")
	}

	now := time.Now().UTC()
	info := ArtifactInfo{
		ID:       id,
		Name:     name,
		StoredAt: now,
	}

	var storedData any = data

	// Attempt to compute size and optionally spill to disk.
	raw := toBytes(data)
	if raw != nil {
		info.SizeBytes = len(raw)

		if s.baseDir != "" && len(raw) >= fileSizeThreshold {
			if err := os.MkdirAll(s.baseDir, 0o755); err != nil {
				return nil, fmt.Errorf("artifact store: create dir: %w", err)
			}
			path := filepath.Join(s.baseDir, id)
			if err := os.WriteFile(path, raw, 0o644); err != nil {
				return nil, fmt.Errorf("artifact store: write file: %w", err)
			}
			info.IsFileBacked = true
			storedData = nil // free memory; data lives on disk
		}
	} else {
		// For non-byte types, estimate size via JSON marshalling.
		if j, err := json.Marshal(data); err == nil {
			info.SizeBytes = len(j)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.artifacts[id] = &artifactEntry{info: info, data: storedData}
	return &info, nil
}

// Retrieve returns the data for the given artifact ID. If the artifact is
// file-backed, the data is read from disk and returned as a []byte.
func (s *ArtifactStore) Retrieve(id string) (any, error) {
	s.mu.RLock()
	entry, ok := s.artifacts[id]
	s.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("artifact store: artifact %q not found", id)
	}

	if entry.info.IsFileBacked {
		path := filepath.Join(s.baseDir, id)
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("artifact store: read file-backed artifact %q: %w", id, err)
		}
		return data, nil
	}

	return entry.data, nil
}

// Has returns true when an artifact with the given ID is present.
func (s *ArtifactStore) Has(id string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.artifacts[id]
	return ok
}

// List returns metadata for all stored artifacts, sorted by ID.
func (s *ArtifactStore) List() []*ArtifactInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	infos := make([]*ArtifactInfo, 0, len(s.artifacts))
	for _, e := range s.artifacts {
		info := e.info // copy
		infos = append(infos, &info)
	}
	sort.Slice(infos, func(i, j int) bool {
		return infos[i].ID < infos[j].ID
	})
	return infos
}

// Remove deletes an artifact by ID. If the artifact is file-backed, the
// backing file is also removed. No-op if the ID does not exist.
func (s *ArtifactStore) Remove(id string) {
	s.mu.Lock()
	entry, ok := s.artifacts[id]
	if ok {
		delete(s.artifacts, id)
	}
	s.mu.Unlock()

	if ok && entry.info.IsFileBacked && s.baseDir != "" {
		os.Remove(filepath.Join(s.baseDir, id))
	}
}

// Clear removes all artifacts and their backing files.
func (s *ArtifactStore) Clear() {
	s.mu.Lock()
	old := s.artifacts
	s.artifacts = make(map[string]*artifactEntry)
	s.mu.Unlock()

	// Clean up any file-backed artifacts outside the lock.
	if s.baseDir != "" {
		for id, entry := range old {
			if entry.info.IsFileBacked {
				os.Remove(filepath.Join(s.baseDir, id))
			}
		}
	}
}

// toBytes converts data to a byte slice if possible, returning nil
// otherwise.
func toBytes(data any) []byte {
	switch v := data.(type) {
	case []byte:
		return v
	case string:
		return []byte(v)
	default:
		return nil
	}
}
