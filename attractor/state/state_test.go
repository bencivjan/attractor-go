package state

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Context: Set / Get / GetString / GetInt
// ---------------------------------------------------------------------------

func TestContext_SetAndGet(t *testing.T) {
	ctx := NewContext()
	ctx.Set("key", "value")
	got := ctx.Get("key")
	if got != "value" {
		t.Errorf("Get('key') = %v, want 'value'", got)
	}
}

func TestContext_GetMissingKey(t *testing.T) {
	ctx := NewContext()
	if got := ctx.Get("missing"); got != nil {
		t.Errorf("Get('missing') = %v, want nil", got)
	}
}

func TestContext_GetString(t *testing.T) {
	ctx := NewContext()
	ctx.Set("name", "alice")
	ctx.Set("count", 42)

	if got := ctx.GetString("name", "default"); got != "alice" {
		t.Errorf("GetString('name') = %q, want %q", got, "alice")
	}
	if got := ctx.GetString("missing", "default"); got != "default" {
		t.Errorf("GetString('missing') = %q, want %q", got, "default")
	}
	// Non-string value returns default.
	if got := ctx.GetString("count", "default"); got != "default" {
		t.Errorf("GetString('count') should return default for non-string, got %q", got)
	}
}

func TestContext_GetInt(t *testing.T) {
	tests := []struct {
		name string
		val  any
		want int
		def  int
	}{
		{"int value", 42, 42, 0},
		{"int64 value", int64(99), 99, 0},
		{"float64 whole", float64(10), 10, 0},
		{"float64 fractional", float64(3.14), 0, 0},
		{"json.Number", json.Number("77"), 77, 0},
		{"string value", "not_int", 0, 0},
		{"missing key", nil, -1, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := NewContext()
			if tt.val != nil {
				ctx.Set("k", tt.val)
			}
			got := ctx.GetInt("k", tt.def)
			if got != tt.want {
				t.Errorf("GetInt() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestContext_SetOverwrite(t *testing.T) {
	ctx := NewContext()
	ctx.Set("k", "first")
	ctx.Set("k", "second")
	if got := ctx.Get("k"); got != "second" {
		t.Errorf("Get('k') = %v, want 'second'", got)
	}
}

// ---------------------------------------------------------------------------
// Context: Clone isolation
// ---------------------------------------------------------------------------

func TestContext_CloneIsolation(t *testing.T) {
	ctx := NewContext()
	ctx.Set("shared", "original")
	ctx.Set("nested", map[string]any{"inner": "value"})
	ctx.AppendLog("log entry")

	clone := ctx.Clone()

	// Clone should have the same values.
	if got := clone.GetString("shared", ""); got != "original" {
		t.Errorf("cloned GetString('shared') = %q, want %q", got, "original")
	}

	// Mutating clone should not affect the original.
	clone.Set("shared", "modified")
	if got := ctx.GetString("shared", ""); got != "original" {
		t.Errorf("original changed after clone mutation: got %q", got)
	}

	// Mutating original should not affect the clone.
	ctx.Set("shared", "re-original")
	if got := clone.GetString("shared", ""); got != "modified" {
		t.Errorf("clone changed after original mutation: got %q", got)
	}

	// Logs should be copied.
	cloneLogs := clone.Logs()
	if len(cloneLogs) != 1 || cloneLogs[0] != "log entry" {
		t.Errorf("clone logs = %v, want [log entry]", cloneLogs)
	}

	// Appending to clone logs should not affect original.
	clone.AppendLog("clone only")
	if len(ctx.Logs()) != 1 {
		t.Errorf("original logs affected by clone AppendLog: %v", ctx.Logs())
	}
}

// ---------------------------------------------------------------------------
// Context: Snapshot
// ---------------------------------------------------------------------------

func TestContext_Snapshot(t *testing.T) {
	ctx := NewContext()
	ctx.Set("a", 1)
	ctx.Set("b", "two")

	snap := ctx.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("Snapshot len = %d, want 2", len(snap))
	}
	if snap["a"] != 1 {
		t.Errorf("snap['a'] = %v, want 1", snap["a"])
	}
	if snap["b"] != "two" {
		t.Errorf("snap['b'] = %v, want 'two'", snap["b"])
	}

	// Mutating the context after snapshot should not affect snap.
	ctx.Set("a", 999)
	if snap["a"] != 1 {
		t.Errorf("snapshot mutated after context change: snap['a'] = %v", snap["a"])
	}
}

// ---------------------------------------------------------------------------
// Context: ApplyUpdates
// ---------------------------------------------------------------------------

func TestContext_ApplyUpdates(t *testing.T) {
	ctx := NewContext()
	ctx.Set("existing", "old")

	ctx.ApplyUpdates(map[string]any{
		"existing": "new",
		"added":    42,
	})

	if got := ctx.GetString("existing", ""); got != "new" {
		t.Errorf("existing = %q, want %q", got, "new")
	}
	if got := ctx.GetInt("added", 0); got != 42 {
		t.Errorf("added = %d, want 42", got)
	}
}

func TestContext_ApplyUpdates_Nil(t *testing.T) {
	ctx := NewContext()
	ctx.Set("k", "v")
	ctx.ApplyUpdates(nil)
	// Should be a no-op.
	if got := ctx.GetString("k", ""); got != "v" {
		t.Errorf("unexpected change after nil ApplyUpdates: got %q", got)
	}
}

func TestContext_ApplyUpdates_Empty(t *testing.T) {
	ctx := NewContext()
	ctx.Set("k", "v")
	ctx.ApplyUpdates(map[string]any{})
	if got := ctx.GetString("k", ""); got != "v" {
		t.Errorf("unexpected change after empty ApplyUpdates: got %q", got)
	}
}

// ---------------------------------------------------------------------------
// Context: AppendLog and Logs
// ---------------------------------------------------------------------------

func TestContext_AppendLog(t *testing.T) {
	ctx := NewContext()
	ctx.AppendLog("first")
	ctx.AppendLog("second")

	logs := ctx.Logs()
	if len(logs) != 2 {
		t.Fatalf("Logs() len = %d, want 2", len(logs))
	}
	if logs[0] != "first" || logs[1] != "second" {
		t.Errorf("Logs() = %v, want [first, second]", logs)
	}
}

func TestContext_LogsReturnsCopy(t *testing.T) {
	ctx := NewContext()
	ctx.AppendLog("entry")

	logs := ctx.Logs()
	logs[0] = "tampered"
	// Original should be unchanged.
	if ctx.Logs()[0] != "entry" {
		t.Error("Logs() returned a reference, not a copy")
	}
}

func TestContext_LogsEmpty(t *testing.T) {
	ctx := NewContext()
	logs := ctx.Logs()
	if len(logs) != 0 {
		t.Errorf("Logs() on new context = %v, want empty", logs)
	}
}

// ---------------------------------------------------------------------------
// Checkpoint: Save / LoadCheckpoint round-trip
// ---------------------------------------------------------------------------

func TestCheckpoint_SaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	now := time.Now().UTC().Truncate(time.Second)
	cp := &Checkpoint{
		Timestamp:      now,
		CurrentNode:    "coder",
		CompletedNodes: []string{"start", "planner"},
		NodeRetries:    map[string]int{"coder": 3},
		ContextValues:  map[string]any{"lang": "go", "count": float64(42)},
		Logs:           []string{"step 1 done", "step 2 done"},
	}

	if err := cp.Save(path); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint error: %v", err)
	}

	if loaded.CurrentNode != "coder" {
		t.Errorf("CurrentNode = %q, want %q", loaded.CurrentNode, "coder")
	}
	if len(loaded.CompletedNodes) != 2 {
		t.Errorf("CompletedNodes len = %d, want 2", len(loaded.CompletedNodes))
	}
	if loaded.NodeRetries["coder"] != 3 {
		t.Errorf("NodeRetries[coder] = %d, want 3", loaded.NodeRetries["coder"])
	}
	if loaded.ContextValues["lang"] != "go" {
		t.Errorf("ContextValues[lang] = %v, want 'go'", loaded.ContextValues["lang"])
	}
	if len(loaded.Logs) != 2 {
		t.Errorf("Logs len = %d, want 2", len(loaded.Logs))
	}
}

func TestCheckpoint_SaveCreatesDirectories(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "deep", "checkpoint.json")

	cp := &Checkpoint{
		Timestamp:   time.Now().UTC(),
		CurrentNode: "test",
		NodeRetries: map[string]int{},
	}

	if err := cp.Save(path); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("Save did not create the file")
	}
}

func TestCheckpoint_SaveEmptyPath(t *testing.T) {
	cp := &Checkpoint{Timestamp: time.Now()}
	err := cp.Save("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestCheckpoint_AtomicWrite(t *testing.T) {
	// Verify the file is written atomically (no partial writes observed).
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	cp := &Checkpoint{
		Timestamp:     time.Now().UTC(),
		CurrentNode:   "node_a",
		NodeRetries:   map[string]int{},
		ContextValues: map[string]any{"key": "value"},
	}
	if err := cp.Save(path); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// Read the file and verify it is valid JSON.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile error: %v", err)
	}
	var loaded Checkpoint
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("Unmarshal error (file may be partially written): %v", err)
	}
	if loaded.CurrentNode != "node_a" {
		t.Errorf("CurrentNode = %q after atomic write", loaded.CurrentNode)
	}

	// No temp files should remain.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".checkpoint-") {
			t.Errorf("temp file left behind: %s", e.Name())
		}
	}
}

func TestLoadCheckpoint_NilMapsInitialised(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	// Write a minimal JSON without node_retries and context.
	data := []byte(`{"timestamp":"2024-01-01T00:00:00Z","current_node":"x","completed_nodes":["a"]}`)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}

	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint error: %v", err)
	}
	if loaded.NodeRetries == nil {
		t.Error("NodeRetries should be initialized to empty map, not nil")
	}
	if loaded.ContextValues == nil {
		t.Error("ContextValues should be initialized to empty map, not nil")
	}
}

func TestLoadCheckpoint_FileNotFound(t *testing.T) {
	_, err := LoadCheckpoint("/nonexistent/path/checkpoint.json")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadCheckpoint_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}
	_, err := LoadCheckpoint(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// ---------------------------------------------------------------------------
// ArtifactStore: basic CRUD
// ---------------------------------------------------------------------------

func TestArtifactStore_StoreAndRetrieve(t *testing.T) {
	store := NewArtifactStore("")
	info, err := store.Store("art1", "My Artifact", "hello data")
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	if info.ID != "art1" {
		t.Errorf("info.ID = %q, want %q", info.ID, "art1")
	}
	if info.Name != "My Artifact" {
		t.Errorf("info.Name = %q, want %q", info.Name, "My Artifact")
	}
	if info.SizeBytes != len("hello data") {
		t.Errorf("info.SizeBytes = %d, want %d", info.SizeBytes, len("hello data"))
	}
	if info.IsFileBacked {
		t.Error("should not be file-backed for in-memory store")
	}

	data, err := store.Retrieve("art1")
	if err != nil {
		t.Fatalf("Retrieve error: %v", err)
	}
	if s, ok := data.(string); !ok || s != "hello data" {
		t.Errorf("Retrieve = %v, want 'hello data'", data)
	}
}

func TestArtifactStore_StoreByteSlice(t *testing.T) {
	store := NewArtifactStore("")
	payload := []byte("byte content")
	info, err := store.Store("b1", "Bytes", payload)
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	if info.SizeBytes != len(payload) {
		t.Errorf("SizeBytes = %d, want %d", info.SizeBytes, len(payload))
	}

	data, err := store.Retrieve("b1")
	if err != nil {
		t.Fatalf("Retrieve error: %v", err)
	}
	if b, ok := data.([]byte); !ok || string(b) != "byte content" {
		t.Errorf("Retrieve = %v, want byte content", data)
	}
}

func TestArtifactStore_StoreStructData(t *testing.T) {
	store := NewArtifactStore("")
	val := map[string]any{"key": "value"}
	info, err := store.Store("s1", "Struct", val)
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	// Size is estimated via JSON marshalling.
	if info.SizeBytes == 0 {
		t.Error("SizeBytes should be non-zero for struct data")
	}
}

func TestArtifactStore_EmptyID(t *testing.T) {
	store := NewArtifactStore("")
	_, err := store.Store("", "test", "data")
	if err == nil {
		t.Fatal("expected error for empty ID")
	}
}

func TestArtifactStore_Has(t *testing.T) {
	store := NewArtifactStore("")
	store.Store("x", "X", "data")

	if !store.Has("x") {
		t.Error("Has('x') should return true")
	}
	if store.Has("y") {
		t.Error("Has('y') should return false")
	}
}

func TestArtifactStore_List(t *testing.T) {
	store := NewArtifactStore("")
	store.Store("b", "B", "data")
	store.Store("a", "A", "data")
	store.Store("c", "C", "data")

	infos := store.List()
	if len(infos) != 3 {
		t.Fatalf("List() len = %d, want 3", len(infos))
	}
	// List should be sorted by ID.
	if infos[0].ID != "a" || infos[1].ID != "b" || infos[2].ID != "c" {
		t.Errorf("List() order: %s, %s, %s; want a, b, c",
			infos[0].ID, infos[1].ID, infos[2].ID)
	}
}

func TestArtifactStore_Remove(t *testing.T) {
	store := NewArtifactStore("")
	store.Store("r1", "R", "data")

	store.Remove("r1")
	if store.Has("r1") {
		t.Error("Has('r1') should be false after Remove")
	}

	// Remove non-existent should be no-op.
	store.Remove("nonexistent")
}

func TestArtifactStore_Clear(t *testing.T) {
	store := NewArtifactStore("")
	store.Store("c1", "C1", "data")
	store.Store("c2", "C2", "data")

	store.Clear()
	if len(store.List()) != 0 {
		t.Error("List() should be empty after Clear")
	}
}

func TestArtifactStore_RetrieveNotFound(t *testing.T) {
	store := NewArtifactStore("")
	_, err := store.Retrieve("nonexistent")
	if err == nil {
		t.Fatal("expected error for non-existent artifact")
	}
}

func TestArtifactStore_Replace(t *testing.T) {
	store := NewArtifactStore("")
	store.Store("r", "Name1", "first")
	store.Store("r", "Name2", "second")

	data, err := store.Retrieve("r")
	if err != nil {
		t.Fatalf("Retrieve error: %v", err)
	}
	if s, ok := data.(string); !ok || s != "second" {
		t.Errorf("Retrieve after replace = %v, want 'second'", data)
	}
}

// ---------------------------------------------------------------------------
// ArtifactStore: file-backed spilling for large payloads
// ---------------------------------------------------------------------------

func TestArtifactStore_FileBackedSpilling(t *testing.T) {
	dir := t.TempDir()
	store := NewArtifactStore(dir)

	// Create a payload larger than fileSizeThreshold (1 MiB).
	large := strings.Repeat("x", fileSizeThreshold+1)

	info, err := store.Store("big", "Big Artifact", large)
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	if !info.IsFileBacked {
		t.Error("large artifact should be file-backed")
	}
	if info.SizeBytes != len(large) {
		t.Errorf("SizeBytes = %d, want %d", info.SizeBytes, len(large))
	}

	// Verify the file exists on disk.
	filePath := filepath.Join(dir, "big")
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		t.Error("file-backed artifact file should exist on disk")
	}

	// Retrieve should read from disk.
	data, err := store.Retrieve("big")
	if err != nil {
		t.Fatalf("Retrieve error: %v", err)
	}
	b, ok := data.([]byte)
	if !ok {
		t.Fatalf("Retrieve returned type %T, want []byte", data)
	}
	if len(b) != len(large) {
		t.Errorf("Retrieve returned wrong data length: got %d, want %d", len(b), len(large))
	}
}

func TestArtifactStore_FileBackedRemove(t *testing.T) {
	dir := t.TempDir()
	store := NewArtifactStore(dir)

	large := strings.Repeat("y", fileSizeThreshold+1)
	store.Store("big2", "Big2", large)

	filePath := filepath.Join(dir, "big2")
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		t.Fatal("file should exist before Remove")
	}

	store.Remove("big2")
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Error("file should be removed after Remove")
	}
}

func TestArtifactStore_FileBackedClear(t *testing.T) {
	dir := t.TempDir()
	store := NewArtifactStore(dir)

	large := strings.Repeat("z", fileSizeThreshold+1)
	store.Store("big3", "Big3", large)

	store.Clear()

	filePath := filepath.Join(dir, "big3")
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Error("file should be removed after Clear")
	}
}

func TestArtifactStore_SmallPayloadNotFileBacked(t *testing.T) {
	dir := t.TempDir()
	store := NewArtifactStore(dir)

	info, err := store.Store("small", "Small", "tiny payload")
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	if info.IsFileBacked {
		t.Error("small payload should not be file-backed")
	}
}

func TestArtifactStore_FileBackedByteSlice(t *testing.T) {
	dir := t.TempDir()
	store := NewArtifactStore(dir)

	large := make([]byte, fileSizeThreshold+1)
	for i := range large {
		large[i] = 'A'
	}

	info, err := store.Store("bytes", "LargeBytes", large)
	if err != nil {
		t.Fatalf("Store error: %v", err)
	}
	if !info.IsFileBacked {
		t.Error("large byte slice should be file-backed")
	}

	data, err := store.Retrieve("bytes")
	if err != nil {
		t.Fatalf("Retrieve error: %v", err)
	}
	if b, ok := data.([]byte); !ok || len(b) != len(large) {
		t.Error("Retrieve returned incorrect data")
	}
}
