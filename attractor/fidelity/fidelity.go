// Package fidelity implements context fidelity control for the Attractor
// pipeline engine (spec Section 5.4). Fidelity modes govern how much prior
// conversation and state is carried into the next node's LLM session,
// enabling control over context window usage across multi-stage pipelines.
//
// The package provides three core operations:
//   - ResolveFidelity: determines the effective fidelity mode from edge, node,
//     and graph attributes using a defined precedence order.
//   - ResolveThreadID: determines the thread key for session reuse when the
//     fidelity mode is "full".
//   - ApplyFidelity: transforms a context snapshot according to the resolved
//     fidelity mode, producing a filtered view suitable for the next handler.
package fidelity

import (
	"fmt"
	"sort"
	"strings"

	"github.com/strongdm/attractor-go/attractor/graph"
)

// ---------------------------------------------------------------------------
// Fidelity modes
// ---------------------------------------------------------------------------

// Mode represents a context fidelity level.
type Mode string

const (
	// Full preserves the complete conversation history by reusing the same
	// LLM session (thread). Context is returned as-is.
	Full Mode = "full"

	// Truncate returns only essential keys: graph.goal, current_node,
	// last_stage, and outcome. Minimal token usage.
	Truncate Mode = "truncate"

	// Compact returns all keys but truncates long values (>500 chars)
	// with a trailing "..." marker.
	Compact Mode = "compact"

	// SummaryLow keeps only the 5 most recent log entries and truncates
	// values longer than 200 characters. Approximately 600 tokens.
	SummaryLow Mode = "summary:low"

	// SummaryMedium keeps the 10 most recent log entries and truncates
	// values longer than 500 characters. Approximately 1500 tokens.
	SummaryMedium Mode = "summary:medium"

	// SummaryHigh keeps the 20 most recent log entries and truncates
	// values longer than 1000 characters. Approximately 3000 tokens.
	SummaryHigh Mode = "summary:high"
)

// DefaultMode is the fallback fidelity mode when no explicit fidelity is
// configured on the edge, node, or graph.
const DefaultMode = Compact

// validModes enumerates all recognized fidelity mode strings for validation.
var validModes = map[Mode]bool{
	Full:          true,
	Truncate:      true,
	Compact:       true,
	SummaryLow:    true,
	SummaryMedium: true,
	SummaryHigh:   true,
}

// IsValid reports whether m is a recognized fidelity mode.
func (m Mode) IsValid() bool {
	return validModes[m]
}

// ParseMode converts a raw string to a Mode. Returns (mode, true) on success
// or (DefaultMode, false) if the string is not a valid fidelity mode.
func ParseMode(raw string) (Mode, bool) {
	m := Mode(strings.TrimSpace(strings.ToLower(raw)))
	if m == "" {
		return DefaultMode, false
	}
	if validModes[m] {
		return m, true
	}
	return DefaultMode, false
}

// ---------------------------------------------------------------------------
// Truncation limits per summary level
// ---------------------------------------------------------------------------

// summaryConfig holds the parameters that vary between summary sub-modes.
type summaryConfig struct {
	maxLogs     int
	maxValueLen int
}

var summaryConfigs = map[Mode]summaryConfig{
	SummaryLow:    {maxLogs: 5, maxValueLen: 200},
	SummaryMedium: {maxLogs: 10, maxValueLen: 500},
	SummaryHigh:   {maxLogs: 20, maxValueLen: 1000},
}

// compactMaxValueLen is the truncation threshold for compact mode.
const compactMaxValueLen = 500

// truncateEssentialKeys are the only keys preserved under the truncate mode.
var truncateEssentialKeys = map[string]bool{
	"goal":         true,
	"current_node": true,
	"last_stage":   true,
	"outcome":      true,
}

// ---------------------------------------------------------------------------
// ResolveFidelity
// ---------------------------------------------------------------------------

// ResolveFidelity determines the effective fidelity mode for a target node
// by walking the precedence chain defined in spec Section 5.4:
//
//  1. Edge fidelity attribute (on the incoming edge)
//  2. Target node fidelity attribute
//  3. Graph default_fidelity attribute
//  4. Default: compact
//
// The edge parameter may be nil when no edge context is available (e.g. the
// start node).
func ResolveFidelity(edge *graph.Edge, targetNode *graph.Node, g *graph.Graph) Mode {
	// Step 1: Edge-level fidelity (highest precedence).
	if edge != nil {
		if f := edge.Fidelity(); f != "" {
			if m, ok := ParseMode(f); ok {
				return m
			}
		}
	}

	// Step 2: Target node fidelity.
	if targetNode != nil {
		if f := targetNode.Fidelity(); f != "" {
			if m, ok := ParseMode(f); ok {
				return m
			}
		}
	}

	// Step 3: Graph default_fidelity.
	if g != nil {
		if f := g.DefaultFidelity(); f != "" {
			if m, ok := ParseMode(f); ok {
				return m
			}
		}
	}

	// Step 4: Default.
	return DefaultMode
}

// ---------------------------------------------------------------------------
// ResolveThreadID
// ---------------------------------------------------------------------------

// ResolveThreadID determines the thread key for LLM session reuse when the
// fidelity mode resolves to "full". Resolution order (spec Section 5.4):
//
//  1. Target node thread_id attribute
//  2. Edge thread_id attribute
//  3. Graph-level default thread (graph attribute "default_thread_id")
//  4. Derived class from the target node (node class attribute)
//  5. Fallback: previous node ID
//
// The edge and previousNodeID parameters may be empty when not applicable.
func ResolveThreadID(edge *graph.Edge, targetNode *graph.Node, g *graph.Graph, previousNodeID string) string {
	// Step 1: Target node thread_id.
	if targetNode != nil {
		if tid := targetNode.ThreadID(); tid != "" {
			return tid
		}
	}

	// Step 2: Edge thread_id.
	if edge != nil {
		if tid := edge.ThreadID(); tid != "" {
			return tid
		}
	}

	// Step 3: Graph-level default thread.
	if g != nil {
		if tid := g.Attrs["default_thread_id"]; tid != "" {
			return tid
		}
	}

	// Step 4: Derived class from the target node.
	if targetNode != nil {
		if cls := targetNode.Class(); cls != "" {
			// Use the first class name as the thread key.
			parts := strings.SplitN(cls, ",", 2)
			return strings.TrimSpace(parts[0])
		}
	}

	// Step 5: Fallback to previous node ID.
	return previousNodeID
}

// ---------------------------------------------------------------------------
// ApplyFidelity
// ---------------------------------------------------------------------------

// ApplyFidelity transforms a context snapshot and log entries according to the
// given fidelity mode. It returns a new snapshot (never mutates the input)
// and a filtered set of log entries.
//
// Mode behaviors:
//   - full:           Returns the snapshot and logs unmodified.
//   - truncate:       Returns only essential keys (goal, current_node, last_stage, outcome).
//   - compact:        Returns all keys but truncates string values longer than 500 chars.
//   - summary:low:    Keeps 5 most recent logs, truncates values > 200 chars.
//   - summary:medium: Keeps 10 most recent logs, truncates values > 500 chars.
//   - summary:high:   Keeps 20 most recent logs, truncates values > 1000 chars.
func ApplyFidelity(mode Mode, snapshot map[string]any, logs []string) (map[string]any, []string) {
	if snapshot == nil {
		snapshot = make(map[string]any)
	}

	switch mode {
	case Full:
		return copySnapshot(snapshot), copyLogs(logs)

	case Truncate:
		return applyTruncate(snapshot), nil

	case Compact:
		return applyCompact(snapshot), copyLogs(logs)

	case SummaryLow, SummaryMedium, SummaryHigh:
		cfg := summaryConfigs[mode]
		return applySummary(snapshot, logs, cfg)

	default:
		// Unrecognized mode falls back to compact behavior.
		return applyCompact(snapshot), copyLogs(logs)
	}
}

// ---------------------------------------------------------------------------
// Mode-specific implementations
// ---------------------------------------------------------------------------

// applyTruncate returns only the essential keys from the snapshot.
func applyTruncate(snapshot map[string]any) map[string]any {
	result := make(map[string]any, len(truncateEssentialKeys))
	for key := range truncateEssentialKeys {
		if v, ok := snapshot[key]; ok {
			result[key] = v
		}
	}
	return result
}

// applyCompact copies all keys but truncates string values exceeding
// compactMaxValueLen characters.
func applyCompact(snapshot map[string]any) map[string]any {
	result := make(map[string]any, len(snapshot))
	for k, v := range snapshot {
		result[k] = truncateValue(v, compactMaxValueLen)
	}
	return result
}

// applySummary filters the snapshot by truncating long values and keeps only
// the N most recent log entries.
func applySummary(snapshot map[string]any, logs []string, cfg summaryConfig) (map[string]any, []string) {
	result := make(map[string]any, len(snapshot))
	for k, v := range snapshot {
		result[k] = truncateValue(v, cfg.maxValueLen)
	}

	filteredLogs := tailLogs(logs, cfg.maxLogs)
	return result, filteredLogs
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// truncateValue truncates string values longer than maxLen, appending "...".
// Non-string values are returned unchanged. For fmt.Stringer types, the
// string representation is truncated.
func truncateValue(v any, maxLen int) any {
	switch s := v.(type) {
	case string:
		if len(s) > maxLen {
			return s[:maxLen] + "..."
		}
		return s
	default:
		// Convert to string via Sprintf for inspection, but only truncate
		// if it is actually a string type. Leave structured values intact.
		return v
	}
}

// tailLogs returns the last n log entries. Returns all entries if n exceeds
// the length, and nil if there are no entries.
func tailLogs(logs []string, n int) []string {
	if len(logs) == 0 {
		return nil
	}
	if n >= len(logs) {
		out := make([]string, len(logs))
		copy(out, logs)
		return out
	}
	start := len(logs) - n
	out := make([]string, n)
	copy(out, logs[start:])
	return out
}

// copySnapshot returns a shallow copy of the snapshot map.
func copySnapshot(snapshot map[string]any) map[string]any {
	result := make(map[string]any, len(snapshot))
	for k, v := range snapshot {
		result[k] = v
	}
	return result
}

// copyLogs returns a copy of the log slice.
func copyLogs(logs []string) []string {
	if logs == nil {
		return nil
	}
	out := make([]string, len(logs))
	copy(out, logs)
	return out
}

// ValidModes returns a sorted list of all valid fidelity mode strings,
// useful for validation error messages.
func ValidModes() []string {
	modes := make([]string, 0, len(validModes))
	for m := range validModes {
		modes = append(modes, string(m))
	}
	sort.Strings(modes)
	return modes
}

// DegradeForCheckpointResume returns the appropriate fidelity mode after
// checkpoint resume. When the previous node used "full" fidelity, in-memory
// LLM sessions cannot be serialized, so we degrade to "summary:high" for
// the first resumed node (spec Section 5.3 step 6).
func DegradeForCheckpointResume(previousFidelity Mode) Mode {
	if previousFidelity == Full {
		return SummaryHigh
	}
	return previousFidelity
}

// FormatFidelityInfo returns a human-readable summary of the fidelity
// resolution for logging/debugging purposes.
func FormatFidelityInfo(mode Mode, threadID string) string {
	if mode == Full && threadID != "" {
		return fmt.Sprintf("fidelity=%s thread_id=%s", mode, threadID)
	}
	return fmt.Sprintf("fidelity=%s", mode)
}
