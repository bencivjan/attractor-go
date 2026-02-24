package fidelity

import (
	"fmt"
	"strings"
	"testing"

	"github.com/strongdm/attractor-go/attractor/graph"
)

// ---------------------------------------------------------------------------
// Mode validation
// ---------------------------------------------------------------------------

func TestModeIsValid(t *testing.T) {
	valid := []Mode{Full, Truncate, Compact, SummaryLow, SummaryMedium, SummaryHigh}
	for _, m := range valid {
		if !m.IsValid() {
			t.Errorf("expected mode %q to be valid", m)
		}
	}

	invalid := []Mode{"", "unknown", "FULL", "Summary:Low"}
	for _, m := range invalid {
		if m.IsValid() {
			t.Errorf("expected mode %q to be invalid", m)
		}
	}
}

func TestParseMode(t *testing.T) {
	tests := []struct {
		input string
		want  Mode
		ok    bool
	}{
		{"full", Full, true},
		{"truncate", Truncate, true},
		{"compact", Compact, true},
		{"summary:low", SummaryLow, true},
		{"summary:medium", SummaryMedium, true},
		{"summary:high", SummaryHigh, true},
		{"  full  ", Full, true},    // whitespace trimmed
		{"COMPACT", Compact, true},  // case insensitive
		{"", DefaultMode, false},
		{"invalid", DefaultMode, false},
		{"summary:", DefaultMode, false},
	}

	for _, tt := range tests {
		got, ok := ParseMode(tt.input)
		if got != tt.want || ok != tt.ok {
			t.Errorf("ParseMode(%q) = (%q, %v), want (%q, %v)", tt.input, got, ok, tt.want, tt.ok)
		}
	}
}

func TestValidModes(t *testing.T) {
	modes := ValidModes()
	if len(modes) != 6 {
		t.Fatalf("expected 6 valid modes, got %d: %v", len(modes), modes)
	}
	// Should be sorted.
	for i := 1; i < len(modes); i++ {
		if modes[i-1] > modes[i] {
			t.Errorf("ValidModes not sorted: %q > %q", modes[i-1], modes[i])
		}
	}
}

// ---------------------------------------------------------------------------
// ResolveFidelity
// ---------------------------------------------------------------------------

func TestResolveFidelity_EdgeTakesPrecedence(t *testing.T) {
	edge := &graph.Edge{From: "a", To: "b", Attrs: map[string]string{"fidelity": "full"}}
	node := &graph.Node{ID: "b", Attrs: map[string]string{"fidelity": "truncate"}}
	g := &graph.Graph{Attrs: map[string]string{"default_fidelity": "compact"}}

	got := ResolveFidelity(edge, node, g)
	if got != Full {
		t.Errorf("expected edge fidelity %q, got %q", Full, got)
	}
}

func TestResolveFidelity_NodeFallback(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{"fidelity": "truncate"}}
	g := &graph.Graph{Attrs: map[string]string{"default_fidelity": "compact"}}

	got := ResolveFidelity(nil, node, g)
	if got != Truncate {
		t.Errorf("expected node fidelity %q, got %q", Truncate, got)
	}
}

func TestResolveFidelity_GraphFallback(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{}}
	g := &graph.Graph{Attrs: map[string]string{"default_fidelity": "summary:medium"}}

	got := ResolveFidelity(nil, node, g)
	if got != SummaryMedium {
		t.Errorf("expected graph fidelity %q, got %q", SummaryMedium, got)
	}
}

func TestResolveFidelity_DefaultCompact(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{}}
	g := &graph.Graph{Attrs: map[string]string{}}

	got := ResolveFidelity(nil, node, g)
	if got != Compact {
		t.Errorf("expected default fidelity %q, got %q", Compact, got)
	}
}

func TestResolveFidelity_InvalidEdgeFidelitySkipped(t *testing.T) {
	edge := &graph.Edge{From: "a", To: "b", Attrs: map[string]string{"fidelity": "bogus"}}
	node := &graph.Node{ID: "b", Attrs: map[string]string{"fidelity": "truncate"}}
	g := &graph.Graph{Attrs: map[string]string{}}

	got := ResolveFidelity(edge, node, g)
	if got != Truncate {
		t.Errorf("expected node fidelity %q when edge has invalid value, got %q", Truncate, got)
	}
}

func TestResolveFidelity_AllNil(t *testing.T) {
	got := ResolveFidelity(nil, nil, nil)
	if got != Compact {
		t.Errorf("expected default %q with all nil inputs, got %q", Compact, got)
	}
}

// ---------------------------------------------------------------------------
// ResolveThreadID
// ---------------------------------------------------------------------------

func TestResolveThreadID_NodeFirst(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{"thread_id": "node-thread"}}
	edge := &graph.Edge{From: "a", To: "b", Attrs: map[string]string{"thread_id": "edge-thread"}}
	g := &graph.Graph{Attrs: map[string]string{"default_thread_id": "graph-thread"}}

	got := ResolveThreadID(edge, node, g, "prev")
	if got != "node-thread" {
		t.Errorf("expected node thread_id %q, got %q", "node-thread", got)
	}
}

func TestResolveThreadID_EdgeFallback(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{}}
	edge := &graph.Edge{From: "a", To: "b", Attrs: map[string]string{"thread_id": "edge-thread"}}

	got := ResolveThreadID(edge, node, nil, "prev")
	if got != "edge-thread" {
		t.Errorf("expected edge thread_id %q, got %q", "edge-thread", got)
	}
}

func TestResolveThreadID_GraphFallback(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{}}
	g := &graph.Graph{Attrs: map[string]string{"default_thread_id": "graph-thread"}}

	got := ResolveThreadID(nil, node, g, "prev")
	if got != "graph-thread" {
		t.Errorf("expected graph thread_id %q, got %q", "graph-thread", got)
	}
}

func TestResolveThreadID_ClassFallback(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{"class": "loop-a,debug"}}
	g := &graph.Graph{Attrs: map[string]string{}}

	got := ResolveThreadID(nil, node, g, "prev")
	if got != "loop-a" {
		t.Errorf("expected class-derived thread_id %q, got %q", "loop-a", got)
	}
}

func TestResolveThreadID_PreviousNodeFallback(t *testing.T) {
	node := &graph.Node{ID: "b", Attrs: map[string]string{}}
	g := &graph.Graph{Attrs: map[string]string{}}

	got := ResolveThreadID(nil, node, g, "prev-node")
	if got != "prev-node" {
		t.Errorf("expected previous node ID %q, got %q", "prev-node", got)
	}
}

func TestResolveThreadID_AllEmpty(t *testing.T) {
	got := ResolveThreadID(nil, nil, nil, "")
	if got != "" {
		t.Errorf("expected empty thread_id, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// ApplyFidelity -- Full
// ---------------------------------------------------------------------------

func TestApplyFidelity_Full(t *testing.T) {
	snap := map[string]any{
		"goal":       "build something",
		"last_stage": "plan",
		"big_value":  strings.Repeat("x", 1000),
	}
	logs := []string{"log1", "log2", "log3"}

	result, resultLogs := ApplyFidelity(Full, snap, logs)

	// Full mode returns everything unchanged.
	if len(result) != len(snap) {
		t.Errorf("full: expected %d keys, got %d", len(snap), len(result))
	}
	if result["big_value"] != snap["big_value"] {
		t.Error("full: big_value should not be truncated")
	}
	if len(resultLogs) != len(logs) {
		t.Errorf("full: expected %d logs, got %d", len(logs), len(resultLogs))
	}
}

func TestApplyFidelity_Full_DoesNotMutateInput(t *testing.T) {
	snap := map[string]any{"key": "value"}
	logs := []string{"log1"}

	result, _ := ApplyFidelity(Full, snap, logs)
	result["new_key"] = "added"

	if _, ok := snap["new_key"]; ok {
		t.Error("full: mutated the input snapshot")
	}
}

// ---------------------------------------------------------------------------
// ApplyFidelity -- Truncate
// ---------------------------------------------------------------------------

func TestApplyFidelity_Truncate(t *testing.T) {
	snap := map[string]any{
		"goal":         "build something",
		"current_node": "plan",
		"last_stage":   "init",
		"outcome":      "success",
		"extra_key":    "should be dropped",
		"another":      42,
	}
	logs := []string{"log1", "log2"}

	result, resultLogs := ApplyFidelity(Truncate, snap, logs)

	// Truncate keeps only essential keys.
	expectedKeys := map[string]bool{"goal": true, "current_node": true, "last_stage": true, "outcome": true}
	if len(result) != len(expectedKeys) {
		t.Errorf("truncate: expected %d keys, got %d: %v", len(expectedKeys), len(result), result)
	}
	for key := range expectedKeys {
		if _, ok := result[key]; !ok {
			t.Errorf("truncate: missing expected key %q", key)
		}
	}
	if _, ok := result["extra_key"]; ok {
		t.Error("truncate: non-essential key 'extra_key' should have been removed")
	}

	// Truncate drops logs entirely.
	if resultLogs != nil {
		t.Errorf("truncate: expected nil logs, got %v", resultLogs)
	}
}

func TestApplyFidelity_Truncate_MissingKeys(t *testing.T) {
	snap := map[string]any{"goal": "test"}

	result, _ := ApplyFidelity(Truncate, snap, nil)
	if len(result) != 1 {
		t.Errorf("truncate: expected 1 key, got %d", len(result))
	}
	if result["goal"] != "test" {
		t.Errorf("truncate: expected goal=%q, got %q", "test", result["goal"])
	}
}

// ---------------------------------------------------------------------------
// ApplyFidelity -- Compact
// ---------------------------------------------------------------------------

func TestApplyFidelity_Compact(t *testing.T) {
	shortValue := "short"
	longValue := strings.Repeat("a", 600)

	snap := map[string]any{
		"short_key": shortValue,
		"long_key":  longValue,
		"int_key":   42,
	}

	result, _ := ApplyFidelity(Compact, snap, nil)

	// Short values pass through.
	if result["short_key"] != shortValue {
		t.Errorf("compact: short_key should be unchanged")
	}

	// Long string values are truncated to 500 chars + "...".
	got, ok := result["long_key"].(string)
	if !ok {
		t.Fatalf("compact: long_key should be a string")
	}
	if len(got) != 503 { // 500 + len("...")
		t.Errorf("compact: long_key length = %d, want 503", len(got))
	}
	if !strings.HasSuffix(got, "...") {
		t.Error("compact: long_key should end with '...'")
	}

	// Non-string values pass through unchanged.
	if result["int_key"] != 42 {
		t.Errorf("compact: int_key should be unchanged, got %v", result["int_key"])
	}
}

func TestApplyFidelity_Compact_BoundaryLength(t *testing.T) {
	exactly500 := strings.Repeat("b", 500)
	snap := map[string]any{"key": exactly500}

	result, _ := ApplyFidelity(Compact, snap, nil)
	got := result["key"].(string)
	if got != exactly500 {
		t.Error("compact: value of exactly 500 chars should not be truncated")
	}
}

// ---------------------------------------------------------------------------
// ApplyFidelity -- Summary modes
// ---------------------------------------------------------------------------

func TestApplyFidelity_SummaryLow(t *testing.T) {
	snap := map[string]any{
		"key": strings.Repeat("c", 300),
	}
	logs := makeLogs(20)

	result, resultLogs := ApplyFidelity(SummaryLow, snap, logs)

	// Values > 200 are truncated.
	got := result["key"].(string)
	if len(got) != 203 {
		t.Errorf("summary:low value len = %d, want 203", len(got))
	}

	// Only last 5 logs.
	if len(resultLogs) != 5 {
		t.Errorf("summary:low logs = %d, want 5", len(resultLogs))
	}
	if resultLogs[0] != "log-15" {
		t.Errorf("summary:low first log = %q, want %q", resultLogs[0], "log-15")
	}
}

func TestApplyFidelity_SummaryMedium(t *testing.T) {
	snap := map[string]any{
		"key": strings.Repeat("d", 600),
	}
	logs := makeLogs(20)

	result, resultLogs := ApplyFidelity(SummaryMedium, snap, logs)

	// Values > 500 are truncated.
	got := result["key"].(string)
	if len(got) != 503 {
		t.Errorf("summary:medium value len = %d, want 503", len(got))
	}

	// Only last 10 logs.
	if len(resultLogs) != 10 {
		t.Errorf("summary:medium logs = %d, want 10", len(resultLogs))
	}
	if resultLogs[0] != "log-10" {
		t.Errorf("summary:medium first log = %q, want %q", resultLogs[0], "log-10")
	}
}

func TestApplyFidelity_SummaryHigh(t *testing.T) {
	snap := map[string]any{
		"key": strings.Repeat("e", 1200),
	}
	logs := makeLogs(30)

	result, resultLogs := ApplyFidelity(SummaryHigh, snap, logs)

	// Values > 1000 are truncated.
	got := result["key"].(string)
	if len(got) != 1003 {
		t.Errorf("summary:high value len = %d, want 1003", len(got))
	}

	// Only last 20 logs.
	if len(resultLogs) != 20 {
		t.Errorf("summary:high logs = %d, want 20", len(resultLogs))
	}
	if resultLogs[0] != "log-10" {
		t.Errorf("summary:high first log = %q, want %q", resultLogs[0], "log-10")
	}
}

func TestApplyFidelity_SummaryWithFewerLogsThanLimit(t *testing.T) {
	logs := makeLogs(3)

	_, resultLogs := ApplyFidelity(SummaryHigh, nil, logs)

	// When fewer logs exist than the limit, return all.
	if len(resultLogs) != 3 {
		t.Errorf("summary:high with 3 logs = %d, want 3", len(resultLogs))
	}
}

// ---------------------------------------------------------------------------
// ApplyFidelity -- Edge cases
// ---------------------------------------------------------------------------

func TestApplyFidelity_NilSnapshot(t *testing.T) {
	result, _ := ApplyFidelity(Full, nil, nil)
	if result == nil {
		t.Error("expected non-nil result for nil snapshot input")
	}
	if len(result) != 0 {
		t.Errorf("expected empty result, got %d keys", len(result))
	}
}

func TestApplyFidelity_EmptyLogs(t *testing.T) {
	_, resultLogs := ApplyFidelity(SummaryLow, nil, nil)
	if resultLogs != nil {
		t.Errorf("expected nil logs, got %v", resultLogs)
	}

	_, resultLogs = ApplyFidelity(SummaryLow, nil, []string{})
	if resultLogs != nil {
		t.Errorf("expected nil logs for empty slice, got %v", resultLogs)
	}
}

func TestApplyFidelity_UnknownModeFallsBackToCompact(t *testing.T) {
	longValue := strings.Repeat("x", 600)
	snap := map[string]any{"key": longValue}

	result, _ := ApplyFidelity(Mode("unknown"), snap, nil)
	got := result["key"].(string)
	if len(got) != 503 {
		t.Errorf("unknown mode should fall back to compact, value len = %d, want 503", len(got))
	}
}

// ---------------------------------------------------------------------------
// DegradeForCheckpointResume
// ---------------------------------------------------------------------------

func TestDegradeForCheckpointResume(t *testing.T) {
	tests := []struct {
		input Mode
		want  Mode
	}{
		{Full, SummaryHigh},       // full degrades to summary:high
		{Truncate, Truncate},      // non-full modes unchanged
		{Compact, Compact},
		{SummaryLow, SummaryLow},
		{SummaryMedium, SummaryMedium},
		{SummaryHigh, SummaryHigh},
	}

	for _, tt := range tests {
		got := DegradeForCheckpointResume(tt.input)
		if got != tt.want {
			t.Errorf("DegradeForCheckpointResume(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// FormatFidelityInfo
// ---------------------------------------------------------------------------

func TestFormatFidelityInfo(t *testing.T) {
	got := FormatFidelityInfo(Full, "thread-1")
	if !strings.Contains(got, "full") || !strings.Contains(got, "thread-1") {
		t.Errorf("expected full + thread info, got %q", got)
	}

	got = FormatFidelityInfo(Compact, "")
	if strings.Contains(got, "thread_id") {
		t.Errorf("compact without thread should not mention thread_id, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makeLogs creates n numbered log entries: "log-0", "log-1", ...
func makeLogs(n int) []string {
	logs := make([]string, n)
	for i := 0; i < n; i++ {
		logs[i] = fmt.Sprintf("log-%d", i)
	}
	return logs
}
