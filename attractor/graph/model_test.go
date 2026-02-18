package graph

import (
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Helper: build a minimal graph for testing
// ---------------------------------------------------------------------------

func makeGraph(nodes map[string]*Node, edges []*Edge, attrs map[string]string) *Graph {
	if attrs == nil {
		attrs = map[string]string{}
	}
	if nodes == nil {
		nodes = map[string]*Node{}
	}
	return &Graph{Name: "test", Nodes: nodes, Edges: edges, Attrs: attrs}
}

func makeNode(id string, attrs map[string]string) *Node {
	if attrs == nil {
		attrs = map[string]string{}
	}
	return &Node{ID: id, Attrs: attrs}
}

func makeEdge(from, to string, attrs map[string]string) *Edge {
	if attrs == nil {
		attrs = map[string]string{}
	}
	return &Edge{From: from, To: to, Attrs: attrs}
}

// ---------------------------------------------------------------------------
// FindStartNode
// ---------------------------------------------------------------------------

func TestFindStartNode_ByShape(t *testing.T) {
	g := makeGraph(
		map[string]*Node{
			"s": makeNode("s", map[string]string{"shape": "Mdiamond"}),
			"a": makeNode("a", nil),
		},
		nil, nil,
	)
	node, err := g.FindStartNode()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if node.ID != "s" {
		t.Errorf("expected start node ID 's', got %q", node.ID)
	}
}

func TestFindStartNode_ByShapeCaseInsensitive(t *testing.T) {
	g := makeGraph(
		map[string]*Node{
			"s": makeNode("s", map[string]string{"shape": "mdiamond"}),
		},
		nil, nil,
	)
	node, err := g.FindStartNode()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if node.ID != "s" {
		t.Errorf("expected start node ID 's', got %q", node.ID)
	}
}

func TestFindStartNode_FallbackID(t *testing.T) {
	tests := []struct {
		name string
		id   string
	}{
		{"lowercase start", "start"},
		{"capitalised Start", "Start"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := makeGraph(
				map[string]*Node{
					tt.id: makeNode(tt.id, nil),
				},
				nil, nil,
			)
			node, err := g.FindStartNode()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if node.ID != tt.id {
				t.Errorf("expected start node ID %q, got %q", tt.id, node.ID)
			}
		})
	}
}

func TestFindStartNode_NotFound(t *testing.T) {
	g := makeGraph(
		map[string]*Node{
			"a": makeNode("a", nil),
		},
		nil, nil,
	)
	_, err := g.FindStartNode()
	if err == nil {
		t.Fatal("expected error when no start node exists")
	}
}

// ---------------------------------------------------------------------------
// FindExitNode
// ---------------------------------------------------------------------------

func TestFindExitNode_ByShape(t *testing.T) {
	g := makeGraph(
		map[string]*Node{
			"e": makeNode("e", map[string]string{"shape": "Msquare"}),
		},
		nil, nil,
	)
	node, err := g.FindExitNode()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if node.ID != "e" {
		t.Errorf("expected exit node ID 'e', got %q", node.ID)
	}
}

func TestFindExitNode_ByShapeCaseInsensitive(t *testing.T) {
	g := makeGraph(
		map[string]*Node{
			"e": makeNode("e", map[string]string{"shape": "msquare"}),
		},
		nil, nil,
	)
	node, err := g.FindExitNode()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if node.ID != "e" {
		t.Errorf("expected exit node ID 'e', got %q", node.ID)
	}
}

func TestFindExitNode_FallbackIDs(t *testing.T) {
	tests := []struct {
		name string
		id   string
	}{
		{"exit", "exit"},
		{"end", "end"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := makeGraph(
				map[string]*Node{
					tt.id: makeNode(tt.id, nil),
				},
				nil, nil,
			)
			node, err := g.FindExitNode()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if node.ID != tt.id {
				t.Errorf("expected exit node ID %q, got %q", tt.id, node.ID)
			}
		})
	}
}

func TestFindExitNode_NotFound(t *testing.T) {
	g := makeGraph(
		map[string]*Node{
			"a": makeNode("a", nil),
		},
		nil, nil,
	)
	_, err := g.FindExitNode()
	if err == nil {
		t.Fatal("expected error when no exit node exists")
	}
}

// ---------------------------------------------------------------------------
// OutgoingEdges / IncomingEdges
// ---------------------------------------------------------------------------

func TestOutgoingEdges(t *testing.T) {
	edges := []*Edge{
		makeEdge("a", "b", nil),
		makeEdge("a", "c", nil),
		makeEdge("b", "c", nil),
	}
	g := makeGraph(nil, edges, nil)

	out := g.OutgoingEdges("a")
	if len(out) != 2 {
		t.Fatalf("expected 2 outgoing edges from 'a', got %d", len(out))
	}

	out = g.OutgoingEdges("b")
	if len(out) != 1 {
		t.Fatalf("expected 1 outgoing edge from 'b', got %d", len(out))
	}

	out = g.OutgoingEdges("c")
	if len(out) != 0 {
		t.Fatalf("expected 0 outgoing edges from 'c', got %d", len(out))
	}
}

func TestIncomingEdges(t *testing.T) {
	edges := []*Edge{
		makeEdge("a", "b", nil),
		makeEdge("a", "c", nil),
		makeEdge("b", "c", nil),
	}
	g := makeGraph(nil, edges, nil)

	inc := g.IncomingEdges("a")
	if len(inc) != 0 {
		t.Fatalf("expected 0 incoming edges to 'a', got %d", len(inc))
	}

	inc = g.IncomingEdges("c")
	if len(inc) != 2 {
		t.Fatalf("expected 2 incoming edges to 'c', got %d", len(inc))
	}
}

func TestOutgoingEdges_NoEdges(t *testing.T) {
	g := makeGraph(nil, nil, nil)
	out := g.OutgoingEdges("nonexistent")
	if len(out) != 0 {
		t.Fatalf("expected 0 outgoing edges, got %d", len(out))
	}
}

// ---------------------------------------------------------------------------
// Node accessor methods
// ---------------------------------------------------------------------------

func TestNode_Label(t *testing.T) {
	tests := []struct {
		name   string
		attrs  map[string]string
		nodeID string
		want   string
	}{
		{"explicit label", map[string]string{"label": "My Label"}, "n1", "My Label"},
		{"defaults to ID", map[string]string{}, "n1", "n1"},
		{"empty label defaults to ID", map[string]string{"label": ""}, "n1", "n1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode(tt.nodeID, tt.attrs)
			if got := n.Label(); got != tt.want {
				t.Errorf("Label() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNode_Shape(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  string
	}{
		{"explicit shape", map[string]string{"shape": "diamond"}, "diamond"},
		{"defaults to box", map[string]string{}, "box"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode("n", tt.attrs)
			if got := n.Shape(); got != tt.want {
				t.Errorf("Shape() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNode_Type(t *testing.T) {
	n := makeNode("n", map[string]string{"type": "conditional"})
	if got := n.Type(); got != "conditional" {
		t.Errorf("Type() = %q, want %q", got, "conditional")
	}
	n2 := makeNode("n2", nil)
	if got := n2.Type(); got != "" {
		t.Errorf("Type() on empty attrs = %q, want empty string", got)
	}
}

func TestNode_Prompt(t *testing.T) {
	n := makeNode("n", map[string]string{"prompt": "Generate code"})
	if got := n.Prompt(); got != "Generate code" {
		t.Errorf("Prompt() = %q, want %q", got, "Generate code")
	}
}

func TestNode_MaxRetries(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  int
	}{
		{"explicit value", map[string]string{"max_retries": "5"}, 5},
		{"default to 0", map[string]string{}, 0},
		{"invalid value", map[string]string{"max_retries": "abc"}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode("n", tt.attrs)
			if got := n.MaxRetries(); got != tt.want {
				t.Errorf("MaxRetries() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestNode_GoalGate(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  bool
	}{
		{"true", map[string]string{"goal_gate": "true"}, true},
		{"false", map[string]string{"goal_gate": "false"}, false},
		{"missing", map[string]string{}, false},
		{"other value", map[string]string{"goal_gate": "yes"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode("n", tt.attrs)
			if got := n.GoalGate(); got != tt.want {
				t.Errorf("GoalGate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNode_Timeout(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  time.Duration
	}{
		{"valid duration", map[string]string{"timeout": "30s"}, 30 * time.Second},
		{"minutes", map[string]string{"timeout": "5m"}, 5 * time.Minute},
		{"missing", map[string]string{}, 0},
		{"empty string", map[string]string{"timeout": ""}, 0},
		{"invalid", map[string]string{"timeout": "notaduration"}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode("n", tt.attrs)
			if got := n.Timeout(); got != tt.want {
				t.Errorf("Timeout() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNode_LLMModel(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  string
	}{
		{"model attr", map[string]string{"model": "gpt-4"}, "gpt-4"},
		{"llm_model attr", map[string]string{"llm_model": "claude-3"}, "claude-3"},
		{"model takes precedence", map[string]string{"model": "gpt-4", "llm_model": "claude-3"}, "gpt-4"},
		{"missing", map[string]string{}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode("n", tt.attrs)
			if got := n.LLMModel(); got != tt.want {
				t.Errorf("LLMModel() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNode_ReasoningEffort(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  string
	}{
		{"explicit", map[string]string{"reasoning_effort": "low"}, "low"},
		{"default to high", map[string]string{}, "high"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := makeNode("n", tt.attrs)
			if got := n.ReasoningEffort(); got != tt.want {
				t.Errorf("ReasoningEffort() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNode_BoolAccessors(t *testing.T) {
	n := makeNode("n", map[string]string{
		"auto_status":   "true",
		"allow_partial": "true",
	})
	if !n.AutoStatus() {
		t.Error("AutoStatus() should be true")
	}
	if !n.AllowPartial() {
		t.Error("AllowPartial() should be true")
	}

	n2 := makeNode("n2", nil)
	if n2.AutoStatus() {
		t.Error("AutoStatus() should default to false")
	}
	if n2.AllowPartial() {
		t.Error("AllowPartial() should default to false")
	}
}

func TestNode_StringAccessors(t *testing.T) {
	n := makeNode("n", map[string]string{
		"retry_target":          "nodeA",
		"fallback_retry_target": "nodeB",
		"fidelity":              "full",
		"thread_id":             "thread-1",
		"class":                 "my-class",
		"llm_provider":          "anthropic",
	})
	if got := n.RetryTarget(); got != "nodeA" {
		t.Errorf("RetryTarget() = %q, want %q", got, "nodeA")
	}
	if got := n.FallbackRetryTarget(); got != "nodeB" {
		t.Errorf("FallbackRetryTarget() = %q, want %q", got, "nodeB")
	}
	if got := n.Fidelity(); got != "full" {
		t.Errorf("Fidelity() = %q, want %q", got, "full")
	}
	if got := n.ThreadID(); got != "thread-1" {
		t.Errorf("ThreadID() = %q, want %q", got, "thread-1")
	}
	if got := n.Class(); got != "my-class" {
		t.Errorf("Class() = %q, want %q", got, "my-class")
	}
	if got := n.LLMProvider(); got != "anthropic" {
		t.Errorf("LLMProvider() = %q, want %q", got, "anthropic")
	}
}

// ---------------------------------------------------------------------------
// Edge accessor methods
// ---------------------------------------------------------------------------

func TestEdge_Label(t *testing.T) {
	e := makeEdge("a", "b", map[string]string{"label": "success path"})
	if got := e.Label(); got != "success path" {
		t.Errorf("Label() = %q, want %q", got, "success path")
	}
}

func TestEdge_Condition(t *testing.T) {
	e := makeEdge("a", "b", map[string]string{"condition": "outcome=success"})
	if got := e.Condition(); got != "outcome=success" {
		t.Errorf("Condition() = %q, want %q", got, "outcome=success")
	}
}

func TestEdge_Weight(t *testing.T) {
	tests := []struct {
		name  string
		attrs map[string]string
		want  int
	}{
		{"explicit", map[string]string{"weight": "10"}, 10},
		{"default to 0", map[string]string{}, 0},
		{"invalid falls back", map[string]string{"weight": "xyz"}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := makeEdge("a", "b", tt.attrs)
			if got := e.Weight(); got != tt.want {
				t.Errorf("Weight() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestEdge_LoopRestart(t *testing.T) {
	e := makeEdge("a", "b", map[string]string{"loop_restart": "true"})
	if !e.LoopRestart() {
		t.Error("LoopRestart() should be true")
	}
	e2 := makeEdge("a", "b", nil)
	if e2.LoopRestart() {
		t.Error("LoopRestart() should default to false")
	}
}

func TestEdge_StringAccessors(t *testing.T) {
	e := makeEdge("a", "b", map[string]string{
		"fidelity":  "compact",
		"thread_id": "t1",
	})
	if got := e.Fidelity(); got != "compact" {
		t.Errorf("Fidelity() = %q, want %q", got, "compact")
	}
	if got := e.ThreadID(); got != "t1" {
		t.Errorf("ThreadID() = %q, want %q", got, "t1")
	}
}

// ---------------------------------------------------------------------------
// Graph-level accessors
// ---------------------------------------------------------------------------

func TestGraph_Accessors(t *testing.T) {
	g := makeGraph(nil, nil, map[string]string{
		"goal":                  "Build a feature",
		"label":                 "Feature Pipeline",
		"model_stylesheet":      "styles.css",
		"default_max_retry":     "10",
		"retry_target":          "coder",
		"fallback_retry_target": "fallback_coder",
		"default_fidelity":      "full",
	})

	if got := g.Goal(); got != "Build a feature" {
		t.Errorf("Goal() = %q, want %q", got, "Build a feature")
	}
	if got := g.Label(); got != "Feature Pipeline" {
		t.Errorf("Label() = %q, want %q", got, "Feature Pipeline")
	}
	if got := g.ModelStylesheet(); got != "styles.css" {
		t.Errorf("ModelStylesheet() = %q, want %q", got, "styles.css")
	}
	if got := g.DefaultMaxRetry(); got != 10 {
		t.Errorf("DefaultMaxRetry() = %d, want %d", got, 10)
	}
	if got := g.RetryTarget(); got != "coder" {
		t.Errorf("RetryTarget() = %q, want %q", got, "coder")
	}
	if got := g.FallbackRetryTarget(); got != "fallback_coder" {
		t.Errorf("FallbackRetryTarget() = %q, want %q", got, "fallback_coder")
	}
	if got := g.DefaultFidelity(); got != "full" {
		t.Errorf("DefaultFidelity() = %q, want %q", got, "full")
	}
}

func TestGraph_DefaultMaxRetry_Default(t *testing.T) {
	g := makeGraph(nil, nil, nil)
	if got := g.DefaultMaxRetry(); got != 50 {
		t.Errorf("DefaultMaxRetry() with no attr = %d, want 50", got)
	}
}

func TestGraph_DefaultMaxRetry_InvalidValue(t *testing.T) {
	g := makeGraph(nil, nil, map[string]string{"default_max_retry": "not_a_number"})
	if got := g.DefaultMaxRetry(); got != 50 {
		t.Errorf("DefaultMaxRetry() with invalid attr = %d, want 50", got)
	}
}
