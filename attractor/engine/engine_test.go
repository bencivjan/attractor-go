package engine

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// Helpers: build test graphs programmatically
// ---------------------------------------------------------------------------

// makeStartExitGraph creates the simplest valid pipeline: Start -> Exit.
func makeStartExitGraph() *graph.Graph {
	g := &graph.Graph{
		Name: "test-start-exit",
		Nodes: map[string]*graph.Node{
			"start": {ID: "start", Attrs: map[string]string{"shape": "Mdiamond", "label": "Start"}},
			"exit":  {ID: "exit", Attrs: map[string]string{"shape": "Msquare", "label": "Exit"}},
		},
		Edges: []*graph.Edge{
			{From: "start", To: "exit", Attrs: map[string]string{}},
		},
		Attrs: map[string]string{},
	}
	return g
}

// makeLinearGraph creates: Start -> A -> B -> Exit.
func makeLinearGraph() *graph.Graph {
	return &graph.Graph{
		Name: "test-linear",
		Nodes: map[string]*graph.Node{
			"start": {ID: "start", Attrs: map[string]string{"shape": "Mdiamond"}},
			"a":     {ID: "a", Attrs: map[string]string{"shape": "box", "label": "Node A"}},
			"b":     {ID: "b", Attrs: map[string]string{"shape": "box", "label": "Node B"}},
			"exit":  {ID: "exit", Attrs: map[string]string{"shape": "Msquare"}},
		},
		Edges: []*graph.Edge{
			{From: "start", To: "a", Attrs: map[string]string{}},
			{From: "a", To: "b", Attrs: map[string]string{}},
			{From: "b", To: "exit", Attrs: map[string]string{}},
		},
		Attrs: map[string]string{},
	}
}

// makeBranchingGraph creates a graph with conditional branching:
//
//	Start -> cond -> (outcome=success) -> good -> Exit
//	                 (outcome=fail)    -> bad  -> Exit
func makeBranchingGraph() *graph.Graph {
	return &graph.Graph{
		Name: "test-branching",
		Nodes: map[string]*graph.Node{
			"start": {ID: "start", Attrs: map[string]string{"shape": "Mdiamond"}},
			"cond":  {ID: "cond", Attrs: map[string]string{"shape": "diamond", "type": "conditional"}},
			"good":  {ID: "good", Attrs: map[string]string{"shape": "box"}},
			"bad":   {ID: "bad", Attrs: map[string]string{"shape": "box"}},
			"exit":  {ID: "exit", Attrs: map[string]string{"shape": "Msquare"}},
		},
		Edges: []*graph.Edge{
			{From: "start", To: "cond", Attrs: map[string]string{}},
			{From: "cond", To: "good", Attrs: map[string]string{"condition": "outcome=success"}},
			{From: "cond", To: "bad", Attrs: map[string]string{"condition": "outcome=fail"}},
			{From: "good", To: "exit", Attrs: map[string]string{}},
			{From: "bad", To: "exit", Attrs: map[string]string{}},
		},
		Attrs: map[string]string{},
	}
}

// makeGoalGateGraph creates a graph with a goal gate that must be satisfied:
//
//	Start -> worker -> Exit
//
// worker has goal_gate=true.
func makeGoalGateGraph(retryTarget string) *graph.Graph {
	g := &graph.Graph{
		Name: "test-goal-gate",
		Nodes: map[string]*graph.Node{
			"start":  {ID: "start", Attrs: map[string]string{"shape": "Mdiamond"}},
			"worker": {ID: "worker", Attrs: map[string]string{"shape": "box", "goal_gate": "true", "retry_target": retryTarget}},
			"exit":   {ID: "exit", Attrs: map[string]string{"shape": "Msquare"}},
		},
		Edges: []*graph.Edge{
			{From: "start", To: "worker", Attrs: map[string]string{}},
			{From: "worker", To: "exit", Attrs: map[string]string{}},
		},
		Attrs: map[string]string{},
	}
	return g
}

func defaultCfg(t *testing.T) Config {
	t.Helper()
	return Config{
		LogsRoot: t.TempDir(),
		Registry: handler.DefaultRegistry(handler.NoopHandler{}),
		MaxSteps: 100,
	}
}

// ---------------------------------------------------------------------------
// Test: Simple start -> exit pipeline runs to completion
// ---------------------------------------------------------------------------

func TestStartExitPipeline(t *testing.T) {
	g := makeStartExitGraph()
	cfg := defaultCfg(t)

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome == nil {
		t.Fatal("expected non-nil outcome")
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected status %q, got %q", state.StatusSuccess, outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: Linear pipeline
// ---------------------------------------------------------------------------

func TestLinearPipeline(t *testing.T) {
	g := makeLinearGraph()
	cfg := defaultCfg(t)

	var visited []string
	cfg.OnEvent = func(e Event) {
		if e.Kind == EventStageStarted {
			if nodeID, ok := e.Data["node"].(string); ok {
				visited = append(visited, nodeID)
			}
		}
	}

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}

	// Should have visited start, a, b (exit is terminal, not "executed" via handler in the same sense).
	expected := []string{"start", "a", "b"}
	if len(visited) < len(expected) {
		t.Errorf("expected at least %d stages started, got %d: %v", len(expected), len(visited), visited)
	}
	for _, e := range expected {
		found := false
		for _, v := range visited {
			if v == e {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected node %q to be visited, visited: %v", e, visited)
		}
	}
}

// ---------------------------------------------------------------------------
// Test: Edge selection -- condition matching
// ---------------------------------------------------------------------------

func TestSelectEdge_ConditionMatch(t *testing.T) {
	g := &graph.Graph{
		Name: "edge-test",
		Nodes: map[string]*graph.Node{
			"n": {ID: "n", Attrs: map[string]string{}},
			"a": {ID: "a", Attrs: map[string]string{}},
			"b": {ID: "b", Attrs: map[string]string{}},
		},
		Edges: []*graph.Edge{
			{From: "n", To: "a", Attrs: map[string]string{"condition": "outcome=success"}},
			{From: "n", To: "b", Attrs: map[string]string{"condition": "outcome=fail"}},
		},
		Attrs: map[string]string{},
	}

	node := g.Nodes["n"]

	// Outcome is success -> should select edge to "a".
	outcome := &state.Outcome{Status: state.StatusSuccess}
	pctx := state.NewContext()
	edge := selectEdge(node, outcome, pctx, g)
	if edge == nil {
		t.Fatal("expected an edge, got nil")
	}
	if edge.To != "a" {
		t.Errorf("expected edge to 'a', got %q", edge.To)
	}

	// Outcome is fail -> should select edge to "b".
	outcome = &state.Outcome{Status: state.StatusFail}
	edge = selectEdge(node, outcome, pctx, g)
	if edge == nil {
		t.Fatal("expected an edge, got nil")
	}
	if edge.To != "b" {
		t.Errorf("expected edge to 'b', got %q", edge.To)
	}
}

// ---------------------------------------------------------------------------
// Test: Edge selection -- preferred label
// ---------------------------------------------------------------------------

func TestSelectEdge_PreferredLabel(t *testing.T) {
	g := &graph.Graph{
		Name: "label-test",
		Nodes: map[string]*graph.Node{
			"n": {ID: "n", Attrs: map[string]string{}},
			"a": {ID: "a", Attrs: map[string]string{}},
			"b": {ID: "b", Attrs: map[string]string{}},
		},
		Edges: []*graph.Edge{
			{From: "n", To: "a", Attrs: map[string]string{"label": "Yes"}},
			{From: "n", To: "b", Attrs: map[string]string{"label": "No"}},
		},
		Attrs: map[string]string{},
	}

	node := g.Nodes["n"]
	outcome := &state.Outcome{Status: state.StatusSuccess, PreferredLabel: "No"}
	pctx := state.NewContext()

	edge := selectEdge(node, outcome, pctx, g)
	if edge == nil {
		t.Fatal("expected an edge, got nil")
	}
	if edge.To != "b" {
		t.Errorf("expected edge to 'b' (label=No), got %q", edge.To)
	}
}

// ---------------------------------------------------------------------------
// Test: Edge selection -- suggested next IDs
// ---------------------------------------------------------------------------

func TestSelectEdge_SuggestedNextIDs(t *testing.T) {
	g := &graph.Graph{
		Name: "suggested-test",
		Nodes: map[string]*graph.Node{
			"n": {ID: "n", Attrs: map[string]string{}},
			"a": {ID: "a", Attrs: map[string]string{}},
			"b": {ID: "b", Attrs: map[string]string{}},
			"c": {ID: "c", Attrs: map[string]string{}},
		},
		Edges: []*graph.Edge{
			{From: "n", To: "a", Attrs: map[string]string{}},
			{From: "n", To: "b", Attrs: map[string]string{}},
			{From: "n", To: "c", Attrs: map[string]string{}},
		},
		Attrs: map[string]string{},
	}

	node := g.Nodes["n"]
	outcome := &state.Outcome{Status: state.StatusSuccess, SuggestedNextIDs: []string{"c"}}
	pctx := state.NewContext()

	edge := selectEdge(node, outcome, pctx, g)
	if edge == nil {
		t.Fatal("expected an edge, got nil")
	}
	if edge.To != "c" {
		t.Errorf("expected edge to 'c' (suggested), got %q", edge.To)
	}
}

// ---------------------------------------------------------------------------
// Test: Edge selection -- weight tiebreak
// ---------------------------------------------------------------------------

func TestSelectEdge_WeightTiebreak(t *testing.T) {
	g := &graph.Graph{
		Name: "weight-test",
		Nodes: map[string]*graph.Node{
			"n": {ID: "n", Attrs: map[string]string{}},
			"a": {ID: "a", Attrs: map[string]string{}},
			"b": {ID: "b", Attrs: map[string]string{}},
		},
		Edges: []*graph.Edge{
			{From: "n", To: "a", Attrs: map[string]string{"weight": "1"}},
			{From: "n", To: "b", Attrs: map[string]string{"weight": "10"}},
		},
		Attrs: map[string]string{},
	}

	node := g.Nodes["n"]
	outcome := &state.Outcome{Status: state.StatusSuccess}
	pctx := state.NewContext()

	edge := selectEdge(node, outcome, pctx, g)
	if edge == nil {
		t.Fatal("expected an edge, got nil")
	}
	if edge.To != "b" {
		t.Errorf("expected edge to 'b' (weight=10), got %q", edge.To)
	}
}

// ---------------------------------------------------------------------------
// Test: Edge selection -- lexical tiebreak
// ---------------------------------------------------------------------------

func TestSelectEdge_LexicalTiebreak(t *testing.T) {
	g := &graph.Graph{
		Name: "lexical-test",
		Nodes: map[string]*graph.Node{
			"n":     {ID: "n", Attrs: map[string]string{}},
			"alpha": {ID: "alpha", Attrs: map[string]string{}},
			"beta":  {ID: "beta", Attrs: map[string]string{}},
		},
		Edges: []*graph.Edge{
			{From: "n", To: "beta", Attrs: map[string]string{}},
			{From: "n", To: "alpha", Attrs: map[string]string{}},
		},
		Attrs: map[string]string{},
	}

	node := g.Nodes["n"]
	outcome := &state.Outcome{Status: state.StatusSuccess}
	pctx := state.NewContext()

	edge := selectEdge(node, outcome, pctx, g)
	if edge == nil {
		t.Fatal("expected an edge, got nil")
	}
	// Equal weight (default 0), lexical tiebreak should pick "alpha".
	if edge.To != "alpha" {
		t.Errorf("expected edge to 'alpha' (lexical), got %q", edge.To)
	}
}

// ---------------------------------------------------------------------------
// Test: Goal gate checking -- all satisfied
// ---------------------------------------------------------------------------

func TestCheckGoalGates_AllSatisfied(t *testing.T) {
	g := makeGoalGateGraph("start")
	pctx := state.NewContext()
	pctx.Set("status.worker", "success")

	allSatisfied, failedNode := checkGoalGates(g, pctx)
	if !allSatisfied {
		t.Error("expected all gates satisfied")
	}
	if failedNode != nil {
		t.Errorf("expected no failed node, got %q", failedNode.ID)
	}
}

// ---------------------------------------------------------------------------
// Test: Goal gate checking -- unsatisfied with retry target
// ---------------------------------------------------------------------------

func TestCheckGoalGates_Unsatisfied(t *testing.T) {
	g := makeGoalGateGraph("start")
	pctx := state.NewContext()
	// Do NOT set status.worker, so it remains unsatisfied.

	allSatisfied, failedNode := checkGoalGates(g, pctx)
	if allSatisfied {
		t.Error("expected gates unsatisfied")
	}
	if failedNode == nil {
		t.Fatal("expected a failed gate node")
	}
	if failedNode.ID != "worker" {
		t.Errorf("expected failed gate 'worker', got %q", failedNode.ID)
	}
}

// ---------------------------------------------------------------------------
// Test: normalizeLabel strips accelerator patterns
// ---------------------------------------------------------------------------

func TestNormalizeLabel(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"[K] Label", "label"},
		{"[Y] Yes", "yes"},
		{"K) Label", "label"},
		{"A) Answer", "answer"},
		{"K - Label", "label"},
		{"B - Beta", "beta"},
		{"plain text", "plain text"},
		{"  UPPER  ", "upper"},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("input=%q", tt.input), func(t *testing.T) {
			got := normalizeLabel(tt.input)
			if got != tt.want {
				t.Errorf("normalizeLabel(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test: executeWithRetry -- success on first try
// ---------------------------------------------------------------------------

func TestExecuteWithRetry_SuccessFirstTry(t *testing.T) {
	node := &graph.Node{ID: "test-node", Attrs: map[string]string{}}
	pctx := state.NewContext()
	g := &graph.Graph{Nodes: map[string]*graph.Node{"test-node": node}, Attrs: map[string]string{}}

	callCount := 0
	h := handler.HandlerFunc(func(ctx context.Context, n *graph.Node, p *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
		callCount++
		return &state.Outcome{Status: state.StatusSuccess}, nil
	})

	policy := RetryPolicy{MaxAttempts: 3, InitialDelay: time.Millisecond, BackoffFactor: 1.0, MaxDelay: time.Second}
	cfg := Config{}

	outcome := executeWithRetry(context.Background(), h, node, pctx, g, t.TempDir(), policy, cfg)
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call, got %d", callCount)
	}
}

// ---------------------------------------------------------------------------
// Test: executeWithRetry -- retry on failure
// ---------------------------------------------------------------------------

func TestExecuteWithRetry_RetryOnFailure(t *testing.T) {
	node := &graph.Node{ID: "test-node", Attrs: map[string]string{}}
	pctx := state.NewContext()
	g := &graph.Graph{Nodes: map[string]*graph.Node{"test-node": node}, Attrs: map[string]string{}}

	callCount := 0
	h := handler.HandlerFunc(func(ctx context.Context, n *graph.Node, p *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
		callCount++
		if callCount < 3 {
			return &state.Outcome{Status: state.StatusFail, FailureReason: "transient"}, nil
		}
		return &state.Outcome{Status: state.StatusSuccess}, nil
	})

	policy := RetryPolicy{MaxAttempts: 5, InitialDelay: time.Millisecond, BackoffFactor: 1.0, MaxDelay: time.Millisecond, Jitter: false}
	cfg := Config{}

	outcome := executeWithRetry(context.Background(), h, node, pctx, g, t.TempDir(), policy, cfg)
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success after retries, got %q", outcome.Status)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls (2 failures + 1 success), got %d", callCount)
	}
}

// ---------------------------------------------------------------------------
// Test: executeWithRetry -- max retries exceeded
// ---------------------------------------------------------------------------

func TestExecuteWithRetry_MaxRetriesExceeded(t *testing.T) {
	node := &graph.Node{ID: "test-node", Attrs: map[string]string{}}
	pctx := state.NewContext()
	g := &graph.Graph{Nodes: map[string]*graph.Node{"test-node": node}, Attrs: map[string]string{}}

	callCount := 0
	h := handler.HandlerFunc(func(ctx context.Context, n *graph.Node, p *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
		callCount++
		return &state.Outcome{Status: state.StatusFail, FailureReason: "permanent"}, nil
	})

	policy := RetryPolicy{MaxAttempts: 2, InitialDelay: time.Millisecond, BackoffFactor: 1.0, MaxDelay: time.Millisecond, Jitter: false}
	cfg := Config{}

	outcome := executeWithRetry(context.Background(), h, node, pctx, g, t.TempDir(), policy, cfg)
	if outcome.Status != state.StatusFail {
		t.Errorf("expected fail, got %q", outcome.Status)
	}
	if callCount != 2 {
		t.Errorf("expected 2 calls (max attempts), got %d", callCount)
	}
}

// ---------------------------------------------------------------------------
// Test: Pipeline with conditional branching based on outcome
// ---------------------------------------------------------------------------

func TestConditionalBranching(t *testing.T) {
	g := makeBranchingGraph()
	cfg := defaultCfg(t)

	// Register a custom handler for the conditional that returns success.
	registry := handler.NewRegistry(handler.NoopHandler{})
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})
	registry.Register("conditional", &handler.ConditionalHandler{})
	cfg.Registry = registry

	var visitedNodes []string
	cfg.OnEvent = func(e Event) {
		if e.Kind == EventStageStarted {
			if nodeID, ok := e.Data["node"].(string); ok {
				visitedNodes = append(visitedNodes, nodeID)
			}
		}
	}

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}

	// The conditional handler returns success, so condition "outcome=success" should match,
	// routing to "good".
	foundGood := false
	foundBad := false
	for _, v := range visitedNodes {
		if v == "good" {
			foundGood = true
		}
		if v == "bad" {
			foundBad = true
		}
	}
	if !foundGood {
		t.Error("expected 'good' node to be visited")
	}
	if foundBad {
		t.Error("did not expect 'bad' node to be visited")
	}
}

// ---------------------------------------------------------------------------
// Test: Pipeline max steps exceeded
// ---------------------------------------------------------------------------

func TestMaxStepsExceeded(t *testing.T) {
	// Create a graph with a loop: Start -> A -> A (self-loop with high weight)
	// and A -> Exit (low weight, so the self-loop wins and we spin until max steps).
	g := &graph.Graph{
		Name: "loop-test",
		Nodes: map[string]*graph.Node{
			"start": {ID: "start", Attrs: map[string]string{"shape": "Mdiamond"}},
			"a":     {ID: "a", Attrs: map[string]string{"shape": "box"}},
			"exit":  {ID: "exit", Attrs: map[string]string{"shape": "Msquare"}},
		},
		Edges: []*graph.Edge{
			{From: "start", To: "a", Attrs: map[string]string{}},
			{From: "a", To: "a", Attrs: map[string]string{"loop_restart": "true", "weight": "10"}},
			{From: "a", To: "exit", Attrs: map[string]string{"weight": "1"}},
		},
		Attrs: map[string]string{},
	}

	cfg := defaultCfg(t)
	cfg.MaxSteps = 5

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusFail {
		t.Errorf("expected fail from max steps, got %q", outcome.Status)
	}
	if !strings.Contains(outcome.FailureReason, "exceeded maximum step count") {
		t.Errorf("expected max steps failure reason, got %q", outcome.FailureReason)
	}
}

// ---------------------------------------------------------------------------
// Test: Context cancellation stops pipeline
// ---------------------------------------------------------------------------

func TestContextCancellation(t *testing.T) {
	g := makeLinearGraph()
	cfg := defaultCfg(t)

	// Register a handler that blocks until context is done.
	callCount := 0
	slowHandler := handler.HandlerFunc(func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
		callCount++
		if callCount > 1 {
			// Block on the second handler call.
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return &state.Outcome{Status: state.StatusSuccess}, nil
	})

	registry := handler.NewRegistry(slowHandler)
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})
	cfg.Registry = registry

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := Run(ctx, g, cfg)
	if err == nil {
		t.Fatal("expected error from context cancellation")
	}
	if !strings.Contains(err.Error(), "context") {
		t.Errorf("expected context-related error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Test: Runner.RunDOT with a simple DOT graph
// ---------------------------------------------------------------------------

func TestRunnerRunDOT(t *testing.T) {
	dotSource := `digraph pipeline {
		start [shape=Mdiamond label="Start"]
		end [shape=Msquare label="Exit"]
		start -> end
	}`

	runner := NewRunner(nil)
	runner.MaxSteps = 50

	outcome, err := runner.RunDOT(context.Background(), dotSource, t.TempDir())
	if err != nil {
		t.Fatalf("RunDOT failed: %v", err)
	}
	if outcome == nil {
		t.Fatal("expected non-nil outcome")
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: bestByWeightThenLexical with empty input
// ---------------------------------------------------------------------------

func TestBestByWeightThenLexical_Empty(t *testing.T) {
	result := bestByWeightThenLexical(nil)
	if result != nil {
		t.Errorf("expected nil for empty input, got %+v", result)
	}
}

// ---------------------------------------------------------------------------
// Test: isTerminal
// ---------------------------------------------------------------------------

func TestIsTerminal(t *testing.T) {
	g := makeStartExitGraph()

	startNode := g.Nodes["start"]
	exitNode := g.Nodes["exit"]

	if isTerminal(startNode, g) {
		t.Error("start node should not be terminal")
	}
	if !isTerminal(exitNode, g) {
		t.Error("exit node should be terminal")
	}
}

// ---------------------------------------------------------------------------
// Test: getRetryTarget resolution order
// ---------------------------------------------------------------------------

func TestGetRetryTarget(t *testing.T) {
	// Node-level retry_target has highest priority.
	node := &graph.Node{ID: "n", Attrs: map[string]string{"retry_target": "nodeTarget"}}
	g := &graph.Graph{
		Attrs: map[string]string{"retry_target": "graphTarget", "fallback_retry_target": "fallbackTarget"},
	}
	target := getRetryTarget(node, g)
	if target != "nodeTarget" {
		t.Errorf("expected 'nodeTarget', got %q", target)
	}

	// Falls through to graph-level retry_target.
	node2 := &graph.Node{ID: "n2", Attrs: map[string]string{}}
	target2 := getRetryTarget(node2, g)
	if target2 != "graphTarget" {
		t.Errorf("expected 'graphTarget', got %q", target2)
	}

	// Falls through to graph-level fallback_retry_target.
	g2 := &graph.Graph{
		Attrs: map[string]string{"fallback_retry_target": "fallbackTarget"},
	}
	target3 := getRetryTarget(node2, g2)
	if target3 != "fallbackTarget" {
		t.Errorf("expected 'fallbackTarget', got %q", target3)
	}
}

// ---------------------------------------------------------------------------
// Test: selectEdge returns nil for no outgoing edges
// ---------------------------------------------------------------------------

func TestSelectEdge_NoEdges(t *testing.T) {
	g := &graph.Graph{
		Nodes: map[string]*graph.Node{
			"n": {ID: "n", Attrs: map[string]string{}},
		},
		Edges: []*graph.Edge{},
		Attrs: map[string]string{},
	}

	node := g.Nodes["n"]
	outcome := &state.Outcome{Status: state.StatusSuccess}
	pctx := state.NewContext()

	edge := selectEdge(node, outcome, pctx, g)
	if edge != nil {
		t.Errorf("expected nil edge for node with no outgoing edges, got %+v", edge)
	}
}

// ---------------------------------------------------------------------------
// Test: Config.maxSteps defaults to 1000 when zero
// ---------------------------------------------------------------------------

func TestConfigMaxStepsDefault(t *testing.T) {
	c := Config{}
	if c.maxSteps() != 1000 {
		t.Errorf("expected default maxSteps=1000, got %d", c.maxSteps())
	}

	c2 := Config{MaxSteps: 42}
	if c2.maxSteps() != 42 {
		t.Errorf("expected maxSteps=42, got %d", c2.maxSteps())
	}
}

// ---------------------------------------------------------------------------
// Test: Event emission
// ---------------------------------------------------------------------------

func TestEventEmission(t *testing.T) {
	g := makeStartExitGraph()
	cfg := defaultCfg(t)

	var events []EventKind
	cfg.OnEvent = func(e Event) {
		events = append(events, e.Kind)
	}

	_, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should see at least pipeline_started and pipeline_completed.
	foundStarted := false
	foundCompleted := false
	for _, e := range events {
		if e == EventPipelineStarted {
			foundStarted = true
		}
		if e == EventPipelineCompleted {
			foundCompleted = true
		}
	}
	if !foundStarted {
		t.Error("expected pipeline_started event")
	}
	if !foundCompleted {
		t.Error("expected pipeline_completed event")
	}
}

// ---------------------------------------------------------------------------
// Test: Goal gate accepts partial_success
// ---------------------------------------------------------------------------

func TestCheckGoalGates_PartialSuccess(t *testing.T) {
	g := makeGoalGateGraph("start")
	pctx := state.NewContext()
	pctx.Set("status.worker", "partial_success")

	allSatisfied, failedNode := checkGoalGates(g, pctx)
	if !allSatisfied {
		t.Error("expected goal gate to accept partial_success")
	}
	if failedNode != nil {
		t.Errorf("expected no failed node, got %q", failedNode.ID)
	}
}

// ---------------------------------------------------------------------------
// Test: Context keys outcome and preferred_label are set after node execution
// ---------------------------------------------------------------------------

func TestContextKeysOutcomeAndPreferredLabel(t *testing.T) {
	g := makeLinearGraph()
	cfg := defaultCfg(t)

	// Register a handler that returns a preferred_label.
	registry := handler.NewRegistry(handler.HandlerFunc(
		func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
			return &state.Outcome{
				Status:         state.StatusSuccess,
				PreferredLabel: "my_label",
			}, nil
		},
	))
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})
	cfg.Registry = registry

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	// Verify context contains outcome and preferred_label by checking the
	// context updates in the final outcome snapshot.
	if v, ok := outcome.ContextUpdates["outcome"]; !ok {
		t.Error("expected 'outcome' key in final context")
	} else if v != "success" {
		t.Errorf("expected outcome='success', got %q", v)
	}
	if v, ok := outcome.ContextUpdates["preferred_label"]; !ok {
		t.Error("expected 'preferred_label' key in final context")
	} else if v != "my_label" {
		t.Errorf("expected preferred_label='my_label', got %q", v)
	}
}

// ---------------------------------------------------------------------------
// Test: Pipeline context keys pipeline.name and pipeline.start_time
// ---------------------------------------------------------------------------

func TestPipelineContextKeys(t *testing.T) {
	g := makeStartExitGraph()
	cfg := defaultCfg(t)

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	snap := outcome.ContextUpdates
	if snap["pipeline.name"] != "test-start-exit" {
		t.Errorf("expected pipeline.name='test-start-exit', got %q", snap["pipeline.name"])
	}
	if snap["pipeline.start_time"] == nil || snap["pipeline.start_time"] == "" {
		t.Error("expected pipeline.start_time to be set")
	}
}

// ---------------------------------------------------------------------------
// Test: max_retries off-by-one: max_retries=2 means 3 total attempts
// ---------------------------------------------------------------------------

func TestBuildRetryPolicy_MaxRetriesOffByOne(t *testing.T) {
	node := &graph.Node{ID: "n", Attrs: map[string]string{"max_retries": "2"}}
	g := &graph.Graph{Attrs: map[string]string{}}
	policy := buildRetryPolicy(node, g)
	// max_retries=2 should yield MaxAttempts=3 (2 retries + 1 initial).
	if policy.MaxAttempts != 3 {
		t.Errorf("expected MaxAttempts=3 for max_retries=2, got %d", policy.MaxAttempts)
	}
}

// ---------------------------------------------------------------------------
// Test: loop_restart resets context and completed nodes
// ---------------------------------------------------------------------------

func TestLoopRestart_ResetsContext(t *testing.T) {
	// Create a graph where:
	// start -> worker -> worker (loop_restart, high weight) -> exit (low weight)
	// The worker handler succeeds first time, sets a context key,
	// then on second invocation checks that the key was cleared.
	callCount := 0
	g := &graph.Graph{
		Name: "loop-restart-test",
		Nodes: map[string]*graph.Node{
			"start":  {ID: "start", Attrs: map[string]string{"shape": "Mdiamond"}},
			"worker": {ID: "worker", Attrs: map[string]string{"shape": "box"}},
			"exit":   {ID: "exit", Attrs: map[string]string{"shape": "Msquare"}},
		},
		Edges: []*graph.Edge{
			{From: "start", To: "worker", Attrs: map[string]string{}},
			{From: "worker", To: "worker", Attrs: map[string]string{"loop_restart": "true", "weight": "10"}},
			{From: "worker", To: "exit", Attrs: map[string]string{"weight": "1"}},
		},
		Attrs: map[string]string{},
	}

	contextCleared := false
	registry := handler.NewRegistry(handler.HandlerFunc(
		func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
			if node.ID == "worker" {
				callCount++
				if callCount == 1 {
					pctx.Set("test_key", "test_value")
				} else if callCount == 2 {
					// After loop_restart, context should be cleared.
					val := pctx.Get("test_key")
					if val == nil {
						contextCleared = true
					}
				}
			}
			return &state.Outcome{Status: state.StatusSuccess}, nil
		},
	))
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})

	cfg := Config{
		LogsRoot: t.TempDir(),
		Registry: registry,
		MaxSteps: 10, // will terminate due to max steps
	}

	Run(context.Background(), g, cfg)

	if callCount < 2 {
		t.Fatalf("expected worker to be called at least 2 times, got %d", callCount)
	}
	if !contextCleared {
		t.Error("expected context to be cleared after loop_restart")
	}
}

// ---------------------------------------------------------------------------
// Test: Manifest.json is written in run directory
// ---------------------------------------------------------------------------

func TestManifestWritten(t *testing.T) {
	g := makeStartExitGraph()
	g.Name = "manifest-test"
	cfg := defaultCfg(t)

	_, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Find the run directory.
	entries, _ := os.ReadDir(cfg.LogsRoot)
	if len(entries) == 0 {
		t.Fatal("expected run directory to exist")
	}

	manifestPath := ""
	for _, e := range entries {
		if e.IsDir() && strings.HasPrefix(e.Name(), "run-") {
			candidate := fmt.Sprintf("%s/%s/manifest.json", cfg.LogsRoot, e.Name())
			if _, err := os.Stat(candidate); err == nil {
				manifestPath = candidate
				break
			}
		}
	}
	if manifestPath == "" {
		t.Fatal("manifest.json not found in run directory")
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("failed to read manifest.json: %v", err)
	}

	if !strings.Contains(string(data), "manifest-test") {
		t.Error("manifest.json should contain the graph name")
	}
	if !strings.Contains(string(data), "start_time") {
		t.Error("manifest.json should contain start_time")
	}
}

// ---------------------------------------------------------------------------
// Test: RunDefault loads and parses the default pipeline
// ---------------------------------------------------------------------------

func TestRunnerRunDefault(t *testing.T) {
	runner := NewRunner(nil)
	runner.MaxSteps = 50

	// RunDefault should succeed at parsing and executing the pipeline.
	// With NoopHandler, all nodes return success, so the pipeline completes.
	outcome, err := runner.RunDefault(context.Background(), t.TempDir())
	if err != nil {
		t.Fatalf("RunDefault failed: %v", err)
	}
	if outcome == nil {
		t.Fatal("expected non-nil outcome")
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q (reason: %s)", outcome.Status, outcome.FailureReason)
	}
}

// ---------------------------------------------------------------------------
// Test: InitialContext values appear in final outcome
// ---------------------------------------------------------------------------

func TestInitialContext_AppliedToRun(t *testing.T) {
	g := makeStartExitGraph()
	cfg := defaultCfg(t)
	cfg.InitialContext = map[string]any{
		"seed_key":  "seed_value",
		"seed_num":  42,
	}

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Fatalf("expected success, got %q", outcome.Status)
	}

	snap := outcome.ContextUpdates
	if snap["seed_key"] != "seed_value" {
		t.Errorf("expected seed_key='seed_value', got %v", snap["seed_key"])
	}
	if snap["seed_num"] != 42 {
		t.Errorf("expected seed_num=42, got %v", snap["seed_num"])
	}
}

// ---------------------------------------------------------------------------
// Test: InitialContext nil is a no-op
// ---------------------------------------------------------------------------

func TestInitialContext_Nil(t *testing.T) {
	g := makeStartExitGraph()
	cfg := defaultCfg(t)
	cfg.InitialContext = nil

	outcome, err := Run(context.Background(), g, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Fatalf("expected success, got %q", outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: RunPipeline with unknown name returns error
// ---------------------------------------------------------------------------

func TestRunnerRunPipeline_NotFound(t *testing.T) {
	runner := NewRunner(nil)
	_, err := runner.RunPipeline(context.Background(), "nonexistent", t.TempDir())
	if err == nil {
		t.Fatal("expected error for unknown pipeline name")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' in error, got: %v", err)
	}
}
