package handler

import (
	"context"
	"strings"
	"testing"

	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/interviewer"
	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeNode(id string, attrs map[string]string) *graph.Node {
	if attrs == nil {
		attrs = map[string]string{}
	}
	return &graph.Node{ID: id, Attrs: attrs}
}

func makeGraph(nodes map[string]*graph.Node, edges []*graph.Edge, attrs map[string]string) *graph.Graph {
	if nodes == nil {
		nodes = map[string]*graph.Node{}
	}
	if edges == nil {
		edges = []*graph.Edge{}
	}
	if attrs == nil {
		attrs = map[string]string{}
	}
	return &graph.Graph{Name: "test", Nodes: nodes, Edges: edges, Attrs: attrs}
}

// ---------------------------------------------------------------------------
// Test: StartHandler
// ---------------------------------------------------------------------------

func TestStartHandler(t *testing.T) {
	h := &StartHandler{}
	node := makeNode("start", map[string]string{"shape": "Mdiamond"})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	if !strings.Contains(outcome.Notes, "start") {
		t.Errorf("expected notes to contain node ID, got %q", outcome.Notes)
	}
}

// ---------------------------------------------------------------------------
// Test: ExitHandler
// ---------------------------------------------------------------------------

func TestExitHandler(t *testing.T) {
	h := &ExitHandler{}
	node := makeNode("exit", map[string]string{"shape": "Msquare"})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	if !strings.Contains(outcome.Notes, "exit") {
		t.Errorf("expected notes to contain node ID, got %q", outcome.Notes)
	}
}

// ---------------------------------------------------------------------------
// Test: ConditionalHandler
// ---------------------------------------------------------------------------

func TestConditionalHandler(t *testing.T) {
	h := &ConditionalHandler{}
	node := makeNode("cond1", map[string]string{"shape": "diamond"})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	if !strings.Contains(outcome.Notes, "cond1") {
		t.Errorf("expected notes to reference node ID, got %q", outcome.Notes)
	}
}

// ---------------------------------------------------------------------------
// Test: NoopHandler
// ---------------------------------------------------------------------------

func TestNoopHandler(t *testing.T) {
	h := NoopHandler{}
	node := makeNode("noop1", map[string]string{"type": "custom_type"})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	if !strings.Contains(outcome.Notes, "noop handler") {
		t.Errorf("expected notes to mention noop handler, got %q", outcome.Notes)
	}
}

// ---------------------------------------------------------------------------
// Test: CodergenHandler -- simulation mode (nil backend)
// ---------------------------------------------------------------------------

func TestCodergenHandler_SimulationMode(t *testing.T) {
	h := &CodergenHandler{Backend: nil}
	node := makeNode("gen1", map[string]string{
		"shape":  "box",
		"prompt": "Write a hello world program",
	})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	if outcome.ContextUpdates == nil {
		t.Fatal("expected context updates")
	}
	if outcome.ContextUpdates["last_stage"] != "gen1" {
		t.Errorf("expected last_stage='gen1', got %v", outcome.ContextUpdates["last_stage"])
	}
}

// ---------------------------------------------------------------------------
// Test: CodergenHandler -- mock backend returning string
// ---------------------------------------------------------------------------

type mockBackendString struct {
	response string
}

func (m *mockBackendString) Run(node *graph.Node, prompt string, ctx *state.Context) (any, error) {
	return m.response, nil
}

func TestCodergenHandler_BackendReturnsString(t *testing.T) {
	h := &CodergenHandler{Backend: &mockBackendString{response: "generated code here"}}
	node := makeNode("gen2", map[string]string{
		"shape":  "box",
		"prompt": "Generate code",
	})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	resp, ok := outcome.ContextUpdates["last_response"].(string)
	if !ok {
		t.Fatal("expected last_response to be a string")
	}
	if !strings.Contains(resp, "generated code here") {
		t.Errorf("expected response in context, got %q", resp)
	}
}

// ---------------------------------------------------------------------------
// Test: CodergenHandler -- mock backend returning *state.Outcome
// ---------------------------------------------------------------------------

type mockBackendOutcome struct {
	outcome *state.Outcome
}

func (m *mockBackendOutcome) Run(node *graph.Node, prompt string, ctx *state.Context) (any, error) {
	return m.outcome, nil
}

func TestCodergenHandler_BackendReturnsOutcome(t *testing.T) {
	terminalOutcome := &state.Outcome{
		Status:        state.StatusFail,
		FailureReason: "model refused",
		Notes:         "terminal failure",
	}
	h := &CodergenHandler{Backend: &mockBackendOutcome{outcome: terminalOutcome}}
	node := makeNode("gen3", map[string]string{
		"shape":  "box",
		"prompt": "Generate code",
	})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The handler should use the terminal outcome returned by the backend.
	if outcome.Status != state.StatusFail {
		t.Errorf("expected fail, got %q", outcome.Status)
	}
	if outcome.FailureReason != "model refused" {
		t.Errorf("expected failure reason 'model refused', got %q", outcome.FailureReason)
	}
}

// ---------------------------------------------------------------------------
// Test: WaitForHumanHandler with AutoApproveInterviewer
// ---------------------------------------------------------------------------

func TestWaitForHumanHandler_AutoApprove(t *testing.T) {
	h := &WaitForHumanHandler{Interviewer: &interviewer.AutoApproveInterviewer{}}
	node := makeNode("human1", map[string]string{
		"shape": "hexagon",
		"label": "Pick a direction",
	})

	nodes := map[string]*graph.Node{
		"human1": node,
		"left":   makeNode("left", nil),
		"right":  makeNode("right", nil),
		"center": makeNode("center", nil),
	}
	edges := []*graph.Edge{
		{From: "human1", To: "left", Attrs: map[string]string{"label": "[L] Left"}},
		{From: "human1", To: "right", Attrs: map[string]string{"label": "[R] Right"}},
		{From: "human1", To: "center", Attrs: map[string]string{"label": "[C] Center"}},
	}
	g := makeGraph(nodes, edges, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	// AutoApprove for multiple choice selects the first option.
	// The first option is "[L] Left" with key "L".
	if len(outcome.SuggestedNextIDs) == 0 {
		t.Fatal("expected at least one suggested next ID")
	}
	if outcome.SuggestedNextIDs[0] != "left" {
		t.Errorf("expected suggested ID 'left', got %q", outcome.SuggestedNextIDs[0])
	}
}

// ---------------------------------------------------------------------------
// Test: WaitForHumanHandler with QueueInterviewer
// ---------------------------------------------------------------------------

func TestWaitForHumanHandler_QueueInterviewer(t *testing.T) {
	// Queue an answer that selects "N" (the second option).
	qi := interviewer.NewQueueInterviewer(interviewer.Answer{
		Value: "N",
		Text:  "No",
	})

	h := &WaitForHumanHandler{Interviewer: qi}
	node := makeNode("human2", map[string]string{
		"shape": "hexagon",
		"label": "Continue?",
	})

	nodes := map[string]*graph.Node{
		"human2": node,
		"yes":    makeNode("yes", nil),
		"no":     makeNode("no", nil),
	}
	edges := []*graph.Edge{
		{From: "human2", To: "yes", Attrs: map[string]string{"label": "[Y] Yes"}},
		{From: "human2", To: "no", Attrs: map[string]string{"label": "[N] No"}},
	}
	g := makeGraph(nodes, edges, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	// Should route to "no" since the queue answer key is "N".
	foundNo := false
	for _, id := range outcome.SuggestedNextIDs {
		if id == "no" {
			foundNo = true
		}
	}
	if !foundNo {
		t.Errorf("expected 'no' in suggested next IDs, got %v", outcome.SuggestedNextIDs)
	}
}

// ---------------------------------------------------------------------------
// Test: ParallelHandler with multiple branches
// ---------------------------------------------------------------------------

func TestParallelHandler(t *testing.T) {
	reg := NewRegistry(NoopHandler{})
	reg.Register("start", &StartHandler{})
	reg.Register("exit", &ExitHandler{})

	h := NewParallelHandler(reg)

	parallelNode := makeNode("par1", map[string]string{
		"shape":        "component",
		"join_policy":  "wait_all",
		"error_policy": "fail_fast",
	})

	branch1 := makeNode("b1", map[string]string{"shape": "box"})
	branch2 := makeNode("b2", map[string]string{"shape": "box"})

	nodes := map[string]*graph.Node{
		"par1": parallelNode,
		"b1":   branch1,
		"b2":   branch2,
	}
	edges := []*graph.Edge{
		{From: "par1", To: "b1", Attrs: map[string]string{}},
		{From: "par1", To: "b2", Attrs: map[string]string{}},
	}
	g := makeGraph(nodes, edges, nil)

	outcome, err := h.Execute(context.Background(), parallelNode, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	results, ok := outcome.ContextUpdates["parallel.results"].(string)
	if !ok {
		t.Fatal("expected parallel.results in context updates")
	}
	if !strings.Contains(results, "b1") || !strings.Contains(results, "b2") {
		t.Errorf("expected both branches in results, got %q", results)
	}
}

// ---------------------------------------------------------------------------
// Test: ParallelHandler with no outgoing edges
// ---------------------------------------------------------------------------

func TestParallelHandler_NoEdges(t *testing.T) {
	reg := NewRegistry(NoopHandler{})
	h := NewParallelHandler(reg)

	parallelNode := makeNode("par2", map[string]string{"shape": "component"})
	g := makeGraph(map[string]*graph.Node{"par2": parallelNode}, nil, nil)

	outcome, err := h.Execute(context.Background(), parallelNode, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success for no-edge parallel, got %q", outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: FanInHandler selecting best branch
// ---------------------------------------------------------------------------

func TestFanInHandler(t *testing.T) {
	h := &FanInHandler{}
	node := makeNode("fanin1", map[string]string{"shape": "tripleoctagon"})
	g := makeGraph(nil, nil, nil)

	pctx := state.NewContext()
	pctx.Set("parallel.results", "branchA:fail,branchB:success,branchC:success")

	outcome, err := h.Execute(context.Background(), node, pctx, g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	bestBranch, ok := outcome.ContextUpdates["parallel.best_branch"].(string)
	if !ok {
		t.Fatal("expected parallel.best_branch in context updates")
	}
	// branchB and branchC both succeeded, lexical tiebreak should pick branchB.
	if bestBranch != "branchB" {
		t.Errorf("expected best branch 'branchB', got %q", bestBranch)
	}
}

// ---------------------------------------------------------------------------
// Test: FanInHandler with no results
// ---------------------------------------------------------------------------

func TestFanInHandler_NoResults(t *testing.T) {
	h := &FanInHandler{}
	node := makeNode("fanin2", map[string]string{})
	g := makeGraph(nil, nil, nil)
	pctx := state.NewContext()

	outcome, err := h.Execute(context.Background(), node, pctx, g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: ToolHandler with simple echo command
// ---------------------------------------------------------------------------

func TestToolHandler_Echo(t *testing.T) {
	h := &ToolHandler{}
	node := makeNode("tool1", map[string]string{
		"shape":        "parallelogram",
		"tool_command": "echo hello",
		"tool_timeout": "5",
	})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
	stdout, ok := outcome.ContextUpdates["tool.stdout"].(string)
	if !ok {
		t.Fatal("expected tool.stdout in context updates")
	}
	if !strings.Contains(stdout, "hello") {
		t.Errorf("expected stdout to contain 'hello', got %q", stdout)
	}
}

// ---------------------------------------------------------------------------
// Test: ToolHandler with no command
// ---------------------------------------------------------------------------

func TestToolHandler_NoCommand(t *testing.T) {
	h := &ToolHandler{}
	node := makeNode("tool2", map[string]string{
		"shape": "parallelogram",
		// No tool_command attribute.
	})
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusFail {
		t.Errorf("expected fail for missing tool_command, got %q", outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: StackManagerLoopHandler incrementing counter
// ---------------------------------------------------------------------------

func TestStackManagerLoopHandler(t *testing.T) {
	h := &StackManagerLoopHandler{}
	node := makeNode("loop1", map[string]string{"shape": "house"})
	g := makeGraph(nil, nil, nil)

	pctx := state.NewContext()

	// First iteration.
	outcome1, err := h.Execute(context.Background(), node, pctx, g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome1.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome1.Status)
	}
	iterVal, ok := outcome1.ContextUpdates["loop.loop1.iteration"]
	if !ok {
		t.Fatal("expected loop counter in context updates")
	}
	if iterVal != 1 {
		t.Errorf("expected iteration 1, got %v", iterVal)
	}

	// Apply updates and run again.
	pctx.ApplyUpdates(outcome1.ContextUpdates)

	outcome2, err := h.Execute(context.Background(), node, pctx, g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	iterVal2 := outcome2.ContextUpdates["loop.loop1.iteration"]
	if iterVal2 != 2 {
		t.Errorf("expected iteration 2, got %v", iterVal2)
	}
}

// ---------------------------------------------------------------------------
// Test: Registry.Resolve by type
// ---------------------------------------------------------------------------

func TestRegistry_ResolveByType(t *testing.T) {
	reg := NewRegistry(NoopHandler{})
	customHandler := &StartHandler{}
	reg.Register("my_type", customHandler)

	node := makeNode("n1", map[string]string{"type": "my_type"})
	resolved := reg.Resolve(node)

	// Should resolve to the registered handler for "my_type".
	if _, ok := resolved.(*StartHandler); !ok {
		t.Errorf("expected *StartHandler, got %T", resolved)
	}
}

// ---------------------------------------------------------------------------
// Test: Registry.Resolve by shape
// ---------------------------------------------------------------------------

func TestRegistry_ResolveByShape(t *testing.T) {
	reg := NewRegistry(NoopHandler{})
	reg.Register("codergen", &CodergenHandler{})

	// No explicit type, but shape "box" maps to "codergen" via ShapeToType.
	node := makeNode("n2", map[string]string{"shape": "box"})
	resolved := reg.Resolve(node)

	if _, ok := resolved.(*CodergenHandler); !ok {
		t.Errorf("expected *CodergenHandler via shape mapping, got %T", resolved)
	}
}

// ---------------------------------------------------------------------------
// Test: Registry.Resolve default fallback
// ---------------------------------------------------------------------------

func TestRegistry_ResolveDefaultFallback(t *testing.T) {
	reg := NewRegistry(NoopHandler{})

	// No type, no matching shape handler registered.
	node := makeNode("n3", map[string]string{"shape": "unknown_shape"})
	resolved := reg.Resolve(node)

	if _, ok := resolved.(NoopHandler); !ok {
		t.Errorf("expected NoopHandler as default, got %T", resolved)
	}
}

// ---------------------------------------------------------------------------
// Test: DefaultRegistry
// ---------------------------------------------------------------------------

func TestDefaultRegistry(t *testing.T) {
	reg := DefaultRegistry(NoopHandler{})

	// Should have start and exit registered.
	startNode := makeNode("s", map[string]string{"type": "start"})
	exitNode := makeNode("e", map[string]string{"type": "exit"})

	startHandler := reg.Resolve(startNode)
	if _, ok := startHandler.(*StartHandler); !ok {
		t.Errorf("expected *StartHandler for start type, got %T", startHandler)
	}

	exitHandler := reg.Resolve(exitNode)
	if _, ok := exitHandler.(*ExitHandler); !ok {
		t.Errorf("expected *ExitHandler for exit type, got %T", exitHandler)
	}

	// Unknown type should fall back to default.
	unknownNode := makeNode("u", map[string]string{"type": "unknown"})
	unknownHandler := reg.Resolve(unknownNode)
	if _, ok := unknownHandler.(NoopHandler); !ok {
		t.Errorf("expected NoopHandler for unknown type, got %T", unknownHandler)
	}
}

// ---------------------------------------------------------------------------
// Test: DefaultRegistryFull
// ---------------------------------------------------------------------------

func TestDefaultRegistryFull(t *testing.T) {
	reg := DefaultRegistryFull(nil, nil)

	// Verify all expected handler types are registered.
	typeTests := []struct {
		typeName    string
		expectType  string // description for error messages
		expectNotNil bool
	}{
		{"start", "StartHandler", true},
		{"exit", "ExitHandler", true},
		{"codergen", "CodergenHandler", true},
		{"wait.human", "WaitForHumanHandler", true},
		{"conditional", "ConditionalHandler", true},
		{"parallel", "ParallelHandler", true},
		{"parallel.fan_in", "FanInHandler", true},
		{"tool", "ToolHandler", true},
		{"stack.manager_loop", "StackManagerLoopHandler", true},
	}

	for _, tt := range typeTests {
		t.Run(tt.typeName, func(t *testing.T) {
			node := makeNode("test", map[string]string{"type": tt.typeName})
			h := reg.Resolve(node)
			if h == nil {
				t.Errorf("expected non-nil handler for type %q", tt.typeName)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test: HandlerFunc adapter
// ---------------------------------------------------------------------------

func TestHandlerFunc(t *testing.T) {
	called := false
	h := HandlerFunc(func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
		called = true
		return &state.Outcome{Status: state.StatusSuccess, Notes: "from func"}, nil
	})

	node := makeNode("fn1", nil)
	g := makeGraph(nil, nil, nil)

	outcome, err := h.Execute(context.Background(), node, state.NewContext(), g, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("expected HandlerFunc to be called")
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q", outcome.Status)
	}
}

// ---------------------------------------------------------------------------
// Test: buildPrompt with $goal expansion
// ---------------------------------------------------------------------------

func TestBuildPrompt_GoalExpansion(t *testing.T) {
	node := makeNode("gen", map[string]string{
		"prompt": "Please implement $goal",
	})
	pctx := state.NewContext()
	g := makeGraph(nil, nil, map[string]string{"goal": "a REST API"})

	prompt := buildPrompt(node, pctx, g)
	if !strings.Contains(prompt, "a REST API") {
		t.Errorf("expected $goal to be expanded, got %q", prompt)
	}
	if strings.Contains(prompt, "$goal") {
		t.Errorf("expected $goal to be replaced, but it remains in %q", prompt)
	}
}

// ---------------------------------------------------------------------------
// Test: buildPrompt falls back to label when prompt is empty
// ---------------------------------------------------------------------------

func TestBuildPrompt_FallbackToLabel(t *testing.T) {
	node := makeNode("gen", map[string]string{
		"label": "Generate tests",
	})
	pctx := state.NewContext()
	g := makeGraph(nil, nil, nil)

	prompt := buildPrompt(node, pctx, g)
	if prompt != "Generate tests" {
		t.Errorf("expected label fallback, got %q", prompt)
	}
}

// ---------------------------------------------------------------------------
// Test: ShapeToType mapping
// ---------------------------------------------------------------------------

func TestShapeToType(t *testing.T) {
	expected := map[string]string{
		"Mdiamond":      "start",
		"Msquare":       "exit",
		"box":           "codergen",
		"hexagon":       "wait.human",
		"diamond":       "conditional",
		"component":     "parallel",
		"tripleoctagon": "parallel.fan_in",
		"parallelogram": "tool",
		"house":         "stack.manager_loop",
	}

	for shape, expectedType := range expected {
		if got, ok := ShapeToType[shape]; !ok {
			t.Errorf("ShapeToType missing shape %q", shape)
		} else if got != expectedType {
			t.Errorf("ShapeToType[%q] = %q, want %q", shape, got, expectedType)
		}
	}
}
