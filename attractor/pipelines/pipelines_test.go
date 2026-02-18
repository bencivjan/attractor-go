package pipelines

import (
	"strings"
	"testing"

	"github.com/strongdm/attractor-go/attractor/parser"
	"github.com/strongdm/attractor-go/attractor/validation"
)

// ---------------------------------------------------------------------------
// Developer pipeline (Default)
// ---------------------------------------------------------------------------

func TestDefault_NotEmpty(t *testing.T) {
	dot := Default()
	if dot == "" {
		t.Fatal("Default() returned empty string")
	}
	if !strings.Contains(dot, "plan_build_verify") {
		t.Error("Default pipeline should contain 'plan_build_verify' graph name")
	}
}

func TestDefault_Parses(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("Default pipeline failed to parse: %v", err)
	}
	if g.Name != "plan_build_verify" {
		t.Errorf("expected graph name 'plan_build_verify', got %q", g.Name)
	}
}

func TestDefault_HasExpectedNodes(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	expectedNodes := []string{
		"start", "high_level_plan", "sprint_breakdown", "implement", "qa_verify",
		"submit_for_evaluation", "evaluation_feedback", "exit",
	}
	for _, id := range expectedNodes {
		if _, ok := g.Nodes[id]; !ok {
			t.Errorf("expected node %q not found in graph", id)
		}
	}
}

func TestDefault_CommunicationNodes(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	tests := []struct {
		nodeID    string
		shape     string
		direction string
	}{
		{"submit_for_evaluation", "doubleoctagon", "outbound"},
		{"evaluation_feedback", "doubleoctagon", "inbound"},
	}

	for _, tt := range tests {
		t.Run(tt.nodeID, func(t *testing.T) {
			node, ok := g.Nodes[tt.nodeID]
			if !ok {
				t.Fatalf("node %q not found", tt.nodeID)
			}
			if node.Shape() != tt.shape {
				t.Errorf("expected shape %q, got %q", tt.shape, node.Shape())
			}
			if node.Type() != "communication" {
				t.Errorf("expected type 'communication', got %q", node.Type())
			}
			if node.Attrs["direction"] != tt.direction {
				t.Errorf("expected direction %q, got %q", tt.direction, node.Attrs["direction"])
			}
		})
	}
}

func TestDefault_ModelAssignments(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	tests := []struct {
		nodeID   string
		model    string
		provider string
	}{
		{"high_level_plan", "claude-opus-4-6", "anthropic"},
		{"sprint_breakdown", "claude-opus-4-6", "anthropic"},
		{"implement", "gpt-5.3-codex", "openai"},
		{"qa_verify", "claude-opus-4-6", "anthropic"},
	}

	for _, tt := range tests {
		t.Run(tt.nodeID, func(t *testing.T) {
			node, ok := g.Nodes[tt.nodeID]
			if !ok {
				t.Fatalf("node %q not found", tt.nodeID)
			}
			if node.LLMModel() != tt.model {
				t.Errorf("node %q: expected model %q, got %q", tt.nodeID, tt.model, node.LLMModel())
			}
			if node.LLMProvider() != tt.provider {
				t.Errorf("node %q: expected provider %q, got %q", tt.nodeID, tt.provider, node.LLMProvider())
			}
		})
	}
}

func TestDefault_GoalGates(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	goalGateNodes := []string{"high_level_plan", "sprint_breakdown", "implement", "qa_verify"}
	for _, id := range goalGateNodes {
		node := g.Nodes[id]
		if !node.GoalGate() {
			t.Errorf("expected node %q to be a goal gate", id)
		}
	}
}

func TestDefault_Validates(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	diags, err := validation.ValidateOrError(g)
	if err != nil {
		t.Fatalf("default pipeline failed validation: %v", err)
	}
	if validation.HasErrors(diags) {
		for _, d := range validation.FilterBySeverity(diags, validation.SeverityError) {
			t.Errorf("validation error: %s", d)
		}
	}
}

func TestDefault_EdgeStructure(t *testing.T) {
	g, err := parser.Parse(Default())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	// Verify the main flow edges exist.
	expectedEdges := []struct{ from, to string }{
		{"start", "high_level_plan"},
		{"high_level_plan", "sprint_breakdown"},
		{"sprint_breakdown", "implement"},
		{"implement", "qa_verify"},
		{"submit_for_evaluation", "exit"},
		{"evaluation_feedback", "implement"},
	}

	for _, ee := range expectedEdges {
		found := false
		for _, edge := range g.Edges {
			if edge.From == ee.from && edge.To == ee.to {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected edge %s -> %s not found", ee.from, ee.to)
		}
	}

	// Verify QA pass now goes to submit_for_evaluation instead of exit.
	var qaToSubmit, qaToImplement bool
	for _, edge := range g.Edges {
		if edge.From == "qa_verify" && edge.To == "submit_for_evaluation" {
			qaToSubmit = true
		}
		if edge.From == "qa_verify" && edge.To == "implement" {
			qaToImplement = true
			if !edge.LoopRestart() {
				t.Error("qa_verify -> implement edge should have loop_restart=true")
			}
		}
	}
	if !qaToSubmit {
		t.Error("expected qa_verify -> submit_for_evaluation edge")
	}
	if !qaToImplement {
		t.Error("expected qa_verify -> implement feedback edge")
	}
}

// ---------------------------------------------------------------------------
// Evaluator pipeline
// ---------------------------------------------------------------------------

func TestEvaluator_NotEmpty(t *testing.T) {
	dot := Evaluator()
	if dot == "" {
		t.Fatal("Evaluator() returned empty string")
	}
	if !strings.Contains(dot, "evaluator") {
		t.Error("Evaluator pipeline should contain 'evaluator' graph name")
	}
}

func TestEvaluator_Parses(t *testing.T) {
	g, err := parser.Parse(Evaluator())
	if err != nil {
		t.Fatalf("Evaluator pipeline failed to parse: %v", err)
	}
	if g.Name != "evaluator" {
		t.Errorf("expected graph name 'evaluator', got %q", g.Name)
	}
}

func TestEvaluator_HasExpectedNodes(t *testing.T) {
	g, err := parser.Parse(Evaluator())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	expectedNodes := []string{
		"start", "receive_submission", "orchestrator", "builder", "qa", "visionary",
		"return_feedback", "exit",
	}
	for _, id := range expectedNodes {
		if _, ok := g.Nodes[id]; !ok {
			t.Errorf("expected node %q not found in graph", id)
		}
	}
}

func TestEvaluator_CommunicationNodes(t *testing.T) {
	g, err := parser.Parse(Evaluator())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	tests := []struct {
		nodeID    string
		direction string
	}{
		{"receive_submission", "inbound"},
		{"return_feedback", "outbound"},
	}

	for _, tt := range tests {
		t.Run(tt.nodeID, func(t *testing.T) {
			node, ok := g.Nodes[tt.nodeID]
			if !ok {
				t.Fatalf("node %q not found", tt.nodeID)
			}
			if node.Shape() != "doubleoctagon" {
				t.Errorf("expected shape 'doubleoctagon', got %q", node.Shape())
			}
			if node.Type() != "communication" {
				t.Errorf("expected type 'communication', got %q", node.Type())
			}
			if node.Attrs["direction"] != tt.direction {
				t.Errorf("expected direction %q, got %q", tt.direction, node.Attrs["direction"])
			}
		})
	}
}

func TestEvaluator_Validates(t *testing.T) {
	g, err := parser.Parse(Evaluator())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	diags, err := validation.ValidateOrError(g)
	if err != nil {
		t.Fatalf("evaluator pipeline failed validation: %v", err)
	}
	if validation.HasErrors(diags) {
		for _, d := range validation.FilterBySeverity(diags, validation.SeverityError) {
			t.Errorf("validation error: %s", d)
		}
	}
}

func TestEvaluator_EdgeStructure(t *testing.T) {
	g, err := parser.Parse(Evaluator())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	expectedEdges := []struct{ from, to string }{
		{"start", "receive_submission"},
		{"receive_submission", "orchestrator"},
		{"orchestrator", "builder"},
		{"builder", "qa"},
		{"qa", "visionary"},
		{"return_feedback", "exit"},
	}

	for _, ee := range expectedEdges {
		found := false
		for _, edge := range g.Edges {
			if edge.From == ee.from && edge.To == ee.to {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected edge %s -> %s not found", ee.from, ee.to)
		}
	}

	// Verify visionary conditional edges.
	var visionaryToExit, visionaryToOrch, visionaryToFeedback bool
	for _, edge := range g.Edges {
		if edge.From == "visionary" {
			switch edge.To {
			case "exit":
				visionaryToExit = true
			case "orchestrator":
				visionaryToOrch = true
			case "return_feedback":
				visionaryToFeedback = true
			}
		}
	}
	if !visionaryToExit {
		t.Error("expected visionary -> exit edge")
	}
	if !visionaryToOrch {
		t.Error("expected visionary -> orchestrator (retry) edge")
	}
	if !visionaryToFeedback {
		t.Error("expected visionary -> return_feedback (fail) edge")
	}
}

// ---------------------------------------------------------------------------
// Factory pipeline
// ---------------------------------------------------------------------------

func TestFactory_NotEmpty(t *testing.T) {
	dot := Factory()
	if dot == "" {
		t.Fatal("Factory() returned empty string")
	}
	if !strings.Contains(dot, "factory") {
		t.Error("Factory pipeline should contain 'factory' graph name")
	}
}

func TestFactory_Parses(t *testing.T) {
	g, err := parser.Parse(Factory())
	if err != nil {
		t.Fatalf("Factory pipeline failed to parse: %v", err)
	}
	if g.Name != "factory" {
		t.Errorf("expected graph name 'factory', got %q", g.Name)
	}
}

func TestFactory_HasExpectedNodes(t *testing.T) {
	g, err := parser.Parse(Factory())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	expectedNodes := []string{
		"start", "high_level_plan", "sprint_breakdown", "implement", "qa_verify",
		"eval_orchestrator", "eval_builder", "eval_qa", "eval_visionary", "exit",
	}
	for _, id := range expectedNodes {
		if _, ok := g.Nodes[id]; !ok {
			t.Errorf("expected node %q not found in graph", id)
		}
	}
}

func TestFactory_Validates(t *testing.T) {
	g, err := parser.Parse(Factory())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	diags, err := validation.ValidateOrError(g)
	if err != nil {
		t.Fatalf("factory pipeline failed validation: %v", err)
	}
	if validation.HasErrors(diags) {
		for _, d := range validation.FilterBySeverity(diags, validation.SeverityError) {
			t.Errorf("validation error: %s", d)
		}
	}
}

func TestFactory_EdgeStructure(t *testing.T) {
	g, err := parser.Parse(Factory())
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	// Developer phase flow.
	devEdges := []struct{ from, to string }{
		{"start", "high_level_plan"},
		{"high_level_plan", "sprint_breakdown"},
		{"sprint_breakdown", "implement"},
		{"implement", "qa_verify"},
	}
	for _, ee := range devEdges {
		found := false
		for _, edge := range g.Edges {
			if edge.From == ee.from && edge.To == ee.to {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected developer edge %s -> %s not found", ee.from, ee.to)
		}
	}

	// Handoff: qa_verify success → eval_orchestrator.
	var qaToEval bool
	for _, edge := range g.Edges {
		if edge.From == "qa_verify" && edge.To == "eval_orchestrator" {
			qaToEval = true
			break
		}
	}
	if !qaToEval {
		t.Error("expected qa_verify -> eval_orchestrator handoff edge")
	}

	// Evaluator phase flow.
	evalEdges := []struct{ from, to string }{
		{"eval_orchestrator", "eval_builder"},
		{"eval_builder", "eval_qa"},
		{"eval_qa", "eval_visionary"},
	}
	for _, ee := range evalEdges {
		found := false
		for _, edge := range g.Edges {
			if edge.From == ee.from && edge.To == ee.to {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected evaluator edge %s -> %s not found", ee.from, ee.to)
		}
	}

	// Visionary outcomes: success→exit, retry→eval_orchestrator, fail→implement.
	var vToExit, vToOrch, vToImpl bool
	for _, edge := range g.Edges {
		if edge.From == "eval_visionary" {
			switch edge.To {
			case "exit":
				vToExit = true
			case "eval_orchestrator":
				vToOrch = true
			case "implement":
				vToImpl = true
			}
		}
	}
	if !vToExit {
		t.Error("expected eval_visionary -> exit edge")
	}
	if !vToOrch {
		t.Error("expected eval_visionary -> eval_orchestrator (retry) edge")
	}
	if !vToImpl {
		t.Error("expected eval_visionary -> implement (fail/rejected) edge")
	}
}

// ---------------------------------------------------------------------------
// Catalog functions
// ---------------------------------------------------------------------------

func TestGet_DefaultName(t *testing.T) {
	dot := Get(DefaultName)
	if dot == "" {
		t.Error("Get(DefaultName) returned empty string")
	}
	if dot != Default() {
		t.Error("Get(DefaultName) should return same as Default()")
	}
}

func TestGet_EvaluatorName(t *testing.T) {
	dot := Get(EvaluatorName)
	if dot == "" {
		t.Error("Get(EvaluatorName) returned empty string")
	}
	if dot != Evaluator() {
		t.Error("Get(EvaluatorName) should return same as Evaluator()")
	}
}

func TestGet_FactoryName(t *testing.T) {
	dot := Get(FactoryName)
	if dot == "" {
		t.Error("Get(FactoryName) returned empty string")
	}
	if dot != Factory() {
		t.Error("Get(FactoryName) should return same as Factory()")
	}
}

func TestGet_Unknown(t *testing.T) {
	dot := Get("nonexistent_pipeline")
	if dot != "" {
		t.Error("Get() for unknown name should return empty string")
	}
}

func TestRegister(t *testing.T) {
	Register("test_pipeline", "digraph test { start [shape=Mdiamond]; exit [shape=Msquare]; start -> exit }")
	dot := Get("test_pipeline")
	if dot == "" {
		t.Error("registered pipeline should be retrievable")
	}
	// Clean up.
	delete(catalog, "test_pipeline")
}

func TestNames(t *testing.T) {
	names := Names()
	expected := map[string]bool{
		DefaultName:   false,
		EvaluatorName: false,
		FactoryName:   false,
	}
	for _, n := range names {
		if _, ok := expected[n]; ok {
			expected[n] = true
		}
	}
	for name, found := range expected {
		if !found {
			t.Errorf("Names() should include %q, got %v", name, names)
		}
	}
}
