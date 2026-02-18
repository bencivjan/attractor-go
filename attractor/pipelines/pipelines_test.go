package pipelines

import (
	"strings"
	"testing"

	"github.com/strongdm/attractor-go/attractor/parser"
	"github.com/strongdm/attractor-go/attractor/validation"
)

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

	expectedNodes := []string{"start", "high_level_plan", "sprint_breakdown", "implement", "qa_verify", "exit"}
	for _, id := range expectedNodes {
		if _, ok := g.Nodes[id]; !ok {
			t.Errorf("expected node %q not found in graph", id)
		}
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

	// Verify QA pass/fail edges.
	var qaToExit, qaToImplement bool
	for _, edge := range g.Edges {
		if edge.From == "qa_verify" && edge.To == "exit" {
			qaToExit = true
		}
		if edge.From == "qa_verify" && edge.To == "implement" {
			qaToImplement = true
			if !edge.LoopRestart() {
				t.Error("qa_verify -> implement edge should have loop_restart=true")
			}
		}
	}
	if !qaToExit {
		t.Error("expected qa_verify -> exit edge")
	}
	if !qaToImplement {
		t.Error("expected qa_verify -> implement feedback edge")
	}
}

func TestGet_DefaultName(t *testing.T) {
	dot := Get(DefaultName)
	if dot == "" {
		t.Error("Get(DefaultName) returned empty string")
	}
	if dot != Default() {
		t.Error("Get(DefaultName) should return same as Default()")
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
	found := false
	for _, n := range names {
		if n == DefaultName {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Names() should include %q, got %v", DefaultName, names)
	}
}
