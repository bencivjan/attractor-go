package validation

import (
	"strings"
	"testing"

	"github.com/strongdm/attractor-go/attractor/graph"
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

func makeEdge(from, to string, attrs map[string]string) *graph.Edge {
	if attrs == nil {
		attrs = map[string]string{}
	}
	return &graph.Edge{From: from, To: to, Attrs: attrs}
}

// validGraph returns a minimal graph that passes all validation rules.
func validGraph() *graph.Graph {
	return &graph.Graph{
		Name: "valid",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"work":  makeNode("work", map[string]string{"type": "codergen", "label": "Do Work", "prompt": "Generate code"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "work", nil),
			makeEdge("work", "exit", nil),
		},
		Attrs: map[string]string{},
	}
}

func findDiag(diags []Diagnostic, rule string) *Diagnostic {
	for _, d := range diags {
		if d.Rule == rule {
			return &d
		}
	}
	return nil
}

func hasDiagRule(diags []Diagnostic, rule string) bool {
	return findDiag(diags, rule) != nil
}

func countDiags(diags []Diagnostic, rule string) int {
	n := 0
	for _, d := range diags {
		if d.Rule == rule {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// Valid graph passes all rules
// ---------------------------------------------------------------------------

func TestValidate_ValidGraph(t *testing.T) {
	g := validGraph()
	diags := Validate(g)
	errs := FilterBySeverity(diags, SeverityError)
	if len(errs) > 0 {
		for _, d := range errs {
			t.Errorf("unexpected error: %s", d)
		}
	}
}

func TestValidateOrError_ValidGraph(t *testing.T) {
	g := validGraph()
	diags, err := ValidateOrError(g)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	errs := FilterBySeverity(diags, SeverityError)
	if len(errs) > 0 {
		t.Errorf("unexpected error diagnostics: %d", len(errs))
	}
}

// ---------------------------------------------------------------------------
// Missing start node -> error
// ---------------------------------------------------------------------------

func TestValidate_MissingStartNode(t *testing.T) {
	g := &graph.Graph{
		Name: "no_start",
		Nodes: map[string]*graph.Node{
			"work": makeNode("work", map[string]string{"type": "codergen", "label": "Work", "prompt": "do"}),
			"exit": makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{makeEdge("work", "exit", nil)},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	d := findDiag(diags, "start_node")
	if d == nil {
		t.Fatal("expected start_node diagnostic")
	}
	if d.Severity != SeverityError {
		t.Errorf("severity = %v, want ERROR", d.Severity)
	}
}

// ---------------------------------------------------------------------------
// Missing terminal node -> error
// ---------------------------------------------------------------------------

func TestValidate_MissingTerminalNode(t *testing.T) {
	g := &graph.Graph{
		Name: "no_exit",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"work":  makeNode("work", map[string]string{"type": "codergen", "label": "Work", "prompt": "do"}),
		},
		Edges: []*graph.Edge{makeEdge("start", "work", nil)},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	if !hasDiagRule(diags, "terminal_node") {
		t.Fatal("expected terminal_node diagnostic")
	}
	d := findDiag(diags, "terminal_node")
	if d.Severity != SeverityError {
		t.Errorf("severity = %v, want ERROR", d.Severity)
	}
}

// ---------------------------------------------------------------------------
// Multiple start nodes -> error
// ---------------------------------------------------------------------------

func TestValidate_MultipleStartNodes(t *testing.T) {
	g := &graph.Graph{
		Name: "multi_start",
		Nodes: map[string]*graph.Node{
			"s1":   makeNode("s1", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"s2":   makeNode("s2", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"exit": makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("s1", "exit", nil),
			makeEdge("s2", "exit", nil),
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	d := findDiag(diags, "start_node")
	if d == nil {
		t.Fatal("expected start_node diagnostic for multiple starts")
	}
	if d.Severity != SeverityError {
		t.Errorf("severity = %v, want ERROR", d.Severity)
	}
	if !strings.Contains(d.Message, "2") {
		t.Errorf("message should mention count of 2: %q", d.Message)
	}
}

// ---------------------------------------------------------------------------
// Unreachable nodes -> error
// ---------------------------------------------------------------------------

func TestValidate_UnreachableNode(t *testing.T) {
	g := &graph.Graph{
		Name: "unreachable",
		Nodes: map[string]*graph.Node{
			"start":    makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"exit":     makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
			"orphan":   makeNode("orphan", map[string]string{"type": "codergen", "label": "Orphan", "prompt": "do"}),
		},
		Edges: []*graph.Edge{makeEdge("start", "exit", nil)},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	reachDiags := FilterByRule(diags, "reachability")
	if len(reachDiags) == 0 {
		t.Fatal("expected reachability diagnostic for orphan node")
	}
	found := false
	for _, d := range reachDiags {
		if d.NodeID == "orphan" {
			found = true
			if d.Severity != SeverityError {
				t.Errorf("severity = %v, want ERROR", d.Severity)
			}
		}
	}
	if !found {
		t.Error("expected reachability diagnostic with NodeID='orphan'")
	}
}

// ---------------------------------------------------------------------------
// Edge targets referencing non-existent nodes -> error
// ---------------------------------------------------------------------------

func TestValidate_EdgeTargetNonExistent(t *testing.T) {
	g := &graph.Graph{
		Name: "bad_edge",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "exit", nil),
			makeEdge("start", "ghost", nil), // ghost does not exist
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	edgeDiags := FilterByRule(diags, "edge_target_exists")
	if len(edgeDiags) == 0 {
		t.Fatal("expected edge_target_exists diagnostic")
	}
	found := false
	for _, d := range edgeDiags {
		if strings.Contains(d.Message, "ghost") {
			found = true
			if d.Severity != SeverityError {
				t.Errorf("severity = %v, want ERROR", d.Severity)
			}
		}
	}
	if !found {
		t.Error("expected diagnostic mentioning 'ghost'")
	}
}

func TestValidate_EdgeSourceNonExistent(t *testing.T) {
	g := &graph.Graph{
		Name: "bad_source",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "exit", nil),
			makeEdge("phantom", "exit", nil), // phantom does not exist
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	edgeDiags := FilterByRule(diags, "edge_target_exists")
	found := false
	for _, d := range edgeDiags {
		if strings.Contains(d.Message, "phantom") {
			found = true
		}
	}
	if !found {
		t.Error("expected diagnostic mentioning 'phantom'")
	}
}

// ---------------------------------------------------------------------------
// Start node with incoming edges -> error
// ---------------------------------------------------------------------------

func TestValidate_StartNodeWithIncomingEdges(t *testing.T) {
	g := &graph.Graph{
		Name: "start_incoming",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"work":  makeNode("work", map[string]string{"type": "codergen", "label": "Work", "prompt": "do"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "work", nil),
			makeEdge("work", "exit", nil),
			makeEdge("work", "start", nil), // incoming to start
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	d := findDiag(diags, "start_no_incoming")
	if d == nil {
		t.Fatal("expected start_no_incoming diagnostic")
	}
	if d.Severity != SeverityError {
		t.Errorf("severity = %v, want ERROR", d.Severity)
	}
	if d.NodeID != "start" {
		t.Errorf("NodeID = %q, want 'start'", d.NodeID)
	}
}

// ---------------------------------------------------------------------------
// Exit node with outgoing edges -> error
// ---------------------------------------------------------------------------

func TestValidate_ExitNodeWithOutgoingEdges(t *testing.T) {
	g := &graph.Graph{
		Name: "exit_outgoing",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
			"extra": makeNode("extra", map[string]string{"type": "codergen", "label": "Extra", "prompt": "do"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "exit", nil),
			makeEdge("exit", "extra", nil), // outgoing from exit
			makeEdge("start", "extra", nil),
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	d := findDiag(diags, "exit_no_outgoing")
	if d == nil {
		t.Fatal("expected exit_no_outgoing diagnostic")
	}
	if d.Severity != SeverityError {
		t.Errorf("severity = %v, want ERROR", d.Severity)
	}
}

// ---------------------------------------------------------------------------
// Invalid condition syntax -> error
// ---------------------------------------------------------------------------

func TestValidate_InvalidConditionSyntax(t *testing.T) {
	g := &graph.Graph{
		Name: "bad_condition",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"work":  makeNode("work", map[string]string{"type": "codergen", "label": "Work", "prompt": "do"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "work", nil),
			makeEdge("work", "exit", map[string]string{"condition": "no_operator_here"}),
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	d := findDiag(diags, "condition_syntax")
	if d == nil {
		t.Fatal("expected condition_syntax diagnostic")
	}
	if d.Severity != SeverityError {
		t.Errorf("severity = %v, want ERROR", d.Severity)
	}
}

func TestValidate_ValidConditionSyntax(t *testing.T) {
	g := validGraph()
	// Add a valid condition to the edge.
	g.Edges[1].Attrs["condition"] = "outcome=success"
	diags := Validate(g)
	if hasDiagRule(diags, "condition_syntax") {
		t.Error("valid condition should not trigger condition_syntax diagnostic")
	}
}

// ---------------------------------------------------------------------------
// Unknown handler type -> warning
// ---------------------------------------------------------------------------

func TestValidate_UnknownHandlerType(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["type"] = "alien_handler"
	diags := Validate(g)
	d := findDiag(diags, "type_known")
	if d == nil {
		t.Fatal("expected type_known diagnostic for unknown handler type")
	}
	if d.Severity != SeverityWarning {
		t.Errorf("severity = %v, want WARNING", d.Severity)
	}
	if d.NodeID != "work" {
		t.Errorf("NodeID = %q, want 'work'", d.NodeID)
	}
}

func TestValidate_KnownHandlerTypes(t *testing.T) {
	knownTypes := []string{
		"start", "exit", "codergen", "wait.human",
		"conditional", "parallel", "parallel.fan_in",
		"tool", "stack.manager_loop",
	}
	for _, typ := range knownTypes {
		t.Run(typ, func(t *testing.T) {
			g := validGraph()
			g.Nodes["work"].Attrs["type"] = typ
			diags := Validate(g)
			if hasDiagRule(diags, "type_known") {
				t.Errorf("type %q should not trigger type_known warning", typ)
			}
		})
	}
}

func TestValidate_EmptyTypeNoWarning(t *testing.T) {
	g := validGraph()
	delete(g.Nodes["work"].Attrs, "type")
	diags := Validate(g)
	// Empty type should not trigger type_known.
	typeKnownDiags := FilterByRule(diags, "type_known")
	if len(typeKnownDiags) > 0 {
		t.Error("empty type should not trigger type_known warning")
	}
}

// ---------------------------------------------------------------------------
// Invalid fidelity -> warning
// ---------------------------------------------------------------------------

func TestValidate_InvalidFidelity_Node(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["fidelity"] = "superfast"
	diags := Validate(g)
	d := findDiag(diags, "fidelity_valid")
	if d == nil {
		t.Fatal("expected fidelity_valid diagnostic")
	}
	if d.Severity != SeverityWarning {
		t.Errorf("severity = %v, want WARNING", d.Severity)
	}
}

func TestValidate_InvalidFidelity_Edge(t *testing.T) {
	g := validGraph()
	g.Edges[0].Attrs["fidelity"] = "invalid_mode"
	diags := Validate(g)
	d := findDiag(diags, "fidelity_valid")
	if d == nil {
		t.Fatal("expected fidelity_valid diagnostic for edge")
	}
	if d.Severity != SeverityWarning {
		t.Errorf("severity = %v, want WARNING", d.Severity)
	}
}

func TestValidate_InvalidFidelity_GraphDefault(t *testing.T) {
	g := validGraph()
	g.Attrs["default_fidelity"] = "bad_fidelity"
	diags := Validate(g)
	fidelityDiags := FilterByRule(diags, "fidelity_valid")
	found := false
	for _, d := range fidelityDiags {
		if strings.Contains(d.Message, "default_fidelity") {
			found = true
		}
	}
	if !found {
		t.Error("expected fidelity_valid diagnostic for graph default_fidelity")
	}
}

func TestValidate_ValidFidelityModes(t *testing.T) {
	validModes := []string{"full", "truncate", "compact", "summary:low", "summary:medium", "summary:high"}
	for _, mode := range validModes {
		t.Run(mode, func(t *testing.T) {
			g := validGraph()
			g.Nodes["work"].Attrs["fidelity"] = mode
			diags := Validate(g)
			if hasDiagRule(diags, "fidelity_valid") {
				t.Errorf("fidelity %q should not trigger fidelity_valid warning", mode)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Goal gate without retry target -> warning
// ---------------------------------------------------------------------------

func TestValidate_GoalGateWithoutRetryTarget(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["goal_gate"] = "true"
	// Deliberately omit retry_target.
	delete(g.Nodes["work"].Attrs, "retry_target")
	diags := Validate(g)
	d := findDiag(diags, "goal_gate_has_retry")
	if d == nil {
		t.Fatal("expected goal_gate_has_retry diagnostic")
	}
	if d.Severity != SeverityWarning {
		t.Errorf("severity = %v, want WARNING", d.Severity)
	}
	if d.NodeID != "work" {
		t.Errorf("NodeID = %q, want 'work'", d.NodeID)
	}
}

func TestValidate_GoalGateWithRetryTarget(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["goal_gate"] = "true"
	g.Nodes["work"].Attrs["retry_target"] = "start"
	diags := Validate(g)
	if hasDiagRule(diags, "goal_gate_has_retry") {
		t.Error("goal_gate with retry_target should not trigger warning")
	}
}

// ---------------------------------------------------------------------------
// LLM node without prompt -> warning
// ---------------------------------------------------------------------------

func TestValidate_LLMNodeWithoutPrompt(t *testing.T) {
	g := &graph.Graph{
		Name: "no_prompt",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"coder": makeNode("coder", map[string]string{"type": "codergen"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "coder", nil),
			makeEdge("coder", "exit", nil),
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	d := findDiag(diags, "prompt_on_llm_nodes")
	if d == nil {
		t.Fatal("expected prompt_on_llm_nodes diagnostic")
	}
	if d.Severity != SeverityWarning {
		t.Errorf("severity = %v, want WARNING", d.Severity)
	}
}

func TestValidate_LLMNodeWithPrompt(t *testing.T) {
	g := validGraph()
	// work node already has prompt, should not trigger warning.
	diags := Validate(g)
	if hasDiagRule(diags, "prompt_on_llm_nodes") {
		t.Error("LLM node with prompt should not trigger warning")
	}
}

func TestValidate_LLMNodeWithDescriptiveLabel(t *testing.T) {
	g := &graph.Graph{
		Name: "label_only",
		Nodes: map[string]*graph.Node{
			"start": makeNode("start", map[string]string{"shape": "Mdiamond", "type": "start"}),
			"coder": makeNode("coder", map[string]string{"type": "codergen", "label": "Generate Code"}),
			"exit":  makeNode("exit", map[string]string{"shape": "Msquare", "type": "exit"}),
		},
		Edges: []*graph.Edge{
			makeEdge("start", "coder", nil),
			makeEdge("coder", "exit", nil),
		},
		Attrs: map[string]string{},
	}
	diags := Validate(g)
	// Since label differs from ID, should not trigger prompt warning.
	promptDiags := FilterByRule(diags, "prompt_on_llm_nodes")
	for _, d := range promptDiags {
		if d.NodeID == "coder" {
			t.Error("coder with descriptive label should not trigger prompt_on_llm_nodes")
		}
	}
}

// ---------------------------------------------------------------------------
// ValidateOrError returns error for ERROR severity
// ---------------------------------------------------------------------------

func TestValidateOrError_ReturnsError(t *testing.T) {
	g := &graph.Graph{
		Name:  "empty",
		Nodes: map[string]*graph.Node{},
		Edges: nil,
		Attrs: map[string]string{},
	}
	diags, err := ValidateOrError(g)
	if err == nil {
		t.Fatal("expected error from ValidateOrError on empty graph")
	}
	if !strings.Contains(err.Error(), "validation failed") {
		t.Errorf("error message = %q, should contain 'validation failed'", err.Error())
	}
	if len(diags) == 0 {
		t.Error("diagnostics should not be empty")
	}
}

func TestValidateOrError_WarningsOnly(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["fidelity"] = "bad_mode"
	diags, err := ValidateOrError(g)
	if err != nil {
		t.Errorf("expected no error for warnings-only graph, got: %v", err)
	}
	if !HasWarnings(diags) {
		t.Error("expected warnings in diagnostics")
	}
}

// ---------------------------------------------------------------------------
// Retry target references non-existent node -> warning
// ---------------------------------------------------------------------------

func TestValidate_RetryTargetNonExistent_Node(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["retry_target"] = "nonexistent_node"
	diags := Validate(g)
	d := findDiag(diags, "retry_target_exists")
	if d == nil {
		t.Fatal("expected retry_target_exists diagnostic")
	}
	if d.Severity != SeverityWarning {
		t.Errorf("severity = %v, want WARNING", d.Severity)
	}
}

func TestValidate_RetryTargetNonExistent_Graph(t *testing.T) {
	g := validGraph()
	g.Attrs["retry_target"] = "ghost_node"
	diags := Validate(g)
	retryDiags := FilterByRule(diags, "retry_target_exists")
	found := false
	for _, d := range retryDiags {
		if strings.Contains(d.Message, "ghost_node") {
			found = true
		}
	}
	if !found {
		t.Error("expected retry_target_exists diagnostic for graph-level retry_target")
	}
}

func TestValidate_FallbackRetryTargetNonExistent(t *testing.T) {
	g := validGraph()
	g.Nodes["work"].Attrs["fallback_retry_target"] = "nowhere"
	diags := Validate(g)
	retryDiags := FilterByRule(diags, "retry_target_exists")
	found := false
	for _, d := range retryDiags {
		if strings.Contains(d.Message, "nowhere") {
			found = true
		}
	}
	if !found {
		t.Error("expected retry_target_exists diagnostic for fallback_retry_target")
	}
}

// ---------------------------------------------------------------------------
// HasErrors / HasWarnings / FilterBySeverity / FilterByRule
// ---------------------------------------------------------------------------

func TestHasErrors(t *testing.T) {
	diags := []Diagnostic{
		{Rule: "r1", Severity: SeverityWarning, Message: "warn"},
	}
	if HasErrors(diags) {
		t.Error("HasErrors should return false for warnings only")
	}
	diags = append(diags, Diagnostic{Rule: "r2", Severity: SeverityError, Message: "err"})
	if !HasErrors(diags) {
		t.Error("HasErrors should return true when errors present")
	}
}

func TestHasWarnings(t *testing.T) {
	diags := []Diagnostic{
		{Rule: "r1", Severity: SeverityError, Message: "err"},
	}
	if HasWarnings(diags) {
		t.Error("HasWarnings should return false for errors only")
	}
	diags = append(diags, Diagnostic{Rule: "r2", Severity: SeverityWarning, Message: "warn"})
	if !HasWarnings(diags) {
		t.Error("HasWarnings should return true when warnings present")
	}
}

func TestFilterBySeverity(t *testing.T) {
	diags := []Diagnostic{
		{Rule: "r1", Severity: SeverityError, Message: "e1"},
		{Rule: "r2", Severity: SeverityWarning, Message: "w1"},
		{Rule: "r3", Severity: SeverityError, Message: "e2"},
		{Rule: "r4", Severity: SeverityInfo, Message: "i1"},
	}
	errors := FilterBySeverity(diags, SeverityError)
	if len(errors) != 2 {
		t.Errorf("FilterBySeverity(ERROR) = %d, want 2", len(errors))
	}
	warnings := FilterBySeverity(diags, SeverityWarning)
	if len(warnings) != 1 {
		t.Errorf("FilterBySeverity(WARNING) = %d, want 1", len(warnings))
	}
	infos := FilterBySeverity(diags, SeverityInfo)
	if len(infos) != 1 {
		t.Errorf("FilterBySeverity(INFO) = %d, want 1", len(infos))
	}
}

func TestFilterByRule(t *testing.T) {
	diags := []Diagnostic{
		{Rule: "start_node", Severity: SeverityError, Message: "a"},
		{Rule: "terminal_node", Severity: SeverityError, Message: "b"},
		{Rule: "start_node", Severity: SeverityError, Message: "c"},
	}
	startDiags := FilterByRule(diags, "start_node")
	if len(startDiags) != 2 {
		t.Errorf("FilterByRule('start_node') = %d, want 2", len(startDiags))
	}
}

// ---------------------------------------------------------------------------
// Severity.String()
// ---------------------------------------------------------------------------

func TestSeverity_String(t *testing.T) {
	tests := []struct {
		sev  Severity
		want string
	}{
		{SeverityError, "ERROR"},
		{SeverityWarning, "WARNING"},
		{SeverityInfo, "INFO"},
		{Severity(99), "UNKNOWN"},
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.sev.String(); got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Diagnostic.String()
// ---------------------------------------------------------------------------

func TestDiagnostic_String(t *testing.T) {
	d := Diagnostic{
		Rule:     "test_rule",
		Severity: SeverityError,
		Message:  "something broke",
		NodeID:   "mynode",
		Fix:      "fix it",
	}
	s := d.String()
	if !strings.Contains(s, "ERROR") {
		t.Errorf("String() should contain 'ERROR': %s", s)
	}
	if !strings.Contains(s, "test_rule") {
		t.Errorf("String() should contain rule name: %s", s)
	}
	if !strings.Contains(s, "mynode") {
		t.Errorf("String() should contain node ID: %s", s)
	}
	if !strings.Contains(s, "fix it") {
		t.Errorf("String() should contain fix: %s", s)
	}
}

func TestDiagnostic_StringWithEdge(t *testing.T) {
	edgePair := [2]string{"a", "b"}
	d := Diagnostic{
		Rule:     "test_rule",
		Severity: SeverityWarning,
		Message:  "edge issue",
		Edge:     &edgePair,
	}
	s := d.String()
	if !strings.Contains(s, "a -> b") {
		t.Errorf("String() should contain edge: %s", s)
	}
}

// ---------------------------------------------------------------------------
// Custom lint rule via extra rules
// ---------------------------------------------------------------------------

type customRule struct{}

func (customRule) Name() string { return "custom_check" }

func (customRule) Apply(g *graph.Graph) []Diagnostic {
	if g.Attrs["custom_attr"] == "" {
		return []Diagnostic{{
			Rule:     "custom_check",
			Severity: SeverityWarning,
			Message:  "missing custom_attr",
		}}
	}
	return nil
}

func TestValidate_CustomRule(t *testing.T) {
	g := validGraph()
	diags := Validate(g, customRule{})
	if !hasDiagRule(diags, "custom_check") {
		t.Error("expected custom_check diagnostic from custom rule")
	}
}

func TestValidate_CustomRulePasses(t *testing.T) {
	g := validGraph()
	g.Attrs["custom_attr"] = "present"
	diags := Validate(g, customRule{})
	if hasDiagRule(diags, "custom_check") {
		t.Error("custom_check should pass when custom_attr is present")
	}
}
