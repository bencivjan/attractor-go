// Package validation implements the pipeline linting and validation system
// for the Attractor pipeline engine. It provides a set of built-in lint rules
// that check structural correctness, reachability, and convention adherence
// of pipeline graphs before execution.
//
// Custom rules can be added by implementing the LintRule interface and passing
// them to Validate or ValidateOrError.
package validation

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/strongdm/attractor-go/attractor/condition"
	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/stylesheet"
)

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

// Severity levels for validation diagnostics.
type Severity int

const (
	// SeverityError indicates a problem that will prevent pipeline execution.
	SeverityError Severity = iota
	// SeverityWarning indicates a problem that may cause unexpected behaviour.
	SeverityWarning
	// SeverityInfo indicates a suggestion for improvement.
	SeverityInfo
)

// String returns the uppercase string representation of the severity level.
func (s Severity) String() string {
	switch s {
	case SeverityError:
		return "ERROR"
	case SeverityWarning:
		return "WARNING"
	case SeverityInfo:
		return "INFO"
	default:
		return "UNKNOWN"
	}
}

// ---------------------------------------------------------------------------
// Diagnostic
// ---------------------------------------------------------------------------

// Diagnostic represents a single validation finding. It identifies the rule
// that produced it, the severity, a human-readable message, and optional
// location information (node or edge).
type Diagnostic struct {
	Rule     string     `json:"rule"`
	Severity Severity   `json:"severity"`
	Message  string     `json:"message"`
	NodeID   string     `json:"node_id,omitempty"`
	Edge     *[2]string `json:"edge,omitempty"`
	Fix      string     `json:"fix,omitempty"`
}

// String returns a formatted diagnostic suitable for log output.
func (d Diagnostic) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[%s] %s: %s", d.Severity.String(), d.Rule, d.Message))
	if d.NodeID != "" {
		sb.WriteString(fmt.Sprintf(" (node: %s)", d.NodeID))
	}
	if d.Edge != nil {
		sb.WriteString(fmt.Sprintf(" (edge: %s -> %s)", d.Edge[0], d.Edge[1]))
	}
	if d.Fix != "" {
		sb.WriteString(fmt.Sprintf(" [fix: %s]", d.Fix))
	}
	return sb.String()
}

// ---------------------------------------------------------------------------
// LintRule interface
// ---------------------------------------------------------------------------

// LintRule is the interface for custom validation rules. Implement this to
// extend the validator with domain-specific checks.
type LintRule interface {
	// Name returns the unique identifier for this rule.
	Name() string
	// Apply runs the rule against the graph and returns any diagnostics.
	Apply(g *graph.Graph) []Diagnostic
}

// ---------------------------------------------------------------------------
// Known handler types and fidelity modes
// ---------------------------------------------------------------------------

// knownHandlerTypes is the set of recognised node type values used by the
// type_known validation rule.
var knownHandlerTypes = map[string]bool{
	"start":              true,
	"exit":               true,
	"codergen":           true,
	"wait.human":         true,
	"conditional":        true,
	"parallel":           true,
	"parallel.fan_in":    true,
	"tool":               true,
	"stack.manager_loop": true,
	"communication":     true,
}

// validFidelityModes is the set of recognised fidelity mode values used by
// the fidelity_valid validation rule.
var validFidelityModes = map[string]bool{
	"full":           true,
	"truncate":       true,
	"compact":        true,
	"summary:low":    true,
	"summary:medium": true,
	"summary:high":   true,
}

// ---------------------------------------------------------------------------
// Built-in lint rules
// ---------------------------------------------------------------------------

// startNodeRule checks that exactly one start node (shape=Mdiamond) exists.
type startNodeRule struct{}

func (startNodeRule) Name() string { return "start_node" }

func (startNodeRule) Apply(g *graph.Graph) []Diagnostic {
	var starts []*graph.Node
	for _, n := range g.Nodes {
		if strings.EqualFold(n.Shape(), "Mdiamond") {
			starts = append(starts, n)
		}
	}

	switch len(starts) {
	case 1:
		return nil
	case 0:
		return []Diagnostic{{
			Rule:     "start_node",
			Severity: SeverityError,
			Message:  "Graph must have exactly one start node (shape=Mdiamond); none found",
			Fix:      "Add a node with shape=Mdiamond as the entry point",
		}}
	default:
		ids := make([]string, len(starts))
		for i, n := range starts {
			ids[i] = n.ID
		}
		sort.Strings(ids)
		return []Diagnostic{{
			Rule:     "start_node",
			Severity: SeverityError,
			Message:  fmt.Sprintf("Graph must have exactly one start node (shape=Mdiamond); found %d: %s", len(starts), strings.Join(ids, ", ")),
			Fix:      "Remove extra start nodes so only one remains",
		}}
	}
}

// terminalNodeRule checks that at least one terminal node (shape=Msquare) exists.
type terminalNodeRule struct{}

func (terminalNodeRule) Name() string { return "terminal_node" }

func (terminalNodeRule) Apply(g *graph.Graph) []Diagnostic {
	for _, n := range g.Nodes {
		if strings.EqualFold(n.Shape(), "Msquare") {
			return nil
		}
	}
	return []Diagnostic{{
		Rule:     "terminal_node",
		Severity: SeverityError,
		Message:  "Graph must have at least one terminal node (shape=Msquare); none found",
		Fix:      "Add a node with shape=Msquare as an exit point",
	}}
}

// reachabilityRule checks that all nodes are reachable from the start node
// via BFS traversal.
type reachabilityRule struct{}

func (reachabilityRule) Name() string { return "reachability" }

func (reachabilityRule) Apply(g *graph.Graph) []Diagnostic {
	start, err := g.FindStartNode()
	if err != nil {
		// Cannot check reachability without a start node; startNodeRule
		// will separately flag this.
		return nil
	}

	reachable := bfs(start.ID, g)
	var diags []Diagnostic

	// Collect unreachable node IDs and sort for deterministic output.
	// Skip inbound communication nodes — they are external entry points
	// that receive input from other pipelines and are not reachable from
	// the start node in standalone mode.
	var unreachable []string
	for id, node := range g.Nodes {
		if !reachable[id] {
			if node.Type() == "communication" && node.Attrs["direction"] == "inbound" {
				continue
			}
			unreachable = append(unreachable, id)
		}
	}
	sort.Strings(unreachable)

	for _, nodeID := range unreachable {
		diags = append(diags, Diagnostic{
			Rule:     "reachability",
			Severity: SeverityError,
			Message:  fmt.Sprintf("Node '%s' is not reachable from start node '%s'", nodeID, start.ID),
			NodeID:   nodeID,
			Fix:      fmt.Sprintf("Add an edge path from '%s' to '%s'", start.ID, nodeID),
		})
	}
	return diags
}

// bfs performs a breadth-first search from startID and returns the set of
// reachable node IDs.
func bfs(startID string, g *graph.Graph) map[string]bool {
	visited := map[string]bool{startID: true}
	queue := []string{startID}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for _, edge := range g.OutgoingEdges(current) {
			if !visited[edge.To] {
				visited[edge.To] = true
				queue = append(queue, edge.To)
			}
		}
	}
	return visited
}

// edgeTargetExistsRule checks that all edge sources and targets reference
// existing nodes.
type edgeTargetExistsRule struct{}

func (edgeTargetExistsRule) Name() string { return "edge_target_exists" }

func (edgeTargetExistsRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, edge := range g.Edges {
		edgePair := [2]string{edge.From, edge.To}

		if _, ok := g.Nodes[edge.From]; !ok {
			diags = append(diags, Diagnostic{
				Rule:     "edge_target_exists",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Edge source '%s' does not reference an existing node", edge.From),
				Edge:     &edgePair,
				Fix:      fmt.Sprintf("Define node '%s' or correct the edge source", edge.From),
			})
		}
		if _, ok := g.Nodes[edge.To]; !ok {
			diags = append(diags, Diagnostic{
				Rule:     "edge_target_exists",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Edge target '%s' does not reference an existing node", edge.To),
				Edge:     &edgePair,
				Fix:      fmt.Sprintf("Define node '%s' or correct the edge target", edge.To),
			})
		}
	}
	return diags
}

// startNoIncomingRule checks that start nodes have no incoming edges.
type startNoIncomingRule struct{}

func (startNoIncomingRule) Name() string { return "start_no_incoming" }

func (startNoIncomingRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, n := range g.Nodes {
		if !strings.EqualFold(n.Shape(), "Mdiamond") {
			continue
		}
		incoming := g.IncomingEdges(n.ID)
		if len(incoming) > 0 {
			diags = append(diags, Diagnostic{
				Rule:     "start_no_incoming",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Start node '%s' must not have incoming edges; found %d", n.ID, len(incoming)),
				NodeID:   n.ID,
				Fix:      fmt.Sprintf("Remove incoming edges to start node '%s'", n.ID),
			})
		}
	}
	return diags
}

// exitNoOutgoingRule checks that exit nodes have no outgoing edges.
type exitNoOutgoingRule struct{}

func (exitNoOutgoingRule) Name() string { return "exit_no_outgoing" }

func (exitNoOutgoingRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, n := range g.Nodes {
		if !strings.EqualFold(n.Shape(), "Msquare") {
			continue
		}
		outgoing := g.OutgoingEdges(n.ID)
		if len(outgoing) > 0 {
			diags = append(diags, Diagnostic{
				Rule:     "exit_no_outgoing",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Exit node '%s' must not have outgoing edges; found %d", n.ID, len(outgoing)),
				NodeID:   n.ID,
				Fix:      fmt.Sprintf("Remove outgoing edges from exit node '%s'", n.ID),
			})
		}
	}
	return diags
}

// conditionSyntaxRule checks that all edge condition expressions parse
// correctly using the condition evaluator.
type conditionSyntaxRule struct{}

func (conditionSyntaxRule) Name() string { return "condition_syntax" }

func (conditionSyntaxRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, edge := range g.Edges {
		cond := edge.Condition()
		if cond == "" {
			continue
		}
		if err := condition.ParseCondition(cond); err != nil {
			edgePair := [2]string{edge.From, edge.To}
			diags = append(diags, Diagnostic{
				Rule:     "condition_syntax",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Edge %s -> %s: invalid condition syntax: %v", edge.From, edge.To, err),
				Edge:     &edgePair,
				Fix:      "Fix the condition expression syntax",
			})
		}
	}
	return diags
}

// stylesheetSyntaxRule checks that the graph's model_stylesheet attribute
// (if present) parses without errors.
type stylesheetSyntaxRule struct{}

func (stylesheetSyntaxRule) Name() string { return "stylesheet_syntax" }

func (stylesheetSyntaxRule) Apply(g *graph.Graph) []Diagnostic {
	ss := g.ModelStylesheet()
	if ss == "" {
		return nil
	}
	// The model_stylesheet attribute may be a stylesheet body or a path.
	// We attempt to parse it as a stylesheet body; if it does not look like
	// a stylesheet (no braces) we skip the check and assume it is a path
	// that will be resolved later.
	if !strings.Contains(ss, "{") {
		return nil
	}
	if _, err := stylesheet.Parse(ss); err != nil {
		return []Diagnostic{{
			Rule:     "stylesheet_syntax",
			Severity: SeverityError,
			Message:  fmt.Sprintf("Graph model_stylesheet has invalid syntax: %v", err),
			Fix:      "Fix the model_stylesheet syntax or remove the attribute",
		}}
	}
	return nil
}

// typeKnownRule warns when a node has an unrecognised type value.
type typeKnownRule struct{}

func (typeKnownRule) Name() string { return "type_known" }

func (typeKnownRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, n := range g.Nodes {
		t := n.Type()
		if t == "" {
			continue
		}
		if !knownHandlerTypes[t] {
			knownList := sortedKeys(knownHandlerTypes)
			diags = append(diags, Diagnostic{
				Rule:     "type_known",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Node '%s' has unrecognized type '%s'", n.ID, t),
				NodeID:   n.ID,
				Fix:      fmt.Sprintf("Use a recognized type: %s", strings.Join(knownList, ", ")),
			})
		}
	}
	return diags
}

// fidelityValidRule warns when a node or edge has an invalid fidelity value.
type fidelityValidRule struct{}

func (fidelityValidRule) Name() string { return "fidelity_valid" }

func (fidelityValidRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	validList := sortedKeys(validFidelityModes)

	// Check node fidelity values.
	for _, n := range g.Nodes {
		f := n.Fidelity()
		if f == "" {
			continue
		}
		if !validFidelityModes[f] {
			diags = append(diags, Diagnostic{
				Rule:     "fidelity_valid",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Node '%s' has invalid fidelity '%s'", n.ID, f),
				NodeID:   n.ID,
				Fix:      fmt.Sprintf("Use a valid fidelity: %s", strings.Join(validList, ", ")),
			})
		}
	}

	// Check edge fidelity values.
	for _, e := range g.Edges {
		f := e.Fidelity()
		if f == "" {
			continue
		}
		if !validFidelityModes[f] {
			edgePair := [2]string{e.From, e.To}
			diags = append(diags, Diagnostic{
				Rule:     "fidelity_valid",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Edge %s -> %s has invalid fidelity '%s'", e.From, e.To, f),
				Edge:     &edgePair,
				Fix:      fmt.Sprintf("Use a valid fidelity: %s", strings.Join(validList, ", ")),
			})
		}
	}

	// Check graph-level default fidelity.
	if df := g.DefaultFidelity(); df != "" && !validFidelityModes[df] {
		diags = append(diags, Diagnostic{
			Rule:     "fidelity_valid",
			Severity: SeverityWarning,
			Message:  fmt.Sprintf("Graph default_fidelity '%s' is not a valid fidelity mode", df),
			Fix:      fmt.Sprintf("Use a valid fidelity: %s", strings.Join(validList, ", ")),
		})
	}

	return diags
}

// retryTargetExistsRule warns when retry_target or fallback_retry_target
// references a non-existent node.
type retryTargetExistsRule struct{}

func (retryTargetExistsRule) Name() string { return "retry_target_exists" }

func (retryTargetExistsRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic

	// Check node-level retry targets.
	for _, n := range g.Nodes {
		if target := n.RetryTarget(); target != "" {
			if _, ok := g.Nodes[target]; !ok {
				diags = append(diags, Diagnostic{
					Rule:     "retry_target_exists",
					Severity: SeverityWarning,
					Message:  fmt.Sprintf("Node '%s' retry_target '%s' does not reference an existing node", n.ID, target),
					NodeID:   n.ID,
					Fix:      "Correct retry_target to reference an existing node",
				})
			}
		}
		if target := n.FallbackRetryTarget(); target != "" {
			if _, ok := g.Nodes[target]; !ok {
				diags = append(diags, Diagnostic{
					Rule:     "retry_target_exists",
					Severity: SeverityWarning,
					Message:  fmt.Sprintf("Node '%s' fallback_retry_target '%s' does not reference an existing node", n.ID, target),
					NodeID:   n.ID,
					Fix:      "Correct fallback_retry_target to reference an existing node",
				})
			}
		}
	}

	// Check graph-level retry targets.
	if target := g.RetryTarget(); target != "" {
		if _, ok := g.Nodes[target]; !ok {
			diags = append(diags, Diagnostic{
				Rule:     "retry_target_exists",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Graph retry_target '%s' does not reference an existing node", target),
				Fix:      "Correct graph-level retry_target to reference an existing node",
			})
		}
	}
	if target := g.FallbackRetryTarget(); target != "" {
		if _, ok := g.Nodes[target]; !ok {
			diags = append(diags, Diagnostic{
				Rule:     "retry_target_exists",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Graph fallback_retry_target '%s' does not reference an existing node", target),
				Fix:      "Correct graph-level fallback_retry_target to reference an existing node",
			})
		}
	}

	return diags
}

// goalGateHasRetryRule warns when a goal_gate=true node lacks a retry_target.
type goalGateHasRetryRule struct{}

func (goalGateHasRetryRule) Name() string { return "goal_gate_has_retry" }

func (goalGateHasRetryRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, n := range g.Nodes {
		if n.GoalGate() && n.RetryTarget() == "" {
			diags = append(diags, Diagnostic{
				Rule:     "goal_gate_has_retry",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Node '%s' has goal_gate=true but no retry_target", n.ID),
				NodeID:   n.ID,
				Fix:      "Add a retry_target attribute to the goal_gate node",
			})
		}
	}
	return diags
}

// promptOnLlmNodesRule warns when a codergen node has neither a prompt nor
// a descriptive label.
type promptOnLlmNodesRule struct{}

func (promptOnLlmNodesRule) Name() string { return "prompt_on_llm_nodes" }

func (promptOnLlmNodesRule) Apply(g *graph.Graph) []Diagnostic {
	var diags []Diagnostic
	for _, n := range g.Nodes {
		// Check both explicit type and shape-based detection.
		isCodergen := n.Type() == "codergen" || strings.EqualFold(n.Shape(), "box")
		if !isCodergen {
			continue
		}
		// Only flag if both prompt is empty and label is the default (same as ID).
		if n.Prompt() == "" && n.Label() == n.ID {
			diags = append(diags, Diagnostic{
				Rule:     "prompt_on_llm_nodes",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Node '%s' is type codergen but has no prompt or descriptive label", n.ID),
				NodeID:   n.ID,
				Fix:      "Add a prompt or label attribute to provide instructions for the LLM",
			})
		}
	}
	return diags
}

// ---------------------------------------------------------------------------
// Built-in rules list
// ---------------------------------------------------------------------------

// builtInRules is the ordered list of all built-in validation rules.
var builtInRules = []LintRule{
	startNodeRule{},
	terminalNodeRule{},
	reachabilityRule{},
	edgeTargetExistsRule{},
	startNoIncomingRule{},
	exitNoOutgoingRule{},
	conditionSyntaxRule{},
	stylesheetSyntaxRule{},
	typeKnownRule{},
	fidelityValidRule{},
	retryTargetExistsRule{},
	goalGateHasRetryRule{},
	promptOnLlmNodesRule{},
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Validate runs all built-in lint rules plus any extra rules against the
// given graph and returns all diagnostics. The diagnostics are returned in
// rule order (built-in rules first, then extra rules).
func Validate(g *graph.Graph, extraRules ...LintRule) []Diagnostic {
	allRules := make([]LintRule, 0, len(builtInRules)+len(extraRules))
	allRules = append(allRules, builtInRules...)
	allRules = append(allRules, extraRules...)

	var diags []Diagnostic
	for _, rule := range allRules {
		diags = append(diags, rule.Apply(g)...)
	}
	return diags
}

// ValidateOrError runs validation and returns an error if any ERROR-severity
// diagnostics are found. The returned diagnostics include all findings
// (errors, warnings, and info), and the error aggregates all error-severity
// messages for easy logging.
func ValidateOrError(g *graph.Graph, extraRules ...LintRule) ([]Diagnostic, error) {
	diags := Validate(g, extraRules...)

	var errMsgs []string
	for _, d := range diags {
		if d.Severity == SeverityError {
			errMsgs = append(errMsgs, d.Message)
		}
	}

	if len(errMsgs) > 0 {
		return diags, errors.New("validation failed: " + strings.Join(errMsgs, "; "))
	}
	return diags, nil
}

// HasErrors returns true if any diagnostic in the slice has ERROR severity.
func HasErrors(diags []Diagnostic) bool {
	for _, d := range diags {
		if d.Severity == SeverityError {
			return true
		}
	}
	return false
}

// HasWarnings returns true if any diagnostic in the slice has WARNING severity.
func HasWarnings(diags []Diagnostic) bool {
	for _, d := range diags {
		if d.Severity == SeverityWarning {
			return true
		}
	}
	return false
}

// FilterBySeverity returns only diagnostics matching the given severity.
func FilterBySeverity(diags []Diagnostic, sev Severity) []Diagnostic {
	var filtered []Diagnostic
	for _, d := range diags {
		if d.Severity == sev {
			filtered = append(filtered, d)
		}
	}
	return filtered
}

// FilterByRule returns only diagnostics from the named rule.
func FilterByRule(diags []Diagnostic, ruleName string) []Diagnostic {
	var filtered []Diagnostic
	for _, d := range diags {
		if d.Rule == ruleName {
			filtered = append(filtered, d)
		}
	}
	return filtered
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// sortedKeys returns the keys of a map[string]bool sorted alphabetically.
func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
