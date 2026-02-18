package graph

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Graph represents a parsed pipeline definition built from a DOT digraph.
// Nodes are keyed by their string ID for O(1) lookup.
type Graph struct {
	Name  string
	Nodes map[string]*Node
	Edges []*Edge
	Attrs map[string]string // graph-level attributes
}

// Goal returns the pipeline goal description.
func (g *Graph) Goal() string { return g.Attrs["goal"] }

// Label returns the human-readable pipeline label.
func (g *Graph) Label() string { return g.Attrs["label"] }

// ModelStylesheet returns the path/name of the model stylesheet to apply.
func (g *Graph) ModelStylesheet() string { return g.Attrs["model_stylesheet"] }

// DefaultMaxRetry returns the default maximum retry count for the pipeline.
// Falls back to 50 if the attribute is missing or unparseable.
func (g *Graph) DefaultMaxRetry() int {
	return attrInt(g.Attrs, "default_max_retry", 50)
}

// RetryTarget returns the default retry target node ID, or empty string if unset.
func (g *Graph) RetryTarget() string { return g.Attrs["retry_target"] }

// FallbackRetryTarget returns the fallback retry target node ID, or empty string if unset.
func (g *Graph) FallbackRetryTarget() string { return g.Attrs["fallback_retry_target"] }

// DefaultFidelity returns the default fidelity level for the pipeline.
func (g *Graph) DefaultFidelity() string { return g.Attrs["default_fidelity"] }

// OutgoingEdges returns all edges originating from the given node.
func (g *Graph) OutgoingEdges(nodeID string) []*Edge {
	var out []*Edge
	for _, e := range g.Edges {
		if e.From == nodeID {
			out = append(out, e)
		}
	}
	return out
}

// IncomingEdges returns all edges arriving at the given node.
func (g *Graph) IncomingEdges(nodeID string) []*Edge {
	var in []*Edge
	for _, e := range g.Edges {
		if e.To == nodeID {
			in = append(in, e)
		}
	}
	return in
}

// FindStartNode locates the pipeline entry point. A start node is identified by
// shape=Mdiamond or by having the ID "start" or "Start".
func (g *Graph) FindStartNode() (*Node, error) {
	for _, n := range g.Nodes {
		if strings.EqualFold(n.Shape(), "Mdiamond") {
			return n, nil
		}
	}
	for _, id := range []string{"start", "Start"} {
		if n, ok := g.Nodes[id]; ok {
			return n, nil
		}
	}
	return nil, fmt.Errorf("no start node found (expected shape=Mdiamond or id=start)")
}

// FindExitNode locates the pipeline exit point. An exit node is identified by
// shape=Msquare or by having the ID "exit" or "end".
func (g *Graph) FindExitNode() (*Node, error) {
	for _, n := range g.Nodes {
		if strings.EqualFold(n.Shape(), "Msquare") {
			return n, nil
		}
	}
	for _, id := range []string{"exit", "end"} {
		if n, ok := g.Nodes[id]; ok {
			return n, nil
		}
	}
	return nil, fmt.Errorf("no exit node found (expected shape=Msquare or id=exit/end)")
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

// Node represents a single pipeline stage.
type Node struct {
	ID    string
	Attrs map[string]string
}

// Label returns the display label, defaulting to the node ID.
func (n *Node) Label() string { return attrString(n.Attrs, "label", n.ID) }

// Shape returns the DOT shape, defaulting to "box".
func (n *Node) Shape() string { return attrString(n.Attrs, "shape", "box") }

// Type returns the handler type (e.g., "tool", "conditional").
func (n *Node) Type() string { return n.Attrs["type"] }

// Prompt returns the LLM prompt text associated with this node.
func (n *Node) Prompt() string { return n.Attrs["prompt"] }

// MaxRetries returns the maximum number of retries, defaulting to 0.
func (n *Node) MaxRetries() int { return attrInt(n.Attrs, "max_retries", 0) }

// GoalGate returns whether this node is a goal gate checkpoint.
func (n *Node) GoalGate() bool { return attrBool(n.Attrs, "goal_gate") }

// RetryTarget returns the node ID to retry from on failure.
func (n *Node) RetryTarget() string { return n.Attrs["retry_target"] }

// FallbackRetryTarget returns the fallback node ID to retry from.
func (n *Node) FallbackRetryTarget() string { return n.Attrs["fallback_retry_target"] }

// Fidelity returns the fidelity level for this node.
func (n *Node) Fidelity() string { return n.Attrs["fidelity"] }

// ThreadID returns the thread scope identifier.
func (n *Node) ThreadID() string { return n.Attrs["thread_id"] }

// Class returns the CSS-style class name for stylesheet matching.
func (n *Node) Class() string { return n.Attrs["class"] }

// Timeout returns the execution timeout as a time.Duration.
// Returns 0 if the attribute is missing or unparseable.
func (n *Node) Timeout() time.Duration {
	raw, ok := n.Attrs["timeout"]
	if !ok || raw == "" {
		return 0
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0
	}
	return d
}

// LLMModel returns the specific LLM model to use.
// Checks both "model" and "llm_model" attributes for compatibility.
func (n *Node) LLMModel() string {
	if v := n.Attrs["model"]; v != "" {
		return v
	}
	return n.Attrs["llm_model"]
}

// LLMProvider returns the LLM provider name (e.g., "openai", "anthropic").
func (n *Node) LLMProvider() string { return n.Attrs["llm_provider"] }

// ReasoningEffort returns the reasoning effort level, defaulting to "high".
func (n *Node) ReasoningEffort() string {
	return attrString(n.Attrs, "reasoning_effort", "high")
}

// AutoStatus returns whether the node automatically reports status.
func (n *Node) AutoStatus() bool { return attrBool(n.Attrs, "auto_status") }

// AllowPartial returns whether partial results are acceptable.
func (n *Node) AllowPartial() bool { return attrBool(n.Attrs, "allow_partial") }

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

// Edge represents a directed transition between two nodes.
type Edge struct {
	From  string
	To    string
	Attrs map[string]string
}

// Label returns the edge display label.
func (e *Edge) Label() string { return e.Attrs["label"] }

// Condition returns the condition expression that gates this transition.
func (e *Edge) Condition() string { return e.Attrs["condition"] }

// Weight returns the edge weight for priority ordering, defaulting to 0
// (Section 2.7).
func (e *Edge) Weight() int { return attrInt(e.Attrs, "weight", 0) }

// Fidelity returns the fidelity level for this edge.
func (e *Edge) Fidelity() string { return e.Attrs["fidelity"] }

// ThreadID returns the thread scope for this edge.
func (e *Edge) ThreadID() string { return e.Attrs["thread_id"] }

// LoopRestart returns whether traversing this edge restarts the loop context.
func (e *Edge) LoopRestart() bool { return attrBool(e.Attrs, "loop_restart") }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// attrString reads a string attribute with a fallback default.
func attrString(attrs map[string]string, key, defaultVal string) string {
	if v, ok := attrs[key]; ok && v != "" {
		return v
	}
	return defaultVal
}

// attrInt reads an integer attribute with a fallback default.
func attrInt(attrs map[string]string, key string, defaultVal int) int {
	raw, ok := attrs[key]
	if !ok || raw == "" {
		return defaultVal
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return defaultVal
	}
	return v
}

// attrBool reads a boolean attribute. Returns true only if the value is exactly "true".
func attrBool(attrs map[string]string, key string) bool {
	return attrs[key] == "true"
}
