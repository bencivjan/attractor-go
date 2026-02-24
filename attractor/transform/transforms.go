// Package transform implements AST transforms that modify a pipeline graph
// before execution begins. Transforms are applied in order after parsing and
// before validation, enabling variable expansion, stylesheet application,
// and other graph rewriting operations.
package transform

import (
	"strings"

	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/stylesheet"
)

// ---------------------------------------------------------------------------
// Transform interface
// ---------------------------------------------------------------------------

// Transform modifies a pipeline graph and returns the (possibly new) graph.
// Transforms must not mutate the input graph if they intend to change the
// structure -- they should produce a new graph with updated nodes/edges.
type Transform interface {
	Apply(g *graph.Graph) *graph.Graph
}

// ---------------------------------------------------------------------------
// ApplyAll
// ---------------------------------------------------------------------------

// ApplyAll applies a sequence of transforms to a graph in order,
// threading the output of each transform as input to the next.
func ApplyAll(g *graph.Graph, transforms ...Transform) *graph.Graph {
	for _, t := range transforms {
		g = t.Apply(g)
	}
	return g
}

// DefaultTransforms returns the standard transform pipeline that should be
// applied before validation and execution. The order matters:
//  1. VariableExpansion  -- expand $goal references in node attributes
//  2. StylesheetApplication -- apply model_stylesheet rules to nodes
func DefaultTransforms() []Transform {
	return []Transform{
		&VariableExpansion{},
		&StylesheetApplication{},
	}
}

// ---------------------------------------------------------------------------
// VariableExpansion
// ---------------------------------------------------------------------------

// VariableExpansion replaces $goal references in node attribute values with
// the graph-level goal attribute. This allows pipeline authors to write
// prompts that reference the overall goal without hard-coding it.
//
// If the graph has no goal attribute, this transform is a no-op.
type VariableExpansion struct{}

func (t *VariableExpansion) Apply(g *graph.Graph) *graph.Graph {
	goal := g.Goal()
	if goal == "" {
		return g
	}

	updatedNodes := make(map[string]*graph.Node, len(g.Nodes))
	for id, node := range g.Nodes {
		expandedAttrs := make(map[string]string, len(node.Attrs))
		for k, v := range node.Attrs {
			expandedAttrs[k] = expandVariables(v, goal)
		}
		updatedNodes[id] = &graph.Node{
			ID:    node.ID,
			Attrs: expandedAttrs,
		}
	}

	return &graph.Graph{
		Name:  g.Name,
		Nodes: updatedNodes,
		Edges: g.Edges,
		Attrs: g.Attrs,
	}
}

// expandVariables replaces all occurrences of $goal with the goal value.
func expandVariables(value, goal string) string {
	return strings.ReplaceAll(value, "$goal", goal)
}

// ---------------------------------------------------------------------------
// StylesheetApplication
// ---------------------------------------------------------------------------

// StylesheetApplication applies model_stylesheet graph attribute rules to
// matching nodes using the CSS-like stylesheet parser (Section 8.2).
//
// Stylesheet syntax:
//
//	selector { key: value; key2: value2; }
//
// Selectors:
//   - "*"          -> all nodes (universal)
//   - ".classname" -> nodes with matching class attribute
//   - "#nodeId"    -> specific node by ID
//   - "shapename"  -> nodes with matching shape attribute
//
// Rules are applied with specificity ordering (universal < shape < class < ID).
// Stylesheet attributes do NOT override attributes already explicitly set on
// a node. This ensures that node-level declarations always take precedence.
type StylesheetApplication struct{}

func (t *StylesheetApplication) Apply(g *graph.Graph) *graph.Graph {
	src := g.ModelStylesheet()
	if src == "" {
		return g
	}

	rules, err := stylesheet.Parse(src)
	if err != nil {
		// Malformed stylesheet -- skip silently to avoid blocking execution.
		return g
	}
	if len(rules) == 0 {
		return g
	}

	updatedNodes := make(map[string]*graph.Node, len(g.Nodes))
	for id, node := range g.Nodes {
		mergedAttrs := copyAttrs(node.Attrs)

		// Resolve effective properties from stylesheet rules with specificity.
		classes := splitClasses(node.Class())
		resolved := stylesheet.Apply(rules, node.ID, node.Shape(), classes)

		// Stylesheet properties do not override existing node attributes.
		for key, value := range resolved {
			if _, exists := mergedAttrs[key]; !exists {
				mergedAttrs[key] = value
			}
		}

		updatedNodes[id] = &graph.Node{
			ID:    node.ID,
			Attrs: mergedAttrs,
		}
	}

	return &graph.Graph{
		Name:  g.Name,
		Nodes: updatedNodes,
		Edges: g.Edges,
		Attrs: g.Attrs,
	}
}

// splitClasses splits a space-separated class attribute into individual class names.
func splitClasses(class string) []string {
	if class == "" {
		return nil
	}
	return strings.Fields(class)
}

// copyAttrs returns a shallow copy of an attribute map.
func copyAttrs(attrs map[string]string) map[string]string {
	out := make(map[string]string, len(attrs))
	for k, v := range attrs {
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// PreambleTransform
// ---------------------------------------------------------------------------

// PreambleTransform synthesises context carryover text for pipeline stages
// that do not use "full" fidelity mode. When the graph or individual nodes
// specify a fidelity mode other than "full", this transform prepends a
// preamble note to the node's prompt explaining that earlier conversation
// context has been summarised or truncated according to the configured
// fidelity mode.
//
// Unlike the other transforms which operate purely on AST attributes,
// this transform is aware of runtime context. It is designed to be applied
// at execution time (not parse time) via runner.RegisterTransform.
type PreambleTransform struct {
	// GraphFidelity is the default fidelity mode from the graph. If empty,
	// defaults to "compact".
	GraphFidelity string
}

func (t *PreambleTransform) Apply(g *graph.Graph) *graph.Graph {
	graphFidelity := t.GraphFidelity
	if graphFidelity == "" {
		graphFidelity = g.DefaultFidelity()
		if graphFidelity == "" {
			graphFidelity = "compact"
		}
	}

	updatedNodes := make(map[string]*graph.Node, len(g.Nodes))
	for id, node := range g.Nodes {
		fidelity := node.Fidelity()
		if fidelity == "" {
			fidelity = graphFidelity
		}

		if fidelity == "full" {
			updatedNodes[id] = node
			continue
		}

		// Build preamble text based on the fidelity mode.
		preamble := buildPreamble(fidelity)
		if preamble == "" {
			updatedNodes[id] = node
			continue
		}

		// Prepend preamble to the existing prompt.
		attrs := copyAttrs(node.Attrs)
		if prompt, ok := attrs["prompt"]; ok && prompt != "" {
			attrs["prompt"] = preamble + "\n\n" + prompt
		} else if label := node.Label(); label != "" {
			// No prompt but has label -- use label as the prompt body.
			attrs["prompt"] = preamble + "\n\n" + label
		}

		updatedNodes[id] = &graph.Node{
			ID:    node.ID,
			Attrs: attrs,
		}
	}

	return &graph.Graph{
		Name:  g.Name,
		Nodes: updatedNodes,
		Edges: g.Edges,
		Attrs: g.Attrs,
	}
}

// buildPreamble returns a context carryover note for the given fidelity mode.
func buildPreamble(fidelity string) string {
	switch strings.ToLower(fidelity) {
	case "truncate":
		return "[Context note: Earlier conversation history has been truncated. Only the most recent exchanges are included below.]"
	case "compact":
		return "[Context note: Earlier conversation history has been compacted. Key decisions and outcomes are preserved, but verbose details have been removed.]"
	case "summary:low":
		return "[Context note: Earlier conversation history has been summarised at a low detail level. Only major milestones and final outcomes are included.]"
	case "summary:medium":
		return "[Context note: Earlier conversation history has been summarised at a medium detail level. Key decisions, outcomes, and important intermediate steps are included.]"
	case "summary:high":
		return "[Context note: Earlier conversation history has been summarised at a high detail level. Most decisions and their rationale are preserved.]"
	default:
		return ""
	}
}
