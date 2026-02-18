package parser

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Simple digraph with two nodes and one edge
// ---------------------------------------------------------------------------

func TestParse_SimpleTwoNodes(t *testing.T) {
	src := `digraph simple {
		a -> b
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if g.Name != "simple" {
		t.Errorf("Name = %q, want %q", g.Name, "simple")
	}
	if len(g.Nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(g.Nodes))
	}
	if _, ok := g.Nodes["a"]; !ok {
		t.Error("missing node 'a'")
	}
	if _, ok := g.Nodes["b"]; !ok {
		t.Error("missing node 'b'")
	}
	if len(g.Edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(g.Edges))
	}
	if g.Edges[0].From != "a" || g.Edges[0].To != "b" {
		t.Errorf("edge = %s -> %s, want a -> b", g.Edges[0].From, g.Edges[0].To)
	}
}

// ---------------------------------------------------------------------------
// Node attributes
// ---------------------------------------------------------------------------

func TestParse_NodeAttributes(t *testing.T) {
	src := `digraph test {
		mynode [label="My Node", shape=diamond, type=tool]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	n, ok := g.Nodes["mynode"]
	if !ok {
		t.Fatal("missing node 'mynode'")
	}
	if got := n.Label(); got != "My Node" {
		t.Errorf("Label() = %q, want %q", got, "My Node")
	}
	if got := n.Shape(); got != "diamond" {
		t.Errorf("Shape() = %q, want %q", got, "diamond")
	}
	if got := n.Type(); got != "tool" {
		t.Errorf("Type() = %q, want %q", got, "tool")
	}
}

// ---------------------------------------------------------------------------
// Edge attributes
// ---------------------------------------------------------------------------

func TestParse_EdgeAttributes(t *testing.T) {
	src := `digraph test {
		a -> b [label="success", condition="outcome=success", weight=10]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(g.Edges))
	}
	e := g.Edges[0]
	if got := e.Label(); got != "success" {
		t.Errorf("Label() = %q, want %q", got, "success")
	}
	if got := e.Condition(); got != "outcome=success" {
		t.Errorf("Condition() = %q, want %q", got, "outcome=success")
	}
	if got := e.Weight(); got != 10 {
		t.Errorf("Weight() = %d, want %d", got, 10)
	}
}

// ---------------------------------------------------------------------------
// Chained edges: a -> b -> c
// ---------------------------------------------------------------------------

func TestParse_ChainedEdges(t *testing.T) {
	src := `digraph test {
		a -> b -> c
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(g.Nodes))
	}
	if len(g.Edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(g.Edges))
	}
	if g.Edges[0].From != "a" || g.Edges[0].To != "b" {
		t.Errorf("edge[0] = %s -> %s, want a -> b", g.Edges[0].From, g.Edges[0].To)
	}
	if g.Edges[1].From != "b" || g.Edges[1].To != "c" {
		t.Errorf("edge[1] = %s -> %s, want b -> c", g.Edges[1].From, g.Edges[1].To)
	}
}

func TestParse_ChainedEdgesWithAttrs(t *testing.T) {
	src := `digraph test {
		a -> b -> c [label="chain"]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(g.Edges))
	}
	// Both edges in the chain get the same attributes.
	for i, e := range g.Edges {
		if got := e.Label(); got != "chain" {
			t.Errorf("edge[%d].Label() = %q, want %q", i, got, "chain")
		}
	}
}

// ---------------------------------------------------------------------------
// Graph-level attributes
// ---------------------------------------------------------------------------

func TestParse_GraphAttributes(t *testing.T) {
	src := `digraph pipeline {
		goal = "Build the feature"
		label = "Pipeline One"
		default_max_retry = "10"
		graph [retry_target=coder]
		a -> b
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if got := g.Goal(); got != "Build the feature" {
		t.Errorf("Goal() = %q, want %q", got, "Build the feature")
	}
	if got := g.Label(); got != "Pipeline One" {
		t.Errorf("Label() = %q, want %q", got, "Pipeline One")
	}
	if got := g.DefaultMaxRetry(); got != 10 {
		t.Errorf("DefaultMaxRetry() = %d, want %d", got, 10)
	}
	if got := g.RetryTarget(); got != "coder" {
		t.Errorf("RetryTarget() = %q, want %q", got, "coder")
	}
}

// ---------------------------------------------------------------------------
// Node defaults and edge defaults
// ---------------------------------------------------------------------------

func TestParse_NodeDefaults(t *testing.T) {
	src := `digraph test {
		node [shape=ellipse, type=codergen]
		a
		b [shape=diamond]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	a := g.Nodes["a"]
	if a == nil {
		t.Fatal("missing node 'a'")
	}
	if got := a.Shape(); got != "ellipse" {
		t.Errorf("a.Shape() = %q, want %q (from node default)", got, "ellipse")
	}
	if got := a.Type(); got != "codergen" {
		t.Errorf("a.Type() = %q, want %q (from node default)", got, "codergen")
	}

	b := g.Nodes["b"]
	if b == nil {
		t.Fatal("missing node 'b'")
	}
	if got := b.Shape(); got != "diamond" {
		t.Errorf("b.Shape() = %q, want %q (overridden)", got, "diamond")
	}
}

func TestParse_EdgeDefaults(t *testing.T) {
	src := `digraph test {
		edge [weight=5]
		a -> b
		c -> d [weight=20]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(g.Edges))
	}
	if got := g.Edges[0].Weight(); got != 5 {
		t.Errorf("edge[0].Weight() = %d, want 5 (from edge default)", got)
	}
	if got := g.Edges[1].Weight(); got != 20 {
		t.Errorf("edge[1].Weight() = %d, want 20 (overridden)", got)
	}
}

// ---------------------------------------------------------------------------
// Subgraphs
// ---------------------------------------------------------------------------

func TestParse_Subgraph(t *testing.T) {
	src := `digraph test {
		a -> b
		subgraph cluster_0 {
			c -> d
		}
		b -> c
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	// Subgraph contents are flattened into the parent graph.
	for _, id := range []string{"a", "b", "c", "d"} {
		if _, ok := g.Nodes[id]; !ok {
			t.Errorf("missing node %q", id)
		}
	}
	if len(g.Edges) != 3 {
		t.Errorf("expected 3 edges, got %d", len(g.Edges))
	}
}

func TestParse_SubgraphAnonymous(t *testing.T) {
	src := `digraph test {
		subgraph {
			x -> y
		}
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if _, ok := g.Nodes["x"]; !ok {
		t.Error("missing node 'x'")
	}
	if _, ok := g.Nodes["y"]; !ok {
		t.Error("missing node 'y'")
	}
}

// ---------------------------------------------------------------------------
// Comments
// ---------------------------------------------------------------------------

func TestParse_LineComments(t *testing.T) {
	src := `digraph test {
		// This is a comment
		a -> b  // trailing comment
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(g.Nodes))
	}
}

func TestParse_BlockComments(t *testing.T) {
	src := `digraph test {
		/* This is a
		   multi-line comment */
		a -> b
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(g.Nodes))
	}
}

func TestParse_CommentInsideQuotedString(t *testing.T) {
	src := `digraph test {
		a [label="has // comment syntax inside"]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	n := g.Nodes["a"]
	if n == nil {
		t.Fatal("missing node 'a'")
	}
	if got := n.Label(); got != "has // comment syntax inside" {
		t.Errorf("Label() = %q, want literal comment syntax preserved in string", got)
	}
}

// ---------------------------------------------------------------------------
// Quoted values with special characters
// ---------------------------------------------------------------------------

func TestParse_QuotedSpecialChars(t *testing.T) {
	src := `digraph test {
		a [label="line1\nline2", prompt="say \"hello\""]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	n := g.Nodes["a"]
	if n == nil {
		t.Fatal("missing node 'a'")
	}
	if got := n.Label(); !strings.Contains(got, "\n") {
		t.Errorf("Label() should contain a newline, got %q", got)
	}
	if got := n.Prompt(); !strings.Contains(got, `"`) {
		t.Errorf("Prompt() should contain a double quote, got %q", got)
	}
}

func TestParse_QuotedNodeID(t *testing.T) {
	src := `digraph test {
		"my node" -> "other node"
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if _, ok := g.Nodes["my node"]; !ok {
		t.Error("missing node 'my node'")
	}
	if _, ok := g.Nodes["other node"]; !ok {
		t.Error("missing node 'other node'")
	}
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

func TestParse_MissingDigraph(t *testing.T) {
	src := `graph test { a -> b }`
	_, err := Parse(src)
	if err == nil {
		t.Fatal("expected error for missing 'digraph' keyword")
	}
}

func TestParse_MissingOpenBrace(t *testing.T) {
	src := `digraph test a -> b }`
	_, err := Parse(src)
	if err == nil {
		t.Fatal("expected error for missing '{'")
	}
}

func TestParse_MissingCloseBrace(t *testing.T) {
	src := `digraph test { a -> b`
	_, err := Parse(src)
	if err == nil {
		t.Fatal("expected error for missing '}'")
	}
}

func TestParse_UnterminatedString(t *testing.T) {
	src := `digraph test {
		a [label="unterminated]
	}`
	_, err := Parse(src)
	if err == nil {
		t.Fatal("expected error for unterminated string")
	}
}

func TestParse_UnterminatedAttributeList(t *testing.T) {
	src := `digraph test {
		a [label="test"
	}`
	// The parser should detect the unterminated attribute list (the '}' is not ']').
	_, err := Parse(src)
	if err == nil {
		t.Fatal("expected error for unterminated attribute list")
	}
}

func TestParse_EmptyInput(t *testing.T) {
	_, err := Parse("")
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestParse_MissingEdgeTarget(t *testing.T) {
	src := `digraph test { a -> }`
	_, err := Parse(src)
	if err == nil {
		t.Fatal("expected error for missing edge target")
	}
}

// ---------------------------------------------------------------------------
// Semicolons as statement separators
// ---------------------------------------------------------------------------

func TestParse_SemicolonSeparators(t *testing.T) {
	src := `digraph test {
		a -> b; b -> c; c [label="end"];
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if len(g.Edges) != 2 {
		t.Errorf("expected 2 edges, got %d", len(g.Edges))
	}
	if len(g.Nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(g.Nodes))
	}
}

// ---------------------------------------------------------------------------
// Complex real-world pipeline graph
// ---------------------------------------------------------------------------

func TestParse_ComplexPipeline(t *testing.T) {
	src := `digraph codegen_pipeline {
		// Graph-level attributes
		goal = "Generate high-quality production code"
		label = "Code Generation Pipeline"
		default_max_retry = "25"
		retry_target = "coder"

		// Node defaults
		node [type=codergen]

		// Pipeline stages
		start [shape=Mdiamond, type=start, label="Begin"]
		planner [label="Plan Implementation", prompt="Create a plan"]
		coder [label="Generate Code", prompt="Write code", max_retries=10, goal_gate=true, retry_target=planner]
		reviewer [label="Review Code", type=conditional, shape=diamond]
		exit [shape=Msquare, type=exit, label="Done"]

		// Flow
		start -> planner
		planner -> coder
		coder -> reviewer
		reviewer -> exit [condition="outcome=success", label="pass"]
		reviewer -> coder [condition="outcome=fail", label="fail", loop_restart=true]
	}`

	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}

	// Graph attributes
	if got := g.Goal(); got != "Generate high-quality production code" {
		t.Errorf("Goal() = %q", got)
	}
	if got := g.DefaultMaxRetry(); got != 25 {
		t.Errorf("DefaultMaxRetry() = %d, want 25", got)
	}
	if got := g.RetryTarget(); got != "coder" {
		t.Errorf("RetryTarget() = %q, want %q", got, "coder")
	}

	// Node count
	if len(g.Nodes) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(g.Nodes))
	}

	// Start node
	startNode, err := g.FindStartNode()
	if err != nil {
		t.Fatalf("FindStartNode error: %v", err)
	}
	if startNode.ID != "start" {
		t.Errorf("start node ID = %q, want %q", startNode.ID, "start")
	}
	if got := startNode.Label(); got != "Begin" {
		t.Errorf("start.Label() = %q, want %q", got, "Begin")
	}

	// Exit node
	exitNode, err := g.FindExitNode()
	if err != nil {
		t.Fatalf("FindExitNode error: %v", err)
	}
	if exitNode.ID != "exit" {
		t.Errorf("exit node ID = %q, want %q", exitNode.ID, "exit")
	}

	// Coder node attributes
	coder := g.Nodes["coder"]
	if coder == nil {
		t.Fatal("missing node 'coder'")
	}
	if got := coder.MaxRetries(); got != 10 {
		t.Errorf("coder.MaxRetries() = %d, want 10", got)
	}
	if !coder.GoalGate() {
		t.Error("coder.GoalGate() should be true")
	}
	if got := coder.RetryTarget(); got != "planner" {
		t.Errorf("coder.RetryTarget() = %q, want %q", got, "planner")
	}

	// Reviewer node
	reviewer := g.Nodes["reviewer"]
	if reviewer == nil {
		t.Fatal("missing node 'reviewer'")
	}
	if got := reviewer.Type(); got != "conditional" {
		t.Errorf("reviewer.Type() = %q, want %q", got, "conditional")
	}

	// Edges
	if len(g.Edges) != 5 {
		t.Fatalf("expected 5 edges, got %d", len(g.Edges))
	}

	// Check conditional edges from reviewer
	outgoing := g.OutgoingEdges("reviewer")
	if len(outgoing) != 2 {
		t.Fatalf("expected 2 outgoing edges from 'reviewer', got %d", len(outgoing))
	}

	var successEdge, failEdge *edge
	for _, e := range outgoing {
		switch e.Condition() {
		case "outcome=success":
			successEdge = (*edge)(nil) // just to check it exists
			_ = successEdge
			if e.To != "exit" {
				t.Errorf("success edge target = %q, want %q", e.To, "exit")
			}
		case "outcome=fail":
			failEdge = (*edge)(nil)
			_ = failEdge
			if e.To != "coder" {
				t.Errorf("fail edge target = %q, want %q", e.To, "coder")
			}
			if !e.LoopRestart() {
				t.Error("fail edge should have loop_restart=true")
			}
		}
	}

	// Node defaults should have been applied (type=codergen) to nodes without overrides
	planner := g.Nodes["planner"]
	if planner == nil {
		t.Fatal("missing node 'planner'")
	}
	// planner does not override type, so it should inherit from node defaults
	if got := planner.Type(); got != "codergen" {
		t.Errorf("planner.Type() = %q, want %q (from node default)", got, "codergen")
	}
}

// ---------------------------------------------------------------------------
// Attribute separator: semicolons inside attribute lists
// ---------------------------------------------------------------------------

func TestParse_AttributeSemicolonSeparator(t *testing.T) {
	src := `digraph test {
		a [label="first"; shape=box]
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	n := g.Nodes["a"]
	if n == nil {
		t.Fatal("missing node 'a'")
	}
	if got := n.Label(); got != "first" {
		t.Errorf("Label() = %q, want %q", got, "first")
	}
	if got := n.Shape(); got != "box" {
		t.Errorf("Shape() = %q, want %q", got, "box")
	}
}

// ---------------------------------------------------------------------------
// Bare graph attribute (key=value outside graph [...])
// ---------------------------------------------------------------------------

func TestParse_BareGraphAttribute(t *testing.T) {
	src := `digraph test {
		label = "My Pipeline"
		goal = "Do things"
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if got := g.Label(); got != "My Pipeline" {
		t.Errorf("Label() = %q, want %q", got, "My Pipeline")
	}
	if got := g.Goal(); got != "Do things" {
		t.Errorf("Goal() = %q, want %q", got, "Do things")
	}
}

// ---------------------------------------------------------------------------
// Node with no attributes (just declaration)
// ---------------------------------------------------------------------------

func TestParse_BareNodeDeclaration(t *testing.T) {
	src := `digraph test {
		mynode
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if _, ok := g.Nodes["mynode"]; !ok {
		t.Error("missing node 'mynode'")
	}
}

// ---------------------------------------------------------------------------
// Hyphenated and qualified node names
// ---------------------------------------------------------------------------

func TestParse_HyphenatedNodeName(t *testing.T) {
	src := `digraph test {
		my-node -> other-node
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if _, ok := g.Nodes["my-node"]; !ok {
		t.Error("missing node 'my-node'")
	}
	if _, ok := g.Nodes["other-node"]; !ok {
		t.Error("missing node 'other-node'")
	}
}

func TestParse_QualifiedNodeName(t *testing.T) {
	src := `digraph test {
		foo.bar -> baz.qux
	}`
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if _, ok := g.Nodes["foo.bar"]; !ok {
		t.Error("missing node 'foo.bar'")
	}
}

// ---------------------------------------------------------------------------
// unused type alias to avoid compile error in complex test (edge is unexported)
// ---------------------------------------------------------------------------

type edge = struct {
	From  string
	To    string
	Attrs map[string]string
}
