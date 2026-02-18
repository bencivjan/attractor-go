// Package parser implements a recursive-descent DOT parser for the Attractor
// pipeline graph subset.
//
// Strategy:
//  1. Strip comments (// line and /* block */)
//  2. Tokenize into a flat token list with line-number tracking
//  3. Parse the token stream with a recursive-descent parser
//
// The parser supports the following DOT subset:
//   - One digraph per file
//   - Directed edges only (->)
//   - Node statements with attribute blocks
//   - Edge statements including chained: A -> B -> C
//   - graph/node/edge default blocks
//   - Subgraph blocks (contents flattened into parent graph)
//   - Comments: // line and /* block */
//   - Quoted strings with escapes: \", \\, \n, \t
//   - Semicolons optional; commas or semicolons separate attributes
//   - Qualified IDs (e.g. foo.bar)
package parser

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/strongdm/attractor-go/attractor/graph"
)

// Parse parses DOT source into a Graph model.
func Parse(source string) (*graph.Graph, error) {
	stripped := stripComments(source)
	tokens, err := tokenize(stripped)
	if err != nil {
		return nil, err
	}
	p := newParser(tokens)
	return p.parseGraph()
}

// =========================================================================
// Phase 1: Comment stripping
// =========================================================================

// stripComments removes // line-comments and /* block-comments */ from src
// while preserving quoted strings intact. Newlines inside block comments are
// preserved so that line numbers remain accurate for error reporting.
func stripComments(src string) string {
	var sb strings.Builder
	sb.Grow(len(src))

	i := 0
	inString := false

	for i < len(src) {
		if inString {
			// Inside a quoted string: pass through, respecting escape sequences.
			if src[i] == '\\' && i+1 < len(src) {
				sb.WriteByte(src[i])
				sb.WriteByte(src[i+1])
				i += 2
			} else {
				if src[i] == '"' {
					inString = false
				}
				sb.WriteByte(src[i])
				i++
			}
		} else if src[i] == '"' {
			inString = true
			sb.WriteByte(src[i])
			i++
		} else if i+1 < len(src) && src[i] == '/' && src[i+1] == '/' {
			// Line comment -- skip to end of line, preserve the newline.
			i += 2
			for i < len(src) && src[i] != '\n' {
				i++
			}
		} else if i+1 < len(src) && src[i] == '/' && src[i+1] == '*' {
			// Block comment -- skip to closing */, preserve newlines for line tracking.
			i += 2
			for i+1 < len(src) && !(src[i] == '*' && src[i+1] == '/') {
				if src[i] == '\n' {
					sb.WriteByte('\n')
				}
				i++
			}
			if i+1 < len(src) {
				i += 2 // skip */
			}
		} else {
			sb.WriteByte(src[i])
			i++
		}
	}
	return sb.String()
}

// =========================================================================
// Phase 2: Tokenization
// =========================================================================

// tokenKind distinguishes the various lexemes the parser works with.
type tokenKind int

const (
	tokIdent        tokenKind = iota // bare identifier or numeric literal
	tokQuotedString                  // "..." with escapes processed
	tokArrow                         // ->
	tokLBrace                        // {
	tokRBrace                        // }
	tokLBracket                      // [
	tokRBracket                      // ]
	tokEquals                        // =
	tokSemicolon                     // ;
	tokComma                         // ,
)

// token is a single lexeme together with its source line number.
type token struct {
	kind  tokenKind
	value string // the textual content (meaningful for Ident/QuotedString)
	line  int    // 1-based source line where this token starts
}

func (t token) String() string {
	switch t.kind {
	case tokIdent:
		return fmt.Sprintf("identifier %q", t.value)
	case tokQuotedString:
		return fmt.Sprintf("string %q", t.value)
	case tokArrow:
		return "'->'"
	case tokLBrace:
		return "'{'"
	case tokRBrace:
		return "'}'"
	case tokLBracket:
		return "'['"
	case tokRBracket:
		return "']'"
	case tokEquals:
		return "'='"
	case tokSemicolon:
		return "';'"
	case tokComma:
		return "','"
	}
	return fmt.Sprintf("token(%d, %q)", t.kind, t.value)
}

// tokenize converts comment-stripped source into a flat token list.
func tokenize(src string) ([]token, error) {
	tokens := make([]token, 0, 256)
	i := 0
	line := 1

	for i < len(src) {
		// Skip whitespace, tracking newlines.
		if src[i] == '\n' {
			line++
			i++
			continue
		}
		if src[i] == '\r' || src[i] == '\t' || src[i] == ' ' {
			i++
			continue
		}

		ch := src[i]

		switch ch {
		case '{':
			tokens = append(tokens, token{kind: tokLBrace, value: "{", line: line})
			i++
		case '}':
			tokens = append(tokens, token{kind: tokRBrace, value: "}", line: line})
			i++
		case '[':
			tokens = append(tokens, token{kind: tokLBracket, value: "[", line: line})
			i++
		case ']':
			tokens = append(tokens, token{kind: tokRBracket, value: "]", line: line})
			i++
		case '=':
			tokens = append(tokens, token{kind: tokEquals, value: "=", line: line})
			i++
		case ';':
			tokens = append(tokens, token{kind: tokSemicolon, value: ";", line: line})
			i++
		case ',':
			tokens = append(tokens, token{kind: tokComma, value: ",", line: line})
			i++
		case '-':
			if i+1 < len(src) && src[i+1] == '>' {
				tokens = append(tokens, token{kind: tokArrow, value: "->", line: line})
				i += 2
			} else {
				// Bare hyphen starts an identifier (e.g. hyphenated node names).
				start := i
				i++
				for i < len(src) && isIdentContinue(rune(src[i])) {
					i++
				}
				tokens = append(tokens, token{kind: tokIdent, value: src[start:i], line: line})
			}
		case '"':
			startLine := line
			s, newI, newLine, err := readQuotedString(src, i, line)
			if err != nil {
				return nil, err
			}
			tokens = append(tokens, token{kind: tokQuotedString, value: s, line: startLine})
			i = newI
			line = newLine
		default:
			r := rune(ch)
			if unicode.IsLetter(r) || r == '_' || unicode.IsDigit(r) {
				start := i
				for i < len(src) && isIdentContinue(rune(src[i])) {
					i++
				}
				tokens = append(tokens, token{kind: tokIdent, value: src[start:i], line: line})
			} else {
				return nil, fmt.Errorf("line %d: unexpected character %q", line, string(ch))
			}
		}
	}

	return tokens, nil
}

// isIdentContinue returns true for characters that may appear in a bare identifier
// (after the first character). Supports letters, digits, underscore, dot, and hyphen
// so that qualified IDs like "foo.bar" and hyphenated names like "my-node" work.
func isIdentContinue(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '.' || r == '-'
}

// readQuotedString reads a "..." string starting at position i (which must point
// to the opening quote). It processes escape sequences and returns the unescaped
// string, the new position after the closing quote, the updated line number, and
// any error.
func readQuotedString(src string, i int, line int) (string, int, int, error) {
	startLine := line
	var sb strings.Builder
	i++ // skip opening "

	for i < len(src) {
		ch := src[i]
		if ch == '\n' {
			line++
		}
		if ch == '\\' && i+1 < len(src) {
			next := src[i+1]
			switch next {
			case 'n':
				sb.WriteByte('\n')
			case 't':
				sb.WriteByte('\t')
			case '\\':
				sb.WriteByte('\\')
			case '"':
				sb.WriteByte('"')
			default:
				sb.WriteByte('\\')
				sb.WriteByte(next)
			}
			if next == '\n' {
				line++
			}
			i += 2
			continue
		}
		if ch == '"' {
			i++ // skip closing "
			return sb.String(), i, line, nil
		}
		sb.WriteByte(ch)
		i++
	}
	return "", i, line, fmt.Errorf("line %d: unterminated string literal", startLine)
}

// =========================================================================
// Phase 3: Recursive-descent parser
// =========================================================================

// parser holds the mutable parse state.
type parser struct {
	tokens []token
	pos    int
}

func newParser(tokens []token) *parser {
	return &parser{tokens: tokens}
}

// peek returns the current token without consuming it.
func (p *parser) peek() (token, bool) {
	if p.pos < len(p.tokens) {
		return p.tokens[p.pos], true
	}
	return token{}, false
}

// advance consumes and returns the current token.
func (p *parser) advance() token {
	t := p.tokens[p.pos]
	p.pos++
	return t
}

// currentLine returns a best-effort line number for error messages.
func (p *parser) currentLine() int {
	if p.pos < len(p.tokens) {
		return p.tokens[p.pos].line
	}
	if len(p.tokens) > 0 {
		return p.tokens[len(p.tokens)-1].line
	}
	return 1
}

// expect consumes the next token if it satisfies pred. Returns an error otherwise.
func (p *parser) expect(pred func(token) bool, desc string) (token, error) {
	t, ok := p.peek()
	if !ok {
		return token{}, fmt.Errorf("line %d: expected %s but reached end of input", p.currentLine(), desc)
	}
	if !pred(t) {
		return token{}, fmt.Errorf("line %d: expected %s but got %s", t.line, desc, t)
	}
	return p.advance(), nil
}

// expectIdent consumes the next token if it is an identifier or quoted string
// and returns the string value.
func (p *parser) expectIdent(desc string) (string, error) {
	t, err := p.expect(func(t token) bool {
		return t.kind == tokIdent || t.kind == tokQuotedString
	}, desc)
	if err != nil {
		return "", err
	}
	return t.value, nil
}

// skipSemicolons consumes any number of semicolons at the current position.
func (p *parser) skipSemicolons() {
	for {
		t, ok := p.peek()
		if !ok || t.kind != tokSemicolon {
			return
		}
		p.advance()
	}
}

// parseAttributeValue consumes one attribute value (quoted string or bare identifier).
func (p *parser) parseAttributeValue() (string, error) {
	t, ok := p.peek()
	if !ok {
		return "", fmt.Errorf("line %d: expected attribute value but reached end of input", p.currentLine())
	}
	switch t.kind {
	case tokQuotedString, tokIdent:
		p.advance()
		return t.value, nil
	default:
		return "", fmt.Errorf("line %d: expected attribute value but got %s", t.line, t)
	}
}

// parseAttrList parses an optional [...] attribute list and returns it as a map.
// If the current token is not '[', returns an empty map and no error.
func (p *parser) parseAttrList() (map[string]string, error) {
	t, ok := p.peek()
	if !ok || t.kind != tokLBracket {
		return map[string]string{}, nil
	}
	p.advance() // consume [

	attrs := make(map[string]string)

	for {
		t, ok := p.peek()
		if !ok {
			return nil, fmt.Errorf("line %d: unterminated attribute list, expected ']'", p.currentLine())
		}
		if t.kind == tokRBracket {
			p.advance()
			return attrs, nil
		}

		// Parse key = value.
		key, err := p.expectIdent("attribute key")
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(func(t token) bool { return t.kind == tokEquals }, "'='"); err != nil {
			return nil, err
		}
		val, err := p.parseAttributeValue()
		if err != nil {
			return nil, err
		}
		attrs[key] = val

		// Optional comma or semicolon separator between attributes.
		if next, ok := p.peek(); ok && (next.kind == tokComma || next.kind == tokSemicolon) {
			p.advance()
		}
	}
}

// ---------------------------------------------------------------------------
// parseState bundles the accumulated state that flows through statement parsing.
// Using a struct avoids passing five maps/slices through every recursive call.
// ---------------------------------------------------------------------------
type parseState struct {
	graphAttrs   map[string]string
	nodeDefaults map[string]string
	edgeDefaults map[string]string
	nodes        map[string]*graph.Node
	edges        []*graph.Edge
}

func newParseState() *parseState {
	return &parseState{
		graphAttrs:   make(map[string]string),
		nodeDefaults: make(map[string]string),
		edgeDefaults: make(map[string]string),
		nodes:        make(map[string]*graph.Node),
		edges:        nil,
	}
}

// parseGraph parses the top-level digraph.
//
//	digraph Name { statement* }
func (p *parser) parseGraph() (*graph.Graph, error) {
	_, err := p.expect(func(t token) bool {
		return t.kind == tokIdent && strings.EqualFold(t.value, "digraph")
	}, "'digraph'")
	if err != nil {
		return nil, err
	}

	name, err := p.expectIdent("graph name")
	if err != nil {
		return nil, err
	}

	if _, err := p.expect(func(t token) bool { return t.kind == tokLBrace }, "'{'"); err != nil {
		return nil, err
	}

	st := newParseState()
	if err := p.parseBody(st); err != nil {
		return nil, err
	}

	if _, err := p.expect(func(t token) bool { return t.kind == tokRBrace }, "'}'"); err != nil {
		return nil, err
	}

	return &graph.Graph{
		Name:  name,
		Nodes: st.nodes,
		Edges: st.edges,
		Attrs: st.graphAttrs,
	}, nil
}

// parseBody parses the statement list inside { ... }.
func (p *parser) parseBody(st *parseState) error {
	for {
		p.skipSemicolons()
		t, ok := p.peek()
		if !ok || t.kind == tokRBrace {
			return nil
		}
		if err := p.parseStatement(st); err != nil {
			return err
		}
		p.skipSemicolons()
	}
}

// parseStatement parses a single statement within a graph body.
func (p *parser) parseStatement(st *parseState) error {
	t, ok := p.peek()
	if !ok {
		return fmt.Errorf("line %d: unexpected end of input", p.currentLine())
	}

	// Check for keyword-led statements (graph/node/edge/subgraph defaults).
	if t.kind == tokIdent {
		lower := strings.ToLower(t.value)
		switch lower {
		case "graph":
			p.advance()
			attrs, err := p.parseAttrList()
			if err != nil {
				return err
			}
			mergeInto(st.graphAttrs, attrs)
			return nil

		case "node":
			p.advance()
			attrs, err := p.parseAttrList()
			if err != nil {
				return err
			}
			mergeInto(st.nodeDefaults, attrs)
			return nil

		case "edge":
			p.advance()
			attrs, err := p.parseAttrList()
			if err != nil {
				return err
			}
			mergeInto(st.edgeDefaults, attrs)
			return nil

		case "subgraph":
			return p.parseSubgraph(st)
		}
	}

	// Otherwise it must be a node statement, edge chain, or bare graph attribute.
	if t.kind == tokIdent || t.kind == tokQuotedString {
		return p.parseNodeOrEdge(st)
	}

	return fmt.Errorf("line %d: unexpected token %s", t.line, t)
}

// parseSubgraph parses:
//
//	subgraph [Name] { statement* }
//
// The contents are flattened into the parent state.
func (p *parser) parseSubgraph(st *parseState) error {
	p.advance() // consume "subgraph"

	// Optional subgraph name.
	if t, ok := p.peek(); ok && (t.kind == tokIdent || t.kind == tokQuotedString) {
		p.advance()
	}

	if _, err := p.expect(func(t token) bool { return t.kind == tokLBrace }, "'{'"); err != nil {
		return err
	}

	// Parse the subgraph body using the parent's state directly so that nodes,
	// edges, and defaults are flattened into the enclosing graph.
	if err := p.parseBody(st); err != nil {
		return err
	}

	if _, err := p.expect(func(t token) bool { return t.kind == tokRBrace }, "'}'"); err != nil {
		return err
	}

	return nil
}

// parseNodeOrEdge parses a statement that starts with an identifier:
//   - edge chain:   id -> id [-> id ...] [attrs]
//   - graph attr:   key = value
//   - node stmt:    id [attrs]
func (p *parser) parseNodeOrEdge(st *parseState) error {
	firstID, err := p.expectIdent("node id")
	if err != nil {
		return err
	}

	t, ok := p.peek()
	if !ok {
		// Bare identifier at end of input -- treat as a node declaration.
		p.ensureNode(st, firstID)
		return nil
	}

	switch t.kind {
	case tokArrow:
		return p.parseEdgeChain(firstID, st)

	case tokEquals:
		// Standalone graph attribute: key = value (e.g. goal = "Build feature").
		p.advance() // consume =
		val, err := p.parseAttributeValue()
		if err != nil {
			return err
		}
		st.graphAttrs[firstID] = val
		return nil

	default:
		// Node statement: id [attrs]
		attrs, err := p.parseAttrList()
		if err != nil {
			return err
		}
		merged := mergeMaps(st.nodeDefaults, attrs)
		if existing, ok := st.nodes[firstID]; ok {
			mergeInto(existing.Attrs, merged)
		} else {
			st.nodes[firstID] = &graph.Node{ID: firstID, Attrs: merged}
		}
		return nil
	}
}

// parseEdgeChain parses: (first id already consumed) -> id [-> id ...] [attrs]
// and expands chained edges into individual Edge values.
func (p *parser) parseEdgeChain(firstID string, st *parseState) error {
	chain := []string{firstID}

	for {
		t, ok := p.peek()
		if !ok || t.kind != tokArrow {
			break
		}
		p.advance() // consume ->
		id, err := p.expectIdent("edge target")
		if err != nil {
			return err
		}
		chain = append(chain, id)
	}

	attrs, err := p.parseAttrList()
	if err != nil {
		return err
	}
	mergedEdgeAttrs := mergeMaps(st.edgeDefaults, attrs)

	// Ensure every node in the chain exists with at least the current defaults.
	for _, id := range chain {
		p.ensureNode(st, id)
	}

	// Expand the chain into individual edges.
	for i := 0; i < len(chain)-1; i++ {
		// Each edge gets its own copy of the merged attributes so that
		// mutations to one edge's map do not affect the others.
		edgeAttrs := copyMap(mergedEdgeAttrs)
		st.edges = append(st.edges, &graph.Edge{
			From:  chain[i],
			To:    chain[i+1],
			Attrs: edgeAttrs,
		})
	}

	return nil
}

// ensureNode creates a node with default attributes if it does not already exist.
func (p *parser) ensureNode(st *parseState, id string) {
	if _, ok := st.nodes[id]; !ok {
		st.nodes[id] = &graph.Node{ID: id, Attrs: copyMap(st.nodeDefaults)}
	}
}

// ---------------------------------------------------------------------------
// Map helpers
// ---------------------------------------------------------------------------

// mergeMaps returns a new map containing all entries from base with overrides
// applied on top. Neither input map is modified.
func mergeMaps(base, overrides map[string]string) map[string]string {
	out := make(map[string]string, len(base)+len(overrides))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overrides {
		out[k] = v
	}
	return out
}

// mergeInto writes all entries from src into dst, overwriting existing keys.
func mergeInto(dst, src map[string]string) {
	for k, v := range src {
		dst[k] = v
	}
}

// copyMap returns a shallow copy of m.
func copyMap(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
