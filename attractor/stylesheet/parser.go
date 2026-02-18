// Package stylesheet implements a CSS-like model stylesheet parser and
// applicator for the Attractor pipeline engine.
//
// Stylesheets allow pipeline authors to declaratively assign LLM
// configuration (model, provider, reasoning effort) to pipeline nodes based
// on their ID, class, or shape, using a familiar CSS-like syntax.
//
// Grammar:
//
//	Stylesheet  ::= Rule+
//	Rule        ::= Selector '{' Declaration (';' Declaration)* ';'? '}'
//	Selector    ::= '*' | '#' Identifier | '.' ClassName | ShapeName
//	Declaration ::= Property ':' PropertyValue | Property '=' PropertyValue
package stylesheet

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------

// SelectorType discriminates the kinds of selector.
type SelectorType int

const (
	// SelectorUniversal matches every node (*).
	SelectorUniversal SelectorType = iota
	// SelectorShape matches nodes by DOT shape name.
	SelectorShape
	// SelectorClass matches nodes with a given class (.class_name).
	SelectorClass
	// SelectorID matches a single node by ID (#node_id).
	SelectorID
)

// Selector represents a parsed selector from a stylesheet rule.
type Selector struct {
	Type  SelectorType
	Value string // shape name, class name, or node ID; empty for universal
}

// Specificity returns an integer representing selector priority.
// Higher values override lower values when multiple rules match.
//
//	0 = universal (*), 1 = shape, 2 = class (.foo), 3 = ID (#bar)
func (s Selector) Specificity() int { return int(s.Type) }

// Matches reports whether the selector matches a node given its ID, shape,
// and class list.
func (s Selector) Matches(nodeID, shape string, classes []string) bool {
	switch s.Type {
	case SelectorUniversal:
		return true
	case SelectorShape:
		return strings.EqualFold(s.Value, shape)
	case SelectorID:
		return s.Value == nodeID
	case SelectorClass:
		for _, c := range classes {
			if c == s.Value {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// String returns a human-readable representation of the selector.
func (s Selector) String() string {
	switch s.Type {
	case SelectorUniversal:
		return "*"
	case SelectorShape:
		return s.Value
	case SelectorClass:
		return "." + s.Value
	case SelectorID:
		return "#" + s.Value
	default:
		return "?"
	}
}

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------

// Rule represents a parsed stylesheet rule consisting of a selector and a
// set of property declarations.
type Rule struct {
	Selector     Selector
	Declarations map[string]string // property -> value
}

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------

// Parse parses a CSS-like stylesheet string into an ordered list of rules.
// It returns an error if the input contains syntax errors.
func Parse(source string) ([]Rule, error) {
	stripped := stripComments(source)
	return parseRules(strings.TrimSpace(stripped), nil)
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

// Apply resolves the effective properties for a node by evaluating all
// rules against the node's ID, shape, and classes. Rules are applied in
// source order; when two rules have equal specificity, the later rule wins.
// Higher-specificity rules always override lower-specificity ones.
func Apply(rules []Rule, nodeID, shape string, classes []string) map[string]string {
	type match struct {
		specificity  int
		index        int
		declarations map[string]string
	}

	var matches []match
	for i, r := range rules {
		if r.Selector.Matches(nodeID, shape, classes) {
			matches = append(matches, match{
				specificity:  r.Selector.Specificity(),
				index:        i,
				declarations: r.Declarations,
			})
		}
	}

	// Sort by specificity ascending, then by source order ascending.
	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].specificity != matches[j].specificity {
			return matches[i].specificity < matches[j].specificity
		}
		return matches[i].index < matches[j].index
	})

	result := make(map[string]string)
	for _, m := range matches {
		for prop, val := range m.declarations {
			result[prop] = val
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// Internal parsing
// ---------------------------------------------------------------------------

// stripComments removes C-style block and line comments from the input.
func stripComments(src string) string {
	var sb strings.Builder
	for i := 0; i < len(src); i++ {
		if i+1 < len(src) && src[i] == '/' && src[i+1] == '/' {
			i += 2
			for i < len(src) && src[i] != '\n' {
				i++
			}
		} else if i+1 < len(src) && src[i] == '/' && src[i+1] == '*' {
			i += 2
			for i+1 < len(src) && !(src[i] == '*' && src[i+1] == '/') {
				i++
			}
			if i+1 < len(src) {
				i++ // skip past closing /
			}
		} else {
			sb.WriteByte(src[i])
		}
	}
	return sb.String()
}

func parseRules(remaining string, acc []Rule) ([]Rule, error) {
	trimmed := strings.TrimSpace(remaining)
	if trimmed == "" {
		return acc, nil
	}

	rule, rest, err := parseSingleRule(trimmed)
	if err != nil {
		return nil, err
	}
	return parseRules(rest, append(acc, rule))
}

func parseSingleRule(input string) (Rule, string, error) {
	braceIdx := strings.Index(input, "{")
	if braceIdx < 0 {
		return Rule{}, "", fmt.Errorf("stylesheet: expected '{' near: %q", truncate(input, 40))
	}

	selectorStr := strings.TrimSpace(input[:braceIdx])
	afterBrace := input[braceIdx+1:]
	closeBraceIdx := strings.Index(afterBrace, "}")
	if closeBraceIdx < 0 {
		return Rule{}, "", fmt.Errorf("stylesheet: expected '}' to close rule for selector %q", selectorStr)
	}

	declsStr := strings.TrimSpace(afterBrace[:closeBraceIdx])
	rest := afterBrace[closeBraceIdx+1:]

	sel, err := parseSelector(selectorStr)
	if err != nil {
		return Rule{}, "", err
	}
	decls, err := parseDeclarations(declsStr)
	if err != nil {
		return Rule{}, "", err
	}

	return Rule{Selector: sel, Declarations: decls}, rest, nil
}

func parseSelector(s string) (Selector, error) {
	s = strings.TrimSpace(s)
	if s == "*" {
		return Selector{Type: SelectorUniversal}, nil
	}
	if strings.HasPrefix(s, "#") {
		name := strings.TrimSpace(s[1:])
		if name == "" {
			return Selector{}, fmt.Errorf("stylesheet: empty ID selector")
		}
		return Selector{Type: SelectorID, Value: name}, nil
	}
	if strings.HasPrefix(s, ".") {
		name := strings.TrimSpace(s[1:])
		if name == "" {
			return Selector{}, fmt.Errorf("stylesheet: empty class selector")
		}
		return Selector{Type: SelectorClass, Value: name}, nil
	}
	// Bare word is a shape selector.
	if isIdentifier(s) {
		return Selector{Type: SelectorShape, Value: s}, nil
	}
	return Selector{}, fmt.Errorf("stylesheet: unrecognized selector: %q (expected *, shape, .class, or #id)", s)
}

func parseDeclarations(s string) (map[string]string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return map[string]string{}, nil
	}

	// Split on semicolons or newlines, filter empties.
	var pairs []string
	for _, part := range strings.FieldsFunc(s, func(r rune) bool {
		return r == ';' || r == '\n'
	}) {
		p := strings.TrimSpace(part)
		if p != "" {
			pairs = append(pairs, p)
		}
	}

	result := make(map[string]string, len(pairs))
	for _, pair := range pairs {
		sepIdx := findSeparator(pair)
		if sepIdx < 0 {
			return nil, fmt.Errorf("stylesheet: expected ':' or '=' in declaration: %q", pair)
		}
		key := strings.TrimSpace(pair[:sepIdx])
		value := unquote(strings.TrimSpace(pair[sepIdx+1:]))
		if key == "" {
			return nil, fmt.Errorf("stylesheet: empty property name in declaration: %q", pair)
		}
		result[key] = value
	}
	return result, nil
}

func findSeparator(pair string) int {
	colonIdx := strings.IndexByte(pair, ':')
	eqIdx := strings.IndexByte(pair, '=')
	switch {
	case colonIdx >= 0 && eqIdx >= 0:
		if colonIdx < eqIdx {
			return colonIdx
		}
		return eqIdx
	case colonIdx >= 0:
		return colonIdx
	case eqIdx >= 0:
		return eqIdx
	default:
		return -1
	}
}

func isIdentifier(s string) bool {
	if len(s) == 0 {
		return false
	}
	for i, ch := range s {
		if i == 0 && !unicode.IsLetter(ch) && ch != '_' {
			return false
		}
		if !unicode.IsLetter(ch) && !unicode.IsDigit(ch) && ch != '_' {
			return false
		}
	}
	return true
}

func unquote(s string) string {
	if len(s) >= 2 {
		first := s[0]
		last := s[len(s)-1]
		if (first == '\'' || first == '"') && first == last {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
