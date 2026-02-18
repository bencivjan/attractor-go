// Package condition implements the condition expression language evaluator
// used by the Attractor pipeline engine to decide which edge to traverse
// after a node handler completes.
//
// Grammar:
//
//	ConditionExpr ::= Clause ( '&&' Clause )*
//	Clause        ::= Key Operator Literal
//	Operator      ::= '=' | '!='
//	Key           ::= 'outcome' | 'preferred_label' | 'context.' Path | bare identifier
//	Literal       ::= QuotedString | BareWord
//
// An empty condition string evaluates to true (unconditional edge).
// Missing context keys resolve to the empty string for comparison.
package condition

import (
	"fmt"
	"strings"

	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Evaluate evaluates a condition expression against an outcome and context.
// An empty condition always returns true. Each clause in the expression must
// be satisfied (logical AND) for the overall result to be true.
func Evaluate(condition string, outcome *state.Outcome, ctx *state.Context) bool {
	condition = strings.TrimSpace(condition)
	if condition == "" {
		return true
	}

	clauses := splitClauses(condition)
	for _, clause := range clauses {
		if !evaluateClause(clause, outcome, ctx) {
			return false
		}
	}
	return true
}

// ParseCondition validates condition syntax without evaluating. It returns
// an error describing the first syntax problem found, or nil if the
// condition is well-formed.
func ParseCondition(condition string) error {
	condition = strings.TrimSpace(condition)
	if condition == "" {
		return nil
	}

	clauses := splitClauses(condition)
	for i, clause := range clauses {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			return fmt.Errorf("condition: empty clause at position %d", i)
		}
		if _, _, _, err := parseClause(clause); err != nil {
			return fmt.Errorf("condition: clause %d: %w", i, err)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// splitClauses splits a condition on the && operator, handling quoted strings
// so that && inside quotes is not treated as a separator.
func splitClauses(condition string) []string {
	var clauses []string
	var current strings.Builder
	inQuote := false
	quoteChar := byte(0)
	runes := []byte(condition)

	for i := 0; i < len(runes); i++ {
		ch := runes[i]

		if inQuote {
			current.WriteByte(ch)
			if ch == quoteChar {
				inQuote = false
			}
			continue
		}

		if ch == '\'' || ch == '"' {
			inQuote = true
			quoteChar = ch
			current.WriteByte(ch)
			continue
		}

		// Look ahead for &&
		if ch == '&' && i+1 < len(runes) && runes[i+1] == '&' {
			clauses = append(clauses, current.String())
			current.Reset()
			i++ // skip second &
			continue
		}

		current.WriteByte(ch)
	}

	clauses = append(clauses, current.String())
	return clauses
}

// parseClause extracts the key, operator, and literal from a single clause
// string. It returns an error if the clause is malformed.
func parseClause(clause string) (key, op, literal string, err error) {
	clause = strings.TrimSpace(clause)

	// Find the operator. We search for != first (longer match) then =.
	opIdx := -1
	opLen := 0
	inQuote := false
	quoteChar := byte(0)

	for i := 0; i < len(clause); i++ {
		ch := clause[i]

		if inQuote {
			if ch == quoteChar {
				inQuote = false
			}
			continue
		}
		if ch == '\'' || ch == '"' {
			inQuote = true
			quoteChar = ch
			continue
		}

		if ch == '!' && i+1 < len(clause) && clause[i+1] == '=' {
			opIdx = i
			opLen = 2
			break
		}
		if ch == '=' {
			opIdx = i
			opLen = 1
			break
		}
	}

	if opIdx < 0 {
		return "", "", "", fmt.Errorf("no operator found in %q (expected = or !=)", clause)
	}

	key = strings.TrimSpace(clause[:opIdx])
	op = clause[opIdx : opIdx+opLen]
	literal = strings.TrimSpace(clause[opIdx+opLen:])

	if key == "" {
		return "", "", "", fmt.Errorf("empty key in clause %q", clause)
	}
	if literal == "" {
		return "", "", "", fmt.Errorf("empty value in clause %q", clause)
	}

	// Strip quotes from literal if present.
	literal = unquote(literal)

	return key, op, literal, nil
}

// evaluateClause evaluates a single key-operator-literal clause against an
// outcome and context.
func evaluateClause(clause string, outcome *state.Outcome, ctx *state.Context) bool {
	clause = strings.TrimSpace(clause)
	if clause == "" {
		return true
	}

	key, op, literal, err := parseClause(clause)
	if err != nil {
		// Malformed clauses evaluate to false to avoid silent misrouting.
		return false
	}

	resolved := resolveKey(key, outcome, ctx)

	switch op {
	case "=":
		return resolved == literal
	case "!=":
		return resolved != literal
	default:
		return false
	}
}

// resolveKey resolves a key reference against the outcome and context.
//
// Recognised keys:
//   - "outcome"          -> outcome.Status (as string)
//   - "preferred_label"  -> outcome.PreferredLabel
//   - "context.<path>"   -> ctx.Get(<path>) converted to string
//   - any bare word      -> treated as context key (ctx.Get(key))
//
// Missing or non-string values resolve to the empty string.
func resolveKey(key string, outcome *state.Outcome, ctx *state.Context) string {
	switch {
	case key == "outcome":
		if outcome == nil {
			return ""
		}
		return string(outcome.Status)

	case key == "preferred_label":
		if outcome == nil {
			return ""
		}
		return outcome.PreferredLabel

	case strings.HasPrefix(key, "context."):
		path := key[len("context."):]
		return contextValueToString(ctx, path)

	default:
		// Bare identifier -- look up in context.
		return contextValueToString(ctx, key)
	}
}

// contextValueToString retrieves a context value and converts it to a
// string for comparison. Nil and unsupported types return "".
func contextValueToString(ctx *state.Context, key string) string {
	if ctx == nil {
		return ""
	}
	v := ctx.Get(key)
	if v == nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	case bool:
		if t {
			return "true"
		}
		return "false"
	case int:
		return fmt.Sprintf("%d", t)
	case int64:
		return fmt.Sprintf("%d", t)
	case float64:
		// Use a compact representation that avoids trailing zeros.
		if t == float64(int64(t)) {
			return fmt.Sprintf("%d", int64(t))
		}
		return fmt.Sprintf("%g", t)
	case fmt.Stringer:
		return t.String()
	default:
		return fmt.Sprintf("%v", t)
	}
}

// unquote strips matching leading and trailing single or double quotes from
// a string. If the string is not quoted (or has mismatched quotes) it is
// returned unchanged.
func unquote(s string) string {
	if len(s) < 2 {
		return s
	}
	first := s[0]
	last := s[len(s)-1]
	if (first == '\'' || first == '"') && first == last {
		return s[1 : len(s)-1]
	}
	return s
}
