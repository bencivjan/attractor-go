package condition

import (
	"testing"

	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func outcome(status state.StageStatus, label string) *state.Outcome {
	return &state.Outcome{
		Status:         status,
		PreferredLabel: label,
	}
}

func ctxWith(kvs ...any) *state.Context {
	ctx := state.NewContext()
	for i := 0; i+1 < len(kvs); i += 2 {
		ctx.Set(kvs[i].(string), kvs[i+1])
	}
	return ctx
}

// ---------------------------------------------------------------------------
// Evaluate tests
// ---------------------------------------------------------------------------

func TestEvaluate_EmptyCondition(t *testing.T) {
	// Empty condition always evaluates to true.
	tests := []struct {
		name      string
		condition string
	}{
		{"empty string", ""},
		{"whitespace only", "   "},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !Evaluate(tt.condition, nil, nil) {
				t.Error("empty condition should evaluate to true")
			}
		})
	}
}

func TestEvaluate_OutcomeSuccess(t *testing.T) {
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("outcome=success", o, nil) {
		t.Error("outcome=success should match StatusSuccess")
	}
}

func TestEvaluate_OutcomeFail(t *testing.T) {
	o := outcome(state.StatusFail, "")
	if !Evaluate("outcome=fail", o, nil) {
		t.Error("outcome=fail should match StatusFail")
	}
}

func TestEvaluate_OutcomeMismatch(t *testing.T) {
	o := outcome(state.StatusSuccess, "")
	if Evaluate("outcome=fail", o, nil) {
		t.Error("outcome=fail should not match StatusSuccess")
	}
}

func TestEvaluate_PreferredLabel(t *testing.T) {
	o := outcome(state.StatusSuccess, "approve")
	if !Evaluate("preferred_label=approve", o, nil) {
		t.Error("preferred_label=approve should match")
	}
	if Evaluate("preferred_label=reject", o, nil) {
		t.Error("preferred_label=reject should not match")
	}
}

func TestEvaluate_ContextDotKey(t *testing.T) {
	ctx := ctxWith("lang", "go")
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("context.lang=go", o, ctx) {
		t.Error("context.lang=go should match")
	}
	if Evaluate("context.lang=python", o, ctx) {
		t.Error("context.lang=python should not match")
	}
}

func TestEvaluate_BareKeyResolution(t *testing.T) {
	ctx := ctxWith("mode", "fast")
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("mode=fast", o, ctx) {
		t.Error("bare key 'mode=fast' should resolve from context")
	}
}

func TestEvaluate_NotEqualsOperator(t *testing.T) {
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("outcome!=fail", o, nil) {
		t.Error("outcome!=fail should be true when outcome is success")
	}
	if Evaluate("outcome!=success", o, nil) {
		t.Error("outcome!=success should be false when outcome is success")
	}
}

func TestEvaluate_AndConjunction(t *testing.T) {
	o := outcome(state.StatusSuccess, "approve")
	ctx := ctxWith("env", "prod")

	if !Evaluate("outcome=success && preferred_label=approve", o, ctx) {
		t.Error("both clauses true, should evaluate to true")
	}
	if Evaluate("outcome=success && preferred_label=reject", o, ctx) {
		t.Error("second clause false, should evaluate to false")
	}
	if Evaluate("outcome=fail && preferred_label=approve", o, ctx) {
		t.Error("first clause false, should evaluate to false")
	}
}

func TestEvaluate_ThreeClauseConjunction(t *testing.T) {
	o := outcome(state.StatusSuccess, "approve")
	ctx := ctxWith("env", "prod")

	if !Evaluate("outcome=success && preferred_label=approve && context.env=prod", o, ctx) {
		t.Error("all three clauses true, should evaluate to true")
	}
	if Evaluate("outcome=success && preferred_label=approve && context.env=staging", o, ctx) {
		t.Error("third clause false, should evaluate to false")
	}
}

func TestEvaluate_QuotedValues(t *testing.T) {
	o := outcome(state.StatusSuccess, "")
	ctx := ctxWith("msg", "hello world")

	if !Evaluate(`context.msg="hello world"`, o, ctx) {
		t.Error("quoted value should match context string with space")
	}
	if !Evaluate(`context.msg='hello world'`, o, ctx) {
		t.Error("single-quoted value should match context string with space")
	}
}

func TestEvaluate_MissingContextKey(t *testing.T) {
	o := outcome(state.StatusSuccess, "")
	ctx := state.NewContext()

	// Missing key resolves to empty string.
	if Evaluate("context.nonexistent=something", o, ctx) {
		t.Error("missing context key should resolve to empty string, not match")
	}
	if !Evaluate("context.nonexistent!=something", o, ctx) {
		t.Error("missing context key != something should be true")
	}
}

func TestEvaluate_NilOutcome(t *testing.T) {
	// Nil outcome should resolve "outcome" and "preferred_label" to "".
	if Evaluate("outcome=success", nil, nil) {
		t.Error("nil outcome should resolve to empty string, not match")
	}
	if !Evaluate("outcome!=success", nil, nil) {
		t.Error("nil outcome != success should be true")
	}
}

func TestEvaluate_NilContext(t *testing.T) {
	o := outcome(state.StatusSuccess, "")
	if Evaluate("context.foo=bar", o, nil) {
		t.Error("nil context should resolve to empty string")
	}
}

func TestEvaluate_IntContextValue(t *testing.T) {
	ctx := ctxWith("count", 42)
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("context.count=42", o, ctx) {
		t.Error("int context value 42 should match string '42'")
	}
}

func TestEvaluate_BoolContextValue(t *testing.T) {
	ctx := ctxWith("enabled", true)
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("context.enabled=true", o, ctx) {
		t.Error("bool context value true should match string 'true'")
	}
}

func TestEvaluate_MalformedClauseReturnsFalse(t *testing.T) {
	// A malformed clause (no operator) evaluates to false.
	o := outcome(state.StatusSuccess, "")
	if Evaluate("just_a_word", o, nil) {
		t.Error("malformed clause without operator should evaluate to false")
	}
}

// ---------------------------------------------------------------------------
// ParseCondition tests
// ---------------------------------------------------------------------------

func TestParseCondition_Valid(t *testing.T) {
	tests := []struct {
		name      string
		condition string
	}{
		{"empty", ""},
		{"simple equals", "outcome=success"},
		{"not equals", "outcome!=fail"},
		{"conjunction", "outcome=success && preferred_label=approve"},
		{"context key", "context.env=prod"},
		{"quoted value", `outcome="success"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ParseCondition(tt.condition); err != nil {
				t.Errorf("ParseCondition(%q) unexpected error: %v", tt.condition, err)
			}
		})
	}
}

func TestParseCondition_Invalid(t *testing.T) {
	tests := []struct {
		name      string
		condition string
	}{
		{"no operator", "justAWord"},
		{"empty clause in conjunction", "outcome=success && "},
		{"empty key", "=value"},
		{"empty value", "key="},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ParseCondition(tt.condition); err == nil {
				t.Errorf("ParseCondition(%q) expected error, got nil", tt.condition)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Edge cases in clause splitting
// ---------------------------------------------------------------------------

func TestEvaluate_AmpersandInsideQuotedValue(t *testing.T) {
	ctx := ctxWith("q", "a&&b")
	o := outcome(state.StatusSuccess, "")
	// The && inside quotes should not be treated as a separator.
	if !Evaluate(`context.q="a&&b"`, o, ctx) {
		t.Error("&& inside quotes should not split clause")
	}
}

// ---------------------------------------------------------------------------
// contextValueToString coverage for float64
// ---------------------------------------------------------------------------

func TestEvaluate_Float64ContextValue(t *testing.T) {
	ctx := ctxWith("rate", float64(3.14))
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("context.rate=3.14", o, ctx) {
		t.Error("float64 context value 3.14 should match string '3.14'")
	}
}

func TestEvaluate_Float64WholeNumber(t *testing.T) {
	ctx := ctxWith("count", float64(10))
	o := outcome(state.StatusSuccess, "")
	if !Evaluate("context.count=10", o, ctx) {
		t.Error("float64 whole number 10 should match string '10'")
	}
}
