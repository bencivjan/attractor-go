package factory

import (
	"testing"

	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// extractDeveloperSubmission
// ---------------------------------------------------------------------------

func TestExtractDeveloperSubmission(t *testing.T) {
	outcome := &state.Outcome{
		Status: state.StatusSuccess,
		ContextUpdates: map[string]any{
			"goal":                    "build a widget",
			"last_response":           "here is the code",
			"status.high_level_plan":  "success",
			"status.sprint_breakdown": "success",
			"status.implement":        "success",
			"pipeline.name":           "plan_build_verify",
			"preferred_label":         "QA Passed",
		},
	}

	sub := extractDeveloperSubmission(outcome)

	if sub["goal"] != "build a widget" {
		t.Errorf("expected goal='build a widget', got %v", sub["goal"])
	}
	if sub["last_response"] != "here is the code" {
		t.Errorf("expected last_response='here is the code', got %v", sub["last_response"])
	}
	// Must NOT leak internal keys.
	for _, key := range []string{"status.high_level_plan", "status.sprint_breakdown", "status.implement", "pipeline.name", "preferred_label"} {
		if _, ok := sub[key]; ok {
			t.Errorf("submission should not contain key %q", key)
		}
	}
	if len(sub) != 2 {
		t.Errorf("expected exactly 2 keys, got %d: %v", len(sub), sub)
	}
}

func TestExtractDeveloperSubmission_NilOutcome(t *testing.T) {
	sub := extractDeveloperSubmission(nil)
	if len(sub) != 0 {
		t.Errorf("expected empty map for nil outcome, got %v", sub)
	}
}

func TestExtractDeveloperSubmission_MissingKeys(t *testing.T) {
	outcome := &state.Outcome{
		Status:         state.StatusSuccess,
		ContextUpdates: map[string]any{"unrelated": "value"},
	}
	sub := extractDeveloperSubmission(outcome)
	if len(sub) != 0 {
		t.Errorf("expected empty map when goal/last_response absent, got %v", sub)
	}
}

// ---------------------------------------------------------------------------
// extractEvaluatorFeedback
// ---------------------------------------------------------------------------

func TestExtractEvaluatorFeedback(t *testing.T) {
	outcome := &state.Outcome{
		Status: state.StatusSuccess,
		ContextUpdates: map[string]any{
			"last_response":           "you need to fix X and Y",
			"status.return_feedback":  "success",
			"status.orchestrator":     "success",
		},
	}

	fb := extractEvaluatorFeedback(outcome)

	if fb["evaluator_feedback"] != "you need to fix X and Y" {
		t.Errorf("expected evaluator_feedback='you need to fix X and Y', got %v", fb["evaluator_feedback"])
	}
	if len(fb) != 1 {
		t.Errorf("expected exactly 1 key, got %d: %v", len(fb), fb)
	}
}

func TestExtractEvaluatorFeedback_NilOutcome(t *testing.T) {
	fb := extractEvaluatorFeedback(nil)
	if len(fb) != 0 {
		t.Errorf("expected empty map for nil outcome, got %v", fb)
	}
}

func TestExtractEvaluatorFeedback_NoLastResponse(t *testing.T) {
	outcome := &state.Outcome{
		Status:         state.StatusSuccess,
		ContextUpdates: map[string]any{"other": "data"},
	}
	fb := extractEvaluatorFeedback(outcome)
	if len(fb) != 0 {
		t.Errorf("expected empty map when last_response absent, got %v", fb)
	}
}

// ---------------------------------------------------------------------------
// isEvaluatorRejection
// ---------------------------------------------------------------------------

func TestIsEvaluatorRejection_Present(t *testing.T) {
	outcome := &state.Outcome{
		Status: state.StatusSuccess,
		ContextUpdates: map[string]any{
			"status.return_feedback": "success",
		},
	}
	if !isEvaluatorRejection(outcome) {
		t.Error("expected rejection when status.return_feedback is present")
	}
}

func TestIsEvaluatorRejection_Absent(t *testing.T) {
	outcome := &state.Outcome{
		Status: state.StatusSuccess,
		ContextUpdates: map[string]any{
			"status.visionary": "success",
		},
	}
	if isEvaluatorRejection(outcome) {
		t.Error("expected no rejection when status.return_feedback is absent")
	}
}

func TestIsEvaluatorRejection_NilOutcome(t *testing.T) {
	if isEvaluatorRejection(nil) {
		t.Error("expected no rejection for nil outcome")
	}
}
