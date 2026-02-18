// Package factory implements the FactoryRunner which orchestrates developer
// and evaluator pipelines as separate, context-isolated executions.
package factory

import (
	"github.com/strongdm/attractor-go/attractor/state"
)

// extractDeveloperSubmission picks the minimal set of context keys that
// cross the developer→evaluator boundary. Only the goal and the developer's
// final response are forwarded; all planning notes, retry history, and
// internal status keys are excluded.
func extractDeveloperSubmission(outcome *state.Outcome) map[string]any {
	result := make(map[string]any)
	if outcome == nil || outcome.ContextUpdates == nil {
		return result
	}
	snap := outcome.ContextUpdates

	if v, ok := snap["goal"]; ok {
		result["goal"] = v
	}
	if v, ok := snap["last_response"]; ok {
		result["last_response"] = v
	}
	return result
}

// extractEvaluatorFeedback maps the evaluator's last_response into the
// evaluator_feedback key that the developer pipeline checks on re-runs.
func extractEvaluatorFeedback(outcome *state.Outcome) map[string]any {
	result := make(map[string]any)
	if outcome == nil || outcome.ContextUpdates == nil {
		return result
	}
	if v, ok := outcome.ContextUpdates["last_response"]; ok {
		result["evaluator_feedback"] = v
	}
	return result
}

// isEvaluatorRejection returns true when the evaluator pipeline visited
// the return_feedback node, which only happens on the FAIL path. Both
// approval and rejection end at exit (SUCCESS), so we detect rejection by
// checking for the status.return_feedback context key.
func isEvaluatorRejection(outcome *state.Outcome) bool {
	if outcome == nil || outcome.ContextUpdates == nil {
		return false
	}
	_, ok := outcome.ContextUpdates["status.return_feedback"]
	return ok
}
