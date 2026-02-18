package factory

import (
	"context"
	"strings"
	"testing"

	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// Test DOT pipelines
// ---------------------------------------------------------------------------

// minimalDeveloperDOT is a trivial pipeline that sets last_response.
const minimalDeveloperDOT = `digraph developer {
	goal = "test goal"
	start [shape=Mdiamond, label="Begin"]
	work  [shape=box, label="Work"]
	exit  [shape=Msquare, label="Done"]
	start -> work -> exit
}`

// minimalEvaluatorApproveDOT always takes the approval path (no return_feedback).
const minimalEvaluatorApproveDOT = `digraph evaluator {
	goal = "evaluate"
	start [shape=Mdiamond, label="Start"]
	exit  [shape=Msquare, label="Done"]
	start -> exit
}`

// minimalEvaluatorRejectDOT visits return_feedback on every run,
// which sets status.return_feedback in context — signaling rejection.
const minimalEvaluatorRejectDOT = `digraph evaluator {
	goal = "evaluate"
	start           [shape=Mdiamond, label="Start"]
	return_feedback [shape=box, label="Return Feedback"]
	exit            [shape=Msquare, label="Done"]
	start -> return_feedback -> exit
}`

// minimalDeveloperFailDOT has fail_node as a goal_gate. When the handler
// fails, the pipeline reaches exit but the goal gate check fails,
// producing an overall pipeline failure.
const minimalDeveloperFailDOT = `digraph developer {
	goal = "test goal"
	default_max_retry = "1"
	start     [shape=Mdiamond, label="Begin"]
	fail_node [shape=box, label="Fail", goal_gate="true"]
	exit      [shape=Msquare, label="Done"]
	start -> fail_node -> exit
}`

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// responseHandler returns a handler that sets last_response in context on success.
func responseHandler() handler.Handler {
	return handler.HandlerFunc(func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
		return &state.Outcome{
			Status:         state.StatusSuccess,
			ContextUpdates: map[string]any{"last_response": "developer output"},
		}, nil
	})
}

// failHandler returns a handler that always fails.
func failHandler() handler.Handler {
	return handler.HandlerFunc(func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
		return &state.Outcome{
			Status:        state.StatusFail,
			FailureReason: "forced failure",
		}, nil
	})
}

func newTestFactory(devDOT, evalDOT string, registry *handler.Registry) *FactoryRunner {
	return &FactoryRunner{
		Registry:      registry,
		MaxSteps:      100,
		MaxRejections: 3,
		DeveloperDOT:  devDOT,
		EvaluatorDOT:  evalDOT,
	}
}

// ---------------------------------------------------------------------------
// Test: Happy path — developer succeeds, evaluator approves first try
// ---------------------------------------------------------------------------

func TestFactoryRunner_HappyPath(t *testing.T) {
	registry := handler.DefaultRegistry(responseHandler())
	factory := newTestFactory(minimalDeveloperDOT, minimalEvaluatorApproveDOT, registry)

	outcome, err := factory.RunWithGoal(context.Background(), "build a thing", t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q (reason: %s)", outcome.Status, outcome.FailureReason)
	}
	if outcome.Iterations != 1 {
		t.Errorf("expected 1 iteration, got %d", outcome.Iterations)
	}
	if outcome.DeveloperOutcome == nil {
		t.Error("expected non-nil developer outcome")
	}
	if outcome.EvaluatorOutcome == nil {
		t.Error("expected non-nil evaluator outcome")
	}
}

// ---------------------------------------------------------------------------
// Test: Rejection then approval — evaluator rejects once, approves on retry
// ---------------------------------------------------------------------------

func TestFactoryRunner_RejectionThenApproval(t *testing.T) {
	evalCallCount := 0

	// Evaluator DOT with conditional branching: judge decides reject or approve.
	// default_max_retry=1 limits retries so the judge handler's first attempt
	// determines the path within each evaluator run.
	conditionalEvalDOT := `digraph evaluator {
		goal = "evaluate"
		default_max_retry = "1"
		start     [shape=Mdiamond, label="Start"]
		judge     [shape=box, label="Judge"]
		return_feedback [shape=box, label="Return Feedback"]
		exit      [shape=Msquare, label="Done"]
		start -> judge
		judge -> return_feedback [condition="outcome=fail", label="Reject", weight="10"]
		judge -> exit            [condition="outcome=success", label="Approve", weight="10"]
		return_feedback -> exit
	}`

	// Judge handler fails on first call (first evaluator run), succeeds on
	// second call (second evaluator run).
	registry := handler.NewRegistry(handler.HandlerFunc(
		func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
			if node.ID == "judge" {
				evalCallCount++
				if evalCallCount == 1 {
					return &state.Outcome{
						Status:         state.StatusFail,
						ContextUpdates: map[string]any{"last_response": "fix your bugs"},
					}, nil
				}
				return &state.Outcome{
					Status:         state.StatusSuccess,
					ContextUpdates: map[string]any{"last_response": "looks good"},
				}, nil
			}
			return &state.Outcome{
				Status:         state.StatusSuccess,
				ContextUpdates: map[string]any{"last_response": "developer output"},
			}, nil
		},
	))
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})

	factory := newTestFactory(minimalDeveloperDOT, conditionalEvalDOT, registry)

	outcome, err := factory.RunWithGoal(context.Background(), "build a thing", t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Errorf("expected success, got %q (reason: %s)", outcome.Status, outcome.FailureReason)
	}
	if outcome.Iterations != 2 {
		t.Errorf("expected 2 iterations, got %d", outcome.Iterations)
	}
}

// ---------------------------------------------------------------------------
// Test: Max rejections exceeded — always rejects, stops at limit
// ---------------------------------------------------------------------------

func TestFactoryRunner_MaxRejectionsExceeded(t *testing.T) {
	registry := handler.DefaultRegistry(handler.HandlerFunc(
		func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
			return &state.Outcome{
				Status:         state.StatusSuccess,
				ContextUpdates: map[string]any{"last_response": "developer output"},
			}, nil
		},
	))

	factory := &FactoryRunner{
		Registry:      registry,
		MaxSteps:      100,
		MaxRejections: 2,
		DeveloperDOT:  minimalDeveloperDOT,
		EvaluatorDOT:  minimalEvaluatorRejectDOT,
	}

	outcome, err := factory.RunWithGoal(context.Background(), "build a thing", t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusFail {
		t.Errorf("expected fail, got %q", outcome.Status)
	}
	if outcome.Iterations != 3 {
		t.Errorf("expected 3 iterations (MaxRejections=2 means 3 attempts), got %d", outcome.Iterations)
	}
	if !strings.Contains(outcome.FailureReason, "rejected") {
		t.Errorf("expected rejection failure reason, got %q", outcome.FailureReason)
	}
}

// ---------------------------------------------------------------------------
// Test: Developer fails — returns immediately without running evaluator
// ---------------------------------------------------------------------------

func TestFactoryRunner_DeveloperFails(t *testing.T) {
	registry := handler.NewRegistry(failHandler())
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})

	factory := newTestFactory(minimalDeveloperFailDOT, minimalEvaluatorApproveDOT, registry)

	outcome, err := factory.RunWithGoal(context.Background(), "build a thing", t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusFail {
		t.Errorf("expected fail, got %q", outcome.Status)
	}
	if outcome.EvaluatorOutcome != nil {
		t.Error("expected nil evaluator outcome when developer fails")
	}
	if outcome.Iterations != 1 {
		t.Errorf("expected 1 iteration, got %d", outcome.Iterations)
	}
}

// ---------------------------------------------------------------------------
// Test: Context isolation — evaluator does NOT see developer planning keys
// ---------------------------------------------------------------------------

func TestFactoryRunner_ContextIsolation(t *testing.T) {
	var evaluatorCtxSnapshot map[string]any

	registry := handler.NewRegistry(handler.HandlerFunc(
		func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
			if g.Name == "developer" {
				return &state.Outcome{
					Status: state.StatusSuccess,
					ContextUpdates: map[string]any{
						"last_response":           "the code",
						"internal_planning_notes": "secret developer notes",
						"retry_history":           "attempt 1 failed, attempt 2 ok",
					},
				}, nil
			}
			// Evaluator: capture what's visible in context.
			evaluatorCtxSnapshot = pctx.Snapshot()
			return &state.Outcome{Status: state.StatusSuccess}, nil
		},
	))
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})

	devDOT := `digraph developer {
		goal = "test isolation"
		start [shape=Mdiamond]
		work  [shape=box]
		exit  [shape=Msquare]
		start -> work -> exit
	}`
	evalDOT := `digraph evaluator {
		goal = "evaluate"
		start  [shape=Mdiamond]
		check  [shape=box]
		exit   [shape=Msquare]
		start -> check -> exit
	}`

	factory := newTestFactory(devDOT, evalDOT, registry)

	outcome, err := factory.RunWithGoal(context.Background(), "test", t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Fatalf("expected success, got %q", outcome.Status)
	}

	if evaluatorCtxSnapshot == nil {
		t.Fatal("evaluator context snapshot was not captured")
	}
	for _, forbidden := range []string{"internal_planning_notes", "retry_history"} {
		if _, ok := evaluatorCtxSnapshot[forbidden]; ok {
			t.Errorf("evaluator context should not contain %q", forbidden)
		}
	}

	if evaluatorCtxSnapshot["goal"] == nil {
		t.Error("evaluator context should contain 'goal'")
	}
}

// ---------------------------------------------------------------------------
// Test: Run (without explicit goal) uses graph-level goal
// ---------------------------------------------------------------------------

func TestFactoryRunner_RunUsesGraphGoal(t *testing.T) {
	var capturedGoal any

	registry := handler.NewRegistry(handler.HandlerFunc(
		func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, lr string) (*state.Outcome, error) {
			if capturedGoal == nil {
				capturedGoal = pctx.Get("goal")
			}
			return &state.Outcome{Status: state.StatusSuccess}, nil
		},
	))
	registry.Register("start", &handler.StartHandler{})
	registry.Register("exit", &handler.ExitHandler{})

	factory := newTestFactory(minimalDeveloperDOT, minimalEvaluatorApproveDOT, registry)

	outcome, err := factory.Run(context.Background(), t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != state.StatusSuccess {
		t.Fatalf("expected success, got %q", outcome.Status)
	}

	if capturedGoal != "test goal" {
		t.Errorf("expected goal='test goal' from graph attr, got %v", capturedGoal)
	}
}
