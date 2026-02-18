package factory

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/strongdm/attractor-go/attractor/engine"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/pipelines"
	"github.com/strongdm/attractor-go/attractor/state"
	"github.com/strongdm/attractor-go/attractor/transform"
)

// FactoryRunner orchestrates developer and evaluator pipelines as completely
// separate executions. Only explicitly defined context keys cross the
// boundary between the two, ensuring the evaluator has no access to the
// developer's internal planning notes, retry history, or status keys.
type FactoryRunner struct {
	// Registry maps node types to handlers.
	Registry *handler.Registry

	// Transforms is the ordered list of graph transforms applied before
	// validation and execution. If nil, DefaultTransforms() is used.
	Transforms []transform.Transform

	// OnEvent is an optional callback for lifecycle events.
	OnEvent func(engine.Event)

	// MaxSteps is the per-pipeline step limit. Defaults to 1000.
	MaxSteps int

	// MaxRejections is the maximum number of evaluator rejections before
	// the factory gives up. Defaults to 3.
	MaxRejections int

	// DeveloperDOT overrides the built-in developer pipeline DOT source.
	// When empty, the embedded developer.dot is used.
	DeveloperDOT string

	// EvaluatorDOT overrides the built-in evaluator pipeline DOT source.
	// When empty, the embedded evaluator.dot is used.
	EvaluatorDOT string
}

// FactoryOutcome captures the result of a full factory run including both
// pipeline outcomes and iteration metadata.
type FactoryOutcome struct {
	// Status is the overall factory result.
	Status state.StageStatus

	// DeveloperOutcome is the final developer pipeline outcome.
	DeveloperOutcome *state.Outcome

	// EvaluatorOutcome is the final evaluator pipeline outcome (nil if the
	// developer failed before reaching evaluation).
	EvaluatorOutcome *state.Outcome

	// Iterations is the number of developer→evaluator cycles executed.
	Iterations int

	// Notes is a human-readable summary of what happened.
	Notes string

	// FailureReason explains why the factory failed, if applicable.
	FailureReason string
}

func (f *FactoryRunner) maxRejections() int {
	if f.MaxRejections > 0 {
		return f.MaxRejections
	}
	return 3
}

func (f *FactoryRunner) developerDOT() string {
	if f.DeveloperDOT != "" {
		return f.DeveloperDOT
	}
	return pipelines.Default()
}

func (f *FactoryRunner) evaluatorDOT() string {
	if f.EvaluatorDOT != "" {
		return f.EvaluatorDOT
	}
	return pipelines.Evaluator()
}

// Run executes the factory loop using the goal from the developer pipeline's
// graph attributes. The logsRoot directory is used as the parent for
// per-iteration subdirectories.
func (f *FactoryRunner) Run(ctx context.Context, logsRoot string) (*FactoryOutcome, error) {
	return f.RunWithGoal(ctx, "", logsRoot)
}

// RunWithGoal executes the factory loop with an explicit goal. If goal is
// empty, the developer pipeline's graph-level goal attribute is used.
func (f *FactoryRunner) RunWithGoal(ctx context.Context, goal string, logsRoot string) (*FactoryOutcome, error) {
	maxReject := f.maxRejections()
	var lastDevOutcome *state.Outcome
	var lastEvalOutcome *state.Outcome

	for iteration := 0; iteration <= maxReject; iteration++ {
		// --- Developer pipeline ---
		devInitCtx := make(map[string]any)
		if goal != "" {
			devInitCtx["goal"] = goal
		}

		// On subsequent iterations, inject evaluator feedback.
		if lastEvalOutcome != nil {
			feedback := extractEvaluatorFeedback(lastEvalOutcome)
			for k, v := range feedback {
				devInitCtx[k] = v
			}
		}

		devLogsDir := filepath.Join(logsRoot, fmt.Sprintf("developer-%03d", iteration))
		devRunner := f.buildRunner(devInitCtx)
		devOutcome, err := devRunner.RunDOT(ctx, f.developerDOT(), devLogsDir)
		if err != nil {
			return &FactoryOutcome{
				Status:           state.StatusFail,
				DeveloperOutcome: devOutcome,
				Iterations:       iteration + 1,
				FailureReason:    fmt.Sprintf("developer pipeline error: %v", err),
			}, err
		}

		lastDevOutcome = devOutcome

		if devOutcome.Status == state.StatusFail {
			return &FactoryOutcome{
				Status:           state.StatusFail,
				DeveloperOutcome: devOutcome,
				Iterations:       iteration + 1,
				Notes:            "Developer pipeline failed",
				FailureReason:    devOutcome.FailureReason,
			}, nil
		}

		// --- Extract submission for evaluator ---
		submission := extractDeveloperSubmission(devOutcome)

		// --- Evaluator pipeline ---
		evalLogsDir := filepath.Join(logsRoot, fmt.Sprintf("evaluator-%03d", iteration))
		evalRunner := f.buildRunner(submission)
		evalOutcome, err := evalRunner.RunDOT(ctx, f.evaluatorDOT(), evalLogsDir)
		if err != nil {
			return &FactoryOutcome{
				Status:           state.StatusFail,
				DeveloperOutcome: devOutcome,
				EvaluatorOutcome: evalOutcome,
				Iterations:       iteration + 1,
				FailureReason:    fmt.Sprintf("evaluator pipeline error: %v", err),
			}, err
		}

		lastEvalOutcome = evalOutcome

		// --- Check evaluator decision ---
		if !isEvaluatorRejection(evalOutcome) {
			// Approved
			return &FactoryOutcome{
				Status:           state.StatusSuccess,
				DeveloperOutcome: devOutcome,
				EvaluatorOutcome: evalOutcome,
				Iterations:       iteration + 1,
				Notes:            fmt.Sprintf("Approved on iteration %d", iteration+1),
			}, nil
		}

		// Rejected — loop continues (unless we've hit the cap).
	}

	return &FactoryOutcome{
		Status:           state.StatusFail,
		DeveloperOutcome: lastDevOutcome,
		EvaluatorOutcome: lastEvalOutcome,
		Iterations:       maxReject + 1,
		Notes:            "Max rejections exceeded",
		FailureReason:    fmt.Sprintf("evaluator rejected %d times, giving up", maxReject+1),
	}, nil
}

func (f *FactoryRunner) buildRunner(initialCtx map[string]any) *engine.Runner {
	registry := f.Registry
	if registry == nil {
		registry = handler.DefaultRegistry(handler.NoopHandler{})
	}

	transforms := f.Transforms
	if transforms == nil {
		transforms = transform.DefaultTransforms()
	}

	return &engine.Runner{
		Registry:       registry,
		Transforms:     transforms,
		OnEvent:        f.OnEvent,
		MaxSteps:       f.MaxSteps,
		InitialContext: initialCtx,
	}
}
