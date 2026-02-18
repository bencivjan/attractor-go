package engine

import (
	"context"
	"fmt"

	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/parser"
	"github.com/strongdm/attractor-go/attractor/pipelines"
	"github.com/strongdm/attractor-go/attractor/state"
	"github.com/strongdm/attractor-go/attractor/transform"
)

// Runner orchestrates the full pipeline lifecycle: parse, validate, transform,
// and execute. It is the primary entry point for callers who want to run a
// pipeline from DOT source or a pre-parsed graph.
type Runner struct {
	// Registry maps node types to their handlers. If nil, a default registry
	// with start/exit/noop handlers is used.
	Registry *handler.Registry

	// Transforms is the ordered list of graph transforms applied before
	// validation and execution. If nil, DefaultTransforms() is used.
	Transforms []transform.Transform

	// OnEvent is an optional callback for lifecycle events emitted during
	// execution. The callback is never invoked concurrently.
	OnEvent func(Event)

	// MaxSteps is a safety limit on the total number of node executions.
	// Defaults to 1000 when zero.
	MaxSteps int
}

// NewRunner creates a Runner with the given handler registry and the default
// transform pipeline. If registry is nil, a registry with only the built-in
// start, exit, and noop handlers is used.
func NewRunner(registry *handler.Registry) *Runner {
	if registry == nil {
		registry = handler.DefaultRegistry(handler.NoopHandler{})
	}
	return &Runner{
		Registry:   registry,
		Transforms: transform.DefaultTransforms(),
	}
}

// RegisterTransform appends a transform to the runner's transform pipeline.
func (r *Runner) RegisterTransform(t transform.Transform) {
	r.Transforms = append(r.Transforms, t)
}

// RunDOT parses DOT source text and runs the pipeline end-to-end.
//
// The lifecycle is:
//
//	PARSE -> TRANSFORM -> VALIDATE -> INITIALIZE -> EXECUTE -> FINALIZE
//
// The logsRoot directory is used as the parent for the run-specific directory
// where checkpoints, status files, and artifacts are written.
func (r *Runner) RunDOT(ctx context.Context, dotSource string, logsRoot string) (*state.Outcome, error) {
	// Phase 1: Parse
	g, err := parser.Parse(dotSource)
	if err != nil {
		return nil, fmt.Errorf("runner: parse failed: %w", err)
	}

	return r.RunGraph(ctx, g, logsRoot)
}

// RunGraph runs a pre-parsed graph through the full pipeline lifecycle:
//
//	TRANSFORM -> VALIDATE -> INITIALIZE -> EXECUTE -> FINALIZE
//
// Validation is performed by the engine.Run function after transforms are
// applied, so callers do not need to validate separately.
func (r *Runner) RunGraph(ctx context.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
	// Phase 2: Transform
	transforms := r.Transforms
	if transforms == nil {
		transforms = transform.DefaultTransforms()
	}
	g = transform.ApplyAll(g, transforms...)

	// Phase 3-6: Validate, Initialize, Execute, Finalize (handled by engine.Run)
	cfg := r.buildConfig(logsRoot)
	return Run(ctx, g, cfg)
}

// RunFromCheckpoint resumes a pipeline from a saved checkpoint file.
// The graph is transformed before resuming. The checkpoint file is loaded,
// and execution continues from the checkpoint's current node with the
// restored context.
func (r *Runner) RunFromCheckpoint(ctx context.Context, g *graph.Graph, checkpointPath string, logsRoot string) (*state.Outcome, error) {
	// Phase 1: Load checkpoint
	cp, err := state.LoadCheckpoint(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("runner: %w", err)
	}

	// Phase 2: Transform
	transforms := r.Transforms
	if transforms == nil {
		transforms = transform.DefaultTransforms()
	}
	g = transform.ApplyAll(g, transforms...)

	// Phase 3-6: Validate and Execute from checkpoint
	cfg := r.buildConfig(logsRoot)
	return ResumeFromCheckpoint(ctx, g, cp, cfg)
}

// RunDefault runs the built-in default pipeline (Plan-Build-Verify).
// This is a convenience method equivalent to RunPipeline with the default
// pipeline name.
func (r *Runner) RunDefault(ctx context.Context, logsRoot string) (*state.Outcome, error) {
	return r.RunPipeline(ctx, pipelines.DefaultName, logsRoot)
}

// RunPipeline runs a named pipeline from the built-in pipeline catalog.
// Use pipelines.Register() to add custom pipelines to the catalog.
func (r *Runner) RunPipeline(ctx context.Context, name string, logsRoot string) (*state.Outcome, error) {
	dotSource := pipelines.Get(name)
	if dotSource == "" {
		return nil, fmt.Errorf("runner: pipeline %q not found in catalog", name)
	}
	return r.RunDOT(ctx, dotSource, logsRoot)
}

// buildConfig creates an engine.Config from the runner's settings.
func (r *Runner) buildConfig(logsRoot string) Config {
	registry := r.Registry
	if registry == nil {
		registry = handler.DefaultRegistry(handler.NoopHandler{})
	}

	return Config{
		LogsRoot: logsRoot,
		Registry: registry,
		OnEvent:  r.OnEvent,
		MaxSteps: r.MaxSteps,
	}
}
