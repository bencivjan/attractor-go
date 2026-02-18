// Package engine implements the core Attractor pipeline execution engine.
// It traverses a parsed and validated pipeline graph from the start node to
// a terminal node, executing handlers at each step, selecting the next edge
// via the 5-step algorithm, enforcing goal gates, and saving checkpoints for
// crash recovery.
package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/strongdm/attractor-go/attractor/condition"
	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/state"
	"github.com/strongdm/attractor-go/attractor/validation"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Config holds settings for a single pipeline run.
type Config struct {
	// LogsRoot is the directory where run logs, checkpoints, and artifacts
	// are written. A unique run subdirectory is created beneath it.
	LogsRoot string

	// Registry maps node types to handlers.
	Registry *handler.Registry

	// OnEvent is an optional callback invoked for lifecycle events during
	// execution. It is never called concurrently.
	OnEvent func(Event)

	// MaxSteps is a safety limit on the total number of node executions in a
	// single run. This prevents infinite loops from consuming resources
	// unboundedly. Defaults to 1000 when zero.
	MaxSteps int

	// InitialContext is an optional set of key-value pairs applied to the
	// pipeline context after graph attributes are mirrored but before
	// execution begins. This allows callers (such as FactoryRunner) to
	// seed the context with values from a prior pipeline run.
	InitialContext map[string]any
}

func (c Config) maxSteps() int {
	if c.MaxSteps > 0 {
		return c.MaxSteps
	}
	return 1000
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

// EventKind distinguishes the lifecycle events emitted during execution.
type EventKind string

const (
	EventPipelineStarted    EventKind = "pipeline_started"
	EventPipelineCompleted  EventKind = "pipeline_completed"
	EventPipelineFailed     EventKind = "pipeline_failed"
	EventStageStarted       EventKind = "stage_started"
	EventStageCompleted     EventKind = "stage_completed"
	EventStageFailed        EventKind = "stage_failed"
	EventStageRetrying      EventKind = "stage_retrying"
	EventCheckpointSaved    EventKind = "checkpoint_saved"
	EventInterviewStarted   EventKind = "interview_started"
	EventInterviewCompleted EventKind = "interview_completed"
	EventParallelStarted    EventKind = "parallel_started"
	EventParallelCompleted  EventKind = "parallel_completed"
)

// Event is a structured lifecycle event emitted during pipeline execution.
type Event struct {
	Kind      EventKind      `json:"kind"`
	Timestamp time.Time      `json:"timestamp"`
	Data      map[string]any `json:"data,omitempty"`
}

// ---------------------------------------------------------------------------
// Retry policy
// ---------------------------------------------------------------------------

// RetryPolicy governs how handler failures are retried.
type RetryPolicy struct {
	MaxAttempts   int
	InitialDelay  time.Duration
	BackoffFactor float64
	MaxDelay      time.Duration
	Jitter        bool
	ShouldRetry   func(error) bool
}

// defaultRetryPolicy returns a sensible baseline retry policy.
func defaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxAttempts:   3,
		InitialDelay:  1 * time.Second,
		BackoffFactor: 2.0,
		MaxDelay:      60 * time.Second,
		Jitter:        true,
	}
}

// ---------------------------------------------------------------------------
// Run -- the core execution entry point
// ---------------------------------------------------------------------------

// Run executes a parsed and validated graph from start to completion.
//
// The execution loop implements the core algorithm from the Attractor spec:
//  1. Find start node
//  2. Loop:
//     a. Check if terminal node -> check goal gates -> break or retry
//     b. Execute node handler with retry policy
//     c. Record completion, apply context updates
//     d. Save checkpoint
//     e. Select next edge (5-step algorithm)
//     f. Handle loop_restart
//     g. Advance to next node
func Run(ctx context.Context, g *graph.Graph, cfg Config) (*state.Outcome, error) {
	// Validate the graph before execution.
	if _, err := validation.ValidateOrError(g); err != nil {
		return nil, err
	}

	// Find the start node.
	startNode, err := g.FindStartNode()
	if err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}

	// Initialize pipeline context with graph-level attributes.
	pctx := state.NewContext()
	mirrorGraphAttributes(g, pctx)

	// Apply caller-provided initial context (e.g. from FactoryRunner).
	if len(cfg.InitialContext) > 0 {
		pctx.ApplyUpdates(cfg.InitialContext)
	}

	// Set pipeline context keys (Section 5.1).
	startTime := time.Now()
	pctx.Set("pipeline.name", g.Name)
	pctx.Set("pipeline.start_time", startTime.Format(time.RFC3339))

	// Create the run directory.
	runID := fmt.Sprintf("run-%s", startTime.UTC().Format("20060102-150405"))
	runDir := filepath.Join(cfg.LogsRoot, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return nil, fmt.Errorf("engine: create run directory: %w", err)
	}

	// Write manifest.json (Section 5.6).
	writeManifest(runDir, g, startTime)

	emit(cfg, Event{Kind: EventPipelineStarted, Timestamp: time.Now(), Data: map[string]any{
		"graph":  g.Name,
		"run_id": runID,
	}})

	// Run the main execution loop.
	outcome, err := executeLoop(ctx, g, startNode, pctx, cfg, runDir)

	// Save final checkpoint.
	snap := pctx.Snapshot()
	finalCP := &state.Checkpoint{
		Timestamp:     time.Now(),
		CurrentNode:   "terminal",
		ContextValues: snap,
		Logs:          pctx.Logs(),
	}
	cpPath := filepath.Join(runDir, "checkpoint.json")
	_ = finalCP.Save(cpPath)

	if err != nil {
		emit(cfg, Event{Kind: EventPipelineFailed, Timestamp: time.Now(), Data: map[string]any{
			"error": err.Error(),
		}})
		return nil, err
	}

	emit(cfg, Event{Kind: EventPipelineCompleted, Timestamp: time.Now(), Data: map[string]any{
		"status": string(outcome.Status),
	}})

	return outcome, nil
}

// ResumeFromCheckpoint resumes execution from a saved checkpoint, continuing
// from the checkpoint's current node with restored context and completed node
// list.
func ResumeFromCheckpoint(ctx context.Context, g *graph.Graph, checkpoint *state.Checkpoint, cfg Config) (*state.Outcome, error) {
	if _, err := validation.ValidateOrError(g); err != nil {
		return nil, err
	}

	resumeNode, ok := g.Nodes[checkpoint.CurrentNode]
	if !ok {
		return nil, fmt.Errorf("engine: checkpoint references node '%s' which is not in the graph", checkpoint.CurrentNode)
	}

	// Restore context from checkpoint.
	pctx := state.NewContext()
	pctx.ApplyUpdates(checkpoint.ContextValues)
	for _, logEntry := range checkpoint.Logs {
		pctx.AppendLog(logEntry)
	}

	runID := fmt.Sprintf("run-%s-resumed", time.Now().UTC().Format("20060102-150405"))
	runDir := filepath.Join(cfg.LogsRoot, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return nil, fmt.Errorf("engine: create run directory: %w", err)
	}

	emit(cfg, Event{Kind: EventPipelineStarted, Timestamp: time.Now(), Data: map[string]any{
		"graph":   g.Name,
		"run_id":  runID,
		"resumed": true,
	}})

	return executeLoop(ctx, g, resumeNode, pctx, cfg, runDir)
}

// ---------------------------------------------------------------------------
// Core execution loop
// ---------------------------------------------------------------------------

// executeLoop is the heart of the engine. It iterates from the given start
// node until a terminal node is reached (or an error/limit is hit).
func executeLoop(
	ctx context.Context,
	g *graph.Graph,
	startNode *graph.Node,
	pctx *state.Context,
	cfg Config,
	runDir string,
) (*state.Outcome, error) {
	currentNode := startNode
	completedNodes := make(map[string]bool)
	nodeOutcomes := make(map[string]*state.Outcome)
	maxSteps := cfg.maxSteps()
	stageStep := 0 // per-loop step counter for stage directory naming

	for step := 0; step < maxSteps; step++ {
		// Check for context cancellation.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Step 2a: Check if this is a terminal node.
		if isTerminal(currentNode, g) {
			allSatisfied, failedGate := checkGoalGates(g, pctx)
			if allSatisfied {
				// All goal gates satisfied (or none defined) -- pipeline complete.
				snap := pctx.Snapshot()
				return &state.Outcome{
					Status:         state.StatusSuccess,
					Notes:          fmt.Sprintf("Pipeline completed at terminal node '%s'", currentNode.ID),
					ContextUpdates: snap,
				}, nil
			}

			// Goal gates unsatisfied -- resolve retry target and jump.
			retryTarget := getRetryTarget(failedGate, g)
			if retryTarget == "" {
				return &state.Outcome{
					Status:        state.StatusFail,
					FailureReason: fmt.Sprintf("Goal gate '%s' unsatisfied and no retry target configured", failedGate.ID),
				}, nil
			}

			targetNode, ok := g.Nodes[retryTarget]
			if !ok {
				return nil, fmt.Errorf("engine: retry target '%s' not found in graph", retryTarget)
			}
			currentNode = targetNode
			continue
		}

		// Step 2b: Execute node handler with retry policy.
		emit(cfg, Event{Kind: EventStageStarted, Timestamp: time.Now(), Data: map[string]any{
			"node":  currentNode.ID,
			"label": currentNode.Label(),
			"step":  step,
		}})

		policy := buildRetryPolicy(currentNode, g)
		h := cfg.Registry.Resolve(currentNode)

		stageDir := filepath.Join(runDir, fmt.Sprintf("%03d-%s", stageStep, currentNode.ID))
		_ = os.MkdirAll(stageDir, 0o755)
		stageStep++

		outcome := executeWithRetry(ctx, h, currentNode, pctx, g, stageDir, policy, cfg)

		// Step 2c: Record completion and apply context updates.
		completedNodes[currentNode.ID] = true
		nodeOutcomes[currentNode.ID] = outcome

		pctx.Set(fmt.Sprintf("status.%s", currentNode.ID), strings.ToLower(string(outcome.Status)))
		pctx.Set("outcome", strings.ToLower(string(outcome.Status)))
		if outcome.PreferredLabel != "" {
			pctx.Set("preferred_label", outcome.PreferredLabel)
		}
		pctx.ApplyUpdates(outcome.ContextUpdates)
		pctx.AppendLog(fmt.Sprintf("[%s] Node '%s' completed with status '%s'",
			time.Now().Format(time.RFC3339), currentNode.ID, outcome.Status))

		if err := writeStatus(stageDir, outcome); err != nil {
			// Non-fatal: log but continue execution.
			pctx.AppendLog(fmt.Sprintf("WARNING: failed to write status for '%s': %s", currentNode.ID, err))
		}

		if outcome.Status == state.StatusFail {
			emit(cfg, Event{Kind: EventStageFailed, Timestamp: time.Now(), Data: map[string]any{
				"node":   currentNode.ID,
				"reason": outcome.FailureReason,
			}})
		} else {
			emit(cfg, Event{Kind: EventStageCompleted, Timestamp: time.Now(), Data: map[string]any{
				"node":   currentNode.ID,
				"status": string(outcome.Status),
			}})
		}

		// Step 2d: Save checkpoint.
		completedList := make([]string, 0, len(completedNodes))
		for id := range completedNodes {
			completedList = append(completedList, id)
		}
		sort.Strings(completedList)

		cp := &state.Checkpoint{
			Timestamp:      time.Now(),
			CurrentNode:    currentNode.ID,
			CompletedNodes: completedList,
			NodeRetries:    make(map[string]int),
			ContextValues:  pctx.Snapshot(),
			Logs:           pctx.Logs(),
		}
		cpPath := filepath.Join(runDir, "checkpoint.json")
		if err := cp.Save(cpPath); err != nil {
			pctx.AppendLog(fmt.Sprintf("WARNING: checkpoint save failed: %s", err))
		} else {
			emit(cfg, Event{Kind: EventCheckpointSaved, Timestamp: time.Now(), Data: map[string]any{
				"path": cpPath,
			}})
		}

		// Step 2e: Select next edge via the 5-step algorithm.
		edge := selectEdge(currentNode, outcome, pctx, g)
		if edge == nil && outcome.Status == state.StatusFail {
			// No matching edge for a failed node -- try retry_target chain
			// (Section 3.7): node retry_target, graph retry_target, fallback.
			retryTarget := getRetryTarget(currentNode, g)
			if retryTarget != "" {
				if targetNode, ok := g.Nodes[retryTarget]; ok {
					currentNode = targetNode
					continue
				}
			}
		}
		if edge == nil {
			// No outgoing edge found -- treat as implicit terminal.
			// Return the last outcome since the pipeline has nowhere to go.
			return outcome, nil
		}

		// Step 2f: Handle loop_restart -- reset mutable context, create fresh
		// log directory, and clear completed tracking (Section 3.2 step 7).
		if edge.LoopRestart() {
			pctx.AppendLog(fmt.Sprintf("Loop restart on edge %s -> %s", edge.From, edge.To))
			pctx.Clear()
			mirrorGraphAttributes(g, pctx)

			// Create a new run subdirectory for the restarted loop.
			restartID := fmt.Sprintf("run-%s-restart", time.Now().UTC().Format("20060102-150405.000"))
			runDir = filepath.Join(cfg.LogsRoot, restartID)
			_ = os.MkdirAll(runDir, 0o755)

			completedNodes = make(map[string]bool)
			nodeOutcomes = make(map[string]*state.Outcome)
			stageStep = 0
		}

		// Step 2g: Advance to next node.
		nextNode, ok := g.Nodes[edge.To]
		if !ok {
			return nil, fmt.Errorf("engine: next node '%s' not found in graph", edge.To)
		}
		currentNode = nextNode
	}

	return &state.Outcome{
		Status:        state.StatusFail,
		FailureReason: fmt.Sprintf("Pipeline exceeded maximum step count (%d)", maxSteps),
	}, nil
}

// ---------------------------------------------------------------------------
// Edge selection -- 5-step algorithm
// ---------------------------------------------------------------------------

// selectEdge implements the 5-step edge selection algorithm:
//  1. Condition-matching edges (evaluate against context/outcome)
//  2. Preferred label match (normalized)
//  3. Suggested next IDs
//  4. Highest weight among unconditional edges
//  5. Lexical tiebreak on target node ID
//
// Returns nil if the node has no outgoing edges.
func selectEdge(node *graph.Node, outcome *state.Outcome, pctx *state.Context, g *graph.Graph) *graph.Edge {
	outEdges := g.OutgoingEdges(node.ID)
	if len(outEdges) == 0 {
		return nil
	}

	// Step 1: Condition-matching edges.
	var conditionEdges []*graph.Edge
	for _, edge := range outEdges {
		cond := edge.Condition()
		if cond == "" {
			continue
		}
		if condition.Evaluate(cond, outcome, pctx) {
			conditionEdges = append(conditionEdges, edge)
		}
	}
	if len(conditionEdges) > 0 {
		return bestByWeightThenLexical(conditionEdges)
	}

	// Step 2: Preferred label match.
	if outcome != nil && outcome.PreferredLabel != "" {
		normalized := normalizeLabel(outcome.PreferredLabel)
		for _, edge := range outEdges {
			if normalizeLabel(edge.Label()) == normalized {
				return edge
			}
		}
	}

	// Step 3: Suggested next IDs from the outcome.
	if outcome != nil && len(outcome.SuggestedNextIDs) > 0 {
		suggestedSet := make(map[string]bool, len(outcome.SuggestedNextIDs))
		for _, id := range outcome.SuggestedNextIDs {
			suggestedSet[id] = true
		}
		var suggestedEdges []*graph.Edge
		for _, edge := range outEdges {
			if suggestedSet[edge.To] {
				suggestedEdges = append(suggestedEdges, edge)
			}
		}
		if len(suggestedEdges) > 0 {
			return bestByWeightThenLexical(suggestedEdges)
		}
	}

	// Steps 4 & 5: Highest weight, then lexical tiebreak.
	// Prefer unconditional edges when available.
	var unconditional []*graph.Edge
	for _, edge := range outEdges {
		if edge.Condition() == "" {
			unconditional = append(unconditional, edge)
		}
	}
	candidates := unconditional
	if len(candidates) == 0 {
		candidates = outEdges
	}
	return bestByWeightThenLexical(candidates)
}

// bestByWeightThenLexical picks the edge with the highest weight. Ties are
// broken by lexicographic order of the target node ID (ascending).
func bestByWeightThenLexical(edges []*graph.Edge) *graph.Edge {
	if len(edges) == 0 {
		return nil
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Weight() != edges[j].Weight() {
			return edges[i].Weight() > edges[j].Weight() // descending weight
		}
		return edges[i].To < edges[j].To // ascending lexical
	})
	return edges[0]
}

// normalizeLabel lowercases, trims whitespace, and strips common accelerator
// prefixes from labels for comparison. Handles patterns like:
//   - "[K] Label" -> "label"
//   - "K) Label"  -> "label"
//   - "K - Label" -> "label"
func normalizeLabel(label string) string {
	trimmed := strings.TrimSpace(strings.ToLower(label))
	if trimmed == "" {
		return trimmed
	}

	// Pattern 1: [X] rest
	if trimmed[0] == '[' {
		if idx := strings.Index(trimmed, "]"); idx >= 0 {
			rest := strings.TrimSpace(trimmed[idx+1:])
			if rest != "" {
				return rest
			}
		}
	}

	// Pattern 2: X) rest
	if idx := strings.Index(trimmed, ")"); idx > 0 && idx < 4 {
		rest := strings.TrimSpace(trimmed[idx+1:])
		if rest != "" {
			return rest
		}
	}

	// Pattern 3: X - rest
	if idx := strings.Index(trimmed, " - "); idx > 0 && idx < 4 {
		rest := strings.TrimSpace(trimmed[idx+3:])
		if rest != "" {
			return rest
		}
	}

	return trimmed
}

// ---------------------------------------------------------------------------
// Terminal and goal gate checking
// ---------------------------------------------------------------------------

// isTerminal checks if a node is a terminal (exit) node. A node is terminal
// if it has an exit shape (Msquare) or type "exit", and has no outgoing edges.
func isTerminal(node *graph.Node, g *graph.Graph) bool {
	shape := node.Shape()
	isExitShape := strings.EqualFold(shape, "Msquare")
	isExitType := node.Type() == "exit"

	if !isExitShape && !isExitType {
		return false
	}
	return len(g.OutgoingEdges(node.ID)) == 0
}

// checkGoalGates checks whether all goal_gate=true nodes have succeeded.
// Returns (true, nil) if all gates pass (or none are defined).
// Returns (false, failedNode) with the first unsatisfied gate node.
func checkGoalGates(g *graph.Graph, pctx *state.Context) (bool, *graph.Node) {
	for _, node := range g.Nodes {
		if !node.GoalGate() {
			continue
		}
		statusKey := fmt.Sprintf("status.%s", node.ID)
		statusStr := pctx.GetString(statusKey, "")
		if statusStr != "success" && statusStr != "partial_success" {
			return false, node
		}
	}
	return true, nil
}

// getRetryTarget resolves the retry target for a failed goal gate node.
// Resolution order: node-level retry_target, graph-level retry_target,
// graph-level fallback_retry_target.
func getRetryTarget(node *graph.Node, g *graph.Graph) string {
	if node != nil {
		if target := node.RetryTarget(); target != "" {
			return target
		}
	}
	if target := g.RetryTarget(); target != "" {
		return target
	}
	return g.FallbackRetryTarget()
}

// ---------------------------------------------------------------------------
// Handler execution with retry
// ---------------------------------------------------------------------------

// executeWithRetry runs the handler for a node, retrying on failure according
// to the retry policy. It respects context cancellation between attempts.
func executeWithRetry(
	ctx context.Context,
	h handler.Handler,
	node *graph.Node,
	pctx *state.Context,
	g *graph.Graph,
	logsRoot string,
	policy RetryPolicy,
	cfg Config,
) *state.Outcome {
	var lastOutcome *state.Outcome

	for attempt := 0; attempt < policy.MaxAttempts; attempt++ {
		// Apply node-level timeout if configured.
		execCtx := ctx
		var cancel context.CancelFunc
		if timeout := node.Timeout(); timeout > 0 {
			execCtx, cancel = context.WithTimeout(ctx, timeout)
		}

		outcome, err := h.Execute(execCtx, node, pctx, g, logsRoot)

		if cancel != nil {
			cancel()
		}

		if err != nil {
			// Handler returned an error -- wrap it into a failure outcome.
			outcome = &state.Outcome{
				Status:        state.StatusFail,
				FailureReason: err.Error(),
			}
		}

		lastOutcome = outcome

		// Check whether we should retry.
		isRetryable := outcome.Status == state.StatusFail || outcome.Status == state.StatusRetry
		if !isRetryable || attempt >= policy.MaxAttempts-1 {
			break
		}

		// If the error is not retryable per the policy callback, stop.
		if err != nil && policy.ShouldRetry != nil && !policy.ShouldRetry(err) {
			break
		}

		emit(cfg, Event{Kind: EventStageRetrying, Timestamp: time.Now(), Data: map[string]any{
			"node":    node.ID,
			"attempt": attempt + 1,
			"status":  string(outcome.Status),
		}})

		// Wait with backoff before retrying.
		delay := delayForAttempt(attempt, policy)
		select {
		case <-ctx.Done():
			return &state.Outcome{
				Status:        state.StatusFail,
				FailureReason: ctx.Err().Error(),
			}
		case <-time.After(delay):
		}
	}

	return lastOutcome
}

// delayForAttempt calculates the backoff delay for a given attempt number.
//
//	delay = min(InitialDelay * BackoffFactor^attempt, MaxDelay)
//
// When Jitter is enabled, the result is multiplied by a random factor in
// [0.5, 1.5) to decorrelate concurrent retriers.
func delayForAttempt(attempt int, policy RetryPolicy) time.Duration {
	delay := float64(policy.InitialDelay) * math.Pow(policy.BackoffFactor, float64(attempt))

	if delay > float64(policy.MaxDelay) {
		delay = float64(policy.MaxDelay)
	}

	if policy.Jitter {
		jitterFactor := 0.5 + rand.Float64() // [0.5, 1.5)
		delay *= jitterFactor
	}

	return time.Duration(delay)
}

// buildRetryPolicy constructs a retry policy for a node, merging node-level
// settings with the graph-level defaults.
func buildRetryPolicy(node *graph.Node, g *graph.Graph) RetryPolicy {
	policy := defaultRetryPolicy()

	// Node-level max_retries overrides the default max attempts.
	// max_retries=N means up to N+1 total executions (spec Section 2.6, 3.5).
	if nodeRetries := node.MaxRetries(); nodeRetries > 0 {
		policy.MaxAttempts = nodeRetries + 1
	}

	// Graph-level default_max_retry sets an upper bound if specified.
	graphMax := g.DefaultMaxRetry()
	if graphMax > 0 && policy.MaxAttempts > graphMax {
		policy.MaxAttempts = graphMax
	}

	return policy
}

// ---------------------------------------------------------------------------
// Status and checkpoint helpers
// ---------------------------------------------------------------------------

// writeStatus writes a status.json file in the given stage directory.
func writeStatus(stageDir string, outcome *state.Outcome) error {
	if stageDir == "" {
		return nil
	}

	data, err := json.MarshalIndent(outcome, "", "  ")
	if err != nil {
		return fmt.Errorf("write status: marshal: %w", err)
	}

	path := filepath.Join(stageDir, "status.json")
	return os.WriteFile(path, data, 0o644)
}

// mirrorGraphAttributes copies graph-level attributes into the pipeline
// context so that handlers can reference them. This follows the Scala
// reference implementation's initializeContext behavior.
func mirrorGraphAttributes(g *graph.Graph, pctx *state.Context) {
	for k, v := range g.Attrs {
		pctx.Set(k, v)
	}
}

// writeManifest writes a manifest.json into the run directory with pipeline
// metadata (Section 5.6).
func writeManifest(runDir string, g *graph.Graph, startTime time.Time) {
	manifest := map[string]any{
		"name":       g.Name,
		"start_time": startTime.Format(time.RFC3339),
		"parameters": g.Attrs,
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(runDir, "manifest.json"), data, 0o644)
}

// ---------------------------------------------------------------------------
// Event emitter
// ---------------------------------------------------------------------------

// emit dispatches an event to the callback if one is configured.
func emit(cfg Config, event Event) {
	if cfg.OnEvent != nil {
		cfg.OnEvent(event)
	}
}
