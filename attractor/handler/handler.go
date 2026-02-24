// Package handler defines the Handler interface and Registry for pipeline node
// execution. Each node type in the pipeline graph maps to a Handler that knows
// how to execute it. The Registry resolves nodes to their handlers using a
// three-step lookup: explicit type attribute, shape-based mapping, then default.
//
// This package provides all built-in handlers for the Attractor pipeline engine:
// start, exit, codergen (LLM), wait.human, conditional, parallel, parallel.fan_in,
// tool, and stack.manager_loop.
package handler

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/strongdm/attractor-go/attractor/condition"
	"github.com/strongdm/attractor-go/attractor/graph"
	"github.com/strongdm/attractor-go/attractor/interviewer"
	"github.com/strongdm/attractor-go/attractor/state"
)

// ---------------------------------------------------------------------------
// Handler interface
// ---------------------------------------------------------------------------

// Handler knows how to execute a specific type of pipeline node.
type Handler interface {
	Execute(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error)
}

// HandlerFunc adapts a plain function to the Handler interface.
type HandlerFunc func(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error)

// Execute implements the Handler interface by delegating to the wrapped function.
func (f HandlerFunc) Execute(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
	return f(ctx, node, pctx, g, logsRoot)
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

// Registry maps node types and shapes to their handlers.
//
// Resolution order:
//  1. Explicit "type" attribute on the node
//  2. Shape-based lookup via the ShapeToType mapping
//  3. Fall back to the default handler
type Registry struct {
	mu             sync.RWMutex
	handlers       map[string]Handler
	defaultHandler Handler
}

// ShapeToType maps DOT shapes to handler type strings. This mapping
// implements the Attractor spec's shape-to-type convention.
var ShapeToType = map[string]string{
	"Mdiamond":      "start",
	"Msquare":       "exit",
	"box":           "codergen",
	"hexagon":       "wait.human",
	"diamond":       "conditional",
	"component":     "parallel",
	"tripleoctagon": "parallel.fan_in",
	"parallelogram":  "tool",
	"house":          "stack.manager_loop",
	"doubleoctagon":  "communication",
}

// NewRegistry creates a Registry with the given default handler.
// The default handler is used when no explicit type or shape mapping resolves.
func NewRegistry(defaultHandler Handler) *Registry {
	return &Registry{
		handlers:       make(map[string]Handler),
		defaultHandler: defaultHandler,
	}
}

// Register adds a handler for the given type string.
func (r *Registry) Register(typeString string, h Handler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.handlers[typeString] = h
}

// Resolve returns the appropriate handler for a node.
func (r *Registry) Resolve(node *graph.Node) Handler {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Step 1: Explicit type attribute.
	if t := node.Type(); t != "" {
		if h, ok := r.handlers[t]; ok {
			return h
		}
	}

	// Step 2: Shape-based resolution.
	if shape := node.Shape(); shape != "" {
		if handlerType, ok := ShapeToType[shape]; ok {
			if h, ok := r.handlers[handlerType]; ok {
				return h
			}
		}
	}

	// Step 3: Default.
	return r.defaultHandler
}

// ---------------------------------------------------------------------------
// CodergenBackend interface
// ---------------------------------------------------------------------------

// CodergenBackend is the pluggable interface for LLM-powered code generation
// stages. The returned value should be either:
//   - A string containing the raw LLM response text on success
//   - A *state.Outcome for terminal/unrecoverable failures
//
// The error return is for unexpected system errors (not LLM failures).
type CodergenBackend interface {
	Run(node *graph.Node, prompt string, ctx *state.Context) (any, error)
}

// ---------------------------------------------------------------------------
// PipelineExecutor interface
// ---------------------------------------------------------------------------

// PipelineExecutor is the interface used by handlers that need to run child
// pipelines (such as StackManagerLoopHandler). It is defined here to avoid
// circular imports between the handler and engine packages -- the engine
// package implements this interface on its Runner type.
type PipelineExecutor interface {
	// ExecuteDOT parses and executes a DOT pipeline, returning the final
	// outcome. The initialContext seeds the child pipeline's context before
	// execution begins.
	ExecuteDOT(ctx context.Context, dotSource string, logsRoot string, initialContext map[string]any) (*state.Outcome, error)
}

// ---------------------------------------------------------------------------
// StartHandler
// ---------------------------------------------------------------------------

// StartHandler is a no-op pass-through for the pipeline entry point.
// It returns Success immediately with no side effects.
type StartHandler struct{}

// Execute marks the entry point of the pipeline and returns success.
func (h *StartHandler) Execute(_ context.Context, node *graph.Node, _ *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Start node '%s' entered", node.ID),
	}, nil
}

// ---------------------------------------------------------------------------
// ExitHandler
// ---------------------------------------------------------------------------

// ExitHandler is a no-op terminal handler for the pipeline exit point.
// It returns Success immediately with no side effects.
type ExitHandler struct{}

// Execute marks the end of the pipeline and returns success.
func (h *ExitHandler) Execute(_ context.Context, node *graph.Node, _ *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Exit node '%s' reached", node.ID),
	}, nil
}

// ---------------------------------------------------------------------------
// CodergenHandler
// ---------------------------------------------------------------------------

// CodergenHandler executes an LLM code-generation stage:
//  1. Builds a prompt from the node's "prompt" attribute (falls back to "label")
//  2. Expands the $goal variable from context or graph attributes
//  3. Creates a stage directory and writes prompt.md
//  4. Invokes the backend (or simulates if nil)
//  5. Writes response.md
//  6. Returns an Outcome with last_stage and last_response context updates
type CodergenHandler struct {
	Backend CodergenBackend // nil = simulation mode
}

// Execute runs the LLM generation stage.
func (h *CodergenHandler) Execute(_ context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
	stageDir := filepath.Join(logsRoot, node.ID)
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		return nil, fmt.Errorf("codergen: create stage dir: %w", err)
	}

	// Build the prompt with variable expansion.
	prompt := buildPrompt(node, pctx, g)

	// Write prompt.md.
	_ = os.WriteFile(filepath.Join(stageDir, "prompt.md"), []byte(prompt), 0o644)

	// Call the backend or simulate.
	var responseText string
	var outcome *state.Outcome

	if h.Backend != nil {
		result, err := h.Backend.Run(node, prompt, pctx)
		if err != nil {
			return nil, fmt.Errorf("codergen: backend error: %w", err)
		}
		switch v := result.(type) {
		case string:
			responseText = v
		case *state.Outcome:
			outcome = v
			responseText = v.Notes
		default:
			responseText = fmt.Sprintf("%v", result)
		}
	} else {
		// Simulation mode: produce a synthetic response.
		promptPreview := prompt
		if len(promptPreview) > 100 {
			promptPreview = promptPreview[:100]
		}
		responseText = fmt.Sprintf("[simulated] Response for: %s", promptPreview)
	}

	// If the backend returned a terminal outcome, use it directly.
	if outcome == nil {
		truncated := responseText
		if len(truncated) > 200 {
			truncated = truncated[:200] + "..."
		}
		outcome = &state.Outcome{
			Status: state.StatusSuccess,
			Notes:  fmt.Sprintf("Codergen stage '%s' completed", node.ID),
			ContextUpdates: map[string]any{
				"last_stage":    node.ID,
				"last_response": truncated,
			},
		}
	}

	// Write response.md.
	_ = os.WriteFile(filepath.Join(stageDir, "response.md"), []byte(responseText), 0o644)

	return outcome, nil
}

// buildPrompt constructs the LLM prompt from node attributes and expands
// the $goal variable using context or graph-level attributes.
func buildPrompt(node *graph.Node, pctx *state.Context, g *graph.Graph) string {
	raw := node.Prompt()
	if raw == "" {
		raw = node.Label()
	}

	// Resolve the effective goal: context value takes precedence over graph.
	goal := pctx.GetString("goal", "")
	if goal == "" {
		goal = g.Goal()
	}

	return strings.ReplaceAll(raw, "$goal", goal)
}

// ---------------------------------------------------------------------------
// WaitForHumanHandler
// ---------------------------------------------------------------------------

// WaitForHumanHandler blocks until a human selects an option. It derives
// choices from the outgoing edges of the current node and presents them
// through the Interviewer abstraction.
//
// Edge labels are parsed for accelerator keys using conventions defined in
// the interviewer.ParseAcceleratorKey function.
type WaitForHumanHandler struct {
	Interviewer interviewer.Interviewer
}

// Execute presents choices to the human and returns the selected path.
func (h *WaitForHumanHandler) Execute(_ context.Context, node *graph.Node, _ *state.Context, g *graph.Graph, _ string) (*state.Outcome, error) {
	outEdges := g.OutgoingEdges(node.ID)

	// Build choices from edge labels.
	type choice struct {
		option   interviewer.Option
		targetID string
	}
	var choices []choice
	for _, edge := range outEdges {
		raw := strings.TrimSpace(edge.Label())
		if raw == "" {
			continue
		}
		opt := interviewer.ParseAcceleratorKey(raw)
		choices = append(choices, choice{option: opt, targetID: edge.To})
	}

	// Build option list for the question.
	options := make([]interviewer.Option, len(choices))
	for i, c := range choices {
		options[i] = c.option
	}

	// Determine question type based on options.
	qType := interviewer.QuestionMultipleChoice
	if len(options) <= 2 {
		allYesNo := true
		for _, opt := range options {
			key := strings.ToLower(opt.Key)
			if key != "y" && key != "n" && key != "yes" && key != "no" {
				allYesNo = false
				break
			}
		}
		if allYesNo {
			qType = interviewer.QuestionYesNo
		}
	}

	question := interviewer.Question{
		Text:    node.Label(),
		Type:    qType,
		Options: options,
		Stage:   node.ID,
	}

	answer := h.Interviewer.Ask(question)

	// Handle timeout and skipped answers by falling back to the node's
	// default_choice attribute. If no default is configured, return FAIL.
	if answer.Value == interviewer.AnswerTimeout || answer.Value == interviewer.AnswerSkipped {
		defaultChoice := node.Attrs["human.default_choice"]
		if defaultChoice == "" {
			return &state.Outcome{
				Status:        state.StatusFail,
				FailureReason: fmt.Sprintf("Human gate '%s' %s with no default_choice", node.ID, string(answer.Value)),
				ContextUpdates: map[string]any{
					"human.gate.selected": string(answer.Value),
				},
			}, nil
		}
		// Use the default choice to select the edge.
		var nextIDs []string
		var defaultLabel string
		for _, c := range choices {
			if strings.EqualFold(c.option.Key, defaultChoice) {
				nextIDs = append(nextIDs, c.targetID)
				defaultLabel = c.option.Label
			}
		}
		if defaultLabel == "" {
			defaultLabel = defaultChoice
		}
		return &state.Outcome{
			Status:           state.StatusSuccess,
			PreferredLabel:   defaultLabel,
			SuggestedNextIDs: nextIDs,
			Notes:            fmt.Sprintf("Human gate '%s' %s, using default_choice: %s", node.ID, string(answer.Value), defaultLabel),
			ContextUpdates: map[string]any{
				"human.gate.selected": defaultChoice,
				"human.gate.label":    defaultLabel,
			},
		}, nil
	}

	selectedKey := string(answer.Value)
	selectedLabel := answer.Text
	if answer.SelectedOption != nil {
		selectedLabel = answer.SelectedOption.Label
	}

	// Map the selected key back to the target node IDs.
	var nextIDs []string
	for _, c := range choices {
		if strings.EqualFold(c.option.Key, selectedKey) {
			nextIDs = append(nextIDs, c.targetID)
		}
	}

	return &state.Outcome{
		Status:           state.StatusSuccess,
		PreferredLabel:   selectedLabel,
		SuggestedNextIDs: nextIDs,
		Notes:            fmt.Sprintf("Human selected: %s", selectedLabel),
		ContextUpdates: map[string]any{
			"human.gate.selected": selectedKey,
			"human.gate.label":    selectedLabel,
		},
	}, nil
}

// ---------------------------------------------------------------------------
// ConditionalHandler
// ---------------------------------------------------------------------------

// ConditionalHandler is a no-op pass-through. Conditional (diamond) nodes
// are routing points where the actual edge evaluation and branching logic
// is handled by the engine's edge selection algorithm.
type ConditionalHandler struct{}

// Execute returns success to let the engine proceed with edge selection.
func (h *ConditionalHandler) Execute(_ context.Context, node *graph.Node, _ *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Conditional node '%s' evaluated", node.ID),
	}, nil
}

// ---------------------------------------------------------------------------
// ParallelHandler
// ---------------------------------------------------------------------------

// JoinPolicy defines how parallel branch results are aggregated.
type JoinPolicy int

const (
	// JoinWaitAll waits for all branches to complete.
	JoinWaitAll JoinPolicy = iota
	// JoinFirstSuccess returns on first successful branch.
	JoinFirstSuccess
	// JoinKOfN requires K successful branches.
	JoinKOfN
	// JoinQuorum requires a majority of branches to succeed.
	JoinQuorum
)

// ErrorPolicy defines how parallel branch failures are handled.
type ErrorPolicy int

const (
	// ErrorFailFast fails immediately on first branch error.
	ErrorFailFast ErrorPolicy = iota
	// ErrorContinue continues despite branch errors.
	ErrorContinue
	// ErrorIgnore ignores all branch errors.
	ErrorIgnore
)

// BranchEventFunc is an optional callback invoked by ParallelHandler when
// a parallel branch starts or completes. This allows the engine to emit
// structured lifecycle events without introducing a circular import.
type BranchEventFunc func(kind string, data map[string]any)

// ParallelHandler fans out to branches concurrently and aggregates results
// according to the node's join_policy and error_policy attributes.
//
// Node attributes:
//   - join_policy: wait_all (default), first_success, k_of_n, quorum
//   - error_policy: fail_fast (default), continue, ignore
//   - max_parallel: concurrency limit (default 4)
//   - k_value: required successes for k_of_n policy (default 1)
//
// Stores results in context under "parallel.results" as a comma-separated
// string of nodeId:status pairs for downstream fan-in consumption.
type ParallelHandler struct {
	// registry is needed to resolve handlers for branch target nodes.
	registry *Registry
	// OnBranchEvent is an optional callback for parallel branch lifecycle events.
	OnBranchEvent BranchEventFunc
}

// NewParallelHandler creates a ParallelHandler that uses the given registry
// to resolve handlers for parallel branch target nodes.
func NewParallelHandler(registry *Registry) *ParallelHandler {
	return &ParallelHandler{registry: registry}
}

// Execute identifies branches from outgoing edges, executes them
// concurrently with bounded parallelism, and applies the join policy.
func (h *ParallelHandler) Execute(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
	outEdges := g.OutgoingEdges(node.ID)
	if len(outEdges) == 0 {
		return &state.Outcome{
			Status: state.StatusSuccess,
			Notes:  fmt.Sprintf("Parallel node '%s' has no outgoing edges", node.ID),
		}, nil
	}

	joinPolicy := parseJoinPolicy(node.Attrs["join_policy"])
	_ = parseErrorPolicy(node.Attrs["error_policy"]) // parsed for future use by other join policies
	maxParallel := attrIntDefault(node.Attrs, "max_parallel", 4)
	kValue := attrIntDefault(node.Attrs, "k_value", 1)

	// Build branch tasks: each branch gets a cloned context for isolation.
	type branchResult struct {
		id      string
		outcome *state.Outcome
	}

	results := make([]branchResult, len(outEdges))
	sem := make(chan struct{}, maxParallel)
	var wg sync.WaitGroup

	for i, edge := range outEdges {
		wg.Add(1)
		go func(idx int, targetID string) {
			defer wg.Done()
			sem <- struct{}{}        // acquire semaphore slot
			defer func() { <-sem }() // release semaphore slot

			// Emit branch started event.
			if h.OnBranchEvent != nil {
				h.OnBranchEvent("parallel_branch_started", map[string]any{
					"parent_node": node.ID,
					"branch":      targetID,
					"index":       idx,
				})
			}

			branchCtx := pctx.Clone()

			targetNode, ok := g.Nodes[targetID]
			if !ok {
				results[idx] = branchResult{
					id: targetID,
					outcome: &state.Outcome{
						Status:        state.StatusFail,
						FailureReason: fmt.Sprintf("target node '%s' not found", targetID),
					},
				}
				// Emit branch completed event on failure.
				if h.OnBranchEvent != nil {
					h.OnBranchEvent("parallel_branch_completed", map[string]any{
						"parent_node": node.ID,
						"branch":      targetID,
						"index":       idx,
						"status":      "fail",
					})
				}
				return
			}

			branchHandler := h.registry.Resolve(targetNode)
			outcome, err := branchHandler.Execute(ctx, targetNode, branchCtx, g, logsRoot)
			if err != nil {
				outcome = &state.Outcome{
					Status:        state.StatusFail,
					FailureReason: err.Error(),
				}
			}

			results[idx] = branchResult{id: targetID, outcome: outcome}

			// Emit branch completed event.
			if h.OnBranchEvent != nil {
				h.OnBranchEvent("parallel_branch_completed", map[string]any{
					"parent_node": node.ID,
					"branch":      targetID,
					"index":       idx,
					"status":      string(outcome.Status),
				})
			}
		}(i, edge.To)
	}

	wg.Wait()

	// Aggregate results.
	var successes, failures int
	resultParts := make([]string, len(results))
	for i, r := range results {
		resultParts[i] = fmt.Sprintf("%s:%s", r.id, r.outcome.Status)
		switch r.outcome.Status {
		case state.StatusSuccess:
			successes++
		case state.StatusFail:
			failures++
		}
	}

	resultsStr := strings.Join(resultParts, ",")

	// Determine aggregate status based on join policy.
	var aggregateStatus state.StageStatus
	switch joinPolicy {
	case JoinWaitAll:
		if failures == 0 {
			aggregateStatus = state.StatusSuccess
		} else if failures > 0 && failures < len(results) {
			// Some branches failed but not all: partial success.
			aggregateStatus = state.StatusPartialSuccess
		} else {
			// All branches failed.
			aggregateStatus = state.StatusFail
		}
	case JoinFirstSuccess:
		if successes > 0 {
			aggregateStatus = state.StatusSuccess
		} else {
			aggregateStatus = state.StatusFail
		}
	case JoinKOfN:
		if successes >= kValue {
			aggregateStatus = state.StatusSuccess
		} else {
			aggregateStatus = state.StatusFail
		}
	case JoinQuorum:
		if successes > len(results)/2 {
			aggregateStatus = state.StatusSuccess
		} else {
			aggregateStatus = state.StatusFail
		}
	default:
		aggregateStatus = state.StatusSuccess
	}

	return &state.Outcome{
		Status: aggregateStatus,
		Notes:  fmt.Sprintf("Parallel '%s': %d/%d succeeded", node.ID, successes, len(results)),
		ContextUpdates: map[string]any{
			"parallel.results": resultsStr,
		},
	}, nil
}

func parseJoinPolicy(s string) JoinPolicy {
	switch strings.ToLower(s) {
	case "first_success":
		return JoinFirstSuccess
	case "k_of_n":
		return JoinKOfN
	case "quorum":
		return JoinQuorum
	default:
		return JoinWaitAll
	}
}

func parseErrorPolicy(s string) ErrorPolicy {
	switch strings.ToLower(s) {
	case "continue":
		return ErrorContinue
	case "ignore":
		return ErrorIgnore
	default:
		return ErrorFailFast
	}
}

// ---------------------------------------------------------------------------
// FanInHandler
// ---------------------------------------------------------------------------

// FanInHandler consolidates parallel branch results. It reads
// "parallel.results" from context (written by ParallelHandler), selects the
// best candidate by heuristic (success ranks highest, then lexical order),
// and stores the winning branch ID in "parallel.fan_in.best_id" and its
// outcome status in "parallel.fan_in.best_outcome".
//
// When the node has a non-empty prompt attribute and a Backend is configured,
// the handler delegates candidate ranking to the LLM instead of using the
// heuristic. If the LLM call fails or no backend is available, it falls back
// to the heuristic selection.
type FanInHandler struct {
	// Backend is an optional LLM backend used for intelligent candidate
	// ranking when the fan-in node has a prompt attribute.
	Backend CodergenBackend
}

// Execute reads parallel results and selects the best branch.
func (h *FanInHandler) Execute(_ context.Context, node *graph.Node, pctx *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	resultsRaw := pctx.GetString("parallel.results", "")
	if resultsRaw == "" {
		return &state.Outcome{
			Status: state.StatusSuccess,
			Notes:  fmt.Sprintf("Fan-in '%s': no parallel results found", node.ID),
			ContextUpdates: map[string]any{
				"parallel.fan_in.best_id":      "",
				"parallel.fan_in.best_outcome": "",
			},
		}, nil
	}

	// Parse entries: "nodeId:status,nodeId:status,..."
	var entries []fanInEntry
	for _, part := range strings.Split(resultsRaw, ",") {
		parts := strings.SplitN(part, ":", 2)
		if len(parts) == 2 {
			entries = append(entries, fanInEntry{id: parts[0], status: parts[1]})
		}
	}

	bestBranch, bestOutcome := h.selectBest(node, pctx, entries)

	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Fan-in '%s': selected best branch '%s' from %d branches", node.ID, bestBranch, len(entries)),
		ContextUpdates: map[string]any{
			"parallel.fan_in.best_id":      bestBranch,
			"parallel.fan_in.best_outcome": bestOutcome,
			"last_stage":                   node.ID,
		},
	}, nil
}

// fanInEntry is used internally for candidate ranking.
type fanInEntry struct {
	id     string
	status string
}

// selectBest picks the best branch from candidates. When the node has a
// prompt and an LLM backend is available, it delegates to the LLM for
// intelligent ranking. Otherwise it falls back to heuristic selection.
func (h *FanInHandler) selectBest(node *graph.Node, pctx *state.Context, entries []fanInEntry) (string, string) {
	// Try LLM-based evaluation if prompt is configured and backend is available.
	if node.Prompt() != "" && h.Backend != nil {
		bestID, bestStatus := h.llmSelectBest(node, pctx, entries)
		if bestID != "" {
			return bestID, bestStatus
		}
		// LLM evaluation failed; fall back to heuristic.
	}
	return heuristicSelectBest(entries)
}

// heuristicSelectBest ranks candidates by status (success > fail > other)
// and breaks ties lexically by branch ID.
func heuristicSelectBest(entries []fanInEntry) (string, string) {
	bestBranch := ""
	bestOutcome := ""
	bestRank := 3
	for _, e := range entries {
		rank := 2
		switch strings.ToLower(e.status) {
		case "success":
			rank = 0
		case "fail":
			rank = 1
		}
		if rank < bestRank || (rank == bestRank && (bestBranch == "" || e.id < bestBranch)) {
			bestBranch = e.id
			bestOutcome = e.status
			bestRank = rank
		}
	}
	return bestBranch, bestOutcome
}

// llmSelectBest uses the CodergenBackend to ask the LLM to pick the best
// branch from the candidates. Returns empty strings if the LLM call fails.
func (h *FanInHandler) llmSelectBest(node *graph.Node, pctx *state.Context, entries []fanInEntry) (string, string) {
	// Build a prompt listing the candidates for the LLM to evaluate.
	var sb strings.Builder
	sb.WriteString(node.Prompt())
	sb.WriteString("\n\nCandidates:\n")
	entryMap := make(map[string]string, len(entries))
	for i, e := range entries {
		sb.WriteString(fmt.Sprintf("%d. Branch '%s' (status: %s)\n", i+1, e.id, e.status))
		entryMap[e.id] = e.status
	}
	sb.WriteString("\nRespond with ONLY the branch ID of the best candidate.")

	result, err := h.Backend.Run(node, sb.String(), pctx)
	if err != nil {
		return "", ""
	}

	responseText, ok := result.(string)
	if !ok {
		return "", ""
	}

	// Parse the LLM response: look for a branch ID in the response text.
	trimmed := strings.TrimSpace(responseText)
	for _, e := range entries {
		if strings.Contains(trimmed, e.id) {
			return e.id, e.status
		}
	}

	// If the response exactly matches a branch ID, use it.
	if status, found := entryMap[trimmed]; found {
		return trimmed, status
	}

	return "", ""
}

// ---------------------------------------------------------------------------
// ToolHandler
// ---------------------------------------------------------------------------

// ToolHandler executes an external tool or shell command. It reads
// tool_command from the node's attributes and executes it via the system
// shell. Stdout and stderr are captured and written to the logs directory.
//
// Node attributes:
//   - tool_command: the shell command to execute (required)
//   - tool_timeout: timeout in seconds (optional, default 60)
//   - tool_cwd: working directory (optional)
//
// Graph-level attributes tool_hooks.pre and tool_hooks.post configure shell
// commands that run before and after each tool invocation respectively.
type ToolHandler struct {
	// PreHook is a shell command run before each tool execution.
	// It receives NODE_ID and TOOL_NAME environment variables.
	PreHook string
	// PostHook is a shell command run after each tool execution.
	// It receives NODE_ID, TOOL_NAME, and TOOL_RESULT environment variables.
	PostHook string
}

// Execute runs the tool command and returns success or failure based on
// the exit code.
func (h *ToolHandler) Execute(ctx context.Context, node *graph.Node, _ *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
	command := node.Attrs["tool_command"]
	if command == "" {
		return &state.Outcome{
			Status:        state.StatusFail,
			FailureReason: fmt.Sprintf("Tool node '%s' has no tool_command attribute", node.ID),
		}, nil
	}

	timeoutSec := attrIntDefault(node.Attrs, "tool_timeout", 60)
	cwd := node.Attrs["tool_cwd"]

	stageDir := filepath.Join(logsRoot, node.ID)
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		return nil, fmt.Errorf("tool: create stage dir: %w", err)
	}

	// Resolve tool hooks: struct fields take precedence, then graph-level attributes.
	preHook := h.PreHook
	if preHook == "" && g != nil {
		preHook = g.Attrs["tool_hooks.pre"]
	}
	postHook := h.PostHook
	if postHook == "" && g != nil {
		postHook = g.Attrs["tool_hooks.post"]
	}

	// Run pre-hook if configured. A non-zero exit skips tool execution.
	toolName := node.Attrs["tool_name"]
	if toolName == "" {
		toolName = node.ID
	}
	if preHook != "" {
		if hookErr := runToolHook(preHook, node.ID, toolName, ""); hookErr != nil {
			return &state.Outcome{
				Status:        state.StatusFail,
				FailureReason: fmt.Sprintf("Tool '%s' pre-hook failed: %v", node.ID, hookErr),
				ContextUpdates: map[string]any{
					"last_stage":     node.ID,
					"tool.exit_code": "-1",
				},
			}, nil
		}
	}

	// Build the command with a timeout derived from the parent context.
	timeout := time.Duration(timeoutSec) * time.Second
	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "sh", "-c", command)
	if cwd != "" {
		cmd.Dir = cwd
	}

	// Capture stdout and stderr separately.
	var stdoutBuf, stderrBuf strings.Builder
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err := cmd.Run()
	stdout := stdoutBuf.String()
	stderr := stderrBuf.String()

	// Write captured output to log files (best effort).
	_ = os.WriteFile(filepath.Join(stageDir, "stdout.txt"), []byte(stdout), 0o644)
	_ = os.WriteFile(filepath.Join(stageDir, "stderr.txt"), []byte(stderr), 0o644)

	if err != nil {
		exitCode := -1
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if cmdCtx.Err() != nil {
			// Timeout or context cancellation.
			exitCode = 137
			stderr = fmt.Sprintf("Process timed out after %ds", timeoutSec)
		}

		truncStderr := stderr
		if len(truncStderr) > 200 {
			truncStderr = truncStderr[:200]
		}

		// Run post-hook on failure (best-effort).
		if postHook != "" {
			_ = runToolHook(postHook, node.ID, toolName, truncStderr)
		}

		return &state.Outcome{
			Status:        state.StatusFail,
			FailureReason: fmt.Sprintf("Tool '%s' failed (exit %d): %s", node.ID, exitCode, truncStderr),
			ContextUpdates: map[string]any{
				"last_stage":     node.ID,
				"tool.stderr":    truncStderr,
				"tool.exit_code": strconv.Itoa(exitCode),
			},
		}, nil
	}

	truncStdout := stdout
	if len(truncStdout) > 200 {
		truncStdout = truncStdout[:200] + "..."
	}

	// Run post-hook on success (best-effort).
	if postHook != "" {
		_ = runToolHook(postHook, node.ID, toolName, truncStdout)
	}

	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Tool '%s' completed (exit 0)", node.ID),
		ContextUpdates: map[string]any{
			"last_stage":     node.ID,
			"tool.output":    truncStdout,
			"tool.exit_code": "0",
		},
	}, nil
}

// ---------------------------------------------------------------------------
// StackManagerLoopHandler
// ---------------------------------------------------------------------------

// StackManagerLoopHandler implements the full supervisor pattern from the
// Attractor spec (Section 4.11). It orchestrates sprint-based iteration by
// running a child pipeline in a loop, observing child telemetry, optionally
// steering the child, and evaluating stop conditions between cycles.
//
// Node attributes:
//   - stack.child_dotfile: DOT source for the child pipeline (falls back to node label as description)
//   - manager.poll_interval: duration between observe/steer cycles (default "1s")
//   - manager.max_cycles: maximum number of supervisor cycles (default 10)
//   - manager.stop_condition: condition expression evaluated each cycle; if true, returns success
//   - manager.actions: comma-separated list of actions per cycle (default "observe,wait")
//     Supported actions: "observe", "steer", "wait"
//   - stack.child_autostart: whether to auto-start the child (default "true")
//
// Context keys set by the handler:
//   - stack.child.iteration: current cycle number (1-based)
//   - stack.child.last_status: status of the most recent child pipeline run
//   - stack.child.total_iterations: total cycles completed
//   - stack.child.outcome: "success" or "fail" based on the last child run
//   - context.stack.child.status: "completed" or "failed" based on the last child run
type StackManagerLoopHandler struct {
	// Executor runs child pipelines. When nil, the handler falls back to
	// the simple counter-increment behavior for backward compatibility.
	Executor PipelineExecutor
}

// Execute implements the supervisor observe/steer/wait loop described in
// the spec. If no Executor is configured, it falls back to incrementing
// a loop counter for backward compatibility with tests and simple setups.
func (h *StackManagerLoopHandler) Execute(ctx context.Context, node *graph.Node, pctx *state.Context, g *graph.Graph, logsRoot string) (*state.Outcome, error) {
	// Parse configuration from node attributes.
	childDOT := node.Attrs["stack.child_dotfile"]
	if childDOT == "" {
		// Fall back to graph-level attribute, then node label as description.
		childDOT = g.Attrs["stack.child_dotfile"]
	}

	pollInterval := parseDurationDefault(node.Attrs["manager.poll_interval"], 1*time.Second)
	maxCycles := attrIntDefault(node.Attrs, "manager.max_cycles", 10)
	stopCondition := node.Attrs["manager.stop_condition"]
	actionsRaw := node.Attrs["manager.actions"]
	if actionsRaw == "" {
		actionsRaw = "observe,wait"
	}
	actions := parseActions(actionsRaw)
	autoStart := node.Attrs["stack.child_autostart"] != "false" // default true

	// If no executor is available or no child DOT is configured, fall back
	// to the simple counter-increment behavior. This preserves backward
	// compatibility for callers that do not inject a PipelineExecutor.
	if h.Executor == nil || childDOT == "" {
		return h.fallbackExecute(node, pctx)
	}

	// Create a stage directory for this manager node's logs.
	stageDir := filepath.Join(logsRoot, node.ID)
	_ = os.MkdirAll(stageDir, 0o755)

	// --- Supervisor loop ---
	var lastChildOutcome *state.Outcome
	totalIterations := 0

	for cycle := 1; cycle <= maxCycles; cycle++ {
		// Check for context cancellation.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		totalIterations = cycle

		// Update iteration tracking in context.
		pctx.Set("stack.child.iteration", cycle)
		pctx.Set("stack.child.total_iterations", totalIterations)

		// --- Observe ---
		if actions["observe"] {
			// Ingest child telemetry: read child status keys from context.
			// The child pipeline writes its own status, so observing is
			// simply acknowledging what context already holds.
			childStatus := pctx.GetString("context.stack.child.status", "")
			childOutcomeStr := pctx.GetString("context.stack.child.outcome", "")

			if childStatus == "completed" || childStatus == "failed" {
				if childOutcomeStr == "success" {
					return &state.Outcome{
						Status: state.StatusSuccess,
						Notes:  fmt.Sprintf("Manager '%s': child completed successfully at cycle %d", node.ID, cycle),
						ContextUpdates: map[string]any{
							"stack.child.iteration":        cycle,
							"stack.child.total_iterations": totalIterations,
							"stack.child.last_status":      "success",
							"last_stage":                   node.ID,
						},
					}, nil
				}
				if childStatus == "failed" {
					return &state.Outcome{
						Status:        state.StatusFail,
						FailureReason: fmt.Sprintf("Manager '%s': child failed at cycle %d", node.ID, cycle),
						ContextUpdates: map[string]any{
							"stack.child.iteration":        cycle,
							"stack.child.total_iterations": totalIterations,
							"stack.child.last_status":      "fail",
							"last_stage":                   node.ID,
						},
					}, nil
				}
			}
		}

		// --- Steer ---
		if actions["steer"] {
			// Write steering context that the child can read on its next
			// iteration. The steer action sets a key indicating intervention
			// is requested; specific steer instructions come from node attrs.
			steerInstruction := node.Attrs["manager.steer_instruction"]
			if steerInstruction != "" {
				pctx.Set("stack.child.steer_instruction", steerInstruction)
			}
			pctx.Set("stack.child.steer_cycle", cycle)
		}

		// --- Execute child pipeline ---
		if autoStart || cycle > 1 {
			childLogsDir := filepath.Join(stageDir, fmt.Sprintf("child-%03d", cycle))

			// Build the initial context for the child from the parent's
			// current snapshot, scoped to keys the child should see.
			childInitCtx := buildChildContext(pctx, cycle)

			childOutcome, err := h.Executor.ExecuteDOT(ctx, childDOT, childLogsDir, childInitCtx)
			if err != nil {
				// ExecuteDOT error is a system failure, not a child pipeline
				// status=fail. Record it and continue to next cycle or bail.
				pctx.Set("context.stack.child.status", "failed")
				pctx.Set("context.stack.child.outcome", "fail")
				pctx.Set("stack.child.last_status", "fail")
				lastChildOutcome = &state.Outcome{
					Status:        state.StatusFail,
					FailureReason: fmt.Sprintf("child pipeline error: %v", err),
				}
			} else {
				lastChildOutcome = childOutcome

				// Update context with child results.
				childStatusStr := strings.ToLower(string(childOutcome.Status))
				pctx.Set("stack.child.last_status", childStatusStr)
				pctx.Set("stack.child.outcome", childStatusStr)

				if childOutcome.Status == state.StatusSuccess {
					pctx.Set("context.stack.child.status", "completed")
					pctx.Set("context.stack.child.outcome", "success")
				} else if childOutcome.Status == state.StatusFail {
					pctx.Set("context.stack.child.status", "failed")
					pctx.Set("context.stack.child.outcome", "fail")
				} else {
					pctx.Set("context.stack.child.status", "running")
					pctx.Set("context.stack.child.outcome", childStatusStr)
				}
			}
		}

		// --- Evaluate stop condition ---
		if stopCondition != "" {
			// Use the condition evaluator to check the stop expression
			// against the current context. We pass a nil outcome since the
			// stop condition operates on context keys, not a handler outcome.
			if condition.Evaluate(stopCondition, nil, pctx) {
				return &state.Outcome{
					Status: state.StatusSuccess,
					Notes:  fmt.Sprintf("Manager '%s': stop condition satisfied at cycle %d", node.ID, cycle),
					ContextUpdates: map[string]any{
						"stack.child.iteration":        cycle,
						"stack.child.total_iterations": totalIterations,
						"stack.child.last_status":      pctx.GetString("stack.child.last_status", ""),
						"last_stage":                   node.ID,
					},
				}, nil
			}
		}

		// Check if the child completed or failed (post-execution observe).
		childStatus := pctx.GetString("context.stack.child.status", "")
		if childStatus == "completed" {
			childOutcomeStr := pctx.GetString("context.stack.child.outcome", "")
			if childOutcomeStr == "success" {
				return &state.Outcome{
					Status: state.StatusSuccess,
					Notes:  fmt.Sprintf("Manager '%s': child completed successfully at cycle %d", node.ID, cycle),
					ContextUpdates: map[string]any{
						"stack.child.iteration":        cycle,
						"stack.child.total_iterations": totalIterations,
						"stack.child.last_status":      "success",
						"last_stage":                   node.ID,
					},
				}, nil
			}
		}
		if childStatus == "failed" {
			return &state.Outcome{
				Status:        state.StatusFail,
				FailureReason: fmt.Sprintf("Manager '%s': child failed at cycle %d", node.ID, cycle),
				ContextUpdates: map[string]any{
					"stack.child.iteration":        cycle,
					"stack.child.total_iterations": totalIterations,
					"stack.child.last_status":      "fail",
					"last_stage":                   node.ID,
				},
			}, nil
		}

		// --- Wait ---
		if actions["wait"] {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(pollInterval):
			}
		}
	}

	// Max cycles exceeded.
	failReason := fmt.Sprintf("Manager '%s': max cycles (%d) exceeded", node.ID, maxCycles)
	lastStatus := "fail"
	if lastChildOutcome != nil {
		lastStatus = strings.ToLower(string(lastChildOutcome.Status))
	}

	return &state.Outcome{
		Status:        state.StatusFail,
		FailureReason: failReason,
		ContextUpdates: map[string]any{
			"stack.child.iteration":        totalIterations,
			"stack.child.total_iterations": totalIterations,
			"stack.child.last_status":      lastStatus,
			"last_stage":                   node.ID,
		},
	}, nil
}

// fallbackExecute provides the simple counter-increment behavior used when
// no PipelineExecutor is available. This preserves backward compatibility.
func (h *StackManagerLoopHandler) fallbackExecute(node *graph.Node, pctx *state.Context) (*state.Outcome, error) {
	counterKey := fmt.Sprintf("loop.%s.iteration", node.ID)
	current := pctx.GetInt(counterKey, 0)
	next := current + 1

	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Stack manager loop '%s' iteration %d", node.ID, next),
		ContextUpdates: map[string]any{
			counterKey:   next,
			"last_stage": node.ID,
		},
	}, nil
}

// parseActions splits a comma-separated action list into a lookup set.
func parseActions(raw string) map[string]bool {
	actions := make(map[string]bool)
	for _, a := range strings.Split(raw, ",") {
		a = strings.TrimSpace(strings.ToLower(a))
		if a != "" {
			actions[a] = true
		}
	}
	return actions
}

// parseDurationDefault parses a duration string with a fallback default.
func parseDurationDefault(raw string, def time.Duration) time.Duration {
	if raw == "" {
		return def
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return def
	}
	return d
}

// buildChildContext creates an initial context map for the child pipeline
// from the parent's current state. It passes through goal, steering
// instructions, and iteration metadata.
func buildChildContext(pctx *state.Context, cycle int) map[string]any {
	childCtx := make(map[string]any)

	// Pass through the goal if set.
	if goal := pctx.GetString("goal", ""); goal != "" {
		childCtx["goal"] = goal
	}

	// Pass through steering instructions.
	if steer := pctx.GetString("stack.child.steer_instruction", ""); steer != "" {
		childCtx["manager.steer_instruction"] = steer
	}

	// Pass iteration metadata.
	childCtx["manager.parent_cycle"] = cycle

	return childCtx
}

// ---------------------------------------------------------------------------
// CommunicationHandler
// ---------------------------------------------------------------------------

// CommunicationHandler is a pass-through for inter-pipeline communication
// nodes (shape=doubleoctagon). It returns Success immediately and records
// the communication direction (inbound/outbound) in context updates so
// downstream logic or the factory pipeline can act on it.
type CommunicationHandler struct{}

// Execute passes through and records direction metadata.
func (h *CommunicationHandler) Execute(_ context.Context, node *graph.Node, _ *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	direction := node.Attrs["direction"]
	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Communication node '%s' (%s) passed through", node.ID, direction),
		ContextUpdates: map[string]any{
			"last_stage":              node.ID,
			"communication.direction": direction,
		},
	}, nil
}

// ---------------------------------------------------------------------------
// NoopHandler
// ---------------------------------------------------------------------------

// NoopHandler returns success with a descriptive note. It is used as the
// default handler when no specific handler is registered for a node type.
type NoopHandler struct{}

// Execute returns success without performing any work.
func (NoopHandler) Execute(_ context.Context, node *graph.Node, _ *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Node '%s' (type=%s) executed by noop handler", node.ID, strings.TrimSpace(node.Type())),
	}, nil
}

// ---------------------------------------------------------------------------
// DefaultRegistry (full)
// ---------------------------------------------------------------------------

// DefaultRegistryFull creates a Registry with all built-in handlers registered.
// The backend parameter provides the LLM backend for CodergenHandler (nil
// for simulation mode). The interv parameter provides the interviewer for
// WaitForHumanHandler (nil falls back to AutoApproveInterviewer). The
// executor parameter provides child pipeline execution for the manager loop
// handler (nil falls back to simple counter-increment behavior).
func DefaultRegistryFull(backend CodergenBackend, interv interviewer.Interviewer, executor PipelineExecutor) *Registry {
	if interv == nil {
		interv = &interviewer.AutoApproveInterviewer{}
	}

	r := NewRegistry(NoopHandler{})

	r.Register("start", &StartHandler{})
	r.Register("exit", &ExitHandler{})
	r.Register("codergen", &CodergenHandler{Backend: backend})
	r.Register("wait.human", &WaitForHumanHandler{Interviewer: interv})
	r.Register("conditional", &ConditionalHandler{})
	r.Register("parallel", NewParallelHandler(r))
	r.Register("parallel.fan_in", &FanInHandler{Backend: backend})
	r.Register("tool", &ToolHandler{})
	r.Register("stack.manager_loop", &StackManagerLoopHandler{Executor: executor})
	r.Register("communication", &CommunicationHandler{})

	return r
}

// DefaultRegistry builds a Registry with the built-in start and exit handlers
// registered. The caller provides the default handler used for all other nodes.
// For a fully-featured registry with all handlers, use DefaultRegistryFull.
func DefaultRegistry(defaultHandler Handler) *Registry {
	r := NewRegistry(defaultHandler)
	r.Register("start", &StartHandler{})
	r.Register("exit", &ExitHandler{})
	return r
}

// ---------------------------------------------------------------------------
// Tool hook helper
// ---------------------------------------------------------------------------

// runToolHook executes a shell command as a tool hook, passing NODE_ID,
// TOOL_NAME, and optionally TOOL_RESULT as environment variables. Returns
// an error if the hook command exits with a non-zero status.
func runToolHook(hookCmd, nodeID, toolName, result string) error {
	if hookCmd == "" {
		return nil
	}
	cmd := exec.Command("sh", "-c", hookCmd)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("NODE_ID=%s", nodeID),
		fmt.Sprintf("TOOL_NAME=%s", toolName),
	)
	if result != "" {
		cmd.Env = append(cmd.Env, fmt.Sprintf("TOOL_RESULT=%s", result))
	}
	return cmd.Run()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// attrIntDefault reads an integer attribute with a fallback default.
func attrIntDefault(attrs map[string]string, key string, def int) int {
	raw, ok := attrs[key]
	if !ok || raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return def
	}
	return v
}
