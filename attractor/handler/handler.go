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
	errorPolicy := parseErrorPolicy(node.Attrs["error_policy"])
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
		} else {
			switch errorPolicy {
			case ErrorIgnore, ErrorContinue:
				aggregateStatus = state.StatusSuccess
			default:
				aggregateStatus = state.StatusFail
			}
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
// and stores the winning branch ID in "parallel.best_branch".
type FanInHandler struct{}

// Execute reads parallel results and selects the best branch.
func (h *FanInHandler) Execute(_ context.Context, node *graph.Node, pctx *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
	resultsRaw := pctx.GetString("parallel.results", "")
	if resultsRaw == "" {
		return &state.Outcome{
			Status: state.StatusSuccess,
			Notes:  fmt.Sprintf("Fan-in '%s': no parallel results found", node.ID),
			ContextUpdates: map[string]any{
				"parallel.best_branch": "",
			},
		}, nil
	}

	// Parse entries: "nodeId:status,nodeId:status,..."
	type entry struct {
		id     string
		status string
	}
	var entries []entry
	for _, part := range strings.Split(resultsRaw, ",") {
		parts := strings.SplitN(part, ":", 2)
		if len(parts) == 2 {
			entries = append(entries, entry{id: parts[0], status: parts[1]})
		}
	}

	// Rank by status: success=0, fail=1, other=2; break ties lexically by ID.
	bestBranch := ""
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
			bestRank = rank
		}
	}

	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Fan-in '%s': selected best branch '%s' from %d branches", node.ID, bestBranch, len(entries)),
		ContextUpdates: map[string]any{
			"parallel.best_branch": bestBranch,
			"last_stage":           node.ID,
		},
	}, nil
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
type ToolHandler struct{}

// Execute runs the tool command and returns success or failure based on
// the exit code.
func (h *ToolHandler) Execute(ctx context.Context, node *graph.Node, _ *state.Context, _ *graph.Graph, logsRoot string) (*state.Outcome, error) {
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
	return &state.Outcome{
		Status: state.StatusSuccess,
		Notes:  fmt.Sprintf("Tool '%s' completed (exit 0)", node.ID),
		ContextUpdates: map[string]any{
			"last_stage":     node.ID,
			"tool.stdout":    truncStdout,
			"tool.exit_code": "0",
		},
	}, nil
}

// ---------------------------------------------------------------------------
// StackManagerLoopHandler
// ---------------------------------------------------------------------------

// StackManagerLoopHandler manages recursive stack-based execution loops.
// It tracks loop iterations via context and delegates to outgoing edges
// for the actual loop body. This is a pass-through that increments the
// loop counter; the engine handles the actual iteration via loop_restart
// edges.
type StackManagerLoopHandler struct{}

// Execute increments the loop counter and returns success.
func (h *StackManagerLoopHandler) Execute(_ context.Context, node *graph.Node, pctx *state.Context, _ *graph.Graph, _ string) (*state.Outcome, error) {
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
// WaitForHumanHandler (nil falls back to AutoApproveInterviewer).
func DefaultRegistryFull(backend CodergenBackend, interv interviewer.Interviewer) *Registry {
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
	r.Register("parallel.fan_in", &FanInHandler{})
	r.Register("tool", &ToolHandler{})
	r.Register("stack.manager_loop", &StackManagerLoopHandler{})
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
