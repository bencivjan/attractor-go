// Package session implements the core agentic loop for the coding agent. A
// Session pairs an LLM client with an execution environment and iterates
// through the think-act cycle: send context to the model, execute any tool
// calls it produces, feed results back, and repeat until the model produces
// a natural-language completion or a limit is reached.
//
// The loop faithfully implements the specification from Section 2.5 of the
// attractor design:
//
//  1. Check limits (max rounds, max turns, abort signal).
//  2. Build LLM request (system prompt + conversation history + tools).
//  3. Call the LLM.
//  4. Record the assistant turn.
//  5. If no tool calls, the loop ends (natural completion).
//  6. Execute tool calls through the execution environment.
//  7. Drain steering messages.
//  8. Loop detection.
//  9. Go to 1.
package session

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/strongdm/attractor-go/codingagent/env"
	"github.com/strongdm/attractor-go/codingagent/events"
	"github.com/strongdm/attractor-go/codingagent/profile"
	"github.com/strongdm/attractor-go/codingagent/truncation"
	"github.com/strongdm/attractor-go/unifiedllm/types"
)

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

// SessionState represents the lifecycle state of a Session.
type SessionState string

const (
	// StateIdle means the session is ready to accept input.
	StateIdle SessionState = "idle"
	// StateProcessing means the session is executing the agentic loop.
	StateProcessing SessionState = "processing"
	// StateAwaitingInput means the session completed a loop iteration and is
	// waiting for the next user message.
	StateAwaitingInput SessionState = "awaiting_input"
	// StateClosed means the session has been terminated.
	StateClosed SessionState = "closed"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Config holds tuning knobs for session behavior.
type Config struct {
	// MaxTurns limits the total number of conversation turns (user + assistant
	// + tool). 0 means unlimited.
	MaxTurns int
	// MaxToolRoundsPerInput limits how many tool-call/result round-trips are
	// allowed per single user input. 0 means unlimited.
	MaxToolRoundsPerInput int
	// DefaultCommandTimeoutMs is the default timeout for shell commands.
	DefaultCommandTimeoutMs int
	// MaxCommandTimeoutMs is the hard upper bound for shell command timeouts.
	MaxCommandTimeoutMs int
	// ReasoningEffort controls the model's reasoning depth ("low", "medium",
	// "high", or "" for provider default).
	ReasoningEffort string
	// ToolOutputLimits overrides default character limits for tool output
	// truncation, keyed by tool name.
	ToolOutputLimits map[string]int
	// EnableLoopDetection enables detection of repeating tool call patterns.
	EnableLoopDetection bool
	// LoopDetectionWindow is the number of recent tool rounds to examine for
	// repeating patterns.
	LoopDetectionWindow int
	// MaxSubagentDepth limits how deeply sub-agents can be nested.
	MaxSubagentDepth int
}

// DefaultConfig returns a Config with sensible production defaults.
func DefaultConfig() Config {
	return Config{
		MaxTurns:                0,
		MaxToolRoundsPerInput:   0,
		DefaultCommandTimeoutMs: 10000,
		MaxCommandTimeoutMs:     600000,
		ReasoningEffort:         "",
		ToolOutputLimits:        nil,
		EnableLoopDetection:     true,
		LoopDetectionWindow:     10,
		MaxSubagentDepth:        1,
	}
}

// ---------------------------------------------------------------------------
// Turn types
// ---------------------------------------------------------------------------

// Turn is a single entry in the conversation history. Each concrete turn
// type records its kind and timestamp.
type Turn interface {
	TurnType() string
	TurnTimestamp() time.Time
}

// UserTurn records user-supplied input.
type UserTurn struct {
	Content   string
	Timestamp time.Time
}

func (t *UserTurn) TurnType() string       { return "user" }
func (t *UserTurn) TurnTimestamp() time.Time { return t.Timestamp }

// AssistantTurn records the model's response including any tool calls,
// reasoning traces, and token usage.
type AssistantTurn struct {
	Content    string
	ToolCalls  []types.ToolCall
	Reasoning  string
	Usage      types.Usage
	ResponseID string
	Timestamp  time.Time
}

func (t *AssistantTurn) TurnType() string       { return "assistant" }
func (t *AssistantTurn) TurnTimestamp() time.Time { return t.Timestamp }

// ToolResultsTurn records the results of one or more tool executions.
type ToolResultsTurn struct {
	Results   []types.ToolResult
	Timestamp time.Time
}

func (t *ToolResultsTurn) TurnType() string       { return "tool_results" }
func (t *ToolResultsTurn) TurnTimestamp() time.Time { return t.Timestamp }

// SystemTurn records a system message injected into the conversation.
type SystemTurn struct {
	Content   string
	Timestamp time.Time
}

func (t *SystemTurn) TurnType() string       { return "system" }
func (t *SystemTurn) TurnTimestamp() time.Time { return t.Timestamp }

// SteeringTurn records a steering message injected between tool rounds.
type SteeringTurn struct {
	Content   string
	Timestamp time.Time
}

func (t *SteeringTurn) TurnType() string       { return "steering" }
func (t *SteeringTurn) TurnTimestamp() time.Time { return t.Timestamp }

// ---------------------------------------------------------------------------
// LLMClient interface
// ---------------------------------------------------------------------------

// LLMClient abstracts the unified LLM SDK so that the session layer is
// decoupled from any specific provider implementation.
type LLMClient interface {
	// Complete sends a synchronous chat-completion request and returns the
	// full response.
	Complete(ctx context.Context, req types.Request) (*types.Response, error)
	// Stream sends a streaming request and returns a channel of events.
	Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error)
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

// Session is the central orchestrator of the coding agent. It manages
// conversation history, drives the agentic loop, and coordinates between
// the LLM and the execution environment.
type Session struct {
	// ID uniquely identifies this session.
	ID string
	// Profile configures the model, tools, and system prompt.
	Profile profile.ProviderProfile
	// ExecutionEnv is where tool calls are executed.
	ExecutionEnv env.ExecutionEnvironment
	// History is the ordered list of conversation turns.
	History []Turn
	// EventEmitter dispatches session events to listeners.
	EventEmitter *events.Emitter
	// Config holds session behavior parameters.
	Config Config
	// State is the current lifecycle state.
	State SessionState
	// LLMClient is the interface to the unified LLM SDK.
	LLMClient LLMClient

	steeringQueue []string
	followupQueue []string
	mu            sync.Mutex
	abortCh       chan struct{}
	depth         int // subagent nesting depth
}

// New creates a Session with the given profile, environment, client, and
// config. It initializes internal state and emits a session_start event.
func New(prof profile.ProviderProfile, execEnv env.ExecutionEnvironment, client LLMClient, cfg Config) *Session {
	s := &Session{
		ID:           generateSessionID(),
		Profile:      prof,
		ExecutionEnv: execEnv,
		History:      nil,
		EventEmitter: events.NewEmitter(),
		Config:       cfg,
		State:        StateIdle,
		LLMClient:    client,
		abortCh:      make(chan struct{}),
	}

	s.EventEmitter.Emit(events.EventSessionStart, s.ID, map[string]any{
		"model":   prof.Model(),
		"profile": prof.ID(),
	})

	return s
}

// Submit processes user input through the agentic loop. It is the primary
// entry point for driving the agent. Submit blocks until the loop completes,
// is aborted, or hits a limit.
func (s *Session) Submit(ctx context.Context, input string) error {
	s.mu.Lock()
	if s.State == StateClosed {
		s.mu.Unlock()
		return fmt.Errorf("session is closed")
	}
	s.State = StateProcessing
	s.mu.Unlock()

	s.EventEmitter.Emit(events.EventUserInput, s.ID, map[string]any{
		"input": input,
	})

	err := s.processInput(ctx, input)

	s.mu.Lock()
	if s.State != StateClosed {
		s.State = StateAwaitingInput
	}
	// Drain follow-up queue.
	var followups []string
	followups = append(followups, s.followupQueue...)
	s.followupQueue = nil
	s.mu.Unlock()

	// Process any queued follow-up messages.
	for _, fu := range followups {
		if fuErr := s.Submit(ctx, fu); fuErr != nil {
			return fuErr
		}
	}

	return err
}

// Steer queues a message to be injected into the conversation after the
// current tool round completes. This is used for mid-loop course correction.
func (s *Session) Steer(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steeringQueue = append(s.steeringQueue, message)
}

// FollowUp queues a message to be processed after the current Submit call
// completes.
func (s *Session) FollowUp(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.followupQueue = append(s.followupQueue, message)
}

// Abort signals the session to stop processing at the next safe point.
func (s *Session) Abort() {
	s.mu.Lock()
	defer s.mu.Unlock()
	select {
	case <-s.abortCh:
		// Already aborted.
	default:
		close(s.abortCh)
	}
}

// Close terminates the session and emits a session_end event. After Close,
// no further Submit calls are accepted.
func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.State == StateClosed {
		return
	}
	s.State = StateClosed
	s.EventEmitter.Emit(events.EventSessionEnd, s.ID, nil)
}

// ---------------------------------------------------------------------------
// Core agentic loop
// ---------------------------------------------------------------------------

// processInput is the heart of the coding agent. It implements the loop
// described in Section 2.5:
//
//  1. Check limits and abort signal.
//  2. Build the LLM request.
//  3. Call the LLM.
//  4. Record the assistant turn.
//  5. If no tool calls -> break (natural completion).
//  6. Execute tool calls.
//  7. Drain steering messages.
//  8. Loop detection.
//  9. Repeat.
func (s *Session) processInput(ctx context.Context, input string) error {
	// Record the user turn.
	s.History = append(s.History, &UserTurn{
		Content:   input,
		Timestamp: time.Now(),
	})

	toolRound := 0

	for {
		// ---- Step 1: Check limits ----

		// Check abort signal.
		select {
		case <-s.abortCh:
			s.EventEmitter.Emit(events.EventWarning, s.ID, map[string]any{
				"message": "session aborted",
			})
			return fmt.Errorf("session aborted")
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Check max turns.
		if s.Config.MaxTurns > 0 && s.countTurns() >= s.Config.MaxTurns {
			s.EventEmitter.Emit(events.EventTurnLimit, s.ID, map[string]any{
				"limit":  s.Config.MaxTurns,
				"reason": "max_turns",
			})
			return fmt.Errorf("maximum turn limit (%d) reached", s.Config.MaxTurns)
		}

		// Check max tool rounds per input.
		if s.Config.MaxToolRoundsPerInput > 0 && toolRound >= s.Config.MaxToolRoundsPerInput {
			s.EventEmitter.Emit(events.EventTurnLimit, s.ID, map[string]any{
				"limit":  s.Config.MaxToolRoundsPerInput,
				"reason": "max_tool_rounds",
			})
			return fmt.Errorf("maximum tool rounds per input (%d) reached", s.Config.MaxToolRoundsPerInput)
		}

		// ---- Step 2: Build LLM request ----
		req := s.buildRequest()

		// ---- Step 3: Call LLM ----
		s.EventEmitter.Emit(events.EventAssistantTextStart, s.ID, map[string]any{
			"tool_round": toolRound,
		})

		resp, err := s.LLMClient.Complete(ctx, req)
		if err != nil {
			s.EventEmitter.Emit(events.EventError, s.ID, map[string]any{
				"error":  err.Error(),
				"source": "llm_complete",
			})
			return fmt.Errorf("LLM complete: %w", err)
		}

		// ---- Step 4: Record assistant turn ----
		assistantText := resp.Text()
		toolCalls := extractToolCalls(resp)
		reasoning := resp.Reasoning()

		assistantTurn := &AssistantTurn{
			Content:    assistantText,
			ToolCalls:  toolCalls,
			Reasoning:  reasoning,
			Usage:      resp.Usage,
			ResponseID: resp.ID,
			Timestamp:  time.Now(),
		}
		s.History = append(s.History, assistantTurn)

		if assistantText != "" {
			s.EventEmitter.Emit(events.EventAssistantTextDelta, s.ID, map[string]any{
				"text": assistantText,
			})
		}

		s.EventEmitter.Emit(events.EventAssistantTextEnd, s.ID, map[string]any{
			"usage":         resp.Usage,
			"finish_reason": resp.FinishReason.Reason,
			"has_tools":     len(toolCalls) > 0,
		})

		// ---- Step 5: No tool calls -> natural completion ----
		if len(toolCalls) == 0 {
			return nil
		}

		// ---- Step 6: Execute tool calls ----
		results := s.executeToolCalls(ctx, toolCalls)
		s.History = append(s.History, &ToolResultsTurn{
			Results:   results,
			Timestamp: time.Now(),
		})

		toolRound++

		// ---- Step 7: Drain steering messages ----
		s.drainSteering()

		// ---- Step 8: Loop detection ----
		if s.Config.EnableLoopDetection && s.detectLoop() {
			s.EventEmitter.Emit(events.EventLoopDetection, s.ID, map[string]any{
				"tool_round": toolRound,
				"message":    "repeating tool call pattern detected",
			})
			// Inject a steering message to break the loop.
			s.History = append(s.History, &SteeringTurn{
				Content: "It appears you are repeating the same tool calls. " +
					"Please reassess your approach and try a different strategy, " +
					"or explain what you are trying to accomplish.",
				Timestamp: time.Now(),
			})
		}

		// ---- Step 9: Continue loop ----
	}
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

// buildRequest assembles the types.Request from the current session state.
func (s *Session) buildRequest() types.Request {
	// System prompt.
	workDir := s.ExecutionEnv.WorkingDirectory()
	systemPrompt := s.Profile.BuildSystemPrompt(workDir, "")

	messages := s.convertHistoryToMessages()

	// Prepend system message.
	allMessages := make([]types.Message, 0, 1+len(messages))
	allMessages = append(allMessages, types.SystemMessage(systemPrompt))
	allMessages = append(allMessages, messages...)

	// Build tool definitions for the request.
	toolDefs := s.convertToolDefinitions()

	req := types.Request{
		Model:           s.Profile.Model(),
		Messages:        allMessages,
		Tools:           toolDefs,
		ReasoningEffort: s.Config.ReasoningEffort,
		ProviderOptions: s.Profile.ProviderOptions(),
	}

	return req
}

// convertToolDefinitions converts the profile's tool definitions to the
// unified LLM types.
func (s *Session) convertToolDefinitions() []types.ToolDefinition {
	defs := s.Profile.Tools()
	result := make([]types.ToolDefinition, len(defs))
	for i, d := range defs {
		result[i] = types.ToolDefinition{
			Name:        d.Name,
			Description: d.Description,
			Parameters:  d.Parameters,
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// History conversion
// ---------------------------------------------------------------------------

// convertHistoryToMessages translates the Turn-based history into the flat
// types.Message list expected by the LLM API.
func (s *Session) convertHistoryToMessages() []types.Message {
	var msgs []types.Message

	for _, turn := range s.History {
		switch t := turn.(type) {
		case *UserTurn:
			msgs = append(msgs, types.UserMessage(t.Content))

		case *AssistantTurn:
			msg := types.Message{
				Role:    types.RoleAssistant,
				Content: buildAssistantContent(t),
			}
			msgs = append(msgs, msg)

		case *ToolResultsTurn:
			for _, r := range t.Results {
				content, isError := toolResultContent(r)
				msgs = append(msgs, types.ToolResultMessage(r.ToolCallID, content, isError))
			}

		case *SystemTurn:
			msgs = append(msgs, types.SystemMessage(t.Content))

		case *SteeringTurn:
			// Steering messages are injected as user messages to guide the
			// model without appearing as system instructions.
			msgs = append(msgs, types.UserMessage(t.Content))
		}
	}

	return msgs
}

// buildAssistantContent constructs the ContentPart slice for an assistant
// turn, including text, reasoning, and tool calls.
func buildAssistantContent(t *AssistantTurn) []types.ContentPart {
	var parts []types.ContentPart

	// Include reasoning/thinking if present.
	if t.Reasoning != "" {
		parts = append(parts, types.ContentPart{
			Kind: types.ContentKindThinking,
			Thinking: &types.ThinkingData{
				Text: t.Reasoning,
			},
		})
	}

	// Include text content if present.
	if t.Content != "" {
		parts = append(parts, types.ContentPart{
			Kind: types.ContentKindText,
			Text: t.Content,
		})
	}

	// Include tool calls.
	for _, tc := range t.ToolCalls {
		parts = append(parts, types.ContentPart{
			Kind: types.ContentKindToolCall,
			ToolCall: &types.ToolCallData{
				ID:        tc.ID,
				Name:      tc.Name,
				Arguments: tc.Arguments,
			},
		})
	}

	// Ensure at least one part exists.
	if len(parts) == 0 {
		parts = append(parts, types.ContentPart{
			Kind: types.ContentKindText,
			Text: "",
		})
	}

	return parts
}

// toolResultContent extracts the string content and error flag from a
// ToolResult.
func toolResultContent(r types.ToolResult) (string, bool) {
	switch v := r.Content.(type) {
	case string:
		return v, r.IsError
	default:
		return fmt.Sprintf("%v", r.Content), r.IsError
	}
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

// executeToolCalls runs all tool calls from a single assistant turn. Tool
// calls are executed sequentially to maintain deterministic ordering and
// prevent resource contention.
func (s *Session) executeToolCalls(ctx context.Context, toolCalls []types.ToolCall) []types.ToolResult {
	results := make([]types.ToolResult, 0, len(toolCalls))
	for _, tc := range toolCalls {
		// Check for abort between tool calls.
		select {
		case <-s.abortCh:
			results = append(results, types.ToolResult{
				ToolCallID: tc.ID,
				Content:    "Tool execution aborted by user.",
				IsError:    true,
			})
			return results
		case <-ctx.Done():
			results = append(results, types.ToolResult{
				ToolCallID: tc.ID,
				Content:    "Tool execution cancelled: " + ctx.Err().Error(),
				IsError:    true,
			})
			return results
		default:
		}

		result := s.executeSingleTool(ctx, tc)
		results = append(results, result)
	}
	return results
}

// executeSingleTool dispatches a single tool call to the registry and
// environment. It handles unknown tools, execution errors, and output
// truncation.
func (s *Session) executeSingleTool(ctx context.Context, tc types.ToolCall) types.ToolResult {
	s.EventEmitter.Emit(events.EventToolCallStart, s.ID, map[string]any{
		"tool_call_id": tc.ID,
		"tool_name":    tc.Name,
		"arguments":    tc.Arguments,
	})

	startTime := time.Now()

	registry := s.Profile.ToolRegistry()
	registered := registry.Get(tc.Name)

	var output string
	var isError bool

	if registered == nil {
		output = fmt.Sprintf("Unknown tool: %q. Available tools: %s", tc.Name, strings.Join(registry.Names(), ", "))
		isError = true
	} else if registered.Executor == nil {
		// Tool is registered but has no executor bound. Fall back to
		// built-in execution based on tool name.
		var err error
		output, err = s.executeBuiltinTool(ctx, tc.Name, tc.Arguments)
		if err != nil {
			output = fmt.Sprintf("Tool %q error: %s", tc.Name, err.Error())
			isError = true
		}
	} else {
		var err error
		output, err = registered.Executor(tc.Arguments, s.ExecutionEnv)
		if err != nil {
			output = fmt.Sprintf("Tool %q error: %s", tc.Name, err.Error())
			isError = true
		}
	}

	// Apply truncation.
	output = truncation.TruncateToolOutput(output, tc.Name, s.Config.ToolOutputLimits)

	durationMs := time.Since(startTime).Milliseconds()

	s.EventEmitter.Emit(events.EventToolCallEnd, s.ID, map[string]any{
		"tool_call_id": tc.ID,
		"tool_name":    tc.Name,
		"duration_ms":  durationMs,
		"is_error":     isError,
		"output_len":   len(output),
	})

	return types.ToolResult{
		ToolCallID: tc.ID,
		Content:    output,
		IsError:    isError,
	}
}

// executeBuiltinTool provides default implementations for the standard tool
// set by dispatching to the ExecutionEnvironment. This is used when a tool
// is registered with a Definition but no Executor (the common case for
// standard tools).
func (s *Session) executeBuiltinTool(ctx context.Context, name string, args map[string]any) (string, error) {
	switch name {
	case "read_file":
		filePath, _ := args["file_path"].(string)
		if filePath == "" {
			return "", fmt.Errorf("file_path is required")
		}
		offset, _ := toInt(args["offset"])
		limit, _ := toInt(args["limit"])
		return s.ExecutionEnv.ReadFile(filePath, offset, limit)

	case "write_file":
		filePath, _ := args["file_path"].(string)
		if filePath == "" {
			return "", fmt.Errorf("file_path is required")
		}
		content, _ := args["content"].(string)
		err := s.ExecutionEnv.WriteFile(filePath, content)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), filePath), nil

	case "edit_file":
		return s.executeEditFile(args)

	case "shell":
		command, _ := args["command"].(string)
		if command == "" {
			return "", fmt.Errorf("command is required")
		}
		timeoutMs, _ := toInt(args["timeout_ms"])
		if timeoutMs <= 0 {
			timeoutMs = s.Config.DefaultCommandTimeoutMs
		}
		if s.Config.MaxCommandTimeoutMs > 0 && timeoutMs > s.Config.MaxCommandTimeoutMs {
			timeoutMs = s.Config.MaxCommandTimeoutMs
		}
		workDir, _ := args["working_dir"].(string)
		result, err := s.ExecutionEnv.ExecCommand(ctx, command, timeoutMs, workDir, nil)
		if err != nil {
			return "", err
		}
		return formatExecResult(result), nil

	case "grep":
		pattern, _ := args["pattern"].(string)
		if pattern == "" {
			return "", fmt.Errorf("pattern is required")
		}
		path, _ := args["path"].(string)
		if path == "" {
			path = s.ExecutionEnv.WorkingDirectory()
		}
		caseInsensitive, _ := args["case_insensitive"].(bool)
		maxResults, _ := toInt(args["max_results"])
		globFilter, _ := args["glob_filter"].(string)
		return s.ExecutionEnv.Grep(pattern, path, caseInsensitive, maxResults, globFilter)

	case "glob":
		pattern, _ := args["pattern"].(string)
		if pattern == "" {
			return "", fmt.Errorf("pattern is required")
		}
		path, _ := args["path"].(string)
		matches, err := s.ExecutionEnv.Glob(pattern, path)
		if err != nil {
			return "", err
		}
		if len(matches) == 0 {
			return "No files matched the pattern.", nil
		}
		return strings.Join(matches, "\n"), nil

	case "list_directory":
		path, _ := args["path"].(string)
		if path == "" {
			return "", fmt.Errorf("path is required")
		}
		depth, _ := toInt(args["depth"])
		entries, err := s.ExecutionEnv.ListDirectory(path, depth)
		if err != nil {
			return "", err
		}
		return formatDirEntries(entries), nil

	default:
		return "", fmt.Errorf("no built-in handler for tool %q", name)
	}
}

// executeEditFile implements the edit_file tool: exact string replacement.
func (s *Session) executeEditFile(args map[string]any) (string, error) {
	filePath, _ := args["file_path"].(string)
	if filePath == "" {
		return "", fmt.Errorf("file_path is required")
	}
	oldString, _ := args["old_string"].(string)
	newString, _ := args["new_string"].(string)
	replaceAll, _ := args["replace_all"].(bool)

	// Read current content (raw, without line numbers).
	content, err := readRawFile(filePath)
	if err != nil {
		return "", fmt.Errorf("reading file for edit: %w", err)
	}

	if oldString == newString {
		return "", fmt.Errorf("old_string and new_string are identical")
	}

	count := strings.Count(content, oldString)
	if count == 0 {
		return "", fmt.Errorf("old_string not found in %s", filePath)
	}
	if count > 1 && !replaceAll {
		return "", fmt.Errorf("old_string found %d times in %s; use replace_all to replace all occurrences", count, filePath)
	}

	var newContent string
	if replaceAll {
		newContent = strings.ReplaceAll(content, oldString, newString)
	} else {
		newContent = strings.Replace(content, oldString, newString, 1)
	}

	if err := s.ExecutionEnv.WriteFile(filePath, newContent); err != nil {
		return "", fmt.Errorf("writing edited file: %w", err)
	}

	replacements := 1
	if replaceAll {
		replacements = count
	}
	return fmt.Sprintf("Successfully edited %s (%d replacement(s))", filePath, replacements), nil
}

// ---------------------------------------------------------------------------
// Steering
// ---------------------------------------------------------------------------

// drainSteering moves all queued steering messages into the conversation
// history.
func (s *Session) drainSteering() {
	s.mu.Lock()
	queue := s.steeringQueue
	s.steeringQueue = nil
	s.mu.Unlock()

	for _, msg := range queue {
		s.History = append(s.History, &SteeringTurn{
			Content:   msg,
			Timestamp: time.Now(),
		})
		s.EventEmitter.Emit(events.EventSteeringInjected, s.ID, map[string]any{
			"message": msg,
		})
	}
}

// ---------------------------------------------------------------------------
// Loop detection
// ---------------------------------------------------------------------------

// detectLoop examines the recent tool-call history for repeating patterns.
// It hashes each tool-round's calls and checks for consecutive duplicates
// within the detection window.
func (s *Session) detectLoop() bool {
	window := s.Config.LoopDetectionWindow
	if window < 2 {
		window = 2
	}

	// Collect hashes of recent tool-results turns (each preceded by an
	// assistant turn with tool calls).
	var hashes []string
	for _, turn := range s.History {
		at, ok := turn.(*AssistantTurn)
		if !ok || len(at.ToolCalls) == 0 {
			continue
		}
		hashes = append(hashes, hashToolCalls(at.ToolCalls))
	}

	if len(hashes) < 2 {
		return false
	}

	// Only look at the last `window` hashes.
	start := 0
	if len(hashes) > window {
		start = len(hashes) - window
	}
	recent := hashes[start:]

	// Check for the same hash repeating 3+ times consecutively. This
	// indicates the model is stuck in an identical tool-call loop.
	if len(recent) >= 3 {
		last := recent[len(recent)-1]
		repeatCount := 0
		for i := len(recent) - 1; i >= 0; i-- {
			if recent[i] == last {
				repeatCount++
			} else {
				break
			}
		}
		if repeatCount >= 3 {
			return true
		}
	}

	// Check for alternating patterns (A-B-A-B).
	if len(recent) >= 4 {
		n := len(recent)
		if recent[n-1] == recent[n-3] && recent[n-2] == recent[n-4] && recent[n-1] != recent[n-2] {
			return true
		}
	}

	return false
}

// hashToolCalls produces a deterministic hash of a set of tool calls based
// on their names and arguments.
func hashToolCalls(calls []types.ToolCall) string {
	h := sha256.New()
	for _, tc := range calls {
		fmt.Fprintf(h, "%s:", tc.Name)
		for k, v := range tc.Arguments {
			fmt.Fprintf(h, "%s=%v;", k, v)
		}
		h.Write([]byte("|"))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// ---------------------------------------------------------------------------
// Counting
// ---------------------------------------------------------------------------

// countTurns returns the total number of turns in the conversation history.
func (s *Session) countTurns() int {
	return len(s.History)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// extractToolCalls converts the response's tool call data into types.ToolCall.
func extractToolCalls(resp *types.Response) []types.ToolCall {
	data := resp.ToolCalls()
	calls := make([]types.ToolCall, len(data))
	for i, d := range data {
		calls[i] = types.ToolCall{
			ID:        d.ID,
			Name:      d.Name,
			Arguments: d.Arguments,
		}
	}
	return calls
}

// readRawFile reads the entire file content as a string without line numbers.
// This is used by edit_file to perform string replacements on the raw content.
func readRawFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// formatExecResult formats a command execution result for the model.
func formatExecResult(r *env.ExecResult) string {
	var b strings.Builder

	if r.Stdout != "" {
		b.WriteString(r.Stdout)
	}

	if r.Stderr != "" {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString("STDERR:\n")
		b.WriteString(r.Stderr)
	}

	if r.TimedOut {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(fmt.Sprintf("Command timed out after %dms", r.DurationMs))
	}

	if r.ExitCode != 0 {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(fmt.Sprintf("Exit code: %d", r.ExitCode))
	}

	if b.Len() == 0 {
		return "(no output)"
	}

	return b.String()
}

// formatDirEntries formats directory entries for the model.
func formatDirEntries(entries []env.DirEntry) string {
	if len(entries) == 0 {
		return "(empty directory)"
	}

	var b strings.Builder
	for _, e := range entries {
		if e.IsDir {
			fmt.Fprintf(&b, "%s/\n", e.Name)
		} else {
			fmt.Fprintf(&b, "%s (%d bytes)\n", e.Name, e.Size)
		}
	}
	return b.String()
}

// toInt converts a value (typically from JSON-decoded arguments) to int.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case float32:
		return int(n), true
	default:
		return 0, false
	}
}

// generateSessionID produces a unique session identifier.
func generateSessionID() string {
	h := sha256.New()
	fmt.Fprintf(h, "%d", time.Now().UnixNano())
	return fmt.Sprintf("sess_%x", h.Sum(nil)[:8])
}
