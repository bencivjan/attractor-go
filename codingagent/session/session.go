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
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/strongdm/attractor-go/codingagent/env"
	"github.com/strongdm/attractor-go/codingagent/events"
	"github.com/strongdm/attractor-go/codingagent/profile"
	"github.com/strongdm/attractor-go/codingagent/tools"
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
	// EnableStreaming enables streaming mode where the session uses
	// LLMClient.Stream() and emits TEXT_DELTA events as tokens arrive.
	EnableStreaming bool
	// ReasoningEffort controls the model's reasoning depth ("low", "medium",
	// "high", or "" for provider default).
	ReasoningEffort string
	// ToolOutputLimits overrides default character limits for tool output
	// truncation, keyed by tool name.
	ToolOutputLimits map[string]int
	// ToolLineLimits overrides default line limits for tool output
	// truncation, keyed by tool name. Applied after character truncation.
	ToolLineLimits map[string]int
	// EnableLoopDetection enables detection of repeating tool call patterns.
	EnableLoopDetection bool
	// LoopDetectionWindow is the number of recent tool rounds to examine for
	// repeating patterns.
	LoopDetectionWindow int
	// MaxSubagentDepth limits how deeply sub-agents can be nested.
	MaxSubagentDepth int
	// UserInstructions contains user-provided instructions that are appended
	// to the end of the system prompt. This allows callers to inject custom
	// guidance that overrides default behavior.
	UserInstructions string
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
	Content            string
	ToolCalls          []types.ToolCall
	Reasoning          string
	ThinkingSignature  string
	Usage              types.Usage
	ResponseID         string
	Timestamp          time.Time
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
	subagents     *SubAgentManager

	// envContext is the cached <environment> block generated at session start.
	envContext string
	// projectDocs is the cached project instruction files discovered at session start.
	projectDocs string
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
	s.subagents = newSubAgentManager(s)

	// Apply profile-specific config overrides.
	// Anthropic/Claude Code convention uses 120s default shell timeout.
	if prof.ID() == "anthropic-claude" && s.Config.DefaultCommandTimeoutMs == 10000 {
		s.Config.DefaultCommandTimeoutMs = 120000
	}

	// Generate cached context at session start per spec Section 6.3.
	s.envContext = buildEnvironmentContext(execEnv, prof)
	s.projectDocs = discoverProjectDocs(execEnv, prof.InstructionFileNames())

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
		s.State = StateIdle
	}
	// Drain follow-up queue.
	var followups []string
	followups = append(followups, s.followupQueue...)
	s.followupQueue = nil
	s.mu.Unlock()

	// Process any queued follow-up messages (these are part of the same
	// processing cycle, so SESSION_END fires only once after all are done).
	for _, fu := range followups {
		if fuErr := s.processInput(ctx, fu); fuErr != nil {
			s.emitSessionEnd()
			return fuErr
		}
	}

	s.emitSessionEnd()
	return err
}

// emitSessionEnd emits SESSION_END if the session is not closed.
func (s *Session) emitSessionEnd() {
	s.mu.Lock()
	closed := s.State == StateClosed
	s.mu.Unlock()
	if !closed {
		s.EventEmitter.Emit(events.EventSessionEnd, s.ID, nil)
	}
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
// It closes the abort channel (which cancels in-flight contexts), emits a
// SESSION_END event with the abort reason, cleans up any running subagents,
// and transitions the session to StateClosed.
func (s *Session) Abort() {
	s.mu.Lock()
	alreadyAborted := false
	select {
	case <-s.abortCh:
		alreadyAborted = true
	default:
		close(s.abortCh)
	}

	if alreadyAborted {
		s.mu.Unlock()
		return
	}

	s.State = StateClosed
	s.mu.Unlock()

	// Clean up all running subagents.
	if s.subagents != nil {
		s.subagents.CloseAll()
	}

	// Emit SESSION_END with abort reason so listeners know the session
	// was terminated rather than completing naturally.
	s.EventEmitter.Emit(events.EventSessionEnd, s.ID, map[string]any{
		"reason": "aborted",
	})
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

	// Drain any pending steering messages before the first LLM call (spec 2.5).
	s.drainSteering()

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

		// Check context window usage.
		s.checkContextUsage()

		// ---- Step 2: Build LLM request ----
		req := s.buildRequest()

		// ---- Step 3+4: Call LLM and record assistant turn ----
		var assistantTurn *AssistantTurn
		var callErr error

		if s.Config.EnableStreaming && s.Profile.SupportsStreaming() {
			assistantTurn, callErr = s.callLLMStreaming(ctx, req, toolRound)
		} else {
			assistantTurn, callErr = s.callLLMBlocking(ctx, req, toolRound)
		}
		if callErr != nil {
			// Unrecoverable errors (authentication, access denied, context
			// length exceeded) should transition the session to CLOSED
			// because retrying will not help.
			if isUnrecoverableError(callErr) {
				s.mu.Lock()
				s.State = StateClosed
				s.mu.Unlock()
			}
			return callErr
		}
		toolCalls := assistantTurn.ToolCalls
		s.History = append(s.History, assistantTurn)

		// ---- Step 5: No tool calls -> natural completion ----
		if len(toolCalls) == 0 {
			// If the response looks like a question, transition to
			// StateAwaitingInput so the caller knows to provide more
			// input before the session can continue.
			if isQuestionText(assistantTurn.Content) {
				s.mu.Lock()
				s.State = StateAwaitingInput
				s.mu.Unlock()
			}
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
// LLM call modes
// ---------------------------------------------------------------------------

// callLLMBlocking calls the LLM in blocking mode and emits text events after
// the full response is received.
func (s *Session) callLLMBlocking(ctx context.Context, req types.Request, toolRound int) (*AssistantTurn, error) {
	resp, err := s.LLMClient.Complete(ctx, req)
	if err != nil {
		s.EventEmitter.Emit(events.EventError, s.ID, map[string]any{
			"error":  err.Error(),
			"source": "llm_complete",
		})
		return nil, fmt.Errorf("LLM complete: %w", err)
	}

	assistantText := resp.Text()
	toolCalls := extractToolCalls(resp)
	reasoning := resp.Reasoning()

	turn := &AssistantTurn{
		Content:           assistantText,
		ToolCalls:         toolCalls,
		Reasoning:         reasoning,
		ThinkingSignature: extractThinkingSignature(resp),
		Usage:             resp.Usage,
		ResponseID:        resp.ID,
		Timestamp:         time.Now(),
	}

	// Emit text events after response arrives.
	s.EventEmitter.Emit(events.EventAssistantTextStart, s.ID, map[string]any{
		"tool_round": toolRound,
	})
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

	return turn, nil
}

// callLLMStreaming calls the LLM in streaming mode, emitting TEXT_DELTA events
// as tokens arrive for real-time UI rendering (spec Section 2.9).
func (s *Session) callLLMStreaming(ctx context.Context, req types.Request, toolRound int) (*AssistantTurn, error) {
	ch, err := s.LLMClient.Stream(ctx, req)
	if err != nil {
		s.EventEmitter.Emit(events.EventError, s.ID, map[string]any{
			"error":  err.Error(),
			"source": "llm_stream",
		})
		return nil, fmt.Errorf("LLM stream: %w", err)
	}

	s.EventEmitter.Emit(events.EventAssistantTextStart, s.ID, map[string]any{
		"tool_round": toolRound,
		"streaming":  true,
	})

	var textBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var accToolCalls []types.ToolCallData
	var usage types.Usage
	var finishReason string

	// Accumulate tool call argument builders keyed by tool call ID.
	type toolCallAccum struct {
		id   string
		name string
		args strings.Builder
	}
	toolCallAccums := make(map[string]*toolCallAccum)

	for evt := range ch {
		switch evt.Type {
		case types.StreamEventTextDelta:
			textBuilder.WriteString(evt.Delta)
			s.EventEmitter.Emit(events.EventAssistantTextDelta, s.ID, map[string]any{
				"text": evt.Delta,
			})

		case types.StreamEventReasoningDelta:
			reasoningBuilder.WriteString(evt.ReasoningDelta)

		case types.StreamEventToolCallStart:
			if evt.ToolCall != nil {
				toolCallAccums[evt.ToolCall.ID] = &toolCallAccum{
					id:   evt.ToolCall.ID,
					name: evt.ToolCall.Name,
				}
			}

		case types.StreamEventToolCallDelta:
			if evt.ToolCall != nil {
				if acc, ok := toolCallAccums[evt.ToolCall.ID]; ok {
					acc.args.WriteString(evt.Delta)
				}
			}

		case types.StreamEventToolCallEnd:
			if evt.ToolCall != nil {
				accToolCalls = append(accToolCalls, *evt.ToolCall)
				// Remove from accumulators.
				delete(toolCallAccums, evt.ToolCall.ID)
			}

		case types.StreamEventFinish:
			if evt.Usage != nil {
				usage = *evt.Usage
			}
			if evt.FinishReason != nil {
				finishReason = evt.FinishReason.Reason
			}

		case types.StreamEventError:
			s.EventEmitter.Emit(events.EventError, s.ID, map[string]any{
				"error":  evt.Error.Error(),
				"source": "llm_stream",
			})
			return nil, fmt.Errorf("LLM stream error: %w", evt.Error)
		}
	}

	// Finalize any partially accumulated tool calls.
	for _, acc := range toolCallAccums {
		var args map[string]any
		if acc.args.Len() > 0 {
			_ = json.Unmarshal([]byte(acc.args.String()), &args)
		}
		accToolCalls = append(accToolCalls, types.ToolCallData{
			ID:        acc.id,
			Name:      acc.name,
			Type:      "function",
			Arguments: args,
		})
	}

	// Convert ToolCallData to ToolCall.
	var toolCalls []types.ToolCall
	for _, tcd := range accToolCalls {
		toolCalls = append(toolCalls, types.ToolCall{
			ID:        tcd.ID,
			Name:      tcd.Name,
			Arguments: tcd.Arguments,
		})
	}

	s.EventEmitter.Emit(events.EventAssistantTextEnd, s.ID, map[string]any{
		"usage":         usage,
		"finish_reason": finishReason,
		"has_tools":     len(toolCalls) > 0,
	})

	return &AssistantTurn{
		Content:   textBuilder.String(),
		ToolCalls: toolCalls,
		Reasoning: reasoningBuilder.String(),
		Usage:     usage,
		Timestamp: time.Now(),
	}, nil
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

// buildRequest assembles the types.Request from the current session state.
func (s *Session) buildRequest() types.Request {
	// System prompt with cached environment context and project docs.
	systemPrompt := s.Profile.BuildSystemPrompt(s.envContext, s.projectDocs)

	// Append user-provided instructions to the end of the system prompt
	// so they take precedence over default behavior.
	if s.Config.UserInstructions != "" {
		systemPrompt += "\n\n# User Instructions\n" + s.Config.UserInstructions
	}

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
		ToolChoice:      &types.ToolChoice{Mode: "auto"},
		Provider:        s.Profile.ProviderName(),
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
				Text:      t.Reasoning,
				Signature: t.ThinkingSignature,
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

// executeToolCalls runs all tool calls from a single assistant turn. When the
// provider profile supports parallel tool calls and multiple calls are present,
// they are executed concurrently. Otherwise they run sequentially.
func (s *Session) executeToolCalls(ctx context.Context, toolCalls []types.ToolCall) []types.ToolResult {
	if s.Profile.SupportsParallelToolCalls() && len(toolCalls) > 1 {
		return s.executeToolCallsParallel(ctx, toolCalls)
	}
	return s.executeToolCallsSequential(ctx, toolCalls)
}

// executeToolCallsSequential runs tool calls one at a time.
func (s *Session) executeToolCallsSequential(ctx context.Context, toolCalls []types.ToolCall) []types.ToolResult {
	results := make([]types.ToolResult, 0, len(toolCalls))
	for _, tc := range toolCalls {
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

// executeToolCallsParallel runs all tool calls concurrently and collects
// results in the original order.
func (s *Session) executeToolCallsParallel(ctx context.Context, toolCalls []types.ToolCall) []types.ToolResult {
	results := make([]types.ToolResult, len(toolCalls))
	var wg sync.WaitGroup
	wg.Add(len(toolCalls))
	for i, tc := range toolCalls {
		go func(idx int, call types.ToolCall) {
			defer wg.Done()
			results[idx] = s.executeSingleTool(ctx, call)
		}(i, tc)
	}
	wg.Wait()
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
	} else if err := validateToolArgs(registered.Definition, tc.Arguments); err != nil {
		output = fmt.Sprintf("Invalid arguments for tool %q: %s", tc.Name, err.Error())
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

	// Preserve full output for the event stream before truncating.
	fullOutput := output

	// Apply truncation for the LLM context.
	output = truncation.TruncateToolOutput(output, tc.Name, s.Config.ToolOutputLimits, s.Config.ToolLineLimits)

	durationMs := time.Since(startTime).Milliseconds()

	// The TOOL_CALL_END event carries the full untruncated output so that
	// host applications always have access to complete output even though
	// the model sees an abbreviated version.
	s.EventEmitter.Emit(events.EventToolCallEnd, s.ID, map[string]any{
		"tool_call_id": tc.ID,
		"tool_name":    tc.Name,
		"duration_ms":  durationMs,
		"is_error":     isError,
		"output":       fullOutput,
		"output_len":   len(fullOutput),
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

	case "apply_patch":
		patch, _ := args["patch"].(string)
		if patch == "" {
			return "", fmt.Errorf("patch is required")
		}
		workDir := s.ExecutionEnv.WorkingDirectory()
		return tools.ApplyPatch(patch, workDir)

	case "read_many_files":
		pathsRaw, _ := args["paths"].([]any)
		if len(pathsRaw) == 0 {
			return "", fmt.Errorf("paths is required and must be a non-empty array")
		}
		var b strings.Builder
		for i, raw := range pathsRaw {
			filePath, ok := raw.(string)
			if !ok {
				continue
			}
			content, err := s.ExecutionEnv.ReadFile(filePath, 0, 0)
			if err != nil {
				fmt.Fprintf(&b, "--- %s ---\nError: %s\n", filePath, err.Error())
			} else {
				fmt.Fprintf(&b, "--- %s ---\n%s\n", filePath, content)
			}
			if i < len(pathsRaw)-1 {
				b.WriteString("\n")
			}
		}
		return b.String(), nil

	case "spawn_agent":
		task, _ := args["task"].(string)
		if task == "" {
			return "", fmt.Errorf("task is required")
		}
		workDir, _ := args["working_dir"].(string)
		model, _ := args["model"].(string)
		maxTurns, _ := toInt(args["max_turns"])
		agentID, err := s.subagents.Spawn(ctx, task, workDir, model, maxTurns)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Spawned subagent %s", agentID), nil

	case "send_input":
		agentID, _ := args["agent_id"].(string)
		if agentID == "" {
			return "", fmt.Errorf("agent_id is required")
		}
		message, _ := args["message"].(string)
		if message == "" {
			return "", fmt.Errorf("message is required")
		}
		if err := s.subagents.SendInput(agentID, message); err != nil {
			return "", err
		}
		return fmt.Sprintf("Message sent to %s", agentID), nil

	case "wait":
		agentID, _ := args["agent_id"].(string)
		if agentID == "" {
			return "", fmt.Errorf("agent_id is required")
		}
		result, err := s.subagents.Wait(agentID)
		if err != nil {
			return "", err
		}
		status := "succeeded"
		if !result.Success {
			status = "failed"
		}
		return fmt.Sprintf("Subagent %s %s (%d turns used):\n\n%s", agentID, status, result.TurnsUsed, result.Output), nil

	case "close_agent":
		agentID, _ := args["agent_id"].(string)
		if agentID == "" {
			return "", fmt.Errorf("agent_id is required")
		}
		finalStatus, err := s.subagents.Close(agentID)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Subagent %s closed (status: %s)", agentID, finalStatus), nil

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

// detectLoop examines the recent tool-call history for repeating patterns
// of length 1, 2, or 3 within the detection window (spec Section 2.10).
// The window must be fully populated before detection activates.
func (s *Session) detectLoop() bool {
	window := s.Config.LoopDetectionWindow
	if window < 3 {
		window = 6 // reasonable default to detect patterns of length 1-3
	}

	// Collect hashes of recent assistant turns with tool calls.
	var hashes []string
	for _, turn := range s.History {
		at, ok := turn.(*AssistantTurn)
		if !ok || len(at.ToolCalls) == 0 {
			continue
		}
		hashes = append(hashes, hashToolCalls(at.ToolCalls))
	}

	// Window must be fully populated before detection activates (spec 2.10).
	if len(hashes) < window {
		return false
	}

	// Use only the last `window` hashes.
	recent := hashes[len(hashes)-window:]

	// Check for repeating patterns of length 1, 2, and 3.
	for patternLen := 1; patternLen <= 3; patternLen++ {
		if window%patternLen != 0 {
			continue
		}
		pattern := recent[:patternLen]
		allMatch := true
		for i := patternLen; i < window; i++ {
			if recent[i] != pattern[i%patternLen] {
				allMatch = false
				break
			}
		}
		if allMatch {
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
		// Sort keys for deterministic hashing (Go map iteration is random).
		keys := make([]string, 0, len(tc.Arguments))
		for k := range tc.Arguments {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(h, "%s=%v;", k, tc.Arguments[k])
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

// checkContextUsage estimates token usage using the ~4 chars/token heuristic
// and emits a warning when usage exceeds 80% of the context window (spec 5.5).
func (s *Session) checkContextUsage() {
	windowSize := s.Profile.ContextWindowSize()
	if windowSize <= 0 {
		return
	}

	totalChars := 0
	for _, turn := range s.History {
		switch t := turn.(type) {
		case *UserTurn:
			totalChars += len(t.Content)
		case *AssistantTurn:
			totalChars += len(t.Content) + len(t.Reasoning)
		case *ToolResultsTurn:
			for _, r := range t.Results {
				if s, ok := r.Content.(string); ok {
					totalChars += len(s)
				}
			}
		case *SystemTurn:
			totalChars += len(t.Content)
		case *SteeringTurn:
			totalChars += len(t.Content)
		}
	}

	approxTokens := totalChars / 4
	threshold := int(float64(windowSize) * 0.8)
	if approxTokens > threshold {
		pct := int(float64(approxTokens) / float64(windowSize) * 100)
		s.EventEmitter.Emit(events.EventWarning, s.ID, map[string]any{
			"message": fmt.Sprintf("Context usage at ~%d%% of context window", pct),
			"approx_tokens": approxTokens,
			"window_size":   windowSize,
		})
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// extractToolCalls converts the response's tool call data into types.ToolCall.
// extractThinkingSignature returns the signature from the first thinking block
// in the response. The Anthropic API requires this signature when replaying
// thinking blocks in subsequent conversation turns.
func extractThinkingSignature(resp *types.Response) string {
	if resp == nil {
		return ""
	}
	for _, p := range resp.Message.Content {
		if p.Kind == types.ContentKindThinking && p.Thinking != nil && p.Thinking.Signature != "" {
			return p.Thinking.Signature
		}
	}
	return ""
}

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

// validateToolArgs checks that required parameters are present and that
// argument types match the JSON Schema declarations (spec Section 3.8).
func validateToolArgs(def tools.Definition, args map[string]any) error {
	if len(def.Parameters) == 0 {
		return nil
	}

	// Check required fields.
	reqSlice, _ := def.Parameters["required"]
	var required []string
	switch r := reqSlice.(type) {
	case []string:
		required = r
	case []any:
		for _, v := range r {
			if s, ok := v.(string); ok {
				required = append(required, s)
			}
		}
	}
	var missing []string
	for _, name := range required {
		if _, exists := args[name]; !exists {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required parameter(s): %s", strings.Join(missing, ", "))
	}

	// Type-check each argument against the properties schema.
	props, _ := def.Parameters["properties"].(map[string]any)
	if props == nil {
		return nil
	}
	var typeErrors []string
	for name, val := range args {
		propSchema, ok := props[name].(map[string]any)
		if !ok {
			continue
		}
		expectedType, _ := propSchema["type"].(string)
		if expectedType == "" {
			continue
		}
		if err := checkJSONType(name, val, expectedType); err != "" {
			typeErrors = append(typeErrors, err)
		}
	}
	if len(typeErrors) > 0 {
		return fmt.Errorf("argument type error(s): %s", strings.Join(typeErrors, "; "))
	}
	return nil
}

// checkJSONType validates that val matches the expected JSON Schema type.
// Returns an error description or empty string on success.
func checkJSONType(name string, val any, expectedType string) string {
	switch expectedType {
	case "string":
		if _, ok := val.(string); !ok {
			return fmt.Sprintf("%s: expected string, got %T", name, val)
		}
	case "integer":
		switch v := val.(type) {
		case float64:
			if v != float64(int(v)) {
				return fmt.Sprintf("%s: expected integer, got float", name)
			}
		case int:
			// ok
		default:
			return fmt.Sprintf("%s: expected integer, got %T", name, val)
		}
	case "number":
		switch val.(type) {
		case float64, int:
			// ok
		default:
			return fmt.Sprintf("%s: expected number, got %T", name, val)
		}
	case "boolean":
		if _, ok := val.(bool); !ok {
			return fmt.Sprintf("%s: expected boolean, got %T", name, val)
		}
	case "object":
		if _, ok := val.(map[string]any); !ok {
			return fmt.Sprintf("%s: expected object, got %T", name, val)
		}
	case "array":
		if _, ok := val.([]any); !ok {
			return fmt.Sprintf("%s: expected array, got %T", name, val)
		}
	}
	return ""
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
		fmt.Fprintf(&b, "[ERROR: Command timed out after %dms. Partial output is shown above.\nYou can retry with a longer timeout by setting the timeout_ms parameter.]", r.DurationMs)
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

// isQuestionText detects whether the model's response looks like a question
// directed at the user. It uses simple heuristics: the text ends with "?" or
// contains common question phrases near the end. This is intentionally
// conservative to avoid false positives on rhetorical or embedded questions.
func isQuestionText(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}

	// Direct check: text ends with a question mark.
	if strings.HasSuffix(trimmed, "?") {
		return true
	}

	// Check the last ~200 characters for question patterns. This catches
	// responses that end with a question followed by a small amount of
	// formatting (e.g. markdown bullet lists of options).
	tail := trimmed
	if len(tail) > 200 {
		tail = tail[len(tail)-200:]
	}
	tail = strings.ToLower(tail)

	questionPhrases := []string{
		"could you clarify",
		"can you clarify",
		"would you like",
		"do you want",
		"shall i",
		"should i",
		"what would you",
		"which option",
		"please let me know",
		"please confirm",
		"what do you think",
		"how would you like",
	}
	for _, phrase := range questionPhrases {
		if strings.Contains(tail, phrase) {
			return true
		}
	}

	return false
}

// isUnrecoverableError checks whether an LLM call error is permanent and
// should not be retried. Authentication failures, access denied responses,
// and context length exceeded errors all fall into this category.
func isUnrecoverableError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())

	unrecoverablePatterns := []string{
		"authentication",
		"auth error",
		"unauthorized",
		"401",
		"403",
		"access denied",
		"permission denied",
		"invalid api key",
		"invalid_api_key",
		"context length exceeded",
		"context_length_exceeded",
		"maximum context length",
		"token limit",
		"max tokens",
		"rate limit", // rate limiting is often temporary, but persistent 429s are unrecoverable at session level
	}
	for _, pattern := range unrecoverablePatterns {
		if strings.Contains(msg, pattern) {
			return true
		}
	}
	return false
}
