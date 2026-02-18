package session

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/strongdm/attractor-go/codingagent/env"
	"github.com/strongdm/attractor-go/codingagent/profile"
	"github.com/strongdm/attractor-go/codingagent/tools"
	"github.com/strongdm/attractor-go/unifiedllm/types"
)

// ---------------------------------------------------------------------------
// Mock LLMClient
// ---------------------------------------------------------------------------

// mockLLMClient returns pre-configured responses in sequence.
type mockLLMClient struct {
	mu        sync.Mutex
	responses []*types.Response
	callCount int
}

func newMockLLMClient(responses ...*types.Response) *mockLLMClient {
	return &mockLLMClient{responses: responses}
}

func (m *mockLLMClient) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	idx := m.callCount
	m.callCount++

	if idx >= len(m.responses) {
		// Return an empty text response as a natural completion fallback.
		return &types.Response{
			ID:    "fallback",
			Model: "mock",
			Message: types.Message{
				Role: types.RoleAssistant,
				Content: []types.ContentPart{
					{Kind: types.ContentKindText, Text: "No more responses configured."},
				},
			},
			FinishReason: types.FinishReasonStop,
		}, nil
	}

	return m.responses[idx], nil
}

func (m *mockLLMClient) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	return nil, fmt.Errorf("streaming not implemented in mock")
}

func (m *mockLLMClient) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// ---------------------------------------------------------------------------
// Mock ExecutionEnvironment
// ---------------------------------------------------------------------------

type mockExecEnv struct {
	workDir     string
	fileContent map[string]string
	execResults map[string]*env.ExecResult
}

func newMockExecEnv() *mockExecEnv {
	return &mockExecEnv{
		workDir:     "/mock/work",
		fileContent: make(map[string]string),
		execResults: make(map[string]*env.ExecResult),
	}
}

func (e *mockExecEnv) ReadFile(path string, offset, limit int) (string, error) {
	content, ok := e.fileContent[path]
	if !ok {
		return "", fmt.Errorf("file not found: %s", path)
	}
	return content, nil
}

func (e *mockExecEnv) WriteFile(path, content string) error {
	e.fileContent[path] = content
	return nil
}

func (e *mockExecEnv) FileExists(path string) bool {
	_, ok := e.fileContent[path]
	return ok
}

func (e *mockExecEnv) ListDirectory(path string, depth int) ([]env.DirEntry, error) {
	return []env.DirEntry{{Name: "file.txt", IsDir: false, Size: 100}}, nil
}

func (e *mockExecEnv) ExecCommand(ctx context.Context, command string, timeoutMs int, workingDir string, envVars map[string]string) (*env.ExecResult, error) {
	if result, ok := e.execResults[command]; ok {
		return result, nil
	}
	return &env.ExecResult{Stdout: "mock output", ExitCode: 0}, nil
}

func (e *mockExecEnv) Grep(pattern, path string, caseInsensitive bool, maxResults int, globFilter string) (string, error) {
	return "mock:1:match", nil
}

func (e *mockExecEnv) Glob(pattern, path string) ([]string, error) {
	return []string{"file.go"}, nil
}

func (e *mockExecEnv) WorkingDirectory() string { return e.workDir }
func (e *mockExecEnv) Platform() string          { return "test" }
func (e *mockExecEnv) OSVersion() string         { return "test 1.0" }
func (e *mockExecEnv) Initialize() error         { return nil }
func (e *mockExecEnv) Cleanup() error            { return nil }

// ---------------------------------------------------------------------------
// Mock ProviderProfile
// ---------------------------------------------------------------------------

type mockProfile struct {
	model    string
	registry *tools.Registry
}

func newMockProfile() *mockProfile {
	reg := tools.NewRegistry()
	// Register a simple echo tool with an executor.
	reg.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "echo_tool",
			Description: "Echoes input",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"message": map[string]any{"type": "string"}}},
		},
		Executor: func(args map[string]any, execEnv any) (string, error) {
			msg, _ := args["message"].(string)
			return "echo: " + msg, nil
		},
	})
	return &mockProfile{
		model:    "mock-model",
		registry: reg,
	}
}

func (p *mockProfile) ID() string                      { return "mock" }
func (p *mockProfile) Model() string                   { return p.model }
func (p *mockProfile) ToolRegistry() *tools.Registry    { return p.registry }
func (p *mockProfile) Tools() []tools.Definition        { return p.registry.Definitions() }
func (p *mockProfile) ProviderOptions() map[string]any  { return nil }
func (p *mockProfile) SupportsReasoning() bool          { return false }
func (p *mockProfile) SupportsStreaming() bool           { return false }
func (p *mockProfile) SupportsParallelToolCalls() bool   { return false }
func (p *mockProfile) ContextWindowSize() int            { return 100000 }
func (p *mockProfile) BuildSystemPrompt(workDir string, projectDocs string) string {
	return "You are a test assistant."
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makeTextResponse creates a mock LLM response with only text content (no tool calls).
func makeTextResponse(text string) *types.Response {
	return &types.Response{
		ID:    "resp-text",
		Model: "mock",
		Message: types.Message{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				{Kind: types.ContentKindText, Text: text},
			},
		},
		FinishReason: types.FinishReasonStop,
	}
}

// makeToolCallResponse creates a mock LLM response with tool calls.
func makeToolCallResponse(calls ...types.ToolCallData) *types.Response {
	var parts []types.ContentPart
	for _, tc := range calls {
		parts = append(parts, types.ContentPart{
			Kind:     types.ContentKindToolCall,
			ToolCall: &tc,
		})
	}
	return &types.Response{
		ID:    "resp-tools",
		Model: "mock",
		Message: types.Message{
			Role:    types.RoleAssistant,
			Content: parts,
		},
		FinishReason: types.FinishReasonToolCalls,
	}
}

func newTestSession(client LLMClient) *Session {
	prof := newMockProfile()
	execEnv := newMockExecEnv()
	cfg := DefaultConfig()
	cfg.EnableLoopDetection = false // Disable for most tests.
	return New(prof, execEnv, client, cfg)
}

// ---------------------------------------------------------------------------
// Test: Session creation and state transitions
// ---------------------------------------------------------------------------

func TestSession_CreationAndStates(t *testing.T) {
	client := newMockLLMClient(makeTextResponse("hello"))
	s := newTestSession(client)

	if s.State != StateIdle {
		t.Errorf("expected initial state idle, got %q", s.State)
	}
	if s.ID == "" {
		t.Error("expected non-empty session ID")
	}

	err := s.Submit(context.Background(), "test input")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if s.State != StateAwaitingInput {
		t.Errorf("expected state awaiting_input after submit, got %q", s.State)
	}

	s.Close()
	if s.State != StateClosed {
		t.Errorf("expected state closed after close, got %q", s.State)
	}

	// Submit after close should error.
	err = s.Submit(context.Background(), "after close")
	if err == nil {
		t.Error("expected error submitting to closed session")
	}
}

// ---------------------------------------------------------------------------
// Test: Submit with text-only response (no tool calls) -> natural completion
// ---------------------------------------------------------------------------

func TestSession_TextOnlyResponse(t *testing.T) {
	client := newMockLLMClient(makeTextResponse("Here is my answer"))
	s := newTestSession(client)

	err := s.Submit(context.Background(), "What is Go?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// History should contain: user turn, assistant turn.
	if len(s.History) != 2 {
		t.Fatalf("expected 2 history entries, got %d", len(s.History))
	}

	userTurn, ok := s.History[0].(*UserTurn)
	if !ok {
		t.Fatalf("expected UserTurn at index 0, got %T", s.History[0])
	}
	if userTurn.Content != "What is Go?" {
		t.Errorf("expected user content 'What is Go?', got %q", userTurn.Content)
	}

	assistantTurn, ok := s.History[1].(*AssistantTurn)
	if !ok {
		t.Fatalf("expected AssistantTurn at index 1, got %T", s.History[1])
	}
	if assistantTurn.Content != "Here is my answer" {
		t.Errorf("expected assistant content, got %q", assistantTurn.Content)
	}
	if len(assistantTurn.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(assistantTurn.ToolCalls))
	}

	if client.CallCount() != 1 {
		t.Errorf("expected 1 LLM call, got %d", client.CallCount())
	}
}

// ---------------------------------------------------------------------------
// Test: Submit with tool calls -> execution -> second LLM call -> completion
// ---------------------------------------------------------------------------

func TestSession_ToolCallAndCompletion(t *testing.T) {
	// First response: tool call. Second response: text completion.
	toolCallResp := makeToolCallResponse(types.ToolCallData{
		ID:        "tc1",
		Name:      "echo_tool",
		Arguments: map[string]any{"message": "hello"},
	})
	textResp := makeTextResponse("Done using the tool")

	client := newMockLLMClient(toolCallResp, textResp)
	s := newTestSession(client)

	err := s.Submit(context.Background(), "Use the echo tool")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// History: user, assistant(tool call), tool results, assistant(text).
	if len(s.History) != 4 {
		t.Fatalf("expected 4 history entries, got %d", len(s.History))
	}

	// Check the tool results turn.
	toolResultsTurn, ok := s.History[2].(*ToolResultsTurn)
	if !ok {
		t.Fatalf("expected ToolResultsTurn at index 2, got %T", s.History[2])
	}
	if len(toolResultsTurn.Results) != 1 {
		t.Fatalf("expected 1 tool result, got %d", len(toolResultsTurn.Results))
	}
	result := toolResultsTurn.Results[0]
	if result.ToolCallID != "tc1" {
		t.Errorf("expected tool call ID 'tc1', got %q", result.ToolCallID)
	}
	content, ok := result.Content.(string)
	if !ok {
		t.Fatalf("expected string content, got %T", result.Content)
	}
	if !strings.Contains(content, "echo: hello") {
		t.Errorf("expected 'echo: hello' in tool output, got %q", content)
	}

	// Check the final assistant turn.
	finalTurn, ok := s.History[3].(*AssistantTurn)
	if !ok {
		t.Fatalf("expected AssistantTurn at index 3, got %T", s.History[3])
	}
	if finalTurn.Content != "Done using the tool" {
		t.Errorf("expected final text, got %q", finalTurn.Content)
	}

	if client.CallCount() != 2 {
		t.Errorf("expected 2 LLM calls, got %d", client.CallCount())
	}
}

// ---------------------------------------------------------------------------
// Test: Abort during processing
// ---------------------------------------------------------------------------

func TestSession_Abort(t *testing.T) {
	// The LLM returns a tool call on the first call. Between the tool
	// execution and the next LLM call, the abort signal is checked.
	// We abort before the second LLM call, which triggers the abort error.
	callCount := 0
	abortingClient := &abortingLLMClient{
		onCall: func() *types.Response {
			callCount++
			if callCount == 1 {
				// First call: return tool calls so the loop continues.
				return makeToolCallResponse(types.ToolCallData{
					ID:        "tc_abort",
					Name:      "echo_tool",
					Arguments: map[string]any{"message": "work"},
				})
			}
			// Should not reach here if abort works.
			return makeTextResponse("should not see this")
		},
	}

	s := newTestSession(abortingClient)

	// Pre-abort the session before submitting.
	s.Abort()

	err := s.Submit(context.Background(), "start work")
	if err == nil {
		t.Fatal("expected error from abort")
	}
	if !strings.Contains(err.Error(), "abort") {
		t.Errorf("expected abort-related error, got: %v", err)
	}
}

// abortingLLMClient calls onCall for each Complete invocation.
type abortingLLMClient struct {
	onCall func() *types.Response
}

func (c *abortingLLMClient) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	return c.onCall(), nil
}

func (c *abortingLLMClient) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	return nil, fmt.Errorf("not implemented")
}

// ---------------------------------------------------------------------------
// Test: MaxTurns limit
// ---------------------------------------------------------------------------

func TestSession_MaxTurns(t *testing.T) {
	// The LLM keeps returning tool calls, never completing naturally.
	toolResp := makeToolCallResponse(types.ToolCallData{
		ID:        "tc1",
		Name:      "echo_tool",
		Arguments: map[string]any{"message": "loop"},
	})

	// Return many tool calls to force the turn limit.
	responses := make([]*types.Response, 50)
	for i := range responses {
		responses[i] = toolResp
	}

	client := newMockLLMClient(responses...)
	s := newTestSession(client)
	// MaxTurns includes user turn + assistant turns + tool results.
	// With MaxTurns=3, after: user(1) + assistant(2) + tool_results(3) -> limit hit.
	s.Config.MaxTurns = 3

	err := s.Submit(context.Background(), "keep going")
	if err == nil {
		t.Fatal("expected error from max turns limit")
	}
	if !strings.Contains(err.Error(), "maximum turn limit") {
		t.Errorf("expected max turn limit error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Test: MaxToolRoundsPerInput limit
// ---------------------------------------------------------------------------

func TestSession_MaxToolRoundsPerInput(t *testing.T) {
	toolResp := makeToolCallResponse(types.ToolCallData{
		ID:        "tc1",
		Name:      "echo_tool",
		Arguments: map[string]any{"message": "round"},
	})

	responses := make([]*types.Response, 50)
	for i := range responses {
		responses[i] = toolResp
	}

	client := newMockLLMClient(responses...)
	s := newTestSession(client)
	s.Config.MaxToolRoundsPerInput = 2

	err := s.Submit(context.Background(), "keep going")
	if err == nil {
		t.Fatal("expected error from max tool rounds limit")
	}
	if !strings.Contains(err.Error(), "maximum tool rounds") {
		t.Errorf("expected max tool rounds error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Test: Loop detection with repeating tool calls
// ---------------------------------------------------------------------------

func TestSession_LoopDetection(t *testing.T) {
	// Create the same tool call response that will repeat.
	toolResp := makeToolCallResponse(types.ToolCallData{
		ID:        "tc1",
		Name:      "echo_tool",
		Arguments: map[string]any{"message": "same"},
	})

	// Need many responses so we can repeat at least 3 times.
	responses := make([]*types.Response, 20)
	for i := range responses {
		responses[i] = toolResp
	}
	// End with text response so the loop eventually finishes after injection.
	responses = append(responses, makeTextResponse("done"))

	client := newMockLLMClient(responses...)
	s := newTestSession(client)
	s.Config.EnableLoopDetection = true
	s.Config.LoopDetectionWindow = 10
	// Set a tool rounds limit as a safety net.
	s.Config.MaxToolRoundsPerInput = 10

	err := s.Submit(context.Background(), "do something")
	// The loop detection should inject a steering message. The test verifies
	// that a SteeringTurn appears in the history.
	// It may still error from the tool rounds limit, which is acceptable.
	_ = err

	// Check that a steering turn was injected.
	foundSteering := false
	for _, turn := range s.History {
		if st, ok := turn.(*SteeringTurn); ok {
			if strings.Contains(st.Content, "repeating") {
				foundSteering = true
				break
			}
		}
	}
	if !foundSteering {
		t.Error("expected a steering turn to be injected due to loop detection")
	}
}

// ---------------------------------------------------------------------------
// Test: Steering message injection
// ---------------------------------------------------------------------------

func TestSession_SteeringInjection(t *testing.T) {
	// First call returns tool call, second returns text.
	toolResp := makeToolCallResponse(types.ToolCallData{
		ID:        "tc1",
		Name:      "echo_tool",
		Arguments: map[string]any{"message": "test"},
	})
	textResp := makeTextResponse("final answer")
	client := newMockLLMClient(toolResp, textResp)

	s := newTestSession(client)

	// Queue a steering message before submit; it will be drained after the
	// tool round.
	s.Steer("Focus on testing only")

	err := s.Submit(context.Background(), "start")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify steering turn is in history.
	foundSteering := false
	for _, turn := range s.History {
		if st, ok := turn.(*SteeringTurn); ok {
			if strings.Contains(st.Content, "Focus on testing") {
				foundSteering = true
				break
			}
		}
	}
	if !foundSteering {
		t.Error("expected steering turn in history")
	}
}

// ---------------------------------------------------------------------------
// Test: History conversion to messages
// ---------------------------------------------------------------------------

func TestSession_ConvertHistoryToMessages(t *testing.T) {
	client := newMockLLMClient()
	s := newTestSession(client)

	// Manually populate history with various turn types.
	s.History = []Turn{
		&UserTurn{Content: "hello"},
		&AssistantTurn{Content: "hi there", ToolCalls: nil},
		&SystemTurn{Content: "system note"},
		&SteeringTurn{Content: "please focus"},
		&ToolResultsTurn{Results: []types.ToolResult{
			{ToolCallID: "tc1", Content: "result data", IsError: false},
		}},
	}

	msgs := s.convertHistoryToMessages()

	if len(msgs) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(msgs))
	}

	// Check roles in order.
	expectedRoles := []types.Role{
		types.RoleUser,      // UserTurn
		types.RoleAssistant, // AssistantTurn
		types.RoleSystem,    // SystemTurn
		types.RoleUser,      // SteeringTurn -> user message
		types.RoleTool,      // ToolResultsTurn
	}

	for i, expected := range expectedRoles {
		if msgs[i].Role != expected {
			t.Errorf("message %d: expected role %q, got %q", i, expected, msgs[i].Role)
		}
	}
}

// ---------------------------------------------------------------------------
// Test: Turn types
// ---------------------------------------------------------------------------

func TestTurnTypes(t *testing.T) {
	ut := &UserTurn{Content: "hi"}
	if ut.TurnType() != "user" {
		t.Errorf("expected 'user', got %q", ut.TurnType())
	}

	at := &AssistantTurn{Content: "hello"}
	if at.TurnType() != "assistant" {
		t.Errorf("expected 'assistant', got %q", at.TurnType())
	}

	trt := &ToolResultsTurn{}
	if trt.TurnType() != "tool_results" {
		t.Errorf("expected 'tool_results', got %q", trt.TurnType())
	}

	st := &SystemTurn{Content: "sys"}
	if st.TurnType() != "system" {
		t.Errorf("expected 'system', got %q", st.TurnType())
	}

	steer := &SteeringTurn{Content: "steer"}
	if steer.TurnType() != "steering" {
		t.Errorf("expected 'steering', got %q", steer.TurnType())
	}
}

// ---------------------------------------------------------------------------
// Test: Close is idempotent
// ---------------------------------------------------------------------------

func TestSession_CloseIdempotent(t *testing.T) {
	client := newMockLLMClient(makeTextResponse("hi"))
	s := newTestSession(client)

	s.Close()
	s.Close() // Should not panic.

	if s.State != StateClosed {
		t.Errorf("expected closed state, got %q", s.State)
	}
}

// ---------------------------------------------------------------------------
// Test: DefaultConfig values
// ---------------------------------------------------------------------------

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.DefaultCommandTimeoutMs != 10000 {
		t.Errorf("expected DefaultCommandTimeoutMs=10000, got %d", cfg.DefaultCommandTimeoutMs)
	}
	if cfg.MaxCommandTimeoutMs != 600000 {
		t.Errorf("expected MaxCommandTimeoutMs=600000, got %d", cfg.MaxCommandTimeoutMs)
	}
	if !cfg.EnableLoopDetection {
		t.Error("expected EnableLoopDetection=true")
	}
	if cfg.LoopDetectionWindow != 10 {
		t.Errorf("expected LoopDetectionWindow=10, got %d", cfg.LoopDetectionWindow)
	}
	if cfg.MaxSubagentDepth != 1 {
		t.Errorf("expected MaxSubagentDepth=1, got %d", cfg.MaxSubagentDepth)
	}
}

// ---------------------------------------------------------------------------
// Test: detectLoop returns false for insufficient history
// ---------------------------------------------------------------------------

func TestDetectLoop_InsufficientHistory(t *testing.T) {
	client := newMockLLMClient()
	s := newTestSession(client)
	s.Config.EnableLoopDetection = true
	s.Config.LoopDetectionWindow = 10

	// No history at all.
	if s.detectLoop() {
		t.Error("expected no loop detected with empty history")
	}

	// Only one assistant turn with tool calls.
	s.History = []Turn{
		&AssistantTurn{ToolCalls: []types.ToolCall{{ID: "1", Name: "tool_a"}}},
	}
	if s.detectLoop() {
		t.Error("expected no loop detected with single tool call turn")
	}
}

// ---------------------------------------------------------------------------
// Test: detectLoop detects 3 consecutive identical tool call rounds
// ---------------------------------------------------------------------------

func TestDetectLoop_ThreeConsecutiveIdentical(t *testing.T) {
	client := newMockLLMClient()
	s := newTestSession(client)
	s.Config.EnableLoopDetection = true
	s.Config.LoopDetectionWindow = 10

	// Three identical tool call assistant turns.
	for i := 0; i < 3; i++ {
		s.History = append(s.History, &AssistantTurn{
			ToolCalls: []types.ToolCall{{ID: "1", Name: "same_tool", Arguments: map[string]any{"x": "y"}}},
		})
	}

	if !s.detectLoop() {
		t.Error("expected loop detected for 3 identical consecutive tool call rounds")
	}
}

// ---------------------------------------------------------------------------
// Test: detectLoop detects alternating pattern (A-B-A-B)
// ---------------------------------------------------------------------------

func TestDetectLoop_AlternatingPattern(t *testing.T) {
	client := newMockLLMClient()
	s := newTestSession(client)
	s.Config.EnableLoopDetection = true
	s.Config.LoopDetectionWindow = 10

	// Alternating A-B-A-B pattern.
	for i := 0; i < 4; i++ {
		name := "tool_a"
		if i%2 == 1 {
			name = "tool_b"
		}
		s.History = append(s.History, &AssistantTurn{
			ToolCalls: []types.ToolCall{{ID: "1", Name: name, Arguments: map[string]any{"n": i % 2}}},
		})
	}

	if !s.detectLoop() {
		t.Error("expected loop detected for A-B-A-B alternating pattern")
	}
}

// ---------------------------------------------------------------------------
// Test: buildAssistantContent
// ---------------------------------------------------------------------------

func TestBuildAssistantContent(t *testing.T) {
	// Turn with reasoning, text, and tool calls.
	turn := &AssistantTurn{
		Content:   "Some text",
		Reasoning: "I think...",
		ToolCalls: []types.ToolCall{{ID: "tc1", Name: "tool_a", Arguments: map[string]any{"x": 1}}},
	}

	parts := buildAssistantContent(turn)
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(parts))
	}
	if parts[0].Kind != types.ContentKindThinking {
		t.Errorf("expected thinking part first, got %q", parts[0].Kind)
	}
	if parts[1].Kind != types.ContentKindText {
		t.Errorf("expected text part second, got %q", parts[1].Kind)
	}
	if parts[2].Kind != types.ContentKindToolCall {
		t.Errorf("expected tool_call part third, got %q", parts[2].Kind)
	}
}

// ---------------------------------------------------------------------------
// Test: buildAssistantContent with empty turn
// ---------------------------------------------------------------------------

func TestBuildAssistantContent_Empty(t *testing.T) {
	turn := &AssistantTurn{}
	parts := buildAssistantContent(turn)
	// Should have at least one empty text part.
	if len(parts) != 1 {
		t.Fatalf("expected 1 part for empty turn, got %d", len(parts))
	}
	if parts[0].Kind != types.ContentKindText {
		t.Errorf("expected text kind, got %q", parts[0].Kind)
	}
}

// ---------------------------------------------------------------------------
// Test: Session with real AnthropicProfile (integration-like)
// ---------------------------------------------------------------------------

func TestSession_WithRealProfile(t *testing.T) {
	prof := profile.NewAnthropicProfile("test-model")
	execEnv := newMockExecEnv()
	client := newMockLLMClient(makeTextResponse("all done"))
	cfg := DefaultConfig()
	cfg.EnableLoopDetection = false

	s := New(prof, execEnv, client, cfg)

	err := s.Submit(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(s.History) != 2 {
		t.Errorf("expected 2 history entries, got %d", len(s.History))
	}
}

// ---------------------------------------------------------------------------
// Test: countTurns
// ---------------------------------------------------------------------------

func TestCountTurns(t *testing.T) {
	client := newMockLLMClient()
	s := newTestSession(client)

	if s.countTurns() != 0 {
		t.Errorf("expected 0 turns, got %d", s.countTurns())
	}

	s.History = append(s.History, &UserTurn{Content: "hi"})
	s.History = append(s.History, &AssistantTurn{Content: "hello"})

	if s.countTurns() != 2 {
		t.Errorf("expected 2 turns, got %d", s.countTurns())
	}
}

// ---------------------------------------------------------------------------
// Test: toInt helper
// ---------------------------------------------------------------------------

func TestToInt(t *testing.T) {
	tests := []struct {
		input    any
		expected int
		ok       bool
	}{
		{42, 42, true},
		{int64(100), 100, true},
		{float64(3.0), 3, true},
		{float32(5.0), 5, true},
		{"not a number", 0, false},
		{nil, 0, false},
	}

	for _, tt := range tests {
		got, ok := toInt(tt.input)
		if ok != tt.ok {
			t.Errorf("toInt(%v): ok = %v, want %v", tt.input, ok, tt.ok)
		}
		if got != tt.expected {
			t.Errorf("toInt(%v) = %d, want %d", tt.input, got, tt.expected)
		}
	}
}

// ---------------------------------------------------------------------------
// Test: toolResultContent helper
// ---------------------------------------------------------------------------

func TestToolResultContent(t *testing.T) {
	// String content.
	r1 := types.ToolResult{ToolCallID: "1", Content: "hello", IsError: false}
	content, isErr := toolResultContent(r1)
	if content != "hello" || isErr {
		t.Errorf("expected ('hello', false), got (%q, %v)", content, isErr)
	}

	// Non-string content.
	r2 := types.ToolResult{ToolCallID: "2", Content: 42, IsError: true}
	content2, isErr2 := toolResultContent(r2)
	if content2 != "42" || !isErr2 {
		t.Errorf("expected ('42', true), got (%q, %v)", content2, isErr2)
	}
}

// ---------------------------------------------------------------------------
// Test: FollowUp queuing
// ---------------------------------------------------------------------------

func TestSession_FollowUp(t *testing.T) {
	// First submit: returns text, which triggers follow-up processing.
	// Follow-up submit: also returns text.
	client := newMockLLMClient(
		makeTextResponse("first answer"),
		makeTextResponse("follow-up answer"),
	)
	s := newTestSession(client)

	// Queue a follow-up before submitting.
	s.FollowUp("follow-up question")

	err := s.Submit(context.Background(), "initial question")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have processed both the initial and follow-up.
	if client.CallCount() != 2 {
		t.Errorf("expected 2 LLM calls (initial + follow-up), got %d", client.CallCount())
	}

	// History should contain turns from both inputs.
	userCount := 0
	for _, turn := range s.History {
		if _, ok := turn.(*UserTurn); ok {
			userCount++
		}
	}
	if userCount != 2 {
		t.Errorf("expected 2 user turns, got %d", userCount)
	}
}
