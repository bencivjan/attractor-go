package types

import (
	"testing"
)

// ---------------------------------------------------------------------------
// Message helper constructors
// ---------------------------------------------------------------------------

func TestSystemMessage(t *testing.T) {
	m := SystemMessage("You are a helpful assistant.")
	if m.Role != RoleSystem {
		t.Errorf("SystemMessage Role = %q, want %q", m.Role, RoleSystem)
	}
	if len(m.Content) != 1 {
		t.Fatalf("SystemMessage Content length = %d, want 1", len(m.Content))
	}
	if m.Content[0].Kind != ContentKindText {
		t.Errorf("SystemMessage Content[0].Kind = %q, want %q", m.Content[0].Kind, ContentKindText)
	}
	if m.Content[0].Text != "You are a helpful assistant." {
		t.Errorf("SystemMessage Content[0].Text = %q, want %q", m.Content[0].Text, "You are a helpful assistant.")
	}
}

func TestUserMessage(t *testing.T) {
	m := UserMessage("Hello, world!")
	if m.Role != RoleUser {
		t.Errorf("UserMessage Role = %q, want %q", m.Role, RoleUser)
	}
	if len(m.Content) != 1 {
		t.Fatalf("UserMessage Content length = %d, want 1", len(m.Content))
	}
	if m.Content[0].Kind != ContentKindText {
		t.Errorf("UserMessage Content[0].Kind = %q, want %q", m.Content[0].Kind, ContentKindText)
	}
	if m.Content[0].Text != "Hello, world!" {
		t.Errorf("UserMessage Content[0].Text = %q, want %q", m.Content[0].Text, "Hello, world!")
	}
}

func TestAssistantMessage(t *testing.T) {
	m := AssistantMessage("I can help with that.")
	if m.Role != RoleAssistant {
		t.Errorf("AssistantMessage Role = %q, want %q", m.Role, RoleAssistant)
	}
	if len(m.Content) != 1 {
		t.Fatalf("AssistantMessage Content length = %d, want 1", len(m.Content))
	}
	if m.Content[0].Kind != ContentKindText {
		t.Errorf("AssistantMessage Content[0].Kind = %q, want %q", m.Content[0].Kind, ContentKindText)
	}
	if m.Content[0].Text != "I can help with that." {
		t.Errorf("AssistantMessage Content[0].Text = %q, want %q", m.Content[0].Text, "I can help with that.")
	}
}

func TestToolResultMessage(t *testing.T) {
	tests := []struct {
		name       string
		toolCallID string
		content    string
		isError    bool
	}{
		{
			name:       "success result",
			toolCallID: "call_123",
			content:    `{"result": 42}`,
			isError:    false,
		},
		{
			name:       "error result",
			toolCallID: "call_456",
			content:    "tool execution failed",
			isError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := ToolResultMessage(tt.toolCallID, tt.content, tt.isError)
			if m.Role != RoleTool {
				t.Errorf("Role = %q, want %q", m.Role, RoleTool)
			}
			if m.ToolCallID != tt.toolCallID {
				t.Errorf("ToolCallID = %q, want %q", m.ToolCallID, tt.toolCallID)
			}
			if len(m.Content) != 1 {
				t.Fatalf("Content length = %d, want 1", len(m.Content))
			}
			part := m.Content[0]
			if part.Kind != ContentKindToolResult {
				t.Errorf("Content[0].Kind = %q, want %q", part.Kind, ContentKindToolResult)
			}
			if part.ToolResult == nil {
				t.Fatal("Content[0].ToolResult is nil")
			}
			if part.ToolResult.ToolCallID != tt.toolCallID {
				t.Errorf("ToolResult.ToolCallID = %q, want %q", part.ToolResult.ToolCallID, tt.toolCallID)
			}
			if part.ToolResult.Content != tt.content {
				t.Errorf("ToolResult.Content = %q, want %q", part.ToolResult.Content, tt.content)
			}
			if part.ToolResult.IsError != tt.isError {
				t.Errorf("ToolResult.IsError = %v, want %v", part.ToolResult.IsError, tt.isError)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Message.Text()
// ---------------------------------------------------------------------------

func TestMessageText(t *testing.T) {
	tests := []struct {
		name    string
		content []ContentPart
		want    string
	}{
		{
			name:    "empty content",
			content: nil,
			want:    "",
		},
		{
			name: "single text part",
			content: []ContentPart{
				{Kind: ContentKindText, Text: "hello"},
			},
			want: "hello",
		},
		{
			name: "multiple text parts concatenated",
			content: []ContentPart{
				{Kind: ContentKindText, Text: "hello "},
				{Kind: ContentKindText, Text: "world"},
			},
			want: "hello world",
		},
		{
			name: "non-text parts ignored",
			content: []ContentPart{
				{Kind: ContentKindText, Text: "before "},
				{Kind: ContentKindImage, Image: &ImageData{URL: "http://example.com/img.png"}},
				{Kind: ContentKindText, Text: "after"},
			},
			want: "before after",
		},
		{
			name: "only non-text parts",
			content: []ContentPart{
				{Kind: ContentKindToolCall, ToolCall: &ToolCallData{ID: "1", Name: "fn"}},
			},
			want: "",
		},
		{
			name: "text interleaved with thinking",
			content: []ContentPart{
				{Kind: ContentKindThinking, Thinking: &ThinkingData{Text: "reasoning..."}},
				{Kind: ContentKindText, Text: "answer"},
			},
			want: "answer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Message{Role: RoleAssistant, Content: tt.content}
			got := m.Text()
			if got != tt.want {
				t.Errorf("Text() = %q, want %q", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Response.Text()
// ---------------------------------------------------------------------------

func TestResponseText(t *testing.T) {
	r := Response{
		Message: Message{
			Role: RoleAssistant,
			Content: []ContentPart{
				{Kind: ContentKindText, Text: "Hello from the model."},
			},
		},
	}
	got := r.Text()
	if got != "Hello from the model." {
		t.Errorf("Response.Text() = %q, want %q", got, "Hello from the model.")
	}
}

func TestResponseTextEmpty(t *testing.T) {
	r := Response{
		Message: Message{
			Role:    RoleAssistant,
			Content: nil,
		},
	}
	got := r.Text()
	if got != "" {
		t.Errorf("Response.Text() on empty message = %q, want %q", got, "")
	}
}

// ---------------------------------------------------------------------------
// Response.ToolCalls()
// ---------------------------------------------------------------------------

func TestResponseToolCalls(t *testing.T) {
	tests := []struct {
		name    string
		content []ContentPart
		want    int
	}{
		{
			name:    "no tool calls",
			content: []ContentPart{{Kind: ContentKindText, Text: "hi"}},
			want:    0,
		},
		{
			name: "single tool call",
			content: []ContentPart{
				{Kind: ContentKindToolCall, ToolCall: &ToolCallData{ID: "c1", Name: "search"}},
			},
			want: 1,
		},
		{
			name: "multiple tool calls mixed with text",
			content: []ContentPart{
				{Kind: ContentKindText, Text: "thinking..."},
				{Kind: ContentKindToolCall, ToolCall: &ToolCallData{ID: "c1", Name: "search"}},
				{Kind: ContentKindToolCall, ToolCall: &ToolCallData{ID: "c2", Name: "calc"}},
			},
			want: 2,
		},
		{
			name: "tool call kind but nil pointer skipped",
			content: []ContentPart{
				{Kind: ContentKindToolCall, ToolCall: nil},
			},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := Response{
				Message: Message{Role: RoleAssistant, Content: tt.content},
			}
			calls := r.ToolCalls()
			if len(calls) != tt.want {
				t.Errorf("ToolCalls() returned %d calls, want %d", len(calls), tt.want)
			}
		})
	}
}

func TestResponseToolCallsData(t *testing.T) {
	args := map[string]any{"query": "weather"}
	r := Response{
		Message: Message{
			Role: RoleAssistant,
			Content: []ContentPart{
				{Kind: ContentKindToolCall, ToolCall: &ToolCallData{
					ID:        "call_abc",
					Name:      "search",
					Arguments: args,
					Type:      "function",
				}},
			},
		},
	}
	calls := r.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	tc := calls[0]
	if tc.ID != "call_abc" {
		t.Errorf("ToolCall.ID = %q, want %q", tc.ID, "call_abc")
	}
	if tc.Name != "search" {
		t.Errorf("ToolCall.Name = %q, want %q", tc.Name, "search")
	}
	if tc.Arguments["query"] != "weather" {
		t.Errorf("ToolCall.Arguments[query] = %v, want %q", tc.Arguments["query"], "weather")
	}
	if tc.Type != "function" {
		t.Errorf("ToolCall.Type = %q, want %q", tc.Type, "function")
	}
}

// ---------------------------------------------------------------------------
// Response.Reasoning()
// ---------------------------------------------------------------------------

func TestResponseReasoning(t *testing.T) {
	tests := []struct {
		name    string
		content []ContentPart
		want    string
	}{
		{
			name:    "no reasoning",
			content: []ContentPart{{Kind: ContentKindText, Text: "answer"}},
			want:    "",
		},
		{
			name: "single thinking block",
			content: []ContentPart{
				{Kind: ContentKindThinking, Thinking: &ThinkingData{Text: "Let me think..."}},
				{Kind: ContentKindText, Text: "answer"},
			},
			want: "Let me think...",
		},
		{
			name: "multiple thinking blocks concatenated",
			content: []ContentPart{
				{Kind: ContentKindThinking, Thinking: &ThinkingData{Text: "Step 1. "}},
				{Kind: ContentKindText, Text: "interim"},
				{Kind: ContentKindThinking, Thinking: &ThinkingData{Text: "Step 2."}},
			},
			want: "Step 1. Step 2.",
		},
		{
			name: "redacted thinking included",
			content: []ContentPart{
				{Kind: ContentKindRedactedThinking, Thinking: &ThinkingData{Text: "redacted content", Redacted: true}},
			},
			want: "redacted content",
		},
		{
			name: "thinking kind with nil pointer skipped",
			content: []ContentPart{
				{Kind: ContentKindThinking, Thinking: nil},
			},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := Response{
				Message: Message{Role: RoleAssistant, Content: tt.content},
			}
			got := r.Reasoning()
			if got != tt.want {
				t.Errorf("Reasoning() = %q, want %q", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Usage.Add()
// ---------------------------------------------------------------------------

func intPtr(v int) *int { return &v }

func TestUsageAdd(t *testing.T) {
	tests := []struct {
		name               string
		a, b               Usage
		wantInput          int
		wantOutput         int
		wantTotal          int
		wantReasoningNil   bool
		wantReasoning      int
		wantCacheReadNil   bool
		wantCacheRead      int
		wantCacheWriteNil  bool
		wantCacheWrite     int
	}{
		{
			name:              "basic addition",
			a:                 Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
			b:                 Usage{InputTokens: 5, OutputTokens: 15, TotalTokens: 20},
			wantInput:         15,
			wantOutput:        35,
			wantTotal:         50,
			wantReasoningNil:  true,
			wantCacheReadNil:  true,
			wantCacheWriteNil: true,
		},
		{
			name:              "both have optional fields",
			a:                 Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30, ReasoningTokens: intPtr(100), CacheReadTokens: intPtr(50)},
			b:                 Usage{InputTokens: 5, OutputTokens: 15, TotalTokens: 20, ReasoningTokens: intPtr(200), CacheReadTokens: intPtr(25)},
			wantInput:         15,
			wantOutput:        35,
			wantTotal:         50,
			wantReasoningNil:  false,
			wantReasoning:     300,
			wantCacheReadNil:  false,
			wantCacheRead:     75,
			wantCacheWriteNil: true,
		},
		{
			name:              "one side has optional field",
			a:                 Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30, ReasoningTokens: intPtr(100)},
			b:                 Usage{InputTokens: 5, OutputTokens: 15, TotalTokens: 20},
			wantInput:         15,
			wantOutput:        35,
			wantTotal:         50,
			wantReasoningNil:  false,
			wantReasoning:     100,
			wantCacheReadNil:  true,
			wantCacheWriteNil: true,
		},
		{
			name:              "other side has optional field",
			a:                 Usage{InputTokens: 0, OutputTokens: 0, TotalTokens: 0},
			b:                 Usage{InputTokens: 0, OutputTokens: 0, TotalTokens: 0, CacheWriteTokens: intPtr(42)},
			wantInput:         0,
			wantOutput:        0,
			wantTotal:         0,
			wantReasoningNil:  true,
			wantCacheReadNil:  true,
			wantCacheWriteNil: false,
			wantCacheWrite:    42,
		},
		{
			name:              "zero values",
			a:                 Usage{},
			b:                 Usage{},
			wantInput:         0,
			wantOutput:        0,
			wantTotal:         0,
			wantReasoningNil:  true,
			wantCacheReadNil:  true,
			wantCacheWriteNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.a.Add(tt.b)

			if result.InputTokens != tt.wantInput {
				t.Errorf("InputTokens = %d, want %d", result.InputTokens, tt.wantInput)
			}
			if result.OutputTokens != tt.wantOutput {
				t.Errorf("OutputTokens = %d, want %d", result.OutputTokens, tt.wantOutput)
			}
			if result.TotalTokens != tt.wantTotal {
				t.Errorf("TotalTokens = %d, want %d", result.TotalTokens, tt.wantTotal)
			}

			if tt.wantReasoningNil {
				if result.ReasoningTokens != nil {
					t.Errorf("ReasoningTokens = %v, want nil", *result.ReasoningTokens)
				}
			} else {
				if result.ReasoningTokens == nil {
					t.Fatal("ReasoningTokens is nil, want non-nil")
				}
				if *result.ReasoningTokens != tt.wantReasoning {
					t.Errorf("ReasoningTokens = %d, want %d", *result.ReasoningTokens, tt.wantReasoning)
				}
			}

			if tt.wantCacheReadNil {
				if result.CacheReadTokens != nil {
					t.Errorf("CacheReadTokens = %v, want nil", *result.CacheReadTokens)
				}
			} else {
				if result.CacheReadTokens == nil {
					t.Fatal("CacheReadTokens is nil, want non-nil")
				}
				if *result.CacheReadTokens != tt.wantCacheRead {
					t.Errorf("CacheReadTokens = %d, want %d", *result.CacheReadTokens, tt.wantCacheRead)
				}
			}

			if tt.wantCacheWriteNil {
				if result.CacheWriteTokens != nil {
					t.Errorf("CacheWriteTokens = %v, want nil", *result.CacheWriteTokens)
				}
			} else {
				if result.CacheWriteTokens == nil {
					t.Fatal("CacheWriteTokens is nil, want non-nil")
				}
				if *result.CacheWriteTokens != tt.wantCacheWrite {
					t.Errorf("CacheWriteTokens = %d, want %d", *result.CacheWriteTokens, tt.wantCacheWrite)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// FinishReason values
// ---------------------------------------------------------------------------

func TestFinishReasonValues(t *testing.T) {
	tests := []struct {
		name   string
		fr     FinishReason
		reason string
	}{
		{"stop", FinishReasonStop, "stop"},
		{"length", FinishReasonLength, "length"},
		{"tool_calls", FinishReasonToolCalls, "tool_calls"},
		{"content_filter", FinishReasonContentFilter, "content_filter"},
		{"error", FinishReasonError, "error"},
		{"other", FinishReasonOther, "other"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fr.Reason != tt.reason {
				t.Errorf("FinishReason.Reason = %q, want %q", tt.fr.Reason, tt.reason)
			}
		})
	}
}

func TestFinishReasonWithRaw(t *testing.T) {
	fr := FinishReason{Reason: "stop", Raw: "end_turn"}
	if fr.Reason != "stop" {
		t.Errorf("Reason = %q, want %q", fr.Reason, "stop")
	}
	if fr.Raw != "end_turn" {
		t.Errorf("Raw = %q, want %q", fr.Raw, "end_turn")
	}
}

// ---------------------------------------------------------------------------
// ContentPart tagged union
// ---------------------------------------------------------------------------

func TestContentPartTaggedUnion(t *testing.T) {
	tests := []struct {
		name string
		part ContentPart
		kind ContentKind
	}{
		{
			name: "text part",
			part: ContentPart{Kind: ContentKindText, Text: "hello"},
			kind: ContentKindText,
		},
		{
			name: "image part",
			part: ContentPart{Kind: ContentKindImage, Image: &ImageData{URL: "http://example.com/img.png", MediaType: "image/png"}},
			kind: ContentKindImage,
		},
		{
			name: "audio part",
			part: ContentPart{Kind: ContentKindAudio, Audio: &AudioData{MediaType: "audio/wav", Data: []byte("raw")}},
			kind: ContentKindAudio,
		},
		{
			name: "document part",
			part: ContentPart{Kind: ContentKindDocument, Document: &DocumentData{FileName: "doc.pdf", MediaType: "application/pdf"}},
			kind: ContentKindDocument,
		},
		{
			name: "tool call part",
			part: ContentPart{Kind: ContentKindToolCall, ToolCall: &ToolCallData{ID: "c1", Name: "fn", Arguments: map[string]any{"x": 1.0}}},
			kind: ContentKindToolCall,
		},
		{
			name: "tool result part",
			part: ContentPart{Kind: ContentKindToolResult, ToolResult: &ToolResultData{ToolCallID: "c1", Content: "ok"}},
			kind: ContentKindToolResult,
		},
		{
			name: "thinking part",
			part: ContentPart{Kind: ContentKindThinking, Thinking: &ThinkingData{Text: "hmm"}},
			kind: ContentKindThinking,
		},
		{
			name: "redacted thinking part",
			part: ContentPart{Kind: ContentKindRedactedThinking, Thinking: &ThinkingData{Redacted: true, Signature: "sig123"}},
			kind: ContentKindRedactedThinking,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.part.Kind != tt.kind {
				t.Errorf("ContentPart.Kind = %q, want %q", tt.part.Kind, tt.kind)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Tool.Definition()
// ---------------------------------------------------------------------------

func TestToolDefinition(t *testing.T) {
	tool := Tool{
		Name:        "search",
		Description: "Search the web",
		Parameters:  map[string]any{"type": "object"},
		Execute:     func(args map[string]any) (string, error) { return "result", nil },
	}
	def := tool.Definition()
	if def.Name != "search" {
		t.Errorf("Definition().Name = %q, want %q", def.Name, "search")
	}
	if def.Description != "Search the web" {
		t.Errorf("Definition().Description = %q, want %q", def.Description, "Search the web")
	}
	if def.Parameters["type"] != "object" {
		t.Errorf("Definition().Parameters[type] = %v, want %q", def.Parameters["type"], "object")
	}
}

// ---------------------------------------------------------------------------
// Role constants
// ---------------------------------------------------------------------------

func TestRoleConstants(t *testing.T) {
	tests := []struct {
		role Role
		want string
	}{
		{RoleSystem, "system"},
		{RoleUser, "user"},
		{RoleAssistant, "assistant"},
		{RoleTool, "tool"},
		{RoleDeveloper, "developer"},
	}
	for _, tt := range tests {
		if string(tt.role) != tt.want {
			t.Errorf("Role %q != %q", tt.role, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// ContentKind constants
// ---------------------------------------------------------------------------

func TestContentKindConstants(t *testing.T) {
	tests := []struct {
		kind ContentKind
		want string
	}{
		{ContentKindText, "text"},
		{ContentKindImage, "image"},
		{ContentKindAudio, "audio"},
		{ContentKindDocument, "document"},
		{ContentKindToolCall, "tool_call"},
		{ContentKindToolResult, "tool_result"},
		{ContentKindThinking, "thinking"},
		{ContentKindRedactedThinking, "redacted_thinking"},
	}
	for _, tt := range tests {
		if string(tt.kind) != tt.want {
			t.Errorf("ContentKind %q != %q", tt.kind, tt.want)
		}
	}
}
