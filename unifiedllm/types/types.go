// Package types defines the core data structures for the Unified LLM Client.
//
// These types provide a provider-agnostic interface for interacting with large
// language models. All major LLM providers (OpenAI, Anthropic, Google, etc.)
// can be mapped onto these shared types, enabling application code to remain
// decoupled from any single provider's API surface.
package types

import (
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

// Role represents the role of a message participant in a conversation.
type Role string

const (
	// RoleSystem is the system prompt role, used to set context and behavior.
	RoleSystem Role = "system"
	// RoleUser represents a human user message.
	RoleUser Role = "user"
	// RoleAssistant represents an assistant (model) response.
	RoleAssistant Role = "assistant"
	// RoleTool represents the result of a tool invocation.
	RoleTool Role = "tool"
	// RoleDeveloper is an alternative to system used by some providers.
	RoleDeveloper Role = "developer"
)

// ---------------------------------------------------------------------------
// ContentKind
// ---------------------------------------------------------------------------

// ContentKind discriminates the type of data carried by a ContentPart.
type ContentKind string

const (
	ContentKindText             ContentKind = "text"
	ContentKindImage            ContentKind = "image"
	ContentKindAudio            ContentKind = "audio"
	ContentKindDocument         ContentKind = "document"
	ContentKindToolCall         ContentKind = "tool_call"
	ContentKindToolResult       ContentKind = "tool_result"
	ContentKindThinking         ContentKind = "thinking"
	ContentKindRedactedThinking ContentKind = "redacted_thinking"
)

// ---------------------------------------------------------------------------
// ContentPart and associated data types
// ---------------------------------------------------------------------------

// ContentPart is a tagged union representing one piece of message content.
// Exactly one of the pointer fields should be non-nil, corresponding to Kind.
type ContentPart struct {
	Kind       ContentKind     `json:"kind"`
	Text       string          `json:"text,omitempty"`
	Image      *ImageData      `json:"image,omitempty"`
	Audio      *AudioData      `json:"audio,omitempty"`
	Document   *DocumentData   `json:"document,omitempty"`
	ToolCall   *ToolCallData   `json:"tool_call,omitempty"`
	ToolResult *ToolResultData `json:"tool_result,omitempty"`
	Thinking   *ThinkingData   `json:"thinking,omitempty"`
}

// ImageData holds the payload for an image content part.
type ImageData struct {
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// AudioData holds the payload for an audio content part.
type AudioData struct {
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

// DocumentData holds the payload for a document content part (e.g. PDF).
type DocumentData struct {
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	FileName  string `json:"file_name,omitempty"`
}

// ToolCallData describes a tool invocation requested by the model.
type ToolCallData struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
	Type      string         `json:"type,omitempty"`
}

// ToolResultData holds the output returned from a tool execution.
type ToolResultData struct {
	ToolCallID     string `json:"tool_call_id"`
	Content        string `json:"content"`
	IsError        bool   `json:"is_error,omitempty"`
	ImageData      []byte `json:"image_data,omitempty"`
	ImageMediaType string `json:"image_media_type,omitempty"`
}

// ThinkingData captures model chain-of-thought or reasoning traces.
type ThinkingData struct {
	Text      string `json:"text,omitempty"`
	Signature string `json:"signature,omitempty"`
	Redacted  bool   `json:"redacted,omitempty"`
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

// Message represents a single turn in a conversation.
type Message struct {
	Role       Role          `json:"role"`
	Content    []ContentPart `json:"content"`
	Name       string        `json:"name,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// Text returns the concatenation of all text content parts in the message.
// Non-text parts are silently ignored. Returns an empty string when the
// message contains no text parts.
func (m Message) Text() string {
	var b strings.Builder
	for _, p := range m.Content {
		if p.Kind == ContentKindText {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// SystemMessage creates a Message with the system role and a single text part.
func SystemMessage(text string) Message {
	return Message{
		Role: RoleSystem,
		Content: []ContentPart{
			{Kind: ContentKindText, Text: text},
		},
	}
}

// UserMessage creates a Message with the user role and a single text part.
func UserMessage(text string) Message {
	return Message{
		Role: RoleUser,
		Content: []ContentPart{
			{Kind: ContentKindText, Text: text},
		},
	}
}

// AssistantMessage creates a Message with the assistant role and a single text part.
func AssistantMessage(text string) Message {
	return Message{
		Role: RoleAssistant,
		Content: []ContentPart{
			{Kind: ContentKindText, Text: text},
		},
	}
}

// ToolResultMessage creates a Message with the tool role containing a single
// tool result content part.
func ToolResultMessage(toolCallID, content string, isError bool) Message {
	return Message{
		Role:       RoleTool,
		ToolCallID: toolCallID,
		Content: []ContentPart{
			{
				Kind: ContentKindToolResult,
				ToolResult: &ToolResultData{
					ToolCallID: toolCallID,
					Content:    content,
					IsError:    isError,
				},
			},
		},
	}
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

// Request represents a complete chat-completion request to an LLM provider.
type Request struct {
	// Model is the provider-specific model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
	Model string `json:"model"`

	// Messages is the conversation history.
	Messages []Message `json:"messages"`

	// Provider optionally pins the request to a specific provider backend.
	Provider string `json:"provider,omitempty"`

	// Tools lists the tool definitions the model may invoke.
	Tools []ToolDefinition `json:"tools,omitempty"`

	// ToolChoice controls how the model selects tools.
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// ResponseFormat constrains the output structure.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Temperature controls randomness. Nil means provider default.
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP controls nucleus sampling. Nil means provider default.
	TopP *float64 `json:"top_p,omitempty"`

	// MaxTokens limits the response length. Nil means provider default.
	MaxTokens *int `json:"max_tokens,omitempty"`

	// StopSequences are sequences that cause the model to stop generating.
	StopSequences []string `json:"stop_sequences,omitempty"`

	// ReasoningEffort hints to providers that support thinking/reasoning
	// (e.g. "low", "medium", "high").
	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	// Metadata is arbitrary key-value data forwarded with the request.
	Metadata map[string]string `json:"metadata,omitempty"`

	// ProviderOptions carries provider-specific parameters that do not map
	// onto the unified schema.
	ProviderOptions map[string]any `json:"provider_options,omitempty"`
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

// Response represents the result of a chat-completion request.
type Response struct {
	// ID is the provider-assigned response identifier.
	ID string `json:"id"`

	// Model is the model that actually served the request.
	Model string `json:"model"`

	// Provider is the provider that served the request.
	Provider string `json:"provider"`

	// Message contains the assistant's reply.
	Message Message `json:"message"`

	// FinishReason indicates why generation stopped.
	FinishReason FinishReason `json:"finish_reason"`

	// Usage contains token consumption statistics.
	Usage Usage `json:"usage"`

	// Raw is the unprocessed provider response for debugging or passthrough.
	Raw map[string]any `json:"raw,omitempty"`

	// Warnings contains non-fatal issues encountered during the request.
	Warnings []Warning `json:"warnings,omitempty"`

	// RateLimit contains rate-limit information returned by the provider.
	RateLimit *RateLimitInfo `json:"rate_limit,omitempty"`
}

// Text is a convenience accessor that returns the concatenated text content
// of the response message.
func (r Response) Text() string {
	return r.Message.Text()
}

// ToolCalls extracts all tool-call content parts from the response message.
func (r Response) ToolCalls() []ToolCallData {
	var calls []ToolCallData
	for _, p := range r.Message.Content {
		if p.Kind == ContentKindToolCall && p.ToolCall != nil {
			calls = append(calls, *p.ToolCall)
		}
	}
	return calls
}

// Reasoning extracts and concatenates all thinking/reasoning content from
// the response message.
func (r Response) Reasoning() string {
	var b strings.Builder
	for _, p := range r.Message.Content {
		if (p.Kind == ContentKindThinking || p.Kind == ContentKindRedactedThinking) && p.Thinking != nil {
			b.WriteString(p.Thinking.Text)
		}
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

// FinishReason describes why the model stopped generating tokens.
type FinishReason struct {
	// Reason is the normalised finish reason.
	Reason string `json:"reason"`
	// Raw is the original provider-specific string, preserved for debugging.
	Raw string `json:"raw,omitempty"`
}

// Pre-defined normalised finish reasons.
var (
	FinishReasonStop          = FinishReason{Reason: "stop"}
	FinishReasonLength        = FinishReason{Reason: "length"}
	FinishReasonToolCalls     = FinishReason{Reason: "tool_calls"}
	FinishReasonContentFilter = FinishReason{Reason: "content_filter"}
	FinishReasonError         = FinishReason{Reason: "error"}
	FinishReasonOther         = FinishReason{Reason: "other"}
)

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

// Usage captures token consumption for a request/response cycle.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`

	// Optional extended token counts (nil when not reported by the provider).
	ReasoningTokens  *int `json:"reasoning_tokens,omitempty"`
	CacheReadTokens  *int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens *int `json:"cache_write_tokens,omitempty"`

	// Raw is the unprocessed usage payload from the provider.
	Raw map[string]any `json:"raw,omitempty"`
}

// Add returns a new Usage that is the sum of u and other. Optional fields
// are summed when present in either operand; if only one side has a value,
// that value is carried through.
func (u Usage) Add(other Usage) Usage {
	result := Usage{
		InputTokens:  u.InputTokens + other.InputTokens,
		OutputTokens: u.OutputTokens + other.OutputTokens,
		TotalTokens:  u.TotalTokens + other.TotalTokens,
	}
	result.ReasoningTokens = addOptionalInt(u.ReasoningTokens, other.ReasoningTokens)
	result.CacheReadTokens = addOptionalInt(u.CacheReadTokens, other.CacheReadTokens)
	result.CacheWriteTokens = addOptionalInt(u.CacheWriteTokens, other.CacheWriteTokens)
	return result
}

// addOptionalInt sums two optional int pointers. Returns nil only when both
// inputs are nil. When one side is nil it is treated as zero.
func addOptionalInt(a, b *int) *int {
	if a == nil && b == nil {
		return nil
	}
	sum := 0
	if a != nil {
		sum += *a
	}
	if b != nil {
		sum += *b
	}
	return &sum
}

// ---------------------------------------------------------------------------
// Tool definitions and choices
// ---------------------------------------------------------------------------

// ToolDefinition describes a tool that the model may invoke.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolChoice controls the model's tool-selection behaviour.
type ToolChoice struct {
	// Mode is one of "auto", "none", "required", or "named".
	Mode string `json:"mode"`
	// ToolName is required when Mode is "named".
	ToolName string `json:"tool_name,omitempty"`
}

// ---------------------------------------------------------------------------
// ResponseFormat
// ---------------------------------------------------------------------------

// ResponseFormat constrains the structure of the model's output.
type ResponseFormat struct {
	// Type is one of "text", "json", or "json_schema".
	Type string `json:"type"`
	// JSONSchema is the schema the output must conform to (used when Type is "json_schema").
	JSONSchema map[string]any `json:"json_schema,omitempty"`
	// Strict, when true, requires the output to strictly match the schema.
	Strict bool `json:"strict,omitempty"`
}

// ---------------------------------------------------------------------------
// Warning
// ---------------------------------------------------------------------------

// Warning represents a non-fatal issue encountered during a request.
type Warning struct {
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
}

// ---------------------------------------------------------------------------
// RateLimitInfo
// ---------------------------------------------------------------------------

// RateLimitInfo contains rate-limit metadata returned by the provider.
type RateLimitInfo struct {
	RequestsRemaining *int       `json:"requests_remaining,omitempty"`
	RequestsLimit     *int       `json:"requests_limit,omitempty"`
	TokensRemaining   *int       `json:"tokens_remaining,omitempty"`
	TokensLimit       *int       `json:"tokens_limit,omitempty"`
	ResetAt           *time.Time `json:"reset_at,omitempty"`
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// StreamEventType discriminates the kind of event received on a stream.
type StreamEventType string

const (
	StreamEventStreamStart    StreamEventType = "stream_start"
	StreamEventTextStart      StreamEventType = "text_start"
	StreamEventTextDelta      StreamEventType = "text_delta"
	StreamEventTextEnd        StreamEventType = "text_end"
	StreamEventReasoningStart StreamEventType = "reasoning_start"
	StreamEventReasoningDelta StreamEventType = "reasoning_delta"
	StreamEventReasoningEnd   StreamEventType = "reasoning_end"
	StreamEventToolCallStart  StreamEventType = "tool_call_start"
	StreamEventToolCallDelta  StreamEventType = "tool_call_delta"
	StreamEventToolCallEnd    StreamEventType = "tool_call_end"
	StreamEventFinish         StreamEventType = "finish"
	StreamEventError          StreamEventType = "error"
	StreamEventProviderEvent  StreamEventType = "provider_event"
)

// StreamEvent is a single event emitted during a streaming response.
type StreamEvent struct {
	Type           StreamEventType `json:"type"`
	Delta          string          `json:"delta,omitempty"`
	TextID         string          `json:"text_id,omitempty"`
	ReasoningDelta string          `json:"reasoning_delta,omitempty"`
	ToolCall       *ToolCallData   `json:"tool_call,omitempty"`
	FinishReason   *FinishReason   `json:"finish_reason,omitempty"`
	Usage          *Usage          `json:"usage,omitempty"`
	Response       *Response       `json:"response,omitempty"`
	Error          error           `json:"-"`
	Raw            map[string]any  `json:"raw,omitempty"`
}

// ---------------------------------------------------------------------------
// High-level Tool (with execute handler)
// ---------------------------------------------------------------------------

// Tool is a high-level tool definition that pairs a schema with an execution
// handler. Use this when building agentic loops that automatically dispatch
// tool calls.
type Tool struct {
	Name        string                                    `json:"name"`
	Description string                                    `json:"description"`
	Parameters  map[string]any                            `json:"parameters,omitempty"`
	Execute     func(args map[string]any) (string, error) `json:"-"`
}

// Definition converts a Tool into a ToolDefinition suitable for inclusion in
// a Request.
func (t Tool) Definition() ToolDefinition {
	return ToolDefinition{
		Name:        t.Name,
		Description: t.Description,
		Parameters:  t.Parameters,
	}
}

// ---------------------------------------------------------------------------
// Extracted ToolCall / ToolResult (convenience wrappers)
// ---------------------------------------------------------------------------

// ToolCall is a convenience type representing a single tool invocation
// extracted from a response.
type ToolCall struct {
	ID           string         `json:"id"`
	Name         string         `json:"name"`
	Arguments    map[string]any `json:"arguments,omitempty"`
	RawArguments string         `json:"raw_arguments,omitempty"`
}

// ToolResult is a convenience type representing the output of executing a
// tool call. Content is typed as any to support structured data as well as
// plain strings.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    any    `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

// ---------------------------------------------------------------------------
// ModelInfo (catalog entry)
// ---------------------------------------------------------------------------

// ModelInfo describes a model available through the provider catalog.
type ModelInfo struct {
	ID                   string   `json:"id"`
	Provider             string   `json:"provider"`
	DisplayName          string   `json:"display_name"`
	ContextWindow        int      `json:"context_window"`
	MaxOutput            int      `json:"max_output"`
	SupportsTools        bool     `json:"supports_tools"`
	SupportsVision       bool     `json:"supports_vision"`
	SupportsReasoning    bool     `json:"supports_reasoning"`
	InputCostPerMillion  *float64 `json:"input_cost_per_million,omitempty"`
	OutputCostPerMillion *float64 `json:"output_cost_per_million,omitempty"`
	Aliases              []string `json:"aliases,omitempty"`
}
