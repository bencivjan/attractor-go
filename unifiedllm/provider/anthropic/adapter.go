// Package anthropic implements the ProviderAdapter for the Anthropic Messages API.
//
// This adapter translates unified types to the Anthropic Messages API format
// (/v1/messages) and back. It handles:
//   - System messages extracted to the top-level "system" parameter
//   - Strict user/assistant alternation with automatic merging
//   - Tool results in user messages with tool_result content blocks
//   - Thinking blocks round-tripped with signatures
//   - max_tokens always set (default 4096)
//   - Beta headers via provider_options
//   - Cache control annotations for prompt caching
//   - Extended thinking with budget_tokens
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/strongdm/attractor-go/unifiedllm/sse"
	"github.com/strongdm/attractor-go/unifiedllm/types"
)

const (
	defaultBaseURL    = "https://api.anthropic.com"
	providerName      = "anthropic"
	defaultMaxTokens  = 4096
	anthropicVersion  = "2023-06-01"
)

// Adapter implements ProviderAdapter for the Anthropic Messages API.
type Adapter struct {
	apiKey             string
	baseURL            string
	httpClient         *http.Client
	headers            map[string]string
	streamReadTimeout  time.Duration
}

// AdapterOption configures an Adapter.
type AdapterOption func(*Adapter)

// WithAPIKey sets the API key for authentication.
func WithAPIKey(key string) AdapterOption {
	return func(a *Adapter) {
		a.apiKey = key
	}
}

// WithBaseURL overrides the default API base URL.
func WithBaseURL(url string) AdapterOption {
	return func(a *Adapter) {
		a.baseURL = strings.TrimRight(url, "/")
	}
}

// WithHTTPClient provides a custom HTTP client for requests.
func WithHTTPClient(c *http.Client) AdapterOption {
	return func(a *Adapter) {
		a.httpClient = c
	}
}

// WithAdapterTimeout configures per-phase timeouts (connect, request,
// stream_read) by building an appropriately configured HTTP client.
func WithAdapterTimeout(t types.AdapterTimeout) AdapterOption {
	return func(a *Adapter) {
		a.httpClient = t.HTTPClient()
		a.streamReadTimeout = t.StreamRead
	}
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(headers map[string]string) AdapterOption {
	return func(a *Adapter) {
		a.headers = headers
	}
}

// New creates an Anthropic adapter with the given options.
func New(opts ...AdapterOption) *Adapter {
	a := &Adapter{
		baseURL:    defaultBaseURL,
		httpClient: http.DefaultClient,
		headers:    make(map[string]string),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// FromEnv creates an Adapter configured from environment variables:
//   - ANTHROPIC_API_KEY (required)
//   - ANTHROPIC_BASE_URL (optional, defaults to https://api.anthropic.com)
func FromEnv() (*Adapter, error) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, types.NewConfigurationError("ANTHROPIC_API_KEY environment variable is not set", nil)
	}

	opts := []AdapterOption{WithAPIKey(key)}

	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}

	return New(opts...), nil
}

// Name returns the canonical provider name.
func (a *Adapter) Name() string {
	return providerName
}

// Complete sends a blocking completion request using the Messages API.
func (a *Adapter) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	body, err := a.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/messages", bytes.NewReader(payload))
	if err != nil {
		return nil, types.NewSDKError("failed to create HTTP request", err)
	}

	a.setHeaders(httpReq, req.ProviderOptions)

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return nil, types.NewAbortError("request cancelled", ctx.Err())
		}
		return nil, types.NewNetworkError("request failed", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewNetworkError("failed to read response body", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, a.parseError(resp.StatusCode, respBody, resp.Header)
	}

	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return nil, types.NewSDKError("failed to parse response JSON", err)
	}

	result, err := a.parseResponse(raw)
	if err != nil {
		return nil, err
	}

	result.RateLimit = parseRateLimitHeaders(resp.Header)
	return result, nil
}

// Stream sends a streaming request using the Messages API with stream=true.
func (a *Adapter) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	body, err := a.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/messages", bytes.NewReader(payload))
	if err != nil {
		return nil, types.NewSDKError("failed to create HTTP request", err)
	}

	a.setHeaders(httpReq, req.ProviderOptions)

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return nil, types.NewAbortError("request cancelled", ctx.Err())
		}
		return nil, types.NewNetworkError("stream request failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, types.NewNetworkError("failed to read error response", readErr)
		}
		return nil, a.parseError(resp.StatusCode, respBody, resp.Header)
	}

	ch := make(chan types.StreamEvent, 64)

	go func() {
		defer close(ch)
		defer resp.Body.Close()

		a.processStream(ctx, resp.Body, ch)
	}()

	return ch, nil
}

// Close releases adapter resources.
func (a *Adapter) Close() error {
	return nil
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

// autoCacheEnabled checks whether automatic cache_control injection is enabled.
// It defaults to true unless provider_options["anthropic"]["auto_cache"] is
// explicitly set to false.
func autoCacheEnabled(providerOptions map[string]any) bool {
	if providerOptions == nil {
		return true
	}
	anthropicOpts, ok := providerOptions["anthropic"].(map[string]any)
	if !ok {
		return true
	}
	if autoCache, ok := anthropicOpts["auto_cache"].(bool); ok {
		return autoCache
	}
	return true
}

// buildRequestBody translates a unified Request into the Anthropic Messages API format.
func (a *Adapter) buildRequestBody(req types.Request, stream bool) (map[string]any, error) {
	body := map[string]any{
		"model": req.Model,
	}

	cacheEnabled := autoCacheEnabled(req.ProviderOptions)

	// Extract system messages.
	var systemParts []map[string]any
	var messages []types.Message

	for _, msg := range req.Messages {
		if msg.Role == types.RoleSystem || msg.Role == types.RoleDeveloper {
			systemParts = append(systemParts, map[string]any{
				"type": "text",
				"text": msg.Text(),
			})
		} else {
			messages = append(messages, msg)
		}
	}

	if len(systemParts) > 0 {
		if cacheEnabled {
			// Inject cache_control on the last system block for prompt caching.
			// This allows Anthropic to cache the system prompt across turns,
			// reducing input token costs by up to 90% for agentic workloads.
			systemParts[len(systemParts)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		body["system"] = systemParts
	}

	// Build messages with alternation enforcement.
	apiMessages, err := a.buildMessages(messages, cacheEnabled)
	if err != nil {
		return nil, err
	}
	body["messages"] = apiMessages

	// max_tokens is required by Anthropic. Use the request value or default.
	maxTokens := defaultMaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	body["max_tokens"] = maxTokens

	// Tools
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, tool := range req.Tools {
			t := map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
			}
			if tool.Parameters != nil {
				t["input_schema"] = tool.Parameters
			} else {
				// Anthropic requires input_schema even if empty.
				t["input_schema"] = map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				}
			}
			tools = append(tools, t)
		}
		// Inject cache_control on the last tool definition for prompt caching.
		// This allows Anthropic to cache the tool definitions across turns,
		// complementing the system prompt and conversation caching.
		if cacheEnabled && len(tools) > 0 {
			tools[len(tools)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		body["tools"] = tools
	}

	// Tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case "auto":
			body["tool_choice"] = map[string]any{"type": "auto"}
		case "none":
			// Anthropic does not have a "none" tool_choice; omit tools instead.
			// But if tools are defined and the caller wants none, we use "auto"
			// and let the model decide. Alternatively, remove tools.
			delete(body, "tools")
		case "required":
			body["tool_choice"] = map[string]any{"type": "any"}
		case "named":
			body["tool_choice"] = map[string]any{
				"type": "tool",
				"name": req.ToolChoice.ToolName,
			}
		default:
			return nil, types.NewUnsupportedToolChoiceError(
				fmt.Sprintf("Anthropic does not support tool_choice mode %q", req.ToolChoice.Mode),
				req.ToolChoice.Mode, nil,
			)
		}
	}

	// Temperature
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}

	// TopP
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}

	// Stop sequences
	if len(req.StopSequences) > 0 {
		body["stop_sequences"] = req.StopSequences
	}

	// Reasoning / thinking
	if req.ReasoningEffort != "" {
		thinking := map[string]any{
			"type": "enabled",
		}
		// Map effort level to budget_tokens.
		switch req.ReasoningEffort {
		case "low":
			thinking["budget_tokens"] = 1024
		case "medium":
			thinking["budget_tokens"] = 4096
		case "high":
			thinking["budget_tokens"] = 16384
		default:
			thinking["budget_tokens"] = 4096
		}
		body["thinking"] = thinking
	}

	// Metadata
	if len(req.Metadata) > 0 {
		body["metadata"] = map[string]any{
			"user_id": req.Metadata["user_id"],
		}
	}

	// Stream
	if stream {
		body["stream"] = true
	}

	// Pass through provider-specific options (except beta headers which go in HTTP headers).
	for k, v := range req.ProviderOptions {
		if k == "anthropic_beta" || k == "beta_headers" || k == "anthropic" {
			continue
		}
		body[k] = v
	}

	return body, nil
}

// buildMessages enforces Anthropic's strict user/assistant alternation by
// merging consecutive same-role messages. Tool results are placed in user
// messages with tool_result content blocks. When cacheEnabled is true,
// cache_control breakpoints are injected on the conversation prefix.
func (a *Adapter) buildMessages(messages []types.Message, cacheEnabled bool) ([]map[string]any, error) {
	if len(messages) == 0 {
		return nil, nil
	}

	var result []map[string]any

	for _, msg := range messages {
		var apiMsg map[string]any

		switch msg.Role {
		case types.RoleUser:
			apiMsg = a.buildUserMessage(msg)

		case types.RoleAssistant:
			apiMsg = a.buildAssistantMessage(msg)

		case types.RoleTool:
			// Tool results become user messages with tool_result blocks.
			apiMsg = a.buildToolResultMessage(msg)
		}

		if apiMsg == nil {
			continue
		}

		// Enforce alternation: merge consecutive same-role messages.
		if len(result) > 0 {
			lastRole, _ := result[len(result)-1]["role"].(string)
			thisRole, _ := apiMsg["role"].(string)
			if lastRole == thisRole {
				result[len(result)-1] = mergeMessages(result[len(result)-1], apiMsg)
				continue
			}
		}

		result = append(result, apiMsg)
	}

	// Anthropic requires the first message to be from the user.
	if len(result) > 0 {
		if role, _ := result[0]["role"].(string); role != "user" {
			// Prepend an empty user message.
			result = append([]map[string]any{{
				"role":    "user",
				"content": ".",
			}}, result...)
		}
	}

	// Inject cache_control on the last user-role message. This creates a
	// cache breakpoint at the conversation prefix boundary, allowing all
	// prior turns to be cached across agentic loop iterations.
	if cacheEnabled {
		injectConversationCacheBreakpoint(result)
	}

	return result, nil
}

// injectConversationCacheBreakpoint adds cache_control to the last content
// block of the last user-role message. This enables Anthropic to cache the
// entire conversation prefix up to this point.
func injectConversationCacheBreakpoint(messages []map[string]any) {
	// Find the last user message.
	for i := len(messages) - 1; i >= 0; i-- {
		role, _ := messages[i]["role"].(string)
		if role != "user" {
			continue
		}

		content := messages[i]["content"]
		switch c := content.(type) {
		case []map[string]any:
			if len(c) > 0 {
				c[len(c)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
			}
		case string:
			// Convert to array form to add cache_control.
			messages[i]["content"] = []map[string]any{
				{
					"type":          "text",
					"text":          c,
					"cache_control": map[string]any{"type": "ephemeral"},
				},
			}
		}
		return
	}
}

// buildUserMessage constructs an Anthropic user message.
func (a *Adapter) buildUserMessage(msg types.Message) map[string]any {
	// Simple text-only message.
	if len(msg.Content) == 1 && msg.Content[0].Kind == types.ContentKindText {
		return map[string]any{
			"role":    "user",
			"content": msg.Content[0].Text,
		}
	}

	var parts []map[string]any
	for _, cp := range msg.Content {
		switch cp.Kind {
		case types.ContentKindText:
			parts = append(parts, map[string]any{
				"type": "text",
				"text": cp.Text,
			})
		case types.ContentKindImage:
			if cp.Image != nil {
				// Auto-resolve local file paths to base64 data.
				img, resolveErr := types.AutoResolveImage(cp.Image)
				if resolveErr != nil {
					// Skip the image but continue building the message.
					parts = append(parts, map[string]any{
						"type": "text",
						"text": fmt.Sprintf("[Failed to resolve image: %v]", resolveErr),
					})
				} else if img.URL != "" {
					parts = append(parts, map[string]any{
						"type": "image",
						"source": map[string]any{
							"type": "url",
							"url":  img.URL,
						},
					})
				} else if len(img.Data) > 0 {
					parts = append(parts, map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": img.MediaType,
							"data":       encodeBase64(img.Data),
						},
					})
				}
			}
		case types.ContentKindAudio:
			// Audio content is not supported by the Anthropic API. Replace with
			// a text placeholder so the model is aware content was omitted.
			parts = append(parts, map[string]any{
				"type": "text",
				"text": "[Audio content not supported by this provider]",
			})
		case types.ContentKindDocument:
			if cp.Document != nil && len(cp.Document.Data) > 0 {
				parts = append(parts, map[string]any{
					"type": "document",
					"source": map[string]any{
						"type":       "base64",
						"media_type": cp.Document.MediaType,
						"data":       encodeBase64(cp.Document.Data),
					},
				})
			}
		}
	}

	if len(parts) == 0 {
		return nil
	}

	return map[string]any{
		"role":    "user",
		"content": parts,
	}
}

// buildAssistantMessage constructs an Anthropic assistant message.
func (a *Adapter) buildAssistantMessage(msg types.Message) map[string]any {
	var parts []map[string]any
	for _, cp := range msg.Content {
		switch cp.Kind {
		case types.ContentKindText:
			parts = append(parts, map[string]any{
				"type": "text",
				"text": cp.Text,
			})
		case types.ContentKindToolCall:
			if cp.ToolCall != nil {
				input := cp.ToolCall.Arguments
				if input == nil {
					input = map[string]any{}
				}
				parts = append(parts, map[string]any{
					"type":  "tool_use",
					"id":    cp.ToolCall.ID,
					"name":  cp.ToolCall.Name,
					"input": input,
				})
			}
		case types.ContentKindThinking:
			if cp.Thinking != nil {
				block := map[string]any{
					"type":    "thinking",
					"thinking": cp.Thinking.Text,
				}
				if cp.Thinking.Signature != "" {
					block["signature"] = cp.Thinking.Signature
				}
				parts = append(parts, block)
			}
		case types.ContentKindRedactedThinking:
			parts = append(parts, map[string]any{
				"type": "redacted_thinking",
			})
		}
	}

	if len(parts) == 0 {
		// Fallback: simple text.
		text := msg.Text()
		if text == "" {
			return nil
		}
		return map[string]any{
			"role":    "assistant",
			"content": text,
		}
	}

	return map[string]any{
		"role":    "assistant",
		"content": parts,
	}
}

// buildToolResultMessage constructs a user message containing tool_result blocks.
func (a *Adapter) buildToolResultMessage(msg types.Message) map[string]any {
	var parts []map[string]any
	for _, cp := range msg.Content {
		if cp.Kind == types.ContentKindToolResult && cp.ToolResult != nil {
			block := map[string]any{
				"type":        "tool_result",
				"tool_use_id": cp.ToolResult.ToolCallID,
				"content":     cp.ToolResult.Content,
			}
			if cp.ToolResult.IsError {
				block["is_error"] = true
			}
			parts = append(parts, block)
		}
	}

	// Fallback: use the message-level ToolCallID.
	if len(parts) == 0 && msg.ToolCallID != "" {
		parts = append(parts, map[string]any{
			"type":        "tool_result",
			"tool_use_id": msg.ToolCallID,
			"content":     msg.Text(),
		})
	}

	if len(parts) == 0 {
		return nil
	}

	return map[string]any{
		"role":    "user",
		"content": parts,
	}
}

// mergeMessages combines two same-role messages by concatenating their content.
func mergeMessages(a, b map[string]any) map[string]any {
	aContent := toContentArray(a["content"])
	bContent := toContentArray(b["content"])
	a["content"] = append(aContent, bContent...)
	return a
}

// toContentArray normalizes message content to an array of content blocks.
func toContentArray(v any) []map[string]any {
	switch c := v.(type) {
	case string:
		return []map[string]any{{"type": "text", "text": c}}
	case []map[string]any:
		return c
	case []any:
		var result []map[string]any
		for _, item := range c {
			if m, ok := item.(map[string]any); ok {
				result = append(result, m)
			}
		}
		return result
	}
	return nil
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// parseResponse translates the Anthropic Messages API response to a unified Response.
func (a *Adapter) parseResponse(body map[string]any) (*types.Response, error) {
	resp := &types.Response{
		Provider: providerName,
		Raw:      body,
		Message: types.Message{
			Role: types.RoleAssistant,
		},
	}

	// ID
	if id, ok := body["id"].(string); ok {
		resp.ID = id
	}

	// Model
	if model, ok := body["model"].(string); ok {
		resp.Model = model
	}

	// Parse content blocks.
	content, _ := body["content"].([]any)
	for _, block := range content {
		blockMap, ok := block.(map[string]any)
		if !ok {
			continue
		}
		blockType, _ := blockMap["type"].(string)

		switch blockType {
		case "text":
			if text, ok := blockMap["text"].(string); ok {
				resp.Message.Content = append(resp.Message.Content, types.ContentPart{
					Kind: types.ContentKindText,
					Text: text,
				})
			}

		case "tool_use":
			tc := types.ToolCallData{Type: "function"}
			if id, ok := blockMap["id"].(string); ok {
				tc.ID = id
			}
			if name, ok := blockMap["name"].(string); ok {
				tc.Name = name
			}
			if input, ok := blockMap["input"].(map[string]any); ok {
				tc.Arguments = input
			}
			resp.Message.Content = append(resp.Message.Content, types.ContentPart{
				Kind:     types.ContentKindToolCall,
				ToolCall: &tc,
			})

		case "thinking":
			td := &types.ThinkingData{}
			if text, ok := blockMap["thinking"].(string); ok {
				td.Text = text
			}
			if sig, ok := blockMap["signature"].(string); ok {
				td.Signature = sig
			}
			resp.Message.Content = append(resp.Message.Content, types.ContentPart{
				Kind:     types.ContentKindThinking,
				Thinking: td,
			})

		case "redacted_thinking":
			resp.Message.Content = append(resp.Message.Content, types.ContentPart{
				Kind: types.ContentKindRedactedThinking,
				Thinking: &types.ThinkingData{
					Redacted: true,
				},
			})
		}
	}

	// Finish reason (stop_reason in Anthropic).
	if stopReason, ok := body["stop_reason"].(string); ok {
		resp.FinishReason = mapAnthropicFinishReason(stopReason)
	}

	// Usage
	if usage, ok := body["usage"].(map[string]any); ok {
		resp.Usage = parseUsage(usage)
	}

	// If the API did not report reasoning tokens explicitly but the response
	// contains thinking blocks, flag that reasoning was used. We cannot know
	// the exact token count from content alone, but we can signal presence by
	// checking for a non-empty thinking block and falling back to the output
	// tokens as an upper-bound estimate when thinking is the sole output.
	if resp.Usage.ReasoningTokens == nil {
		hasThinking := false
		for _, cp := range resp.Message.Content {
			if cp.Kind == types.ContentKindThinking || cp.Kind == types.ContentKindRedactedThinking {
				hasThinking = true
				break
			}
		}
		if hasThinking {
			// Mark reasoning tokens as present. Without an explicit count from
			// the API we set to 0 to indicate "present but unknown count" rather
			// than fabricating a number.
			zero := 0
			resp.Usage.ReasoningTokens = &zero
		}
	}

	return resp, nil
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// processStream reads SSE events from the Anthropic Messages API stream and
// translates them to unified StreamEvents.
func (a *Adapter) processStream(ctx context.Context, reader io.Reader, ch chan<- types.StreamEvent) {
	parser := sse.NewParser(reader)

	var currentToolCallID string
	var currentToolCallName string
	var currentBlockType string // "text", "tool_use", or "thinking"

	for {
		select {
		case <-ctx.Done():
			ch <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewAbortError("stream cancelled", ctx.Err()),
			}
			return
		default:
		}

		evt, err := parser.Next()
		if err == io.EOF {
			return
		}
		if err != nil {
			ch <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewStreamError("SSE parse error", err),
			}
			return
		}

		if evt.Data == "" {
			continue
		}

		var data map[string]any
		if err := json.Unmarshal([]byte(evt.Data), &data); err != nil {
			ch <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewStreamError("failed to parse SSE data", err),
			}
			return
		}

		eventType := evt.Type
		if eventType == "" {
			if t, ok := data["type"].(string); ok {
				eventType = t
			}
		}

		switch eventType {
		case "message_start":
			ch <- types.StreamEvent{
				Type: types.StreamEventStreamStart,
				Raw:  data,
			}

		case "content_block_start":
			if contentBlock, ok := data["content_block"].(map[string]any); ok {
				currentBlockType, _ = contentBlock["type"].(string)
				switch currentBlockType {
				case "text":
					ch <- types.StreamEvent{
						Type: types.StreamEventTextStart,
						Raw:  data,
					}
				case "tool_use":
					currentToolCallID, _ = contentBlock["id"].(string)
					currentToolCallName, _ = contentBlock["name"].(string)
					ch <- types.StreamEvent{
						Type: types.StreamEventToolCallStart,
						ToolCall: &types.ToolCallData{
							ID:   currentToolCallID,
							Name: currentToolCallName,
							Type: "function",
						},
						Raw: data,
					}
				case "thinking":
					ch <- types.StreamEvent{
						Type: types.StreamEventReasoningStart,
						Raw:  data,
					}
				}
			}

		case "content_block_delta":
			if delta, ok := data["delta"].(map[string]any); ok {
				deltaType, _ := delta["type"].(string)
				switch deltaType {
				case "text_delta":
					if text, ok := delta["text"].(string); ok {
						ch <- types.StreamEvent{
							Type:  types.StreamEventTextDelta,
							Delta: text,
							Raw:   data,
						}
					}
				case "input_json_delta":
					if partial, ok := delta["partial_json"].(string); ok {
						ch <- types.StreamEvent{
							Type:  types.StreamEventToolCallDelta,
							Delta: partial,
							ToolCall: &types.ToolCallData{
								ID:   currentToolCallID,
								Name: currentToolCallName,
								Type: "function",
							},
							Raw: data,
						}
					}
				case "thinking_delta":
					if thinking, ok := delta["thinking"].(string); ok {
						ch <- types.StreamEvent{
							Type:           types.StreamEventReasoningDelta,
							ReasoningDelta: thinking,
							Raw:            data,
						}
					}
				case "signature_delta":
					// Signature data for thinking round-trips -- forward as provider event.
					ch <- types.StreamEvent{
						Type: types.StreamEventProviderEvent,
						Raw:  data,
					}
				}
			}

		case "content_block_stop":
			// Emit the correct end event based on the tracked block type.
			switch currentBlockType {
			case "tool_use":
				ch <- types.StreamEvent{
					Type: types.StreamEventToolCallEnd,
					ToolCall: &types.ToolCallData{
						ID:   currentToolCallID,
						Name: currentToolCallName,
						Type: "function",
					},
					Raw: data,
				}
				currentToolCallID = ""
				currentToolCallName = ""
			case "thinking":
				ch <- types.StreamEvent{
					Type: types.StreamEventReasoningEnd,
					Raw:  data,
				}
			default:
				ch <- types.StreamEvent{
					Type: types.StreamEventTextEnd,
					Raw:  data,
				}
			}
			currentBlockType = ""

		case "message_delta":
			// Contains stop_reason and usage delta.
			var finishEvt types.StreamEvent
			finishEvt.Type = types.StreamEventFinish
			finishEvt.Raw = data

			if delta, ok := data["delta"].(map[string]any); ok {
				if stopReason, ok := delta["stop_reason"].(string); ok {
					fr := mapAnthropicFinishReason(stopReason)
					finishEvt.FinishReason = &fr
				}
			}
			if usage, ok := data["usage"].(map[string]any); ok {
				u := parseUsage(usage)
				finishEvt.Usage = &u
			}

			ch <- finishEvt

		case "message_stop":
			// End of stream -- channel will be closed by deferred close.

		case "ping":
			// Keep-alive; ignore.

		case "error":
			errMsg := "stream error"
			if errData, ok := data["error"].(map[string]any); ok {
				if msg, ok := errData["message"].(string); ok {
					errMsg = msg
				}
			}
			ch <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewStreamError(errMsg, nil),
				Raw:   data,
			}

		default:
			ch <- types.StreamEvent{
				Type: types.StreamEventProviderEvent,
				Raw:  data,
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

// parseError extracts error information from a non-200 response and returns
// the appropriate typed error.
func (a *Adapter) parseError(statusCode int, body []byte, headers http.Header) error {
	var raw map[string]any
	_ = json.Unmarshal(body, &raw)

	message := fmt.Sprintf("Anthropic API error (HTTP %d)", statusCode)
	errorCode := ""

	// Anthropic error format: {"type": "error", "error": {"type": "...", "message": "..."}}
	if errObj, ok := raw["error"].(map[string]any); ok {
		if msg, ok := errObj["message"].(string); ok && msg != "" {
			message = msg
		}
		if errType, ok := errObj["type"].(string); ok {
			errorCode = errType
		}
	}

	var retryAfter *float64
	if ra := headers.Get("Retry-After"); ra != "" {
		if seconds, err := strconv.ParseFloat(ra, 64); err == nil {
			retryAfter = &seconds
		}
	}

	return types.ErrorFromStatusCode(statusCode, message, providerName, errorCode, raw, retryAfter)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// setHeaders applies authentication and standard headers to an HTTP request.
func (a *Adapter) setHeaders(req *http.Request, providerOptions map[string]any) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)

	// Collect beta header values from provider options.
	// Support two conventions:
	//   1. Nested: providerOptions["anthropic"]["beta_headers"] (preferred)
	//   2. Top-level: providerOptions["anthropic_beta"] (backwards compat)
	var betaParts []string

	if providerOptions != nil {
		// First, check the nested convention: providerOptions["anthropic"]["beta_headers"].
		if anthropicOpts, ok := providerOptions["anthropic"].(map[string]any); ok {
			if beta, ok := anthropicOpts["beta_headers"].(string); ok {
				betaParts = append(betaParts, beta)
			}
			if betas, ok := anthropicOpts["beta_headers"].([]string); ok {
				betaParts = append(betaParts, betas...)
			}
			if betas, ok := anthropicOpts["beta_headers"].([]any); ok {
				for _, b := range betas {
					if s, ok := b.(string); ok {
						betaParts = append(betaParts, s)
					}
				}
			}
		}

		// Fall back to the top-level convention: providerOptions["anthropic_beta"].
		if beta, ok := providerOptions["anthropic_beta"].(string); ok {
			betaParts = append(betaParts, beta)
		}
		if betas, ok := providerOptions["anthropic_beta"].([]string); ok {
			betaParts = append(betaParts, betas...)
		}
		if betas, ok := providerOptions["anthropic_beta"].([]any); ok {
			for _, b := range betas {
				if s, ok := b.(string); ok {
					betaParts = append(betaParts, s)
				}
			}
		}
	}

	// Always enable prompt caching beta for cost optimization.
	hasCachingBeta := false
	for _, b := range betaParts {
		if strings.Contains(b, "prompt-caching") {
			hasCachingBeta = true
			break
		}
	}
	if !hasCachingBeta {
		betaParts = append(betaParts, "prompt-caching-2024-07-31")
	}

	if len(betaParts) > 0 {
		req.Header.Set("anthropic-beta", strings.Join(betaParts, ","))
	}

	for k, v := range a.headers {
		req.Header.Set(k, v)
	}
}

// mapAnthropicFinishReason maps an Anthropic stop_reason to a unified FinishReason.
func mapAnthropicFinishReason(stopReason string) types.FinishReason {
	switch stopReason {
	case "end_turn":
		return types.FinishReason{Reason: "stop", Raw: stopReason}
	case "stop_sequence":
		return types.FinishReason{Reason: "stop", Raw: stopReason}
	case "max_tokens":
		return types.FinishReason{Reason: "length", Raw: stopReason}
	case "tool_use":
		return types.FinishReason{Reason: "tool_calls", Raw: stopReason}
	default:
		return types.FinishReason{Reason: "other", Raw: stopReason}
	}
}

// parseUsage extracts token usage from the Anthropic usage object.
func parseUsage(usage map[string]any) types.Usage {
	u := types.Usage{
		Raw: usage,
	}

	if v, ok := toInt(usage["input_tokens"]); ok {
		u.InputTokens = v
	}
	if v, ok := toInt(usage["output_tokens"]); ok {
		u.OutputTokens = v
	}
	u.TotalTokens = u.InputTokens + u.OutputTokens

	// Cache token counts.
	if v, ok := toInt(usage["cache_creation_input_tokens"]); ok {
		u.CacheWriteTokens = &v
	}
	if v, ok := toInt(usage["cache_read_input_tokens"]); ok {
		u.CacheReadTokens = &v
	}

	// Reasoning/thinking tokens. The Anthropic API may report these directly
	// in the usage object (e.g. as "thinking_tokens" or within an
	// "output_tokens_details" sub-object). Check both locations.
	if v, ok := toInt(usage["thinking_tokens"]); ok {
		u.ReasoningTokens = &v
	} else if details, ok := usage["output_tokens_details"].(map[string]any); ok {
		if v, ok := toInt(details["thinking_tokens"]); ok {
			u.ReasoningTokens = &v
		}
	}

	return u
}

// parseRateLimitHeaders extracts rate limit information from response headers.
func parseRateLimitHeaders(headers http.Header) *types.RateLimitInfo {
	info := &types.RateLimitInfo{}
	hasAny := false

	if v := headers.Get("anthropic-ratelimit-requests-remaining"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.RequestsRemaining = &n
			hasAny = true
		}
	}
	if v := headers.Get("anthropic-ratelimit-requests-limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.RequestsLimit = &n
			hasAny = true
		}
	}
	if v := headers.Get("anthropic-ratelimit-tokens-remaining"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.TokensRemaining = &n
			hasAny = true
		}
	}
	if v := headers.Get("anthropic-ratelimit-tokens-limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.TokensLimit = &n
			hasAny = true
		}
	}

	if !hasAny {
		return nil
	}
	return info
}

// toInt converts a JSON number (float64) to int.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	}
	return 0, false
}

// encodeBase64 encodes bytes as standard base64.
func encodeBase64(data []byte) string {
	const base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	var b strings.Builder
	b.Grow(((len(data) + 2) / 3) * 4)

	for i := 0; i < len(data); i += 3 {
		var val uint32
		remaining := len(data) - i
		for j := 0; j < 3; j++ {
			val <<= 8
			if j < remaining {
				val |= uint32(data[i+j])
			}
		}
		b.WriteByte(base64Chars[(val>>18)&0x3F])
		b.WriteByte(base64Chars[(val>>12)&0x3F])
		if remaining > 1 {
			b.WriteByte(base64Chars[(val>>6)&0x3F])
		} else {
			b.WriteByte('=')
		}
		if remaining > 2 {
			b.WriteByte(base64Chars[val&0x3F])
		} else {
			b.WriteByte('=')
		}
	}

	return b.String()
}
