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
	apiKey     string
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
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

// buildRequestBody translates a unified Request into the Anthropic Messages API format.
func (a *Adapter) buildRequestBody(req types.Request, stream bool) (map[string]any, error) {
	body := map[string]any{
		"model": req.Model,
	}

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
		// Use array form to support cache_control annotations.
		body["system"] = systemParts
	}

	// Build messages with alternation enforcement.
	apiMessages, err := a.buildMessages(messages)
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
		if k == "anthropic_beta" || k == "beta_headers" {
			continue
		}
		body[k] = v
	}

	return body, nil
}

// buildMessages enforces Anthropic's strict user/assistant alternation by
// merging consecutive same-role messages. Tool results are placed in user
// messages with tool_result content blocks.
func (a *Adapter) buildMessages(messages []types.Message) ([]map[string]any, error) {
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

	return result, nil
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
				if cp.Image.URL != "" {
					parts = append(parts, map[string]any{
						"type": "image",
						"source": map[string]any{
							"type": "url",
							"url":  cp.Image.URL,
						},
					})
				} else if len(cp.Image.Data) > 0 {
					parts = append(parts, map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": cp.Image.MediaType,
							"data":       encodeBase64(cp.Image.Data),
						},
					})
				}
			}
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
				blockType, _ := contentBlock["type"].(string)
				switch blockType {
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
			// Determine what type of block ended. The index can help, but we
			// use the current state to decide.
			if currentToolCallID != "" {
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
			} else {
				// Could be text end or reasoning end. Send text end as default.
				ch <- types.StreamEvent{
					Type: types.StreamEventTextEnd,
					Raw:  data,
				}
			}

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

	// Beta headers from provider options.
	if providerOptions != nil {
		if beta, ok := providerOptions["anthropic_beta"].(string); ok {
			req.Header.Set("anthropic-beta", beta)
		}
		if betas, ok := providerOptions["anthropic_beta"].([]string); ok {
			req.Header.Set("anthropic-beta", strings.Join(betas, ","))
		}
		if betas, ok := providerOptions["anthropic_beta"].([]any); ok {
			var parts []string
			for _, b := range betas {
				if s, ok := b.(string); ok {
					parts = append(parts, s)
				}
			}
			if len(parts) > 0 {
				req.Header.Set("anthropic-beta", strings.Join(parts, ","))
			}
		}
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
