// Package openai implements the ProviderAdapter for the OpenAI Responses API.
//
// This adapter translates unified types to the OpenAI Responses API format
// (/v1/responses) and back. It handles:
//   - System messages as the "instructions" parameter
//   - User/assistant/tool messages as the "input" array
//   - Tool definitions as function-type tools
//   - Streaming via SSE with the Responses API stream format
//   - Reasoning effort for models that support extended thinking
//   - Structured output / response format constraints
package openai

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
	defaultBaseURL = "https://api.openai.com"
	providerName   = "openai"
)

// Adapter implements ProviderAdapter for the OpenAI API.
type Adapter struct {
	apiKey     string
	baseURL    string
	orgID      string
	projectID  string
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

// WithOrgID sets the OpenAI organization ID header.
func WithOrgID(id string) AdapterOption {
	return func(a *Adapter) {
		a.orgID = id
	}
}

// WithProjectID sets the OpenAI project ID header.
func WithProjectID(id string) AdapterOption {
	return func(a *Adapter) {
		a.projectID = id
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

// New creates an OpenAI adapter with the given options.
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
//   - OPENAI_API_KEY (required)
//   - OPENAI_BASE_URL (optional, defaults to https://api.openai.com)
//   - OPENAI_ORG_ID (optional)
//   - OPENAI_PROJECT_ID (optional)
func FromEnv() (*Adapter, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, types.NewConfigurationError("OPENAI_API_KEY environment variable is not set", nil)
	}

	opts := []AdapterOption{WithAPIKey(key)}

	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}
	if orgID := os.Getenv("OPENAI_ORG_ID"); orgID != "" {
		opts = append(opts, WithOrgID(orgID))
	}
	if projectID := os.Getenv("OPENAI_PROJECT_ID"); projectID != "" {
		opts = append(opts, WithProjectID(projectID))
	}

	return New(opts...), nil
}

// Name returns the canonical provider name.
func (a *Adapter) Name() string {
	return providerName
}

// Complete sends a blocking completion request using the Responses API.
func (a *Adapter) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	body, err := a.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/responses", bytes.NewReader(payload))
	if err != nil {
		return nil, types.NewSDKError("failed to create HTTP request", err)
	}

	a.setHeaders(httpReq)

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

// Stream sends a streaming request using the Responses API with stream=true.
func (a *Adapter) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	body, err := a.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/responses", bytes.NewReader(payload))
	if err != nil {
		return nil, types.NewSDKError("failed to create HTTP request", err)
	}

	a.setHeaders(httpReq)

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

// Close releases adapter resources. The OpenAI adapter uses the shared
// http.DefaultClient by default so there is nothing to close.
func (a *Adapter) Close() error {
	return nil
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

// buildRequestBody translates a unified Request into the OpenAI Responses API
// request format.
func (a *Adapter) buildRequestBody(req types.Request, stream bool) (map[string]any, error) {
	body := map[string]any{
		"model": req.Model,
	}

	// Extract system messages as "instructions".
	var instructions []string
	var input []map[string]any

	for _, msg := range req.Messages {
		switch msg.Role {
		case types.RoleSystem, types.RoleDeveloper:
			instructions = append(instructions, msg.Text())

		case types.RoleUser:
			item := a.buildUserInput(msg)
			input = append(input, item)

		case types.RoleAssistant:
			item := a.buildAssistantInput(msg)
			input = append(input, item)

		case types.RoleTool:
			items := a.buildToolResultInputs(msg)
			input = append(input, items...)
		}
	}

	if len(instructions) > 0 {
		body["instructions"] = strings.Join(instructions, "\n\n")
	}

	if len(input) > 0 {
		body["input"] = input
	}

	// Tools
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, tool := range req.Tools {
			t := map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        tool.Name,
					"description": tool.Description,
				},
			}
			if tool.Parameters != nil {
				t["function"].(map[string]any)["parameters"] = tool.Parameters
			}
			tools = append(tools, t)
		}
		body["tools"] = tools
	}

	// Tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case "auto":
			body["tool_choice"] = "auto"
		case "none":
			body["tool_choice"] = "none"
		case "required":
			body["tool_choice"] = "required"
		case "named":
			body["tool_choice"] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": req.ToolChoice.ToolName,
				},
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

	// MaxTokens -> max_output_tokens in Responses API
	if req.MaxTokens != nil {
		body["max_output_tokens"] = *req.MaxTokens
	}

	// Reasoning effort
	if req.ReasoningEffort != "" {
		body["reasoning"] = map[string]any{
			"effort": req.ReasoningEffort,
		}
	}

	// Response format
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json":
			body["text"] = map[string]any{
				"format": map[string]any{
					"type": "json_object",
				},
			}
		case "json_schema":
			format := map[string]any{
				"type":   "json_schema",
				"schema": req.ResponseFormat.JSONSchema,
			}
			if req.ResponseFormat.Strict {
				format["strict"] = true
			}
			body["text"] = map[string]any{
				"format": format,
			}
		}
	}

	// Metadata
	if len(req.Metadata) > 0 {
		body["metadata"] = req.Metadata
	}

	// Stream
	if stream {
		body["stream"] = true
	}

	// Pass through any provider-specific options.
	for k, v := range req.ProviderOptions {
		body[k] = v
	}

	return body, nil
}

// buildUserInput constructs a Responses API input item for a user message.
func (a *Adapter) buildUserInput(msg types.Message) map[string]any {
	item := map[string]any{
		"role": "user",
	}

	// If all content is text, use a simple string content.
	if len(msg.Content) == 1 && msg.Content[0].Kind == types.ContentKindText {
		item["content"] = msg.Content[0].Text
		return item
	}

	// Multi-part content.
	var parts []map[string]any
	for _, cp := range msg.Content {
		switch cp.Kind {
		case types.ContentKindText:
			parts = append(parts, map[string]any{
				"type": "input_text",
				"text": cp.Text,
			})
		case types.ContentKindImage:
			if cp.Image != nil {
				if cp.Image.URL != "" {
					parts = append(parts, map[string]any{
						"type":      "input_image",
						"image_url": cp.Image.URL,
					})
				} else if len(cp.Image.Data) > 0 {
					parts = append(parts, map[string]any{
						"type":      "input_image",
						"image_url": fmt.Sprintf("data:%s;base64,%s", cp.Image.MediaType, encodeBase64(cp.Image.Data)),
					})
				}
			}
		}
	}

	item["content"] = parts
	return item
}

// buildAssistantInput constructs a Responses API input item for an assistant message.
func (a *Adapter) buildAssistantInput(msg types.Message) map[string]any {
	item := map[string]any{
		"role": "assistant",
	}

	// Build content parts for the assistant output.
	var parts []map[string]any
	for _, cp := range msg.Content {
		switch cp.Kind {
		case types.ContentKindText:
			parts = append(parts, map[string]any{
				"type": "output_text",
				"text": cp.Text,
			})
		case types.ContentKindToolCall:
			if cp.ToolCall != nil {
				argJSON, _ := json.Marshal(cp.ToolCall.Arguments)
				parts = append(parts, map[string]any{
					"type":      "function_call",
					"id":        cp.ToolCall.ID,
					"name":      cp.ToolCall.Name,
					"arguments": string(argJSON),
				})
			}
		}
	}

	if len(parts) == 1 && parts[0]["type"] == "output_text" {
		item["content"] = parts[0]["text"]
	} else if len(parts) > 0 {
		item["content"] = parts
	}

	return item
}

// buildToolResultInputs constructs Responses API input items for tool results.
// Each tool result content part becomes a separate function_call_output item.
func (a *Adapter) buildToolResultInputs(msg types.Message) []map[string]any {
	var items []map[string]any
	for _, cp := range msg.Content {
		if cp.Kind == types.ContentKindToolResult && cp.ToolResult != nil {
			items = append(items, map[string]any{
				"type":    "function_call_output",
				"call_id": cp.ToolResult.ToolCallID,
				"output":  cp.ToolResult.Content,
			})
		}
	}
	// Fallback: if no tool result parts but there is a ToolCallID on the message.
	if len(items) == 0 && msg.ToolCallID != "" {
		items = append(items, map[string]any{
			"type":    "function_call_output",
			"call_id": msg.ToolCallID,
			"output":  msg.Text(),
		})
	}
	return items
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// parseResponse translates the OpenAI Responses API response to a unified Response.
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

	// Parse output items.
	output, _ := body["output"].([]any)
	for _, item := range output {
		itemMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		itemType, _ := itemMap["type"].(string)

		switch itemType {
		case "message":
			// Parse message content.
			a.parseMessageOutput(itemMap, resp)

		case "function_call":
			// Tool call output item.
			tc := types.ToolCallData{
				Type: "function",
			}
			if id, ok := itemMap["id"].(string); ok {
				tc.ID = id
			}
			if name, ok := itemMap["name"].(string); ok {
				tc.Name = name
			}
			if args, ok := itemMap["arguments"].(string); ok {
				var parsed map[string]any
				if err := json.Unmarshal([]byte(args), &parsed); err == nil {
					tc.Arguments = parsed
				}
			}
			resp.Message.Content = append(resp.Message.Content, types.ContentPart{
				Kind:     types.ContentKindToolCall,
				ToolCall: &tc,
			})

		case "reasoning":
			// Reasoning/thinking output.
			if summary, ok := itemMap["summary"].([]any); ok {
				for _, s := range summary {
					if sMap, ok := s.(map[string]any); ok {
						if text, ok := sMap["text"].(string); ok {
							resp.Message.Content = append(resp.Message.Content, types.ContentPart{
								Kind: types.ContentKindThinking,
								Thinking: &types.ThinkingData{
									Text: text,
								},
							})
						}
					}
				}
			}
		}
	}

	// Finish reason (status field in Responses API).
	if status, ok := body["status"].(string); ok {
		resp.FinishReason = mapOpenAIFinishReason(status)
	}

	// If there are tool calls but no explicit tool_calls finish reason, set it.
	if len(resp.ToolCalls()) > 0 && resp.FinishReason.Reason != "tool_calls" {
		resp.FinishReason = types.FinishReason{Reason: "tool_calls", Raw: resp.FinishReason.Raw}
	}

	// Usage
	if usage, ok := body["usage"].(map[string]any); ok {
		resp.Usage = parseUsage(usage)
	}

	return resp, nil
}

// parseMessageOutput extracts content from a "message" type output item.
func (a *Adapter) parseMessageOutput(itemMap map[string]any, resp *types.Response) {
	content, _ := itemMap["content"].([]any)
	for _, c := range content {
		cMap, ok := c.(map[string]any)
		if !ok {
			continue
		}
		cType, _ := cMap["type"].(string)
		switch cType {
		case "output_text":
			if text, ok := cMap["text"].(string); ok {
				resp.Message.Content = append(resp.Message.Content, types.ContentPart{
					Kind: types.ContentKindText,
					Text: text,
				})
			}
		case "refusal":
			if refusal, ok := cMap["refusal"].(string); ok {
				resp.Warnings = append(resp.Warnings, types.Warning{
					Message: refusal,
					Code:    "refusal",
				})
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// processStream reads SSE events from the OpenAI Responses API stream and
// translates them to unified StreamEvents on the channel.
func (a *Adapter) processStream(ctx context.Context, reader io.Reader, ch chan<- types.StreamEvent) {
	parser := sse.NewParser(reader)

	// Track state for accumulating tool call arguments.
	var currentToolCallID string
	var currentToolCallName string

	for {
		// Check for context cancellation.
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
		case "response.created":
			ch <- types.StreamEvent{
				Type: types.StreamEventStreamStart,
				Raw:  data,
			}

		case "response.output_item.added":
			item, _ := data["item"].(map[string]any)
			if item != nil {
				itemType, _ := item["type"].(string)
				switch itemType {
				case "function_call":
					currentToolCallID, _ = item["id"].(string)
					currentToolCallName, _ = item["name"].(string)
					ch <- types.StreamEvent{
						Type: types.StreamEventToolCallStart,
						ToolCall: &types.ToolCallData{
							ID:   currentToolCallID,
							Name: currentToolCallName,
							Type: "function",
						},
						Raw: data,
					}
				case "message":
					// Message item started -- no unified event needed.
				case "reasoning":
					ch <- types.StreamEvent{
						Type: types.StreamEventReasoningStart,
						Raw:  data,
					}
				}
			}

		case "response.content_part.added":
			// A content part within a message was added. Signal text start.
			ch <- types.StreamEvent{
				Type: types.StreamEventTextStart,
				Raw:  data,
			}

		case "response.output_text.delta":
			if delta, ok := data["delta"].(string); ok {
				ch <- types.StreamEvent{
					Type:  types.StreamEventTextDelta,
					Delta: delta,
					Raw:   data,
				}
			}

		case "response.output_text.done":
			ch <- types.StreamEvent{
				Type: types.StreamEventTextEnd,
				Raw:  data,
			}

		case "response.function_call_arguments.delta":
			if delta, ok := data["delta"].(string); ok {
				ch <- types.StreamEvent{
					Type:  types.StreamEventToolCallDelta,
					Delta: delta,
					ToolCall: &types.ToolCallData{
						ID:   currentToolCallID,
						Name: currentToolCallName,
						Type: "function",
					},
					Raw: data,
				}
			}

		case "response.function_call_arguments.done":
			var args map[string]any
			if argStr, ok := data["arguments"].(string); ok {
				_ = json.Unmarshal([]byte(argStr), &args)
			}
			ch <- types.StreamEvent{
				Type: types.StreamEventToolCallEnd,
				ToolCall: &types.ToolCallData{
					ID:        currentToolCallID,
					Name:      currentToolCallName,
					Type:      "function",
					Arguments: args,
				},
				Raw: data,
			}
			currentToolCallID = ""
			currentToolCallName = ""

		case "response.reasoning_summary_part.added":
			ch <- types.StreamEvent{
				Type: types.StreamEventReasoningStart,
				Raw:  data,
			}

		case "response.reasoning_summary_text.delta":
			if delta, ok := data["delta"].(string); ok {
				ch <- types.StreamEvent{
					Type:           types.StreamEventReasoningDelta,
					ReasoningDelta: delta,
					Raw:            data,
				}
			}

		case "response.reasoning_summary_text.done":
			ch <- types.StreamEvent{
				Type: types.StreamEventReasoningEnd,
				Raw:  data,
			}

		case "response.completed", "response.done":
			// Extract final usage and finish reason from the response object.
			var finishEvt types.StreamEvent
			finishEvt.Type = types.StreamEventFinish
			finishEvt.Raw = data

			if response, ok := data["response"].(map[string]any); ok {
				if status, ok := response["status"].(string); ok {
					fr := mapOpenAIFinishReason(status)
					finishEvt.FinishReason = &fr
				}
				if usage, ok := response["usage"].(map[string]any); ok {
					u := parseUsage(usage)
					finishEvt.Usage = &u
				}
			}

			ch <- finishEvt

		case "response.failed":
			errMsg := "response failed"
			if response, ok := data["response"].(map[string]any); ok {
				if e, ok := response["error"].(map[string]any); ok {
					if msg, ok := e["message"].(string); ok {
						errMsg = msg
					}
				}
			}
			ch <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewSDKError(errMsg, nil),
				Raw:   data,
			}

		default:
			// Forward unrecognized events as provider events.
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

	message := fmt.Sprintf("OpenAI API error (HTTP %d)", statusCode)
	errorCode := ""

	// OpenAI error format: {"error": {"message": "...", "type": "...", "code": "..."}}
	if errObj, ok := raw["error"].(map[string]any); ok {
		if msg, ok := errObj["message"].(string); ok && msg != "" {
			message = msg
		}
		if code, ok := errObj["code"].(string); ok {
			errorCode = code
		}
		if errType, ok := errObj["type"].(string); ok && errorCode == "" {
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
func (a *Adapter) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+a.apiKey)

	if a.orgID != "" {
		req.Header.Set("OpenAI-Organization", a.orgID)
	}
	if a.projectID != "" {
		req.Header.Set("OpenAI-Project", a.projectID)
	}

	for k, v := range a.headers {
		req.Header.Set(k, v)
	}
}

// mapOpenAIFinishReason maps an OpenAI response status to a unified FinishReason.
func mapOpenAIFinishReason(status string) types.FinishReason {
	switch status {
	case "completed":
		return types.FinishReason{Reason: "stop", Raw: status}
	case "incomplete":
		return types.FinishReason{Reason: "length", Raw: status}
	case "cancelled":
		return types.FinishReason{Reason: "stop", Raw: status}
	case "failed":
		return types.FinishReason{Reason: "error", Raw: status}
	default:
		return types.FinishReason{Reason: "other", Raw: status}
	}
}

// parseUsage extracts token usage from the OpenAI usage object.
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
	if v, ok := toInt(usage["total_tokens"]); ok {
		u.TotalTokens = v
	} else {
		u.TotalTokens = u.InputTokens + u.OutputTokens
	}

	// Extended token counts.
	if outputDetails, ok := usage["output_tokens_details"].(map[string]any); ok {
		if v, ok := toInt(outputDetails["reasoning_tokens"]); ok {
			u.ReasoningTokens = &v
		}
	}
	if inputDetails, ok := usage["input_tokens_details"].(map[string]any); ok {
		if v, ok := toInt(inputDetails["cached_tokens"]); ok {
			u.CacheReadTokens = &v
		}
	}

	return u
}

// parseRateLimitHeaders extracts rate limit information from response headers.
func parseRateLimitHeaders(headers http.Header) *types.RateLimitInfo {
	info := &types.RateLimitInfo{}
	hasAny := false

	if v := headers.Get("x-ratelimit-remaining-requests"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.RequestsRemaining = &n
			hasAny = true
		}
	}
	if v := headers.Get("x-ratelimit-limit-requests"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.RequestsLimit = &n
			hasAny = true
		}
	}
	if v := headers.Get("x-ratelimit-remaining-tokens"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.TokensRemaining = &n
			hasAny = true
		}
	}
	if v := headers.Get("x-ratelimit-limit-tokens"); v != "" {
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
