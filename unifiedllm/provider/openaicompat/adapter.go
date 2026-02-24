// Package openaicompat implements a ProviderAdapter for OpenAI-compatible
// endpoints that use the Chat Completions API (/v1/chat/completions).
//
// Many third-party services (vLLM, Ollama, Together AI, Groq, etc.) expose
// an OpenAI-compatible Chat Completions API. This adapter uses the standard
// Chat Completions format rather than the Responses API, making it compatible
// with these services.
//
// Key differences from the native OpenAI adapter:
//   - Uses /v1/chat/completions (not /v1/responses)
//   - Uses messages array with role/content (not input array)
//   - Uses {"type": "function", "function": {...}} tool wrapper format
//   - Uses data: {"choices": [...]} streaming format (not Responses API events)
//   - Does not support reasoning tokens or built-in tools
package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/strongdm/attractor-go/unifiedllm/sse"
	"github.com/strongdm/attractor-go/unifiedllm/types"
)

const (
	providerName = "openai-compat"
)

// Adapter implements ProviderAdapter for OpenAI-compatible Chat Completions endpoints.
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
	return func(a *Adapter) { a.apiKey = key }
}

// WithBaseURL sets the base URL for the API endpoint.
func WithBaseURL(url string) AdapterOption {
	return func(a *Adapter) { a.baseURL = strings.TrimRight(url, "/") }
}

// WithHTTPClient provides a custom HTTP client.
func WithHTTPClient(c *http.Client) AdapterOption {
	return func(a *Adapter) { a.httpClient = c }
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(headers map[string]string) AdapterOption {
	return func(a *Adapter) { a.headers = headers }
}

// New creates an OpenAI-compatible adapter with the given options.
func New(opts ...AdapterOption) *Adapter {
	a := &Adapter{
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// FromEnv creates an Adapter configured from environment variables:
//   - OPENAI_COMPAT_API_KEY (required)
//   - OPENAI_COMPAT_BASE_URL (required)
func FromEnv() (*Adapter, error) {
	key := os.Getenv("OPENAI_COMPAT_API_KEY")
	if key == "" {
		return nil, types.NewConfigurationError("OPENAI_COMPAT_API_KEY environment variable is not set", nil)
	}
	baseURL := os.Getenv("OPENAI_COMPAT_BASE_URL")
	if baseURL == "" {
		return nil, types.NewConfigurationError("OPENAI_COMPAT_BASE_URL environment variable is not set", nil)
	}
	return New(WithAPIKey(key), WithBaseURL(baseURL)), nil
}

// Name returns the canonical provider name.
func (a *Adapter) Name() string { return providerName }

// Complete sends a blocking Chat Completions request.
func (a *Adapter) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	body, err := a.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, types.NewSDKError("failed to create HTTP request", err)
	}
	a.setHeaders(httpReq)

	httpResp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, types.NewNetworkError("HTTP request failed", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, types.NewNetworkError("failed to read response body", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, types.ErrorFromStatusCode(httpResp.StatusCode, string(respBody), providerName, "", nil, nil)
	}

	return a.parseResponse(respBody)
}

// Stream sends a streaming Chat Completions request.
func (a *Adapter) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	body, err := a.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, types.NewSDKError("failed to create HTTP request", err)
	}
	a.setHeaders(httpReq)

	httpResp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, types.NewNetworkError("HTTP request failed", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, types.ErrorFromStatusCode(httpResp.StatusCode, string(body), providerName, "", nil, nil)
	}

	out := make(chan types.StreamEvent, 64)
	go func() {
		defer close(out)
		defer httpResp.Body.Close()
		a.processStream(ctx, httpResp.Body, out)
	}()

	return out, nil
}

// Close releases resources. The default HTTP client has no resources to release.
func (a *Adapter) Close() error { return nil }

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

func (a *Adapter) buildRequestBody(req types.Request, stream bool) (map[string]any, error) {
	body := map[string]any{
		"model": req.Model,
	}

	// Messages: standard Chat Completions format.
	var messages []map[string]any
	for _, msg := range req.Messages {
		switch msg.Role {
		case types.RoleSystem:
			messages = append(messages, map[string]any{
				"role":    "system",
				"content": msg.Text(),
			})
		case types.RoleUser:
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": msg.Text(),
			})
		case types.RoleAssistant:
			item := map[string]any{
				"role": "assistant",
			}
			text := msg.Text()
			if text != "" {
				item["content"] = text
			}
			// Extract tool calls from content parts.
			var tcs []map[string]any
			for _, cp := range msg.Content {
				if cp.Kind == types.ContentKindToolCall && cp.ToolCall != nil {
					argsJSON, _ := json.Marshal(cp.ToolCall.Arguments)
					tcs = append(tcs, map[string]any{
						"id":   cp.ToolCall.ID,
						"type": "function",
						"function": map[string]any{
							"name":      cp.ToolCall.Name,
							"arguments": string(argsJSON),
						},
					})
				}
			}
			if len(tcs) > 0 {
				item["tool_calls"] = tcs
			}
			messages = append(messages, item)
		case types.RoleTool:
			for _, cp := range msg.Content {
				if cp.Kind == types.ContentKindToolResult && cp.ToolResult != nil {
					messages = append(messages, map[string]any{
						"role":         "tool",
						"tool_call_id": cp.ToolResult.ToolCallID,
						"content":      cp.ToolResult.Content,
					})
				}
			}
		}
	}
	body["messages"] = messages

	// Tools: Chat Completions format with {"type": "function", "function": {...}} wrapper.
	if len(req.Tools) > 0 {
		var tools []map[string]any
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

	// Tool choice.
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
		default:
			return nil, types.NewUnsupportedToolChoiceError(
				fmt.Sprintf("OpenAI-compatible adapter does not support tool_choice mode %q", req.ToolChoice.Mode),
				req.ToolChoice.Mode, nil,
			)
		}
	}

	// Temperature.
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.MaxTokens != nil {
		body["max_tokens"] = *req.MaxTokens
	}
	if len(req.StopSequences) > 0 {
		body["stop"] = req.StopSequences
	}

	// Response format.
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object":
			body["response_format"] = map[string]any{"type": "json_object"}
		case "json_schema":
			body["response_format"] = map[string]any{
				"type": "json_schema",
				"json_schema": map[string]any{
					"name":   "response_schema",
					"schema": req.ResponseFormat.JSONSchema,
					"strict": req.ResponseFormat.Strict,
				},
			}
		}
	}

	if stream {
		body["stream"] = true
	}

	return body, nil
}

func (a *Adapter) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if a.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+a.apiKey)
	}
	for k, v := range a.headers {
		req.Header.Set(k, v)
	}
}

// ---------------------------------------------------------------------------
// Response parsing (blocking)
// ---------------------------------------------------------------------------

func (a *Adapter) parseResponse(body []byte) (*types.Response, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, types.NewSDKError("failed to parse response JSON", err)
	}

	choices, _ := raw["choices"].([]any)
	if len(choices) == 0 {
		return nil, types.NewSDKError("response contains no choices", nil)
	}

	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	finishReason, _ := choice["finish_reason"].(string)

	var content []types.ContentPart

	// Text content.
	if text, ok := message["content"].(string); ok && text != "" {
		content = append(content, types.ContentPart{
			Kind: types.ContentKindText,
			Text: text,
		})
	}

	// Tool calls.
	if toolCalls, ok := message["tool_calls"].([]any); ok {
		for _, tcRaw := range toolCalls {
			tc, _ := tcRaw.(map[string]any)
			id, _ := tc["id"].(string)
			fn, _ := tc["function"].(map[string]any)
			name, _ := fn["name"].(string)
			argsStr, _ := fn["arguments"].(string)

			var args map[string]any
			_ = json.Unmarshal([]byte(argsStr), &args)

			content = append(content, types.ContentPart{
				Kind: types.ContentKindToolCall,
				ToolCall: &types.ToolCallData{
					ID:        id,
					Name:      name,
					Type:      "function",
					Arguments: args,
				},
			})
		}
	}

	resp := &types.Response{
		ID: fmt.Sprintf("%v", raw["id"]),
		Message: types.Message{
			Role:    types.RoleAssistant,
			Content: content,
		},
		FinishReason: mapFinishReason(finishReason),
		Provider:     providerName,
	}

	// Usage.
	if usage, ok := raw["usage"].(map[string]any); ok {
		resp.Usage = parseUsage(usage)
	}

	return resp, nil
}

func mapFinishReason(reason string) types.FinishReason {
	switch reason {
	case "stop":
		return types.FinishReasonStop
	case "tool_calls":
		return types.FinishReasonToolCalls
	case "length":
		return types.FinishReasonLength
	case "content_filter":
		return types.FinishReasonContentFilter
	default:
		return types.FinishReasonStop
	}
}

func parseUsage(usage map[string]any) types.Usage {
	toInt := func(v any) int {
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		default:
			return 0
		}
	}
	u := types.Usage{
		InputTokens:  toInt(usage["prompt_tokens"]),
		OutputTokens: toInt(usage["completion_tokens"]),
	}
	u.TotalTokens = u.InputTokens + u.OutputTokens
	return u
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

func (a *Adapter) processStream(ctx context.Context, body io.Reader, out chan<- types.StreamEvent) {
	parser := sse.NewParser(body)

	// Accumulated tool call state.
	type partialToolCall struct {
		id      string
		name    string
		argsStr strings.Builder
	}
	pendingToolCalls := make(map[int]*partialToolCall)

	for {
		event, err := parser.Next()
		if err != nil {
			if err == io.EOF {
				// Stream ended normally.
				return
			}
			out <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewNetworkError("stream read error", err),
			}
			return
		}

		if event.Data == "" {
			continue
		}

		var chunk map[string]any
		if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
			continue
		}

		choices, _ := chunk["choices"].([]any)
		if len(choices) == 0 {
			// May be a usage-only chunk.
			if usage, ok := chunk["usage"].(map[string]any); ok {
				u := parseUsage(usage)
				out <- types.StreamEvent{
					Type:  types.StreamEventFinish,
					Usage: &u,
				}
			}
			continue
		}

		choice, _ := choices[0].(map[string]any)
		delta, _ := choice["delta"].(map[string]any)
		finishReason, _ := choice["finish_reason"].(string)

		if delta != nil {
			// Text delta.
			if content, ok := delta["content"].(string); ok && content != "" {
				out <- types.StreamEvent{
					Type:  types.StreamEventTextDelta,
					Delta: content,
				}
			}

			// Tool call deltas.
			if tcs, ok := delta["tool_calls"].([]any); ok {
				for _, tcRaw := range tcs {
					tc, _ := tcRaw.(map[string]any)
					idx := 0
					if idxF, ok := tc["index"].(float64); ok {
						idx = int(idxF)
					}

					ptc, exists := pendingToolCalls[idx]
					if !exists {
						ptc = &partialToolCall{}
						pendingToolCalls[idx] = ptc
					}

					if id, ok := tc["id"].(string); ok {
						ptc.id = id
					}
					if fn, ok := tc["function"].(map[string]any); ok {
						if name, ok := fn["name"].(string); ok {
							ptc.name = name
							// Emit tool call start.
							out <- types.StreamEvent{
								Type: types.StreamEventToolCallStart,
								ToolCall: &types.ToolCallData{
									ID:   ptc.id,
									Name: ptc.name,
									Type: "function",
								},
							}
						}
						if args, ok := fn["arguments"].(string); ok {
							ptc.argsStr.WriteString(args)
						}
					}
				}
			}
		}

		// Finish reason.
		if finishReason != "" {
			// Finalize any pending tool calls.
			for _, ptc := range pendingToolCalls {
				var args map[string]any
				_ = json.Unmarshal([]byte(ptc.argsStr.String()), &args)
				out <- types.StreamEvent{
					Type: types.StreamEventToolCallEnd,
					ToolCall: &types.ToolCallData{
						ID:        ptc.id,
						Name:      ptc.name,
						Type:      "function",
						Arguments: args,
					},
				}
			}

			fr := mapFinishReason(finishReason)
			out <- types.StreamEvent{
				Type:         types.StreamEventFinish,
				FinishReason: &fr,
			}
		}
	}
}
