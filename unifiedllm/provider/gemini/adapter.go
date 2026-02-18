// Package gemini implements the ProviderAdapter for the Google Gemini API.
//
// This adapter translates unified types to the Gemini generateContent API format
// (/v1beta/models/*/generateContent) and back. It handles:
//   - System messages as the systemInstruction parameter
//   - User/model role mapping (Gemini uses "model" instead of "assistant")
//   - Tool calls with synthetic unique IDs (Gemini does not assign call IDs)
//   - Tool results using function name instead of call ID
//   - Streaming via ?alt=sse query parameter
//   - Multi-part content with text and image support
package gemini

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
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
	defaultBaseURL = "https://generativelanguage.googleapis.com"
	providerName   = "gemini"
)

// Adapter implements ProviderAdapter for the Google Gemini API.
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

// New creates a Gemini adapter with the given options.
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
//   - GEMINI_API_KEY (required)
//   - GEMINI_BASE_URL (optional)
func FromEnv() (*Adapter, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, types.NewConfigurationError("GEMINI_API_KEY environment variable is not set", nil)
	}

	opts := []AdapterOption{WithAPIKey(key)}

	if baseURL := os.Getenv("GEMINI_BASE_URL"); baseURL != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}

	return New(opts...), nil
}

// Name returns the canonical provider name.
func (a *Adapter) Name() string {
	return providerName
}

// Complete sends a blocking completion request to the Gemini API.
func (a *Adapter) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	body, err := a.buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", a.baseURL, req.Model, a.apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
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
		return nil, a.parseError(resp.StatusCode, respBody)
	}

	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return nil, types.NewSDKError("failed to parse response JSON", err)
	}

	return a.parseResponse(raw, req.Model)
}

// Stream sends a streaming request to the Gemini API using ?alt=sse.
func (a *Adapter) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	body, err := a.buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, types.NewSDKError("failed to marshal request body", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", a.baseURL, req.Model, a.apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
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
		return nil, a.parseError(resp.StatusCode, respBody)
	}

	ch := make(chan types.StreamEvent, 64)

	go func() {
		defer close(ch)
		defer resp.Body.Close()

		a.processStream(ctx, resp.Body, ch, req.Model)
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

// buildRequestBody translates a unified Request into the Gemini API format.
func (a *Adapter) buildRequestBody(req types.Request) (map[string]any, error) {
	body := make(map[string]any)

	// Extract system messages into systemInstruction.
	var systemParts []map[string]any
	var messages []types.Message

	for _, msg := range req.Messages {
		if msg.Role == types.RoleSystem || msg.Role == types.RoleDeveloper {
			systemParts = append(systemParts, map[string]any{
				"text": msg.Text(),
			})
		} else {
			messages = append(messages, msg)
		}
	}

	if len(systemParts) > 0 {
		body["systemInstruction"] = map[string]any{
			"parts": systemParts,
		}
	}

	// Build contents array.
	contents := a.buildContents(messages)
	if len(contents) > 0 {
		body["contents"] = contents
	}

	// Generation config.
	genConfig := make(map[string]any)

	if req.Temperature != nil {
		genConfig["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		genConfig["topP"] = *req.TopP
	}
	if req.MaxTokens != nil {
		genConfig["maxOutputTokens"] = *req.MaxTokens
	}
	if len(req.StopSequences) > 0 {
		genConfig["stopSequences"] = req.StopSequences
	}

	// Response format.
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json":
			genConfig["responseMimeType"] = "application/json"
		case "json_schema":
			genConfig["responseMimeType"] = "application/json"
			if req.ResponseFormat.JSONSchema != nil {
				genConfig["responseSchema"] = req.ResponseFormat.JSONSchema
			}
		}
	}

	// Reasoning effort -> thinking config.
	if req.ReasoningEffort != "" {
		thinkingConfig := map[string]any{
			"includeThoughts": true,
		}
		switch req.ReasoningEffort {
		case "low":
			thinkingConfig["thinkingBudget"] = 1024
		case "medium":
			thinkingConfig["thinkingBudget"] = 4096
		case "high":
			thinkingConfig["thinkingBudget"] = 16384
		}
		genConfig["thinkingConfig"] = thinkingConfig
	}

	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}

	// Tools
	if len(req.Tools) > 0 {
		var funcDecls []map[string]any
		for _, tool := range req.Tools {
			decl := map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
			}
			if tool.Parameters != nil {
				decl["parameters"] = tool.Parameters
			}
			funcDecls = append(funcDecls, decl)
		}
		body["tools"] = []map[string]any{
			{"functionDeclarations": funcDecls},
		}
	}

	// Tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case "auto":
			body["toolConfig"] = map[string]any{
				"functionCallingConfig": map[string]any{
					"mode": "AUTO",
				},
			}
		case "none":
			body["toolConfig"] = map[string]any{
				"functionCallingConfig": map[string]any{
					"mode": "NONE",
				},
			}
		case "required":
			body["toolConfig"] = map[string]any{
				"functionCallingConfig": map[string]any{
					"mode": "ANY",
				},
			}
		case "named":
			body["toolConfig"] = map[string]any{
				"functionCallingConfig": map[string]any{
					"mode":                "ANY",
					"allowedFunctionNames": []string{req.ToolChoice.ToolName},
				},
			}
		}
	}

	// Pass through provider-specific options.
	for k, v := range req.ProviderOptions {
		body[k] = v
	}

	return body, nil
}

// buildContents translates unified messages into Gemini contents array.
func (a *Adapter) buildContents(messages []types.Message) []map[string]any {
	var contents []map[string]any

	for _, msg := range messages {
		switch msg.Role {
		case types.RoleUser:
			content := a.buildUserContent(msg)
			if content != nil {
				contents = append(contents, content)
			}

		case types.RoleAssistant:
			content := a.buildModelContent(msg)
			if content != nil {
				contents = append(contents, content)
			}

		case types.RoleTool:
			content := a.buildFunctionResponseContent(msg)
			if content != nil {
				contents = append(contents, content)
			}
		}
	}

	return contents
}

// buildUserContent constructs a Gemini user content entry.
func (a *Adapter) buildUserContent(msg types.Message) map[string]any {
	var parts []map[string]any

	for _, cp := range msg.Content {
		switch cp.Kind {
		case types.ContentKindText:
			parts = append(parts, map[string]any{
				"text": cp.Text,
			})
		case types.ContentKindImage:
			if cp.Image != nil {
				if len(cp.Image.Data) > 0 {
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": cp.Image.MediaType,
							"data":     encodeBase64(cp.Image.Data),
						},
					})
				} else if cp.Image.URL != "" {
					parts = append(parts, map[string]any{
						"fileData": map[string]any{
							"mimeType": cp.Image.MediaType,
							"fileUri":  cp.Image.URL,
						},
					})
				}
			}
		}
	}

	if len(parts) == 0 {
		return nil
	}

	return map[string]any{
		"role":  "user",
		"parts": parts,
	}
}

// buildModelContent constructs a Gemini model content entry for assistant messages.
func (a *Adapter) buildModelContent(msg types.Message) map[string]any {
	var parts []map[string]any

	for _, cp := range msg.Content {
		switch cp.Kind {
		case types.ContentKindText:
			parts = append(parts, map[string]any{
				"text": cp.Text,
			})
		case types.ContentKindToolCall:
			if cp.ToolCall != nil {
				fc := map[string]any{
					"name": cp.ToolCall.Name,
				}
				if cp.ToolCall.Arguments != nil {
					fc["args"] = cp.ToolCall.Arguments
				}
				parts = append(parts, map[string]any{
					"functionCall": fc,
				})
			}
		case types.ContentKindThinking:
			if cp.Thinking != nil {
				parts = append(parts, map[string]any{
					"thought": true,
					"text":    cp.Thinking.Text,
				})
			}
		}
	}

	if len(parts) == 0 {
		return nil
	}

	return map[string]any{
		"role":  "model",
		"parts": parts,
	}
}

// buildFunctionResponseContent constructs a Gemini function response content entry.
// Gemini uses the function name instead of call ID for matching responses.
func (a *Adapter) buildFunctionResponseContent(msg types.Message) map[string]any {
	var parts []map[string]any

	for _, cp := range msg.Content {
		if cp.Kind == types.ContentKindToolResult && cp.ToolResult != nil {
			// Try to parse the content as JSON for structured responses.
			var responseData any
			if err := json.Unmarshal([]byte(cp.ToolResult.Content), &responseData); err != nil {
				// Not JSON; wrap as a text response.
				responseData = map[string]any{
					"result": cp.ToolResult.Content,
				}
			}

			// We need the function name. Try to find it from the tool call ID
			// by looking at the Name field on the message.
			name := msg.Name
			if name == "" {
				// Fall back to the tool call ID which may contain the name.
				name = cp.ToolResult.ToolCallID
			}

			parts = append(parts, map[string]any{
				"functionResponse": map[string]any{
					"name":     name,
					"response": responseData,
				},
			})
		}
	}

	// Fallback: use message-level ToolCallID.
	if len(parts) == 0 && msg.ToolCallID != "" {
		name := msg.Name
		if name == "" {
			name = msg.ToolCallID
		}

		var responseData any
		text := msg.Text()
		if err := json.Unmarshal([]byte(text), &responseData); err != nil {
			responseData = map[string]any{"result": text}
		}

		parts = append(parts, map[string]any{
			"functionResponse": map[string]any{
				"name":     name,
				"response": responseData,
			},
		})
	}

	if len(parts) == 0 {
		return nil
	}

	return map[string]any{
		"role":  "user",
		"parts": parts,
	}
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// parseResponse translates the Gemini API response to a unified Response.
func (a *Adapter) parseResponse(body map[string]any, model string) (*types.Response, error) {
	resp := &types.Response{
		Provider: providerName,
		Model:    model,
		Raw:      body,
		Message: types.Message{
			Role: types.RoleAssistant,
		},
	}

	// Parse candidates.
	candidates, _ := body["candidates"].([]any)
	if len(candidates) > 0 {
		candidate, ok := candidates[0].(map[string]any)
		if ok {
			a.parseCandidateContent(candidate, resp)

			// Finish reason.
			if reason, ok := candidate["finishReason"].(string); ok {
				resp.FinishReason = mapGeminiFinishReason(reason)
			}
		}
	}

	// Check for prompt feedback (safety blocks).
	if feedback, ok := body["promptFeedback"].(map[string]any); ok {
		if blockReason, ok := feedback["blockReason"].(string); ok {
			resp.FinishReason = types.FinishReason{Reason: "content_filter", Raw: blockReason}
			resp.Warnings = append(resp.Warnings, types.Warning{
				Message: fmt.Sprintf("prompt blocked: %s", blockReason),
				Code:    "prompt_blocked",
			})
		}
	}

	// Usage metadata.
	if usage, ok := body["usageMetadata"].(map[string]any); ok {
		resp.Usage = parseUsage(usage)
	}

	// If there are tool calls, adjust the finish reason.
	if len(resp.ToolCalls()) > 0 && resp.FinishReason.Reason != "tool_calls" {
		resp.FinishReason = types.FinishReason{Reason: "tool_calls", Raw: resp.FinishReason.Raw}
	}

	return resp, nil
}

// parseCandidateContent extracts content from a Gemini candidate.
func (a *Adapter) parseCandidateContent(candidate map[string]any, resp *types.Response) {
	content, ok := candidate["content"].(map[string]any)
	if !ok {
		return
	}

	parts, _ := content["parts"].([]any)
	for _, part := range parts {
		partMap, ok := part.(map[string]any)
		if !ok {
			continue
		}

		// Text part.
		if text, ok := partMap["text"].(string); ok {
			// Check if this is a thought/reasoning part.
			if thought, ok := partMap["thought"].(bool); ok && thought {
				resp.Message.Content = append(resp.Message.Content, types.ContentPart{
					Kind: types.ContentKindThinking,
					Thinking: &types.ThinkingData{
						Text: text,
					},
				})
			} else {
				resp.Message.Content = append(resp.Message.Content, types.ContentPart{
					Kind: types.ContentKindText,
					Text: text,
				})
			}
		}

		// Function call part.
		if fc, ok := partMap["functionCall"].(map[string]any); ok {
			tc := types.ToolCallData{
				Type: "function",
				ID:   generateSyntheticID(),
			}
			if name, ok := fc["name"].(string); ok {
				tc.Name = name
			}
			if args, ok := fc["args"].(map[string]any); ok {
				tc.Arguments = args
			}
			resp.Message.Content = append(resp.Message.Content, types.ContentPart{
				Kind:     types.ContentKindToolCall,
				ToolCall: &tc,
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// processStream reads SSE events from the Gemini streaming API and translates
// them to unified StreamEvents.
func (a *Adapter) processStream(ctx context.Context, reader io.Reader, ch chan<- types.StreamEvent, model string) {
	parser := sse.NewParser(reader)
	sentStart := false
	var lastUsage *types.Usage

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
			// Send finish event if we have accumulated usage.
			if lastUsage != nil {
				fr := types.FinishReasonStop
				ch <- types.StreamEvent{
					Type:         types.StreamEventFinish,
					FinishReason: &fr,
					Usage:        lastUsage,
				}
			}
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

		if !sentStart {
			ch <- types.StreamEvent{
				Type: types.StreamEventStreamStart,
				Raw:  data,
			}
			sentStart = true
		}

		// Check for errors in the response.
		if errObj, ok := data["error"].(map[string]any); ok {
			errMsg := "Gemini stream error"
			if msg, ok := errObj["message"].(string); ok {
				errMsg = msg
			}
			ch <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewStreamError(errMsg, nil),
				Raw:   data,
			}
			return
		}

		// Parse candidates.
		candidates, _ := data["candidates"].([]any)
		if len(candidates) == 0 {
			// Could be usage-only update.
			if usage, ok := data["usageMetadata"].(map[string]any); ok {
				u := parseUsage(usage)
				lastUsage = &u
			}
			continue
		}

		candidate, ok := candidates[0].(map[string]any)
		if !ok {
			continue
		}

		content, _ := candidate["content"].(map[string]any)
		if content == nil {
			// Check for finish reason without content.
			if reason, ok := candidate["finishReason"].(string); ok {
				fr := mapGeminiFinishReason(reason)
				var usage *types.Usage
				if u, ok := data["usageMetadata"].(map[string]any); ok {
					parsed := parseUsage(u)
					usage = &parsed
				}
				ch <- types.StreamEvent{
					Type:         types.StreamEventFinish,
					FinishReason: &fr,
					Usage:        usage,
					Raw:          data,
				}
			}
			continue
		}

		parts, _ := content["parts"].([]any)
		for _, part := range parts {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}

			// Text delta.
			if text, ok := partMap["text"].(string); ok {
				if thought, ok := partMap["thought"].(bool); ok && thought {
					ch <- types.StreamEvent{
						Type:           types.StreamEventReasoningDelta,
						ReasoningDelta: text,
						Raw:            data,
					}
				} else {
					ch <- types.StreamEvent{
						Type:  types.StreamEventTextDelta,
						Delta: text,
						Raw:   data,
					}
				}
			}

			// Function call.
			if fc, ok := partMap["functionCall"].(map[string]any); ok {
				tc := &types.ToolCallData{
					Type: "function",
					ID:   generateSyntheticID(),
				}
				if name, ok := fc["name"].(string); ok {
					tc.Name = name
				}
				if args, ok := fc["args"].(map[string]any); ok {
					tc.Arguments = args
				}

				// For Gemini, function calls arrive as complete objects in streaming,
				// so emit start and end together.
				ch <- types.StreamEvent{
					Type:     types.StreamEventToolCallStart,
					ToolCall: tc,
					Raw:      data,
				}
				ch <- types.StreamEvent{
					Type:     types.StreamEventToolCallEnd,
					ToolCall: tc,
					Raw:      data,
				}
			}
		}

		// Check for finish reason on this chunk.
		if reason, ok := candidate["finishReason"].(string); ok {
			fr := mapGeminiFinishReason(reason)
			var usage *types.Usage
			if u, ok := data["usageMetadata"].(map[string]any); ok {
				parsed := parseUsage(u)
				usage = &parsed
				lastUsage = nil // Already emitting it.
			}
			ch <- types.StreamEvent{
				Type:         types.StreamEventFinish,
				FinishReason: &fr,
				Usage:        usage,
				Raw:          data,
			}
		}

		// Track usage for the final event.
		if usage, ok := data["usageMetadata"].(map[string]any); ok {
			u := parseUsage(usage)
			lastUsage = &u
		}
	}
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

// parseError extracts error information from a non-200 response.
func (a *Adapter) parseError(statusCode int, body []byte) error {
	var raw map[string]any
	_ = json.Unmarshal(body, &raw)

	message := fmt.Sprintf("Gemini API error (HTTP %d)", statusCode)
	errorCode := ""

	// Gemini error format: {"error": {"code": 400, "message": "...", "status": "INVALID_ARGUMENT"}}
	if errObj, ok := raw["error"].(map[string]any); ok {
		if msg, ok := errObj["message"].(string); ok && msg != "" {
			message = msg
		}
		if status, ok := errObj["status"].(string); ok {
			errorCode = status
		}
	}

	return types.ErrorFromStatusCode(statusCode, message, providerName, errorCode, raw, nil)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// setHeaders applies standard headers to an HTTP request. Gemini uses API key
// in the URL query string, not in headers.
func (a *Adapter) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")

	for k, v := range a.headers {
		req.Header.Set(k, v)
	}
}

// mapGeminiFinishReason maps a Gemini finish reason to a unified FinishReason.
func mapGeminiFinishReason(reason string) types.FinishReason {
	switch reason {
	case "STOP":
		return types.FinishReason{Reason: "stop", Raw: reason}
	case "MAX_TOKENS":
		return types.FinishReason{Reason: "length", Raw: reason}
	case "SAFETY":
		return types.FinishReason{Reason: "content_filter", Raw: reason}
	case "RECITATION":
		return types.FinishReason{Reason: "content_filter", Raw: reason}
	case "BLOCKLIST":
		return types.FinishReason{Reason: "content_filter", Raw: reason}
	case "PROHIBITED_CONTENT":
		return types.FinishReason{Reason: "content_filter", Raw: reason}
	case "SPII":
		return types.FinishReason{Reason: "content_filter", Raw: reason}
	case "MALFORMED_FUNCTION_CALL":
		return types.FinishReason{Reason: "error", Raw: reason}
	default:
		return types.FinishReason{Reason: "other", Raw: reason}
	}
}

// parseUsage extracts token usage from the Gemini usageMetadata object.
func parseUsage(usage map[string]any) types.Usage {
	u := types.Usage{
		Raw: usage,
	}

	if v, ok := toInt(usage["promptTokenCount"]); ok {
		u.InputTokens = v
	}
	if v, ok := toInt(usage["candidatesTokenCount"]); ok {
		u.OutputTokens = v
	}
	if v, ok := toInt(usage["totalTokenCount"]); ok {
		u.TotalTokens = v
	} else {
		u.TotalTokens = u.InputTokens + u.OutputTokens
	}

	// Reasoning/thinking tokens (Gemini 2.0+ with thinking enabled).
	if v, ok := toInt(usage["thoughtsTokenCount"]); ok {
		u.ReasoningTokens = &v
	}

	// Cached content tokens.
	if v, ok := toInt(usage["cachedContentTokenCount"]); ok {
		u.CacheReadTokens = &v
	}

	return u
}

// generateSyntheticID creates a unique ID for tool calls since Gemini does not
// assign call IDs natively. The format is "call_" followed by 12 random hex
// characters.
func generateSyntheticID() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		// Fallback to a simple counter-based approach if crypto/rand fails,
		// which should never happen in practice.
		return fmt.Sprintf("call_%d", syntheticCounter())
	}
	return "call_" + hex.EncodeToString(b)
}

// syntheticCounter provides a simple fallback counter. It is not goroutine-safe
// but is only used as a last resort if crypto/rand fails.
var syntheticSeq uint64

func syntheticCounter() uint64 {
	syntheticSeq++
	return syntheticSeq
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

// encodeBase64 encodes bytes as standard base64. This is a self-contained
// implementation to avoid importing encoding/base64 and keep the zero external
// dependency constraint.
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

// Unused import suppressor for strconv (used in other adapters, included here
// for consistency even though Gemini doesn't use header-based rate limits).
var _ = strconv.Itoa
