// Package unifiedllm provides high-level convenience functions for interacting
// with LLM providers through the unified client. It wraps the lower-level
// client.Client with tool execution loops, retry logic, streaming support,
// and structured output generation.
//
// The primary entry points are:
//   - Generate: blocking generation with automatic tool execution loops
//   - Stream: streaming generation with tool loop support
//   - GenerateObject: structured output validated against a JSON schema
//
// All functions use a module-level default client that is lazily initialized
// from environment variables, or callers can supply an explicit client via
// GenerateOptions.Client.
package unifiedllm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/strongdm/attractor-go/unifiedllm/client"
	"github.com/strongdm/attractor-go/unifiedllm/retry"
	"github.com/strongdm/attractor-go/unifiedllm/types"
)

// ---------------------------------------------------------------------------
// Default client management
// ---------------------------------------------------------------------------

// defaultClient is the module-level default, lazily initialized from env.
var (
	defaultClient     *client.Client
	defaultClientOnce sync.Once
	defaultClientErr  error
	defaultClientMu   sync.Mutex
)

// SetDefaultClient overrides the module-level default client. This is useful
// when the caller has already constructed a client with specific provider
// configurations and wants all top-level functions to use it.
func SetDefaultClient(c *client.Client) {
	defaultClientMu.Lock()
	defer defaultClientMu.Unlock()
	defaultClient = c
	defaultClientErr = nil
	// Reset the Once so that subsequent calls to getDefaultClient return the
	// manually-set client instead of attempting env-based initialization.
	defaultClientOnce = sync.Once{}
	// Mark as done since we already have a client.
	defaultClientOnce.Do(func() {})
}

// getDefaultClient returns the default client, initializing from env if needed.
func getDefaultClient() (*client.Client, error) {
	defaultClientMu.Lock()
	defer defaultClientMu.Unlock()

	defaultClientOnce.Do(func() {
		defaultClient, defaultClientErr = client.FromEnv()
	})

	if defaultClientErr != nil {
		return nil, defaultClientErr
	}
	if defaultClient == nil {
		return nil, types.NewConfigurationError("no default client configured; call SetDefaultClient or set provider API keys in environment", nil)
	}
	return defaultClient, nil
}

// ---------------------------------------------------------------------------
// GenerateOptions
// ---------------------------------------------------------------------------

// GenerateOptions configures a Generate, Stream, or GenerateObject call.
type GenerateOptions struct {
	// Model is the provider-specific model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
	Model string

	// Prompt is a simple text prompt converted to a single user message.
	// Mutually exclusive with Messages.
	Prompt string

	// Messages is the full message history. Mutually exclusive with Prompt.
	Messages []types.Message

	// System is the system message prepended to the conversation.
	System string

	// Tools are the tools available for the model to invoke. Tools with
	// non-nil Execute handlers participate in the automatic tool execution loop.
	Tools []types.Tool

	// ToolChoice controls how the model selects tools.
	ToolChoice *types.ToolChoice

	// MaxToolRounds is the maximum number of tool execution loop iterations.
	// Each round consists of: execute tool calls from the model, append results,
	// and call the model again. Default: 1. Set to 0 to disable the tool loop
	// entirely (tool calls will be returned but not executed).
	MaxToolRounds int

	// ResponseFormat constrains the output structure.
	ResponseFormat *types.ResponseFormat

	// Temperature controls randomness. Nil means provider default.
	Temperature *float64

	// TopP controls nucleus sampling. Nil means provider default.
	TopP *float64

	// MaxTokens limits the response length. Nil means provider default.
	MaxTokens *int

	// StopSequences are sequences that cause the model to stop generating.
	StopSequences []string

	// ReasoningEffort hints to providers that support thinking/reasoning.
	ReasoningEffort string

	// Provider optionally pins the request to a specific provider backend.
	Provider string

	// ProviderOptions carries provider-specific parameters.
	ProviderOptions map[string]any

	// MaxRetries is the maximum number of retry attempts for transient failures
	// on individual LLM calls (not whole operations). Default: 2.
	MaxRetries int

	// Client overrides the default client for this call.
	Client *client.Client
}

// ---------------------------------------------------------------------------
// GenerateResult
// ---------------------------------------------------------------------------

// GenerateResult contains the output from a Generate or GenerateObject call.
type GenerateResult struct {
	// Text is the concatenated text content from the final response.
	Text string

	// Reasoning is the concatenated reasoning/thinking content from the final response.
	Reasoning string

	// ToolCalls contains the tool calls from the final response (if any remained unexecuted).
	ToolCalls []types.ToolCall

	// ToolResults contains the results from all tool executions across all steps.
	ToolResults []types.ToolResult

	// FinishReason indicates why generation stopped.
	FinishReason types.FinishReason

	// Usage is the token usage from the final step.
	Usage types.Usage

	// TotalUsage is the aggregated token usage across all steps.
	TotalUsage types.Usage

	// Steps contains the detailed results from each LLM call in the tool loop.
	Steps []StepResult

	// Response is the raw response from the final LLM call.
	Response *types.Response
}

// StepResult captures the output from a single LLM call within the tool loop.
type StepResult struct {
	// Text is the text content from this step's response.
	Text string

	// Reasoning is the reasoning/thinking content from this step's response.
	Reasoning string

	// ToolCalls contains the tool calls requested by the model in this step.
	ToolCalls []types.ToolCall

	// ToolResults contains the results from executing tool calls in this step.
	ToolResults []types.ToolResult

	// FinishReason indicates why this step's generation stopped.
	FinishReason types.FinishReason

	// Usage is the token usage for this step.
	Usage types.Usage

	// Response is the raw response from this step's LLM call.
	Response *types.Response
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

// Generate is the primary blocking generation function. It handles prompt
// standardization, tool execution loops, multi-step orchestration, and retries.
//
// Tool loop: when tools with Execute handlers are provided and the model
// responds with tool calls, Generate automatically executes them, appends
// results, and calls the model again. This continues until the model responds
// without tool calls, MaxToolRounds is reached, or the context is cancelled.
//
// MaxToolRounds semantics: value N means up to N rounds of tool execution.
// Total LLM calls is at most MaxToolRounds + 1 (initial call + N rounds).
// Default is 1.
func Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	c, err := resolveClient(opts.Client)
	if err != nil {
		return nil, err
	}

	req, err := buildRequest(opts)
	if err != nil {
		return nil, err
	}

	maxRounds := opts.MaxToolRounds
	if maxRounds <= 0 && hasExecutableTools(opts.Tools) {
		maxRounds = 1
	}

	steps, err := toolLoop(ctx, c, req, opts.Tools, maxRounds, buildRetryPolicy(opts.MaxRetries))
	if err != nil {
		return nil, err
	}

	return buildGenerateResult(steps), nil
}

// ---------------------------------------------------------------------------
// StreamResult
// ---------------------------------------------------------------------------

// StreamResult wraps a streaming generation with tool loop support. It provides
// access to the event channel for real-time processing and the accumulated
// response after the stream completes.
type StreamResult struct {
	events chan types.StreamEvent
	done   chan struct{}
	resp   *types.Response
	mu     sync.Mutex
	err    error
}

// Events returns the channel of stream events. Read from this channel to
// process events in real time. The channel is closed when the stream ends.
func (sr *StreamResult) Events() <-chan types.StreamEvent {
	return sr.events
}

// Response returns the accumulated response after the stream ends. Returns
// nil if the stream has not yet completed.
func (sr *StreamResult) Response() *types.Response {
	<-sr.done
	sr.mu.Lock()
	defer sr.mu.Unlock()
	return sr.resp
}

// Err returns any error that occurred during streaming. Returns nil if the
// stream completed successfully. Blocks until the stream ends.
func (sr *StreamResult) Err() error {
	<-sr.done
	sr.mu.Lock()
	defer sr.mu.Unlock()
	return sr.err
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

// Stream is the primary streaming generation function. It accepts the same
// parameters as Generate. When tools are present, the stream pauses during
// tool execution and resumes with the next model response. Tool execution
// results are emitted as synthetic events on the stream.
func Stream(ctx context.Context, opts GenerateOptions) (*StreamResult, error) {
	c, err := resolveClient(opts.Client)
	if err != nil {
		return nil, err
	}

	req, err := buildRequest(opts)
	if err != nil {
		return nil, err
	}

	maxRounds := opts.MaxToolRounds
	if maxRounds <= 0 && hasExecutableTools(opts.Tools) {
		maxRounds = 1
	}

	sr := &StreamResult{
		events: make(chan types.StreamEvent, 64),
		done:   make(chan struct{}),
	}

	go func() {
		defer close(sr.events)
		defer close(sr.done)

		streamErr := streamToolLoop(ctx, c, req, opts.Tools, maxRounds, sr.events)
		sr.mu.Lock()
		sr.err = streamErr
		sr.mu.Unlock()
	}()

	return sr, nil
}

// ---------------------------------------------------------------------------
// GenerateObject
// ---------------------------------------------------------------------------

// GenerateObject produces structured output validated against a JSON schema.
// It uses provider-native structured output (OpenAI json_schema, Gemini
// responseSchema) by setting ResponseFormat on the request. For providers
// that do not support native structured output, it falls back to prompt-based
// extraction.
//
// The schema parameter should be a JSON Schema object (map[string]any) that
// describes the expected output structure.
//
// GenerateResult.Text contains the raw JSON string. Callers should unmarshal
// it into their target type.
func GenerateObject(ctx context.Context, opts GenerateOptions, schema map[string]any) (*GenerateResult, error) {
	// Set up structured output via ResponseFormat.
	opts.ResponseFormat = &types.ResponseFormat{
		Type:       "json_schema",
		JSONSchema: schema,
		Strict:     true,
	}

	// If no system prompt is set, add one that instructs the model to
	// produce valid JSON matching the schema. This helps providers that
	// do not have native structured output support.
	if opts.System == "" {
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, types.NewSDKError("failed to marshal schema for system prompt", err)
		}
		opts.System = fmt.Sprintf(
			"You must respond with valid JSON that conforms to the following JSON Schema. "+
				"Do not include any text outside the JSON object.\n\nSchema:\n%s",
			string(schemaBytes),
		)
	}

	result, err := Generate(ctx, opts)
	if err != nil {
		return nil, err
	}

	// Validate that the output is valid JSON.
	text := strings.TrimSpace(result.Text)
	if text == "" {
		return nil, types.NewNoObjectGeneratedError(
			"model produced empty output; expected JSON matching the provided schema",
			nil,
		)
	}

	// Attempt to parse the JSON to verify it is well-formed.
	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		// Try to extract JSON from markdown code fences that some models produce.
		extracted := extractJSON(text)
		if extracted != "" {
			if jsonErr := json.Unmarshal([]byte(extracted), &parsed); jsonErr == nil {
				result.Text = extracted
				return result, nil
			}
		}
		return nil, types.NewNoObjectGeneratedError(
			fmt.Sprintf("model output is not valid JSON: %v", err),
			err,
		)
	}

	result.Text = text
	return result, nil
}

// ---------------------------------------------------------------------------
// StreamAccumulator
// ---------------------------------------------------------------------------

// StreamAccumulator collects stream events into a complete Response. It is
// safe to call Process from a single goroutine reading the event channel.
type StreamAccumulator struct {
	textParts      []string
	reasoningParts []string
	toolCalls      map[string]*types.ToolCallData
	finishReason   *types.FinishReason
	usage          *types.Usage
	response       *types.Response
}

// NewStreamAccumulator creates a new StreamAccumulator ready to process events.
func NewStreamAccumulator() *StreamAccumulator {
	return &StreamAccumulator{
		toolCalls: make(map[string]*types.ToolCallData),
	}
}

// Process incorporates a single stream event into the accumulated state.
func (sa *StreamAccumulator) Process(event types.StreamEvent) {
	switch event.Type {
	case types.StreamEventTextDelta:
		sa.textParts = append(sa.textParts, event.Delta)

	case types.StreamEventReasoningDelta:
		sa.reasoningParts = append(sa.reasoningParts, event.ReasoningDelta)

	case types.StreamEventToolCallStart:
		if event.ToolCall != nil {
			sa.toolCalls[event.ToolCall.ID] = &types.ToolCallData{
				ID:   event.ToolCall.ID,
				Name: event.ToolCall.Name,
				Type: event.ToolCall.Type,
			}
		}

	case types.StreamEventToolCallDelta:
		if event.ToolCall != nil {
			if tc, ok := sa.toolCalls[event.ToolCall.ID]; ok {
				// Merge arguments incrementally. Tool call deltas may carry
				// partial argument data that needs to be accumulated.
				if tc.Arguments == nil && event.ToolCall.Arguments != nil {
					tc.Arguments = event.ToolCall.Arguments
				} else if event.ToolCall.Arguments != nil {
					for k, v := range event.ToolCall.Arguments {
						tc.Arguments[k] = v
					}
				}
			}
		}

	case types.StreamEventToolCallEnd:
		// Tool call is finalized; nothing additional to do.

	case types.StreamEventFinish:
		if event.FinishReason != nil {
			sa.finishReason = event.FinishReason
		}
		if event.Usage != nil {
			sa.usage = event.Usage
		}
		if event.Response != nil {
			sa.response = event.Response
		}
	}
}

// Response returns the accumulated Response. If a complete Response was
// delivered via a finish event, that is returned directly. Otherwise, a
// synthetic Response is constructed from the accumulated parts.
func (sa *StreamAccumulator) Response() *types.Response {
	if sa.response != nil {
		return sa.response
	}

	// Build a synthetic response from accumulated parts.
	var content []types.ContentPart

	// Add reasoning parts.
	reasoning := strings.Join(sa.reasoningParts, "")
	if reasoning != "" {
		content = append(content, types.ContentPart{
			Kind:     types.ContentKindThinking,
			Thinking: &types.ThinkingData{Text: reasoning},
		})
	}

	// Add text parts.
	text := strings.Join(sa.textParts, "")
	if text != "" {
		content = append(content, types.ContentPart{
			Kind: types.ContentKindText,
			Text: text,
		})
	}

	// Add tool calls.
	for _, tc := range sa.toolCalls {
		content = append(content, types.ContentPart{
			Kind:     types.ContentKindToolCall,
			ToolCall: tc,
		})
	}

	resp := &types.Response{
		Message: types.Message{
			Role:    types.RoleAssistant,
			Content: content,
		},
	}

	if sa.finishReason != nil {
		resp.FinishReason = *sa.finishReason
	}
	if sa.usage != nil {
		resp.Usage = *sa.usage
	}

	return resp
}

// Text returns the concatenated text content accumulated so far.
func (sa *StreamAccumulator) Text() string {
	return strings.Join(sa.textParts, "")
}

// Reasoning returns the concatenated reasoning content accumulated so far.
func (sa *StreamAccumulator) Reasoning() string {
	return strings.Join(sa.reasoningParts, "")
}

// ---------------------------------------------------------------------------
// Internal: request building
// ---------------------------------------------------------------------------

// resolveClient returns the client to use, preferring the explicit option.
func resolveClient(explicit *client.Client) (*client.Client, error) {
	if explicit != nil {
		return explicit, nil
	}
	return getDefaultClient()
}

// buildRequest constructs a types.Request from GenerateOptions.
func buildRequest(opts GenerateOptions) (types.Request, error) {
	// Validate mutual exclusivity.
	if opts.Prompt != "" && len(opts.Messages) > 0 {
		return types.Request{}, types.NewConfigurationError(
			"Prompt and Messages are mutually exclusive; provide one or the other",
			nil,
		)
	}

	req := types.Request{
		Model:           opts.Model,
		Provider:        opts.Provider,
		Temperature:     opts.Temperature,
		TopP:            opts.TopP,
		MaxTokens:       opts.MaxTokens,
		StopSequences:   opts.StopSequences,
		ReasoningEffort: opts.ReasoningEffort,
		ResponseFormat:  opts.ResponseFormat,
		ToolChoice:      opts.ToolChoice,
		ProviderOptions: opts.ProviderOptions,
	}

	// Build messages.
	var messages []types.Message

	// Prepend system message if provided.
	if opts.System != "" {
		messages = append(messages, types.SystemMessage(opts.System))
	}

	if opts.Prompt != "" {
		// Convert simple prompt to a single user message.
		messages = append(messages, types.UserMessage(opts.Prompt))
	} else if len(opts.Messages) > 0 {
		messages = append(messages, opts.Messages...)
	}

	if len(messages) == 0 {
		return types.Request{}, types.NewConfigurationError(
			"either Prompt or Messages must be provided",
			nil,
		)
	}

	req.Messages = messages

	// Convert Tool definitions.
	if len(opts.Tools) > 0 {
		defs := make([]types.ToolDefinition, len(opts.Tools))
		for i, t := range opts.Tools {
			defs[i] = t.Definition()
		}
		req.Tools = defs
	}

	return req, nil
}

// buildRetryPolicy constructs a retry.Policy from the maxRetries option.
func buildRetryPolicy(maxRetries int) retry.Policy {
	p := retry.DefaultPolicy()
	if maxRetries > 0 {
		p.MaxRetries = maxRetries
	}
	return p
}

// hasExecutableTools returns true if any tool has an Execute handler.
func hasExecutableTools(tools []types.Tool) bool {
	for _, t := range tools {
		if t.Execute != nil {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Internal: tool loop (blocking)
// ---------------------------------------------------------------------------

// toolLoop implements the multi-step tool execution loop for Generate.
//
// On each iteration it:
//  1. Calls the LLM (with retry) to get a response.
//  2. Records the step.
//  3. If the response contains tool calls and there are executable tools and
//     rounds remain, it executes the tools, appends tool result messages to
//     the conversation, and loops.
//  4. Otherwise, it returns the accumulated steps.
func toolLoop(ctx context.Context, c *client.Client, req types.Request, tools []types.Tool, maxRounds int, retryPolicy retry.Policy) ([]StepResult, error) {
	var steps []StepResult
	toolMap := buildToolMap(tools)

	for round := 0; ; round++ {
		// Check context before each LLM call.
		if err := ctx.Err(); err != nil {
			return steps, types.NewAbortError("context cancelled during tool loop", err)
		}

		// Call the LLM with retry wrapping individual calls.
		resp, err := retry.Do(ctx, retryPolicy, func() (*types.Response, error) {
			return c.Complete(ctx, req)
		})
		if err != nil {
			return steps, err
		}

		// Extract tool calls and text from the response.
		respToolCalls := extractToolCalls(resp)
		step := StepResult{
			Text:         resp.Text(),
			Reasoning:    resp.Reasoning(),
			ToolCalls:    respToolCalls,
			FinishReason: resp.FinishReason,
			Usage:        resp.Usage,
			Response:     resp,
		}

		// Determine if we should execute tools this round.
		executableCalls := filterExecutableCalls(respToolCalls, toolMap)

		if len(executableCalls) > 0 && round < maxRounds {
			// Execute all tool calls concurrently.
			toolResults := executeAllTools(toolMap, executableCalls)
			step.ToolResults = toolResults

			// Append the assistant message (with tool calls) to the conversation.
			req.Messages = append(req.Messages, resp.Message)

			// Append tool result messages to the conversation.
			for _, tr := range toolResults {
				contentStr, _ := toolResultToString(tr.Content)
				req.Messages = append(req.Messages, types.ToolResultMessage(
					tr.ToolCallID,
					contentStr,
					tr.IsError,
				))
			}

			steps = append(steps, step)
			continue
		}

		// No more tool execution: final step.
		steps = append(steps, step)
		break
	}

	return steps, nil
}

// ---------------------------------------------------------------------------
// Internal: streaming tool loop
// ---------------------------------------------------------------------------

// streamToolLoop runs the streaming generation with tool loop support. Events
// are forwarded to the output channel. During tool execution rounds, the
// stream pauses while tools run and then resumes with the next LLM call.
func streamToolLoop(ctx context.Context, c *client.Client, req types.Request, tools []types.Tool, maxRounds int, out chan<- types.StreamEvent) error {
	toolMap := buildToolMap(tools)

	for round := 0; ; round++ {
		if err := ctx.Err(); err != nil {
			out <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: types.NewAbortError("context cancelled during stream tool loop", err),
			}
			return types.NewAbortError("context cancelled during stream tool loop", err)
		}

		// Start the stream.
		eventCh, err := c.Stream(ctx, req)
		if err != nil {
			out <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: err,
			}
			return err
		}

		// Accumulate the stream and forward events.
		acc := NewStreamAccumulator()
		var streamErr error

		for event := range eventCh {
			acc.Process(event)

			// Forward the event to the caller.
			select {
			case out <- event:
			case <-ctx.Done():
				return types.NewAbortError("context cancelled while forwarding stream events", ctx.Err())
			}

			if event.Type == types.StreamEventError {
				streamErr = event.Error
			}
		}

		if streamErr != nil {
			return streamErr
		}

		// Get the accumulated response.
		resp := acc.Response()
		if resp == nil {
			return nil
		}

		// Check for tool calls to execute.
		respToolCalls := extractToolCalls(resp)
		executableCalls := filterExecutableCalls(respToolCalls, toolMap)

		if len(executableCalls) > 0 && round < maxRounds {
			// Execute tools.
			toolResults := executeAllTools(toolMap, executableCalls)

			// Append assistant message and tool results to conversation.
			req.Messages = append(req.Messages, resp.Message)

			for _, tr := range toolResults {
				contentStr, _ := toolResultToString(tr.Content)
				req.Messages = append(req.Messages, types.ToolResultMessage(
					tr.ToolCallID,
					contentStr,
					tr.IsError,
				))
			}

			// Continue to the next round (which will start a new stream).
			continue
		}

		// No more tool execution, we are done.
		return nil
	}
}

// ---------------------------------------------------------------------------
// Internal: tool execution
// ---------------------------------------------------------------------------

// buildToolMap creates a name-to-tool lookup map.
func buildToolMap(tools []types.Tool) map[string]types.Tool {
	m := make(map[string]types.Tool, len(tools))
	for _, t := range tools {
		m[t.Name] = t
	}
	return m
}

// extractToolCalls extracts ToolCall values from a Response.
func extractToolCalls(resp *types.Response) []types.ToolCall {
	if resp == nil {
		return nil
	}
	rawCalls := resp.ToolCalls()
	if len(rawCalls) == 0 {
		return nil
	}
	calls := make([]types.ToolCall, len(rawCalls))
	for i, tc := range rawCalls {
		calls[i] = types.ToolCall{
			ID:        tc.ID,
			Name:      tc.Name,
			Arguments: tc.Arguments,
		}
	}
	return calls
}

// filterExecutableCalls returns only the tool calls that have a matching tool
// with an Execute handler.
func filterExecutableCalls(calls []types.ToolCall, toolMap map[string]types.Tool) []types.ToolCall {
	var executable []types.ToolCall
	for _, tc := range calls {
		if tool, ok := toolMap[tc.Name]; ok && tool.Execute != nil {
			executable = append(executable, tc)
		}
	}
	return executable
}

// executeAllTools executes tool calls concurrently using goroutines and a
// WaitGroup. Each tool call is dispatched to its corresponding Execute handler.
// Results are collected in order. If a tool is not found or has no handler, an
// error result is returned for that call.
func executeAllTools(toolMap map[string]types.Tool, toolCalls []types.ToolCall) []types.ToolResult {
	results := make([]types.ToolResult, len(toolCalls))
	var wg sync.WaitGroup

	for i, tc := range toolCalls {
		wg.Add(1)
		go func(idx int, call types.ToolCall) {
			defer wg.Done()

			tool, ok := toolMap[call.Name]
			if !ok || tool.Execute == nil {
				results[idx] = types.ToolResult{
					ToolCallID: call.ID,
					Content:    fmt.Sprintf("error: tool %q not found or has no execute handler", call.Name),
					IsError:    true,
				}
				return
			}

			// Execute the tool, recovering from panics to prevent one tool
			// from crashing the entire loop.
			content, err := safeExecuteTool(tool, call.Arguments)
			if err != nil {
				results[idx] = types.ToolResult{
					ToolCallID: call.ID,
					Content:    fmt.Sprintf("error executing tool %q: %v", call.Name, err),
					IsError:    true,
				}
				return
			}

			results[idx] = types.ToolResult{
				ToolCallID: call.ID,
				Content:    content,
				IsError:    false,
			}
		}(i, tc)
	}

	wg.Wait()
	return results
}

// safeExecuteTool calls a tool's Execute handler and recovers from panics,
// converting them to errors.
func safeExecuteTool(tool types.Tool, args map[string]any) (result string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic in tool %q: %v", tool.Name, r)
		}
	}()
	return tool.Execute(args)
}

// ---------------------------------------------------------------------------
// Internal: helpers
// ---------------------------------------------------------------------------

// buildGenerateResult constructs a GenerateResult from accumulated steps.
func buildGenerateResult(steps []StepResult) *GenerateResult {
	if len(steps) == 0 {
		return &GenerateResult{}
	}

	// The final step determines the primary result.
	final := steps[len(steps)-1]

	// Aggregate usage across all steps.
	var totalUsage types.Usage
	for _, s := range steps {
		totalUsage = totalUsage.Add(s.Usage)
	}

	// Collect all tool results across all steps.
	var allToolResults []types.ToolResult
	for _, s := range steps {
		allToolResults = append(allToolResults, s.ToolResults...)
	}

	return &GenerateResult{
		Text:         final.Text,
		Reasoning:    final.Reasoning,
		ToolCalls:    final.ToolCalls,
		ToolResults:  allToolResults,
		FinishReason: final.FinishReason,
		Usage:        final.Usage,
		TotalUsage:   totalUsage,
		Steps:        steps,
		Response:     final.Response,
	}
}

// toolResultToString converts a ToolResult.Content (which is typed as any)
// to a string suitable for inclusion in a message.
func toolResultToString(content any) (string, error) {
	switch v := content.(type) {
	case string:
		return v, nil
	case nil:
		return "", nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v), err
		}
		return string(b), nil
	}
}

// extractJSON attempts to extract a JSON object or array from text that may
// be wrapped in markdown code fences (e.g. ```json ... ```).
func extractJSON(text string) string {
	// Try to find JSON in markdown code fences.
	fenceStart := strings.Index(text, "```")
	if fenceStart < 0 {
		return ""
	}

	// Skip the opening fence and any language tag.
	afterFence := text[fenceStart+3:]
	newline := strings.IndexByte(afterFence, '\n')
	if newline < 0 {
		return ""
	}
	content := afterFence[newline+1:]

	// Find the closing fence.
	fenceEnd := strings.Index(content, "```")
	if fenceEnd < 0 {
		return ""
	}

	return strings.TrimSpace(content[:fenceEnd])
}
