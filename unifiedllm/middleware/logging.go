// Package middleware provides common middleware implementations for the
// unified LLM client. Middleware wraps provider calls in an onion pattern,
// allowing cross-cutting concerns like logging, cost tracking, and rate
// limiting to be applied uniformly across all providers.
package middleware

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/strongdm/attractor-go/unifiedllm/types"
)

// Logging returns a middleware that logs request/response details including
// the model, provider, latency, token usage, and finish reason. Errors are
// logged at the error level. All output goes to the standard logger.
//
// This middleware is intended for development and debugging. Production
// deployments should use structured logging middleware tailored to their
// observability stack.
func Logging() func(ctx context.Context, req types.Request, next func(context.Context, types.Request) (*types.Response, error)) (*types.Response, error) {
	return func(ctx context.Context, req types.Request, next func(context.Context, types.Request) (*types.Response, error)) (*types.Response, error) {
		start := time.Now()

		provider := req.Provider
		if provider == "" {
			provider = "(auto)"
		}

		log.Printf("[unifiedllm] request: model=%s provider=%s messages=%d tools=%d",
			req.Model, provider, len(req.Messages), len(req.Tools))

		resp, err := next(ctx, req)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("[unifiedllm] error: model=%s provider=%s elapsed=%s error=%v",
				req.Model, provider, elapsed, err)
			return nil, err
		}

		log.Printf("[unifiedllm] response: model=%s provider=%s elapsed=%s finish=%s input_tokens=%d output_tokens=%d total_tokens=%d",
			resp.Model, resp.Provider, elapsed, resp.FinishReason.Reason,
			resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)

		if len(resp.Warnings) > 0 {
			for _, w := range resp.Warnings {
				log.Printf("[unifiedllm] warning: code=%s message=%s", w.Code, w.Message)
			}
		}

		return resp, nil
	}
}

// CostTracker accumulates token usage across multiple requests. It is safe
// for concurrent use. Attach its Middleware method to the client to
// automatically track usage for every completion request.
type CostTracker struct {
	TotalUsage types.Usage
	mu         sync.Mutex
}

// Middleware returns a middleware function that records token usage from each
// successful response into the tracker's TotalUsage. The middleware does not
// modify the request or response; it only observes.
func (ct *CostTracker) Middleware() func(ctx context.Context, req types.Request, next func(context.Context, types.Request) (*types.Response, error)) (*types.Response, error) {
	return func(ctx context.Context, req types.Request, next func(context.Context, types.Request) (*types.Response, error)) (*types.Response, error) {
		resp, err := next(ctx, req)
		if err != nil {
			return nil, err
		}

		ct.mu.Lock()
		ct.TotalUsage = ct.TotalUsage.Add(resp.Usage)
		ct.mu.Unlock()

		return resp, nil
	}
}

// GetUsage returns a snapshot of the current accumulated usage. Safe for
// concurrent access.
func (ct *CostTracker) GetUsage() types.Usage {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.TotalUsage
}

// Reset zeroes out the accumulated usage. Safe for concurrent access.
func (ct *CostTracker) Reset() {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.TotalUsage = types.Usage{}
}
