// Package client provides the unified LLM client that routes requests to
// provider-specific adapters and applies middleware in an onion pattern.
//
// The Client is the main entry point for application code. It abstracts away
// the differences between LLM providers behind a common Request/Response
// interface, allowing callers to switch providers without changing application
// logic.
package client

import (
	"context"
	"fmt"
	"os"

	"github.com/strongdm/attractor-go/unifiedllm/catalog"
	"github.com/strongdm/attractor-go/unifiedllm/types"
)

// ProviderAdapter is the interface that every provider backend must implement.
// Each adapter translates between the unified types and the provider's native
// API format.
type ProviderAdapter interface {
	// Name returns the canonical provider name (e.g. "openai", "anthropic", "gemini").
	Name() string

	// Complete sends a blocking completion request and returns the full response.
	Complete(ctx context.Context, req types.Request) (*types.Response, error)

	// Stream sends a streaming request and returns a channel of events.
	// The channel is closed when the stream ends or an error occurs.
	// Errors are delivered as StreamEvent with Type == StreamEventError.
	Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error)

	// Close releases any resources held by the adapter (HTTP clients, connections, etc.).
	Close() error
}

// Middleware wraps provider calls in an onion pattern. Each middleware receives
// the context, request, and a next function to call the inner layer. Middleware
// can inspect/modify the request, call next, and inspect/modify the response.
type Middleware func(ctx context.Context, req types.Request, next func(context.Context, types.Request) (*types.Response, error)) (*types.Response, error)

// Client holds registered providers, routes requests to the appropriate adapter,
// and applies middleware to every call.
type Client struct {
	providers       map[string]ProviderAdapter
	defaultProvider string
	middleware      []Middleware
}

// Option configures a Client during construction.
type Option func(*Client)

// WithProvider registers a provider adapter under the given name.
func WithProvider(name string, adapter ProviderAdapter) Option {
	return func(c *Client) {
		c.providers[name] = adapter
	}
}

// WithDefaultProvider sets the default provider used when a request does not
// specify a provider and the model is not in the catalog.
func WithDefaultProvider(name string) Option {
	return func(c *Client) {
		c.defaultProvider = name
	}
}

// WithMiddleware appends one or more middleware functions to the chain.
// Middleware is applied in the order provided (first added = outermost layer).
func WithMiddleware(m ...Middleware) Option {
	return func(c *Client) {
		c.middleware = append(c.middleware, m...)
	}
}

// New creates a Client with the given options. At least one provider must be
// registered via WithProvider or the client will be unable to serve requests.
func New(opts ...Option) *Client {
	c := &Client{
		providers: make(map[string]ProviderAdapter),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// FromEnv creates a Client by inspecting environment variables for known
// provider API keys. The following variables are checked:
//   - OPENAI_API_KEY: registers the OpenAI provider
//   - ANTHROPIC_API_KEY: registers the Anthropic provider
//   - GEMINI_API_KEY: registers the Gemini provider
//
// Only providers with non-empty keys are registered. The first registered
// provider becomes the default. Returns an error if no provider keys are found.
func FromEnv() (*Client, error) {
	c := &Client{
		providers: make(map[string]ProviderAdapter),
	}

	// Track registration order so the first provider becomes default.
	var firstProvider string

	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		_ = key
		firstProvider = "openai"
	}

	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		_ = key
		if firstProvider == "" {
			firstProvider = "anthropic"
		}
	}

	if key := os.Getenv("GEMINI_API_KEY"); key != "" {
		_ = key
		if firstProvider == "" {
			firstProvider = "gemini"
		}
	}

	if len(c.providers) == 0 && firstProvider == "" {
		return nil, types.NewConfigurationError(
			"no LLM provider API keys found in environment; set at least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY",
			nil,
		)
	}

	if c.defaultProvider == "" {
		c.defaultProvider = firstProvider
	}

	return c, nil
}

// RegisterProvider registers a provider adapter at runtime.
func (c *Client) RegisterProvider(name string, adapter ProviderAdapter) {
	c.providers[name] = adapter
	if c.defaultProvider == "" {
		c.defaultProvider = name
	}
}

// SetDefaultProvider changes the default provider at runtime.
func (c *Client) SetDefaultProvider(name string) {
	c.defaultProvider = name
}

// Complete sends a blocking completion request. The request is routed to the
// appropriate provider based on the Provider field, the Model field (via catalog
// lookup), or the default provider. Middleware is applied around the call.
func (c *Client) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
	adapter, err := c.resolveProvider(req)
	if err != nil {
		return nil, err
	}

	final := func(ctx context.Context, req types.Request) (*types.Response, error) {
		return adapter.Complete(ctx, req)
	}

	return c.applyMiddleware(ctx, req, final)
}

// Stream sends a streaming request and returns a channel of events.
func (c *Client) Stream(ctx context.Context, req types.Request) (<-chan types.StreamEvent, error) {
	adapter, err := c.resolveProvider(req)
	if err != nil {
		return nil, err
	}
	return adapter.Stream(ctx, req)
}

// Close releases all registered provider resources.
func (c *Client) Close() error {
	var errs []error
	for name, adapter := range c.providers {
		if err := adapter.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing provider %s: %w", name, err))
		}
	}
	if len(errs) > 0 {
		msg := fmt.Sprintf("errors closing %d provider(s)", len(errs))
		combined := errs[0]
		for _, e := range errs[1:] {
			combined = fmt.Errorf("%w; %w", combined, e)
		}
		return types.NewSDKError(msg, combined)
	}
	return nil
}

// resolveProvider determines which adapter to use for the given request.
func (c *Client) resolveProvider(req types.Request) (ProviderAdapter, error) {
	providerName := req.Provider

	if providerName == "" && req.Model != "" {
		if info := catalog.GetModelInfo(req.Model); info != nil {
			providerName = info.Provider
		}
	}

	if providerName == "" {
		providerName = c.defaultProvider
	}

	if providerName == "" {
		return nil, types.NewConfigurationError(
			"cannot resolve provider: no provider specified, model not found in catalog, and no default provider configured",
			nil,
		)
	}

	adapter, ok := c.providers[providerName]
	if !ok {
		return nil, types.NewConfigurationError(
			fmt.Sprintf("provider %q is not registered; registered providers: %v", providerName, c.providerNames()),
			nil,
		)
	}

	return adapter, nil
}

// applyMiddleware chains middleware in onion pattern.
func (c *Client) applyMiddleware(ctx context.Context, req types.Request, final func(context.Context, types.Request) (*types.Response, error)) (*types.Response, error) {
	if len(c.middleware) == 0 {
		return final(ctx, req)
	}

	handler := final
	for i := len(c.middleware) - 1; i >= 0; i-- {
		mw := c.middleware[i]
		next := handler
		handler = func(ctx context.Context, req types.Request) (*types.Response, error) {
			return mw(ctx, req, next)
		}
	}

	return handler(ctx, req)
}

// providerNames returns the list of registered provider names.
func (c *Client) providerNames() []string {
	names := make([]string, 0, len(c.providers))
	for name := range c.providers {
		names = append(names, name)
	}
	return names
}
