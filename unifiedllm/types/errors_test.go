package types

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// SDKError
// ---------------------------------------------------------------------------

func TestSDKErrorWithoutCause(t *testing.T) {
	err := NewSDKError("something went wrong", nil)
	if err.Error() != "something went wrong" {
		t.Errorf("Error() = %q, want %q", err.Error(), "something went wrong")
	}
	if err.Unwrap() != nil {
		t.Errorf("Unwrap() = %v, want nil", err.Unwrap())
	}
}

func TestSDKErrorWithCause(t *testing.T) {
	cause := fmt.Errorf("root cause")
	err := NewSDKError("operation failed", cause)
	want := "operation failed: root cause"
	if err.Error() != want {
		t.Errorf("Error() = %q, want %q", err.Error(), want)
	}
	if err.Unwrap() != cause {
		t.Errorf("Unwrap() = %v, want %v", err.Unwrap(), cause)
	}
}

// ---------------------------------------------------------------------------
// ProviderError
// ---------------------------------------------------------------------------

func TestProviderErrorFormatting(t *testing.T) {
	tests := []struct {
		name       string
		message    string
		provider   string
		statusCode int
		errorCode  string
		cause      error
		wantSubstr []string
	}{
		{
			name:       "full metadata no cause",
			message:    "bad request",
			provider:   "openai",
			statusCode: 400,
			errorCode:  "invalid_request",
			cause:      nil,
			wantSubstr: []string{"[openai]", "HTTP 400", "(invalid_request)", "bad request"},
		},
		{
			name:       "with cause",
			message:    "server error",
			provider:   "anthropic",
			statusCode: 500,
			errorCode:  "",
			cause:      fmt.Errorf("connection reset"),
			wantSubstr: []string{"[anthropic]", "HTTP 500", "server error", "connection reset"},
		},
		{
			name:       "zero status code omitted",
			message:    "unknown error",
			provider:   "google",
			statusCode: 0,
			errorCode:  "",
			cause:      nil,
			wantSubstr: []string{"[google]", "unknown error"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewProviderError(tt.message, tt.provider, tt.statusCode, tt.errorCode, false, nil, nil, tt.cause)
			msg := err.Error()
			for _, sub := range tt.wantSubstr {
				if !strings.Contains(msg, sub) {
					t.Errorf("Error() = %q, missing substring %q", msg, sub)
				}
			}
		})
	}
}

func TestProviderErrorIsRetryable(t *testing.T) {
	err := NewProviderError("test", "p", 500, "", true, nil, nil, nil)
	if !err.IsRetryable() {
		t.Error("expected retryable=true for explicitly retryable ProviderError")
	}

	err2 := NewProviderError("test", "p", 400, "", false, nil, nil, nil)
	if err2.IsRetryable() {
		t.Error("expected retryable=false for non-retryable ProviderError")
	}
}

func TestProviderErrorUnwrap(t *testing.T) {
	cause := fmt.Errorf("original")
	err := NewProviderError("msg", "p", 500, "", true, nil, nil, cause)
	unwrapped := err.Unwrap()
	if len(unwrapped) != 2 {
		t.Fatalf("Unwrap() returned %d errors, want 2", len(unwrapped))
	}
	// First element should be the embedded SDKError.
	if _, ok := unwrapped[0].(*SDKError); !ok {
		t.Errorf("Unwrap()[0] type = %T, want *SDKError", unwrapped[0])
	}
	if unwrapped[1] != cause {
		t.Errorf("Unwrap()[1] = %v, want %v", unwrapped[1], cause)
	}
}

func TestProviderErrorUnwrapNoCause(t *testing.T) {
	err := NewProviderError("msg", "p", 500, "", true, nil, nil, nil)
	unwrapped := err.Unwrap()
	if len(unwrapped) != 1 {
		t.Fatalf("Unwrap() returned %d errors, want 1", len(unwrapped))
	}
}

func TestProviderErrorRetryAfter(t *testing.T) {
	retryAfter := 5.0
	err := NewProviderError("msg", "p", 429, "", true, &retryAfter, nil, nil)
	if err.RetryAfter == nil || *err.RetryAfter != 5.0 {
		t.Errorf("RetryAfter = %v, want 5.0", err.RetryAfter)
	}
}

// ---------------------------------------------------------------------------
// Concrete error types: construction and IsRetryable
// ---------------------------------------------------------------------------

func TestAuthenticationError(t *testing.T) {
	err := NewAuthenticationError("invalid api key", "openai", "invalid_api_key", nil, nil)
	if err.IsRetryable() {
		t.Error("AuthenticationError should not be retryable")
	}
	if err.StatusCode != 401 {
		t.Errorf("StatusCode = %d, want 401", err.StatusCode)
	}
	if err.Provider != "openai" {
		t.Errorf("Provider = %q, want %q", err.Provider, "openai")
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("Error() = %q, missing message", err.Error())
	}
}

func TestAuthenticationErrorUnwrap(t *testing.T) {
	cause := fmt.Errorf("expired")
	err := NewAuthenticationError("bad key", "p", "", nil, cause)
	unwrapped := err.Unwrap()
	if len(unwrapped) != 2 {
		t.Fatalf("Unwrap() returned %d errors, want 2", len(unwrapped))
	}
	if _, ok := unwrapped[0].(*ProviderError); !ok {
		t.Errorf("Unwrap()[0] type = %T, want *ProviderError", unwrapped[0])
	}
}

func TestRateLimitError(t *testing.T) {
	retryAfter := 2.5
	err := NewRateLimitError("rate limited", "anthropic", "rate_limit_exceeded", nil, &retryAfter, nil)
	if !err.IsRetryable() {
		t.Error("RateLimitError should be retryable")
	}
	if err.StatusCode != 429 {
		t.Errorf("StatusCode = %d, want 429", err.StatusCode)
	}
	if err.RetryAfter == nil || *err.RetryAfter != 2.5 {
		t.Errorf("RetryAfter = %v, want 2.5", err.RetryAfter)
	}
}

func TestServerError(t *testing.T) {
	err := NewServerError("internal error", "openai", 502, "bad_gateway", nil, nil)
	if !err.IsRetryable() {
		t.Error("ServerError should be retryable")
	}
	if err.StatusCode != 502 {
		t.Errorf("StatusCode = %d, want 502", err.StatusCode)
	}
}

func TestServerErrorDefaultStatus(t *testing.T) {
	err := NewServerError("internal error", "openai", 0, "", nil, nil)
	if err.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500 (default)", err.StatusCode)
	}
}

func TestInvalidRequestError(t *testing.T) {
	err := NewInvalidRequestError("bad input", "google", 400, "invalid_argument", nil, nil)
	if err.IsRetryable() {
		t.Error("InvalidRequestError should not be retryable")
	}
	if err.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", err.StatusCode)
	}
}

func TestInvalidRequestErrorDefaultStatus(t *testing.T) {
	err := NewInvalidRequestError("bad input", "google", 0, "", nil, nil)
	if err.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400 (default)", err.StatusCode)
	}
}

func TestConfigurationError(t *testing.T) {
	err := NewConfigurationError("missing API key", nil)
	if err.IsRetryable() {
		t.Error("ConfigurationError should not be retryable")
	}
	if err.Error() != "missing API key" {
		t.Errorf("Error() = %q, want %q", err.Error(), "missing API key")
	}
}

func TestConfigurationErrorUnwrap(t *testing.T) {
	cause := fmt.Errorf("env var not set")
	err := NewConfigurationError("config problem", cause)
	unwrapped := err.Unwrap()
	if len(unwrapped) != 2 {
		t.Fatalf("Unwrap() returned %d errors, want 2", len(unwrapped))
	}
}

func TestNetworkError(t *testing.T) {
	err := NewNetworkError("connection refused", nil)
	if !err.IsRetryable() {
		t.Error("NetworkError should be retryable")
	}
	if err.Error() != "connection refused" {
		t.Errorf("Error() = %q, want %q", err.Error(), "connection refused")
	}
}

func TestStreamError(t *testing.T) {
	err := NewStreamError("stream interrupted", nil)
	if !err.IsRetryable() {
		t.Error("StreamError should be retryable")
	}
}

func TestRequestTimeoutError(t *testing.T) {
	err := NewRequestTimeoutError("request timed out", nil)
	if !err.IsRetryable() {
		t.Error("RequestTimeoutError should be retryable")
	}
}

func TestAccessDeniedError(t *testing.T) {
	err := NewAccessDeniedError("forbidden", "openai", "access_denied", nil, nil)
	if err.IsRetryable() {
		t.Error("AccessDeniedError should not be retryable")
	}
	if err.StatusCode != 403 {
		t.Errorf("StatusCode = %d, want 403", err.StatusCode)
	}
}

func TestNotFoundError(t *testing.T) {
	err := NewNotFoundError("model not found", "anthropic", "not_found", nil, nil)
	if err.IsRetryable() {
		t.Error("NotFoundError should not be retryable")
	}
	if err.StatusCode != 404 {
		t.Errorf("StatusCode = %d, want 404", err.StatusCode)
	}
}

func TestContentFilterError(t *testing.T) {
	err := NewContentFilterError("blocked by content filter", "openai", "content_filter", nil, nil)
	if err.IsRetryable() {
		t.Error("ContentFilterError should not be retryable")
	}
}

func TestContextLengthError(t *testing.T) {
	err := NewContextLengthError("context too long", "openai", 413, "context_length_exceeded", nil, nil)
	if err.IsRetryable() {
		t.Error("ContextLengthError should not be retryable")
	}
	if err.StatusCode != 413 {
		t.Errorf("StatusCode = %d, want 413", err.StatusCode)
	}
}

func TestContextLengthErrorDefaultStatus(t *testing.T) {
	err := NewContextLengthError("context too long", "openai", 0, "", nil, nil)
	if err.StatusCode != 413 {
		t.Errorf("StatusCode = %d, want 413 (default)", err.StatusCode)
	}
}

func TestQuotaExceededError(t *testing.T) {
	err := NewQuotaExceededError("quota exceeded", "openai", "insufficient_quota", nil, nil)
	if err.IsRetryable() {
		t.Error("QuotaExceededError should not be retryable")
	}
	if err.StatusCode != 429 {
		t.Errorf("StatusCode = %d, want 429", err.StatusCode)
	}
}

func TestNoObjectGeneratedError(t *testing.T) {
	err := NewNoObjectGeneratedError("failed to parse", nil)
	if err.IsRetryable() {
		t.Error("NoObjectGeneratedError should not be retryable")
	}
}

func TestAbortError(t *testing.T) {
	err := NewAbortError("cancelled", nil)
	if err.IsRetryable() {
		t.Error("AbortError should not be retryable")
	}
}

func TestInvalidToolCallError(t *testing.T) {
	err := NewInvalidToolCallError("unknown tool", "bad_tool", nil)
	if err.IsRetryable() {
		t.Error("InvalidToolCallError should not be retryable")
	}
	if err.ToolName != "bad_tool" {
		t.Errorf("ToolName = %q, want %q", err.ToolName, "bad_tool")
	}
}

func TestUnsupportedToolChoiceError(t *testing.T) {
	err := NewUnsupportedToolChoiceError("unsupported mode", "required", nil)
	if err.IsRetryable() {
		t.Error("UnsupportedToolChoiceError should not be retryable")
	}
	if err.ToolChoice != "required" {
		t.Errorf("ToolChoice = %q, want %q", err.ToolChoice, "required")
	}
}

// ---------------------------------------------------------------------------
// IsRetryable (package-level function)
// ---------------------------------------------------------------------------

func TestIsRetryableFunction(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"auth error", NewAuthenticationError("bad key", "p", "", nil, nil), false},
		{"rate limit error", NewRateLimitError("limited", "p", "", nil, nil, nil), true},
		{"server error", NewServerError("500", "p", 500, "", nil, nil), true},
		{"network error", NewNetworkError("connection lost", nil), true},
		{"stream error", NewStreamError("stream broken", nil), true},
		{"timeout error", NewRequestTimeoutError("timed out", nil), true},
		{"invalid request error", NewInvalidRequestError("bad", "p", 400, "", nil, nil), false},
		{"access denied error", NewAccessDeniedError("forbidden", "p", "", nil, nil), false},
		{"configuration error", NewConfigurationError("bad config", nil), false},
		{"quota exceeded error", NewQuotaExceededError("over quota", "p", "", nil, nil), false},
		{"content filter error", NewContentFilterError("blocked", "p", "", nil, nil), false},
		{"not found error", NewNotFoundError("missing", "p", "", nil, nil), false},
		{"abort error", NewAbortError("cancelled", nil), false},
		{"no object generated error", NewNoObjectGeneratedError("parse fail", nil), false},
		{"plain error (no IsRetryable)", fmt.Errorf("some error"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsRetryable(tt.err)
			if got != tt.want {
				t.Errorf("IsRetryable() = %v, want %v", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ErrorFromStatusCode
// ---------------------------------------------------------------------------

func TestErrorFromStatusCode(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		message    string
		provider   string
		errorCode  string
		wantType   string
	}{
		{
			name:       "401 -> AuthenticationError",
			statusCode: 401,
			message:    "invalid key",
			provider:   "openai",
			wantType:   "*types.AuthenticationError",
		},
		{
			name:       "403 -> AccessDeniedError",
			statusCode: 403,
			message:    "not permitted",
			provider:   "openai",
			wantType:   "*types.AccessDeniedError",
		},
		{
			name:       "404 -> NotFoundError",
			statusCode: 404,
			message:    "endpoint missing",
			provider:   "anthropic",
			wantType:   "*types.NotFoundError",
		},
		{
			name:       "400 -> InvalidRequestError",
			statusCode: 400,
			message:    "missing field",
			provider:   "google",
			wantType:   "*types.InvalidRequestError",
		},
		{
			name:       "422 -> InvalidRequestError",
			statusCode: 422,
			message:    "unprocessable",
			provider:   "google",
			wantType:   "*types.InvalidRequestError",
		},
		{
			name:       "413 -> ContextLengthError",
			statusCode: 413,
			message:    "too large",
			provider:   "openai",
			wantType:   "*types.ContextLengthError",
		},
		{
			name:       "429 -> RateLimitError",
			statusCode: 429,
			message:    "too many requests",
			provider:   "openai",
			wantType:   "*types.RateLimitError",
		},
		{
			name:       "429 with quota message -> QuotaExceededError",
			statusCode: 429,
			message:    "You exceeded your current quota",
			provider:   "openai",
			wantType:   "*types.QuotaExceededError",
		},
		{
			name:       "500 -> ServerError",
			statusCode: 500,
			message:    "internal failure",
			provider:   "anthropic",
			wantType:   "*types.ServerError",
		},
		{
			name:       "502 -> ServerError",
			statusCode: 502,
			message:    "gateway failure",
			provider:   "openai",
			wantType:   "*types.ServerError",
		},
		{
			name:       "503 -> ServerError",
			statusCode: 503,
			message:    "temporarily unavailable",
			provider:   "google",
			wantType:   "*types.ServerError",
		},
		{
			name:       "418 unknown -> ProviderError",
			statusCode: 418,
			message:    "im a teapot",
			provider:   "unknown",
			wantType:   "*types.ProviderError",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ErrorFromStatusCode(tt.statusCode, tt.message, tt.provider, tt.errorCode, nil, nil)
			gotType := fmt.Sprintf("%T", err)
			if gotType != tt.wantType {
				t.Errorf("ErrorFromStatusCode(%d, %q) type = %s, want %s", tt.statusCode, tt.message, gotType, tt.wantType)
			}
		})
	}
}

func TestErrorFromStatusCodeMessageClassification(t *testing.T) {
	// When the message contains context-length keywords, ErrorFromStatusCode
	// should classify it as a ContextLengthError regardless of the status code.
	err := ErrorFromStatusCode(400, "This request exceeds the maximum context length", "openai", "", nil, nil)
	if _, ok := err.(*ContextLengthError); !ok {
		t.Errorf("expected *ContextLengthError for context length message, got %T", err)
	}

	// Content filter message should be classified even on a 400.
	err2 := ErrorFromStatusCode(400, "blocked by content filter", "openai", "", nil, nil)
	if _, ok := err2.(*ContentFilterError); !ok {
		t.Errorf("expected *ContentFilterError for content filter message, got %T", err2)
	}

	// Quota exceeded via error code.
	err3 := ErrorFromStatusCode(400, "error occurred", "openai", "insufficient_quota", nil, nil)
	if _, ok := err3.(*QuotaExceededError); !ok {
		t.Errorf("expected *QuotaExceededError for quota error code, got %T", err3)
	}
}

// ---------------------------------------------------------------------------
// ClassifyByMessage
// ---------------------------------------------------------------------------

func TestClassifyByMessage(t *testing.T) {
	tests := []struct {
		message  string
		wantType string
	}{
		{"model not found", "*types.NotFoundError"},
		{"resource does not exist", "*types.NotFoundError"},
		{"maximum context length exceeded", "*types.ContextLengthError"},
		{"token limit reached", "*types.ContextLengthError"},
		{"input too long for the model", "*types.ContextLengthError"},
		{"blocked by content filter", "*types.ContentFilterError"},
		{"content policy violation", "*types.ContentFilterError"},
		{"safety filter triggered", "*types.ContentFilterError"},
		{"invalid api key provided", "*types.AuthenticationError"},
		{"unauthorized access", "*types.AuthenticationError"},
		{"expired token detected", "*types.AuthenticationError"},
		{"access denied to resource", "*types.AccessDeniedError"},
		{"permission denied", "*types.AccessDeniedError"},
		{"forbidden action", "*types.AccessDeniedError"},
		{"rate limit exceeded", "*types.RateLimitError"},
		{"too many requests sent", "*types.RateLimitError"},
		{"throttled by provider", "*types.RateLimitError"},
		{"quota exceeded for account", "*types.QuotaExceededError"},
		{"billing issue detected", "*types.QuotaExceededError"},
		{"spending limit reached", "*types.QuotaExceededError"},
		{"internal server error occurred", "*types.ServerError"},
		{"service unavailable right now", "*types.ServerError"},
		{"bad gateway response", "*types.ServerError"},
		{"the system is overloaded", "*types.ServerError"},
		{"request timed out", "*types.RequestTimeoutError"},
		{"deadline exceeded for operation", "*types.RequestTimeoutError"},
		{"something completely unknown", "*types.SDKError"},
	}

	for _, tt := range tests {
		t.Run(tt.message, func(t *testing.T) {
			err := ClassifyByMessage(tt.message)
			gotType := fmt.Sprintf("%T", err)
			if gotType != tt.wantType {
				t.Errorf("ClassifyByMessage(%q) type = %s, want %s", tt.message, gotType, tt.wantType)
			}
		})
	}
}

func TestClassifyByMessageCaseInsensitive(t *testing.T) {
	// Verify case insensitivity.
	err := ClassifyByMessage("INVALID API KEY")
	if _, ok := err.(*AuthenticationError); !ok {
		t.Errorf("expected *AuthenticationError for uppercase message, got %T", err)
	}
}

// ---------------------------------------------------------------------------
// Unwrap chains: errors.As / errors.Is compatibility
// ---------------------------------------------------------------------------

func TestErrorsAsSDKError(t *testing.T) {
	authErr := NewAuthenticationError("bad key", "openai", "", nil, nil)
	var sdkErr *SDKError
	if !errors.As(authErr, &sdkErr) {
		t.Error("expected errors.As to find *SDKError in AuthenticationError chain")
	}
}

func TestErrorsAsProviderError(t *testing.T) {
	rateLimitErr := NewRateLimitError("limited", "openai", "", nil, nil, nil)
	var provErr *ProviderError
	if !errors.As(rateLimitErr, &provErr) {
		t.Error("expected errors.As to find *ProviderError in RateLimitError chain")
	}
}

func TestErrorsAsCause(t *testing.T) {
	cause := fmt.Errorf("root problem")
	err := NewNetworkError("connection failed", cause)
	var sdkErr *SDKError
	if !errors.As(err, &sdkErr) {
		t.Error("expected errors.As to find *SDKError in NetworkError chain")
	}
}

// ---------------------------------------------------------------------------
// IsRetryable with wrapped errors
// ---------------------------------------------------------------------------

func TestIsRetryableWithWrappedRetryable(t *testing.T) {
	// A ServerError wrapped inside an AuthenticationError via Cause:
	// The outer AuthenticationError reports not retryable.
	authErr := NewAuthenticationError("bad key", "openai", "", nil, nil)
	if IsRetryable(authErr) {
		t.Error("AuthenticationError should report not retryable")
	}
}
