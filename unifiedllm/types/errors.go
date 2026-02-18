// Package types defines the shared type system for the unified LLM client,
// including a structured error hierarchy that classifies failures from LLM
// providers and the SDK itself into actionable, retryable categories.
package types

import (
	"fmt"
	"net/http"
	"strings"
)

// ---------------------------------------------------------------------------
// Base error
// ---------------------------------------------------------------------------

// SDKError is the root of the error hierarchy. Every error surfaced by the
// unified LLM client either is an SDKError or embeds one.
type SDKError struct {
	Message string
	Cause   error
}

func (e *SDKError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

func (e *SDKError) Unwrap() error { return e.Cause }

// NewSDKError constructs a base SDK error.
func NewSDKError(message string, cause error) *SDKError {
	return &SDKError{Message: message, Cause: cause}
}

// ---------------------------------------------------------------------------
// Provider error (abstract base for all provider-originated failures)
// ---------------------------------------------------------------------------

// ProviderError represents an error returned by an LLM provider's API.
type ProviderError struct {
	SDKError
	Provider   string
	StatusCode int
	ErrorCode  string
	Retryable  bool
	RetryAfter *float64
	Raw        map[string]any
}

func (e *ProviderError) Error() string {
	parts := []string{fmt.Sprintf("[%s]", e.Provider)}
	if e.StatusCode != 0 {
		parts = append(parts, fmt.Sprintf("HTTP %d", e.StatusCode))
	}
	if e.ErrorCode != "" {
		parts = append(parts, fmt.Sprintf("(%s)", e.ErrorCode))
	}
	parts = append(parts, e.Message)
	msg := strings.Join(parts, " ")
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", msg, e.Cause)
	}
	return msg
}

func (e *ProviderError) IsRetryable() bool { return e.Retryable }

func (e *ProviderError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// NewProviderError constructs a ProviderError with the supplied metadata.
func NewProviderError(message, provider string, statusCode int, errorCode string, retryable bool, retryAfter *float64, raw map[string]any, cause error) *ProviderError {
	return &ProviderError{
		SDKError:   SDKError{Message: message, Cause: cause},
		Provider:   provider,
		StatusCode: statusCode,
		ErrorCode:  errorCode,
		Retryable:  retryable,
		RetryAfter: retryAfter,
		Raw:        raw,
	}
}

// ---------------------------------------------------------------------------
// Specific provider errors
// ---------------------------------------------------------------------------

// AuthenticationError indicates the request was rejected because the API key
// or token is missing, invalid, or expired (HTTP 401).
type AuthenticationError struct{ ProviderError }

func NewAuthenticationError(message, provider, errorCode string, raw map[string]any, cause error) *AuthenticationError {
	return &AuthenticationError{
		ProviderError: *NewProviderError(message, provider, http.StatusUnauthorized, errorCode, false, nil, raw, cause),
	}
}

func (e *AuthenticationError) IsRetryable() bool { return false }

func (e *AuthenticationError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// RateLimitError indicates the caller has exceeded the provider's rate limit (HTTP 429).
type RateLimitError struct{ ProviderError }

func NewRateLimitError(message, provider, errorCode string, raw map[string]any, retryAfter *float64, cause error) *RateLimitError {
	return &RateLimitError{
		ProviderError: *NewProviderError(message, provider, http.StatusTooManyRequests, errorCode, true, retryAfter, raw, cause),
	}
}

func (e *RateLimitError) IsRetryable() bool { return true }

func (e *RateLimitError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// ServerError indicates the provider returned an internal server error (HTTP 500-599).
type ServerError struct{ ProviderError }

func NewServerError(message, provider string, statusCode int, errorCode string, raw map[string]any, cause error) *ServerError {
	if statusCode == 0 {
		statusCode = http.StatusInternalServerError
	}
	return &ServerError{
		ProviderError: *NewProviderError(message, provider, statusCode, errorCode, true, nil, raw, cause),
	}
}

func (e *ServerError) IsRetryable() bool { return true }

func (e *ServerError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// InvalidRequestError indicates the request was malformed (HTTP 400 or 422).
type InvalidRequestError struct{ ProviderError }

func NewInvalidRequestError(message, provider string, statusCode int, errorCode string, raw map[string]any, cause error) *InvalidRequestError {
	if statusCode == 0 {
		statusCode = http.StatusBadRequest
	}
	return &InvalidRequestError{
		ProviderError: *NewProviderError(message, provider, statusCode, errorCode, false, nil, raw, cause),
	}
}

func (e *InvalidRequestError) IsRetryable() bool { return false }

func (e *InvalidRequestError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// ConfigurationError indicates the SDK was misconfigured.
type ConfigurationError struct{ SDKError }

func NewConfigurationError(message string, cause error) *ConfigurationError {
	return &ConfigurationError{
		SDKError: SDKError{Message: message, Cause: cause},
	}
}

func (e *ConfigurationError) IsRetryable() bool { return false }

func (e *ConfigurationError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// NoObjectGeneratedError indicates that structured output extraction failed.
type NoObjectGeneratedError struct{ SDKError }

func NewNoObjectGeneratedError(message string, cause error) *NoObjectGeneratedError {
	return &NoObjectGeneratedError{
		SDKError: SDKError{Message: message, Cause: cause},
	}
}

func (e *NoObjectGeneratedError) IsRetryable() bool { return false }

func (e *NoObjectGeneratedError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// AbortError indicates the request was explicitly cancelled by the caller.
type AbortError struct{ SDKError }

func NewAbortError(message string, cause error) *AbortError {
	return &AbortError{
		SDKError: SDKError{Message: message, Cause: cause},
	}
}

func (e *AbortError) IsRetryable() bool { return false }

func (e *AbortError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// NetworkError indicates a transport-level failure such as DNS resolution
// failure, connection refused, or TLS handshake error.
type NetworkError struct{ SDKError }

func NewNetworkError(message string, cause error) *NetworkError {
	return &NetworkError{
		SDKError: SDKError{Message: message, Cause: cause},
	}
}

func (e *NetworkError) IsRetryable() bool { return true }

func (e *NetworkError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// StreamError indicates a failure that occurred while reading a streaming response.
type StreamError struct{ SDKError }

func NewStreamError(message string, cause error) *StreamError {
	return &StreamError{
		SDKError: SDKError{Message: message, Cause: cause},
	}
}

func (e *StreamError) IsRetryable() bool { return true }

func (e *StreamError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// RequestTimeoutError indicates the request exceeded the configured timeout.
type RequestTimeoutError struct{ SDKError }

func NewRequestTimeoutError(message string, cause error) *RequestTimeoutError {
	return &RequestTimeoutError{
		SDKError: SDKError{Message: message, Cause: cause},
	}
}

func (e *RequestTimeoutError) IsRetryable() bool { return true }

func (e *RequestTimeoutError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// AccessDeniedError indicates the authenticated identity lacks permission (HTTP 403).
type AccessDeniedError struct{ ProviderError }

func NewAccessDeniedError(message, provider, errorCode string, raw map[string]any, cause error) *AccessDeniedError {
	return &AccessDeniedError{
		ProviderError: *NewProviderError(message, provider, http.StatusForbidden, errorCode, false, nil, raw, cause),
	}
}

func (e *AccessDeniedError) IsRetryable() bool { return false }

func (e *AccessDeniedError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// NotFoundError indicates the requested resource does not exist (HTTP 404).
type NotFoundError struct{ ProviderError }

func NewNotFoundError(message, provider, errorCode string, raw map[string]any, cause error) *NotFoundError {
	return &NotFoundError{
		ProviderError: *NewProviderError(message, provider, http.StatusNotFound, errorCode, false, nil, raw, cause),
	}
}

func (e *NotFoundError) IsRetryable() bool { return false }

func (e *NotFoundError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// ContentFilterError indicates the request or response was blocked by content moderation.
type ContentFilterError struct{ ProviderError }

func NewContentFilterError(message, provider, errorCode string, raw map[string]any, cause error) *ContentFilterError {
	return &ContentFilterError{
		ProviderError: *NewProviderError(message, provider, 0, errorCode, false, nil, raw, cause),
	}
}

func (e *ContentFilterError) IsRetryable() bool { return false }

func (e *ContentFilterError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// ContextLengthError indicates the request exceeded the model's maximum context window.
type ContextLengthError struct{ ProviderError }

func NewContextLengthError(message, provider string, statusCode int, errorCode string, raw map[string]any, cause error) *ContextLengthError {
	if statusCode == 0 {
		statusCode = http.StatusRequestEntityTooLarge
	}
	return &ContextLengthError{
		ProviderError: *NewProviderError(message, provider, statusCode, errorCode, false, nil, raw, cause),
	}
}

func (e *ContextLengthError) IsRetryable() bool { return false }

func (e *ContextLengthError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// QuotaExceededError indicates the caller has exhausted their billing quota.
type QuotaExceededError struct{ ProviderError }

func NewQuotaExceededError(message, provider, errorCode string, raw map[string]any, cause error) *QuotaExceededError {
	return &QuotaExceededError{
		ProviderError: *NewProviderError(message, provider, http.StatusTooManyRequests, errorCode, false, nil, raw, cause),
	}
}

func (e *QuotaExceededError) IsRetryable() bool { return false }

func (e *QuotaExceededError) Unwrap() []error {
	errs := []error{&e.ProviderError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// InvalidToolCallError indicates the model produced a tool call that does not
// match any declared tool or has invalid arguments.
type InvalidToolCallError struct {
	SDKError
	ToolName string
}

func NewInvalidToolCallError(message, toolName string, cause error) *InvalidToolCallError {
	return &InvalidToolCallError{
		SDKError: SDKError{Message: message, Cause: cause},
		ToolName: toolName,
	}
}

func (e *InvalidToolCallError) IsRetryable() bool { return false }

func (e *InvalidToolCallError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// UnsupportedToolChoiceError indicates the provider does not support the
// requested tool_choice mode.
type UnsupportedToolChoiceError struct {
	SDKError
	ToolChoice string
}

func NewUnsupportedToolChoiceError(message, toolChoice string, cause error) *UnsupportedToolChoiceError {
	return &UnsupportedToolChoiceError{
		SDKError:   SDKError{Message: message, Cause: cause},
		ToolChoice: toolChoice,
	}
}

func (e *UnsupportedToolChoiceError) IsRetryable() bool { return false }

func (e *UnsupportedToolChoiceError) Unwrap() []error {
	errs := []error{&e.SDKError}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// ---------------------------------------------------------------------------
// ErrorFromStatusCode
// ---------------------------------------------------------------------------

// ErrorFromStatusCode maps an HTTP status code to the most appropriate typed
// error. When the status code alone is ambiguous, message-based classification
// is attempted. If no specific type matches, a generic ProviderError is returned.
func ErrorFromStatusCode(statusCode int, message, provider, errorCode string, raw map[string]any, retryAfter *float64) error {
	// Attempt message-based refinement first.
	if classified := classifyByMessageAndCode(statusCode, message, provider, errorCode, raw, retryAfter); classified != nil {
		return classified
	}

	switch statusCode {
	case http.StatusUnauthorized:
		return NewAuthenticationError(message, provider, errorCode, raw, nil)
	case http.StatusForbidden:
		return NewAccessDeniedError(message, provider, errorCode, raw, nil)
	case http.StatusNotFound:
		return NewNotFoundError(message, provider, errorCode, raw, nil)
	case http.StatusBadRequest:
		return NewInvalidRequestError(message, provider, statusCode, errorCode, raw, nil)
	case http.StatusUnprocessableEntity:
		return NewInvalidRequestError(message, provider, statusCode, errorCode, raw, nil)
	case http.StatusRequestEntityTooLarge:
		return NewContextLengthError(message, provider, statusCode, errorCode, raw, nil)
	case http.StatusTooManyRequests:
		if matchesQuotaExceeded(message) {
			return NewQuotaExceededError(message, provider, errorCode, raw, nil)
		}
		return NewRateLimitError(message, provider, errorCode, raw, retryAfter, nil)
	default:
		if statusCode >= 500 && statusCode < 600 {
			return NewServerError(message, provider, statusCode, errorCode, raw, nil)
		}
		return NewProviderError(message, provider, statusCode, errorCode, false, retryAfter, raw, nil)
	}
}

// classifyByMessageAndCode handles cases where the HTTP status code is
// insufficient and the error message or provider error code is needed.
func classifyByMessageAndCode(statusCode int, message, provider, errorCode string, raw map[string]any, retryAfter *float64) error {
	lower := strings.ToLower(message)
	lowerCode := strings.ToLower(errorCode)

	if matchesContextLength(lower) || matchesContextLength(lowerCode) {
		return NewContextLengthError(message, provider, statusCode, errorCode, raw, nil)
	}
	if matchesContentFilter(lower) || matchesContentFilter(lowerCode) {
		return NewContentFilterError(message, provider, errorCode, raw, nil)
	}
	if matchesQuotaExceeded(lower) || matchesQuotaExceeded(lowerCode) {
		return NewQuotaExceededError(message, provider, errorCode, raw, nil)
	}

	return nil
}

// ---------------------------------------------------------------------------
// ClassifyByMessage
// ---------------------------------------------------------------------------

// ClassifyByMessage attempts to determine the error type from the error message
// text alone.
func ClassifyByMessage(message string) error {
	lower := strings.ToLower(message)

	switch {
	case matchesNotFound(lower):
		return NewNotFoundError(message, "unknown", "", nil, nil)
	case matchesContextLength(lower):
		return NewContextLengthError(message, "unknown", 0, "", nil, nil)
	case matchesContentFilter(lower):
		return NewContentFilterError(message, "unknown", "", nil, nil)
	case matchesAuthentication(lower):
		return NewAuthenticationError(message, "unknown", "", nil, nil)
	case matchesAccessDenied(lower):
		return NewAccessDeniedError(message, "unknown", "", nil, nil)
	case matchesRateLimit(lower):
		return NewRateLimitError(message, "unknown", "", nil, nil, nil)
	case matchesQuotaExceeded(lower):
		return NewQuotaExceededError(message, "unknown", "", nil, nil)
	case matchesServerError(lower):
		return NewServerError(message, "unknown", 0, "", nil, nil)
	case matchesTimeout(lower):
		return NewRequestTimeoutError(message, nil)
	default:
		return NewSDKError(message, nil)
	}
}

// ---------------------------------------------------------------------------
// Message-matching helpers (private)
// ---------------------------------------------------------------------------

func matchesNotFound(lower string) bool {
	return containsAny(lower, "not found", "model not found", "resource not found", "does not exist", "no such model", "unknown model")
}

func matchesContextLength(lower string) bool {
	return containsAny(lower, "context length", "context window", "context_length", "token limit", "tokens exceed", "maximum context", "max tokens", "too many tokens", "input too long", "request too large", "payload too large")
}

func matchesContentFilter(lower string) bool {
	return containsAny(lower, "content filter", "content_filter", "content moderation", "content policy", "content_policy", "safety filter", "safety system", "blocked by", "flagged by", "harmful content")
}

func matchesAuthentication(lower string) bool {
	return containsAny(lower, "invalid api key", "invalid_api_key", "api key", "authentication", "unauthorized", "invalid token", "expired token", "invalid credentials")
}

func matchesAccessDenied(lower string) bool {
	return containsAny(lower, "access denied", "forbidden", "permission denied", "insufficient permissions", "not allowed")
}

func matchesRateLimit(lower string) bool {
	return containsAny(lower, "rate limit", "rate_limit", "too many requests", "throttled", "requests per minute", "requests per second")
}

func matchesQuotaExceeded(lower string) bool {
	return containsAny(lower, "quota exceeded", "quota_exceeded", "billing", "insufficient_quota", "exceeded your current quota", "spending limit", "usage limit")
}

func matchesServerError(lower string) bool {
	return containsAny(lower, "internal server error", "internal error", "server error", "service unavailable", "bad gateway", "gateway timeout", "overloaded")
}

func matchesTimeout(lower string) bool {
	return containsAny(lower, "timeout", "timed out", "deadline exceeded", "request took too long")
}

func containsAny(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Retryable interface
// ---------------------------------------------------------------------------

// Retryable is satisfied by any error that can report whether it is safe to retry.
type Retryable interface {
	error
	IsRetryable() bool
}

// IsRetryable reports whether err (or any error in its Unwrap chain) is retryable.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	type unwrapper interface {
		Unwrap() error
	}
	type multiUnwrapper interface {
		Unwrap() []error
	}

	for err != nil {
		if r, ok := err.(Retryable); ok {
			return r.IsRetryable()
		}
		switch u := err.(type) {
		case multiUnwrapper:
			for _, inner := range u.Unwrap() {
				if IsRetryable(inner) {
					return true
				}
			}
			return false
		case unwrapper:
			err = u.Unwrap()
		default:
			return false
		}
	}
	return false
}
