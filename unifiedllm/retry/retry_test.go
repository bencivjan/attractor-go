package retry

import (
	"context"
	"errors"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// DefaultPolicy values
// ---------------------------------------------------------------------------

func TestDefaultPolicy(t *testing.T) {
	p := DefaultPolicy()

	if p.MaxRetries != 2 {
		t.Errorf("MaxRetries = %d, want 2", p.MaxRetries)
	}
	if p.BaseDelay != 1*time.Second {
		t.Errorf("BaseDelay = %v, want 1s", p.BaseDelay)
	}
	if p.MaxDelay != 60*time.Second {
		t.Errorf("MaxDelay = %v, want 60s", p.MaxDelay)
	}
	if p.BackoffMultiplier != 2.0 {
		t.Errorf("BackoffMultiplier = %f, want 2.0", p.BackoffMultiplier)
	}
	if !p.Jitter {
		t.Error("Jitter should be true by default")
	}
	if p.OnRetry != nil {
		t.Error("OnRetry should be nil by default")
	}
}

// ---------------------------------------------------------------------------
// DelayForAttempt: exponential backoff without jitter
// ---------------------------------------------------------------------------

func TestDelayForAttemptExponentialBackoff(t *testing.T) {
	p := Policy{
		BaseDelay:         100 * time.Millisecond,
		MaxDelay:          10 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{0, 100 * time.Millisecond},  // 100ms * 2^0 = 100ms
		{1, 200 * time.Millisecond},  // 100ms * 2^1 = 200ms
		{2, 400 * time.Millisecond},  // 100ms * 2^2 = 400ms
		{3, 800 * time.Millisecond},  // 100ms * 2^3 = 800ms
		{4, 1600 * time.Millisecond}, // 100ms * 2^4 = 1600ms
	}

	for _, tt := range tests {
		got := p.DelayForAttempt(tt.attempt)
		if got != tt.want {
			t.Errorf("DelayForAttempt(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// DelayForAttempt: max delay capping
// ---------------------------------------------------------------------------

func TestDelayForAttemptMaxDelayCapping(t *testing.T) {
	p := Policy{
		BaseDelay:         1 * time.Second,
		MaxDelay:          5 * time.Second,
		BackoffMultiplier: 10.0,
		Jitter:            false,
	}

	// Attempt 0: 1s * 10^0 = 1s (under max)
	got0 := p.DelayForAttempt(0)
	if got0 != 1*time.Second {
		t.Errorf("DelayForAttempt(0) = %v, want 1s", got0)
	}

	// Attempt 1: 1s * 10^1 = 10s, capped to 5s
	got1 := p.DelayForAttempt(1)
	if got1 != 5*time.Second {
		t.Errorf("DelayForAttempt(1) = %v, want 5s (capped)", got1)
	}

	// Attempt 10: way over max, should be capped to 5s
	got10 := p.DelayForAttempt(10)
	if got10 != 5*time.Second {
		t.Errorf("DelayForAttempt(10) = %v, want 5s (capped)", got10)
	}
}

// ---------------------------------------------------------------------------
// DelayForAttempt: jitter produces delay in expected range
// ---------------------------------------------------------------------------

func TestDelayForAttemptWithJitter(t *testing.T) {
	p := Policy{
		BaseDelay:         1 * time.Second,
		MaxDelay:          60 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            true,
	}

	// With jitter, delay = base * 2^attempt * [0.5, 1.5)
	// For attempt 0: base delay = 1s, so jittered range is [0.5s, 1.5s)
	minExpected := 500 * time.Millisecond
	maxExpected := 1500 * time.Millisecond

	// Run multiple times to ensure it falls in range.
	for i := 0; i < 100; i++ {
		got := p.DelayForAttempt(0)
		if got < minExpected || got >= maxExpected {
			t.Errorf("iteration %d: DelayForAttempt(0) = %v, want in [%v, %v)", i, got, minExpected, maxExpected)
		}
	}
}

// ---------------------------------------------------------------------------
// Do[T]: success on first try
// ---------------------------------------------------------------------------

func TestDoSuccessFirstTry(t *testing.T) {
	p := Policy{
		MaxRetries:        3,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	result, err := Do(context.Background(), p, func() (string, error) {
		calls++
		return "success", nil
	})

	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if result != "success" {
		t.Errorf("Do() result = %q, want %q", result, "success")
	}
	if calls != 1 {
		t.Errorf("function called %d times, want 1", calls)
	}
}

// ---------------------------------------------------------------------------
// Do[T]: success after retries
// ---------------------------------------------------------------------------

func TestDoSuccessAfterRetries(t *testing.T) {
	p := Policy{
		MaxRetries:        3,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	result, err := Do(context.Background(), p, func() (int, error) {
		calls++
		if calls < 3 {
			return 0, &retryableError{msg: "transient"}
		}
		return 42, nil
	})

	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if result != 42 {
		t.Errorf("Do() result = %d, want 42", result)
	}
	if calls != 3 {
		t.Errorf("function called %d times, want 3 (1 initial + 2 retries)", calls)
	}
}

// ---------------------------------------------------------------------------
// Do[T]: max retries exceeded
// ---------------------------------------------------------------------------

func TestDoMaxRetriesExceeded(t *testing.T) {
	p := Policy{
		MaxRetries:        2,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	_, err := Do(context.Background(), p, func() (string, error) {
		calls++
		return "", &retryableError{msg: "always fails"}
	})

	if err == nil {
		t.Fatal("Do() expected error after max retries, got nil")
	}
	// 1 initial call + 2 retries = 3 total calls
	if calls != 3 {
		t.Errorf("function called %d times, want 3 (1 initial + 2 retries)", calls)
	}
}

// ---------------------------------------------------------------------------
// Do[T]: non-retryable error stops immediately
// ---------------------------------------------------------------------------

func TestDoNonRetryableStopsImmediately(t *testing.T) {
	p := Policy{
		MaxRetries:        5,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	_, err := Do(context.Background(), p, func() (string, error) {
		calls++
		return "", &nonRetryableError{msg: "permanent failure"}
	})

	if err == nil {
		t.Fatal("Do() expected error, got nil")
	}
	// Should have called only once (initial) since the error is not retryable.
	if calls != 1 {
		t.Errorf("function called %d times, want 1 (non-retryable stops retries)", calls)
	}
}

// ---------------------------------------------------------------------------
// DoVoid: success
// ---------------------------------------------------------------------------

func TestDoVoidSuccess(t *testing.T) {
	p := Policy{
		MaxRetries:        2,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	err := DoVoid(context.Background(), p, func() error {
		calls++
		return nil
	})

	if err != nil {
		t.Fatalf("DoVoid() error = %v", err)
	}
	if calls != 1 {
		t.Errorf("function called %d times, want 1", calls)
	}
}

// ---------------------------------------------------------------------------
// DoVoid: failure after retries
// ---------------------------------------------------------------------------

func TestDoVoidFailure(t *testing.T) {
	p := Policy{
		MaxRetries:        2,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	err := DoVoid(context.Background(), p, func() error {
		calls++
		return &retryableError{msg: "always fails"}
	})

	if err == nil {
		t.Fatal("DoVoid() expected error, got nil")
	}
	if calls != 3 {
		t.Errorf("function called %d times, want 3", calls)
	}
}

// ---------------------------------------------------------------------------
// DoVoid: success after retries
// ---------------------------------------------------------------------------

func TestDoVoidSuccessAfterRetry(t *testing.T) {
	p := Policy{
		MaxRetries:        3,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	calls := 0
	err := DoVoid(context.Background(), p, func() error {
		calls++
		if calls < 2 {
			return &retryableError{msg: "transient"}
		}
		return nil
	})

	if err != nil {
		t.Fatalf("DoVoid() error = %v", err)
	}
	if calls != 2 {
		t.Errorf("function called %d times, want 2", calls)
	}
}

// ---------------------------------------------------------------------------
// Context cancellation during retry
// ---------------------------------------------------------------------------

func TestDoContextCancellation(t *testing.T) {
	p := Policy{
		MaxRetries:        10,
		BaseDelay:         100 * time.Millisecond,
		MaxDelay:          1 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	ctx, cancel := context.WithCancel(context.Background())
	calls := 0

	go func() {
		// Cancel after a short delay, before retries complete.
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := Do(ctx, p, func() (string, error) {
		calls++
		return "", &retryableError{msg: "keep trying"}
	})

	if !errors.Is(err, context.Canceled) {
		t.Errorf("Do() error = %v, want context.Canceled", err)
	}
}

func TestDoContextDeadlineExceeded(t *testing.T) {
	p := Policy{
		MaxRetries:        10,
		BaseDelay:         100 * time.Millisecond,
		MaxDelay:          1 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := Do(ctx, p, func() (string, error) {
		return "", &retryableError{msg: "keep trying"}
	})

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Do() error = %v, want context.DeadlineExceeded", err)
	}
}

func TestDoVoidContextCancellation(t *testing.T) {
	p := Policy{
		MaxRetries:        10,
		BaseDelay:         100 * time.Millisecond,
		MaxDelay:          1 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            false,
	}

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	err := DoVoid(ctx, p, func() error {
		return &retryableError{msg: "keep trying"}
	})

	if !errors.Is(err, context.Canceled) {
		t.Errorf("DoVoid() error = %v, want context.Canceled", err)
	}
}

// ---------------------------------------------------------------------------
// OnRetry callback
// ---------------------------------------------------------------------------

func TestOnRetryCallback(t *testing.T) {
	var attempts []int
	p := Policy{
		MaxRetries:        3,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		Jitter:            false,
		OnRetry: func(err error, attempt int, delay time.Duration) {
			attempts = append(attempts, attempt)
		},
	}

	calls := 0
	_, _ = Do(context.Background(), p, func() (string, error) {
		calls++
		if calls <= 3 {
			return "", &retryableError{msg: "fail"}
		}
		return "ok", nil
	})

	// 1 initial + 3 retries = 4 calls, OnRetry called for attempts 0, 1, 2
	if len(attempts) != 3 {
		t.Fatalf("OnRetry called %d times, want 3", len(attempts))
	}
	for i, got := range attempts {
		if got != i {
			t.Errorf("OnRetry attempt[%d] = %d, want %d", i, got, i)
		}
	}
}

// ---------------------------------------------------------------------------
// IsRetryable (package-level)
// ---------------------------------------------------------------------------

func TestIsRetryablePackageLevel(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "retryable error",
			err:  &retryableError{msg: "retry me"},
			want: true,
		},
		{
			name: "non-retryable error",
			err:  &nonRetryableError{msg: "do not retry"},
			want: false,
		},
		{
			name: "plain error (no IsRetryable) defaults to true",
			err:  errors.New("generic"),
			want: true,
		},
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
// Test helpers
// ---------------------------------------------------------------------------

type retryableError struct {
	msg string
}

func (e *retryableError) Error() string      { return e.msg }
func (e *retryableError) IsRetryable() bool   { return true }

type nonRetryableError struct {
	msg string
}

func (e *nonRetryableError) Error() string    { return e.msg }
func (e *nonRetryableError) IsRetryable() bool { return false }
