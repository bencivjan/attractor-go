package retry

import (
	"context"
	"math"
	"math/rand/v2"
	"time"
)

// Policy configures retry behavior for transient failures.
type Policy struct {
	// MaxRetries is the total number of retry attempts, not counting the
	// initial call. Default: 2.
	MaxRetries int

	// BaseDelay is the initial delay before the first retry. Default: 1s.
	BaseDelay time.Duration

	// MaxDelay is the upper bound on any single retry delay. Default: 60s.
	MaxDelay time.Duration

	// BackoffMultiplier is the exponential factor applied to BaseDelay on
	// each successive retry. Default: 2.0.
	BackoffMultiplier float64

	// Jitter adds randomization to the delay to avoid thundering-herd
	// effects. When enabled, the computed delay is multiplied by a random
	// factor in [0.5, 1.5). Default: true.
	Jitter bool

	// OnRetry is an optional callback invoked before each retry attempt.
	OnRetry func(err error, attempt int, delay time.Duration)
}

// DefaultPolicy returns a Policy with sensible production defaults.
func DefaultPolicy() Policy {
	return Policy{
		MaxRetries:        2,
		BaseDelay:         1 * time.Second,
		MaxDelay:          60 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            true,
	}
}

// DelayForAttempt calculates the delay for attempt n (0-indexed).
func (p Policy) DelayForAttempt(attempt int) time.Duration {
	delay := float64(p.BaseDelay) * math.Pow(p.BackoffMultiplier, float64(attempt))

	if delay > float64(p.MaxDelay) {
		delay = float64(p.MaxDelay)
	}

	if p.Jitter {
		jitterFactor := 0.5 + rand.Float64()
		delay *= jitterFactor
	}

	return time.Duration(delay)
}

// retryable is an interface that errors can implement to indicate whether
// they should be retried.
type retryable interface {
	IsRetryable() bool
}

// IsRetryable checks whether err should be retried.
func IsRetryable(err error) bool {
	if r, ok := err.(retryable); ok {
		return r.IsRetryable()
	}
	return true
}

// Do executes fn with the given retry policy.
func Do[T any](ctx context.Context, policy Policy, fn func() (T, error)) (T, error) {
	var zero T

	result, err := fn()
	if err == nil {
		return result, nil
	}

	for attempt := 0; attempt < policy.MaxRetries; attempt++ {
		if !IsRetryable(err) {
			return zero, err
		}

		delay := policy.DelayForAttempt(attempt)

		if policy.OnRetry != nil {
			policy.OnRetry(err, attempt, delay)
		}

		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-time.After(delay):
		}

		result, err = fn()
		if err == nil {
			return result, nil
		}
	}

	return zero, err
}

// DoVoid executes fn with the given retry policy for functions that do not return a value.
func DoVoid(ctx context.Context, policy Policy, fn func() error) error {
	_, err := Do(ctx, policy, func() (struct{}, error) {
		return struct{}{}, fn()
	})
	return err
}
