// Package events provides a lightweight, synchronous event system for the
// coding agent session. Events are emitted during the agentic loop to notify
// listeners of session lifecycle changes, tool executions, streaming text,
// and diagnostic conditions such as loop detection or turn limits.
package events

import (
	"sync"
	"time"
)

// EventKind discriminates the type of session event.
type EventKind string

const (
	// EventSessionStart is emitted when a session begins processing.
	EventSessionStart EventKind = "session_start"
	// EventSessionEnd is emitted when a session finishes or is closed.
	EventSessionEnd EventKind = "session_end"
	// EventUserInput is emitted when user input is received.
	EventUserInput EventKind = "user_input"
	// EventAssistantTextStart marks the beginning of assistant text output.
	EventAssistantTextStart EventKind = "assistant_text_start"
	// EventAssistantTextDelta carries an incremental chunk of assistant text.
	EventAssistantTextDelta EventKind = "assistant_text_delta"
	// EventAssistantTextEnd marks the end of assistant text output.
	EventAssistantTextEnd EventKind = "assistant_text_end"
	// EventToolCallStart is emitted when a tool call begins execution.
	EventToolCallStart EventKind = "tool_call_start"
	// EventToolCallOutputDelta carries incremental tool output.
	EventToolCallOutputDelta EventKind = "tool_call_output_delta"
	// EventToolCallEnd is emitted when a tool call finishes.
	EventToolCallEnd EventKind = "tool_call_end"
	// EventSteeringInjected is emitted when a steering message is injected.
	EventSteeringInjected EventKind = "steering_injected"
	// EventTurnLimit is emitted when a turn or tool-round limit is reached.
	EventTurnLimit EventKind = "turn_limit"
	// EventLoopDetection is emitted when the loop detector fires.
	EventLoopDetection EventKind = "loop_detection"
	// EventError is emitted on non-fatal errors during the loop.
	EventError EventKind = "error"
	// EventWarning is emitted for diagnostic warnings.
	EventWarning EventKind = "warning"
)

// SessionEvent is a single event emitted during a session's lifecycle.
type SessionEvent struct {
	Kind      EventKind      `json:"kind"`
	Timestamp time.Time      `json:"timestamp"`
	SessionID string         `json:"session_id"`
	Data      map[string]any `json:"data,omitempty"`
}

// Emitter is a synchronous event dispatcher. Listeners are invoked in
// registration order on the goroutine that calls Emit. Emitter is safe
// for concurrent use.
type Emitter struct {
	mu        sync.RWMutex
	listeners []func(SessionEvent)
}

// NewEmitter creates an Emitter with no listeners.
func NewEmitter() *Emitter {
	return &Emitter{}
}

// On registers a listener that will be called for every subsequent event.
// Listeners are invoked synchronously in registration order.
func (e *Emitter) On(listener func(SessionEvent)) {
	if listener == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.listeners = append(e.listeners, listener)
}

// Emit dispatches an event to all registered listeners. It constructs the
// SessionEvent with the current timestamp and the supplied parameters.
func (e *Emitter) Emit(kind EventKind, sessionID string, data map[string]any) {
	evt := SessionEvent{
		Kind:      kind,
		Timestamp: time.Now(),
		SessionID: sessionID,
		Data:      data,
	}

	// Take a snapshot of listeners under read lock to allow concurrent Emit
	// calls and to prevent holding the lock during listener execution.
	e.mu.RLock()
	snapshot := make([]func(SessionEvent), len(e.listeners))
	copy(snapshot, e.listeners)
	e.mu.RUnlock()

	for _, fn := range snapshot {
		fn(evt)
	}
}

// ListenerCount returns the number of registered listeners.
func (e *Emitter) ListenerCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.listeners)
}
