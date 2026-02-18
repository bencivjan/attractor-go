package sse

import (
	"io"
	"strings"
	"testing"
)

func TestBasicDataEvent(t *testing.T) {
	input := "data: hello world\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "hello world" {
		t.Errorf("Data = %q, want %q", evt.Data, "hello world")
	}
	if evt.Type != "" {
		t.Errorf("Type = %q, want empty", evt.Type)
	}
	if evt.ID != "" {
		t.Errorf("ID = %q, want empty", evt.ID)
	}
}

func TestEventTypeField(t *testing.T) {
	input := "event: message\ndata: payload\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Type != "message" {
		t.Errorf("Type = %q, want %q", evt.Type, "message")
	}
	if evt.Data != "payload" {
		t.Errorf("Data = %q, want %q", evt.Data, "payload")
	}
}

func TestMultiLineData(t *testing.T) {
	input := "data: line1\ndata: line2\ndata: line3\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	want := "line1\nline2\nline3"
	if evt.Data != want {
		t.Errorf("Data = %q, want %q", evt.Data, want)
	}
}

func TestIDField(t *testing.T) {
	input := "id: 42\ndata: test\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.ID != "42" {
		t.Errorf("ID = %q, want %q", evt.ID, "42")
	}
}

func TestIDFieldWithNullByteIgnored(t *testing.T) {
	// Per SSE spec, IDs containing null characters should be ignored.
	input := "id: bad\x00id\ndata: test\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.ID != "" {
		t.Errorf("ID = %q, want empty (null byte should cause ignore)", evt.ID)
	}
}

func TestRetryDirective(t *testing.T) {
	input := "retry: 3000\ndata: test\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Retry != 3000 {
		t.Errorf("Retry = %d, want %d", evt.Retry, 3000)
	}
}

func TestRetryInvalidValue(t *testing.T) {
	// Non-integer retry value should be silently ignored (Retry stays 0).
	input := "retry: abc\ndata: test\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Retry != 0 {
		t.Errorf("Retry = %d, want 0 (invalid value ignored)", evt.Retry)
	}
}

func TestCommentLinesIgnored(t *testing.T) {
	input := ": this is a comment\ndata: actual data\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "actual data" {
		t.Errorf("Data = %q, want %q", evt.Data, "actual data")
	}
}

func TestMultipleComments(t *testing.T) {
	input := ": comment 1\n: comment 2\ndata: value\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "value" {
		t.Errorf("Data = %q, want %q", evt.Data, "value")
	}
}

func TestEmptyLinesBeforeEvent(t *testing.T) {
	// Leading blank lines (before any field) should be skipped.
	input := "\n\ndata: hello\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "hello" {
		t.Errorf("Data = %q, want %q", evt.Data, "hello")
	}
}

func TestDoneSentinel(t *testing.T) {
	input := "data: [DONE]\n\n"
	p := NewParser(strings.NewReader(input))

	_, err := p.Next()
	if err != io.EOF {
		t.Errorf("Next() error = %v, want io.EOF", err)
	}
}

func TestDoneSentinelMidStream(t *testing.T) {
	input := "data: first event\n\ndata: [DONE]\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("first Next() error = %v", err)
	}
	if evt.Data != "first event" {
		t.Errorf("first Data = %q, want %q", evt.Data, "first event")
	}

	_, err = p.Next()
	if err != io.EOF {
		t.Errorf("second Next() error = %v, want io.EOF", err)
	}
}

func TestEOFHandling(t *testing.T) {
	// Empty input should return EOF immediately.
	p := NewParser(strings.NewReader(""))

	_, err := p.Next()
	if err != io.EOF {
		t.Errorf("Next() on empty input error = %v, want io.EOF", err)
	}
}

func TestEOFAfterLastEvent(t *testing.T) {
	input := "data: last\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("first Next() error = %v", err)
	}
	if evt.Data != "last" {
		t.Errorf("Data = %q, want %q", evt.Data, "last")
	}

	_, err = p.Next()
	if err != io.EOF {
		t.Errorf("second Next() error = %v, want io.EOF", err)
	}
}

func TestEventWithoutTrailingNewline(t *testing.T) {
	// If the stream ends without a trailing blank line, the accumulated
	// event should still be returned.
	input := "data: no trailing blank"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "no trailing blank" {
		t.Errorf("Data = %q, want %q", evt.Data, "no trailing blank")
	}
}

func TestMultipleEvents(t *testing.T) {
	input := "data: event1\n\ndata: event2\n\ndata: event3\n\n"
	p := NewParser(strings.NewReader(input))

	for i, want := range []string{"event1", "event2", "event3"} {
		evt, err := p.Next()
		if err != nil {
			t.Fatalf("event %d: Next() error = %v", i+1, err)
		}
		if evt.Data != want {
			t.Errorf("event %d: Data = %q, want %q", i+1, evt.Data, want)
		}
	}

	_, err := p.Next()
	if err != io.EOF {
		t.Errorf("after all events: Next() error = %v, want io.EOF", err)
	}
}

func TestFullEventAllFields(t *testing.T) {
	input := "event: update\nid: 7\nretry: 5000\ndata: full event\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Type != "update" {
		t.Errorf("Type = %q, want %q", evt.Type, "update")
	}
	if evt.ID != "7" {
		t.Errorf("ID = %q, want %q", evt.ID, "7")
	}
	if evt.Retry != 5000 {
		t.Errorf("Retry = %d, want %d", evt.Retry, 5000)
	}
	if evt.Data != "full event" {
		t.Errorf("Data = %q, want %q", evt.Data, "full event")
	}
}

func TestDataWithColonInValue(t *testing.T) {
	// The value after "data: " may itself contain colons.
	input := "data: key: value\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "key: value" {
		t.Errorf("Data = %q, want %q", evt.Data, "key: value")
	}
}

func TestDataWithNoSpaceAfterColon(t *testing.T) {
	// SSE spec: if there is no space after the colon, the value starts right away.
	input := "data:no-space\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "no-space" {
		t.Errorf("Data = %q, want %q", evt.Data, "no-space")
	}
}

func TestFieldWithNoColon(t *testing.T) {
	// A line with no colon is treated as a field name with empty value.
	// Unknown fields are ignored, so this should not disrupt parsing.
	input := "unknownfield\ndata: hello\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "hello" {
		t.Errorf("Data = %q, want %q", evt.Data, "hello")
	}
}

func TestEmptyDataLine(t *testing.T) {
	// "data:" with no value should produce an empty string data line.
	input := "data:\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if evt.Data != "" {
		t.Errorf("Data = %q, want empty string", evt.Data)
	}
}

func TestMultipleDataLinesWithEmpty(t *testing.T) {
	// Multiple data lines including empty ones are joined with newlines.
	input := "data: first\ndata:\ndata: third\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	want := "first\n\nthird"
	if evt.Data != want {
		t.Errorf("Data = %q, want %q", evt.Data, want)
	}
}

func TestOnlyComments(t *testing.T) {
	// Stream containing only comments should return EOF.
	input := ": comment only\n: another comment\n"
	p := NewParser(strings.NewReader(input))

	_, err := p.Next()
	if err != io.EOF {
		t.Errorf("Next() on comments-only input error = %v, want io.EOF", err)
	}
}

func TestJSONData(t *testing.T) {
	// Typical usage with JSON data payloads.
	input := "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
	p := NewParser(strings.NewReader(input))

	evt, err := p.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	want := `{"choices":[{"delta":{"content":"Hello"}}]}`
	if evt.Data != want {
		t.Errorf("Data = %q, want %q", evt.Data, want)
	}
}
