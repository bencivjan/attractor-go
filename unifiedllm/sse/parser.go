package sse

import (
	"bufio"
	"io"
	"strconv"
	"strings"
)

// Event represents a parsed SSE event.
type Event struct {
	Type  string // event type (from "event:" line)
	Data  string // event data (from "data:" lines, joined with newlines)
	ID    string // event ID (from "id:" line)
	Retry int    // reconnection time in ms (from "retry:" line)
}

// Parser reads SSE events from an io.Reader according to the SSE specification.
type Parser struct {
	scanner *bufio.Scanner
}

// NewParser creates a new SSE parser that reads from r.
func NewParser(r io.Reader) *Parser {
	return &Parser{
		scanner: bufio.NewScanner(r),
	}
}

// Next returns the next SSE event from the stream. It returns io.EOF when
// the stream is exhausted or when a "[DONE]" data sentinel is encountered.
func (p *Parser) Next() (Event, error) {
	var evt Event
	var dataLines []string
	hasFields := false

	for p.scanner.Scan() {
		line := p.scanner.Text()

		if line == "" {
			if !hasFields {
				continue
			}
			evt.Data = strings.Join(dataLines, "\n")
			return evt, nil
		}

		if strings.HasPrefix(line, ":") {
			continue
		}

		field, value := parseLine(line)

		switch field {
		case "data":
			if value == "[DONE]" {
				return Event{}, io.EOF
			}
			dataLines = append(dataLines, value)
			hasFields = true
		case "event":
			evt.Type = value
			hasFields = true
		case "id":
			if !strings.Contains(value, "\x00") {
				evt.ID = value
			}
			hasFields = true
		case "retry":
			ms, err := strconv.Atoi(value)
			if err == nil && ms >= 0 {
				evt.Retry = ms
			}
			hasFields = true
		default:
		}
	}

	if err := p.scanner.Err(); err != nil {
		return Event{}, err
	}

	if hasFields {
		evt.Data = strings.Join(dataLines, "\n")
		return evt, nil
	}

	return Event{}, io.EOF
}

func parseLine(line string) (field, value string) {
	idx := strings.IndexByte(line, ':')
	if idx < 0 {
		return line, ""
	}
	field = line[:idx]
	value = line[idx+1:]
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return field, value
}
