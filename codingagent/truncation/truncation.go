// Package truncation implements tool output truncation strategies to keep
// conversation context within manageable limits. It supports both character-based
// and line-based truncation with configurable modes (head+tail or tail-only)
// on a per-tool basis.
package truncation

import (
	"fmt"
	"strings"
)

// Mode controls which portion of the output is preserved when truncation
// is required.
type Mode string

const (
	// ModeHeadTail preserves both the beginning and end of the output,
	// removing the middle. This is useful for tools like shell output where
	// both the initial lines and final lines are informative.
	ModeHeadTail Mode = "head_tail"
	// ModeTail preserves only the end of the output. This is useful for
	// tools like grep where the most recent or final matches are typically
	// the most relevant.
	ModeTail Mode = "tail"
)

// DefaultCharLimits defines the maximum character count per tool name.
var DefaultCharLimits = map[string]int{
	"read_file":   50000,
	"shell":       30000,
	"grep":        20000,
	"glob":        20000,
	"edit_file":   10000,
	"apply_patch": 10000,
	"write_file":  1000,
	"spawn_agent": 20000,
}

// DefaultLineLimits defines the maximum line count per tool name. Applied
// after character truncation.
var DefaultLineLimits = map[string]int{
	"shell": 256,
	"grep":  200,
	"glob":  500,
}

// DefaultModes defines the truncation mode per tool name.
var DefaultModes = map[string]Mode{
	"read_file":   ModeHeadTail,
	"shell":       ModeHeadTail,
	"grep":        ModeTail,
	"glob":        ModeTail,
	"edit_file":   ModeTail,
	"apply_patch": ModeTail,
	"write_file":  ModeTail,
	"spawn_agent": ModeHeadTail,
}

// defaultCharLimit is applied when a tool has no entry in DefaultCharLimits.
const defaultCharLimit = 30000

// Truncate applies character-based truncation to output. If the output is
// within maxChars no changes are made. When truncation is needed, the mode
// determines which portion is kept.
func Truncate(output string, maxChars int, mode Mode) string {
	if maxChars <= 0 || len(output) <= maxChars {
		return output
	}

	removed := len(output) - maxChars

	switch mode {
	case ModeHeadTail:
		// Split budget 50/50 between head and tail (spec 5.1).
		headBudget := maxChars / 2
		tailBudget := maxChars - headBudget

		head := output[:headBudget]
		tail := output[len(output)-tailBudget:]

		marker := fmt.Sprintf("\n\n[WARNING: Tool output was truncated. %d characters were removed from the middle. "+
			"The full output is available in the event stream. "+
			"If you need to see specific parts, re-run the tool with more targeted parameters.]\n\n", removed)
		return head + marker + tail

	case ModeTail:
		tail := output[len(output)-maxChars:]
		marker := fmt.Sprintf("[WARNING: Tool output was truncated. First %d characters were removed. "+
			"The full output is available in the event stream.]\n\n", removed)
		return marker + tail

	default:
		// Unknown mode falls back to tail truncation.
		tail := output[len(output)-maxChars:]
		marker := fmt.Sprintf("[WARNING: Tool output was truncated. First %d characters were removed. "+
			"The full output is available in the event stream.]\n\n", removed)
		return marker + tail
	}
}

// TruncateLines applies line-based truncation. If the output has more than
// maxLines lines, excess lines are removed from the middle. The first half
// and last half of the allowed lines are preserved.
func TruncateLines(output string, maxLines int) string {
	if maxLines <= 0 {
		return output
	}

	lines := strings.Split(output, "\n")
	if len(lines) <= maxLines {
		return output
	}

	removed := len(lines) - maxLines

	// Split: keep first half and last half of the budget.
	headLines := maxLines / 2
	tailLines := maxLines - headLines

	head := lines[:headLines]
	tail := lines[len(lines)-tailLines:]

	marker := fmt.Sprintf("\n--- truncated %d lines ---\n", removed)
	return strings.Join(head, "\n") + marker + strings.Join(tail, "\n")
}

// TruncateToolOutput applies the full truncation pipeline for a named tool:
// 1. Character-based truncation using the tool's configured limit and mode.
// 2. Line-based truncation using the tool's configured line limit.
//
// customCharLimits overrides DefaultCharLimits; customLineLimits overrides
// DefaultLineLimits. Either may be nil to use defaults.
func TruncateToolOutput(output, toolName string, customCharLimits, customLineLimits map[string]int) string {
	if output == "" {
		return output
	}

	// Determine character limit.
	charLimit := defaultCharLimit
	if customCharLimits != nil {
		if limit, ok := customCharLimits[toolName]; ok {
			charLimit = limit
		}
	} else if limit, ok := DefaultCharLimits[toolName]; ok {
		charLimit = limit
	}

	// Determine mode.
	mode := ModeTail
	if m, ok := DefaultModes[toolName]; ok {
		mode = m
	}

	// Step 1: character truncation.
	result := Truncate(output, charLimit, mode)

	// Step 2: line truncation.
	lineLimit := 0
	if customLineLimits != nil {
		if limit, ok := customLineLimits[toolName]; ok {
			lineLimit = limit
		}
	} else if limit, ok := DefaultLineLimits[toolName]; ok {
		lineLimit = limit
	}
	if lineLimit > 0 {
		result = TruncateLines(result, lineLimit)
	}

	return result
}
