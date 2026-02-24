// Package tools contains the apply_patch implementation for the v4a patch format
// used by the OpenAI profile.
package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// PatchOperation represents a single file operation in a v4a patch.
type PatchOperation struct {
	Type    string // "add", "delete", "update"
	Path    string
	MoveTo  string // for rename operations
	Content string // for add: the full file content
	Hunks   []Hunk // for update: the diff hunks
}

// Hunk represents a change hunk within an update operation.
type Hunk struct {
	ContextHint string // text after "@@" on the hunk header line
	Lines       []HunkLine
}

// HunkLine represents a single line within a hunk.
type HunkLine struct {
	Type    byte   // ' ' = context, '-' = delete, '+' = add
	Content string // the line content without the prefix
}

// ParsePatch parses a v4a format patch string into a list of operations.
func ParsePatch(patch string) ([]PatchOperation, error) {
	lines := strings.Split(patch, "\n")
	var ops []PatchOperation

	i := 0

	// Find "*** Begin Patch"
	for i < len(lines) {
		if strings.TrimSpace(lines[i]) == "*** Begin Patch" {
			i++
			break
		}
		i++
	}

	for i < len(lines) {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		if trimmed == "*** End Patch" || trimmed == "" && i == len(lines)-1 {
			break
		}

		if strings.HasPrefix(line, "*** Add File: ") {
			path := strings.TrimPrefix(line, "*** Add File: ")
			path = strings.TrimSpace(path)
			i++

			// Collect added lines (all prefixed with +).
			var content []string
			for i < len(lines) {
				if strings.HasPrefix(lines[i], "*** ") {
					break
				}
				if strings.HasPrefix(lines[i], "+") {
					content = append(content, lines[i][1:])
				}
				i++
			}

			ops = append(ops, PatchOperation{
				Type:    "add",
				Path:    path,
				Content: strings.Join(content, "\n"),
			})
			continue
		}

		if strings.HasPrefix(line, "*** Delete File: ") {
			path := strings.TrimPrefix(line, "*** Delete File: ")
			path = strings.TrimSpace(path)
			i++

			ops = append(ops, PatchOperation{
				Type: "delete",
				Path: path,
			})
			continue
		}

		if strings.HasPrefix(line, "*** Update File: ") {
			path := strings.TrimPrefix(line, "*** Update File: ")
			path = strings.TrimSpace(path)
			i++

			op := PatchOperation{
				Type: "update",
				Path: path,
			}

			// Check for Move to.
			if i < len(lines) && strings.HasPrefix(lines[i], "*** Move to: ") {
				op.MoveTo = strings.TrimSpace(strings.TrimPrefix(lines[i], "*** Move to: "))
				i++
			}

			// Parse hunks.
			for i < len(lines) {
				if strings.HasPrefix(lines[i], "*** ") && !strings.HasPrefix(lines[i], "*** End of File") {
					break
				}

				if strings.HasPrefix(lines[i], "@@ ") {
					hint := strings.TrimPrefix(lines[i], "@@ ")
					hint = strings.TrimSpace(hint)
					i++

					var hunkLines []HunkLine
					for i < len(lines) {
						l := lines[i]
						if strings.HasPrefix(l, "@@ ") || (strings.HasPrefix(l, "*** ") && !strings.HasPrefix(l, "*** End of File")) {
							break
						}
						if strings.HasPrefix(l, "*** End of File") {
							i++
							break
						}
						if len(l) == 0 {
							// Empty line treated as context with empty content.
							hunkLines = append(hunkLines, HunkLine{Type: ' ', Content: ""})
							i++
							continue
						}
						prefix := l[0]
						content := ""
						if len(l) > 1 {
							content = l[1:]
						}
						switch prefix {
						case ' ', '-', '+':
							hunkLines = append(hunkLines, HunkLine{Type: prefix, Content: content})
						default:
							// Treat as context line (could be whitespace issues).
							hunkLines = append(hunkLines, HunkLine{Type: ' ', Content: l})
						}
						i++
					}

					op.Hunks = append(op.Hunks, Hunk{
						ContextHint: hint,
						Lines:       hunkLines,
					})
				} else {
					i++
				}
			}

			ops = append(ops, op)
			continue
		}

		// Skip unrecognized lines.
		i++
	}

	return ops, nil
}

// ApplyPatch applies a v4a format patch to the filesystem rooted at workDir.
// Returns a summary of operations performed.
func ApplyPatch(patch string, workDir string) (string, error) {
	ops, err := ParsePatch(patch)
	if err != nil {
		return "", fmt.Errorf("parse patch: %w", err)
	}

	if len(ops) == 0 {
		return "", fmt.Errorf("no operations found in patch")
	}

	var results []string

	for _, op := range ops {
		path := filepath.Join(workDir, op.Path)

		switch op.Type {
		case "add":
			dir := filepath.Dir(path)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return "", fmt.Errorf("create directory for %s: %w", op.Path, err)
			}
			if err := os.WriteFile(path, []byte(op.Content), 0o644); err != nil {
				return "", fmt.Errorf("write new file %s: %w", op.Path, err)
			}
			results = append(results, fmt.Sprintf("Added: %s", op.Path))

		case "delete":
			if err := os.Remove(path); err != nil {
				return "", fmt.Errorf("delete file %s: %w", op.Path, err)
			}
			results = append(results, fmt.Sprintf("Deleted: %s", op.Path))

		case "update":
			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("read file %s for update: %w", op.Path, err)
			}

			content := string(data)
			content, err = applyHunks(content, op.Hunks)
			if err != nil {
				return "", fmt.Errorf("apply hunks to %s: %w", op.Path, err)
			}

			targetPath := path
			if op.MoveTo != "" {
				targetPath = filepath.Join(workDir, op.MoveTo)
				targetDir := filepath.Dir(targetPath)
				if err := os.MkdirAll(targetDir, 0o755); err != nil {
					return "", fmt.Errorf("create directory for %s: %w", op.MoveTo, err)
				}
			}

			if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
				return "", fmt.Errorf("write updated file: %w", err)
			}

			if op.MoveTo != "" && path != targetPath {
				_ = os.Remove(path) // Remove the old file.
				results = append(results, fmt.Sprintf("Updated and moved: %s -> %s", op.Path, op.MoveTo))
			} else {
				results = append(results, fmt.Sprintf("Updated: %s", op.Path))
			}
		}
	}

	return strings.Join(results, "\n"), nil
}

// applyHunks applies a series of hunks to file content.
func applyHunks(content string, hunks []Hunk) (string, error) {
	lines := strings.Split(content, "\n")

	// Apply hunks in reverse order to preserve line numbers.
	// First, find the position for each hunk.
	type positioned struct {
		hunk  Hunk
		start int
	}
	var positioned_hunks []positioned

	for _, hunk := range hunks {
		pos, err := findHunkPosition(lines, hunk)
		if err != nil {
			return "", err
		}
		positioned_hunks = append(positioned_hunks, positioned{hunk: hunk, start: pos})
	}

	// Apply in reverse order of position to avoid shifting.
	for i := len(positioned_hunks) - 1; i >= 0; i-- {
		ph := positioned_hunks[i]
		lines = applyHunkAtPosition(lines, ph.hunk, ph.start)
	}

	return strings.Join(lines, "\n"), nil
}

// findHunkPosition locates where a hunk should be applied in the file.
func findHunkPosition(lines []string, hunk Hunk) (int, error) {
	// Build the context/delete pattern to match.
	var pattern []string
	for _, hl := range hunk.Lines {
		if hl.Type == ' ' || hl.Type == '-' {
			pattern = append(pattern, hl.Content)
		}
	}

	if len(pattern) == 0 {
		// No context lines - this is a pure insertion. Use the context hint.
		if hunk.ContextHint != "" {
			for i, line := range lines {
				if strings.TrimSpace(line) == strings.TrimSpace(hunk.ContextHint) {
					return i, nil
				}
			}
		}
		return 0, nil
	}

	// Try exact match first.
	for i := 0; i <= len(lines)-len(pattern); i++ {
		if matchesAt(lines, i, pattern) {
			return i, nil
		}
	}

	// Fuzzy match: normalize whitespace.
	for i := 0; i <= len(lines)-len(pattern); i++ {
		if fuzzyMatchesAt(lines, i, pattern) {
			return i, nil
		}
	}

	return 0, fmt.Errorf("could not find hunk position (context hint: %q, first pattern line: %q)", hunk.ContextHint, pattern[0])
}

// matchesAt checks if the pattern matches exactly at position pos.
func matchesAt(lines []string, pos int, pattern []string) bool {
	for j, p := range pattern {
		if pos+j >= len(lines) || lines[pos+j] != p {
			return false
		}
	}
	return true
}

// fuzzyMatchesAt checks if the pattern matches at position pos after
// normalizing whitespace.
func fuzzyMatchesAt(lines []string, pos int, pattern []string) bool {
	for j, p := range pattern {
		if pos+j >= len(lines) {
			return false
		}
		if normalizeWhitespace(lines[pos+j]) != normalizeWhitespace(p) {
			return false
		}
	}
	return true
}

// normalizeWhitespace collapses all whitespace to single spaces and trims.
func normalizeWhitespace(s string) string {
	fields := strings.Fields(s)
	return strings.Join(fields, " ")
}

// applyHunkAtPosition applies a hunk's changes starting at the given position.
func applyHunkAtPosition(lines []string, hunk Hunk, pos int) []string {
	// Count how many original lines this hunk consumes.
	oldCount := 0
	for _, hl := range hunk.Lines {
		if hl.Type == ' ' || hl.Type == '-' {
			oldCount++
		}
	}

	// Build the replacement lines.
	var newLines []string
	for _, hl := range hunk.Lines {
		switch hl.Type {
		case ' ':
			newLines = append(newLines, hl.Content)
		case '+':
			newLines = append(newLines, hl.Content)
		case '-':
			// Removed line; do not include.
		}
	}

	// Splice: replace lines[pos:pos+oldCount] with newLines.
	result := make([]string, 0, len(lines)-oldCount+len(newLines))
	result = append(result, lines[:pos]...)
	result = append(result, newLines...)
	if pos+oldCount < len(lines) {
		result = append(result, lines[pos+oldCount:]...)
	}

	return result
}
