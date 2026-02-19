//go:build !js

package env

import (
	"bufio"
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// ---------------------------------------------------------------------------
// LocalEnvironment
// ---------------------------------------------------------------------------

// LocalEnvironment executes everything on the host machine.
type LocalEnvironment struct {
	workDir   string
	platform  string
	osVersion string
}

// NewLocalEnvironment creates a LocalEnvironment rooted at workDir.
func NewLocalEnvironment(workDir string) *LocalEnvironment {
	return &LocalEnvironment{
		workDir:   workDir,
		platform:  runtime.GOOS,
		osVersion: "",
	}
}

// Initialize detects the OS version and validates the working directory.
func (e *LocalEnvironment) Initialize() error {
	// Ensure working directory exists.
	info, err := os.Stat(e.workDir)
	if err != nil {
		return fmt.Errorf("working directory %q: %w", e.workDir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("working directory %q is not a directory", e.workDir)
	}

	// Detect OS version.
	e.osVersion = detectOSVersion()
	return nil
}

// Cleanup is a no-op for the local environment.
func (e *LocalEnvironment) Cleanup() error {
	return nil
}

// WorkingDirectory returns the configured working directory.
func (e *LocalEnvironment) WorkingDirectory() string {
	return e.workDir
}

// Platform returns the runtime OS.
func (e *LocalEnvironment) Platform() string {
	return e.platform
}

// OSVersion returns the detected OS version string.
func (e *LocalEnvironment) OSVersion() string {
	return e.osVersion
}

// ReadFile reads a file and returns its contents with line numbers. When
// offset > 0, lines before offset are skipped. When limit > 0, at most
// limit lines are returned.
func (e *LocalEnvironment) ReadFile(path string, offset, limit int) (string, error) {
	path = e.resolvePath(path)

	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}
	defer f.Close()

	var b strings.Builder
	scanner := bufio.NewScanner(f)
	// Increase scanner buffer for large lines.
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	lineNum := 0
	linesWritten := 0

	for scanner.Scan() {
		lineNum++

		// Skip lines before offset (1-based).
		if offset > 0 && lineNum < offset {
			continue
		}

		// Respect limit.
		if limit > 0 && linesWritten >= limit {
			break
		}

		line := scanner.Text()
		// Truncate very long lines to prevent context overflow.
		const maxLineLen = 2000
		if len(line) > maxLineLen {
			line = line[:maxLineLen] + "..."
		}

		fmt.Fprintf(&b, "%6d\t%s\n", lineNum, line)
		linesWritten++
	}

	if err := scanner.Err(); err != nil {
		return b.String(), fmt.Errorf("read file scan: %w", err)
	}

	return b.String(), nil
}

// WriteFile writes content to path, creating any necessary parent directories.
func (e *LocalEnvironment) WriteFile(path, content string) error {
	path = e.resolvePath(path)

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create parent dirs: %w", err)
	}

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Errorf("write file: %w", err)
	}
	return nil
}

// FileExists reports whether a file or directory exists at path.
func (e *LocalEnvironment) FileExists(path string) bool {
	path = e.resolvePath(path)
	_, err := os.Stat(path)
	return err == nil
}

// ListDirectory lists entries up to the given depth (0 = immediate children only).
func (e *LocalEnvironment) ListDirectory(path string, depth int) ([]DirEntry, error) {
	path = e.resolvePath(path)

	var entries []DirEntry
	err := e.walkDir(path, path, depth, 0, &entries)
	if err != nil {
		return nil, fmt.Errorf("list directory: %w", err)
	}
	return entries, nil
}

func (e *LocalEnvironment) walkDir(root, current string, maxDepth, currentDepth int, entries *[]DirEntry) error {
	dirEntries, err := os.ReadDir(current)
	if err != nil {
		return err
	}

	for _, de := range dirEntries {
		info, err := de.Info()
		if err != nil {
			continue
		}

		relPath, _ := filepath.Rel(root, filepath.Join(current, de.Name()))
		*entries = append(*entries, DirEntry{
			Name:  relPath,
			IsDir: de.IsDir(),
			Size:  info.Size(),
		})

		if de.IsDir() && currentDepth < maxDepth {
			if err := e.walkDir(root, filepath.Join(current, de.Name()), maxDepth, currentDepth+1, entries); err != nil {
				// Skip directories we cannot read rather than failing entirely.
				continue
			}
		}
	}
	return nil
}

// ExecCommand runs a shell command with timeout enforcement. The command is
// spawned in its own process group. On timeout, SIGTERM is sent first; if
// the process does not exit within 5 seconds, SIGKILL is sent.
func (e *LocalEnvironment) ExecCommand(ctx context.Context, command string, timeoutMs int, workingDir string, envVars map[string]string) (*ExecResult, error) {
	if workingDir == "" {
		workingDir = e.workDir
	} else {
		workingDir = e.resolvePathFrom(workingDir)
	}

	if timeoutMs <= 0 {
		timeoutMs = 10000
	}

	timeout := time.Duration(timeoutMs) * time.Millisecond
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = workingDir

	// Set up process group for clean termination.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	// Build environment: inherit current env, then overlay provided vars.
	// Filter out potentially dangerous variables.
	cmdEnv := filterEnv(os.Environ())
	for k, v := range envVars {
		cmdEnv = append(cmdEnv, k+"="+v)
	}
	cmd.Env = cmdEnv

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Start()
	if err != nil {
		return &ExecResult{
			ExitCode:   -1,
			DurationMs: time.Since(start).Milliseconds(),
		}, fmt.Errorf("exec start: %w", err)
	}

	// Wait for the command to finish or the context to expire.
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	var timedOut bool
	select {
	case err = <-done:
		// Command completed (possibly with non-zero exit).
	case <-ctx.Done():
		timedOut = true
		// Send SIGTERM to the process group.
		if cmd.Process != nil {
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		}

		// Give the process 5 seconds to terminate gracefully.
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			if cmd.Process != nil {
				_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			}
			<-done
		}
	}

	duration := time.Since(start).Milliseconds()

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if timedOut {
			exitCode = 124 // Convention: 124 = timed out
		} else {
			exitCode = -1
		}
	}

	return &ExecResult{
		Stdout:     stdout.String(),
		Stderr:     stderr.String(),
		ExitCode:   exitCode,
		TimedOut:   timedOut,
		DurationMs: duration,
	}, nil
}

// Grep searches for pattern matches in files. It attempts to use ripgrep (rg)
// for performance, falling back to a Go-native regex search.
func (e *LocalEnvironment) Grep(pattern, path string, caseInsensitive bool, maxResults int, globFilter string) (string, error) {
	path = e.resolvePath(path)

	// Try ripgrep first.
	if rgPath, err := exec.LookPath("rg"); err == nil {
		return e.grepWithRipgrep(rgPath, pattern, path, caseInsensitive, maxResults, globFilter)
	}

	// Fallback to native Go regex search.
	return e.grepNative(pattern, path, caseInsensitive, maxResults, globFilter)
}

func (e *LocalEnvironment) grepWithRipgrep(rgPath, pattern, path string, caseInsensitive bool, maxResults int, globFilter string) (string, error) {
	args := []string{"-n", "--no-heading", "--color=never"}

	if caseInsensitive {
		args = append(args, "-i")
	}
	if maxResults > 0 {
		args = append(args, fmt.Sprintf("--max-count=%d", maxResults))
	}
	if globFilter != "" {
		args = append(args, "--glob", globFilter)
	}

	args = append(args, pattern, path)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, rgPath, args...)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		// Exit code 1 means no matches, which is not an error.
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			return "", nil
		}
		// Exit code 2 means an actual error.
		if stderr.Len() > 0 {
			return "", fmt.Errorf("ripgrep: %s", stderr.String())
		}
		return "", fmt.Errorf("ripgrep: %w", err)
	}

	return stdout.String(), nil
}

func (e *LocalEnvironment) grepNative(pattern, path string, caseInsensitive bool, maxResults int, globFilter string) (string, error) {
	flags := ""
	if caseInsensitive {
		flags = "(?i)"
	}
	re, err := regexp.Compile(flags + pattern)
	if err != nil {
		return "", fmt.Errorf("invalid pattern: %w", err)
	}

	var results strings.Builder
	matchCount := 0

	err = filepath.WalkDir(path, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil // Skip inaccessible entries.
		}
		if d.IsDir() {
			// Skip hidden directories.
			if d.Name() != "." && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}

		// Apply glob filter if specified.
		if globFilter != "" {
			matched, _ := filepath.Match(globFilter, d.Name())
			if !matched {
				return nil
			}
		}

		// Skip binary files by checking extension.
		if isBinaryExtension(filepath.Ext(p)) {
			return nil
		}

		f, err := os.Open(p)
		if err != nil {
			return nil
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 1*1024*1024)
		lineNum := 0
		for scanner.Scan() {
			lineNum++
			line := scanner.Text()
			if re.MatchString(line) {
				relPath, _ := filepath.Rel(e.workDir, p)
				if relPath == "" {
					relPath = p
				}
				fmt.Fprintf(&results, "%s:%d:%s\n", relPath, lineNum, line)
				matchCount++
				if maxResults > 0 && matchCount >= maxResults {
					return filepath.SkipAll
				}
			}
		}
		return nil
	})

	if err != nil {
		return results.String(), fmt.Errorf("grep walk: %w", err)
	}
	return results.String(), nil
}

// Glob returns file paths matching the pattern under the given root path.
func (e *LocalEnvironment) Glob(pattern, path string) ([]string, error) {
	if path == "" {
		path = e.workDir
	} else {
		path = e.resolvePath(path)
	}

	// If pattern contains "**", walk manually; otherwise use filepath.Glob.
	if strings.Contains(pattern, "**") {
		return e.globRecursive(pattern, path)
	}

	fullPattern := filepath.Join(path, pattern)
	matches, err := filepath.Glob(fullPattern)
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}

	// Return paths relative to the root path.
	var relMatches []string
	for _, m := range matches {
		rel, _ := filepath.Rel(path, m)
		if rel == "" {
			rel = m
		}
		relMatches = append(relMatches, rel)
	}
	return relMatches, nil
}

func (e *LocalEnvironment) globRecursive(pattern, root string) ([]string, error) {
	// Simplified ** expansion: walk the tree and match each relative path
	// against the pattern. For a production implementation, consider the
	// doublestar library for full glob semantics.
	var matches []string

	// Convert ** pattern to a regex for matching.
	regexPattern := globToRegex(pattern)
	re, err := regexp.Compile("^" + regexPattern + "$")
	if err != nil {
		return nil, fmt.Errorf("glob pattern compile: %w", err)
	}

	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		// Skip hidden directories.
		if d.IsDir() && d.Name() != "." && strings.HasPrefix(d.Name(), ".") {
			return filepath.SkipDir
		}
		rel, _ := filepath.Rel(root, p)
		if rel == "." {
			return nil
		}
		if re.MatchString(rel) {
			matches = append(matches, rel)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("glob walk: %w", err)
	}
	return matches, nil
}

// resolvePath makes path absolute relative to the working directory.
func (e *LocalEnvironment) resolvePath(path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(e.workDir, path)
}

// resolvePathFrom is identical to resolvePath but named for clarity at call sites.
func (e *LocalEnvironment) resolvePathFrom(path string) string {
	return e.resolvePath(path)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// detectOSVersion returns a human-readable OS version string.
func detectOSVersion() string {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("sw_vers", "-productVersion").Output()
		if err == nil {
			return "macOS " + strings.TrimSpace(string(out))
		}
	case "linux":
		// Try /etc/os-release.
		data, err := os.ReadFile("/etc/os-release")
		if err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "PRETTY_NAME=") {
					return strings.Trim(strings.TrimPrefix(line, "PRETTY_NAME="), "\"")
				}
			}
		}
	}

	// Fallback to uname.
	out, err := exec.Command("uname", "-sr").Output()
	if err == nil {
		return strings.TrimSpace(string(out))
	}
	return runtime.GOOS
}

// filterEnv removes sensitive environment variables from the inherited env.
func filterEnv(environ []string) []string {
	blocked := map[string]bool{
		"AWS_SECRET_ACCESS_KEY": true,
		"AWS_SESSION_TOKEN":    true,
		"OPENAI_API_KEY":       true,
		"ANTHROPIC_API_KEY":    true,
		"GOOGLE_API_KEY":       true,
		"GITHUB_TOKEN":         true,
	}

	var filtered []string
	for _, e := range environ {
		key, _, ok := strings.Cut(e, "=")
		if ok && blocked[key] {
			continue
		}
		filtered = append(filtered, e)
	}
	return filtered
}

// globToRegex converts a glob pattern with ** support to a regular expression.
func globToRegex(pattern string) string {
	var b strings.Builder
	i := 0
	for i < len(pattern) {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				// ** matches any path segment(s).
				if i+2 < len(pattern) && pattern[i+2] == '/' {
					b.WriteString("(.*/)?")
					i += 3
					continue
				}
				b.WriteString(".*")
				i += 2
				continue
			}
			// Single * matches anything except path separator.
			b.WriteString("[^/]*")
		case '?':
			b.WriteString("[^/]")
		case '.':
			b.WriteString("\\.")
		case '\\':
			b.WriteString("\\\\")
		case '{':
			b.WriteString("(")
		case '}':
			b.WriteString(")")
		case ',':
			b.WriteString("|")
		default:
			b.WriteByte(ch)
		}
		i++
	}
	return b.String()
}

// isBinaryExtension reports whether the file extension indicates a binary file.
func isBinaryExtension(ext string) bool {
	switch strings.ToLower(ext) {
	case ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a",
		".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
		".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
		".mp3", ".mp4", ".avi", ".mov", ".wav",
		".pdf", ".doc", ".docx", ".xls", ".xlsx",
		".wasm", ".class", ".pyc":
		return true
	}
	return false
}
