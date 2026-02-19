// Package env defines the ExecutionEnvironment interface and a local
// implementation. The ExecutionEnvironment abstracts file system access,
// command execution, and search operations so that the coding agent can
// operate against different backends (local machine, container, remote VM)
// through a single interface.
package env

import "context"

// ExecResult captures the outcome of running a shell command.
type ExecResult struct {
	Stdout     string
	Stderr     string
	ExitCode   int
	TimedOut   bool
	DurationMs int64
}

// DirEntry represents a single entry in a directory listing.
type DirEntry struct {
	Name  string
	IsDir bool
	Size  int64
}

// ExecutionEnvironment abstracts where tools run. Implementations may target
// the local OS, a Docker container, a remote VM, or a sandboxed environment.
type ExecutionEnvironment interface {
	// ReadFile reads the file at path. If offset > 0, skip that many lines
	// from the start. If limit > 0, return at most that many lines. Lines
	// are returned with "  N\t" prefixes (cat -n style).
	ReadFile(path string, offset, limit int) (string, error)

	// WriteFile writes content to path, creating parent directories as needed.
	WriteFile(path, content string) error

	// FileExists reports whether the file or directory at path exists.
	FileExists(path string) bool

	// ListDirectory returns entries in the directory at path, recursing up
	// to depth levels (0 = only immediate children).
	ListDirectory(path string, depth int) ([]DirEntry, error)

	// ExecCommand runs a shell command with the given timeout, working
	// directory, and environment variables. The command is executed in its
	// own process group to allow clean termination.
	ExecCommand(ctx context.Context, command string, timeoutMs int, workingDir string, envVars map[string]string) (*ExecResult, error)

	// Grep searches for a regex pattern in files at or under path. If
	// caseInsensitive is true the match is case-folded. maxResults caps the
	// number of matching lines returned. globFilter limits which files are
	// searched (e.g. "*.go").
	Grep(pattern, path string, caseInsensitive bool, maxResults int, globFilter string) (string, error)

	// Glob returns file paths matching the glob pattern rooted at path.
	Glob(pattern, path string) ([]string, error)

	// WorkingDirectory returns the base working directory for this environment.
	WorkingDirectory() string

	// Platform returns the OS name (e.g. "darwin", "linux").
	Platform() string

	// OSVersion returns the OS version string.
	OSVersion() string

	// Initialize prepares the environment for use.
	Initialize() error

	// Cleanup releases any resources held by the environment.
	Cleanup() error
}
