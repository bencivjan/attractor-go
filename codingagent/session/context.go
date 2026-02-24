package session

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/strongdm/attractor-go/codingagent/env"
	"github.com/strongdm/attractor-go/codingagent/profile"
)

// buildEnvironmentContext generates the structured <environment> block
// included in every system prompt. It is generated once at session start
// and cached for the session lifetime.
func buildEnvironmentContext(execEnv env.ExecutionEnvironment, prof profile.ProviderProfile) string {
	workDir := execEnv.WorkingDirectory()
	platform := execEnv.Platform()
	osVersion := execEnv.OSVersion()
	model := prof.Model()
	cutoff := prof.KnowledgeCutoff()
	date := time.Now().Format("2006-01-02")

	// Detect git repository.
	isGitRepo := false
	gitBranch := ""

	gitStatus := ""
	gitRecentCommits := ""

	result, err := execEnv.ExecCommand(context.Background(), "git rev-parse --is-inside-work-tree", 5000, workDir, nil)
	if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) == "true" {
		isGitRepo = true
		branchResult, _ := execEnv.ExecCommand(context.Background(), "git branch --show-current", 5000, workDir, nil)
		if branchResult != nil && branchResult.ExitCode == 0 {
			gitBranch = strings.TrimSpace(branchResult.Stdout)
		}
		// Short status: count modified and untracked files.
		statusResult, _ := execEnv.ExecCommand(context.Background(), "git status --short", 5000, workDir, nil)
		if statusResult != nil && statusResult.ExitCode == 0 {
			statusOut := strings.TrimSpace(statusResult.Stdout)
			if statusOut != "" {
				lines := strings.Split(statusOut, "\n")
				gitStatus = fmt.Sprintf("%d file(s) with changes", len(lines))
			} else {
				gitStatus = "clean"
			}
		}
		// Recent commits.
		logResult, _ := execEnv.ExecCommand(context.Background(), "git log --oneline -5 2>/dev/null", 5000, workDir, nil)
		if logResult != nil && logResult.ExitCode == 0 {
			gitRecentCommits = strings.TrimSpace(logResult.Stdout)
		}
	}

	var b strings.Builder
	b.WriteString("<environment>\n")
	fmt.Fprintf(&b, "Working directory: %s\n", workDir)
	fmt.Fprintf(&b, "Is git repository: %v\n", isGitRepo)
	if isGitRepo && gitBranch != "" {
		fmt.Fprintf(&b, "Git branch: %s\n", gitBranch)
	}
	if gitStatus != "" {
		fmt.Fprintf(&b, "Git status: %s\n", gitStatus)
	}
	if gitRecentCommits != "" {
		fmt.Fprintf(&b, "Recent commits:\n%s\n", gitRecentCommits)
	}
	fmt.Fprintf(&b, "Platform: %s\n", platform)
	if osVersion != "" {
		fmt.Fprintf(&b, "OS version: %s\n", osVersion)
	}
	fmt.Fprintf(&b, "Today's date: %s\n", date)
	fmt.Fprintf(&b, "Model: %s\n", model)
	if cutoff != "" {
		fmt.Fprintf(&b, "Knowledge cutoff: %s\n", cutoff)
	}
	b.WriteString("</environment>")

	return b.String()
}

// projectDocBudget is the maximum total bytes of project instruction files
// to include in the system prompt.
const projectDocBudget = 32 * 1024

// discoverProjectDocs walks from the git root (or working directory) to the
// current working directory, loading recognized instruction files for the
// active provider profile. Root-level files are loaded first; deeper files
// are appended with higher precedence. The total is capped at 32KB.
func discoverProjectDocs(execEnv env.ExecutionEnvironment, fileNames []string) string {
	if len(fileNames) == 0 {
		return ""
	}

	workDir := execEnv.WorkingDirectory()

	// Find git root.
	gitRoot := ""
	result, err := execEnv.ExecCommand(context.Background(), "git rev-parse --show-toplevel", 5000, workDir, nil)
	if err == nil && result.ExitCode == 0 {
		gitRoot = strings.TrimSpace(result.Stdout)
	}

	startDir := gitRoot
	if startDir == "" {
		startDir = workDir
	}

	// Build directory chain from startDir to workDir.
	dirs := directoryChain(startDir, workDir)

	var docs []string
	totalBytes := 0

	for _, dir := range dirs {
		for _, fileName := range fileNames {
			path := filepath.Join(dir, fileName)

			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			content := string(data)

			if totalBytes+len(content) > projectDocBudget {
				remaining := projectDocBudget - totalBytes
				if remaining > 0 {
					docs = append(docs, fmt.Sprintf("# %s\n%s", path, content[:remaining]))
					docs = append(docs, "\n[Project instructions truncated at 32KB]")
				}
				return strings.Join(docs, "\n\n")
			}

			docs = append(docs, fmt.Sprintf("# %s\n%s", path, content))
			totalBytes += len(content)
		}
	}

	return strings.Join(docs, "\n\n")
}

// directoryChain returns a list of directories from 'from' to 'to',
// inclusive of both endpoints. If 'to' is not under 'from', both are
// returned.
func directoryChain(from, to string) []string {
	from = filepath.Clean(from)
	to = filepath.Clean(to)

	if from == to {
		return []string{from}
	}

	rel, err := filepath.Rel(from, to)
	if err != nil || strings.HasPrefix(rel, "..") {
		return []string{from, to}
	}

	chain := []string{from}
	parts := strings.Split(rel, string(filepath.Separator))
	current := from
	for _, part := range parts {
		current = filepath.Join(current, part)
		chain = append(chain, current)
	}

	return chain
}
