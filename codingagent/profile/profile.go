// Package profile defines provider-specific configurations for the coding
// agent. A ProviderProfile encapsulates the model identifier, tool registry,
// system prompt construction, and capability flags for a given LLM family
// (Anthropic, OpenAI, Gemini). This enables the session layer to remain
// provider-agnostic while respecting each model's unique strengths and
// constraints.
package profile

import (
	"fmt"
	"strings"

	"github.com/strongdm/attractor-go/codingagent/tools"
)

// ProviderProfile defines the contract for a model-family configuration.
type ProviderProfile interface {
	// ID returns a unique identifier for this profile (e.g. "anthropic-claude").
	ID() string
	// ProviderName returns the canonical provider key used for request routing
	// (e.g. "openai", "anthropic", "gemini"). This must match the provider
	// names registered in the unified LLM client.
	ProviderName() string
	// Model returns the provider-specific model identifier.
	Model() string
	// ToolRegistry returns the tool registry for this profile.
	ToolRegistry() *tools.Registry
	// BuildSystemPrompt constructs the system prompt incorporating the
	// environment context, project documentation, and provider-specific
	// instructions.
	BuildSystemPrompt(envContext string, projectDocs string) string
	// Tools returns the list of tool definitions to include in LLM requests.
	Tools() []tools.Definition
	// ProviderOptions returns provider-specific request options (e.g.
	// anthropic-specific headers or parameters).
	ProviderOptions() map[string]any
	// SupportsReasoning reports whether the model supports extended thinking.
	SupportsReasoning() bool
	// SupportsStreaming reports whether the model supports streaming responses.
	SupportsStreaming() bool
	// SupportsParallelToolCalls reports whether the model can issue multiple
	// tool calls in a single response.
	SupportsParallelToolCalls() bool
	// ContextWindowSize returns the model's context window in tokens.
	ContextWindowSize() int
	// InstructionFileNames returns the project instruction files this profile
	// should load (e.g. ["AGENTS.md", "CLAUDE.md"] for Anthropic).
	InstructionFileNames() []string
	// KnowledgeCutoff returns the model family's knowledge cutoff date string
	// (e.g. "2025-04").
	KnowledgeCutoff() string
}

// ---------------------------------------------------------------------------
// BaseProfile
// ---------------------------------------------------------------------------

// BaseProfile provides a reusable ProviderProfile implementation. Concrete
// profiles are constructed via the NewAnthropicProfile, NewOpenAIProfile, and
// NewGeminiProfile factories.
type BaseProfile struct {
	id                        string
	providerName              string
	model                     string
	registry                  *tools.Registry
	supportsReasoning         bool
	supportsStreaming          bool
	supportsParallelToolCalls bool
	contextWindowSize         int
	providerOptions           map[string]any
	systemPromptTemplate      string
	instructionFileNames      []string
	knowledgeCutoff           string
}

func (p *BaseProfile) ID() string                      { return p.id }
func (p *BaseProfile) ProviderName() string             { return p.providerName }
func (p *BaseProfile) Model() string                   { return p.model }
func (p *BaseProfile) ToolRegistry() *tools.Registry   { return p.registry }
func (p *BaseProfile) ProviderOptions() map[string]any { return p.providerOptions }
func (p *BaseProfile) SupportsReasoning() bool         { return p.supportsReasoning }
func (p *BaseProfile) SupportsStreaming() bool          { return p.supportsStreaming }
func (p *BaseProfile) SupportsParallelToolCalls() bool { return p.supportsParallelToolCalls }
func (p *BaseProfile) ContextWindowSize() int          { return p.contextWindowSize }
func (p *BaseProfile) InstructionFileNames() []string  { return p.instructionFileNames }
func (p *BaseProfile) KnowledgeCutoff() string         { return p.knowledgeCutoff }

// SetModel overrides the model identifier for this profile. This is used
// when spawning subagents with a different model than the parent session.
func (p *BaseProfile) SetModel(model string) { p.model = model }

// Tools returns all tool definitions from the registry.
func (p *BaseProfile) Tools() []tools.Definition {
	if p.registry == nil {
		return nil
	}
	return p.registry.Definitions()
}

// BuildSystemPrompt constructs the system prompt by substituting the
// environment context and project docs into the template.
func (p *BaseProfile) BuildSystemPrompt(envContext string, projectDocs string) string {
	prompt := p.systemPromptTemplate
	prompt = strings.ReplaceAll(prompt, "{{ENVIRONMENT}}", envContext)
	prompt = strings.ReplaceAll(prompt, "{{PROJECT_DOCS}}", projectDocs)
	return prompt
}

// ---------------------------------------------------------------------------
// Anthropic Profile
// ---------------------------------------------------------------------------

// NewAnthropicProfile creates a profile aligned with Claude Code conventions.
// It registers the standard coding-agent tool set and uses an Anthropic-style
// system prompt.
func NewAnthropicProfile(model string) *BaseProfile {
	registry := tools.NewRegistry()
	registerStandardTools(registry)

	return &BaseProfile{
		id:                        "anthropic-claude",
		providerName:              "anthropic",
		model:                     model,
		registry:                  registry,
		supportsReasoning:         true,
		supportsStreaming:          true,
		supportsParallelToolCalls: true,
		contextWindowSize:         200000,
		providerOptions: map[string]any{
			"anthropic_beta": []string{
				"interleaved-thinking-2025-05-14",
				"prompt-caching-2024-07-31",
				"token-efficient-tools-2025-02-19",
			},
		},
		systemPromptTemplate: anthropicSystemPrompt,
		instructionFileNames: []string{"AGENTS.md", "CLAUDE.md"},
		knowledgeCutoff:      "2025-04",
	}
}

// ---------------------------------------------------------------------------
// OpenAI Profile
// ---------------------------------------------------------------------------

// NewOpenAIProfile creates a profile aligned with codex-rs conventions.
func NewOpenAIProfile(model string) *BaseProfile {
	registry := tools.NewRegistry()
	registerStandardTools(registry)
	registerApplyPatchTool(registry)

	return &BaseProfile{
		id:                        "openai-codex",
		providerName:              "openai",
		model:                     model,
		registry:                  registry,
		supportsReasoning:         true,
		supportsStreaming:          true,
		supportsParallelToolCalls: true,
		contextWindowSize:         128000,
		providerOptions:           map[string]any{},
		systemPromptTemplate:      openAISystemPrompt,
		instructionFileNames:      []string{"AGENTS.md", ".codex/instructions.md"},
		knowledgeCutoff:           "2025-04",
	}
}

// ---------------------------------------------------------------------------
// Gemini Profile
// ---------------------------------------------------------------------------

// NewGeminiProfile creates a profile aligned with gemini-cli conventions.
func NewGeminiProfile(model string) *BaseProfile {
	registry := tools.NewRegistry()
	registerStandardTools(registry)
	registerWebTools(registry)
	registerReadManyFilesTool(registry)

	return &BaseProfile{
		id:                        "gemini-cli",
		providerName:              "gemini",
		model:                     model,
		registry:                  registry,
		supportsReasoning:         false,
		supportsStreaming:          true,
		supportsParallelToolCalls: true,
		contextWindowSize:         1000000,
		providerOptions:           map[string]any{},
		systemPromptTemplate:      geminiSystemPrompt,
		instructionFileNames:      []string{"AGENTS.md", "GEMINI.md"},
		knowledgeCutoff:           "2025-04",
	}
}

// ---------------------------------------------------------------------------
// Standard tool registration
// ---------------------------------------------------------------------------

// registerStandardTools populates the registry with the core developer tools
// shared across all profiles. The executors are stubs here; the session layer
// binds them to the ExecutionEnvironment at runtime via the tools.Registry.
func registerStandardTools(registry *tools.Registry) {
	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "read_file",
			Description: "Read the contents of a file at the given path. Returns the file contents with line numbers.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{
						"type":        "string",
						"description": "The absolute path to the file to read",
					},
					"offset": map[string]any{
						"type":        "integer",
						"description": "Line number to start reading from (1-based)",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "Maximum number of lines to read",
					},
				},
				"required": []string{"file_path"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "write_file",
			Description: "Write content to a file, creating parent directories as needed. Overwrites existing files.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{
						"type":        "string",
						"description": "The absolute path to the file to write",
					},
					"content": map[string]any{
						"type":        "string",
						"description": "The content to write to the file",
					},
				},
				"required": []string{"file_path", "content"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "edit_file",
			Description: "Perform exact string replacement in a file. The old_string must appear exactly once in the file unless replace_all is true.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{
						"type":        "string",
						"description": "The absolute path to the file to edit",
					},
					"old_string": map[string]any{
						"type":        "string",
						"description": "The exact text to find and replace",
					},
					"new_string": map[string]any{
						"type":        "string",
						"description": "The replacement text",
					},
					"replace_all": map[string]any{
						"type":        "boolean",
						"description": "If true, replace all occurrences",
					},
				},
				"required": []string{"file_path", "old_string", "new_string"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "shell",
			Description: "Execute a shell command. Returns stdout, stderr, and exit code.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "The shell command to execute",
					},
					"description": map[string]any{
						"type":        "string",
						"description": "A short description of what the command does (for logging)",
					},
					"timeout_ms": map[string]any{
						"type":        "integer",
						"description": "Timeout in milliseconds (default 10000, max 600000)",
					},
					"working_dir": map[string]any{
						"type":        "string",
						"description": "Working directory for the command",
					},
				},
				"required": []string{"command"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "grep",
			Description: "Search for a regex pattern in files. Returns matching lines with file paths and line numbers.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{
						"type":        "string",
						"description": "The regex pattern to search for",
					},
					"path": map[string]any{
						"type":        "string",
						"description": "File or directory to search in",
					},
					"case_insensitive": map[string]any{
						"type":        "boolean",
						"description": "Perform case-insensitive matching",
					},
					"max_results": map[string]any{
						"type":        "integer",
						"description": "Maximum number of results to return",
					},
					"glob_filter": map[string]any{
						"type":        "string",
						"description": "Glob pattern to filter files (e.g. '*.go')",
					},
				},
				"required": []string{"pattern"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "glob",
			Description: "Find files matching a glob pattern. Supports ** for recursive matching.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{
						"type":        "string",
						"description": "The glob pattern (e.g. '**/*.go')",
					},
					"path": map[string]any{
						"type":        "string",
						"description": "Root directory for the search",
					},
				},
				"required": []string{"pattern"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "list_directory",
			Description: "List files and directories at the given path.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Directory path to list",
					},
					"depth": map[string]any{
						"type":        "integer",
						"description": "Recursion depth (0 = immediate children)",
					},
				},
				"required": []string{"path"},
			},
		},
	})

	// Subagent tools (Section 7).
	registerSubAgentTools(registry)
}

// registerSubAgentTools adds the spawn_agent, send_input, wait, and
// close_agent tools shared across all profiles.
func registerSubAgentTools(registry *tools.Registry) {
	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "spawn_agent",
			Description: "Spawn a subagent to handle a scoped task autonomously. The subagent runs its own agentic loop with independent conversation history.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"task": map[string]any{
						"type":        "string",
						"description": "Natural language task description for the subagent",
					},
					"working_dir": map[string]any{
						"type":        "string",
						"description": "Subdirectory to scope the agent to (optional)",
					},
					"model": map[string]any{
						"type":        "string",
						"description": "Model override (default: parent's model)",
					},
					"max_turns": map[string]any{
						"type":        "integer",
						"description": "Turn limit for the subagent (0 = unlimited)",
					},
				},
				"required": []string{"task"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "send_input",
			Description: "Send a message to a running subagent.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"agent_id": map[string]any{
						"type":        "string",
						"description": "The ID of the subagent",
					},
					"message": map[string]any{
						"type":        "string",
						"description": "The message to send",
					},
				},
				"required": []string{"agent_id", "message"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "wait",
			Description: "Wait for a subagent to complete and return its result.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"agent_id": map[string]any{
						"type":        "string",
						"description": "The ID of the subagent to wait for",
					},
				},
				"required": []string{"agent_id"},
			},
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "close_agent",
			Description: "Terminate a subagent.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"agent_id": map[string]any{
						"type":        "string",
						"description": "The ID of the subagent to terminate",
					},
				},
				"required": []string{"agent_id"},
			},
		},
	})
}

// registerApplyPatchTool adds the apply_patch tool (v4a format) used by the
// OpenAI profile. OpenAI models are specifically trained on this format and
// produce significantly better edits when using it.
func registerApplyPatchTool(registry *tools.Registry) {
	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name: "apply_patch",
			Description: "Apply code changes using the v4a patch format. Supports creating, deleting, " +
				"and modifying files in a single operation. Use this tool for all file edits.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"patch": map[string]any{
						"type":        "string",
						"description": "The patch content in v4a format (*** Begin Patch ... *** End Patch)",
					},
				},
				"required": []string{"patch"},
			},
		},
	})
}

// registerReadManyFilesTool adds the read_many_files tool used by the Gemini
// profile. Gemini's large context window makes it efficient to read multiple
// files in a single tool call, reducing round-trips. The executor is nil
// because the session layer provides a built-in handler.
func registerReadManyFilesTool(registry *tools.Registry) {
	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "read_many_files",
			Description: "Read the contents of multiple files in a single operation. Returns each file's contents concatenated with file path headers. More efficient than multiple read_file calls.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"paths": map[string]any{
						"type":        "array",
						"description": "List of absolute file paths to read",
						"items": map[string]any{
							"type": "string",
						},
					},
				},
				"required": []string{"paths"},
			},
		},
	})
}

// registerWebTools adds web_search and web_fetch stub tools used by the
// Gemini profile. These tools are defined so the model can reference them;
// the executors return "not implemented" until a backend is wired in.
func registerWebTools(registry *tools.Registry) {
	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "web_search",
			Description: "Search the web for information. Returns a list of search results with titles, URLs, and snippets.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "The search query",
					},
					"max_results": map[string]any{
						"type":        "integer",
						"description": "Maximum number of results to return (default 5)",
					},
				},
				"required": []string{"query"},
			},
		},
		Executor: func(args map[string]any, env any) (string, error) {
			return "web_search is not implemented", nil
		},
	})

	registry.Register(&tools.RegisteredTool{
		Definition: tools.Definition{
			Name:        "web_fetch",
			Description: "Fetch the content of a web page at the given URL. Returns the page content as text.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The URL to fetch",
					},
					"extract_text": map[string]any{
						"type":        "boolean",
						"description": "If true, extract text content only (strip HTML)",
					},
				},
				"required": []string{"url"},
			},
		},
		Executor: func(args map[string]any, env any) (string, error) {
			return "web_fetch is not implemented", nil
		},
	})
}

// ---------------------------------------------------------------------------
// System prompt templates
// ---------------------------------------------------------------------------

var anthropicSystemPrompt = fmt.Sprintf(`You are Claude, an AI assistant by Anthropic. You are an expert software engineer with deep experience in designing, building, and scaling high-performance software systems.

{{ENVIRONMENT}}

%s

Tool usage guidelines:
- Prefer dedicated tools (read_file, edit_file, grep, glob) over shell commands for file operations.
- Use edit_file for all file modifications. It uses old_string/new_string exact matching.
  The old_string must be unique in the file. Provide enough surrounding context to ensure uniqueness.
- Use read_file before editing to understand file structure and find the exact text to match.
- Use shell for build commands, test execution, git operations, and other system tasks.
- Use grep and glob for codebase exploration rather than shell find/grep.

Shell command safety:
- Never run destructive commands (rm -rf, git push --force, etc.) without explicit user request.
- Always quote file paths containing spaces.
- Use timeouts for long-running commands.
- Prefer non-interactive commands; avoid commands requiring stdin input.

When making changes:
- Read relevant files before editing to understand context and conventions.
- Make targeted, minimal changes using edit_file old_string/new_string format.
- Verify changes compile and pass tests when possible.
- Follow existing code conventions, patterns, and naming styles.
- Explain your reasoning for non-obvious changes.

{{PROJECT_DOCS}}`, coreInstructions)

var openAISystemPrompt = fmt.Sprintf(`You are a coding assistant powered by OpenAI. You help developers write, debug, and improve code efficiently.

{{ENVIRONMENT}}

%s

Tool usage guidelines:
- Use apply_patch for all file modifications. It uses the v4a patch format:
  *** Begin Patch / *** End Patch with per-file sections containing context and change lines.
- Use read_file before editing to understand file contents and structure.
- Use shell for build, test, and git operations.
- Use grep and glob for codebase search and discovery.
- Be precise and efficient with tool calls; minimize unnecessary round-trips.

When making changes:
- Read relevant files first to understand context.
- Construct patches carefully with correct context lines.
- Group related changes into a single apply_patch call when possible.
- Verify changes compile and pass tests.
- Follow existing code conventions and patterns.

{{PROJECT_DOCS}}`, coreInstructions)

var geminiSystemPrompt = fmt.Sprintf(`You are a coding assistant powered by Google Gemini. You help developers understand, write, and maintain code across large codebases.

{{ENVIRONMENT}}

%s

Tool usage guidelines:
- Use read_many_files to efficiently load multiple related files in a single call.
- Use edit_file for targeted modifications using old_string/new_string matching.
- Use grep and glob for codebase exploration and discovery.
- Use shell for build commands, tests, and system operations.
- Use web_search and web_fetch when you need external documentation or references.
- Check for GEMINI.md in the project root for project-specific conventions.

When making changes:
- Leverage your large context window to read broadly before making changes.
- Read relevant files and related code before editing.
- Make targeted, minimal changes that follow existing patterns.
- Verify changes compile and pass tests when possible.
- Follow existing code conventions and naming styles.

{{PROJECT_DOCS}}`, coreInstructions)

const coreInstructions = `Key guidelines:
- Always use absolute file paths
- Read files before editing them
- Prefer targeted edits over full file rewrites
- Run tests after making changes to verify correctness
- Do not commit changes unless explicitly asked
- Do not make changes outside the scope of what was requested`
