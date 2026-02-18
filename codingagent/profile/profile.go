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
	// Model returns the provider-specific model identifier.
	Model() string
	// ToolRegistry returns the tool registry for this profile.
	ToolRegistry() *tools.Registry
	// BuildSystemPrompt constructs the system prompt incorporating the
	// working directory, project documentation, and provider-specific
	// instructions.
	BuildSystemPrompt(workDir string, projectDocs string) string
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
}

// ---------------------------------------------------------------------------
// BaseProfile
// ---------------------------------------------------------------------------

// BaseProfile provides a reusable ProviderProfile implementation. Concrete
// profiles are constructed via the NewAnthropicProfile, NewOpenAIProfile, and
// NewGeminiProfile factories.
type BaseProfile struct {
	id                        string
	model                     string
	registry                  *tools.Registry
	supportsReasoning         bool
	supportsStreaming          bool
	supportsParallelToolCalls bool
	contextWindowSize         int
	providerOptions           map[string]any
	systemPromptTemplate      string
}

func (p *BaseProfile) ID() string                    { return p.id }
func (p *BaseProfile) Model() string                 { return p.model }
func (p *BaseProfile) ToolRegistry() *tools.Registry  { return p.registry }
func (p *BaseProfile) ProviderOptions() map[string]any { return p.providerOptions }
func (p *BaseProfile) SupportsReasoning() bool        { return p.supportsReasoning }
func (p *BaseProfile) SupportsStreaming() bool         { return p.supportsStreaming }
func (p *BaseProfile) SupportsParallelToolCalls() bool { return p.supportsParallelToolCalls }
func (p *BaseProfile) ContextWindowSize() int          { return p.contextWindowSize }

// Tools returns all tool definitions from the registry.
func (p *BaseProfile) Tools() []tools.Definition {
	if p.registry == nil {
		return nil
	}
	return p.registry.Definitions()
}

// BuildSystemPrompt constructs the system prompt by substituting the working
// directory and project docs into the template.
func (p *BaseProfile) BuildSystemPrompt(workDir string, projectDocs string) string {
	prompt := p.systemPromptTemplate
	prompt = strings.ReplaceAll(prompt, "{{WORKING_DIR}}", workDir)
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
		model:                     model,
		registry:                  registry,
		supportsReasoning:         true,
		supportsStreaming:          true,
		supportsParallelToolCalls: true,
		contextWindowSize:         200000,
		providerOptions:           map[string]any{},
		systemPromptTemplate:      anthropicSystemPrompt,
	}
}

// ---------------------------------------------------------------------------
// OpenAI Profile
// ---------------------------------------------------------------------------

// NewOpenAIProfile creates a profile aligned with codex-rs conventions.
func NewOpenAIProfile(model string) *BaseProfile {
	registry := tools.NewRegistry()
	registerStandardTools(registry)

	return &BaseProfile{
		id:                        "openai-codex",
		model:                     model,
		registry:                  registry,
		supportsReasoning:         true,
		supportsStreaming:          true,
		supportsParallelToolCalls: true,
		contextWindowSize:         128000,
		providerOptions:           map[string]any{},
		systemPromptTemplate:      openAISystemPrompt,
	}
}

// ---------------------------------------------------------------------------
// Gemini Profile
// ---------------------------------------------------------------------------

// NewGeminiProfile creates a profile aligned with gemini-cli conventions.
func NewGeminiProfile(model string) *BaseProfile {
	registry := tools.NewRegistry()
	registerStandardTools(registry)

	return &BaseProfile{
		id:                        "gemini-cli",
		model:                     model,
		registry:                  registry,
		supportsReasoning:         false,
		supportsStreaming:          true,
		supportsParallelToolCalls: true,
		contextWindowSize:         1000000,
		providerOptions:           map[string]any{},
		systemPromptTemplate:      geminiSystemPrompt,
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
}

// ---------------------------------------------------------------------------
// System prompt templates
// ---------------------------------------------------------------------------

var anthropicSystemPrompt = fmt.Sprintf(`You are an expert software engineer with deep experience in designing, building, and scaling high-performance software systems.

Working directory: {{WORKING_DIR}}

%s

You have access to tools for reading files, writing files, editing files, running shell commands, and searching the codebase. Use these tools to understand the codebase and make changes.

When making changes:
- Read relevant files before editing to understand context
- Make targeted, minimal changes
- Verify your changes compile and pass tests when possible
- Follow existing code conventions and patterns

{{PROJECT_DOCS}}`, coreInstructions)

var openAISystemPrompt = fmt.Sprintf(`You are a coding assistant with access to developer tools.

Working directory: {{WORKING_DIR}}

%s

You can read, write, and edit files, run shell commands, and search the codebase. Be precise and efficient with tool calls.

{{PROJECT_DOCS}}`, coreInstructions)

var geminiSystemPrompt = fmt.Sprintf(`You are a helpful coding assistant.

Working directory: {{WORKING_DIR}}

%s

Tools are available for file operations, command execution, and code search. Use them to explore and modify the codebase as needed.

{{PROJECT_DOCS}}`, coreInstructions)

const coreInstructions = `Key guidelines:
- Always use absolute file paths
- Read files before editing them
- Prefer targeted edits over full file rewrites
- Run tests after making changes to verify correctness
- Do not commit changes unless explicitly asked
- Do not make changes outside the scope of what was requested`
