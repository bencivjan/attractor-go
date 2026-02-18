// Package tools provides a registry for tool definitions and executors used by
// the coding agent. Each registered tool pairs a schema (Definition) with an
// execution function, enabling the agentic loop to discover available tools and
// dispatch invocations at runtime.
package tools

import "sync"

// Definition is a tool definition suitable for inclusion in an LLM request.
// Parameters follows JSON Schema conventions.
type Definition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// RegisteredTool pairs a Definition with an executor function. The executor
// receives parsed arguments and an opaque environment handle, and returns
// the tool output as a string or an error.
type RegisteredTool struct {
	Definition Definition
	// Executor runs the tool. The env parameter is the ExecutionEnvironment
	// (typed as any to avoid a circular import).
	Executor func(args map[string]any, env any) (string, error)
}

// Registry maps tool names to registered tools. It is safe for concurrent use.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]*RegisteredTool
}

// NewRegistry creates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]*RegisteredTool),
	}
}

// Register adds a tool to the registry. If a tool with the same name already
// exists it is silently replaced.
func (r *Registry) Register(tool *RegisteredTool) {
	if tool == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Definition.Name] = tool
}

// Unregister removes a tool by name. It is a no-op if the name is not present.
func (r *Registry) Unregister(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.tools, name)
}

// Get returns the registered tool with the given name, or nil if not found.
func (r *Registry) Get(name string) *RegisteredTool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tools[name]
}

// Definitions returns a snapshot of all tool definitions in the registry.
// The order is non-deterministic.
func (r *Registry) Definitions() []Definition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	defs := make([]Definition, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, t.Definition)
	}
	return defs
}

// Names returns a snapshot of all registered tool names. The order is
// non-deterministic.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

// Len returns the number of registered tools.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}
