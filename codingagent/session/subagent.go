// Package session — subagent management.
//
// A subagent is a child Session spawned by the parent to handle a scoped task.
// It runs its own agentic loop with independent conversation history but shares
// the parent's execution environment. Depth limiting prevents recursive spawning
// (default max depth: 1, configurable via Config.MaxSubagentDepth).
package session

import (
	"context"
	"fmt"
	"sync"

	"github.com/strongdm/attractor-go/codingagent/env"
	"github.com/strongdm/attractor-go/codingagent/events"
	"github.com/strongdm/attractor-go/codingagent/profile"
)

// ---------------------------------------------------------------------------
// SubAgent types
// ---------------------------------------------------------------------------

// SubAgentStatus represents the lifecycle state of a subagent.
type SubAgentStatus string

const (
	SubAgentRunning   SubAgentStatus = "running"
	SubAgentCompleted SubAgentStatus = "completed"
	SubAgentFailed    SubAgentStatus = "failed"
)

// SubAgentHandle tracks a spawned subagent.
type SubAgentHandle struct {
	ID      string
	Session *Session
	Status  SubAgentStatus
	Result  *SubAgentResult

	mu     sync.Mutex
	doneCh chan struct{}
}

// SubAgentResult is the outcome of a subagent's run.
type SubAgentResult struct {
	Output    string
	Success   bool
	TurnsUsed int
}

// ---------------------------------------------------------------------------
// SubAgent manager (embedded in Session)
// ---------------------------------------------------------------------------

// SubAgentManager manages child sessions for a parent session.
type SubAgentManager struct {
	parent  *Session
	agents  map[string]*SubAgentHandle
	counter int
	mu      sync.Mutex
}

// newSubAgentManager creates a manager for the given parent session.
func newSubAgentManager(parent *Session) *SubAgentManager {
	return &SubAgentManager{
		parent: parent,
		agents: make(map[string]*SubAgentHandle),
	}
}

// Spawn creates a child session for the given task. Returns the agent ID.
func (m *SubAgentManager) Spawn(ctx context.Context, task string, workingDir string, model string, maxTurns int) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check depth limit.
	parentDepth := m.parent.depth
	maxDepth := m.parent.Config.MaxSubagentDepth
	if maxDepth > 0 && parentDepth >= maxDepth {
		return "", fmt.Errorf("subagent depth limit reached (current depth: %d, max: %d)", parentDepth, maxDepth)
	}

	// Generate agent ID.
	m.counter++
	agentID := fmt.Sprintf("agent-%s-%d", m.parent.ID[:8], m.counter)

	// Build child config.
	childCfg := m.parent.Config
	if maxTurns > 0 {
		childCfg.MaxToolRoundsPerInput = maxTurns
	}

	// Resolve profile. If model override is specified and differs from the
	// parent's model, clone the parent's profile with the new model. This
	// enables subagents to use cheaper or more capable models as needed.
	childProfile := m.parent.Profile
	if model != "" && model != m.parent.Profile.Model() {
		childProfile = cloneProfileWithModel(m.parent.Profile, model)
	}

	// Resolve execution environment. If workingDir is specified, create a
	// scoped environment.
	childEnv := m.parent.ExecutionEnv
	if workingDir != "" {
		scopedEnv, err := scopeEnvironment(childEnv, workingDir)
		if err != nil {
			return "", fmt.Errorf("scope environment to %s: %w", workingDir, err)
		}
		childEnv = scopedEnv
	}

	// Create child session.
	child := newChildSession(agentID, childProfile, childEnv, m.parent.LLMClient, childCfg, parentDepth+1)

	handle := &SubAgentHandle{
		ID:      agentID,
		Session: child,
		Status:  SubAgentRunning,
		doneCh:  make(chan struct{}),
	}

	m.agents[agentID] = handle

	// Run the subagent asynchronously.
	go m.runSubAgent(ctx, handle, task)

	return agentID, nil
}

// SendInput sends a message to a running subagent.
func (m *SubAgentManager) SendInput(agentID string, message string) error {
	m.mu.Lock()
	handle, ok := m.agents[agentID]
	m.mu.Unlock()

	if !ok {
		return fmt.Errorf("subagent %q not found", agentID)
	}

	handle.mu.Lock()
	defer handle.mu.Unlock()

	if handle.Status != SubAgentRunning {
		return fmt.Errorf("subagent %q is not running (status: %s)", agentID, handle.Status)
	}

	// Inject a steering message into the child's queue.
	handle.Session.Steer(message)
	return nil
}

// Wait blocks until a subagent completes and returns its result.
func (m *SubAgentManager) Wait(agentID string) (*SubAgentResult, error) {
	m.mu.Lock()
	handle, ok := m.agents[agentID]
	m.mu.Unlock()

	if !ok {
		return nil, fmt.Errorf("subagent %q not found", agentID)
	}

	// Wait for completion.
	<-handle.doneCh

	handle.mu.Lock()
	defer handle.mu.Unlock()

	if handle.Result == nil {
		return &SubAgentResult{Output: "", Success: false, TurnsUsed: 0}, nil
	}
	return handle.Result, nil
}

// Close terminates a subagent.
func (m *SubAgentManager) Close(agentID string) (SubAgentStatus, error) {
	m.mu.Lock()
	handle, ok := m.agents[agentID]
	m.mu.Unlock()

	if !ok {
		return "", fmt.Errorf("subagent %q not found", agentID)
	}

	handle.mu.Lock()
	if handle.Status == SubAgentRunning {
		handle.Session.Abort()
		handle.Status = SubAgentFailed
		handle.Result = &SubAgentResult{
			Output:    "Terminated by parent",
			Success:   false,
			TurnsUsed: len(handle.Session.History),
		}
		select {
		case <-handle.doneCh:
		default:
			close(handle.doneCh)
		}
	}
	status := handle.Status
	handle.mu.Unlock()

	return status, nil
}

// CloseAll terminates all running subagents. This is used during graceful
// shutdown to ensure no child sessions are left in a running state.
func (m *SubAgentManager) CloseAll() {
	m.mu.Lock()
	ids := make([]string, 0, len(m.agents))
	for id, handle := range m.agents {
		handle.mu.Lock()
		if handle.Status == SubAgentRunning {
			ids = append(ids, id)
		}
		handle.mu.Unlock()
	}
	m.mu.Unlock()

	for _, id := range ids {
		_, _ = m.Close(id)
	}
}

// runSubAgent executes the subagent's agentic loop.
func (m *SubAgentManager) runSubAgent(ctx context.Context, handle *SubAgentHandle, task string) {
	defer func() {
		select {
		case <-handle.doneCh:
		default:
			close(handle.doneCh)
		}
	}()

	err := handle.Session.Submit(ctx, task)

	handle.mu.Lock()
	defer handle.mu.Unlock()

	turnsUsed := len(handle.Session.History)

	if err != nil {
		handle.Status = SubAgentFailed
		handle.Result = &SubAgentResult{
			Output:    fmt.Sprintf("Error: %v", err),
			Success:   false,
			TurnsUsed: turnsUsed,
		}
		return
	}

	// Extract final assistant output.
	output := extractFinalOutput(handle.Session)
	handle.Status = SubAgentCompleted
	handle.Result = &SubAgentResult{
		Output:    output,
		Success:   true,
		TurnsUsed: turnsUsed,
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// newChildSession creates a Session for use as a subagent. It shares the
// parent's profile and client but has its own history and event emitter.
func newChildSession(id string, prof profile.ProviderProfile, execEnv env.ExecutionEnvironment, client LLMClient, cfg Config, depth int) *Session {
	s := &Session{
		ID:           id,
		Profile:      prof,
		ExecutionEnv: execEnv,
		History:      nil,
		EventEmitter: events.NewEmitter(),
		Config:       cfg,
		State:        StateIdle,
		LLMClient:    client,
		abortCh:      make(chan struct{}),
		depth:        depth,
	}

	// Generate cached context.
	s.envContext = buildEnvironmentContext(execEnv, prof)
	s.projectDocs = discoverProjectDocs(execEnv, prof.InstructionFileNames())

	return s
}

// extractFinalOutput pulls the last assistant message text from a session's
// history.
func extractFinalOutput(s *Session) string {
	for i := len(s.History) - 1; i >= 0; i-- {
		if at, ok := s.History[i].(*AssistantTurn); ok && at.Content != "" {
			return at.Content
		}
	}
	return ""
}

// cloneProfileWithModel creates a new profile with the given model override.
// If the profile is a *BaseProfile (or wraps one via the ModelSettable
// interface), the model is set directly. Otherwise, the parent profile is
// returned unchanged since we cannot modify the model on an opaque interface.
func cloneProfileWithModel(parent profile.ProviderProfile, model string) profile.ProviderProfile {
	type modelSettable interface {
		profile.ProviderProfile
		SetModel(string)
	}

	// If the profile supports SetModel, create a shallow clone and set it.
	if bp, ok := parent.(*profile.BaseProfile); ok {
		// Construct a new profile from the same provider factory so we
		// get an independent copy.
		var cloned *profile.BaseProfile
		switch bp.ProviderName() {
		case "anthropic":
			cloned = profile.NewAnthropicProfile(model)
		case "openai":
			cloned = profile.NewOpenAIProfile(model)
		case "gemini":
			cloned = profile.NewGeminiProfile(model)
		default:
			// Unknown provider; use the parent as-is.
			return parent
		}
		return cloned
	}

	// If it implements the modelSettable interface directly, use it.
	if ms, ok := parent.(modelSettable); ok {
		ms.SetModel(model)
		return ms
	}

	return parent
}

// scopeEnvironment creates a scoped execution environment for the given
// working directory. If the environment supports scoping, it uses that;
// otherwise it returns the parent environment unchanged.
func scopeEnvironment(parent env.ExecutionEnvironment, workDir string) (env.ExecutionEnvironment, error) {
	// Check if the environment supports scoping via the ScopedEnvironment
	// interface.
	type scopable interface {
		Scope(workDir string) (env.ExecutionEnvironment, error)
	}
	if s, ok := parent.(scopable); ok {
		return s.Scope(workDir)
	}
	// Fall back: return parent as-is. The subagent will use the same
	// working directory but can use working_dir args on shell calls.
	return parent, nil
}
