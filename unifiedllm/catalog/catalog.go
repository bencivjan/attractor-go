// Package catalog provides a registry of known LLM models and their
// capabilities. The catalog allows the client to resolve model names to
// providers and to look up model metadata without hard-coding knowledge
// of every model into the routing layer.
package catalog

import (
	"strings"
	"sync"

	"github.com/strongdm/attractor-go/unifiedllm/types"
)

// registry holds the global model catalog, keyed by model ID.
var (
	registry   = make(map[string]*types.ModelInfo)
	registryMu sync.RWMutex
)

// Models is the built-in catalog of known models across all supported providers.
// Each entry describes a model's capabilities, context window, and cost
// information where available. These are registered into the global registry
// at package init time.
var Models = []types.ModelInfo{
	// -------------------------------------------------------------------------
	// Anthropic
	// -------------------------------------------------------------------------
	{
		ID:                "claude-opus-4-6",
		Provider:          "anthropic",
		DisplayName:       "Claude Opus 4.6",
		ContextWindow:     200000,
		MaxOutput:         32000,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},
	{
		ID:                "claude-sonnet-4-5",
		Provider:          "anthropic",
		DisplayName:       "Claude Sonnet 4.5",
		ContextWindow:     200000,
		MaxOutput:         16000,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},

	// -------------------------------------------------------------------------
	// OpenAI
	// -------------------------------------------------------------------------
	{
		ID:                "gpt-5.2",
		Provider:          "openai",
		DisplayName:       "GPT-5.2",
		ContextWindow:     1047576,
		MaxOutput:         32768,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},
	{
		ID:                "gpt-5.2-mini",
		Provider:          "openai",
		DisplayName:       "GPT-5.2 Mini",
		ContextWindow:     1047576,
		MaxOutput:         16384,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},
	{
		ID:                "gpt-5.2-codex",
		Provider:          "openai",
		DisplayName:       "GPT-5.2 Codex",
		ContextWindow:     1047576,
		MaxOutput:         32768,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},
	{
		ID:                "gpt-5.3-codex",
		Provider:          "openai",
		DisplayName:       "GPT-5.3 Codex",
		ContextWindow:     1047576,
		MaxOutput:         32768,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},

	// -------------------------------------------------------------------------
	// Gemini
	// -------------------------------------------------------------------------
	{
		ID:                "gemini-3-pro-preview",
		Provider:          "gemini",
		DisplayName:       "Gemini 3 Pro (Preview)",
		ContextWindow:     1048576,
		MaxOutput:         65536,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},
	{
		ID:                "gemini-3-flash-preview",
		Provider:          "gemini",
		DisplayName:       "Gemini 3 Flash (Preview)",
		ContextWindow:     1048576,
		MaxOutput:         65536,
		SupportsTools:     true,
		SupportsVision:    true,
		SupportsReasoning: true,
	},
}

func init() {
	for _, m := range Models {
		Register(m)
	}
}

// Register adds a model to the global catalog. If a model with the same ID
// already exists, it is overwritten.
func Register(info types.ModelInfo) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registered := info // copy
	registry[info.ID] = &registered
	for _, alias := range info.Aliases {
		registry[alias] = &registered
	}
}

// GetModelInfo looks up a model by ID or alias. Returns nil if not found.
func GetModelInfo(model string) *types.ModelInfo {
	registryMu.RLock()
	defer registryMu.RUnlock()

	if info, ok := registry[model]; ok {
		return info
	}

	// Try case-insensitive lookup as a fallback.
	lower := strings.ToLower(model)
	for key, info := range registry {
		if strings.ToLower(key) == lower {
			return info
		}
	}

	return nil
}

// ListModels returns all models for the specified provider. If provider is
// empty, all models in the catalog are returned. The returned slice is a
// copy and safe to modify.
func ListModels(provider string) []types.ModelInfo {
	registryMu.RLock()
	defer registryMu.RUnlock()

	// Use the Models slice to preserve ordering rather than iterating
	// the unordered map.
	var result []types.ModelInfo
	for _, m := range Models {
		if provider == "" || m.Provider == provider {
			result = append(result, m)
		}
	}

	// Also include any models that were registered externally (not in the
	// built-in Models slice).
	builtIn := make(map[string]bool, len(Models))
	for _, m := range Models {
		builtIn[m.ID] = true
	}
	for id, info := range registry {
		if builtIn[id] {
			continue
		}
		// Only include primary IDs, not aliases.
		if id != info.ID {
			continue
		}
		if provider == "" || info.Provider == provider {
			result = append(result, *info)
		}
	}

	return result
}

// GetLatestModel returns the first model matching the given provider and
// capability. Supported capability values are "tools", "vision", "reasoning",
// and "" (any). Returns nil if no matching model is found.
//
// The catalog is ordered with the most capable / newest models first within
// each provider section, so the first match is considered the "latest".
func GetLatestModel(provider string, capability string) *types.ModelInfo {
	registryMu.RLock()
	defer registryMu.RUnlock()

	capability = strings.ToLower(capability)

	for i := range Models {
		m := &Models[i]
		if m.Provider != provider {
			continue
		}

		switch capability {
		case "tools":
			if m.SupportsTools {
				return m
			}
		case "vision":
			if m.SupportsVision {
				return m
			}
		case "reasoning":
			if m.SupportsReasoning {
				return m
			}
		case "":
			return m
		}
	}

	return nil
}
