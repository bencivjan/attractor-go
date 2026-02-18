// Package pipelines provides embedded pipeline definitions for the Attractor
// engine. The default pipeline (Plan-Build-Verify) is embedded at compile
// time and available as a DOT source string.
package pipelines

import _ "embed"

//go:embed plan_build_verify.dot
var planBuildVerifyDOT string

// DefaultName is the identifier for the built-in default pipeline.
const DefaultName = "plan_build_verify"

// Default returns the DOT source for the default Plan-Build-Verify pipeline.
//
// The pipeline stages are:
//  1. High-Level Plan     (Claude Opus)  — architecture and strategy
//  2. Sprint Breakdown    (Claude Opus)  — decompose into sprint-sized work
//  3. Implement           (Codex)        — write production code
//  4. QA Verification     (Claude Opus)  — verify implementation matches plans
func Default() string {
	return planBuildVerifyDOT
}

// catalog holds named pipelines for lookup.
var catalog = map[string]string{
	DefaultName: planBuildVerifyDOT,
}

// Get returns the DOT source for a named pipeline, or empty string if not found.
func Get(name string) string {
	return catalog[name]
}

// Register adds a named pipeline to the catalog. Existing entries with the
// same name are overwritten.
func Register(name, dotSource string) {
	catalog[name] = dotSource
}

// Names returns the names of all registered pipelines.
func Names() []string {
	names := make([]string, 0, len(catalog))
	for name := range catalog {
		names = append(names, name)
	}
	return names
}
