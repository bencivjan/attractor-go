// Package pipelines provides embedded pipeline definitions for the Attractor
// engine. Three pipelines are embedded at compile time:
//   - developer (Plan-Build-Verify) — the default development pipeline
//   - evaluator — reviews submissions against a project vision
//   - factory — combined end-to-end pipeline wiring developer and evaluator
package pipelines

import _ "embed"

//go:embed developer.dot
var developerDOT string

//go:embed evaluator.dot
var evaluatorDOT string

//go:embed factory.dot
var factoryDOT string

// Pipeline name constants.
const (
	// DefaultName is the identifier for the built-in default pipeline.
	DefaultName = "plan_build_verify"

	// EvaluatorName is the identifier for the evaluator pipeline.
	EvaluatorName = "evaluator"

	// FactoryName is the identifier for the combined factory pipeline.
	FactoryName = "factory"
)

// Default returns the DOT source for the default Plan-Build-Verify pipeline.
//
// The pipeline stages are:
//  1. High-Level Plan     (Claude Opus)  — architecture and strategy
//  2. Sprint Breakdown    (Claude Opus)  — decompose into sprint-sized work
//  3. Implement           (Codex)        — write production code
//  4. QA Verification     (Claude Opus)  — verify implementation matches plans
func Default() string {
	return developerDOT
}

// Evaluator returns the DOT source for the Evaluator pipeline.
func Evaluator() string {
	return evaluatorDOT
}

// Factory returns the DOT source for the combined Factory pipeline.
func Factory() string {
	return factoryDOT
}

// catalog holds named pipelines for lookup.
var catalog = map[string]string{
	DefaultName:   developerDOT,
	EvaluatorName: evaluatorDOT,
	FactoryName:   factoryDOT,
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
