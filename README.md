# attractor-go

A production-grade Go implementation of the [Attractor specification](https://github.com/strongdm/attractor) — a DOT-based pipeline runner for orchestrating multi-stage AI workflows.

Attractor lets you define complex LLM pipelines as directed graphs in Graphviz DOT syntax. Nodes are tasks (LLM calls, human gates, parallel fan-outs), edges are transitions with conditions and weights, and the engine handles traversal, retries, checkpointing, and human interaction. This implementation covers the full spec and pairs it with a unified LLM client and coding agent runtime.

## Why Go

Go is the right language for a pipeline orchestrator. The properties that matter most — deterministic concurrency, fast startup, static binaries, and a type system that catches structural errors at compile time — are exactly what Go provides without ceremony.

### Concurrency that maps to the problem

Attractor pipelines fan out to parallel branches, poll child pipelines, and stream events to frontends. Go's goroutines and channels map directly to these patterns. The `parallel` handler runs branches concurrently with semaphore-bounded parallelism, each branch getting an isolated context clone. No thread pool configuration, no executor framework — just `go func()` with a `sync.WaitGroup` and a buffered channel for results. The `sync.RWMutex` on Context gives concurrent read access during parallel execution while serializing writes, which is exactly the access pattern pipeline state requires.

### Type safety where it counts

The graph model (`attractor/graph`) uses typed accessors — `node.GoalGate()` returns `bool`, `node.MaxRetries()` returns `int`, `node.Timeout()` returns `time.Duration`. Attribute parsing happens once at parse time, not scattered across handlers at runtime. The handler registry maps type strings to `Handler` interface implementations with compile-time guarantees. If a handler doesn't implement `Execute(node, context, graph, logsRoot) Outcome`, it doesn't compile.

### Zero-overhead extensibility

New handlers, lint rules, transforms, and LLM providers are all interfaces with single-method contracts. Adding a custom handler is:

```go
registry.Register("my_type", myHandler{})
```

No reflection, no annotation processing, no dependency injection container. The handler registry is a `map[string]Handler` with a `Resolve` method. Custom lint rules implement `LintRule` with an `Apply(graph) []Diagnostic` method. Transforms implement `Apply(graph) Graph`. The patterns are obvious and the compiler enforces them.

### Production behavior by default

Checkpoints use atomic write-then-rename to prevent corruption on crash. The artifact store spills to disk at 100KB to prevent memory bloat on long-running pipelines. Retry policies use exponential backoff with jitter to avoid thundering herd. Context cloning uses JSON marshal/unmarshal for deep copy — not the fastest approach, but correct by construction and impossible to accidentally share mutable state between parallel branches.

## Architecture

```
attractor-go/
├── attractor/           Pipeline engine
│   ├── parser/          Three-phase DOT parser (strip → tokenize → parse)
│   ├── graph/           Typed graph model with O(1) node lookup
│   ├── engine/          Core execution loop and checkpoint-aware runner
│   ├── handler/         Pluggable handler registry (codergen, human, parallel, tool, ...)
│   ├── state/           Thread-safe context, checkpointing, artifact store
│   ├── condition/       Condition expression evaluator (=, !=, &&)
│   ├── validation/      13 lint rules across three severity levels
│   ├── stylesheet/      CSS-like model configuration with specificity cascade
│   ├── transform/       AST transforms (variable expansion, stylesheet application)
│   └── pipelines/       Embedded pipeline definitions (.dot files)
├── unifiedllm/          Provider-agnostic LLM client
│   ├── provider/        Adapters for Anthropic, OpenAI, Gemini
│   ├── client/          Multi-provider client with middleware chain
│   ├── types/           Unified message/content/tool types
│   ├── retry/           Retry with backoff for LLM calls
│   └── sse/             Server-sent events parser for streaming
├── codingagent/         Coding agent runtime
│   ├── session/         Agent session lifecycle management
│   ├── tools/           Tool registry for agent capabilities
│   ├── profile/         Agent profile configuration
│   ├── events/          Typed event stream
│   ├── env/             Environment detection
│   └── truncation/      Context window management
└── go.mod
```

### How a pipeline runs

1. **Parse** — The DOT parser strips comments, tokenizes (handling quoted strings, qualified identifiers, hyphenated names), and builds a graph via recursive descent.
2. **Transform** — Variable expansion replaces `$goal` in prompts. The stylesheet transform applies CSS-like model rules with specificity cascade (universal < class < ID).
3. **Validate** — 13 lint rules check structural correctness (start/exit nodes, reachability via BFS, edge targets), semantic validity (condition syntax, retry targets), and conventions (type names, fidelity modes). Errors block execution; warnings don't.
4. **Execute** — The engine traverses from the start node. Each node's handler is resolved from the registry, executed with retry policy, and the outcome drives edge selection through a deterministic 5-step algorithm: condition match → preferred label → suggested IDs → weight → lexical tiebreak.
5. **Checkpoint** — After each node, state is atomically saved. Resume restores context, completed nodes, and retry counters. Goal gates are enforced before exit.

### Edge selection

The routing algorithm is the heart of the engine. After a node completes, outgoing edges are evaluated in strict priority order:

1. **Condition match** — Edges with `condition` expressions that evaluate to true against the current context and outcome
2. **Preferred label** — If the outcome suggests a label, match it (normalized: lowercase, trimmed, accelerator keys stripped)
3. **Suggested next IDs** — Explicit node IDs from the outcome
4. **Highest weight** — Among unconditional edges, highest `weight` wins
5. **Lexical tiebreak** — Alphabetical by target node ID

This gives pipeline authors precise control over routing while keeping behavior deterministic and inspectable.

## Unified LLM Client

The `unifiedllm` package provides a single interface across LLM providers. The design prioritizes correctness over convenience — every provider adapter implements the same `ProviderAdapter` interface, and the client resolves providers through explicit configuration, model catalog lookup, or environment auto-detection.

Key capabilities:
- **Streaming with tool loops** — `Stream()` pauses on tool calls, executes them, and resumes. `Generate()` runs blocking tool loops up to `MaxToolRounds`.
- **Structured output** — `GenerateObject()` validates responses against JSON schemas with fallback markdown fence extraction.
- **Middleware** — Onion-pattern request/response interceptors for logging, metrics, or request modification.
- **Unified types** — Messages, content blocks (text, image, audio, document, tool calls, thinking), and usage tracking work identically across providers.

## Included Pipelines

### Plan-Build-Verify (`developer.dot`)

A four-stage software development pipeline: high-level planning (Claude Opus) → sprint breakdown (Claude Opus) → implementation (Codex) → QA verification (Claude Opus). QA failure loops back to implementation with feedback. Goal gates ensure quality before exit.

### Evaluator (`evaluator.dot`)

A four-stage evaluation pipeline for reviewing submissions against a project vision: orchestration (Claude Opus) → tool building (Codex) → QA testing (Claude Opus) → visionary judgment (Claude Opus). The visionary maintains the high-level goal and returns structured, actionable feedback to the submitter.

## Spec Compliance

This implementation covers the full [Attractor specification](https://github.com/strongdm/attractor), including:

- Complete DOT subset parser with chained edges, subgraphs, default blocks, and multi-line attributes
- All 8 built-in handler types (start, exit, codergen, wait.human, conditional, parallel, fan-in, tool, manager loop)
- 5-step edge selection algorithm with condition evaluation
- Goal gate enforcement with retry target fallback chain
- Exponential backoff with jitter retry policies
- Thread-safe context with RWMutex, deep cloning for parallel isolation
- Atomic checkpointing with crash recovery and resume
- CSS-like model stylesheet with specificity cascade
- Composable AST transforms (variable expansion, stylesheet application)
- 13 built-in lint rules with extensible rule interface
- Human-in-the-loop via Interviewer abstraction (auto-approve, console, callback, queue, recording)
- Artifact store with automatic disk spillover
- Context fidelity modes (full, truncate, compact, summary:low/medium/high)
- Event streaming for pipeline lifecycle observability

## License

See [LICENSE](LICENSE) for details.
