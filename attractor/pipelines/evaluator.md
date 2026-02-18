# Evaluator Pipeline

A four-stage pipeline that evaluates submissions against a project vision using role-separated AI agents. Rather than having a single LLM judge quality, the evaluator decomposes the problem: one agent plans the evaluation, another builds the testing tools, a third executes the tests, and a fourth judges the results against the original vision.

![Evaluator Pipeline](evaluator.png)

## Why this structure

Most AI evaluation approaches ask a single model to read code and decide if it's good. This fails for two reasons: the model has no way to actually run the code, and it conflates understanding the goal with measuring the result.

The evaluator pipeline solves both problems by separating concerns into four roles that mirror how human teams evaluate software — a lead who scopes the review, a developer who builds test infrastructure, a QA engineer who runs the tests, and a product owner who decides if the result matches the vision.

## Stages

### 1. Orchestrator (Claude Opus)

Receives the submission and the project vision. Produces a **delegation plan** with two sections:

- **Builder Tasks** — specific tools, test harnesses, scripts, and fixtures the builder must create. Each task has a clear purpose tied to an evaluation dimension (correctness, completeness, quality, vision alignment).
- **QA Checklist** — specific checks QA must perform using those tools, each tied to a success criterion from the vision.

The orchestrator is a goal gate (`goal_gate=true`), meaning the pipeline cannot exit successfully unless this stage succeeds. It has 2 retries on failure.

**Model:** `claude-opus-4-6` (Anthropic)

### 2. Builder (Codex 5.3)

A toolsmith that builds everything the QA stage needs. For each tool in the delegation plan, the builder produces:

- Runnable code that QA can invoke directly
- Usage instructions (CLI flags, env vars, expected I/O)
- Self-contained setup and teardown
- Structured output (JSON or parseable markers) for programmatic checking

The builder does not evaluate the submission — it only builds instruments. This is a goal gate with 3 retries, and its `retry_target` points back to itself so failures retry the build rather than restarting the whole pipeline.

**Model:** `gpt-5.3-codex` (OpenAI)

### 3. QA (Claude Opus)

Executes the builder's tools against the submission, following the orchestrator's checklist item by item. For each check:

1. Run the relevant tool or harness
2. Record raw output
3. Determine PASS or FAIL with justification
4. Note unexpected behavior even on passing checks

Produces a structured **evaluation report** with a summary (pass/fail counts, health assessment), per-check results (checklist item, tool used, outcome, evidence), and observations not covered by the checklist.

QA is deliberately objective — it reports what happened, not what it thinks should happen. Interpretation is the visionary's job.

**Model:** `claude-opus-4-6` (Anthropic)

### 4. Visionary (Claude Opus)

The keeper of the project's high-level goal. Reads the QA report and judges the submission against the vision on four dimensions:

1. **Vision alignment** — does the submission move the project toward its goal?
2. **Completeness** — are there gaps between delivery and vision?
3. **Quality bar** — does the work meet the standard the vision implies?
4. **Regression** — does the submission break anything that previously worked?

This is a conditional gate (diamond shape) with three exit conditions:

| Outcome | Condition | What happens |
|---------|-----------|--------------|
| **Approved** | `outcome=success` | Submission satisfies the vision. Pipeline exits with success summary. |
| **Approved (partial)** | `outcome=partial_success` | Submission is acceptable with caveats noted. |
| **Rejected** | `outcome=fail` | Submission falls short. Structured feedback is returned to the submitter. |

On rejection, the visionary produces actionable feedback: what's missing (referencing QA evidence), why it matters to the vision, and concrete next steps. This feedback exits the pipeline and goes directly to the submitter.

**Model:** `claude-opus-4-6` (Anthropic)

## Configuration

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `default_max_retry` | 3 | Global retry ceiling for any stage |
| `fallback_retry_target` | `builder` | If a stage fails with no retry target, restart from the builder |
| `reasoning_effort` | high (via stylesheet) | All stages use high reasoning effort |

## Data flow

```
Submission ──► Orchestrator ──► Builder ──► QA ──► Visionary ──► Result
                  │                │          │         │
                  │                │          │         ├─► Approved
                  ▼                ▼          ▼         ├─► Approved (partial)
            delegation plan   tools &    evaluation    └─► Rejected + feedback
            (builder tasks    harnesses   report
             + QA checklist)
```

Each stage's output becomes context for the next. The orchestrator's delegation plan tells the builder what to build and the QA what to check. The builder's tools are what QA exercises. The QA report is what the visionary judges. No stage needs to understand the full pipeline — each has a focused role with clear inputs and outputs.

## Goal gates

All four working stages (orchestrator, builder, QA, visionary) are goal gates. The pipeline cannot exit successfully unless every stage has a success or partial success outcome. If any gate is unsatisfied when the pipeline reaches the exit node, the engine routes to the `retry_target` (or `fallback_retry_target`) instead of exiting.

## When to use this pipeline

- **Pull request review** — evaluate code changes against a feature spec or architectural vision
- **Agent output validation** — verify that an AI coding agent's output matches the original task description
- **Continuous evaluation** — run as part of a CI/CD pipeline to gate deployments against quality criteria
- **Spec compliance** — check an implementation against a specification document
