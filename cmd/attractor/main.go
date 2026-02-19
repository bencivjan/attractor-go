// Command attractor runs AI pipelines defined as Graphviz DOT graphs.
//
// Usage:
//
//	attractor <command> [flags]
//
// Commands: run, validate, list, factory
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/strongdm/attractor-go/attractor/engine"
	"github.com/strongdm/attractor-go/attractor/factory"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/parser"
	"github.com/strongdm/attractor-go/attractor/pipelines"
	"github.com/strongdm/attractor-go/attractor/state"
	"github.com/strongdm/attractor-go/attractor/transform"
	"github.com/strongdm/attractor-go/attractor/validation"
)

const usage = `attractor — run AI pipelines defined as Graphviz DOT graphs

Usage:
  attractor <command> [flags]

Commands:
  run        Execute a pipeline from a .dot file or built-in name
  validate   Lint a .dot file and report errors/warnings
  list       List built-in pipeline names
  factory    Run the developer→evaluator factory loop

Run "attractor <command> --help" for details on each command.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "run":
		cmdRun(args)
	case "validate":
		cmdValidate(args)
	case "list":
		cmdList(args)
	case "factory":
		cmdFactory(args)
	case "--help", "-h", "help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "attractor: unknown command %q\n\n%s", cmd, usage)
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

func cmdRun(args []string) {
	fs := flag.NewFlagSet("attractor run", flag.ExitOnError)
	file := fs.String("file", "", "Path to a .dot pipeline file")
	pipeline := fs.String("pipeline", "", "Name of a built-in pipeline (see 'attractor list')")
	logs := fs.String("logs", "attractor-logs", "Directory for run logs, checkpoints, and artifacts")
	maxSteps := fs.Int("max-steps", 0, "Maximum node executions per run (0 = default 1000)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `attractor run — execute a pipeline

Usage:
  attractor run -file <path.dot> [-logs <dir>] [-max-steps <n>]
  attractor run -pipeline <name>  [-logs <dir>] [-max-steps <n>]

Provide either -file (a .dot file on disk) or -pipeline (a built-in name).
The pipeline runs from start to exit, executing handlers at each node.

Flags:
`)
		fs.PrintDefaults()
	}
	fs.Parse(args)

	if *file == "" && *pipeline == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "\nerror: provide -file or -pipeline")
		os.Exit(1)
	}
	if *file != "" && *pipeline != "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "\nerror: provide -file or -pipeline, not both")
		os.Exit(1)
	}

	runner := &engine.Runner{
		Registry:   handler.DefaultRegistry(handler.NoopHandler{}),
		Transforms: transform.DefaultTransforms(),
		MaxSteps:   *maxSteps,
	}

	var (
		outcome *state.Outcome
		err     error
	)

	if *file != "" {
		dot, readErr := os.ReadFile(*file)
		if readErr != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", readErr)
			os.Exit(1)
		}
		outcome, err = runner.RunDOT(context.Background(), string(dot), *logs)
	} else {
		outcome, err = runner.RunPipeline(context.Background(), *pipeline, *logs)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Status: %s\n", outcome.Status)
	if outcome.Notes != "" {
		fmt.Printf("Notes:  %s\n", outcome.Notes)
	}
	if outcome.FailureReason != "" {
		fmt.Printf("Reason: %s\n", outcome.FailureReason)
	}
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

func cmdValidate(args []string) {
	fs := flag.NewFlagSet("attractor validate", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `attractor validate — lint a .dot pipeline file

Usage:
  attractor validate <file.dot> [<file2.dot> ...]

Parses each file, applies transforms, and runs the 13 built-in lint rules.
Exits with code 1 if any ERROR-severity diagnostics are found.

Severity levels:
  ERROR    Pipeline will not execute correctly
  WARNING  Potential issue, pipeline may still run
  INFO     Suggestion for improvement
`)
	}
	fs.Parse(args)

	files := fs.Args()
	if len(files) == 0 {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "\nerror: provide at least one .dot file")
		os.Exit(1)
	}

	hasErrors := false
	for _, path := range files {
		dot, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			hasErrors = true
			continue
		}

		g, err := parser.Parse(string(dot))
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: parse error: %v\n", path, err)
			hasErrors = true
			continue
		}

		g = transform.ApplyAll(g, transform.DefaultTransforms()...)
		diags := validation.Validate(g)

		if len(diags) == 0 {
			fmt.Printf("%s: ok\n", path)
			continue
		}

		for _, d := range diags {
			fmt.Printf("%s: [%s] %s: %s\n", path, d.Severity, d.Rule, d.Message)
		}

		if validation.HasErrors(diags) {
			hasErrors = true
		}
	}

	if hasErrors {
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

func cmdList(args []string) {
	fs := flag.NewFlagSet("attractor list", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `attractor list — list built-in pipeline names

Usage:
  attractor list

Prints the names of all registered built-in pipelines. Use a name with
"attractor run -pipeline <name>" to execute one.
`)
	}
	fs.Parse(args)

	names := pipelines.Names()
	if len(names) == 0 {
		fmt.Println("(no built-in pipelines registered)")
		return
	}
	for _, name := range names {
		fmt.Println(name)
	}
}

// ---------------------------------------------------------------------------
// factory
// ---------------------------------------------------------------------------

func cmdFactory(args []string) {
	fs := flag.NewFlagSet("attractor factory", flag.ExitOnError)
	goal := fs.String("goal", "", "Project goal (overrides graph-level goal attribute)")
	logs := fs.String("logs", "attractor-logs", "Directory for run logs")
	maxSteps := fs.Int("max-steps", 0, "Maximum node executions per pipeline run (0 = default 1000)")
	maxReject := fs.Int("max-rejections", 3, "Maximum evaluator rejections before giving up")
	devDOT := fs.String("developer", "", "Path to custom developer .dot (default: built-in)")
	evalDOT := fs.String("evaluator", "", "Path to custom evaluator .dot (default: built-in)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `attractor factory — run the developer→evaluator factory loop

Usage:
  attractor factory -goal "build a feature" [-logs <dir>] [flags]

Runs the developer pipeline (plan→implement→QA) then hands the result to the
evaluator pipeline (orchestrate→build tools→QA→visionary) for review. If the
evaluator rejects, the developer re-runs with feedback. Repeats up to
-max-rejections times.

The two pipelines run with complete context isolation — only goal and
last_response cross from developer to evaluator; only evaluator_feedback
crosses back.

Flags:
`)
		fs.PrintDefaults()
	}
	fs.Parse(args)

	f := &factory.FactoryRunner{
		Registry:      handler.DefaultRegistry(handler.NoopHandler{}),
		Transforms:    transform.DefaultTransforms(),
		MaxSteps:      *maxSteps,
		MaxRejections: *maxReject,
	}

	if *devDOT != "" {
		dot, err := os.ReadFile(*devDOT)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading developer dot: %v\n", err)
			os.Exit(1)
		}
		f.DeveloperDOT = string(dot)
	}
	if *evalDOT != "" {
		dot, err := os.ReadFile(*evalDOT)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading evaluator dot: %v\n", err)
			os.Exit(1)
		}
		f.EvaluatorDOT = string(dot)
	}

	var (
		outcome *factory.FactoryOutcome
		err     error
	)

	if *goal != "" {
		outcome, err = f.RunWithGoal(context.Background(), *goal, *logs)
	} else {
		outcome, err = f.Run(context.Background(), *logs)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Status:     %s\n", outcome.Status)
	fmt.Printf("Iterations: %d\n", outcome.Iterations)
	if outcome.Notes != "" {
		fmt.Printf("Notes:      %s\n", outcome.Notes)
	}
	if outcome.FailureReason != "" {
		fmt.Printf("Reason:     %s\n", outcome.FailureReason)
	}

	// Print abbreviated dev/eval status.
	parts := []string{}
	if outcome.DeveloperOutcome != nil {
		parts = append(parts, fmt.Sprintf("developer=%s", outcome.DeveloperOutcome.Status))
	}
	if outcome.EvaluatorOutcome != nil {
		parts = append(parts, fmt.Sprintf("evaluator=%s", outcome.EvaluatorOutcome.Status))
	}
	if len(parts) > 0 {
		fmt.Printf("Pipelines:  %s\n", strings.Join(parts, ", "))
	}
}
