// Package server provides an HTTP API for the Attractor pipeline engine.
//
// It exposes endpoints for submitting DOT pipelines, monitoring execution
// progress via SSE, managing human interaction gates, and inspecting
// pipeline state (context, checkpoint, graph). The server integrates with
// engine.Runner to execute pipelines in background goroutines.
//
// Spec reference: Section 9.5 (HTTP Server Mode).
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/strongdm/attractor-go/attractor/engine"
	"github.com/strongdm/attractor-go/attractor/handler"
	"github.com/strongdm/attractor-go/attractor/interviewer"
	"github.com/strongdm/attractor-go/attractor/state"
	"github.com/strongdm/attractor-go/attractor/transform"
)

// ---------------------------------------------------------------------------
// Pipeline status
// ---------------------------------------------------------------------------

// PipelineStatus represents the lifecycle state of a pipeline run.
type PipelineStatus string

const (
	StatusPending   PipelineStatus = "pending"
	StatusRunning   PipelineStatus = "running"
	StatusCompleted PipelineStatus = "completed"
	StatusFailed    PipelineStatus = "failed"
	StatusCancelled PipelineStatus = "cancelled"
)

// ---------------------------------------------------------------------------
// PendingQuestion
// ---------------------------------------------------------------------------

// PendingQuestion represents a human interaction question that is waiting
// for an answer via the HTTP API. It wraps the interviewer.Question with
// a unique ID and a channel for delivering the answer back to the blocked
// goroutine.
type PendingQuestion struct {
	ID       string               `json:"id"`
	Question interviewer.Question `json:"question"`
	answerCh chan interviewer.Answer
}

// ---------------------------------------------------------------------------
// PipelineRun
// ---------------------------------------------------------------------------

// PipelineRun tracks all state for a single pipeline execution.
type PipelineRun struct {
	ID         string         `json:"id"`
	Status     PipelineStatus `json:"status"`
	CreatedAt  time.Time      `json:"created_at"`
	FinishedAt *time.Time     `json:"finished_at,omitempty"`
	DOTSource  string         `json:"dot_source,omitempty"`
	Error      string         `json:"error,omitempty"`

	// outcome holds the final engine outcome after completion.
	outcome *state.Outcome

	// cancel cancels the pipeline's context.Context.
	cancel context.CancelFunc

	// pipelineCtx is the state.Context used during execution.
	pipelineCtx *state.Context

	// checkpoint is the most recently saved checkpoint.
	checkpoint *state.Checkpoint

	// events is a buffered list of all events emitted during execution.
	events []engine.Event

	// eventSubscribers holds channels for SSE consumers.
	eventSubscribers []chan engine.Event

	// questions holds pending human interaction questions keyed by question ID.
	questions map[string]*PendingQuestion

	// mu serializes access to mutable fields.
	mu sync.RWMutex
}

// addEvent appends an event and fans it out to SSE subscribers.
func (pr *PipelineRun) addEvent(ev engine.Event) {
	pr.mu.Lock()
	pr.events = append(pr.events, ev)
	subs := make([]chan engine.Event, len(pr.eventSubscribers))
	copy(subs, pr.eventSubscribers)
	pr.mu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- ev:
		default:
			// Subscriber is slow; drop the event rather than blocking
			// the engine goroutine.
		}
	}
}

// subscribe creates a new SSE event channel for a consumer. The caller
// receives all future events. The returned channel is buffered to absorb
// transient slowness.
func (pr *PipelineRun) subscribe() chan engine.Event {
	ch := make(chan engine.Event, 64)
	pr.mu.Lock()
	pr.eventSubscribers = append(pr.eventSubscribers, ch)
	pr.mu.Unlock()
	return ch
}

// unsubscribe removes the channel from the subscriber list and closes it.
func (pr *PipelineRun) unsubscribe(ch chan engine.Event) {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	for i, sub := range pr.eventSubscribers {
		if sub == ch {
			pr.eventSubscribers = append(pr.eventSubscribers[:i], pr.eventSubscribers[i+1:]...)
			break
		}
	}
	close(ch)
}

// closeAllSubscribers shuts down all SSE channels. Called when the
// pipeline terminates.
func (pr *PipelineRun) closeAllSubscribers() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	for _, ch := range pr.eventSubscribers {
		close(ch)
	}
	pr.eventSubscribers = nil
}

// addQuestion registers a pending question and returns the PendingQuestion
// so the WebInterviewer can block on its answer channel.
func (pr *PipelineRun) addQuestion(q interviewer.Question) *PendingQuestion {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	id := generateID()
	pq := &PendingQuestion{
		ID:       id,
		Question: q,
		answerCh: make(chan interviewer.Answer, 1),
	}
	pr.questions[id] = pq
	return pq
}

// removeQuestion removes a pending question after it has been answered.
func (pr *PipelineRun) removeQuestion(id string) {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	delete(pr.questions, id)
}

// listQuestions returns a snapshot of all currently pending questions.
func (pr *PipelineRun) listQuestions() []*PendingQuestion {
	pr.mu.RLock()
	defer pr.mu.RUnlock()
	out := make([]*PendingQuestion, 0, len(pr.questions))
	for _, pq := range pr.questions {
		out = append(out, pq)
	}
	return out
}

// ---------------------------------------------------------------------------
// WebInterviewer
// ---------------------------------------------------------------------------

// WebInterviewer implements the interviewer.Interviewer interface by blocking
// on a channel until an answer arrives via the HTTP API. When the engine
// reaches a wait.human node, the handler calls Ask(), which registers a
// PendingQuestion on the PipelineRun and blocks. The HTTP endpoints
// GET /questions and POST /questions/{qid}/answer allow external clients
// to list pending questions and submit answers, respectively.
type WebInterviewer struct {
	run *PipelineRun
}

// Ask registers a pending question and blocks until an HTTP client submits
// an answer. If the pipeline context is cancelled while waiting, it returns
// a timeout answer.
func (w *WebInterviewer) Ask(q interviewer.Question) interviewer.Answer {
	pq := w.run.addQuestion(q)

	// Emit an event so SSE consumers know a question is pending.
	w.run.addEvent(engine.Event{
		Kind:      engine.EventInterviewStarted,
		Timestamp: time.Now(),
		Data: map[string]any{
			"question_id": pq.ID,
			"stage":       q.Stage,
			"text":        q.Text,
			"type":        q.Type.String(),
		},
	})

	answer := <-pq.answerCh
	w.run.removeQuestion(pq.ID)

	w.run.addEvent(engine.Event{
		Kind:      engine.EventInterviewCompleted,
		Timestamp: time.Now(),
		Data: map[string]any{
			"question_id": pq.ID,
			"stage":       q.Stage,
			"answer":      string(answer.Value),
		},
	})

	return answer
}

// AskMultiple presents multiple questions sequentially.
func (w *WebInterviewer) AskMultiple(questions []interviewer.Question) []interviewer.Answer {
	answers := make([]interviewer.Answer, len(questions))
	for i, q := range questions {
		answers[i] = w.Ask(q)
	}
	return answers
}

// Inform sends a non-interactive informational message as an SSE event.
func (w *WebInterviewer) Inform(message, stage string) {
	w.run.addEvent(engine.Event{
		Kind:      engine.EventStageStarted,
		Timestamp: time.Now(),
		Data: map[string]any{
			"type":    "inform",
			"message": message,
			"stage":   stage,
		},
	})
}

// Verify interface compliance at compile time.
var _ interviewer.Interviewer = (*WebInterviewer)(nil)

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

// Server is the HTTP API server for the Attractor pipeline engine.
type Server struct {
	// LogsRoot is the parent directory for pipeline run artifacts.
	LogsRoot string

	// Registry is the handler registry. If nil, DefaultRegistryFull is
	// used with the WebInterviewer for each pipeline run.
	Registry *handler.Registry

	// MaxSteps overrides the default maximum step count for pipeline runs.
	MaxSteps int

	// Backend is the optional CodergenBackend for LLM-powered stages.
	Backend handler.CodergenBackend

	// runs stores active and completed pipeline runs.
	runs sync.Map // map[string]*PipelineRun

	// httpServer is the underlying net/http server.
	httpServer *http.Server

	// logger provides structured logging.
	logger *log.Logger
}

// NewServer creates a Server with the given logs directory.
func NewServer(logsRoot string) *Server {
	return &Server{
		LogsRoot: logsRoot,
		logger:   log.New(os.Stderr, "[attractor-server] ", log.LstdFlags),
	}
}

// ListenAndServe starts the HTTP server on the given address.
func (s *Server) ListenAndServe(addr string) error {
	mux := http.NewServeMux()
	s.registerRoutes(mux)

	s.httpServer = &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      0, // SSE requires no write timeout
		IdleTimeout:       120 * time.Second,
	}

	s.logger.Printf("listening on %s", addr)
	return s.httpServer.ListenAndServe()
}

// Serve starts the HTTP server on an existing listener. This is useful
// for tests that need to bind to an ephemeral port.
func (s *Server) Serve(ln net.Listener) error {
	mux := http.NewServeMux()
	s.registerRoutes(mux)

	s.httpServer = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      0,
		IdleTimeout:       120 * time.Second,
	}

	s.logger.Printf("serving on %s", ln.Addr())
	return s.httpServer.Serve(ln)
}

// Shutdown gracefully shuts down the server. It cancels all running
// pipelines and waits for the HTTP server to drain.
func (s *Server) Shutdown(ctx context.Context) error {
	// Cancel all running pipelines.
	s.runs.Range(func(key, value any) bool {
		pr := value.(*PipelineRun)
		pr.mu.RLock()
		cancelFn := pr.cancel
		status := pr.Status
		pr.mu.RUnlock()
		if status == StatusRunning && cancelFn != nil {
			cancelFn()
		}
		return true
	})

	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Route registration
// ---------------------------------------------------------------------------

func (s *Server) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /pipelines", s.handleCreatePipeline)
	mux.HandleFunc("GET /pipelines/{id}", s.handleGetPipeline)
	mux.HandleFunc("GET /pipelines/{id}/events", s.handleGetEvents)
	mux.HandleFunc("POST /pipelines/{id}/cancel", s.handleCancelPipeline)
	mux.HandleFunc("GET /pipelines/{id}/graph", s.handleGetGraph)
	mux.HandleFunc("GET /pipelines/{id}/questions", s.handleGetQuestions)
	mux.HandleFunc("POST /pipelines/{id}/questions/{qid}/answer", s.handleAnswerQuestion)
	mux.HandleFunc("GET /pipelines/{id}/checkpoint", s.handleGetCheckpoint)
	mux.HandleFunc("GET /pipelines/{id}/context", s.handleGetContext)

	// Health check endpoint.
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
}

// ---------------------------------------------------------------------------
// POST /pipelines -- Submit DOT source and start execution
// ---------------------------------------------------------------------------

type createPipelineRequest struct {
	DOTSource      string         `json:"dot_source"`
	InitialContext map[string]any `json:"initial_context,omitempty"`
}

type createPipelineResponse struct {
	ID     string         `json:"id"`
	Status PipelineStatus `json:"status"`
}

func (s *Server) handleCreatePipeline(w http.ResponseWriter, r *http.Request) {
	var req createPipelineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse("invalid request body: "+err.Error()))
		return
	}
	if strings.TrimSpace(req.DOTSource) == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse("dot_source is required"))
		return
	}

	pipelineID := generateID()
	now := time.Now()

	pr := &PipelineRun{
		ID:        pipelineID,
		Status:    StatusPending,
		CreatedAt: now,
		DOTSource: req.DOTSource,
		questions: make(map[string]*PendingQuestion),
	}

	s.runs.Store(pipelineID, pr)

	// Launch the pipeline in a background goroutine.
	go s.executePipeline(pr, req.InitialContext)

	writeJSON(w, http.StatusAccepted, createPipelineResponse{
		ID:     pipelineID,
		Status: StatusPending,
	})
}

// executePipeline runs the pipeline engine in a goroutine. It creates a
// per-run WebInterviewer so that wait.human nodes block until HTTP clients
// submit answers.
func (s *Server) executePipeline(pr *PipelineRun, initialContext map[string]any) {
	ctx, cancel := context.WithCancel(context.Background())

	pr.mu.Lock()
	pr.cancel = cancel
	pr.Status = StatusRunning
	pr.mu.Unlock()

	defer cancel()

	// Create a WebInterviewer bound to this pipeline run.
	webInterviewer := &WebInterviewer{run: pr}

	// Build a per-run registry that uses the WebInterviewer for human gates.
	// We pass nil for the PipelineExecutor initially and set it after creating
	// the runner, because the runner itself implements PipelineExecutor.
	registry := handler.DefaultRegistryFull(s.Backend, webInterviewer, nil)

	runner := &engine.Runner{
		Registry:       registry,
		Transforms:     transform.DefaultTransforms(),
		MaxSteps:       s.MaxSteps,
		InitialContext: initialContext,
		OnEvent: func(ev engine.Event) {
			pr.addEvent(ev)

			// Capture checkpoint-saved events to track the latest checkpoint.
			if ev.Kind == engine.EventCheckpointSaved {
				if cpPath, ok := ev.Data["path"].(string); ok {
					cp, err := state.LoadCheckpoint(cpPath)
					if err == nil {
						pr.mu.Lock()
						pr.checkpoint = cp
						pr.mu.Unlock()
					}
				}
			}
		},
	}

	// Create the logs directory for this run.
	logsDir := s.LogsRoot
	if logsDir == "" {
		logsDir = "attractor-logs"
	}

	outcome, err := runner.RunDOT(ctx, pr.DOTSource, logsDir)

	now := time.Now()
	pr.mu.Lock()
	pr.FinishedAt = &now
	pr.outcome = outcome

	if ctx.Err() == context.Canceled && pr.Status == StatusRunning {
		pr.Status = StatusCancelled
		pr.Error = "pipeline cancelled"
	} else if err != nil {
		pr.Status = StatusFailed
		pr.Error = err.Error()
	} else if outcome != nil && outcome.Status == state.StatusFail {
		pr.Status = StatusFailed
		pr.Error = outcome.FailureReason
	} else {
		pr.Status = StatusCompleted
	}
	pr.mu.Unlock()

	// Close all SSE subscriber channels to signal end-of-stream.
	pr.closeAllSubscribers()

	s.logger.Printf("pipeline %s finished: %s", pr.ID, pr.Status)
}

// ---------------------------------------------------------------------------
// GET /pipelines/{id} -- Get pipeline status
// ---------------------------------------------------------------------------

type getPipelineResponse struct {
	ID         string         `json:"id"`
	Status     PipelineStatus `json:"status"`
	CreatedAt  time.Time      `json:"created_at"`
	FinishedAt *time.Time     `json:"finished_at,omitempty"`
	Error      string         `json:"error,omitempty"`
	Outcome    *state.Outcome `json:"outcome,omitempty"`
	EventCount int            `json:"event_count"`
}

func (s *Server) handleGetPipeline(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	pr.mu.RLock()
	resp := getPipelineResponse{
		ID:         pr.ID,
		Status:     pr.Status,
		CreatedAt:  pr.CreatedAt,
		FinishedAt: pr.FinishedAt,
		Error:      pr.Error,
		Outcome:    pr.outcome,
		EventCount: len(pr.events),
	}
	pr.mu.RUnlock()

	writeJSON(w, http.StatusOK, resp)
}

// ---------------------------------------------------------------------------
// GET /pipelines/{id}/events -- SSE stream
// ---------------------------------------------------------------------------

func (s *Server) handleGetEvents(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, errorResponse("streaming not supported"))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	// First, replay all historical events so the client catches up.
	pr.mu.RLock()
	historical := make([]engine.Event, len(pr.events))
	copy(historical, pr.events)
	status := pr.Status
	pr.mu.RUnlock()

	for _, ev := range historical {
		if err := writeSSEEvent(w, ev); err != nil {
			return
		}
		flusher.Flush()
	}

	// If the pipeline has already finished, close after replay.
	if status != StatusRunning && status != StatusPending {
		writeSSEDone(w)
		flusher.Flush()
		return
	}

	// Subscribe to live events.
	ch := pr.subscribe()
	defer pr.unsubscribe(ch)

	for {
		select {
		case ev, open := <-ch:
			if !open {
				// Pipeline finished; channel closed.
				writeSSEDone(w)
				flusher.Flush()
				return
			}
			if err := writeSSEEvent(w, ev); err != nil {
				return
			}
			flusher.Flush()

		case <-r.Context().Done():
			// Client disconnected.
			return
		}
	}
}

// ---------------------------------------------------------------------------
// POST /pipelines/{id}/cancel -- Cancel a running pipeline
// ---------------------------------------------------------------------------

func (s *Server) handleCancelPipeline(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	pr.mu.RLock()
	status := pr.Status
	cancelFn := pr.cancel
	pr.mu.RUnlock()

	if status != StatusRunning {
		writeJSON(w, http.StatusConflict, errorResponse(
			fmt.Sprintf("pipeline is %s, cannot cancel", status)))
		return
	}

	if cancelFn != nil {
		cancelFn()
	}

	// Also unblock any pending questions with a timeout answer so the
	// engine goroutine does not hang.
	pr.mu.RLock()
	pending := make([]*PendingQuestion, 0, len(pr.questions))
	for _, pq := range pr.questions {
		pending = append(pending, pq)
	}
	pr.mu.RUnlock()

	for _, pq := range pending {
		select {
		case pq.answerCh <- interviewer.TimeoutAnswer:
		default:
		}
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "cancelling"})
}

// ---------------------------------------------------------------------------
// GET /pipelines/{id}/graph -- Get DOT source
// ---------------------------------------------------------------------------

func (s *Server) handleGetGraph(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	w.Header().Set("Content-Type", "text/vnd.graphviz")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(pr.DOTSource))
}

// ---------------------------------------------------------------------------
// GET /pipelines/{id}/questions -- List pending questions
// ---------------------------------------------------------------------------

type questionResponse struct {
	ID       string               `json:"id"`
	Question interviewer.Question `json:"question"`
}

func (s *Server) handleGetQuestions(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	pending := pr.listQuestions()
	resp := make([]questionResponse, len(pending))
	for i, pq := range pending {
		resp[i] = questionResponse{
			ID:       pq.ID,
			Question: pq.Question,
		}
	}

	writeJSON(w, http.StatusOK, resp)
}

// ---------------------------------------------------------------------------
// POST /pipelines/{id}/questions/{qid}/answer -- Submit answer
// ---------------------------------------------------------------------------

type answerRequest struct {
	Value          string            `json:"value"`
	Text           string            `json:"text,omitempty"`
	SelectedOption *interviewer.Option `json:"selected_option,omitempty"`
}

func (s *Server) handleAnswerQuestion(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	qid := r.PathValue("qid")
	if qid == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse("question ID is required"))
		return
	}

	pr.mu.RLock()
	pq, exists := pr.questions[qid]
	pr.mu.RUnlock()

	if !exists {
		writeJSON(w, http.StatusNotFound, errorResponse("question not found"))
		return
	}

	var req answerRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse("invalid request body: "+err.Error()))
		return
	}

	answer := interviewer.Answer{
		Value:          interviewer.AnswerValue(req.Value),
		Text:           req.Text,
		SelectedOption: req.SelectedOption,
	}

	// Deliver the answer to the blocked WebInterviewer.Ask() call.
	select {
	case pq.answerCh <- answer:
		writeJSON(w, http.StatusOK, map[string]string{"status": "accepted"})
	default:
		// Channel already has an answer (race condition or duplicate submission).
		writeJSON(w, http.StatusConflict, errorResponse("question already answered"))
	}
}

// ---------------------------------------------------------------------------
// GET /pipelines/{id}/checkpoint -- Get checkpoint state
// ---------------------------------------------------------------------------

func (s *Server) handleGetCheckpoint(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	pr.mu.RLock()
	cp := pr.checkpoint
	pr.mu.RUnlock()

	if cp == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"checkpoint": nil,
			"message":    "no checkpoint available yet",
		})
		return
	}

	writeJSON(w, http.StatusOK, cp)
}

// ---------------------------------------------------------------------------
// GET /pipelines/{id}/context -- Get context key-value store
// ---------------------------------------------------------------------------

func (s *Server) handleGetContext(w http.ResponseWriter, r *http.Request) {
	pr, ok := s.lookupPipeline(w, r)
	if !ok {
		return
	}

	// The pipeline context is available from the checkpoint's context values,
	// which is the most recent snapshot saved by the engine.
	pr.mu.RLock()
	cp := pr.checkpoint
	pr.mu.RUnlock()

	if cp == nil {
		writeJSON(w, http.StatusOK, map[string]any{})
		return
	}

	writeJSON(w, http.StatusOK, cp.ContextValues)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// lookupPipeline extracts the pipeline ID from the URL and looks it up.
// Returns false and writes an error response if the pipeline is not found.
func (s *Server) lookupPipeline(w http.ResponseWriter, r *http.Request) (*PipelineRun, bool) {
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse("pipeline ID is required"))
		return nil, false
	}

	val, ok := s.runs.Load(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, errorResponse("pipeline not found"))
		return nil, false
	}

	return val.(*PipelineRun), true
}

// writeJSON encodes v as JSON and writes it with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.Encode(v)
}

// errorResponse builds a standard error response body.
func errorResponse(message string) map[string]string {
	return map[string]string{"error": message}
}

// writeSSEEvent writes a single Server-Sent Event to the response writer.
func writeSSEEvent(w http.ResponseWriter, ev engine.Event) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Kind, data)
	return err
}

// writeSSEDone writes a terminal SSE event indicating the stream is finished.
func writeSSEDone(w http.ResponseWriter) {
	fmt.Fprint(w, "event: done\ndata: {}\n\n")
}

// generateID produces a random 16-character hex string suitable for use as
// a pipeline or question identifier.
func generateID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		// Fallback to timestamp-based ID if crypto/rand fails.
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}
