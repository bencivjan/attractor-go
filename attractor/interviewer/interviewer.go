// Package interviewer defines the human-in-the-loop interaction abstraction
// for the Attractor pipeline engine. It provides the Interviewer interface
// used by handlers that require human input (e.g. WaitForHumanHandler), along
// with several ready-made implementations for testing, automation, and
// interactive use.
package interviewer

import (
	"regexp"
	"strings"
	"sync"
	"unicode"
)

// ---------------------------------------------------------------------------
// QuestionType
// ---------------------------------------------------------------------------

// QuestionType discriminates the kind of input expected from the human.
type QuestionType int

const (
	// QuestionYesNo expects a binary yes/no answer.
	QuestionYesNo QuestionType = iota
	// QuestionMultipleChoice expects selection from a list of options.
	QuestionMultipleChoice
	// QuestionFreeform expects free-text input.
	QuestionFreeform
	// QuestionConfirmation expects an explicit confirmation (similar to yes/no).
	QuestionConfirmation
)

// String returns the human-readable name of the question type.
func (qt QuestionType) String() string {
	switch qt {
	case QuestionYesNo:
		return "yes_no"
	case QuestionMultipleChoice:
		return "multiple_choice"
	case QuestionFreeform:
		return "freeform"
	case QuestionConfirmation:
		return "confirmation"
	default:
		return "unknown"
	}
}

// ---------------------------------------------------------------------------
// Option
// ---------------------------------------------------------------------------

// Option represents a selectable choice presented to the user.
type Option struct {
	Key   string `json:"key"`
	Label string `json:"label"`
}

// ---------------------------------------------------------------------------
// Question
// ---------------------------------------------------------------------------

// Question represents a prompt posed to a human (or automated) reviewer
// during pipeline execution.
type Question struct {
	Text           string         `json:"text"`
	Type           QuestionType   `json:"type"`
	Options        []Option       `json:"options"`
	Default        *Answer        `json:"default,omitempty"`
	TimeoutSeconds float64        `json:"timeout_seconds,omitempty"`
	Stage          string         `json:"stage"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

// ---------------------------------------------------------------------------
// AnswerValue
// ---------------------------------------------------------------------------

// AnswerValue represents sentinel answer values for non-textual responses.
type AnswerValue string

const (
	// AnswerYes indicates affirmative.
	AnswerYes AnswerValue = "yes"
	// AnswerNo indicates negative.
	AnswerNo AnswerValue = "no"
	// AnswerSkipped indicates the question was skipped.
	AnswerSkipped AnswerValue = "skipped"
	// AnswerTimeout indicates the question timed out without a response.
	AnswerTimeout AnswerValue = "timeout"
)

// ---------------------------------------------------------------------------
// Answer
// ---------------------------------------------------------------------------

// Answer represents the response from a human (or automated) reviewer.
type Answer struct {
	Value          AnswerValue `json:"value"`
	SelectedOption *Option     `json:"selected_option,omitempty"`
	Text           string      `json:"text,omitempty"`
}

// Predefined sentinel answers for common responses.
var (
	YesAnswer     = Answer{Value: AnswerYes}
	NoAnswer      = Answer{Value: AnswerNo}
	SkippedAnswer = Answer{Value: AnswerSkipped}
	TimeoutAnswer = Answer{Value: AnswerTimeout}
)

// ---------------------------------------------------------------------------
// Interviewer interface
// ---------------------------------------------------------------------------

// Interviewer is the core abstraction for human-in-the-loop pipeline stages.
// Implementations range from fully automated (AutoApproveInterviewer) to
// interactive (console, web UI) to test-oriented (QueueInterviewer).
type Interviewer interface {
	// Ask presents a single question and blocks until an answer is available.
	Ask(q Question) Answer

	// AskMultiple presents multiple questions sequentially and returns the
	// answers in the same order.
	AskMultiple(questions []Question) []Answer

	// Inform sends a non-interactive informational message to the reviewer.
	Inform(message, stage string)
}

// ---------------------------------------------------------------------------
// AutoApproveInterviewer
// ---------------------------------------------------------------------------

// AutoApproveInterviewer always approves or selects the first option. It is
// used for fully automated pipeline runs where no human input is required.
type AutoApproveInterviewer struct{}

// Ask returns an automatic approval response based on the question type.
func (a *AutoApproveInterviewer) Ask(q Question) Answer {
	switch q.Type {
	case QuestionYesNo, QuestionConfirmation:
		return YesAnswer
	case QuestionMultipleChoice:
		if len(q.Options) > 0 {
			opt := q.Options[0]
			return Answer{
				Value:          AnswerValue(opt.Key),
				SelectedOption: &opt,
				Text:           opt.Label,
			}
		}
		return YesAnswer
	case QuestionFreeform:
		if q.Default != nil {
			return *q.Default
		}
		return Answer{Value: "approved", Text: "auto-approved"}
	default:
		return YesAnswer
	}
}

// AskMultiple answers each question sequentially using Ask.
func (a *AutoApproveInterviewer) AskMultiple(questions []Question) []Answer {
	answers := make([]Answer, len(questions))
	for i, q := range questions {
		answers[i] = a.Ask(q)
	}
	return answers
}

// Inform is a no-op for the auto-approve interviewer.
func (a *AutoApproveInterviewer) Inform(message, stage string) {}

// ---------------------------------------------------------------------------
// QueueInterviewer
// ---------------------------------------------------------------------------

// QueueInterviewer reads answers from a pre-filled queue. When the queue is
// exhausted it falls back to a default answer. This implementation is
// designed for testing and replay scenarios.
type QueueInterviewer struct {
	answers  []Answer
	fallback Answer
	pos      int
	mu       sync.Mutex
}

// NewQueueInterviewer creates a QueueInterviewer pre-loaded with the given
// answers. Once all answers are consumed, subsequent calls to Ask return
// AnswerYes as the fallback.
func NewQueueInterviewer(answers ...Answer) *QueueInterviewer {
	return &QueueInterviewer{
		answers:  answers,
		fallback: YesAnswer,
	}
}

// NewQueueInterviewerWithFallback creates a QueueInterviewer with an
// explicit fallback answer used when the queue is exhausted.
func NewQueueInterviewerWithFallback(fallback Answer, answers ...Answer) *QueueInterviewer {
	return &QueueInterviewer{
		answers:  answers,
		fallback: fallback,
	}
}

// Ask returns the next queued answer, or the fallback if the queue is empty.
func (q *QueueInterviewer) Ask(_ Question) Answer {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.pos < len(q.answers) {
		a := q.answers[q.pos]
		q.pos++
		return a
	}
	return q.fallback
}

// AskMultiple answers each question sequentially using Ask.
func (q *QueueInterviewer) AskMultiple(questions []Question) []Answer {
	answers := make([]Answer, len(questions))
	for i, question := range questions {
		answers[i] = q.Ask(question)
	}
	return answers
}

// Inform is a no-op for the queue interviewer.
func (q *QueueInterviewer) Inform(message, stage string) {}

// Remaining returns the number of unconsumed answers still in the queue.
func (q *QueueInterviewer) Remaining() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	remaining := len(q.answers) - q.pos
	if remaining < 0 {
		return 0
	}
	return remaining
}

// ---------------------------------------------------------------------------
// CallbackInterviewer
// ---------------------------------------------------------------------------

// CallbackInterviewer delegates all questions to a caller-supplied callback
// function. This is useful for programmatic integrations and custom UIs.
type CallbackInterviewer struct {
	Callback func(Question) Answer
	OnInform func(message, stage string)
}

// Ask delegates to the callback function.
func (c *CallbackInterviewer) Ask(q Question) Answer {
	if c.Callback == nil {
		return YesAnswer
	}
	return c.Callback(q)
}

// AskMultiple answers each question sequentially using Ask.
func (c *CallbackInterviewer) AskMultiple(questions []Question) []Answer {
	answers := make([]Answer, len(questions))
	for i, q := range questions {
		answers[i] = c.Ask(q)
	}
	return answers
}

// Inform delegates to the OnInform callback if set, otherwise no-op.
func (c *CallbackInterviewer) Inform(message, stage string) {
	if c.OnInform != nil {
		c.OnInform(message, stage)
	}
}

// ---------------------------------------------------------------------------
// RecordingInterviewer
// ---------------------------------------------------------------------------

// QAPair holds a recorded question-answer exchange.
type QAPair struct {
	Question Question
	Answer   Answer
}

// RecordingInterviewer wraps a delegate interviewer and records all
// question-answer pairs for later inspection. Thread-safe.
type RecordingInterviewer struct {
	delegate Interviewer
	records  []QAPair
	mu       sync.Mutex
}

// NewRecordingInterviewer creates a RecordingInterviewer wrapping the given
// delegate.
func NewRecordingInterviewer(delegate Interviewer) *RecordingInterviewer {
	return &RecordingInterviewer{delegate: delegate}
}

// Ask delegates to the wrapped interviewer and records the exchange.
func (r *RecordingInterviewer) Ask(q Question) Answer {
	a := r.delegate.Ask(q)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, QAPair{Question: q, Answer: a})
	return a
}

// AskMultiple answers each question sequentially using Ask.
func (r *RecordingInterviewer) AskMultiple(questions []Question) []Answer {
	answers := make([]Answer, len(questions))
	for i, q := range questions {
		answers[i] = r.Ask(q)
	}
	return answers
}

// Inform delegates to the wrapped interviewer.
func (r *RecordingInterviewer) Inform(message, stage string) {
	r.delegate.Inform(message, stage)
}

// Recorded returns a copy of all recorded question-answer pairs.
func (r *RecordingInterviewer) Recorded() []QAPair {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]QAPair, len(r.records))
	copy(out, r.records)
	return out
}

// ---------------------------------------------------------------------------
// ParseAcceleratorKey
// ---------------------------------------------------------------------------

// Accelerator patterns for edge labels.
var (
	bracketPattern = regexp.MustCompile(`^\[(\w+)\]\s*(.+)$`)
	parenPattern   = regexp.MustCompile(`^(\w+)\)\s*(.+)$`)
	dashPattern    = regexp.MustCompile(`^(\w+)\s*-\s*(.+)$`)
)

// ParseAcceleratorKey extracts a shortcut key and label from common edge
// label patterns used in pipeline graphs. Supported formats:
//
//	"[Y] Yes"   -> key="Y", label="Yes"
//	"Y) Yes"    -> key="Y", label="Yes"
//	"Y - Yes"   -> key="Y", label="Yes"
//	"Yes"       -> key="Y" (first character, uppercased), label="Yes"
//
// Returns an Option with the extracted key and label.
func ParseAcceleratorKey(raw string) Option {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Option{Key: "?", Label: ""}
	}

	if m := bracketPattern.FindStringSubmatch(raw); m != nil {
		return Option{
			Key:   strings.ToUpper(m[1]),
			Label: strings.TrimSpace(m[2]),
		}
	}
	if m := parenPattern.FindStringSubmatch(raw); m != nil {
		return Option{
			Key:   strings.ToUpper(m[1]),
			Label: strings.TrimSpace(m[2]),
		}
	}
	if m := dashPattern.FindStringSubmatch(raw); m != nil {
		return Option{
			Key:   strings.ToUpper(m[1]),
			Label: strings.TrimSpace(m[2]),
		}
	}

	// Fallback: first character as key, full string as label.
	key := "?"
	for _, r := range raw {
		key = string(unicode.ToUpper(r))
		break
	}
	return Option{Key: key, Label: strings.TrimSpace(raw)}
}
