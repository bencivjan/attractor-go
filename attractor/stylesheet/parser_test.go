package stylesheet

import (
	"testing"
)

// ---------------------------------------------------------------------------
// Test: Universal selector (*) parsing
// ---------------------------------------------------------------------------

func TestParse_UniversalSelector(t *testing.T) {
	source := `* { model: gpt-4o }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	r := rules[0]
	if r.Selector.Type != SelectorUniversal {
		t.Errorf("expected SelectorUniversal, got %d", r.Selector.Type)
	}
	if r.Selector.Value != "" {
		t.Errorf("expected empty value for universal, got %q", r.Selector.Value)
	}
	if r.Selector.String() != "*" {
		t.Errorf("expected '*' string, got %q", r.Selector.String())
	}
	if r.Declarations["model"] != "gpt-4o" {
		t.Errorf("expected model='gpt-4o', got %q", r.Declarations["model"])
	}
}

// ---------------------------------------------------------------------------
// Test: Class selector (.class) parsing
// ---------------------------------------------------------------------------

func TestParse_ClassSelector(t *testing.T) {
	source := `.fast { model: gpt-4o-mini; temperature: 0.5 }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	r := rules[0]
	if r.Selector.Type != SelectorClass {
		t.Errorf("expected SelectorClass, got %d", r.Selector.Type)
	}
	if r.Selector.Value != "fast" {
		t.Errorf("expected value 'fast', got %q", r.Selector.Value)
	}
	if r.Selector.String() != ".fast" {
		t.Errorf("expected '.fast' string, got %q", r.Selector.String())
	}
	if r.Declarations["model"] != "gpt-4o-mini" {
		t.Errorf("expected model='gpt-4o-mini', got %q", r.Declarations["model"])
	}
	if r.Declarations["temperature"] != "0.5" {
		t.Errorf("expected temperature='0.5', got %q", r.Declarations["temperature"])
	}
}

// ---------------------------------------------------------------------------
// Test: ID selector (#id) parsing
// ---------------------------------------------------------------------------

func TestParse_IDSelector(t *testing.T) {
	source := `#review_stage { provider: anthropic; model: claude-sonnet-4-20250514 }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	r := rules[0]
	if r.Selector.Type != SelectorID {
		t.Errorf("expected SelectorID, got %d", r.Selector.Type)
	}
	if r.Selector.Value != "review_stage" {
		t.Errorf("expected value 'review_stage', got %q", r.Selector.Value)
	}
	if r.Selector.String() != "#review_stage" {
		t.Errorf("expected '#review_stage' string, got %q", r.Selector.String())
	}
	if r.Declarations["provider"] != "anthropic" {
		t.Errorf("expected provider='anthropic', got %q", r.Declarations["provider"])
	}
}

// ---------------------------------------------------------------------------
// Test: Shape selector (box, diamond, etc.) parsing
// ---------------------------------------------------------------------------

func TestParse_ShapeSelector(t *testing.T) {
	source := `box { model: gpt-4o; reasoning_effort: high }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	r := rules[0]
	if r.Selector.Type != SelectorShape {
		t.Errorf("expected SelectorShape, got %d", r.Selector.Type)
	}
	if r.Selector.Value != "box" {
		t.Errorf("expected value 'box', got %q", r.Selector.Value)
	}
	if r.Selector.String() != "box" {
		t.Errorf("expected 'box' string, got %q", r.Selector.String())
	}
	if r.Declarations["reasoning_effort"] != "high" {
		t.Errorf("expected reasoning_effort='high', got %q", r.Declarations["reasoning_effort"])
	}
}

// ---------------------------------------------------------------------------
// Test: Property parsing with = separator
// ---------------------------------------------------------------------------

func TestParse_EqualsSeparator(t *testing.T) {
	source := `* { model = gpt-4o }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	if rules[0].Declarations["model"] != "gpt-4o" {
		t.Errorf("expected model='gpt-4o', got %q", rules[0].Declarations["model"])
	}
}

// ---------------------------------------------------------------------------
// Test: Property with quoted value
// ---------------------------------------------------------------------------

func TestParse_QuotedValue(t *testing.T) {
	source := `* { model: "gpt-4o" }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	// Quotes should be stripped.
	if rules[0].Declarations["model"] != "gpt-4o" {
		t.Errorf("expected model='gpt-4o' (unquoted), got %q", rules[0].Declarations["model"])
	}
}

// ---------------------------------------------------------------------------
// Test: Multiple rules
// ---------------------------------------------------------------------------

func TestParse_MultipleRules(t *testing.T) {
	source := `
		* { model: gpt-4o }
		.fast { model: gpt-4o-mini }
		#special { model: claude-opus-4-20250514; provider: anthropic }
	`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 3 {
		t.Fatalf("expected 3 rules, got %d", len(rules))
	}

	if rules[0].Selector.Type != SelectorUniversal {
		t.Errorf("rule 0: expected universal, got %d", rules[0].Selector.Type)
	}
	if rules[1].Selector.Type != SelectorClass {
		t.Errorf("rule 1: expected class, got %d", rules[1].Selector.Type)
	}
	if rules[2].Selector.Type != SelectorID {
		t.Errorf("rule 2: expected ID, got %d", rules[2].Selector.Type)
	}
}

// ---------------------------------------------------------------------------
// Test: Apply function with specificity cascade
// ---------------------------------------------------------------------------

func TestApply_SpecificityCascade(t *testing.T) {
	rules := []Rule{
		{
			Selector:     Selector{Type: SelectorUniversal},
			Declarations: map[string]string{"model": "default-model", "temperature": "0.7"},
		},
		{
			Selector:     Selector{Type: SelectorShape, Value: "box"},
			Declarations: map[string]string{"model": "shape-model"},
		},
		{
			Selector:     Selector{Type: SelectorClass, Value: "premium"},
			Declarations: map[string]string{"model": "class-model"},
		},
		{
			Selector:     Selector{Type: SelectorID, Value: "node1"},
			Declarations: map[string]string{"model": "id-model"},
		},
	}

	// Node matches all four rules. ID has highest specificity.
	result := Apply(rules, "node1", "box", []string{"premium"})
	if result["model"] != "id-model" {
		t.Errorf("expected ID selector to win, got model=%q", result["model"])
	}
	// temperature should come from universal (only one that sets it).
	if result["temperature"] != "0.7" {
		t.Errorf("expected temperature='0.7' from universal, got %q", result["temperature"])
	}
}

// ---------------------------------------------------------------------------
// Test: Apply -- class beats shape
// ---------------------------------------------------------------------------

func TestApply_ClassBeatsShape(t *testing.T) {
	rules := []Rule{
		{
			Selector:     Selector{Type: SelectorShape, Value: "box"},
			Declarations: map[string]string{"model": "shape-model"},
		},
		{
			Selector:     Selector{Type: SelectorClass, Value: "premium"},
			Declarations: map[string]string{"model": "class-model"},
		},
	}

	result := Apply(rules, "nodeX", "box", []string{"premium"})
	if result["model"] != "class-model" {
		t.Errorf("expected class to beat shape, got model=%q", result["model"])
	}
}

// ---------------------------------------------------------------------------
// Test: Apply -- later rule wins at same specificity
// ---------------------------------------------------------------------------

func TestApply_LaterRuleWinsAtSameSpecificity(t *testing.T) {
	rules := []Rule{
		{
			Selector:     Selector{Type: SelectorUniversal},
			Declarations: map[string]string{"model": "first"},
		},
		{
			Selector:     Selector{Type: SelectorUniversal},
			Declarations: map[string]string{"model": "second"},
		},
	}

	result := Apply(rules, "anyNode", "anyShape", nil)
	if result["model"] != "second" {
		t.Errorf("expected later rule to win at same specificity, got model=%q", result["model"])
	}
}

// ---------------------------------------------------------------------------
// Test: Apply -- no matching rules
// ---------------------------------------------------------------------------

func TestApply_NoMatch(t *testing.T) {
	rules := []Rule{
		{
			Selector:     Selector{Type: SelectorID, Value: "other"},
			Declarations: map[string]string{"model": "specific"},
		},
	}

	result := Apply(rules, "nodeA", "box", nil)
	if len(result) != 0 {
		t.Errorf("expected empty result for no matching rules, got %v", result)
	}
}

// ---------------------------------------------------------------------------
// Test: Selector.Matches
// ---------------------------------------------------------------------------

func TestSelector_Matches(t *testing.T) {
	tests := []struct {
		name    string
		sel     Selector
		nodeID  string
		shape   string
		classes []string
		want    bool
	}{
		{"universal matches all", Selector{Type: SelectorUniversal}, "any", "any", nil, true},
		{"shape match", Selector{Type: SelectorShape, Value: "box"}, "n", "box", nil, true},
		{"shape case insensitive", Selector{Type: SelectorShape, Value: "Box"}, "n", "box", nil, true},
		{"shape mismatch", Selector{Type: SelectorShape, Value: "diamond"}, "n", "box", nil, false},
		{"id match", Selector{Type: SelectorID, Value: "mynode"}, "mynode", "", nil, true},
		{"id mismatch", Selector{Type: SelectorID, Value: "mynode"}, "other", "", nil, false},
		{"class match", Selector{Type: SelectorClass, Value: "fast"}, "n", "", []string{"fast", "gpu"}, true},
		{"class mismatch", Selector{Type: SelectorClass, Value: "slow"}, "n", "", []string{"fast"}, false},
		{"class empty list", Selector{Type: SelectorClass, Value: "fast"}, "n", "", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.sel.Matches(tt.nodeID, tt.shape, tt.classes)
			if got != tt.want {
				t.Errorf("Matches() = %v, want %v", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test: Selector.Specificity
// ---------------------------------------------------------------------------

func TestSelector_Specificity(t *testing.T) {
	if (Selector{Type: SelectorUniversal}).Specificity() != 0 {
		t.Error("universal specificity should be 0")
	}
	if (Selector{Type: SelectorShape}).Specificity() != 1 {
		t.Error("shape specificity should be 1")
	}
	if (Selector{Type: SelectorClass}).Specificity() != 2 {
		t.Error("class specificity should be 2")
	}
	if (Selector{Type: SelectorID}).Specificity() != 3 {
		t.Error("ID specificity should be 3")
	}
}

// ---------------------------------------------------------------------------
// Test: Comments are stripped
// ---------------------------------------------------------------------------

func TestParse_Comments(t *testing.T) {
	source := `
		// This is a line comment
		* { model: gpt-4o }
		/* This is a block comment */
		.fast { temperature: 0.2 }
	`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 2 {
		t.Fatalf("expected 2 rules after stripping comments, got %d", len(rules))
	}
}

// ---------------------------------------------------------------------------
// Test: Error -- missing opening brace
// ---------------------------------------------------------------------------

func TestParse_Error_MissingOpenBrace(t *testing.T) {
	source := `* model: gpt-4o }`
	_, err := Parse(source)
	if err == nil {
		t.Fatal("expected parse error for missing '{'")
	}
}

// ---------------------------------------------------------------------------
// Test: Error -- missing closing brace
// ---------------------------------------------------------------------------

func TestParse_Error_MissingCloseBrace(t *testing.T) {
	source := `* { model: gpt-4o`
	_, err := Parse(source)
	if err == nil {
		t.Fatal("expected parse error for missing '}'")
	}
}

// ---------------------------------------------------------------------------
// Test: Error -- empty ID selector
// ---------------------------------------------------------------------------

func TestParse_Error_EmptyIDSelector(t *testing.T) {
	source := `# { model: gpt-4o }`
	_, err := Parse(source)
	if err == nil {
		t.Fatal("expected parse error for empty ID selector")
	}
}

// ---------------------------------------------------------------------------
// Test: Error -- empty class selector
// ---------------------------------------------------------------------------

func TestParse_Error_EmptyClassSelector(t *testing.T) {
	source := `. { model: gpt-4o }`
	_, err := Parse(source)
	if err == nil {
		t.Fatal("expected parse error for empty class selector")
	}
}

// ---------------------------------------------------------------------------
// Test: Error -- bad declaration (no separator)
// ---------------------------------------------------------------------------

func TestParse_Error_BadDeclaration(t *testing.T) {
	source := `* { model gpt-4o }`
	_, err := Parse(source)
	if err == nil {
		t.Fatal("expected parse error for declaration without : or =")
	}
}

// ---------------------------------------------------------------------------
// Test: Error -- unrecognized selector
// ---------------------------------------------------------------------------

func TestParse_Error_UnrecognizedSelector(t *testing.T) {
	source := `123bad { model: gpt-4o }`
	_, err := Parse(source)
	if err == nil {
		t.Fatal("expected parse error for invalid selector")
	}
}

// ---------------------------------------------------------------------------
// Test: Empty stylesheet
// ---------------------------------------------------------------------------

func TestParse_EmptyStylesheet(t *testing.T) {
	rules, err := Parse("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 0 {
		t.Errorf("expected 0 rules, got %d", len(rules))
	}
}

// ---------------------------------------------------------------------------
// Test: Whitespace-only stylesheet
// ---------------------------------------------------------------------------

func TestParse_WhitespaceOnly(t *testing.T) {
	rules, err := Parse("   \n\t  ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 0 {
		t.Errorf("expected 0 rules, got %d", len(rules))
	}
}

// ---------------------------------------------------------------------------
// Test: Empty declaration block
// ---------------------------------------------------------------------------

func TestParse_EmptyDeclarations(t *testing.T) {
	source := `* {}`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	if len(rules[0].Declarations) != 0 {
		t.Errorf("expected 0 declarations, got %d", len(rules[0].Declarations))
	}
}

// ---------------------------------------------------------------------------
// Test: Newline-separated declarations
// ---------------------------------------------------------------------------

func TestParse_NewlineSeparatedDeclarations(t *testing.T) {
	source := `* {
		model: gpt-4o
		temperature: 0.7
		provider: openai
	}`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	if len(rules[0].Declarations) != 3 {
		t.Errorf("expected 3 declarations, got %d: %v", len(rules[0].Declarations), rules[0].Declarations)
	}
}

// ---------------------------------------------------------------------------
// Test: Single-quoted values
// ---------------------------------------------------------------------------

func TestParse_SingleQuotedValue(t *testing.T) {
	source := `* { model: 'gpt-4o' }`
	rules, err := Parse(source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rules[0].Declarations["model"] != "gpt-4o" {
		t.Errorf("expected single-quoted value to be unquoted, got %q", rules[0].Declarations["model"])
	}
}
