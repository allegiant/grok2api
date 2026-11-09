package provider

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestSummarizeRequestShapeRedactsContentAndBoundsLabels(t *testing.T) {
	body := `{"system":"PRIVATE_SYSTEM_TEXT","tools":[{"name":"PRIVATE_TOOL_NAME"}],"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"PRIVATE_THINKING","signature":"PRIVATE_SIGNATURE"}]},{"role":"assistant","content":[{"type":"tool_use","name":"PRIVATE_TOOL_NAME","input":{"command":"PRIVATE_TOOL_ARGUMENT"}}]},{"role":"user","content":[{"type":"tool_result","content":"PRIVATE_TOOL_RESULT"}]}]}`
	shape := SummarizeRequestShape([]byte(body), "messages")
	encoded, err := json.Marshal(shape)
	if err != nil {
		t.Fatal(err)
	}
	got := string(encoded)
	for _, secret := range []string{"PRIVATE_SYSTEM_TEXT", "PRIVATE_TOOL_NAME", "PRIVATE_THINKING", "PRIVATE_SIGNATURE", "PRIVATE_TOOL_ARGUMENT", "PRIVATE_TOOL_RESULT"} {
		if strings.Contains(got, secret) {
			t.Fatalf("request summary leaked %q: %s", secret, got)
		}
	}
	if shape.Protocol != "messages" || shape.PayloadFormat != "messages" || len(shape.Items) != 4 || shape.ToolDefinitions != 1 {
		t.Fatalf("unexpected shape: %#v", shape)
	}
	if shape.Items[1].Role != "assistant" || len(shape.Items[1].ContentTypes) != 1 || shape.Items[1].ContentTypes[0] != "thinking" || shape.Items[1].SignatureBytes != len("PRIVATE_SIGNATURE") {
		t.Fatalf("thinking block shape was not retained: %#v", shape.Items[1])
	}
	if shape.Items[2].ToolCalls != 1 || shape.Items[3].ToolResults != 1 {
		t.Fatalf("tool structure was not retained: %#v", shape.Items)
	}
}

func TestObserveUpstreamResponsesPreservesBytesAndRedactsSummary(t *testing.T) {
	ciphertext := "PRIVATE_RAW_CIPHERTEXT"
	prompt := "PRIVATE_RAW_REASONING"
	argument := `{"token":"PRIVATE_TOOL_ARGUMENT"}`
	frames := []string{
		`event: response.output_item.added` + "\n" + `data: {"type":"response.output_item.added","item":{"id":"PRIVATE_ID","type":"reasoning"}}`,
		`event: response.reasoning_summary_text.delta` + "\n" + `data: {"type":"response.reasoning_summary_text.delta","delta":"` + prompt + `"}`,
		`event: response.output_item.done` + "\n" + `data: {"type":"response.output_item.done","item":{"id":"PRIVATE_ID","type":"reasoning","encrypted_content":"` + ciphertext + `","summary":[{"type":"summary_text","text":"PRIVATE_SUMMARY"}]}}`,
		`event: response.output_item.added` + "\n" + `data: {"type":"response.output_item.added","item":{"id":"PRIVATE_CALL_ID","type":"function_call"}}`,
		`event: response.function_call_arguments.delta` + "\n" + `data: {"type":"response.function_call_arguments.delta","delta":` + mustJSON(t, argument) + `}`,
		`event: response.in_progress` + "\n" + `data: {"type":"response.in_progress","usage":{"output_tokens":20,"output_tokens_details":{"reasoning_tokens":8}}}`,
		`event: response.completed` + "\n" + `data: {"type":"response.completed","response":{"usage":{"output_tokens":34,"output_tokens_details":{"reasoning_tokens":0}},"output":[{"type":"function_call_output","output":"PRIVATE_TOOL_OUTPUT"}]}}`,
	}
	fixture := strings.Join(frames, "\n\n") + "\n\n"
	diagnostic := &QualityRequestDiagnostic{}
	body := diagnostic.ObserveUpstreamResponses(io.NopCloser(strings.NewReader(fixture)))
	gotBytes, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if string(gotBytes) != fixture {
		t.Fatal("raw response observer changed upstream SSE bytes")
	}
	_, _, summary := diagnostic.Snapshot()
	if summary.ReasoningItemEvents == 0 || summary.ReasoningTextEvents != 1 || summary.ReasoningTextParts != 1 || summary.ReasoningPayloadBytes != len(ciphertext) {
		t.Fatalf("raw reasoning summary = %#v", summary)
	}
	if summary.FunctionCallItemEvents == 0 || summary.FunctionCallArgumentEvents != 1 || summary.FunctionCallOutputEvents == 0 {
		t.Fatalf("raw tool summary = %#v", summary)
	}
	if !summary.UsageReported || summary.OutputTokens != 34 || summary.ReasoningTokens != 0 || !summary.Terminal {
		t.Fatalf("raw usage/terminal summary = %#v", summary)
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{ciphertext, prompt, argument, "PRIVATE_SUMMARY", "PRIVATE_TOOL_OUTPUT", "PRIVATE_ID", "PRIVATE_CALL_ID"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("raw response summary leaked %q: %s", secret, encoded)
		}
	}
}

func TestRawResponsesSummaryBoundsEventTypeLists(t *testing.T) {
	fixture := strings.Repeat("data: {\"type\":\"response.created\"}\n\n", maxRawResponsesTypes+5)
	diagnostic := &QualityRequestDiagnostic{}
	body := diagnostic.ObserveUpstreamResponses(io.NopCloser(strings.NewReader(fixture)))
	if _, err := io.Copy(io.Discard, body); err != nil {
		t.Fatal(err)
	}
	_ = body.Close()
	_, _, summary := diagnostic.Snapshot()
	if len(summary.Events) != maxRawResponsesTypes || !summary.EventsTruncated || summary.EventCount != maxRawResponsesTypes+5 {
		t.Fatalf("raw event type list was not bounded: %#v", summary)
	}
}

func TestSummarizeRequestControlsAreSafeAndUseful(t *testing.T) {
	body := `{"instructions":"PRIVATE_INSTRUCTIONS","reasoning":{"effort":"HIGH","summary":"DETAILED"},"output_config":{"effort":"MAX"},"thinking":{"type":"ADAPTIVE","budget_tokens":8192},"tool_choice":{"type":"function","name":"PRIVATE_TOOL_NAME"},"parallel_tool_calls":false,"max_output_tokens":4096,"previous_response_id":"PRIVATE_RESPONSE_ID","tools":[{"type":"function","name":"PRIVATE_TOOL_NAME"}],"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"PRIVATE_PROMPT"}]}]}`
	shape := SummarizeRequestShape([]byte(body), "responses")
	if shape.PayloadBytes != len(body) || shape.ReasoningEffort != "high" || shape.OutputConfigEffort != "max" || shape.ReasoningSummary != "detailed" || shape.ThinkingType != "adaptive" {
		t.Fatalf("request controls were not summarized: %#v", shape)
	}
	if shape.ThinkingBudgetTokens == nil || *shape.ThinkingBudgetTokens != 8192 || shape.ToolChoice != "function" || shape.ParallelToolCalls == nil || *shape.ParallelToolCalls {
		t.Fatalf("thinking/tool controls were not summarized: %#v", shape)
	}
	if shape.OutputTokenLimit == nil || *shape.OutputTokenLimit != 4096 || shape.OutputTokenLimitSource != "max_output_tokens" || !shape.PreviousResponseIDSeen {
		t.Fatalf("token/replay controls were not summarized: %#v", shape)
	}
	encoded, err := json.Marshal(shape)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"PRIVATE_INSTRUCTIONS", "PRIVATE_TOOL_NAME", "PRIVATE_RESPONSE_ID", "PRIVATE_PROMPT"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("request controls leaked %q: %s", secret, encoded)
		}
	}
}

func TestSummarizeRequestControlsCollapseUnknownLabels(t *testing.T) {
	shape := SummarizeRequestShape([]byte(`{"reasoning":{"effort":"PRIVATE_EFFORT","summary":"PRIVATE_SUMMARY"},"tool_choice":{"type":"PRIVATE_CHOICE","name":"PRIVATE_TOOL_NAME"}}`), "responses")
	if shape.ReasoningEffort != "other" || shape.ReasoningSummary != "other" || shape.ToolChoice != "other" {
		t.Fatalf("unknown control labels were not collapsed: %#v", shape)
	}
	encoded, err := json.Marshal(shape)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"PRIVATE_EFFORT", "PRIVATE_SUMMARY", "PRIVATE_CHOICE", "PRIVATE_TOOL_NAME"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("unknown request control leaked %q: %s", secret, encoded)
		}
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func TestSummarizeResponsesRequestShapeAndInputLimit(t *testing.T) {
	body := `{"instructions":"PRIVATE_INSTRUCTIONS","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"PRIVATE_PROMPT"}]},{"type":"reasoning","encrypted_content":"PRIVATE_ENCRYPTED_CONTENT"},{"type":"function_call_output","call_id":"PRIVATE_ID","output":"PRIVATE_OUTPUT"}]}`
	shape := SummarizeRequestShape([]byte(body), "responses")
	encoded, err := json.Marshal(shape)
	if err != nil {
		t.Fatal(err)
	}
	got := string(encoded)
	for _, secret := range []string{"PRIVATE_INSTRUCTIONS", "PRIVATE_PROMPT", "PRIVATE_ENCRYPTED_CONTENT", "PRIVATE_ID", "PRIVATE_OUTPUT"} {
		if strings.Contains(got, secret) {
			t.Fatalf("Responses summary leaked %q: %s", secret, got)
		}
	}
	if shape.Protocol != "responses" || shape.PayloadFormat != "responses" || len(shape.Items) != 4 || shape.Items[0].Type != "instructions" || shape.Items[1].Type != "message" || shape.Items[2].Type != "reasoning" || shape.Items[2].EncryptedContentBytes != len("PRIVATE_ENCRYPTED_CONTENT") || shape.Items[3].Type != "function_call_output" || shape.Items[3].ToolResults != 1 {
		t.Fatalf("unexpected Responses shape: %#v", shape)
	}

	convertedMessages := SummarizeRequestShape([]byte(`{"input":[{"type":"message","role":"assistant","content":[{"type":"reasoning"}]}]}`), "messages")
	if convertedMessages.Protocol != "messages" || convertedMessages.PayloadFormat != "responses" {
		t.Fatalf("converted Messages payload format was not distinguished: %#v", convertedMessages)
	}

	tooLarge := make([]byte, maxRequestShapeBodyBytes+1)
	limited := SummarizeRequestShape(tooLarge, "responses")
	if !limited.Truncated || len(limited.Items) != 0 {
		t.Fatalf("oversized body was not bounded: %#v", limited)
	}
}
