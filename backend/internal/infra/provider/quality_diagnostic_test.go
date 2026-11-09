package provider

import (
	"encoding/json"
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
