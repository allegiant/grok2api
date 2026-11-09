package gateway

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

func TestPeekQualityStreamDiagnosticIsBoundedAndContentFree(t *testing.T) {
	ciphertext := "PRIVATE_CIPHERTEXT_VALUE"
	argument := `{"command":"PRIVATE_TOOL_ARGUMENT"}`
	finalArgument := `{"command":"PRIVATE_FINAL_ARGUMENT"}`
	fixture := sse(
		`data: {"type":"response.output_item.added","item":{"id":"private-id","type":"reasoning"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"private-id","type":"reasoning","encrypted_content":"`+ciphertext+`"}}`,
		`data: {"type":"response.function_call_arguments.delta","delta":`+mustJSONString(t, argument)+`}`,
		`data: {"type":"response.completed","response":{"id":"private-response-id","usage":{"output_tokens":57,"output_tokens_details":{"reasoning_tokens":0}},"output":[{"id":"private-call-id","type":"function_call","arguments":`+mustJSONString(t, finalArgument)+`}]}}`,
	)
	replay, verdict, usage, metadata, err := peekQualityStream(
		context.Background(),
		io.NopCloser(strings.NewReader(fixture)),
		qualityProtocolResponses,
		QualityRetryRuntime{MinOutputTokens: 8, HoldTimeout: time.Second},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer replay.Close()
	if verdict != QualityWithhold || usage.OutputTokens != 57 || usage.ReasoningTokens != 0 {
		t.Fatalf("unexpected quality result: verdict=%s usage=%#v", verdict, usage)
	}
	if !metadata.Stream.Terminal || !metadata.Stream.UsageReported || metadata.Stream.ReasoningPayloadBytes != len(ciphertext) {
		t.Fatalf("stream summary missed terminal/usage/cipher length: %#v", metadata.Stream)
	}
	if len(metadata.Stream.Events) == 0 || len(metadata.Stream.OutputItemTypes) == 0 {
		t.Fatalf("stream summary missed event or item types: %#v", metadata.Stream)
	}
	encoded, err := json.Marshal(metadata.Stream)
	if err != nil {
		t.Fatal(err)
	}
	got := string(encoded)
	for _, secret := range []string{ciphertext, "PRIVATE_TOOL_ARGUMENT", "PRIVATE_FINAL_ARGUMENT", "private-id", "private-call-id", "private-response-id"} {
		if strings.Contains(got, secret) {
			t.Fatalf("stream summary leaked %q: %s", secret, got)
		}
	}
}

func TestQualityStreamDiagnosticCapsUnknownTypes(t *testing.T) {
	state := qualityScanState{}
	for i := 0; i < qualityDiagnosticMaxTypes+8; i++ {
		state.addDiagnosticEvent("PRIVATE_EVENT_NAME")
	}
	if len(state.diagnostic.Events) != qualityDiagnosticMaxTypes || !state.diagnostic.EventsTruncated || state.diagnostic.EventCount != qualityDiagnosticMaxTypes+8 {
		t.Fatalf("event summary did not cap its type list: %#v", state.diagnostic)
	}
	for _, event := range state.diagnostic.Events {
		if event != "other" {
			t.Fatalf("untrusted event type was retained: %#v", state.diagnostic.Events)
		}
	}
}

func mustJSONString(t *testing.T, value string) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}
