package gateway

import "testing"

const tuiCompactionPrompt = "Your task is to produce a faithful, concise summary of the conversation so far. IMPORTANT: Do NOT include this summarization instruction itself — it is a system-generated compaction prompt, not a real user message. Output the final summary inside a single <summary> block."

func TestIsResponsesCompactionRequest(t *testing.T) {
	tests := []struct {
		name string
		body string
		want responsesCompactionKind
	}{
		{name: "remote v2", body: `{"input":[{"role":"user","content":"continue"},{"type":"compaction_trigger"}]}`, want: responsesCompactionTrigger},
		{name: "not terminal", body: `{"input":[{"type":"compaction_trigger"},{"role":"user","content":"continue"}]}`, want: responsesCompactionTrigger},
		{name: "tui last user string", body: `{"input":[{"role":"user","content":"continue"},{"role":"user","content":"` + tuiCompactionPrompt + `"}]}`, want: responsesCompactionTUI},
		{name: "tui last user parts", body: `{"input":[{"role":"user","content":[{"type":"input_text","text":"` + tuiCompactionPrompt + `"}]}]}`, want: responsesCompactionTUI},
		{name: "tui chat messages", body: `{"messages":[{"role":"user","content":"` + tuiCompactionPrompt + `"}]}`, want: responsesCompactionTUI},
		{name: "prompt only in history", body: `{"input":[{"role":"user","content":"` + tuiCompactionPrompt + `"},{"role":"user","content":"continue the task"}]}`, want: responsesCompactionNone},
		{name: "claude code summary before tool result", body: `{"messages":[{"role":"user","content":[{"type":"text","text":"` + claudeCodeCompactionMarker + `"}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}`, want: responsesCompactionTUI},
		{name: "claude code prompt followed by normal user text", body: `{"messages":[{"role":"user","content":"` + claudeCodeCompactionMarker + `"},{"role":"user","content":"continue the task"}]}`, want: responsesCompactionNone},
		{name: "claude code partial summary", body: `{"messages":[{"role":"user","content":"Your task is to create a detailed summary of the RECENT portion of the conversation — the messages that follow earlier retained context."}]}`, want: responsesCompactionTUI},
		{name: "assistant prompt", body: `{"input":[{"role":"assistant","content":"` + tuiCompactionPrompt + `"}]}`, want: responsesCompactionNone},
		{name: "system prompt", body: `{"messages":[{"role":"system","content":"` + tuiCompactionPrompt + `"}]}`, want: responsesCompactionNone},
		{name: "missing role", body: `{"input":[{"content":"` + tuiCompactionPrompt + `"}]}`, want: responsesCompactionNone},
		{name: "normal responses", body: `{"input":[{"role":"user","content":"continue"}]}`, want: responsesCompactionNone},
		{name: "string input", body: `{"input":"continue"}`, want: responsesCompactionNone},
		{name: "invalid json", body: `{`, want: responsesCompactionNone},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := classifyResponsesCompactionRequest([]byte(test.body)); got != test.want {
				t.Fatalf("classifyResponsesCompactionRequest() = %d, want %d", got, test.want)
			}
			if got := isResponsesCompactionRequest([]byte(test.body)); got != (test.want != responsesCompactionNone) {
				t.Fatalf("isResponsesCompactionRequest() = %t, want %t", got, test.want != responsesCompactionNone)
			}
		})
	}
}
