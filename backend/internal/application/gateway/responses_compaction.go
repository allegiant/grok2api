package gateway

import (
	"encoding/json"
	"strings"
)

// Distinctive lines from known compaction prompts. Codex remote-v2 uses
// compaction_trigger. Grok TUI appends the first sentence as the last user
// item. Claude Code 2.1.331 has three summary prompts, and all of them start
// with the second sentence.
const (
	clientCompactionPromptMarker = "it is a system-generated compaction prompt, not a real user message"
	claudeCodeCompactionMarker   = "Your task is to create a detailed summary of"
)

type responsesCompactionKind uint8

const (
	responsesCompactionNone responsesCompactionKind = iota
	responsesCompactionTrigger
	responsesCompactionTUI
)

// isResponsesCompactionRequest detects a context-compaction turn without
// retaining the request body. Codex remote compaction v2 sends
// compaction_trigger; Grok TUI sends the canonical summary prompt as the
// last input/message item. The Provider adapter still requires the trigger
// before it rewrites the body into an encrypted compaction blob.
func isResponsesCompactionRequest(body []byte) bool {
	return classifyResponsesCompactionRequest(body) != responsesCompactionNone
}

func classifyResponsesCompactionRequest(body []byte) responsesCompactionKind {
	var payload struct {
		Input    []json.RawMessage `json:"input"`
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return responsesCompactionNone
	}
	if hasCompactionTrigger(payload.Input) {
		return responsesCompactionTrigger
	}
	if latestUserTextLooksLikeCompactionPrompt(payload.Input) || latestUserTextLooksLikeCompactionPrompt(payload.Messages) {
		return responsesCompactionTUI
	}
	return responsesCompactionNone
}

func hasCompactionTrigger(items []json.RawMessage) bool {
	for _, raw := range items {
		var item struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(raw, &item) != nil {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(item.Type), "compaction_trigger") {
			return true
		}
	}
	return false
}

func latestUserTextLooksLikeCompactionPrompt(items []json.RawMessage) bool {
	for i := len(items) - 1; i >= 0; i-- {
		var item struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(items[i], &item) != nil || !strings.EqualFold(strings.TrimSpace(item.Role), "user") {
			continue
		}
		text := strings.TrimSpace(extractContentText(item.Content))
		if text == "" {
			continue
		}
		return looksLikeCompactionPrompt(text)
	}
	return false
}

func looksLikeCompactionPrompt(text string) bool {
	return strings.Contains(text, clientCompactionPromptMarker) || strings.Contains(text, claudeCodeCompactionMarker)
}

func extractContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var builder strings.Builder
		for _, part := range parts {
			builder.WriteString(part.Text)
		}
		return builder.String()
	}
	return ""
}
