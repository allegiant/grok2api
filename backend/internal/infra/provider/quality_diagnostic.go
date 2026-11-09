package provider

import (
	"encoding/json"
	"strings"
)

const (
	maxRequestShapeBodyBytes    = 8 << 20
	maxRequestShapeItems        = 24
	maxRequestShapeContentTypes = 12
	maxRequestShapeToolCount    = 10000
)

// RequestShapeSummary contains only protocol labels, bounded counts, and
// encrypted payload lengths. It must never contain values from request content.
type RequestShapeSummary struct {
	Protocol        string             `json:"protocol"`
	PayloadFormat   string             `json:"payload_format"`
	Items           []RequestShapeItem `json:"items,omitempty"`
	ToolDefinitions int                `json:"tool_definitions,omitempty"`
	Truncated       bool               `json:"truncated,omitempty"`
	InvalidJSON     bool               `json:"invalid_json,omitempty"`
}

// RequestShapeItem describes one message or Responses input item without
// retaining text, tool names, arguments, identifiers, or ciphertext.
type RequestShapeItem struct {
	Role                  string   `json:"role,omitempty"`
	Type                  string   `json:"type,omitempty"`
	ContentTypes          []string `json:"content_types,omitempty"`
	ContentTypesTruncated bool     `json:"content_types_truncated,omitempty"`
	ToolCalls             int      `json:"tool_calls,omitempty"`
	ToolResults           int      `json:"tool_results,omitempty"`
	EncryptedContentBytes int      `json:"encrypted_content_bytes,omitempty"`
	SignatureBytes        int      `json:"signature_bytes,omitempty"`
}

// SummarizeRequestShape extracts a bounded structural view of supported
// Messages, Chat Completions, or Responses request JSON. Unknown labels are
// collapsed to "other" so clients cannot smuggle arbitrary strings into logs.
func SummarizeRequestShape(body []byte, operation string) RequestShapeSummary {
	shape := RequestShapeSummary{Protocol: requestShapeProtocol(operation), PayloadFormat: "unknown"}
	if len(body) == 0 {
		return shape
	}
	if len(body) > maxRequestShapeBodyBytes {
		shape.Truncated = true
		return shape
	}

	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil || root == nil {
		shape.InvalidJSON = true
		return shape
	}
	if tools := rawArrayLength(root["tools"]); tools > 0 {
		shape.ToolDefinitions = tools
	}
	if system, ok := root["system"]; ok {
		shape.appendItem(summarizeRequestItem(system, "system", ""))
	}
	if instructions, ok := root["instructions"]; ok {
		shape.appendItem(summarizeRequestItem(instructions, "", "instructions"))
	}
	if messages, ok := root["messages"]; ok {
		if shape.Protocol == "chat" {
			shape.PayloadFormat = "chat"
		} else {
			shape.PayloadFormat = "messages"
		}
		var rawMessages []json.RawMessage
		if json.Unmarshal(messages, &rawMessages) != nil {
			shape.InvalidJSON = true
		} else {
			shape.appendRawItems(rawMessages)
		}
	} else if input, ok := root["input"]; ok {
		shape.PayloadFormat = "responses"
		var rawItems []json.RawMessage
		if json.Unmarshal(input, &rawItems) == nil {
			shape.appendRawItems(rawItems)
		} else {
			var text string
			if json.Unmarshal(input, &text) == nil {
				shape.appendItem(RequestShapeItem{Role: "user", Type: "text_input", ContentTypes: []string{"text"}})
			} else {
				shape.InvalidJSON = true
			}
		}
	}
	return shape
}

func requestShapeProtocol(operation string) string {
	switch strings.ToLower(strings.TrimSpace(operation)) {
	case "chat":
		return "chat"
	case "messages", "anthropic":
		return "messages"
	case "responses":
		return "responses"
	default:
		return "other"
	}
}

func (s *RequestShapeSummary) appendRawItems(rawItems []json.RawMessage) {
	if s == nil {
		return
	}
	limit := min(len(rawItems), maxRequestShapeItems+1)
	for _, raw := range rawItems[:limit] {
		if len(s.Items) >= maxRequestShapeItems {
			s.Truncated = true
			return
		}
		s.appendItem(summarizeRequestItem(raw, "", ""))
	}
	if len(rawItems) > maxRequestShapeItems {
		s.Truncated = true
	}
}

func (s *RequestShapeSummary) appendItem(item RequestShapeItem) {
	if s == nil {
		return
	}
	if len(s.Items) >= maxRequestShapeItems {
		s.Truncated = true
		return
	}
	s.Items = append(s.Items, item)
}

func summarizeRequestItem(raw json.RawMessage, defaultRole, defaultType string) RequestShapeItem {
	item := RequestShapeItem{Role: defaultRole, Type: defaultType}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil || object == nil {
		var text string
		if json.Unmarshal(raw, &text) == nil {
			if item.Type == "" && item.Role == "" {
				item.Type = "text_input"
				item.Role = "user"
			}
			if item.Type != "" {
				item.Type = safeRequestItemType(item.Type)
			}
			if item.Role != "" {
				item.Role = safeRequestRole(item.Role)
			}
			if text != "" {
				item.addContentType("text")
			}
			return item
		}
		var blocks []json.RawMessage
		if json.Unmarshal(raw, &blocks) == nil && item.Role != "" {
			item.Role = safeRequestRole(item.Role)
			collectRequestContentShape(&item, raw)
			return item
		}
		item.Type = "other"
		return item
	}
	if role := rawString(object["role"]); role != "" {
		item.Role = safeRequestRole(role)
	} else if item.Role != "" {
		item.Role = safeRequestRole(item.Role)
	}
	if kind := rawString(object["type"]); kind != "" {
		item.Type = safeRequestItemType(kind)
	} else if item.Type != "" {
		item.Type = safeRequestItemType(item.Type)
	}
	item.EncryptedContentBytes = rawStringLength(object["encrypted_content"])
	item.SignatureBytes = rawStringLength(object["signature"])
	if content, ok := object["content"]; ok {
		collectRequestContentShape(&item, content)
	}
	if summary, ok := object["summary"]; ok {
		collectRequestContentShape(&item, summary)
	}
	if calls, ok := object["tool_calls"]; ok {
		item.ToolCalls = rawArrayLength(calls)
		item.addContentType("tool_call")
	}
	if _, ok := object["function_call"]; ok {
		item.ToolCalls = min(item.ToolCalls+1, maxRequestShapeToolCount)
		item.addContentType("tool_call")
	}
	switch item.Type {
	case "function_call", "custom_tool_call", "computer_call", "shell_call", "local_shell_call", "mcp_call":
		item.ToolCalls = min(item.ToolCalls+1, maxRequestShapeToolCount)
	case "function_call_output", "custom_tool_call_output", "computer_call_output", "shell_call_output", "mcp_call_output":
		item.ToolResults = min(item.ToolResults+1, maxRequestShapeToolCount)
	}
	if item.Role == "tool" {
		item.ToolResults = min(item.ToolResults+1, maxRequestShapeToolCount)
	}
	return item
}

func collectRequestContentShape(item *RequestShapeItem, raw json.RawMessage) {
	if item == nil || len(raw) == 0 || string(raw) == "null" {
		return
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		if text != "" {
			item.addContentType("text")
		}
		return
	}
	var blocks []json.RawMessage
	if json.Unmarshal(raw, &blocks) == nil {
		limit := min(len(blocks), maxRequestShapeContentTypes)
		for _, block := range blocks[:limit] {
			collectRequestBlockShape(item, block)
		}
		if len(blocks) > limit {
			item.ContentTypesTruncated = true
		}
		return
	}
	collectRequestBlockShape(item, raw)
}

func collectRequestBlockShape(item *RequestShapeItem, raw json.RawMessage) {
	var block map[string]json.RawMessage
	if json.Unmarshal(raw, &block) != nil || block == nil {
		item.addContentType("other")
		return
	}
	kind := rawString(block["type"])
	if kind == "" {
		kind = "other"
	}
	item.addContentType(safeRequestContentType(kind))
	item.EncryptedContentBytes = max(item.EncryptedContentBytes, rawStringLength(block["encrypted_content"]))
	item.SignatureBytes = max(item.SignatureBytes, rawStringLength(block["signature"]))
	switch safeRequestContentType(kind) {
	case "tool_use", "tool_call":
		item.ToolCalls = min(item.ToolCalls+1, maxRequestShapeToolCount)
	case "tool_result", "tool_call_output":
		item.ToolResults = min(item.ToolResults+1, maxRequestShapeToolCount)
	}
}

func (item *RequestShapeItem) addContentType(kind string) {
	if item == nil || kind == "" {
		return
	}
	if len(item.ContentTypes) >= maxRequestShapeContentTypes {
		item.ContentTypesTruncated = true
		return
	}
	item.ContentTypes = append(item.ContentTypes, kind)
}

func safeRequestRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "developer", "user", "assistant", "tool":
		return strings.ToLower(strings.TrimSpace(role))
	default:
		return "other"
	}
}

func safeRequestItemType(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "message", "reasoning", "function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output", "computer_call", "computer_call_output", "shell_call", "shell_call_output", "local_shell_call", "mcp_call", "mcp_call_output", "web_search_call", "web_search_call_output", "image_generation_call", "code_interpreter_call", "text_input", "instructions":
		return strings.ToLower(strings.TrimSpace(kind))
	default:
		return "other"
	}
}

func safeRequestContentType(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "text", "input_text", "output_text", "summary_text", "refusal", "tool_use", "tool_result", "tool_call", "tool_call_output", "image", "input_image", "output_image", "document", "file", "audio", "input_audio", "output_audio", "thinking", "redacted_thinking", "reasoning":
		return strings.ToLower(strings.TrimSpace(kind))
	default:
		return "other"
	}
}

func rawString(raw json.RawMessage) string {
	var value string
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil {
		return ""
	}
	return value
}

func rawStringLength(raw json.RawMessage) int {
	return len(rawString(raw))
}

func rawArrayLength(raw json.RawMessage) int {
	var values []json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &values) != nil {
		return 0
	}
	return min(len(values), maxRequestShapeToolCount)
}
