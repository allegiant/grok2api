package provider

import (
	"bytes"
	"encoding/json"
	"io"
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

const (
	maxRawResponsesEventBytes = 256 << 10
	maxRawResponsesTypes      = 32
)

// RawResponsesStreamSummary describes raw upstream events before conversion to
// Chat or Anthropic. It contains only whitelisted types, counts, and token usage.
type RawResponsesStreamSummary struct {
	Events                     []string `json:"events,omitempty"`
	EventCount                 int      `json:"event_count,omitempty"`
	EventsTruncated            bool     `json:"events_truncated,omitempty"`
	OutputItemTypes            []string `json:"output_item_types,omitempty"`
	OutputItemsTruncated       bool     `json:"output_items_truncated,omitempty"`
	ReasoningItemEvents        int      `json:"reasoning_item_events,omitempty"`
	ReasoningTextEvents        int      `json:"reasoning_text_events,omitempty"`
	ReasoningTextParts         int      `json:"reasoning_text_parts,omitempty"`
	FunctionCallItemEvents     int      `json:"function_call_item_events,omitempty"`
	FunctionCallOutputEvents   int      `json:"function_call_output_events,omitempty"`
	FunctionCallArgumentEvents int      `json:"function_call_argument_events,omitempty"`
	OutputTextEvents           int      `json:"output_text_events,omitempty"`
	ReasoningPayloadBytes      int      `json:"reasoning_payload_bytes,omitempty"`
	UsageReported              bool     `json:"usage_reported"`
	OutputTokens               int64    `json:"output_tokens,omitempty"`
	ReasoningTokens            int64    `json:"reasoning_tokens,omitempty"`
	Terminal                   bool     `json:"terminal"`
	Truncated                  bool     `json:"truncated"`
	MalformedEvents            int      `json:"malformed_events,omitempty"`
}

type rawResponsesEventBuffer struct {
	line            []byte
	event           string
	data            []byte
	discardingLine  bool
	discardingFrame bool
}

// ObserveUpstreamResponses wraps a raw Responses SSE body with a content-free
// observer. Bytes are passed through unchanged to the protocol converter.
func (d *QualityRequestDiagnostic) ObserveUpstreamResponses(body io.ReadCloser) io.ReadCloser {
	if d == nil || body == nil {
		return body
	}
	return &qualityDiagnosticResponsesBody{ReadCloser: body, diagnostic: d}
}

type qualityDiagnosticResponsesBody struct {
	io.ReadCloser
	diagnostic *QualityRequestDiagnostic
}

func (b *qualityDiagnosticResponsesBody) Read(dst []byte) (int, error) {
	n, err := b.ReadCloser.Read(dst)
	if n > 0 || err == io.EOF {
		b.diagnostic.observeRawResponsesChunk(dst[:n], err == io.EOF)
	}
	return n, err
}

func (d *QualityRequestDiagnostic) observeRawResponsesChunk(chunk []byte, eof bool) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	buffer := &d.rawEventBuffer
	for _, value := range chunk {
		if buffer.discardingLine {
			if value == '\n' {
				buffer.discardingLine = false
			}
			continue
		}
		if value == '\n' {
			line := bytes.TrimSuffix(buffer.line, []byte{'\r'})
			d.processRawResponsesLine(line)
			buffer.line = buffer.line[:0]
			continue
		}
		if len(buffer.line) >= maxRawResponsesEventBytes {
			buffer.line = buffer.line[:0]
			buffer.discardingLine = true
			buffer.discardingFrame = true
			d.rawResponse.Truncated = true
			continue
		}
		buffer.line = append(buffer.line, value)
	}
	if eof {
		if buffer.discardingLine {
			buffer.discardingLine = false
			buffer.discardingFrame = true
		} else if len(buffer.line) > 0 {
			d.processRawResponsesLine(bytes.TrimSuffix(buffer.line, []byte{'\r'}))
			buffer.line = buffer.line[:0]
		}
		d.finishRawResponsesFrame()
	}
}

func (d *QualityRequestDiagnostic) processRawResponsesLine(line []byte) {
	buffer := &d.rawEventBuffer
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		d.finishRawResponsesFrame()
		return
	}
	if buffer.discardingFrame {
		return
	}
	if bytes.HasPrefix(trimmed, []byte("event:")) {
		buffer.event = safeRawResponsesEventType(strings.TrimSpace(string(bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("event:"))))))
		return
	}
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return
	}
	payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
	separator := 0
	if len(buffer.data) > 0 {
		separator = 1
	}
	if len(buffer.data)+separator+len(payload) > maxRawResponsesEventBytes {
		buffer.data = buffer.data[:0]
		buffer.discardingFrame = true
		d.rawResponse.Truncated = true
		return
	}
	if separator > 0 {
		buffer.data = append(buffer.data, '\n')
	}
	buffer.data = append(buffer.data, payload...)
}

func (d *QualityRequestDiagnostic) finishRawResponsesFrame() {
	buffer := &d.rawEventBuffer
	if !buffer.discardingFrame && len(buffer.data) > 0 {
		d.observeRawResponsesPayload(buffer.event, buffer.data)
	}
	buffer.event = ""
	buffer.data = buffer.data[:0]
	buffer.discardingFrame = false
}

func (d *QualityRequestDiagnostic) observeRawResponsesPayload(eventName string, payload []byte) {
	if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
		d.rawResponse.EventCount++
		d.rawResponse.Terminal = true
		appendRawDiagnosticType(&d.rawResponse.Events, &d.rawResponse.EventsTruncated, "done")
		return
	}
	var root map[string]json.RawMessage
	if json.Unmarshal(payload, &root) != nil || root == nil {
		d.rawResponse.MalformedEvents++
		return
	}
	eventType := rawString(root["type"])
	if eventType == "" {
		eventType = eventName
	}
	if eventType == "" {
		d.rawResponse.MalformedEvents++
		return
	}
	eventType = safeRawResponsesEventType(eventType)
	d.rawResponse.EventCount++
	appendRawDiagnosticType(&d.rawResponse.Events, &d.rawResponse.EventsTruncated, eventType)
	switch eventType {
	case "response.completed", "response.incomplete", "response.failed", "error":
		d.rawResponse.Terminal = true
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		if rawString(root["delta"]) != "" {
			d.rawResponse.ReasoningTextEvents++
		}
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		d.rawResponse.ReasoningTextParts += countRawReasoningTextParts(root["part"])
	case "response.output_text.delta":
		if rawString(root["delta"]) != "" {
			d.rawResponse.OutputTextEvents++
		}
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta", "response.mcp_call_arguments.delta":
		d.rawResponse.FunctionCallArgumentEvents++
	case "response.output_item.added", "response.output_item.done":
		d.observeRawResponsesItem(root["item"])
	}
	if usage, ok := root["usage"]; ok {
		d.observeRawResponsesUsage(usage)
	}
	if responseRaw, ok := root["response"]; ok {
		var response struct {
			Output []json.RawMessage `json:"output"`
			Usage  json.RawMessage   `json:"usage"`
		}
		if json.Unmarshal(responseRaw, &response) == nil {
			if len(response.Usage) > 0 {
				d.observeRawResponsesUsage(response.Usage)
			}
			for _, item := range response.Output {
				d.observeRawResponsesItem(item)
			}
		}
	}
}

func (d *QualityRequestDiagnostic) observeRawResponsesItem(raw json.RawMessage) {
	var item struct {
		Type             string          `json:"type"`
		EncryptedContent string          `json:"encrypted_content"`
		Content          json.RawMessage `json:"content"`
		Summary          json.RawMessage `json:"summary"`
	}
	if json.Unmarshal(raw, &item) != nil {
		d.rawResponse.MalformedEvents++
		return
	}
	itemType := safeRawResponsesItemType(item.Type)
	appendRawDiagnosticType(&d.rawResponse.OutputItemTypes, &d.rawResponse.OutputItemsTruncated, itemType)
	switch itemType {
	case "reasoning":
		d.rawResponse.ReasoningItemEvents++
		d.rawResponse.ReasoningPayloadBytes = max(d.rawResponse.ReasoningPayloadBytes, len(item.EncryptedContent))
		d.rawResponse.ReasoningTextParts += countRawReasoningTextParts(item.Content)
		d.rawResponse.ReasoningTextParts += countRawReasoningTextParts(item.Summary)
	case "function_call", "custom_tool_call", "computer_call", "shell_call", "local_shell_call", "mcp_call":
		d.rawResponse.FunctionCallItemEvents++
	case "function_call_output", "custom_tool_call_output", "computer_call_output", "shell_call_output", "mcp_call_output":
		d.rawResponse.FunctionCallOutputEvents++
	}
}

func countRawReasoningTextParts(raw json.RawMessage) int {
	if len(raw) == 0 {
		return 0
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		count := 0
		for _, part := range parts {
			if (part.Type == "reasoning_text" || part.Type == "summary_text") && part.Text != "" {
				count++
			}
		}
		return count
	}
	var part struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &part) == nil && (part.Type == "reasoning_text" || part.Type == "summary_text") && part.Text != "" {
		return 1
	}
	return 0
}

func (d *QualityRequestDiagnostic) observeRawResponsesUsage(raw json.RawMessage) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return
	}
	var usage struct {
		OutputTokens        int64 `json:"output_tokens"`
		CompletionTokens    int64 `json:"completion_tokens"`
		ReasoningTokens     int64 `json:"reasoning_tokens"`
		OutputTokensDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
		CompletionTokenDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	}
	if json.Unmarshal(raw, &usage) != nil {
		d.rawResponse.MalformedEvents++
		return
	}
	d.rawResponse.UsageReported = true
	outputTokens := usage.OutputTokens
	if outputTokens == 0 {
		outputTokens = usage.CompletionTokens
	}
	reasoningTokens := usage.OutputTokensDetails.ReasoningTokens
	if reasoningTokens == 0 {
		reasoningTokens = usage.CompletionTokenDetails.ReasoningTokens
	}
	if reasoningTokens == 0 {
		reasoningTokens = usage.ReasoningTokens
	}
	d.rawResponse.OutputTokens = outputTokens
	d.rawResponse.ReasoningTokens = reasoningTokens
}

func appendRawDiagnosticType(values *[]string, truncated *bool, value string) {
	if values == nil || truncated == nil {
		return
	}
	if len(*values) >= maxRawResponsesTypes {
		*truncated = true
		return
	}
	*values = append(*values, value)
}

func safeRawResponsesEventType(value string) string {
	switch value {
	case "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done", "response.content_part.added", "response.content_part.done", "response.output_text.delta", "response.output_text.done", "response.reasoning_text.delta", "response.reasoning_text.done", "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done", "response.reasoning_summary_part.added", "response.reasoning_summary_part.done", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done", "response.mcp_call_arguments.delta", "response.mcp_call_arguments.done", "response.completed", "response.incomplete", "response.failed", "error":
		return value
	default:
		return "other"
	}
}

func safeRawResponsesItemType(value string) string {
	switch value {
	case "message", "reasoning", "function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output", "computer_call", "computer_call_output", "shell_call", "shell_call_output", "local_shell_call", "mcp_call", "mcp_call_output", "web_search_call", "web_search_call_output", "image_generation_call", "code_interpreter_call":
		return value
	default:
		return "other"
	}
}

func cloneRequestShape(shape RequestShapeSummary) RequestShapeSummary {
	shape.Items = append([]RequestShapeItem(nil), shape.Items...)
	for i := range shape.Items {
		shape.Items[i].ContentTypes = append([]string(nil), shape.Items[i].ContentTypes...)
	}
	return shape
}
