package provider

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
)

const (
	maxRequestShapeBodyBytes        = 8 << 20
	maxRequestShapeItems            = 24
	maxRequestShapeContentTypes     = 12
	maxRequestShapeToolCount        = 10000
	maxResponsesHistoryCompareItems = 4096
)

// RequestShapeSummary contains only protocol labels, bounded counts, and
// encrypted payload lengths. It must never contain values from request content.
type RequestShapeSummary struct {
	Protocol               string             `json:"protocol"`
	PayloadFormat          string             `json:"payload_format"`
	PayloadBytes           int                `json:"payload_bytes,omitempty"`
	Items                  []RequestShapeItem `json:"items,omitempty"`
	ToolDefinitions        int                `json:"tool_definitions,omitempty"`
	XSearchToolPresent     bool               `json:"x_search_tool_present,omitempty"`
	ReasoningEffort        string             `json:"reasoning_effort,omitempty"`
	OutputConfigEffort     string             `json:"output_config_effort,omitempty"`
	ReasoningSummary       string             `json:"reasoning_summary,omitempty"`
	ThinkingType           string             `json:"thinking_type,omitempty"`
	ThinkingBudgetTokens   *int64             `json:"thinking_budget_tokens,omitempty"`
	ToolChoice             string             `json:"tool_choice,omitempty"`
	ParallelToolCalls      *bool              `json:"parallel_tool_calls,omitempty"`
	OutputTokenLimit       *int64             `json:"output_token_limit,omitempty"`
	OutputTokenLimitSource string             `json:"output_token_limit_source,omitempty"`
	PreviousResponseIDSeen bool               `json:"previous_response_id_seen"`
	Truncated              bool               `json:"truncated,omitempty"`
	InvalidJSON            bool               `json:"invalid_json,omitempty"`
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

// ResponsesHistoryIntegritySummary records only counts and equality results;
// it never retains request text, tool identifiers, arguments, or ciphertext.
type ResponsesHistoryIntegritySummary struct {
	Comparable                      bool `json:"comparable"`
	Truncated                       bool `json:"truncated,omitempty"`
	IncomingInputItems              int  `json:"incoming_input_items,omitempty"`
	UpstreamInputItems              int  `json:"upstream_input_items,omitempty"`
	InputItemOrderMatch             bool `json:"input_item_order_match"`
	IncomingReasoningItems          int  `json:"incoming_reasoning_items,omitempty"`
	UpstreamReasoningItems          int  `json:"upstream_reasoning_items,omitempty"`
	IncomingEncryptedReasoningItems int  `json:"incoming_encrypted_reasoning_items,omitempty"`
	UpstreamEncryptedReasoningItems int  `json:"upstream_encrypted_reasoning_items,omitempty"`
	EncryptedReasoningContentMatch  bool `json:"encrypted_reasoning_content_match"`
	IncomingToolCallItems           int  `json:"incoming_tool_call_items,omitempty"`
	UpstreamToolCallItems           int  `json:"upstream_tool_call_items,omitempty"`
	IncomingToolCallOutputItems     int  `json:"incoming_tool_call_output_items,omitempty"`
	UpstreamToolCallOutputItems     int  `json:"upstream_tool_call_output_items,omitempty"`
	ToolCallReferencesMatch         bool `json:"tool_call_references_match"`
	PreviousResponseIDSeen          bool `json:"previous_response_id_seen"`
	PreviousResponseIDMatch         bool `json:"previous_response_id_match"`
	XSearchInjected                 bool `json:"x_search_injected"`
}

// SummarizeRequestShape extracts a bounded structural view of supported
// Messages, Chat Completions, or Responses request JSON. Unknown labels are
// collapsed to "other" so clients cannot smuggle arbitrary strings into logs.
func SummarizeRequestShape(body []byte, operation string) RequestShapeSummary {
	shape := RequestShapeSummary{Protocol: requestShapeProtocol(operation), PayloadFormat: "unknown", PayloadBytes: len(body)}
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
	summarizeRequestControls(&shape, root)
	if tools := rawArrayLength(root["tools"]); tools > 0 {
		shape.ToolDefinitions = tools
		shape.XSearchToolPresent = requestHasXSearchTool(root["tools"])
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

// CompareResponsesRequestHistory compares protected history fields in memory and
// returns only counts and booleans suitable for structured logging.
func CompareResponsesRequestHistory(incoming, upstream []byte, operation string) *ResponsesHistoryIntegritySummary {
	if requestShapeProtocol(operation) != "responses" {
		return nil
	}
	summary := &ResponsesHistoryIntegritySummary{}
	if len(incoming) > maxRequestShapeBodyBytes || len(upstream) > maxRequestShapeBodyBytes {
		summary.Truncated = true
		return summary
	}
	incomingRoot, incomingValid := decodeResponsesRequestRoot(incoming)
	upstreamRoot, upstreamValid := decodeResponsesRequestRoot(upstream)
	if incomingValid && upstreamValid {
		incomingID, incomingIDSeen, incomingIDValid := responsesPreviousID(incomingRoot)
		upstreamID, upstreamIDSeen, upstreamIDValid := responsesPreviousID(upstreamRoot)
		summary.PreviousResponseIDSeen = incomingIDSeen || upstreamIDSeen
		summary.PreviousResponseIDMatch = incomingIDSeen == upstreamIDSeen && incomingIDValid && upstreamIDValid && incomingID == upstreamID
		summary.XSearchInjected = !requestHasXSearchTool(incomingRoot["tools"]) && requestHasXSearchTool(upstreamRoot["tools"])
	}

	incomingHistory, incomingComparable, incomingTruncated := decodeResponsesHistory(incomingRoot, incomingValid)
	upstreamHistory, upstreamComparable, upstreamTruncated := decodeResponsesHistory(upstreamRoot, upstreamValid)
	summary.Truncated = incomingTruncated || upstreamTruncated
	if incomingComparable {
		summary.IncomingInputItems = incomingHistory.itemCount
		summary.IncomingReasoningItems = incomingHistory.reasoningItems
		summary.IncomingEncryptedReasoningItems = incomingHistory.encryptedReasoningItems
		summary.IncomingToolCallItems = incomingHistory.toolCallItems
		summary.IncomingToolCallOutputItems = incomingHistory.toolCallOutputItems
	}
	if upstreamComparable {
		summary.UpstreamInputItems = upstreamHistory.itemCount
		summary.UpstreamReasoningItems = upstreamHistory.reasoningItems
		summary.UpstreamEncryptedReasoningItems = upstreamHistory.encryptedReasoningItems
		summary.UpstreamToolCallItems = upstreamHistory.toolCallItems
		summary.UpstreamToolCallOutputItems = upstreamHistory.toolCallOutputItems
	}
	if !incomingComparable || !upstreamComparable {
		return summary
	}

	summary.Comparable = true
	summary.InputItemOrderMatch = equalResponsesHistoryItems(incomingHistory.items, upstreamHistory.items)
	summary.EncryptedReasoningContentMatch = equalEncryptedReasoning(incomingHistory.reasoning, upstreamHistory.reasoning)
	summary.ToolCallReferencesMatch = equalResponsesToolReferences(incomingHistory.toolReferences, upstreamHistory.toolReferences)
	return summary
}

type responsesHistoryItem struct {
	typeName string
	role     string
}

type encryptedReasoningValue struct {
	present bool
	valid   bool
	content string
}

type responsesToolReference struct {
	typeName string
	id       string
	callID   string
}

type responsesRequestHistory struct {
	itemCount               int
	items                   []responsesHistoryItem
	reasoningItems          int
	encryptedReasoningItems int
	reasoning               []encryptedReasoningValue
	toolCallItems           int
	toolCallOutputItems     int
	toolReferences          []responsesToolReference
}

func decodeResponsesRequestRoot(body []byte) (map[string]json.RawMessage, bool) {
	var root map[string]json.RawMessage
	if len(body) == 0 || json.Unmarshal(body, &root) != nil || root == nil {
		return nil, false
	}
	return root, true
}

func responsesPreviousID(root map[string]json.RawMessage) (string, bool, bool) {
	raw, seen := root["previous_response_id"]
	if !seen {
		return "", false, true
	}
	var id string
	if json.Unmarshal(raw, &id) != nil {
		return "", true, false
	}
	return id, true, true
}

func decodeResponsesHistory(root map[string]json.RawMessage, rootValid bool) (responsesRequestHistory, bool, bool) {
	if !rootValid {
		return responsesRequestHistory{}, false, false
	}
	rawInput := bytes.TrimSpace(root["input"])
	if len(rawInput) == 0 || rawInput[0] != '[' {
		return responsesRequestHistory{}, false, false
	}
	decoder := json.NewDecoder(bytes.NewReader(rawInput))
	if _, err := decoder.Token(); err != nil {
		return responsesRequestHistory{}, false, false
	}
	history := responsesRequestHistory{}
	for decoder.More() {
		if history.itemCount >= maxResponsesHistoryCompareItems {
			return history, false, true
		}
		var rawItem json.RawMessage
		if decoder.Decode(&rawItem) != nil {
			return responsesRequestHistory{}, false, false
		}
		history.itemCount++
		var item map[string]json.RawMessage
		if json.Unmarshal(rawItem, &item) != nil || item == nil {
			history.items = append(history.items, responsesHistoryItem{})
			continue
		}
		typeName := rawString(item["type"])
		history.items = append(history.items, responsesHistoryItem{typeName: typeName, role: rawString(item["role"])})
		if typeName == "reasoning" {
			history.reasoningItems++
			rawEncrypted, present := item["encrypted_content"]
			value := encryptedReasoningValue{present: present, valid: !present}
			if present {
				history.encryptedReasoningItems++
				value.valid = json.Unmarshal(rawEncrypted, &value.content) == nil
			}
			history.reasoning = append(history.reasoning, value)
		}
		if isResponsesToolCallType(typeName) {
			history.toolCallItems++
			history.toolReferences = append(history.toolReferences, responsesToolReference{
				typeName: typeName,
				id:       rawString(item["id"]),
				callID:   rawString(item["call_id"]),
			})
			continue
		}
		if isResponsesToolOutputType(typeName) {
			history.toolCallOutputItems++
			history.toolReferences = append(history.toolReferences, responsesToolReference{
				typeName: typeName,
				callID:   rawString(item["call_id"]),
			})
		}
	}
	if _, err := decoder.Token(); err != nil {
		return responsesRequestHistory{}, false, false
	}
	return history, true, false
}

func isResponsesToolCallType(typeName string) bool {
	switch typeName {
	case "function_call", "custom_tool_call", "computer_call", "shell_call", "local_shell_call", "mcp_call":
		return true
	default:
		return false
	}
}

func isResponsesToolOutputType(typeName string) bool {
	switch typeName {
	case "function_call_output", "custom_tool_call_output", "computer_call_output", "shell_call_output", "mcp_call_output":
		return true
	default:
		return false
	}
}

func equalResponsesHistoryItems(left, right []responsesHistoryItem) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func equalEncryptedReasoning(left, right []encryptedReasoningValue) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if !left[i].valid || !right[i].valid || left[i] != right[i] {
			return false
		}
	}
	return true
}

func equalResponsesToolReferences(left, right []responsesToolReference) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
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

func summarizeRequestControls(shape *RequestShapeSummary, root map[string]json.RawMessage) {
	if shape == nil {
		return
	}
	var reasoning, outputConfig, thinking map[string]json.RawMessage
	_ = json.Unmarshal(root["reasoning"], &reasoning)
	_ = json.Unmarshal(root["output_config"], &outputConfig)
	_ = json.Unmarshal(root["thinking"], &thinking)

	shape.ReasoningEffort = safeRequestOption(rawString(reasoning["effort"]), "none", "minimal", "low", "medium", "high", "xhigh", "max")
	if shape.ReasoningEffort == "" {
		shape.ReasoningEffort = safeRequestOption(rawString(root["reasoning_effort"]), "none", "minimal", "low", "medium", "high", "xhigh", "max")
	}
	shape.OutputConfigEffort = safeRequestOption(rawString(outputConfig["effort"]), "none", "minimal", "low", "medium", "high", "xhigh", "max")
	shape.ReasoningSummary = safeRequestOption(rawString(reasoning["summary"]), "auto", "concise", "detailed")
	shape.ThinkingType = safeRequestOption(rawString(thinking["type"]), "enabled", "adaptive", "disabled")
	if shape.ThinkingType == "" {
		shape.ThinkingType = safeRequestOption(rawString(root["thinking"]), "enabled", "adaptive", "disabled")
	}
	if budget, ok := rawInt64(thinking["budget_tokens"]); ok {
		shape.ThinkingBudgetTokens = &budget
	}
	shape.ToolChoice = safeRequestToolChoice(root["tool_choice"])
	if raw, ok := root["parallel_tool_calls"]; ok {
		var parallel bool
		if json.Unmarshal(raw, &parallel) == nil {
			shape.ParallelToolCalls = &parallel
		}
	}
	for _, key := range []string{"max_output_tokens", "max_completion_tokens", "max_tokens"} {
		if value, ok := rawInt64(root[key]); ok {
			shape.OutputTokenLimit = &value
			shape.OutputTokenLimitSource = key
			break
		}
	}
	_, shape.PreviousResponseIDSeen = root["previous_response_id"]
}

func safeRequestOption(value string, allowed ...string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return ""
	}
	for _, candidate := range allowed {
		if value == candidate {
			return value
		}
	}
	return "other"
}

func safeRequestToolChoice(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	if value := rawString(raw); value != "" {
		return safeRequestOption(value, "auto", "none", "required", "any")
	}
	var choice map[string]json.RawMessage
	if json.Unmarshal(raw, &choice) != nil || choice == nil {
		return "other"
	}
	return safeRequestOption(rawString(choice["type"]), "auto", "none", "required", "any", "function", "tool", "custom")
}

func rawInt64(raw json.RawMessage) (int64, bool) {
	var value int64
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil {
		return 0, false
	}
	return value, true
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

func requestHasXSearchTool(raw json.RawMessage) bool {
	var tools []json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &tools) != nil {
		return false
	}
	for _, rawTool := range tools {
		var tool struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(rawTool, &tool) == nil && tool.Type == "x_search" {
			return true
		}
	}
	return false
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
