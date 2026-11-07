package gateway

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/chenyme/grok2api/backend/internal/domain/audit"
	"github.com/chenyme/grok2api/backend/internal/pkg/reasoningreplay"
)

const (
	qualityProtocolChat                = "chat"
	qualityProtocolResponses           = "responses"
	qualityProtocolAnthropic           = "anthropic"
	qualityReasoningSSEComment         = ": grok2api-reasoning-start"
	qualityReasoningDeltaSSEComment    = ": grok2api-reasoning-delta"
	qualityReasoningEvidenceSSEComment = ": grok2api-reasoning-evidence"
	qualityHoldMaxBufferBytes          = 8 << 20
)

type qualityScanState struct {
	protocol                        string
	pending                         []byte
	hasThinking                     bool
	reasoningStarted                bool
	visibleRunes                    int
	aggregateRunes                  int
	semanticOutput                  bool
	reasoningTokens                 int64
	outputTokens                    int64
	encryptedBytes                  int
	validEncryptedContent           bool
	minEncryptedBytes               int
	encryptedBytesPerReasoningToken int
	usage                           Usage
	responseID                      string
	terminal                        bool
	holdExpired                     bool
	firstVisibleAt                  time.Time
}

type qualityReadResult struct {
	data []byte
	err  error
}

// qualityReadPump is the sole reader of the upstream body. It lets the hold
// timer win while an upstream Read is blocked, then remains the continuation
// reader for the response body after the held prefix is replayed.
type qualityReadPump struct {
	source    io.ReadCloser
	results   chan qualityReadResult
	done      chan struct{}
	closeOnce sync.Once
	pending   []byte
	finalErr  error
}

func newQualityReadPump(source io.ReadCloser) *qualityReadPump {
	pump := &qualityReadPump{
		source:  source,
		results: make(chan qualityReadResult),
		done:    make(chan struct{}),
	}
	go pump.run()
	return pump
}

func (p *qualityReadPump) run() {
	defer close(p.results)
	buf := make([]byte, 4096)
	for {
		n, err := p.source.Read(buf)
		if n == 0 && err == nil {
			continue
		}
		result := qualityReadResult{err: err}
		if n > 0 {
			result.data = append([]byte(nil), buf[:n]...)
		}
		select {
		case p.results <- result:
		case <-p.done:
			return
		}
		if err != nil {
			return
		}
	}
}

func (p *qualityReadPump) Read(dst []byte) (int, error) {
	for len(p.pending) == 0 {
		if p.finalErr != nil {
			return 0, p.finalErr
		}
		result, ok := <-p.results
		if !ok {
			p.finalErr = io.EOF
			return 0, io.EOF
		}
		p.pending = result.data
		p.finalErr = result.err
		if len(p.pending) == 0 && p.finalErr != nil {
			return 0, p.finalErr
		}
	}
	n := copy(dst, p.pending)
	p.pending = p.pending[n:]
	return n, nil
}

func (p *qualityReadPump) DiscardReplayCapture() {
	reasoningreplay.DiscardCapture(p.source)
}

func (p *qualityReadPump) Close() error {
	var err error
	p.closeOnce.Do(func() {
		close(p.done)
		err = p.source.Close()
	})
	return err
}

func qualityProtocolForOperation(operation audit.Operation) string {
	switch operation {
	case audit.OperationChat:
		return qualityProtocolChat
	case audit.OperationMessages:
		return qualityProtocolAnthropic
	default:
		return qualityProtocolResponses
	}
}

func (s *qualityScanState) signals() QualityStreamSignals {
	// Visible tokens come only from streamed content deltas (chat content /
	// responses output_text / message items). Do not lift from
	// usage.output − usage.reasoning: chat often reports completion_tokens
	// that still include reasoning, which made dumps look like long answers.
	visibleRunes := max(s.visibleRunes, s.aggregateRunes)
	visible := int64((visibleRunes + 3) / 4)
	output := s.outputTokens
	if s.usage.Reported && s.usage.OutputTokens > output {
		output = s.usage.OutputTokens
	}
	// Usage.reasoning_tokens is not proof of thinking. 降智 accounts report
	// hundreds of reasoning tokens on completed while the stream never sent
	// reasoning_text / reasoning_summary deltas (TUI shows no thoughts).
	// A non-empty encrypted_content stub is also not thinking until it meets
	// the ciphertext floor (default 256 bytes, or 4 bytes per reasoning token).
	reasoningTokens := max(s.reasoningTokens, s.usage.ReasoningTokens)
	floor := encryptedThinkingFloor(s.minEncryptedBytes, s.encryptedBytesPerReasoningToken, reasoningTokens)
	if s.validEncryptedContent && reasoningTokens > 0 && s.encryptedBytes > 0 {
		if s.minEncryptedBytes <= defaultMinEncryptedBytes {
			floor = int64(s.encryptedBytes)
		} else if int64(s.encryptedBytes) >= int64(s.minEncryptedBytes) {
			floor = int64(s.minEncryptedBytes)
		}
	}
	hasThinking := s.hasThinking || int64(s.encryptedBytes) >= floor
	firstVisible := !s.firstVisibleAt.IsZero()
	var flushMS int64
	if firstVisible {
		flushMS = time.Since(s.firstVisibleAt).Milliseconds()
	}
	return QualityStreamSignals{
		HasThinking:       hasThinking,
		HasReasoningDelta: s.hasThinking,
		ReasoningStarted:  s.reasoningStarted || hasThinking,
		VisibleTokens:     visible,
		ReasoningTokens:   reasoningTokens,
		OutputTokens:      output,
		EncryptedBytes:    s.encryptedBytes,
		EncryptedFloor:    floor,
		UsageReported:     s.usage.Reported,
		FirstVisible:      firstVisible,
		VisibleFlushMS:    flushMS,
		Terminal:          s.terminal,
		HoldExpired:       s.holdExpired,
	}
}

// ObserveQualityChunk feeds one SSE chunk into the hold classifier state.
// This is the shipped scanner used by peekQualityStream.
func ObserveQualityChunk(state *qualityScanState, chunk []byte) {
	if state == nil || len(chunk) == 0 {
		return
	}
	state.pending = append(state.pending, chunk...)
	for {
		index := bytes.IndexByte(state.pending, '\n')
		if index < 0 {
			if len(state.pending) > qualityHoldMaxBufferBytes {
				state.pending = nil
			}
			return
		}
		line := bytes.TrimSpace(state.pending[:index])
		state.pending = state.pending[index+1:]
		if len(line) == 0 {
			continue
		}
		if bytes.Equal(line, []byte(qualityReasoningSSEComment)) {
			// Timing stub only. 降智 still emits this, then usage.reasoning_tokens=0.
			state.reasoningStarted = true
			continue
		}
		if bytes.Equal(line, []byte(qualityReasoningDeltaSSEComment)) {
			state.reasoningStarted = true
			state.hasThinking = true
			continue
		}
		if bytes.HasPrefix(line, []byte(qualityReasoningEvidenceSSEComment)) {
			// Protocol converters cannot expose encrypted_content in every public
			// JSON contract. This internal SSE comment preserves its byte length so
			// short stubs cannot bypass the ciphertext floor.
			state.reasoningStarted = true
			fields := strings.Fields(strings.TrimSpace(strings.TrimPrefix(string(line), qualityReasoningEvidenceSSEComment)))
			if len(fields) > 0 {
				if count, err := strconv.Atoi(fields[0]); err == nil && count > state.encryptedBytes {
					state.encryptedBytes = count
				}
				for _, flag := range fields[1:] {
					if strings.EqualFold(flag, "valid") {
						state.validEncryptedContent = true
					}
				}
			}
			continue
		}
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if bytes.Equal(payload, []byte("[DONE]")) {
			state.terminal = true
			continue
		}
		observeQualityPayload(state, payload)
	}
}

func observeQualityPayload(state *qualityScanState, payload []byte) {
	switch state.protocol {
	case qualityProtocolChat:
		observeQualityChat(state, payload)
	case qualityProtocolAnthropic:
		observeQualityAnthropic(state, payload)
	default:
		observeQualityResponses(state, payload)
	}
}

func observeQualityChat(state *qualityScanState, payload []byte) {
	var event struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Delta struct {
				Content          string `json:"content"`
				Reasoning        string `json:"reasoning"`
				ReasoningContent string `json:"reasoning_content"`
				ThinkingContent  string `json:"thinking_content"`
				ToolCalls        []any  `json:"tool_calls"`
				FunctionCall     any    `json:"function_call"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens            int64 `json:"prompt_tokens"`
			CompletionTokens        int64 `json:"completion_tokens"`
			TotalTokens             int64 `json:"total_tokens"`
			CompletionTokensDetails struct {
				ReasoningTokens int64 `json:"reasoning_tokens"`
			} `json:"completion_tokens_details"`
		} `json:"usage"`
	}
	if json.Unmarshal(payload, &event) != nil {
		return
	}
	if state.responseID == "" {
		state.responseID = event.ID
	}
	if event.Usage != nil {
		state.usage.Reported = true
		state.usage.InputTokens = event.Usage.PromptTokens
		state.usage.OutputTokens = event.Usage.CompletionTokens
		state.usage.ReasoningTokens = event.Usage.CompletionTokensDetails.ReasoningTokens
		state.usage.TotalTokens = event.Usage.TotalTokens
		state.usage.ResponseModel = event.Model
		state.outputTokens = event.Usage.CompletionTokens
		state.reasoningTokens = event.Usage.CompletionTokensDetails.ReasoningTokens
	}
	for _, choice := range event.Choices {
		delta := choice.Delta
		if strings.TrimSpace(delta.Reasoning) != "" || strings.TrimSpace(delta.ReasoningContent) != "" || strings.TrimSpace(delta.ThinkingContent) != "" {
			state.hasThinking = true
		}
		if delta.Content != "" {
			noteVisibleContent(state, delta.Content)
		}
		if len(delta.ToolCalls) > 0 || delta.FunctionCall != nil {
			state.semanticOutput = true
		}
		if choice.FinishReason != "" {
			state.terminal = true
		}
	}
}

type qualityResponsesOutputItem struct {
	ID               string          `json:"id"`
	Type             string          `json:"type"`
	EncryptedContent string          `json:"encrypted_content"`
	Arguments        string          `json:"arguments"`
	Input            string          `json:"input"`
	Command          string          `json:"command"`
	Content          json.RawMessage `json:"content"`
	Summary          json.RawMessage `json:"summary"`
}

type qualityResponsesUsage struct {
	OutputTokens            int64 `json:"output_tokens"`
	OutputTokensCamel       int64 `json:"outputTokens"`
	CompletionTokens        int64 `json:"completion_tokens"`
	CompletionTokensCamel   int64 `json:"completionTokens"`
	InputTokens             int64 `json:"input_tokens"`
	InputTokensCamel        int64 `json:"inputTokens"`
	PromptTokens            int64 `json:"prompt_tokens"`
	PromptTokensCamel       int64 `json:"promptTokens"`
	TotalTokens             int64 `json:"total_tokens"`
	TotalTokensCamel        int64 `json:"totalTokens"`
	ReasoningTokens         int64 `json:"reasoning_tokens"`
	ReasoningTokensCamel    int64 `json:"reasoningTokens"`
	ThinkingTokens          int64 `json:"thinking_tokens"`
	ThinkingTokensCamel     int64 `json:"thinkingTokens"`
	NumServerSideToolsUsed  int64 `json:"num_server_side_tools_used"`
	NumServerSideToolsCamel int64 `json:"numServerSideToolsUsed"`
	OutputTokensDetails     struct {
		ReasoningTokens      int64 `json:"reasoning_tokens"`
		ReasoningTokensCamel int64 `json:"reasoningTokens"`
		ThinkingTokens       int64 `json:"thinking_tokens"`
		ThinkingTokensCamel  int64 `json:"thinkingTokens"`
	} `json:"output_tokens_details"`
	OutputTokensDetailsCamel struct {
		ReasoningTokens      int64 `json:"reasoning_tokens"`
		ReasoningTokensCamel int64 `json:"reasoningTokens"`
		ThinkingTokens       int64 `json:"thinking_tokens"`
		ThinkingTokensCamel  int64 `json:"thinkingTokens"`
	} `json:"outputTokensDetails"`
	CompletionTokensDetails struct {
		ReasoningTokens      int64 `json:"reasoning_tokens"`
		ReasoningTokensCamel int64 `json:"reasoningTokens"`
	} `json:"completion_tokens_details"`
}

func (u *qualityResponsesUsage) applyTo(state *qualityScanState, model string) {
	if u == nil || state == nil {
		return
	}
	input := firstPositiveInt64(u.InputTokens, u.InputTokensCamel, u.PromptTokens, u.PromptTokensCamel)
	output := firstPositiveInt64(u.OutputTokens, u.OutputTokensCamel, u.CompletionTokens, u.CompletionTokensCamel)
	total := firstPositiveInt64(u.TotalTokens, u.TotalTokensCamel)
	if total == 0 && (input > 0 || output > 0) {
		total = input + output
	}
	reasoning := firstPositiveInt64(
		u.OutputTokensDetails.ReasoningTokens,
		u.OutputTokensDetails.ReasoningTokensCamel,
		u.OutputTokensDetails.ThinkingTokens,
		u.OutputTokensDetails.ThinkingTokensCamel,
		u.OutputTokensDetailsCamel.ReasoningTokens,
		u.OutputTokensDetailsCamel.ReasoningTokensCamel,
		u.OutputTokensDetailsCamel.ThinkingTokens,
		u.OutputTokensDetailsCamel.ThinkingTokensCamel,
		u.CompletionTokensDetails.ReasoningTokens,
		u.CompletionTokensDetails.ReasoningTokensCamel,
		u.ReasoningTokens,
		u.ReasoningTokensCamel,
		u.ThinkingTokens,
		u.ThinkingTokensCamel,
	)
	serverTools := firstPositiveInt64(u.NumServerSideToolsUsed, u.NumServerSideToolsCamel)
	state.usage.Reported = true
	state.usage.InputTokens = input
	state.usage.OutputTokens = output
	state.usage.ReasoningTokens = reasoning
	state.usage.TotalTokens = total
	state.usage.NumServerSideToolsUsed = serverTools
	if model != "" {
		state.usage.ResponseModel = model
	}
	state.outputTokens = output
	state.reasoningTokens = reasoning
}

func firstPositiveInt64(values ...int64) int64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func noteEncryptedBytes(state *qualityScanState, blob string) {
	trimmed := strings.TrimSpace(blob)
	if n := len(trimmed); n > state.encryptedBytes {
		state.encryptedBytes = n
	}
	if isValidGrokEncryptedContent(trimmed) {
		state.validEncryptedContent = true
	}
}

func isValidGrokEncryptedContent(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > qualityHoldMaxBufferBytes {
		return false
	}
	if strings.HasPrefix(value, "gAAAA") || strings.Contains(value, "=") {
		return false
	}
	decoded, err := base64.RawStdEncoding.DecodeString(value)
	if err != nil || len(decoded) < 50 {
		return false
	}
	var counts [256]int
	for _, b := range decoded {
		counts[b]++
	}
	size := float64(len(decoded))
	entropy := 0.0
	for _, count := range counts {
		if count == 0 {
			continue
		}
		p := float64(count) / size
		entropy -= p * math.Log2(p)
	}
	maxSymbols := min(len(decoded), 256)
	if maxSymbols <= 1 {
		return false
	}
	return entropy/math.Log2(float64(maxSymbols)) >= 0.85
}

func responsesReasoningPartsHaveText(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var plain string
	if json.Unmarshal(raw, &plain) == nil {
		return strings.TrimSpace(plain) != ""
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		for _, part := range parts {
			if strings.TrimSpace(part.Text) != "" {
				return true
			}
		}
	}
	return false
}

func noteResponsesReasoningItem(state *qualityScanState, item qualityResponsesOutputItem) {
	if !strings.EqualFold(strings.TrimSpace(item.Type), "reasoning") {
		return
	}
	if strings.TrimSpace(item.ID) != "" {
		state.reasoningStarted = true
	}
	if responsesReasoningPartsHaveText(item.Summary) || responsesReasoningPartsHaveText(item.Content) {
		state.hasThinking = true
	}
	noteEncryptedBytes(state, item.EncryptedContent)
}

func observeQualityResponses(state *qualityScanState, payload []byte) {
	var event struct {
		Type     string                 `json:"type"`
		Delta    json.RawMessage        `json:"delta"`
		Text     string                 `json:"text"`
		Part     json.RawMessage        `json:"part"`
		Item     json.RawMessage        `json:"item"`
		Usage    *qualityResponsesUsage `json:"usage"`
		Response *struct {
			ID     string                 `json:"id"`
			Model  string                 `json:"model"`
			Output []json.RawMessage      `json:"output"`
			Usage  *qualityResponsesUsage `json:"usage"`
		} `json:"response"`
	}
	if json.Unmarshal(payload, &event) != nil {
		return
	}
	deltaText := ""
	if len(event.Delta) > 0 {
		_ = json.Unmarshal(event.Delta, &deltaText)
	}
	switch event.Type {
	case "response.completed", "response.incomplete", "response.failed":
		state.terminal = true
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		if strings.TrimSpace(deltaText) != "" {
			state.hasThinking = true
		}
	case "response.reasoning_text.done", "response.reasoning_summary_text.done":
		if strings.TrimSpace(event.Text) != "" || strings.TrimSpace(deltaText) != "" {
			state.hasThinking = true
		}
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		if responsesReasoningPartsHaveText(event.Part) {
			state.hasThinking = true
		}
	case "response.output_item.added", "response.output_item.done":
		var item qualityResponsesOutputItem
		if len(event.Item) > 0 && json.Unmarshal(event.Item, &item) == nil {
			noteResponsesReasoningItem(state, item)
			state.aggregateRunes = max(state.aggregateRunes, observeQualityResponsesOutputItem(state, item))
		}
	case "response.output_text.delta":
		if deltaText != "" {
			noteVisibleContent(state, deltaText)
		}
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta", "response.mcp_call_arguments.delta":
		if deltaText != "" {
			state.semanticOutput = true
		}
	}
	if event.Usage != nil && (event.Response == nil || event.Response.Usage == nil) {
		event.Usage.applyTo(state, "")
	}
	if event.Response != nil {
		if state.responseID == "" {
			state.responseID = event.Response.ID
		}
		if event.Response.Usage != nil {
			event.Response.Usage.applyTo(state, event.Response.Model)
		}
		aggregateRunes := 0
		for _, rawItem := range event.Response.Output {
			var item qualityResponsesOutputItem
			if json.Unmarshal(rawItem, &item) != nil {
				continue
			}
			noteResponsesReasoningItem(state, item)
			aggregateRunes += observeQualityResponsesOutputItem(state, item)
		}
		state.aggregateRunes = max(state.aggregateRunes, aggregateRunes)
	}
}

func observeQualityResponsesOutputItem(state *qualityScanState, item qualityResponsesOutputItem) int {
	if state == nil {
		return 0
	}
	visibleRunes := 0
	switch item.Type {
	case "", "reasoning":
		return 0
	case "message":
		if len(item.Content) == 0 {
			return 0
		}
		var plainText string
		if json.Unmarshal(item.Content, &plainText) == nil {
			if plainText != "" {
				visibleRunes += utf8.RuneCountInString(plainText)
				state.semanticOutput = true
			}
			return visibleRunes
		}
		var parts []struct {
			Type    string `json:"type"`
			Text    string `json:"text"`
			Refusal string `json:"refusal"`
		}
		if json.Unmarshal(item.Content, &parts) == nil {
			for _, content := range parts {
				text := content.Text
				if text == "" {
					text = content.Refusal
				}
				if text != "" {
					visibleRunes += utf8.RuneCountInString(text)
					state.semanticOutput = true
					continue
				}
				if content.Type != "" && content.Type != "output_text" && content.Type != "refusal" {
					state.semanticOutput = true
				}
			}
		}
	default:
		// Function, shell, MCP and other call items are meaningful output even
		// when the provider omits usage and argument-delta events.
		state.semanticOutput = true
	}
	return visibleRunes
}

func observeQualityAnthropic(state *qualityScanState, payload []byte) {
	var event struct {
		Type         string `json:"type"`
		ContentBlock struct {
			Type string `json:"type"`
			Text string `json:"text"`
			Data string `json:"data"`
		} `json:"content_block"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			PartialJSON string `json:"partial_json"`
			Signature   string `json:"signature"`
		} `json:"delta"`
		Usage *struct {
			OutputTokens        int64 `json:"output_tokens"`
			OutputTokensDetails struct {
				ThinkingTokens int64 `json:"thinking_tokens"`
			} `json:"output_tokens_details"`
		} `json:"usage"`
	}
	if json.Unmarshal(payload, &event) != nil {
		return
	}
	switch event.Type {
	case "message_stop":
		state.terminal = true
	case "content_block_start":
		switch event.ContentBlock.Type {
		case "thinking":
			state.reasoningStarted = true
		case "redacted_thinking":
			state.reasoningStarted = true
			noteEncryptedBytes(state, event.ContentBlock.Data)
		case "text":
			if event.ContentBlock.Text != "" {
				noteVisibleContent(state, event.ContentBlock.Text)
				state.semanticOutput = true
			}
		case "":
		default:
			state.semanticOutput = true
		}
	case "content_block_delta":
		if event.Delta.Type == "thinking_delta" && strings.TrimSpace(event.Delta.Thinking) != "" {
			state.hasThinking = true
		}
		if event.Delta.Type == "signature_delta" && strings.TrimSpace(event.Delta.Signature) != "" {
			// Anthropic Messages represents Responses encrypted_content as a
			// signature delta. Length is judged against the ciphertext floor.
			state.reasoningStarted = true
			noteEncryptedBytes(state, event.Delta.Signature)
		}
		if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
			noteVisibleContent(state, event.Delta.Text)
		}
		if event.Delta.Type == "input_json_delta" && event.Delta.PartialJSON != "" {
			state.semanticOutput = true
		}
	}
	if event.Usage != nil {
		state.usage.Reported = true
		state.usage.OutputTokens = event.Usage.OutputTokens
		state.usage.ReasoningTokens = event.Usage.OutputTokensDetails.ThinkingTokens
		state.outputTokens = event.Usage.OutputTokens
		state.reasoningTokens = event.Usage.OutputTokensDetails.ThinkingTokens
	}
}

func noteVisibleContent(state *qualityScanState, text string) {
	if text == "" {
		return
	}
	if state.firstVisibleAt.IsZero() {
		state.firstVisibleAt = time.Now()
	}
	state.visibleRunes += utf8.RuneCountInString(text)
}

func peekQualityStream(ctx context.Context, body io.ReadCloser, protocol string, cfg QualityRetryRuntime) (io.ReadCloser, QualityVerdict, Usage, string, error) {
	cfg = normalizeQualityRetry(cfg)
	if body == nil {
		return io.NopCloser(bytes.NewReader(nil)), QualityWait, Usage{}, "", errQualityEmptyStream
	}
	pump := newQualityReadPump(body)
	state := qualityScanState{
		protocol:                        protocol,
		minEncryptedBytes:               cfg.MinEncryptedBytes,
		encryptedBytesPerReasoningToken: cfg.EncryptedBytesPerReasoningToken,
	}
	var held bytes.Buffer
	holdTimer := time.NewTimer(cfg.HoldTimeout)
	defer holdTimer.Stop()
	for {
		sig := state.signals()
		if verdict := ClassifyQualityHold(sig, cfg.MinOutputTokens); verdict != QualityWait {
			return newPrefixReplay(&held, pump), verdict, state.usage, state.responseID, nil
		}
		// A completed empty stream must rotate immediately. Waiting for idle
		// timeout after response.completed / [DONE] surfaces HTTP 200 with 0
		// tokens and makes Grok TUI retry for 50–120s.
		if sig.Terminal {
			return finishQualityPeek(&held, pump, &state, cfg)
		}

		select {
		case <-ctx.Done():
			reasoningreplay.DiscardCapture(pump)
			_ = pump.Close()
			return io.NopCloser(bytes.NewReader(held.Bytes())), QualityWait, state.usage, state.responseID, qualityPeekAbortError(ctx, ctx.Err())
		case <-holdTimer.C:
			state.holdExpired = true
			sig.HoldExpired = true
			if verdict := ClassifyQualityHold(sig, cfg.MinOutputTokens); verdict != QualityWait {
				return newPrefixReplay(&held, pump), verdict, state.usage, state.responseID, nil
			}
		case result, ok := <-pump.results:
			if !ok {
				return finishQualityPeek(&held, pump, &state, cfg)
			}
			if len(result.data) > 0 {
				if held.Len()+len(result.data) > qualityHoldMaxBufferBytes {
					_, _ = held.Write(result.data)
					return newPrefixReplay(&held, pump), QualityDeliver, state.usage, state.responseID, nil
				}
				_, _ = held.Write(result.data)
				ObserveQualityChunk(&state, result.data)
			}
			if result.err == io.EOF {
				return finishQualityPeek(&held, pump, &state, cfg)
			}
			if result.err != nil {
				reasoningreplay.DiscardCapture(pump)
				_ = pump.Close()
				return io.NopCloser(bytes.NewReader(held.Bytes())), QualityWait, state.usage, state.responseID, qualityPeekAbortError(ctx, result.err)
			}
		}
	}
}

func finishQualityPeek(held *bytes.Buffer, pump *qualityReadPump, state *qualityScanState, cfg QualityRetryRuntime) (io.ReadCloser, QualityVerdict, Usage, string, error) {
	if state == nil {
		return io.NopCloser(bytes.NewReader(nil)), QualityWait, Usage{}, "", errQualityEmptyStream
	}
	if len(state.pending) > 0 {
		// Process a final valid SSE data line even when the upstream omitted its
		// trailing newline.
		ObserveQualityChunk(state, []byte{'\n'})
	}
	state.terminal = true
	signals := state.signals()
	if !qualityHasValidThinking(signals) && signals.ReasoningTokens <= 0 && signals.OutputTokens <= 0 && signals.VisibleTokens <= 0 {
		if state.semanticOutput {
			if signals.ReasoningStarted || signals.EncryptedBytes > 0 {
				return newPrefixReplay(held, pump), QualityWithhold, state.usage, state.responseID, nil
			}
			return newPrefixReplay(held, pump), QualityDeliver, state.usage, state.responseID, nil
		}
		return newPrefixReplay(held, pump), QualityWait, state.usage, state.responseID, errQualityEmptyStream
	}
	return newPrefixReplay(held, pump), ClassifyQualityHold(signals, cfg.MinOutputTokens), state.usage, state.responseID, nil
}

func newPrefixReplay(held *bytes.Buffer, rest io.ReadCloser) io.ReadCloser {
	if rest == nil {
		rest = io.NopCloser(bytes.NewReader(nil))
	}
	if held == nil || held.Len() == 0 {
		return rest
	}
	return &replayReadCloser{Reader: io.MultiReader(bytes.NewReader(held.Bytes()), rest), source: rest}
}
