package gateway

import (
	"github.com/chenyme/grok2api/backend/internal/domain/audit"
	modeldomain "github.com/chenyme/grok2api/backend/internal/domain/model"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
)

type qualityShapeSignals struct {
	items      int
	reasoning  int
	encrypted  int
	signatures int
}

func qualityShapeSignalsFrom(shape provider.RequestShapeSummary) qualityShapeSignals {
	signals := qualityShapeSignals{items: len(shape.Items)}
	for _, item := range shape.Items {
		if item.Type == "reasoning" || requestShapeHasContent(item, "thinking") || requestShapeHasContent(item, "redacted_thinking") {
			signals.reasoning++
		}
		if item.EncryptedContentBytes > 0 {
			signals.encrypted++
		}
		if item.SignatureBytes > 0 {
			signals.signatures++
		}
	}
	return signals
}

func requestShapeHasContent(item provider.RequestShapeItem, want string) bool {
	for _, value := range item.ContentTypes {
		if value == want {
			return true
		}
	}
	return false
}

// logQualityStreamObserved records one compact line for every quality hold
// decision, including delivers. Full request shapes stay on the withhold path.
// Counts taken from a truncated shape are partial; history_* fields are complete
// only when the Responses integrity compare ran.
func (s *Service) logQualityStreamObserved(
	requestID string,
	accountID uint64,
	route modeldomain.Route,
	operation audit.Operation,
	verdict QualityVerdict,
	action QualityRetryAction,
	attempt int,
	usage Usage,
	stream qualityStreamDiagnostic,
	incoming, upstream provider.RequestShapeSummary,
	raw provider.RawResponsesStreamSummary,
	history *provider.ResponsesHistoryIntegritySummary,
	continuation qualityRequestContinuationInfo,
) {
	if s == nil || s.logger == nil {
		return
	}
	incomingSignals := qualityShapeSignalsFrom(incoming)
	upstreamSignals := qualityShapeSignalsFrom(upstream)
	comparable := false
	encryptedMatch := false
	orderMatch := false
	toolMatch := false
	historyIncomingReasoning := 0
	historyUpstreamReasoning := 0
	historyIncomingEncrypted := 0
	historyUpstreamEncrypted := 0
	xSearch := upstream.XSearchToolPresent
	if history != nil {
		comparable = history.Comparable
		encryptedMatch = history.EncryptedReasoningContentMatch
		orderMatch = history.InputItemOrderMatch
		toolMatch = history.ToolCallReferencesMatch
		historyIncomingReasoning = history.IncomingReasoningItems
		historyUpstreamReasoning = history.UpstreamReasoningItems
		historyIncomingEncrypted = history.IncomingEncryptedReasoningItems
		historyUpstreamEncrypted = history.UpstreamEncryptedReasoningItems
		if history.XSearchInjected {
			xSearch = true
		}
	}
	s.logger.Info("quality_stream_observed",
		"request_id", requestID,
		"account_id", accountID,
		"provider", route.Provider,
		"public_model", route.PublicID,
		"upstream_model", route.UpstreamModel,
		"protocol", qualityProtocolForOperation(operation),
		"verdict", string(verdict),
		"action", string(action),
		"quality_attempt", attempt,
		"output_tokens", usage.OutputTokens,
		"reasoning_tokens", usage.ReasoningTokens,
		"has_thinking", stream.HasThinking,
		"reasoning_started", stream.ReasoningStarted,
		"encrypted_content_valid", stream.EncryptedContentValid,
		"usage_reported", stream.UsageReported,
		"terminal", stream.Terminal,
		"tail_item_type", continuation.TailItemType,
		"tail_item_role", continuation.TailItemRole,
		"tool_continuation", continuation.IsToolContinuation,
		"tool_loop_context", continuation.HasToolLoopContext,
		"incoming_items", incomingSignals.items,
		"incoming_reasoning_items", incomingSignals.reasoning,
		"incoming_encrypted_reasoning_items", incomingSignals.encrypted,
		"incoming_signature_items", incomingSignals.signatures,
		"incoming_truncated", incoming.Truncated,
		"incoming_thinking_type", incoming.ThinkingType,
		"upstream_items", upstreamSignals.items,
		"upstream_reasoning_items", upstreamSignals.reasoning,
		"upstream_encrypted_reasoning_items", upstreamSignals.encrypted,
		"upstream_truncated", upstream.Truncated,
		"reasoning_effort", upstream.ReasoningEffort,
		"history_comparable", comparable,
		"history_encrypted_match", encryptedMatch,
		"history_order_match", orderMatch,
		"history_tool_match", toolMatch,
		"history_incoming_reasoning_items", historyIncomingReasoning,
		"history_upstream_reasoning_items", historyUpstreamReasoning,
		"history_incoming_encrypted_reasoning_items", historyIncomingEncrypted,
		"history_upstream_encrypted_reasoning_items", historyUpstreamEncrypted,
		"x_search_injected", xSearch,
		"raw_reasoning_items", raw.ReasoningItemEvents,
		"raw_reasoning_text_events", raw.ReasoningTextEvents,
		"raw_function_calls", raw.FunctionCallItemEvents,
		"raw_output_text_events", raw.OutputTextEvents,
		"raw_terminal", raw.Terminal,
		"raw_usage_reported", raw.UsageReported,
		"raw_reasoning_tokens", raw.ReasoningTokens,
	)
}
