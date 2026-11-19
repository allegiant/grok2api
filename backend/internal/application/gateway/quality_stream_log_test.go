package gateway

import (
	"bytes"
	"log/slog"
	"testing"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/domain/audit"
	modeldomain "github.com/chenyme/grok2api/backend/internal/domain/model"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
)

func TestQualityShapeSignalsFrom(t *testing.T) {
	shape := provider.RequestShapeSummary{
		Items: []provider.RequestShapeItem{
			{Role: "assistant", ContentTypes: []string{"thinking", "tool_use"}, SignatureBytes: 892, ToolCalls: 1},
			{Type: "reasoning", EncryptedContentBytes: 892},
			{Role: "user", ContentTypes: []string{"text"}},
		},
	}
	got := qualityShapeSignalsFrom(shape)
	if got.items != 3 || got.reasoning != 2 || got.encrypted != 1 || got.signatures != 1 {
		t.Fatalf("signals = %+v", got)
	}
}

func TestLogQualityStreamObservedIncludesDeliver(t *testing.T) {
	var buf bytes.Buffer
	service := &Service{logger: slog.New(slog.NewJSONHandler(&buf, nil))}
	history := &provider.ResponsesHistoryIntegritySummary{
		Comparable: true, EncryptedReasoningContentMatch: true, InputItemOrderMatch: true,
		ToolCallReferencesMatch: true, IncomingReasoningItems: 4, UpstreamReasoningItems: 4,
		IncomingEncryptedReasoningItems: 4, UpstreamEncryptedReasoningItems: 4, XSearchInjected: true,
	}
	service.logQualityStreamObserved("req-1", 7, modeldomain.Route{PublicID: "grok", UpstreamModel: "grok-4", Provider: accountdomain.ProviderBuild}, audit.OperationMessages, QualityDeliver, QualityActionDeliver, 1, Usage{OutputTokens: 12, ReasoningTokens: 30}, qualityStreamDiagnostic{HasThinking: true, UsageReported: true}, provider.RequestShapeSummary{ThinkingType: "adaptive"}, provider.RequestShapeSummary{ReasoningEffort: "xhigh", XSearchToolPresent: true}, provider.RawResponsesStreamSummary{ReasoningItemEvents: 1, ReasoningTokens: 30, Terminal: true, UsageReported: true}, history)
	line := buf.String()
	for _, want := range []string{`"msg":"quality_stream_observed"`, `"verdict":"deliver"`, `"reasoning_tokens":30`, `"history_encrypted_match":true`, `"raw_reasoning_items":1`, `"incoming_thinking_type":"adaptive"`} {
		if !bytes.Contains([]byte(line), []byte(want)) {
			t.Fatalf("log missing %s: %s", want, line)
		}
	}
}
