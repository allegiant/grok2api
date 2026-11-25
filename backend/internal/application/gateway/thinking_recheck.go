package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	egressapp "github.com/chenyme/grok2api/backend/internal/application/egress"
	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/domain/audit"
	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/qualityguard"
	"github.com/chenyme/grok2api/backend/internal/infra/security"
)

const thinkingRecheckDefaultModel = "grok-4.7"

// RecheckDueMissingThinking probes accounts whose missing-thinking cooldown has
// elapsed. Ordinary selection keeps them out until a probe shows thinking again.
func (s *Service) RecheckDueMissingThinking(ctx context.Context, accounts []accountdomain.Credential, input egressapp.QualityProbeInput) {
	for _, credential := range accounts {
		if ctx.Err() != nil {
			return
		}
		if err := s.recheckMissingThinking(ctx, credential, input); err != nil {
			s.logger.Warn("thinking_recheck_failed", "account_id", credential.ID, "provider", credential.Provider, "error", err)
		}
	}
}

func (s *Service) recheckMissingThinking(ctx context.Context, credential accountdomain.Credential, input egressapp.QualityProbeInput) error {
	key, err := s.clientKeys.Get(ctx, input.ClientKeyID)
	if err != nil {
		return fmt.Errorf("读取质量探测 Client Key: %w", err)
	}
	if !key.IsAvailable(time.Now().UTC()) {
		return fmt.Errorf("质量探测 Client Key 已禁用或过期")
	}
	model := strings.TrimSpace(input.Model)
	if model == "" {
		model = thinkingRecheckDefaultModel
	}
	prompt := strings.TrimSpace(input.Prompt)
	if prompt == "" {
		prompt = qualityguard.ProbePrompt
	}
	maxTokens := input.MaxOutputTokens
	if maxTokens <= 0 {
		maxTokens = egressapp.DefaultQualityProbeMaxOutputTokens
	}
	publicModel, ok := qualityProbeBuildPublicModel(model)
	if !ok {
		return fmt.Errorf("质量复查模型必须属于 Grok Build")
	}
	requestIDPart, err := security.NewOpaqueToken(12)
	if err != nil {
		return err
	}
	body, err := json.Marshal(map[string]any{
		"model":          model,
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
		"stream":         true,
		"stream_options": map[string]bool{"include_usage": true},
		"max_tokens":     maxTokens,
	})
	if err != nil {
		return err
	}
	result, err := s.CreateChatCompletion(infraegress.WithQualityProbe(ctx), Input{
		RequestID: "thinking_recheck_" + requestIDPart, ClientKey: key, PublicModel: publicModel, Body: body,
		Streaming: true, Operation: audit.OperationChat,
		ForcedEgressNodeID: credential.EgressNodeID, ForcedAccountID: credential.ID, thinkingRecheck: true,
	})
	if err != nil {
		return err
	}
	defer result.Body.Close()
	if result.StatusCode < http.StatusOK || result.StatusCode >= http.StatusMultipleChoices {
		payload, _ := io.ReadAll(io.LimitReader(result.Body, 32<<10))
		return fmt.Errorf("质量复查上游返回 %d: %s", result.StatusCode, strings.TrimSpace(string(payload)))
	}
	thinking, err := thinkingRecheckStreamHasThinking(result.Body)
	if err != nil {
		return err
	}
	if thinking {
		if err := s.selector.clearMissingThinkingStrike(ctx, credential); err != nil {
			return err
		}
		s.logger.Info("thinking_recheck_released", "account_id", credential.ID, "provider", credential.Provider)
		return nil
	}
	cooldown := s.qualityRetryConfig().AccountCooldown
	if cooldown <= 0 {
		cooldown = defaultMissingThinkingCooldown
	}
	until := time.Now().UTC().Add(cooldown)
	if err := s.selector.deferMissingThinkingRecheck(ctx, credential, until); err != nil {
		return err
	}
	s.logger.Info("thinking_recheck_held", "account_id", credential.ID, "provider", credential.Provider, "cooldown", cooldown)
	return nil
}

func thinkingRecheckStreamHasThinking(body io.Reader) (bool, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	totalBytes := 0
	terminal := false
	thinking := false
	for scanner.Scan() {
		line := scanner.Bytes()
		totalBytes += len(line) + 1
		if totalBytes > qualityProbeMaxStreamBytes {
			return false, fmt.Errorf("质量复查响应超过 %d MiB", qualityProbeMaxStreamBytes>>20)
		}
		line = []byte(strings.TrimSpace(string(line)))
		if bytes.Equal(line, []byte(": grok2api-reasoning-start")) {
			thinking = true
			continue
		}
		if !strings.HasPrefix(string(line), "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(string(line), "data:"))
		if payload == "[DONE]" {
			terminal = true
			break
		}
		var event qualityProbeChatEvent
		if json.Unmarshal([]byte(payload), &event) != nil {
			continue
		}
		if event.Usage != nil && event.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
			thinking = true
		}
		for _, choice := range event.Choices {
			delta := choice.Delta
			if delta.Reasoning != "" || delta.ReasoningContent != "" || delta.ThinkingContent != "" {
				thinking = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return false, fmt.Errorf("读取质量复查流: %w", err)
	}
	if !terminal {
		return false, fmt.Errorf("质量复查流未正常结束")
	}
	return thinking, nil
}
