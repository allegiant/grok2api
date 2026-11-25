package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

func TestThinkingRecheckHeldOnlyAfterMissingThinkingCooldown(t *testing.T) {
	now := time.Date(2026, 11, 25, 12, 0, 0, 0, time.UTC)
	future := now.Add(time.Hour)
	past := now.Add(-time.Minute)
	if thinkingRecheckHeld(account.Credential{LastError: account.LastErrorMissingThinking, CooldownUntil: &future}, now) {
		t.Fatal("active cooldown is already excluded by the cooldown check")
	}
	if !thinkingRecheckHeld(account.Credential{LastError: account.LastErrorMissingThinking, CooldownUntil: &past}, now) {
		t.Fatal("expired missing-thinking cooldown must stay out of the pool")
	}
	if thinkingRecheckHeld(account.Credential{LastError: "upstream timeout", CooldownUntil: &past}, now) {
		t.Fatal("ordinary failure cooldown should return when the timer elapses")
	}
	if !thinkingRecheckHeld(account.Credential{LastError: account.LastErrorMissingThinking}, now) {
		t.Fatal("missing-thinking strike without a future cooldown must stay out")
	}
}

func TestThinkingRecheckStreamRequiresReasoning(t *testing.T) {
	thinking, err := thinkingRecheckStreamHasThinking(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}],\"usage\":{\"completion_tokens\":5}}\n\ndata: [DONE]\n"))
	if err != nil || thinking {
		t.Fatalf("thinking=%v err=%v", thinking, err)
	}
	thinking, err = thinkingRecheckStreamHasThinking(strings.NewReader("data: {\"choices\":[{\"delta\":{\"reasoning\":\"plan\"}}]}\n\ndata: [DONE]\n"))
	if err != nil || !thinking {
		t.Fatalf("thinking=%v err=%v", thinking, err)
	}
}
