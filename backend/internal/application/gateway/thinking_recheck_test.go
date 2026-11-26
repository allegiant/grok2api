package gateway

import (
	"strings"
	"testing"
	"time"

	egressapp "github.com/chenyme/grok2api/backend/internal/application/egress"
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
	if thinkingRecheckHeld(account.Credential{LastError: account.LastErrorMissingThinking}, now) {
		t.Fatal("a cleared cooldown must return the account even while the strike remains")
	}
}

func TestThinkingRecheckUsesProfileMarkerAndThinking(t *testing.T) {
	profile := egressapp.QualityProbeInput{Expected: "QUALITY_OK", MatchMode: egressapp.MatchLastLine, RequireThinking: true}
	plain, err := observeThinkingRecheck(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hello\\nQUALITY_OK\"}}],\"usage\":{\"completion_tokens\":80,\"completion_tokens_details\":{\"reasoning_tokens\":0}}}\n\ndata: [DONE]\n"))
	if err != nil || thinkingRecheckPassed(plain, profile) {
		t.Fatalf("marker without reasoning released: %#v err=%v", plain, err)
	}
	thought, err := observeThinkingRecheck(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"1. A B C D\\nQUALITY_OK\"}}],\"usage\":{\"completion_tokens\":90,\"completion_tokens_details\":{\"reasoning_tokens\":40}}}\n\ndata: [DONE]\n"))
	if err != nil || !thinkingRecheckPassed(thought, profile) {
		t.Fatalf("profile pass = %#v err=%v", thought, err)
	}
}
