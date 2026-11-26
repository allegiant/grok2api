package qualityguard

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
)

// ActiveProbeProfile is the probe profile currently marked for use.
type ActiveProbeProfile struct {
	ID              string
	Name            string
	Prompt          string
	ExpectedText    string
	MatchMode       string
	RequireThinking bool
	MaxOutputTokens int
}

type probeProfileFile struct {
	ActiveProfileID string                      `json:"active_profile_id"`
	Profiles        map[string]probeProfileJSON `json:"profiles"`
}

type probeProfileJSON struct {
	Name            string `json:"name"`
	Prompt          string `json:"prompt"`
	ExpectedText    string `json:"expected_text"`
	MatchMode       string `json:"match_mode"`
	RequireThinking bool   `json:"require_thinking"`
	MaxOutputTokens int    `json:"max_output_tokens"`
}

// LoadActiveProbeProfile reads the quality-guard profile file and returns the
// profile selected as current. A missing path or file reports found=false.
func LoadActiveProbeProfile(path string) (ActiveProbeProfile, bool, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return ActiveProbeProfile{}, false, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return ActiveProbeProfile{}, false, nil
		}
		return ActiveProbeProfile{}, false, err
	}
	var file probeProfileFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return ActiveProbeProfile{}, false, err
	}
	id := strings.TrimSpace(file.ActiveProfileID)
	profile, ok := file.Profiles[id]
	if !ok || strings.TrimSpace(profile.Prompt) == "" {
		return ActiveProbeProfile{}, false, nil
	}
	return ActiveProbeProfile{
		ID: id, Name: strings.TrimSpace(profile.Name), Prompt: strings.TrimSpace(profile.Prompt),
		ExpectedText: strings.TrimSpace(profile.ExpectedText), MatchMode: strings.TrimSpace(profile.MatchMode),
		RequireThinking: profile.RequireThinking, MaxOutputTokens: profile.MaxOutputTokens,
	}, true, nil
}
