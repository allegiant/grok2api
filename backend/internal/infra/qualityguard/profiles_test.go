package qualityguard

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadActiveProbeProfile(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "profiles.json")
	raw := []byte(`{"version":1,"active_profile_id":"custom-1","profiles":{"custom-1":{"name":"多步推理诊断","prompt":"Solve the schedule.\nQUALITY_OK","expected_text":"QUALITY_OK","match_mode":"last_line","require_thinking":true,"max_output_tokens":256}}}`)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	profile, ok, err := LoadActiveProbeProfile(path)
	if err != nil || !ok || profile.Name != "多步推理诊断" || profile.Prompt != "Solve the schedule.\nQUALITY_OK" || !profile.RequireThinking || profile.MaxOutputTokens != 256 {
		t.Fatalf("profile=%#v ok=%v err=%v", profile, ok, err)
	}
	_, ok, err = LoadActiveProbeProfile(filepath.Join(directory, "missing.json"))
	if err != nil || ok {
		t.Fatalf("missing ok=%v err=%v", ok, err)
	}
}
