package web

import "testing"

func TestGatewayActivityIgnoresControlFrames(t *testing.T) {
	activity := newGatewayActivity()
	if activity.observeFrame([]byte(`{"event":{"type":"pong"}}`)) {
		t.Fatal("pong reset the idle window")
	}
	if !activity.observeFrame([]byte(`{"event":{"type":"response.output_text.delta","delta":"hello"}}`)) {
		t.Fatal("text delta did not count as activity")
	}
	if activity.observeFrame([]byte(`{"event":{"type":"response.output_text.delta","delta":"  "}}`)) {
		t.Fatal("blank delta counted as activity")
	}
}
