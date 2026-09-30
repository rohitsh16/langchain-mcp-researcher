package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestExtractSections(t *testing.T) {
	samplePaper := `
1. Abstract:
Context management in LLMs requires dynamic memory pruning and token allocation under strict budgets.

1. Introduction
Modern agents need durable cross-session context. We propose a 6-pass deterministic allocator.

4. Conclusion
Our experiments demonstrate that budget-bounded caching improves hit rates by 35%.

References
[1] Vaswani et al. Attention is all you need.
`

	sections := ExtractSections(samplePaper)
	if len(sections) == 0 {
		t.Fatalf("expected extracted sections, got empty map")
	}

	if abs, ok := sections["Abstract"]; !ok || !strings.Contains(abs, "Context management") {
		t.Errorf("expected Abstract section with 'Context management', got: %v", sections)
	}

	if intro, ok := sections["Introduction"]; !ok || !strings.Contains(intro, "Modern agents") {
		t.Errorf("expected Introduction section, got: %v", sections)
	}

	if conc, ok := sections["Conclusion"]; !ok || !strings.Contains(conc, "budget-bounded") {
		t.Errorf("expected Conclusion section, got: %v", sections)
	}
}

func TestPDFMCPToolsList(t *testing.T) {
	tools := getTools()
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Name] = true
	}
	if !names["read_pdf"] || !names["read_pdf_url"] {
		t.Errorf("missing expected tools, got %v", names)
	}
}

func TestPDFMCPProtocolRoundtrip(t *testing.T) {
	input := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}` + "\n"

	inBuf := bytes.NewBufferString(input)
	outBuf := &bytes.Buffer{}

	if err := Run(inBuf, outBuf); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(outBuf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 response lines, got %d", len(lines))
	}

	var initResp JSONRPCResponse
	if err := json.Unmarshal([]byte(lines[0]), &initResp); err != nil {
		t.Fatalf("failed to parse init resp: %v", err)
	}
	if initResp.Error != nil {
		t.Errorf("init had error: %v", initResp.Error)
	}

	var toolsResp JSONRPCResponse
	if err := json.Unmarshal([]byte(lines[1]), &toolsResp); err != nil {
		t.Fatalf("failed to parse tools resp: %v", err)
	}
	if toolsResp.Error != nil {
		t.Errorf("tools had error: %v", toolsResp.Error)
	}
}
