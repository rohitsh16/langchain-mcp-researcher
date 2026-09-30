package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestSearchArticles(t *testing.T) {
	resp := searchArticles("mcp", 5)
	if len(resp.Articles) == 0 {
		t.Fatalf("expected articles matching 'mcp', got 0")
	}
	found := false
	for _, a := range resp.Articles {
		if strings.Contains(strings.ToLower(a.Title), "model context protocol") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected Model Context Protocol article, got: %v", resp.Articles)
	}
}

func TestGetPattern(t *testing.T) {
	p, err := getPattern("mcp")
	if err != nil {
		t.Fatalf("unexpected error getting pattern: %v", err)
	}
	if !strings.Contains(p.Pattern, "Model Context Protocol") {
		t.Errorf("pattern mismatch: %s", p.Pattern)
	}
}

func TestSWEMCPToolsList(t *testing.T) {
	tools := getTools()
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Name] = true
	}
	if !names["search_swe_articles"] || !names["get_swe_pattern"] {
		t.Errorf("missing tools, got: %v", names)
	}
}

func TestSWEMCPProtocolRoundtrip(t *testing.T) {
	input := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_swe_pattern","arguments":{"pattern":"mcp"}}}` + "\n"

	inBuf := bytes.NewBufferString(input)
	outBuf := &bytes.Buffer{}

	if err := Run(inBuf, outBuf); err != nil {
		t.Fatalf("Run error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(outBuf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}

	var callResp JSONRPCResponse
	if err := json.Unmarshal([]byte(lines[1]), &callResp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if callResp.Error != nil {
		t.Errorf("call error: %v", callResp.Error)
	}
}
