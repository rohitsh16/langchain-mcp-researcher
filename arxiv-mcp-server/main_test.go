package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

const sampleAtomXML = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Search</title>
  <entry>
    <id>http://arxiv.org/abs/2312.11805v1</id>
    <title>Agentic Context Management: A Survey</title>
    <summary>This paper reviews state-of-the-art context window management in autonomous agents.</summary>
    <published>2023-12-18T18:00:00Z</published>
    <updated>2023-12-19T10:00:00Z</updated>
    <author><name>Alice Johnson</name></author>
    <author><name>Bob Smith</name></author>
    <link href="http://arxiv.org/abs/2312.11805v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2312.11805v1" rel="related" title="pdf" type="application/pdf"/>
    <category term="cs.AI"/>
    <category term="cs.SE"/>
  </entry>
</feed>`

func TestParseAtomFeed(t *testing.T) {
	papers, err := parseAtomFeed([]byte(sampleAtomXML))
	if err != nil {
		t.Fatalf("unexpected error parsing feed: %v", err)
	}
	if len(papers) != 1 {
		t.Fatalf("expected 1 paper, got %d", len(papers))
	}
	p := papers[0]
	if p.ID != "2312.11805v1" {
		t.Errorf("expected ID 2312.11805v1, got %s", p.ID)
	}
	if p.Title != "Agentic Context Management: A Survey" {
		t.Errorf("title mismatch: %s", p.Title)
	}
	if len(p.Authors) != 2 || p.Authors[0] != "Alice Johnson" {
		t.Errorf("authors mismatch: %v", p.Authors)
	}
	if p.PDFURL != "http://arxiv.org/pdf/2312.11805v1" {
		t.Errorf("pdf url mismatch: %s", p.PDFURL)
	}
	if len(p.Categories) != 2 || p.Categories[0] != "cs.AI" {
		t.Errorf("categories mismatch: %v", p.Categories)
	}
}

func TestMCPToolsList(t *testing.T) {
	tools := getTools()
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	toolNames := map[string]bool{}
	for _, tool := range tools {
		toolNames[tool.Name] = true
	}
	if !toolNames["search_arxiv"] || !toolNames["get_arxiv_paper"] {
		t.Errorf("expected search_arxiv and get_arxiv_paper, got %v", toolNames)
	}
}

func TestMCPProtocolRoundtrip(t *testing.T) {
	input := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}` + "\n"

	inBuf := bytes.NewBufferString(input)
	outBuf := &bytes.Buffer{}

	if err := Run(inBuf, outBuf); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(outBuf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 response lines, got %d. Output: %s", len(lines), outBuf.String())
	}

	var initResp JSONRPCResponse
	if err := json.Unmarshal([]byte(lines[0]), &initResp); err != nil {
		t.Fatalf("failed to unmarshal init response: %v", err)
	}
	if initResp.Error != nil {
		t.Fatalf("init response had error: %v", initResp.Error)
	}

	var toolsResp JSONRPCResponse
	if err := json.Unmarshal([]byte(lines[1]), &toolsResp); err != nil {
		t.Fatalf("failed to unmarshal tools response: %v", err)
	}
	if toolsResp.Error != nil {
		t.Fatalf("tools response had error: %v", toolsResp.Error)
	}
}
