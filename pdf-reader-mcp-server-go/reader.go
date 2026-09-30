package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ledongthuc/pdf"
)

// PDFDocument holds extracted content and metadata.
type PDFDocument struct {
	TotalPages int               `json:"total_pages"`
	ReadPages  int               `json:"read_pages"`
	Text       string            `json:"text"`
	Sections   map[string]string `json:"sections,omitempty"`
}

var pdfHTTPClient = &http.Client{
	Timeout: 60 * time.Second,
}

// ReadPDF reads a local PDF file and extracts text up to maxPages.
func ReadPDF(path string, maxPages int) (*PDFDocument, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF %s: %w", path, err)
	}
	defer f.Close()

	numPages := r.NumPage()
	if maxPages <= 0 || maxPages > numPages {
		maxPages = numPages
	}

	var sb strings.Builder
	for i := 1; i <= maxPages; i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}
		text, err := p.GetPlainText(nil)
		if err != nil {
			continue
		}
		sb.WriteString(fmt.Sprintf("\n--- Page %d ---\n", i))
		sb.WriteString(cleanPDFText(text))
		sb.WriteString("\n")
	}

	fullText := strings.TrimSpace(sb.String())
	sections := ExtractSections(fullText)

	return &PDFDocument{
		TotalPages: numPages,
		ReadPages:  maxPages,
		Text:       fullText,
		Sections:   sections,
	}, nil
}

// ReadPDFURL downloads a PDF from a URL into a temporary file and extracts text.
func ReadPDFURL(url string, maxPages int) (*PDFDocument, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("invalid PDF URL: %w", err)
	}
	req.Header.Set("User-Agent", "LangChain-MCP-Researcher/1.0 (PDF-Reader)")

	resp, err := pdfHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download PDF from %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("PDF server returned status %s for %s", resp.Status, url)
	}

	tmpFile, err := os.CreateTemp("", "paper-*.pdf")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		return nil, fmt.Errorf("failed to save PDF to temp file: %w", err)
	}

	return ReadPDF(tmpFile.Name(), maxPages)
}

// ReadPDFBytes reads PDF from memory.
func ReadPDFBytes(data []byte, maxPages int) (*PDFDocument, error) {
	readerAt := bytes.NewReader(data)
	r, err := pdf.NewReader(readerAt, int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("failed to parse PDF bytes: %w", err)
	}

	numPages := r.NumPage()
	if maxPages <= 0 || maxPages > numPages {
		maxPages = numPages
	}

	var sb strings.Builder
	for i := 1; i <= maxPages; i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}
		text, err := p.GetPlainText(nil)
		if err != nil {
			continue
		}
		sb.WriteString(fmt.Sprintf("\n--- Page %d ---\n", i))
		sb.WriteString(cleanPDFText(text))
		sb.WriteString("\n")
	}

	fullText := strings.TrimSpace(sb.String())
	sections := ExtractSections(fullText)

	return &PDFDocument{
		TotalPages: numPages,
		ReadPages:  maxPages,
		Text:       fullText,
		Sections:   sections,
	}, nil
}

// cleanPDFText normalizes spacing and artifacts.
func cleanPDFText(s string) string {
	var cleaned []string
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			cleaned = append(cleaned, trimmed)
		}
	}
	return strings.Join(cleaned, "\n")
}

// ExtractSections parses standard academic sections.
func ExtractSections(text string) map[string]string {
	sections := make(map[string]string)
	patterns := map[string]*regexp.Regexp{
		"Abstract":     regexp.MustCompile(`(?i)(?:^|\n)\s*(?:1\.?\s*)?abstract[:\s\n]+([\s\S]*?)(?:\n\s*(?:1\.?\s*)?introduction|\n\s*(?:2\.?\s*)|\z)`),
		"Introduction": regexp.MustCompile(`(?i)(?:^|\n)\s*(?:1\.?\s*)?introduction[:\s\n]+([\s\S]*?)(?:\n\s*(?:2\.?\s*)|\z)`),
		"Conclusion":   regexp.MustCompile(`(?i)(?:^|\n)\s*(?:\d+\.?\s*)?conclusion[s]?[:\s\n]+([\s\S]*?)(?:\n\s*references|\z)`),
	}

	for name, re := range patterns {
		matches := re.FindStringSubmatch(text)
		if len(matches) > 1 {
			content := strings.TrimSpace(matches[1])
			if len(content) > 1500 {
				content = content[:1500] + "... [truncated]"
			}
			if content != "" {
				sections[name] = content
			}
		}
	}
	return sections
}
