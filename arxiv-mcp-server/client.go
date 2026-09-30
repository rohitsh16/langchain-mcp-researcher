package main

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Paper represents an arXiv paper metadata.
type Paper struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	Summary     string   `json:"summary"`
	Authors     []string `json:"authors"`
	Published   string   `json:"published"`
	Updated     string   `json:"updated"`
	PDFURL      string   `json:"pdf_url"`
	EntryURL    string   `json:"entry_url"`
	Categories  []string `json:"categories"`
}

// Atom XML structs for arXiv API response.
type atomFeed struct {
	XMLName xml.Name    `xml:"feed"`
	Title   string      `xml:"title"`
	Entries []atomEntry `xml:"entry"`
}

type atomEntry struct {
	ID        string       `xml:"id"`
	Title     string       `xml:"title"`
	Summary   string       `xml:"summary"`
	Published string       `xml:"published"`
	Updated   string       `xml:"updated"`
	Authors   []atomAuthor `xml:"author"`
	Links     []atomLink   `xml:"link"`
	Category  []atomCat    `xml:"category"`
}

type atomAuthor struct {
	Name string `xml:"name"`
}

type atomLink struct {
	Href  string `xml:"href,attr"`
	Rel   string `xml:"rel,attr"`
	Title string `xml:"title,attr"`
	Type  string `xml:"type,attr"`
}

type atomCat struct {
	Term string `xml:"term,attr"`
}

var httpClient = &http.Client{
	Timeout: 30 * time.Second,
}

// cleanText trims spaces and normalizes newlines.
func cleanText(s string) string {
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, l := range lines {
		t := strings.TrimSpace(l)
		if t != "" {
			cleaned = append(cleaned, t)
		}
	}
	return strings.Join(cleaned, " ")
}

// extractPaperID extracts the clean arXiv ID (e.g. 2312.11805 or abs/2312.11805) from full URI.
func extractPaperID(rawID string) string {
	parts := strings.Split(rawID, "/abs/")
	if len(parts) == 2 {
		return parts[1]
	}
	parts = strings.Split(rawID, "/")
	return parts[len(parts)-1]
}

// parseAtomFeed parses raw Atom XML bytes into []Paper.
func parseAtomFeed(data []byte) ([]Paper, error) {
	var feed atomFeed
	if err := xml.Unmarshal(data, &feed); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Atom feed: %w", err)
	}

	papers := make([]Paper, 0, len(feed.Entries))
	for _, entry := range feed.Entries {
		cleanID := extractPaperID(entry.ID)
		if cleanID == "" {
			continue
		}

		authors := make([]string, 0, len(entry.Authors))
		for _, a := range entry.Authors {
			name := strings.TrimSpace(a.Name)
			if name != "" {
				authors = append(authors, name)
			}
		}

		cats := make([]string, 0, len(entry.Category))
		for _, c := range entry.Category {
			if c.Term != "" {
				cats = append(cats, c.Term)
			}
		}

		var pdfURL string
		entryURL := entry.ID
		for _, l := range entry.Links {
			if l.Title == "pdf" || l.Type == "application/pdf" {
				pdfURL = l.Href
			}
			if l.Rel == "alternate" {
				entryURL = l.Href
			}
		}
		if pdfURL == "" && cleanID != "" {
			pdfURL = fmt.Sprintf("https://arxiv.org/pdf/%s.pdf", cleanID)
		}

		papers = append(papers, Paper{
			ID:         cleanID,
			Title:      cleanText(entry.Title),
			Summary:    cleanText(entry.Summary),
			Authors:    authors,
			Published:  entry.Published,
			Updated:    entry.Updated,
			PDFURL:     pdfURL,
			EntryURL:   entryURL,
			Categories: cats,
		})
	}
	return papers, nil
}

// SearchArxiv queries the arXiv API for papers matching query.
func SearchArxiv(query string, maxResults int, sortBy string) ([]Paper, error) {
	if maxResults <= 0 {
		maxResults = 5
	}
	if maxResults > 25 {
		maxResults = 25
	}

	if sortBy == "" {
		sortBy = "relevance"
	}

	apiURL := fmt.Sprintf(
		"http://export.arxiv.org/api/query?search_query=all:%s&start=0&max_results=%d&sortBy=%s&sortOrder=descending",
		url.QueryEscape(query),
		maxResults,
		url.QueryEscape(sortBy),
	)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("User-Agent", "LangChain-MCP-Researcher/1.0 (rohitshukla)")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("arxiv request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("arxiv API returned status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return parseAtomFeed(body)
}

// GetArxivPaper retrieves paper metadata by arXiv ID.
func GetArxivPaper(paperID string) (*Paper, error) {
	cleanID := strings.TrimSpace(paperID)
	cleanID = strings.TrimPrefix(cleanID, "arxiv:")
	cleanID = strings.TrimPrefix(cleanID, "arXiv:")

	apiURL := fmt.Sprintf(
		"http://export.arxiv.org/api/query?id_list=%s",
		url.QueryEscape(cleanID),
	)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("User-Agent", "LangChain-MCP-Researcher/1.0 (rohitshukla)")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("arxiv request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("arxiv API returned status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	papers, err := parseAtomFeed(body)
	if err != nil {
		return nil, err
	}
	if len(papers) == 0 {
		return nil, fmt.Errorf("paper %s not found on arXiv", cleanID)
	}
	return &papers[0], nil
}
