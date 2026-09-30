package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/rohitsh16/software-eng-mcp-server-go/models"
)

type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      any           `json:"id,omitempty"`
	Result  any           `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type MCPTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type ToolCallResult struct {
	Content []ToolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

type ToolContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type PatternGuide struct {
	Pattern         string   `json:"pattern"`
	Category        string   `json:"category"`
	Summary         string   `json:"summary"`
	Tradeoffs       []string `json:"tradeoffs"`
	Recommendations []string `json:"recommendations"`
	ReferenceURL    string   `json:"reference_url"`
}

// Built-in curated engineering knowledge database
var engineeringArticles = []models.Article{
	{
		Title:  "Model Context Protocol: Standardizing Tool Calling Across Autonomous Coding Agents",
		URL:    "https://modelcontextprotocol.io/introduction",
		Author: "Anthropic / Open Standards",
	},
	{
		Title:  "ContextOS: Local-First Deterministic Context Infrastructure for Multi-Agent Workspaces",
		URL:    "https://github.com/rohitsh16/ContextOS",
		Author: "ContextOS Architecture Team",
	},
	{
		Title:  "Building Resilient Research Agents: LangChain and Tool-Use Best Practices",
		URL:    "https://python.langchain.com/docs/concepts/agents/",
		Author: "Harrison Chase & LangChain Community",
	},
	{
		Title:  "Designing High-Performance In-Process Storage with SQLite WAL and FTS5",
		URL:    "https://sqlite.org/wal.html",
		Author: "D. Richard Hipp",
	},
	{
		Title:  "Retrieval-Augmented Generation (RAG) Systems: Architecture, Chunking, and Re-ranking",
		URL:    "https://arxiv.org/abs/2005.11401",
		Author: "Lewis et al. (Facebook AI Research)",
	},
	{
		Title:  "Event Sourcing and CQRS in Distributed Go Services",
		URL:    "https://martinfowler.com/eaaDev/EventSourcing.html",
		Author: "Martin Fowler",
	},
	{
		Title:  "Concurrency in Go: Safe Goroutines, Worker Pools, and Mutex Patterns",
		URL:    "https://go.dev/doc/effective_go#concurrency",
		Author: "Go Team",
	},
	{
		Title:  "Context Window Compaction and Attention Budgeting in Long-Context LLMs",
		URL:    "https://arxiv.org/abs/2310.03025",
		Author: "DeepMind Research",
	},
}

var patternGuides = map[string]PatternGuide{
	"mcp": {
		Pattern:  "Model Context Protocol (MCP) Stdio Architecture",
		Category: "Agent Interoperability",
		Summary:  "MCP standardizes bidirectional JSON-RPC 2.0 transport over stdin/stdout or SSE, decoupling LLMs from proprietary tool APIs.",
		Tradeoffs: []string{
			"Stdio provides zero-network, local process isolation and minimal latency",
			"Requires robust process lifecycle management and non-blocking IO",
		},
		Recommendations: []string{
			"Use newline-delimited JSON messages",
			"Implement graceful SIGINT/SIGTERM termination",
			"Include complete JSON Schema for tool input arguments",
		},
		ReferenceURL: "https://spec.modelcontextprotocol.io",
	},
	"context-management": {
		Pattern:  "Deterministic Token Allocation & 6-Pass Context Planning",
		Category: "Context Optimization",
		Summary:  "Allocates prompt budget across system instructions, active task graph, relevant code symbols, recent trace events, and durable memories.",
		Tradeoffs: []string{
			"Ensures total prompt tokens stay strictly below LLM window limits",
			"Prevents needle-in-a-haystack attention degradation",
		},
		Recommendations: []string{
			"Score items by lexical overlap (BM25) and reciprocal rank fusion (RRF)",
			"Prune low-density redundant nodes",
			"Record durable engineering decisions across chat sessions",
		},
		ReferenceURL: "https://github.com/rohitsh16/ContextOS",
	},
	"research-agent": {
		Pattern:  "Multi-Server Tool-Augmented Research Loop",
		Category: "AI Agent Architecture",
		Summary:  "A LangChain coordinator dispatches queries across specialized MCP servers (arXiv search, PDF extraction, SWE pattern lookup) and synthesizes findings.",
		Tradeoffs: []string{
			"High quality structured reports grounded in real academic and engineering literature",
			"Requires orchestration and handling of slow external network calls",
		},
		Recommendations: []string{
			"Run paper searches with strict pagination",
			"Extract section summaries (Abstract + Conclusion) before reading full PDFs",
			"Ground conclusions in cited sources",
		},
		ReferenceURL: "https://python.langchain.com/docs/concepts/agents/",
	},
}

func getTools() []MCPTool {
	return []MCPTool{
		{
			Name:        "search_swe_articles",
			Description: "Search curated software engineering, system design, and AI architecture articles by topic.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"topic": map[string]any{
						"type":        "string",
						"description": "Engineering topic or keyword (e.g., 'mcp', 'context', 'concurrency', 'rag', 'sqlite')",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "Maximum number of articles to return (default: 5)",
					},
				},
				"required": []string{"topic"},
			},
		},
		{
			Name:        "get_swe_pattern",
			Description: "Fetch architectural pattern guidelines, trade-offs, and recommendations for modern software and agent engineering.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{
						"type":        "string",
						"description": "Pattern name (options: 'mcp', 'context-management', 'research-agent')",
					},
				},
				"required": []string{"pattern"},
			},
		},
	}
}

func searchArticles(topic string, limit int) models.ArticleResponse {
	if limit <= 0 {
		limit = 5
	}
	tLower := strings.ToLower(topic)

	var matched []models.Article
	for _, art := range engineeringArticles {
		if strings.Contains(strings.ToLower(art.Title), tLower) ||
			strings.Contains(strings.ToLower(art.Author), tLower) ||
			strings.Contains(strings.ToLower(art.URL), tLower) {
			matched = append(matched, art)
		}
	}

	// Fallback to all articles up to limit if no specific keyword match
	if len(matched) == 0 {
		matched = engineeringArticles
	}

	if len(matched) > limit {
		matched = matched[:limit]
	}

	return models.ArticleResponse{
		Articles: matched,
	}
}

func getPattern(key string) (*PatternGuide, error) {
	kLower := strings.ToLower(strings.TrimSpace(key))
	for k, p := range patternGuides {
		if strings.Contains(kLower, k) || strings.Contains(strings.ToLower(p.Pattern), kLower) {
			return &p, nil
		}
	}
	return nil, fmt.Errorf("pattern '%s' not found. Available patterns: 'mcp', 'context-management', 'research-agent'", key)
}

func handleToolCall(name string, rawArgs json.RawMessage) ToolCallResult {
	switch name {
	case "search_swe_articles":
		var req models.ArticleRequest
		if err := json.Unmarshal(rawArgs, &req); err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("invalid arguments: %v", err)}},
				IsError: true,
			}
		}
		if req.Topic == "" {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: "topic argument is required"}},
				IsError: true,
			}
		}
		resp := searchArticles(req.Topic, req.Limit)
		data, _ := json.MarshalIndent(resp, "", "  ")
		return ToolCallResult{
			Content: []ToolContent{{Type: "text", Text: string(data)}},
		}

	case "get_swe_pattern":
		var args struct {
			Pattern string `json:"pattern"`
		}
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("invalid arguments: %v", err)}},
				IsError: true,
			}
		}
		if args.Pattern == "" {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: "pattern argument is required"}},
				IsError: true,
			}
		}
		p, err := getPattern(args.Pattern)
		if err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: err.Error()}},
				IsError: true,
			}
		}
		data, _ := json.MarshalIndent(p, "", "  ")
		return ToolCallResult{
			Content: []ToolContent{{Type: "text", Text: string(data)}},
		}

	default:
		return ToolCallResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("unknown tool: %s", name)}},
			IsError: true,
		}
	}
}

func handleRequest(req JSONRPCRequest) *JSONRPCResponse {
	switch req.Method {
	case "initialize":
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities": map[string]any{
					"tools": map[string]any{},
				},
				"serverInfo": map[string]any{
					"name":    "software-eng-mcp-server-go",
					"version": "1.0.0",
				},
			},
		}

	case "notifications/initialized":
		return nil

	case "ping":
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  map[string]any{},
		}

	case "tools/list":
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"tools": getTools(),
			},
		}

	case "tools/call":
		var params struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil {
			return &JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error: &JSONRPCError{
					Code:    -32602,
					Message: fmt.Sprintf("invalid params: %v", err),
				},
			}
		}
		res := handleToolCall(params.Name, params.Arguments)
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  res,
		}

	default:
		if req.ID == nil {
			return nil
		}
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &JSONRPCError{
				Code:    -32601,
				Message: fmt.Sprintf("method not found: %s", req.Method),
			},
		}
	}
}

func Run(r io.Reader, w io.Writer) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)
	encoder := json.NewEncoder(w)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var req JSONRPCRequest
		if err := json.Unmarshal(line, &req); err != nil {
			resp := JSONRPCResponse{
				JSONRPC: "2.0",
				Error: &JSONRPCError{
					Code:    -32700,
					Message: "parse error",
				},
			}
			_ = encoder.Encode(resp)
			continue
		}

		resp := handleRequest(req)
		if resp != nil {
			if err := encoder.Encode(resp); err != nil {
				return err
			}
		}
	}
	return scanner.Err()
}

func main() {
	if err := Run(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "software-eng-mcp-server error: %v\n", err)
		os.Exit(1)
	}
}
