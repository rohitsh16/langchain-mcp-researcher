package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
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

func getTools() []MCPTool {
	return []MCPTool{
		{
			Name:        "read_pdf",
			Description: "Read text and extract sections from a local PDF file.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Absolute or relative path to the PDF file",
					},
					"max_pages": map[string]any{
						"type":        "integer",
						"description": "Maximum number of pages to read (default: 5 to conserve context)",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "read_pdf_url",
			Description: "Download and extract text/sections from a remote PDF URL (e.g. arXiv PDF link).",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "Direct URL to the PDF file",
					},
					"max_pages": map[string]any{
						"type":        "integer",
						"description": "Maximum number of pages to extract (default: 5)",
					},
				},
				"required": []string{"url"},
			},
		},
	}
}

func handleToolCall(name string, rawArgs json.RawMessage) ToolCallResult {
	switch name {
	case "read_pdf":
		var args struct {
			Path     string `json:"path"`
			MaxPages int    `json:"max_pages"`
		}
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("invalid arguments: %v", err)}},
				IsError: true,
			}
		}
		if args.Path == "" {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: "path argument is required"}},
				IsError: true,
			}
		}
		if args.MaxPages <= 0 {
			args.MaxPages = 5
		}
		doc, err := ReadPDF(args.Path, args.MaxPages)
		if err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("failed to read PDF: %v", err)}},
				IsError: true,
			}
		}
		data, _ := json.MarshalIndent(doc, "", "  ")
		return ToolCallResult{
			Content: []ToolContent{{Type: "text", Text: string(data)}},
		}

	case "read_pdf_url":
		var args struct {
			URL      string `json:"url"`
			MaxPages int    `json:"max_pages"`
		}
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("invalid arguments: %v", err)}},
				IsError: true,
			}
		}
		if args.URL == "" {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: "url argument is required"}},
				IsError: true,
			}
		}
		if args.MaxPages <= 0 {
			args.MaxPages = 5
		}
		doc, err := ReadPDFURL(args.URL, args.MaxPages)
		if err != nil {
			return ToolCallResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("failed to read PDF from URL: %v", err)}},
				IsError: true,
			}
		}
		data, _ := json.MarshalIndent(doc, "", "  ")
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
					"name":    "pdf-reader-mcp-server-go",
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
	scanner.Buffer(make([]byte, 1024*1024), 20*1024*1024)
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
		fmt.Fprintf(os.Stderr, "pdf-reader-mcp-server error: %v\n", err)
		os.Exit(1)
	}
}
