# LangChain MCP Research Assistant

An autonomous research agent orchestrating high-performance Golang Model Context Protocol (MCP) microservices to conduct academic literature reviews, extract PDF paper sections, and synthesize system design patterns.

## Features

- **Multi-Server Stdio MCP Client**: Connects natively to Go MCP binaries over standard I/O pipes using the standard JSON-RPC 2.0 protocol.
- **Dynamic Tool Discovery**: Auto-detects all tools (`search_arxiv`, `get_arxiv_paper`, `read_pdf`, `read_pdf_url`, `search_swe_articles`, `get_swe_pattern`).
- **Autonomous Synthesis**: Merges academic findings from arXiv with production software engineering design patterns.
- **Self-Contained & Resilient**: Operates with LLM API keys (`OPENAI_API_KEY`) or in offline deterministic mode with zero external dependencies.

## Quickstart

### 1. Build the Go MCP Servers

From the repository root:
```bash
make build-all
```
This produces `arxiv-mcp-server`, `pdf-reader-mcp-server`, and `software-eng-mcp-server` in `bin/`.

### 2. Verify Server Integrations

```bash
cd langchain-agent
python3 agent_runner.py --test
```

### 3. Run a Research Query

```bash
python3 agent_runner.py --query "Deterministic token budgeting in autonomous agents"
```

Or run interactively:
```bash
python3 agent_runner.py
```

Generated reports are automatically formatted and saved in `langchain-agent/reports/`.
