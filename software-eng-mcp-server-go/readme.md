# Software Engineering MCP Server (Go)

A Model Context Protocol (MCP) JSON-RPC 2.0 server in Go delivering software engineering articles, architectural design patterns, and engineering trade-offs.

## Tools Exposed

### `search_swe_articles`
Search curated software engineering and AI system design articles by keyword.
- **Inputs**:
  - `topic` (string, required): Topic keyword (e.g. `mcp`, `context`, `concurrency`, `rag`, `sqlite`).
  - `limit` (int, optional): Max articles to return (default: `5`).

### `get_swe_pattern`
Fetch deep architectural pattern guidelines, trade-offs, and implementation recommendations.
- **Inputs**:
  - `pattern` (string, required): Pattern name (`mcp`, `context-management`, `research-agent`).

## Running Standalone

```bash
CC=/usr/bin/clang go build -ldflags="-linkmode=external" -o software-eng-mcp-server .
./software-eng-mcp-server
```
