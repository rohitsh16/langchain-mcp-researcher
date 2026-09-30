UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CC ?= /usr/bin/clang
	export CC
endif

BINDIR=bin
LDFLAGS=-ldflags="-linkmode=external"

.PHONY: all build-all test clean ctx-stats ctx-index run-agent test-agent

all: build-all

build-all:
	@mkdir -p $(BINDIR)
	@echo "🔨 Building Go MCP servers..."
	CC=$(CC) go build $(LDFLAGS) -o $(BINDIR)/arxiv-mcp-server ./arxiv-mcp-server
	CC=$(CC) go build $(LDFLAGS) -o $(BINDIR)/pdf-reader-mcp-server ./pdf-reader-mcp-server-go
	CC=$(CC) go build $(LDFLAGS) -o $(BINDIR)/software-eng-mcp-server ./software-eng-mcp-server-go
	@echo "✅ All MCP servers compiled successfully into $(BINDIR)/"

test:
	@echo "🧪 Running unit tests across all Go modules..."
	CC=$(CC) go test $(LDFLAGS) -v ./arxiv-mcp-server/... ./pdf-reader-mcp-server-go/... ./software-eng-mcp-server-go/...

test-agent: build-all
	@echo "🧪 Testing LangChain MCP agent integration..."
	python3 langchain-agent/agent_runner.py --test

run-agent: build-all
	@python3 langchain-agent/agent_runner.py

ctx-stats:
	@./bin/ctx stats -repo .

ctx-index:
	@./bin/ctx index -repo .

clean:
	@rm -f $(BINDIR)/arxiv-mcp-server $(BINDIR)/pdf-reader-mcp-server $(BINDIR)/software-eng-mcp-server
	@echo "🧹 Cleaned Go MCP binaries."
