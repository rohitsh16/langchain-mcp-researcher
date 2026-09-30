UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CC ?= /usr/bin/clang
	export CC
endif

BINDIR=bin
LDFLAGS=-ldflags="-linkmode=external"

.PHONY: all build-all test clean ctx-stats ctx-index run-agent test-agent unit-test smoke-test research-smoke benchmark ci

PYTHON ?= .venv/bin/python3
PYTEST ?= .venv/bin/pytest

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
	@$(PYTHON) langchain-agent/agent_runner.py --test

run-agent: build-all
	@$(PYTHON) langchain-agent/agent_runner.py

ctx-stats:
	@./bin/ctx stats -repo .

ctx-index:
	@./bin/ctx index -repo .

unit-test:
	@echo "🧪 Running Python correctness & research test suites..."
	@$(PYTEST) ai_correctness_mcp_server/tests research/tests

smoke-test:
	@echo "🚀 Running AI correctness MCP server end-to-end smoke test..."
	@$(PYTHON) -m ai_correctness_mcp_server --smoke-test

research-smoke:
	@echo "🔬 Running research experiment runner smoke test..."
	@$(PYTHON) -c "from ai_correctness_mcp_server.experiments.runner import ExperimentRunner; runner = ExperimentRunner(); res = runner.run_calibration_experiment('EXP-SMOKE-001', 'Calibration Smoke', 'Coverage test', sample_size=60); print('EXP RESULT:', res)"

benchmark:
	@echo "📊 Running factual and reasoning benchmark adapters..."
	@$(PYTHON) -c "from ai_correctness_mcp_server.benchmarks import SimpleQABenchmark, MathReasoningBenchmark; from ai_correctness_mcp_server.verification.exact import ExactVerifier; b = SimpleQABenchmark(); print('SimpleQA Items:', len(b.data)); m = MathReasoningBenchmark(); print('Math Problems:', len(m.problems))"

ci: test unit-test smoke-test
	@echo "🎉 All Go and Python CI suites passed successfully!"

clean:
	@rm -f $(BINDIR)/arxiv-mcp-server $(BINDIR)/pdf-reader-mcp-server $(BINDIR)/software-eng-mcp-server
	@echo "🧹 Cleaned Go MCP binaries."
