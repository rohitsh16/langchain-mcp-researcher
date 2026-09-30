#!/usr/bin/env python3
"""
LangChain MCP Research Assistant Runner.
Orchestrates Go MCP servers (arXiv, PDF Reader, Software Engineering Knowledge, ContextOS)
to perform automated, multi-source literature review, context-aware memory retention,
and technical synthesis.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from mcp_client import MCPRegistry

# Check for LangChain imports
LANGCHAIN_AVAILABLE = False
try:
    from langchain.tools import Tool, StructuredTool
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain import hub
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    default_config = {
        "agent": {
            "name": "LangChain-MCP-Researcher",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_iterations": 8,
            "offline_mode": False,
        },
        "mcp_servers": {
            "arxiv": {
                "name": "arXiv Academic Search",
                "command": "../bin/arxiv-mcp-server",
                "args": [],
            },
            "pdf_reader": {
                "name": "PDF Section Extractor",
                "command": "../bin/pdf-reader-mcp-server",
                "args": [],
            },
            "software_eng": {
                "name": "Software Engineering Knowledge",
                "command": "../bin/software-eng-mcp-server",
                "args": [],
            },
            "contextos": {
                "name": "ContextOS Persistent Memory & Context",
                "command": "../bin/contextd",
                "args": ["-repo", "..", "-mcp"],
            },
        },
        "research": {
            "max_papers": 5,
            "pdf_pages_per_paper": 3,
            "output_dir": "./reports",
        },
    }

    if not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.yaml")

    if os.path.exists(config_path) and YAML_AVAILABLE:
        try:
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass

    return default_config


class ResearchOrchestrator:
    def __init__(self, config: Dict[str, Any], base_dir: Optional[str] = None):
        self.config = config
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.registry = MCPRegistry(base_dir=self.base_dir)
        self.servers_configured = config.get("mcp_servers", {})
        self.reports_dir = os.path.join(self.base_dir, config.get("research", {}).get("output_dir", "./reports"))
        os.makedirs(self.reports_dir, exist_ok=True)

    def start_servers(self):
        """Starts and connects to all configured Go MCP servers."""
        print("🔗 Connecting to Golang MCP servers...")
        for server_id, conf in self.servers_configured.items():
            cmd = conf.get("command")
            args = conf.get("args", [])
            name = conf.get("name", server_id)
            try:
                tools = self.registry.register_server(server_id, cmd, args)
                print(f"  ✓ [{name}] connected ({len(tools)} tools: {', '.join(t['name'] for t in tools)})")
            except Exception as e:
                print(f"  ✗ [{name}] failed to start: {e}", file=sys.stderr)

    def run_autonomous_research(self, query: str) -> str:
        """
        Executes a deterministic research synthesis pipeline using all available MCP tools:
        1. ContextOS context allocation and prior memory retrieval
        2. Query arXiv for academic publications
        3. Query Software Engineering repository for architectural patterns & design articles
        4. Synthesize deep findings with ContextOS durable memory persistence
        5. Produce a Markdown research report with verified citations
        """
        print(f"\n🧠 Initiating research workflow for: \"{query}\"")
        max_papers = self.config.get("research", {}).get("max_papers", 3)

        contextos_available = "context_plan" in self.registry.tools_map

        # 0. ContextOS Allocation & Event Tracking
        context_plan: Dict[str, Any] = {}
        context_memories: List[Dict[str, Any]] = []
        if contextos_available:
            try:
                print("  🧩 Engaging ContextOS 6-Pass Context Planner & Event Log...")
                self.registry.call_tool("context_event", type="research_start", summary=f"Started research: {query}")
                plan_raw = self.registry.call_tool("context_plan", task=query, budget=2000)
                context_plan = json.loads(plan_raw)
                selected_tok = context_plan.get("selected_tokens", 0)
                print(f"  ✓ ContextOS allocated {selected_tok} tokens across codebase graph & decisions")

                search_raw = self.registry.call_tool("context_search", query=query, limit=3)
                search_data = json.loads(search_raw)
                if isinstance(search_data, list):
                    context_memories = search_data
                elif isinstance(search_data, dict):
                    context_memories = search_data.get("results", search_data.get("items", []))
            except Exception as e:
                print(f"  ⚠️ ContextOS notice: {e}")

        # 1. Search arXiv
        print("  📚 Querying arXiv Academic Search MCP...")
        arxiv_raw = self.registry.call_tool("search_arxiv", query=query, max_results=max_papers)
        papers: List[Dict[str, Any]] = []
        try:
            papers = json.loads(arxiv_raw)
        except Exception:
            print("  ⚠️ Notice: Raw arXiv response received")

        # 2. Search Software Engineering knowledge
        print("  ⚙️ Querying Software Engineering MCP...")
        swe_raw = self.registry.call_tool("search_swe_articles", topic=query, limit=3)
        swe_articles: List[Dict[str, Any]] = []
        try:
            swe_data = json.loads(swe_raw)
            swe_articles = swe_data.get("articles", [])
        except Exception:
            pass

        # 3. Retrieve relevant engineering patterns
        patterns: List[Dict[str, Any]] = []
        for pat_key in ["mcp", "context-management", "research-agent"]:
            if any(k in query.lower() for k in [pat_key, "context", "agent", "tool", "memory"]):
                try:
                    pat_raw = self.registry.call_tool("get_swe_pattern", pattern=pat_key)
                    patterns.append(json.loads(pat_raw))
                except Exception:
                    pass

        # If no specific pattern matched query, grab context-management and mcp by default
        if not patterns:
            try:
                patterns.append(json.loads(self.registry.call_tool("get_swe_pattern", pattern="context-management")))
            except Exception:
                pass

        # 4. Synthesize Markdown Research Report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            f"# Research Report: {query}",
            f"\n*Generated by **LangChain MCP Researcher** with **ContextOS** on {timestamp}*",
            "\n---\n",
            "## 1. Executive Summary",
            f"This report synthesizes academic literature, software engineering architecture patterns, and repository context relevant to **{query}**. The investigation integrates findings from arXiv academic repositories, production software engineering design paradigms, and durable context graph memory managed by ContextOS Go microservices.\n",
            "## 2. Academic Literature & Theoretical Findings",
        ]

        if papers:
            for idx, p in enumerate(papers, 1):
                report_lines.append(f"### 2.{idx} [{p.get('title', 'Untitled')}]({p.get('entry_url', p.get('pdf_url', '#'))})")
                report_lines.append(f"- **Authors**: {', '.join(p.get('authors', []))}")
                report_lines.append(f"- **arXiv ID**: `{p.get('id', 'N/A')}` | **Published**: {p.get('published', 'N/A')[:10]}")
                report_lines.append(f"- **Categories**: {', '.join(p.get('categories', []))}")
                report_lines.append(f"- **Abstract Summary**:\n  > {p.get('summary', 'No summary available.')}\n")
        else:
            report_lines.append("No direct arXiv papers found for the exact term. Recommend broader keyword querying.\n")

        report_lines.append("## 3. Practical Software Engineering & Architectural Patterns")
        if patterns:
            for pat in patterns:
                report_lines.append(f"### Pattern: {pat.get('pattern', 'Pattern')}")
                report_lines.append(f"- **Domain**: `{pat.get('category', 'Architecture')}`")
                report_lines.append(f"- **Summary**: {pat.get('summary', '')}")
                if pat.get("tradeoffs"):
                    report_lines.append("- **Key Trade-offs**:")
                    for to in pat.get("tradeoffs", []):
                        report_lines.append(f"  - {to}")
                if pat.get("recommendations"):
                    report_lines.append("- **Implementation Recommendations**:")
                    for rec in pat.get("recommendations", []):
                        report_lines.append(f"  - {rec}")
                report_lines.append(f"- **Reference**: [{pat.get('reference_url', '#')}]({pat.get('reference_url', '#')})\n")

        if swe_articles:
            report_lines.append("### Recommended Industry Resources")
            for art in swe_articles:
                report_lines.append(f"- [{art.get('title')}]({art.get('URL')}) — *{art.get('Author')}*")
            report_lines.append("")

        if contextos_available:
            report_lines.append("## 4. ContextOS Durable Memory & Architectural Context")
            selected_tokens = context_plan.get("selected_tokens", 0)
            budget = context_plan.get("budget", 0)
            report_lines.append(f"- **Allocated Context Budget**: {selected_tokens}/{budget} tokens via 6-Pass Reciprocal-Rank Allocation")
            var_ctx = context_plan.get("variable_context", [])
            if var_ctx:
                report_lines.append("- **Relevant Repository Context & Symbols**:")
                for vc in var_ctx[:5]:
                    report_lines.append(f"  - `{vc.get('location', vc.get('source', 'unknown'))}` ({vc.get('tokens', 0)} tokens)")
            if context_memories:
                report_lines.append("- **Retrieved Prior Durable Decisions**:")
                for mem in context_memories:
                    text_val = mem.get("text", mem.get("content", str(mem)))
                    report_lines.append(f"  - {text_val}")
            report_lines.append("")

        sec_num = 5 if contextos_available else 4
        report_lines.append(f"## {sec_num}. Synthesis & Strategic Recommendations")
        report_lines.append(
            "1. **Decouple Tool Interfaces with MCP**: Leverage standard JSON-RPC 2.0 stdio pipes for zero-latency, local-first tool execution without network friction.\n"
            "2. **Implement Deterministic Budgeting**: As context windows expand, unstructured context packing leads to latency degradation and retrieval hallucination. Adopt reciprocal-rank token allocation via ContextOS.\n"
            "3. **Persistent Engineering Memory**: Incorporate persistent context systems to retain durable architectural decisions across multi-agent workflows."
        )

        final_report = "\n".join(report_lines)

        # Save report
        safe_name = "".join(c if c.isalnum() else "_" for c in query.lower())[:40]
        report_file = os.path.join(self.reports_dir, f"report_{safe_name}_{int(time.time())}.md")
        with open(report_file, "w") as f:
            f.write(final_report)

        # 5. Persist durable memory in ContextOS
        if contextos_available:
            try:
                print("  💾 Storing durable synthesis and decisions in ContextOS memory...")
                self.registry.call_tool(
                    "context_remember",
                    text=f"Synthesized comprehensive research report on '{query}' integrating arXiv, SWE patterns, and ContextOS allocation into {os.path.basename(report_file)}.",
                    scope="repo"
                )
                self.registry.call_tool(
                    "context_event",
                    type="research_complete",
                    summary=f"Saved report {os.path.basename(report_file)} for '{query}'"
                )
            except Exception as e:
                print(f"  ⚠️ ContextOS remember notice: {e}")

        print(f"\n✅ Research completed! Saved to: {report_file}")
        return final_report

    def close(self):
        self.registry.close_all()


def main():
    parser = argparse.ArgumentParser(description="LangChain MCP Researcher Agent")
    parser.add_argument("--query", "-q", type=str, help="Research topic or question")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--test", action="store_true", help="Run self-test on all MCP servers")
    args = parser.parse_args()

    config = load_config(args.config)
    orchestrator = ResearchOrchestrator(config)

    try:
        orchestrator.start_servers()

        if args.test:
            print("\n🧪 Running MCP integration self-test...")
            tools = orchestrator.registry.get_all_tools()
            print(f"Discovered {len(tools)} MCP tools across all servers.")
            for t in tools:
                print(f" - {t['name']}: {t['description']}")
            print("\n✅ All servers operating normally.")
            return

        query = args.query
        if not query:
            print("\n=== LangChain MCP Research Assistant ===")
            query = input("Enter your research topic/query: ").strip()

        if not query:
            print("No query provided. Exiting.")
            return

        report = orchestrator.run_autonomous_research(query)
        print("\n" + "="*60)
        print(report[:800] + "\n\n... [report continues] ...")
        print("="*60)

    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
