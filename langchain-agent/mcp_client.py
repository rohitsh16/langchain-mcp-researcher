"""
Stdio Model Context Protocol (MCP) Client for Python and LangChain.
Connects directly to Go MCP server binaries over standard input/output streams.
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("mcp_client")


class StdioMCPClient:
    """A client that communicates with an MCP server via stdin/stdout."""

    def __init__(self, command: str, args: Optional[List[str]] = None, cwd: Optional[str] = None):
        self.command = command
        self.args = args or []
        self.cwd = cwd or os.getcwd()
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0

    def start(self):
        """Starts the server subprocess and executes initialization handshake."""
        cmd_path = self.command
        if cmd_path in ("python", "python3") and sys.executable:
            cmd_path = sys.executable
        elif not os.path.isabs(cmd_path):
            local_path = os.path.abspath(os.path.join(self.cwd, cmd_path))
            if os.path.exists(local_path):
                cmd_path = local_path

        self.process = subprocess.Popen(
            [cmd_path] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )

        # 1. Initialize
        self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "langchain-agent-client", "version": "1.0.0"},
            },
        )

        # 2. Notification initialized
        self._send_notification("notifications/initialized")

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(f"MCP server process {self.command} is not running")

        self._request_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            req["params"] = params

        req_str = json.dumps(req) + "\n"
        self.process.stdin.write(req_str)
        self.process.stdin.flush()

        # Read response line
        line = self.process.stdout.readline()
        if not line:
            stderr_out = self.process.stderr.read()
            raise RuntimeError(f"Server closed connection unexpectedly. Stderr: {stderr_out}")

        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"MCP error {resp['error'].get('code')}: {resp['error'].get('message')}")

        return resp.get("result", {})

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        if not self.process or self.process.poll() is not None:
            return
        notif = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notif["params"] = params
        self.process.stdin.write(json.dumps(notif) + "\n")
        self.process.stdin.flush()

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools exposed by the MCP server."""
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Call a specific tool on the MCP server and return concatenated text content."""
        params = {"name": name, "arguments": arguments or {}}
        result = self._send_request("tools/call", params)
        content_items = result.get("content", [])
        texts = [item.get("text", "") for item in content_items if item.get("type") == "text"]
        return "\n".join(texts)

    def close(self):
        """Terminate the server process."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
            finally:
                self.process = None


class MCPRegistry:
    """Manages connections to multiple MCP servers and exports LangChain-compatible tools."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.clients: Dict[str, StdioMCPClient] = {}
        self.tools_map: Dict[str, StdioMCPClient] = {}

    def register_server(self, server_id: str, command: str, args: Optional[List[str]] = None, cwd: Optional[str] = None):
        server_cwd = os.path.abspath(os.path.join(self.base_dir, cwd)) if cwd else self.base_dir
        client = StdioMCPClient(command, args=args, cwd=server_cwd)
        client.start()
        self.clients[server_id] = client

        tools = client.list_tools()
        for t in tools:
            name = t["name"]
            self.tools_map[name] = client
        return tools

    def call_tool(self, name: str, **kwargs) -> str:
        if name not in self.tools_map:
            raise KeyError(f"Tool '{name}' not found on any registered MCP server")
        client = self.tools_map[name]
        return client.call_tool(name, kwargs)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        for client in self.clients.values():
            all_tools.extend(client.list_tools())
        return all_tools

    def close_all(self):
        for client in self.clients.values():
            client.close()
        self.clients.clear()
        self.tools_map.clear()
