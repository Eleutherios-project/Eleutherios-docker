#!/usr/bin/env python3
"""
Aegis Insight - MCP stdio Wrapper (newline-delimited JSON)
Claude Desktop sends raw JSON lines, not Content-Length framed messages.
"""
import sys
import os
import json
import requests

AEGIS_MCP_URL = os.getenv("AEGIS_MCP_URL", "http://localhost:8100")
DEBUG = os.getenv("AEGIS_MCP_DEBUG", "false").lower() == "true"

def log(msg: str):
    if DEBUG:
        print(f"[aegis-mcp] {msg}", file=sys.stderr, flush=True)

def send(obj):
    """Send JSON response as a single line"""
    line = json.dumps(obj)
    print(line, flush=True)
    log(f"Sent: {line[:100]}...")

def call_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{AEGIS_MCP_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def handle_tool(name: str, args: dict) -> dict:
    try:
        if name == "analyze_topic":
            result = call_api("/mcp/analyze_topic", "POST", {
                "topic": args.get("topic"),
                "detail": args.get("detail", "standard"),
                "domain": args.get("domain"),
                "max_claims": args.get("max_claims", 200)
            })
        elif name == "assess_source":
            result = call_api("/mcp/assess_source", "POST", {
                "source_identifier": args.get("source_identifier"),
                "detail": args.get("detail", "standard")
            })
        elif name == "get_perspectives":
            result = call_api("/mcp/get_perspectives", "POST", {
                "topic": args.get("topic"),
                "max_clusters": args.get("max_clusters", 5),
                "claims_per_cluster": args.get("claims_per_cluster", 5)
            })
        elif name == "get_claim_context":
            result = call_api("/mcp/get_claim_context", "POST", {
                "claim_id": args.get("claim_id"),
                "include_graph": args.get("include_graph", False)
            })
        elif name == "list_domains":
            result = call_api("/mcp/list_domains", "GET")
        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
        
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}

TOOLS = [
    {
        "name": "analyze_topic",
        "description": "Analyze a topic for suppression patterns, coordination signatures, and epistemic manipulation in the Aegis knowledge graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to analyze"},
                "detail": {"type": "string", "enum": ["abbreviated", "standard", "verbose"], "default": "standard"},
                "domain": {"type": "string", "description": "Optional domain scope"},
                "max_claims": {"type": "integer", "default": 200}
            },
            "required": ["topic"]
        }
    },
    {
        "name": "assess_source",
        "description": "Assess a source's position in the knowledge topology.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_identifier": {"type": "string", "description": "Source name or path"},
                "detail": {"type": "string", "enum": ["abbreviated", "standard", "verbose"], "default": "standard"}
            },
            "required": ["source_identifier"]
        }
    },
    {
        "name": "get_perspectives",
        "description": "Get clustered perspectives on a topic with representative claims.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "max_clusters": {"type": "integer", "default": 5},
                "claims_per_cluster": {"type": "integer", "default": 5}
            },
            "required": ["topic"]
        }
    },
    {
        "name": "get_claim_context",
        "description": "Get full context for a specific claim by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "include_graph": {"type": "boolean", "default": False}
            },
            "required": ["claim_id"]
        }
    },
    {
        "name": "list_domains",
        "description": "List available knowledge domains.",
        "inputSchema": {"type": "object", "properties": {}}
    }
]

def main():
    log("Starting (newline-delimited mode)...")
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        log(f"Received: {line[:100]}...")
        
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"JSON parse error: {e}")
            continue
        
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})
        
        log(f"Method: {method}, id: {mid}")
        
        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "aegis_insight", "version": "1.0.0"}
                }
            })
        elif method == "notifications/initialized":
            pass  # No response needed
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            result = handle_tool(name, args)
            send({"jsonrpc": "2.0", "id": mid, "result": result})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "notifications/cancelled":
            log(f"Cancelled: {params}")
        else:
            log(f"Unknown method: {method}")
            if mid is not None:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown: {method}"}})

if __name__ == "__main__":
    main()
