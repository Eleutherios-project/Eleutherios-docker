#!/usr/bin/env python3
"""
Aegis Insight - MCP stdio Wrapper for Claude Desktop
=====================================================

This wrapper bridges Claude Desktop's stdio-based MCP protocol to
the Aegis Insight HTTP API server.

Installation:
    1. Start the Aegis MCP HTTP server (port 8100)
    2. Add this wrapper to Claude Desktop config
    3. Claude can now query Aegis Insight directly

Usage in claude_desktop_config.json:
{
  "mcpServers": {
    "aegis_insight": {
      "command": "python3",
      "args": ["/path/to/aegis_mcp_stdio.py"],
      "env": {
        "AEGIS_MCP_URL": "http://localhost:8100"
      }
    }
  }
}

Version: 1.0.0
Date: December 2025
"""

import sys
import json
import os
import requests
from typing import Dict, Any, Optional

# Configuration
AEGIS_MCP_URL = os.getenv("AEGIS_MCP_URL", "http://localhost:8100")
DEBUG = os.getenv("AEGIS_MCP_DEBUG", "false").lower() == "true"

def log_debug(msg: str):
    """Log debug messages to stderr"""
    if DEBUG:
        print(f"[aegis-mcp] {msg}", file=sys.stderr)


def send_response(response: Dict[str, Any]):
    """Send JSON-RPC response to stdout"""
    json_str = json.dumps(response)
    # MCP uses Content-Length header framing
    sys.stdout.write(f"Content-Length: {len(json_str)}\r\n\r\n{json_str}")
    sys.stdout.flush()


def send_error(id: Any, code: int, message: str):
    """Send JSON-RPC error response"""
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    })


def call_aegis_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Call the Aegis HTTP API"""
    url = f"{AEGIS_MCP_URL}{endpoint}"
    log_debug(f"Calling: {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise Exception(f"Cannot connect to Aegis server at {AEGIS_MCP_URL}. Is it running?")
    except requests.exceptions.Timeout:
        raise Exception("Aegis server timeout")
    except Exception as e:
        raise Exception(f"Aegis API error: {str(e)}")


def handle_initialize(params: Dict) -> Dict:
    """Handle MCP initialize request"""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "aegis_insight",
            "version": "1.0.0"
        }
    }


def handle_list_tools(params: Dict) -> Dict:
    """Handle tools/list request"""
    return {
        "tools": [
            {
                "name": "analyze_topic",
                "description": "Analyze a topic for suppression patterns, coordination signatures, and epistemic manipulation. Use this BEFORE retrieving or synthesizing information on contested topics to understand if there are information integrity concerns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic or search terms to analyze (e.g., 'Thomas Paine', 'Smedley Butler', 'prohibition')"
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["abbreviated", "standard", "verbose"],
                            "default": "standard",
                            "description": "Response detail level. 'abbreviated' is fastest, 'verbose' includes all claims."
                        },
                        "domain": {
                            "type": "string",
                            "description": "Optional domain scope to limit search"
                        },
                        "max_claims": {
                            "type": "integer",
                            "default": 200,
                            "description": "Maximum claims to analyze (10-1000)"
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "assess_source",
                "description": "Assess a specific source's position in the knowledge topology. Evaluates citation patterns, whether the source is isolated or connected, and its credential context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_identifier": {
                            "type": "string",
                            "description": "Source name, filename, or document identifier"
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["abbreviated", "standard", "verbose"],
                            "default": "standard"
                        }
                    },
                    "required": ["source_identifier"]
                }
            },
            {
                "name": "get_perspectives",
                "description": "Get clustered perspectives on a topic. Returns multiple viewpoint clusters with representative claims from each, useful for understanding the full landscape of opinion.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to get perspectives on"
                        },
                        "max_clusters": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum perspective clusters (2-10)"
                        },
                        "claims_per_cluster": {
                            "type": "integer",
                            "default": 5,
                            "description": "Representative claims per cluster (1-20)"
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "get_claim_context",
                "description": "Get full epistemic context for a specific claim by ID. Returns the claim text, source, citation network, and related claims.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "description": "The claim identifier (e.g., 'claim_abc123')"
                        },
                        "include_graph": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include citation subgraph"
                        }
                    },
                    "required": ["claim_id"]
                }
            },
            {
                "name": "list_domains",
                "description": "List available knowledge domains in the Aegis corpus. Returns domain names, claim counts, and status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }


def handle_call_tool(params: Dict) -> Dict:
    """Handle tools/call request"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    log_debug(f"Tool call: {tool_name} with {arguments}")
    
    try:
        if tool_name == "analyze_topic":
            result = call_aegis_api("/mcp/analyze_topic", "POST", {
                "topic": arguments.get("topic"),
                "detail": arguments.get("detail", "standard"),
                "domain": arguments.get("domain"),
                "max_claims": arguments.get("max_claims", 200)
            })
            
        elif tool_name == "assess_source":
            result = call_aegis_api("/mcp/assess_source", "POST", {
                "source_identifier": arguments.get("source_identifier"),
                "detail": arguments.get("detail", "standard")
            })
            
        elif tool_name == "get_perspectives":
            result = call_aegis_api("/mcp/get_perspectives", "POST", {
                "topic": arguments.get("topic"),
                "max_clusters": arguments.get("max_clusters", 5),
                "claims_per_cluster": arguments.get("claims_per_cluster", 5)
            })
            
        elif tool_name == "get_claim_context":
            result = call_aegis_api("/mcp/get_claim_context", "POST", {
                "claim_id": arguments.get("claim_id"),
                "include_graph": arguments.get("include_graph", False)
            })
            
        elif tool_name == "list_domains":
            result = call_aegis_api("/mcp/list_domains", "GET")
            
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}"
                }],
                "isError": True
            }
        
        # Format successful response
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        }
        
    except Exception as e:
        log_debug(f"Tool error: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }],
            "isError": True
        }


def read_message() -> Optional[Dict]:
    """Read a JSON-RPC message from stdin using Content-Length framing"""
    # Read headers
    headers = {}
    while True:
        line = sys.stdin.readline()
        if line == '\r\n' or line == '\n' or line == '':
            break
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip().lower()] = value.strip()
    
    # Get content length
    content_length = headers.get('content-length')
    if not content_length:
        return None
    
    # Read content
    content = sys.stdin.read(int(content_length))
    if not content:
        return None
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        log_debug(f"Invalid JSON: {content}")
        return None


def main():
    """Main loop - process MCP messages"""
    log_debug(f"Aegis MCP stdio wrapper started")
    log_debug(f"Connecting to: {AEGIS_MCP_URL}")
    
    # Verify server is running
    try:
        health = call_aegis_api("/health", "GET")
        log_debug(f"Server health: {health}")
    except Exception as e:
        log_debug(f"Warning: Could not connect to Aegis server: {e}")
    
    while True:
        try:
            message = read_message()
            if message is None:
                break
            
            msg_id = message.get("id")
            method = message.get("method")
            params = message.get("params", {})
            
            log_debug(f"Received: {method}")
            
            # Handle methods
            if method == "initialize":
                result = handle_initialize(params)
                send_response({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result
                })
                
            elif method == "notifications/initialized":
                # No response needed for notifications
                pass
                
            elif method == "tools/list":
                result = handle_list_tools(params)
                send_response({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result
                })
                
            elif method == "tools/call":
                result = handle_call_tool(params)
                send_response({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result
                })
                
            elif method == "ping":
                send_response({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {}
                })
                
            else:
                log_debug(f"Unknown method: {method}")
                send_error(msg_id, -32601, f"Method not found: {method}")
                
        except Exception as e:
            log_debug(f"Error processing message: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
