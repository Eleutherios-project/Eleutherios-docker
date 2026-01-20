# Aegis Insight - Claude Desktop MCP Configuration
# ================================================
#
# This file provides example configurations for connecting Claude Desktop
# to your Aegis Insight instance via the Model Context Protocol (MCP).
#
# Location: 
#   - macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
#   - Windows: %APPDATA%\Claude\claude_desktop_config.json
#   - Linux: ~/.config/Claude/claude_desktop_config.json

# ============================================================================
# OPTION 1: HTTP Transport (Recommended for Docker deployments)
# ============================================================================
# 
# Use this configuration when running Aegis in Docker containers.
# The MCP server exposes an HTTP endpoint that Claude can connect to directly.
#
# Prerequisites:
#   - Aegis Docker containers running (docker-compose up -d)
#   - MCP server accessible at http://localhost:8001
#
# claude_desktop_config.json:

{
  "mcpServers": {
    "aegis-insight": {
      "url": "http://localhost:8001/mcp",
      "transport": "http",
      "description": "Aegis Insight - Epistemic analysis infrastructure"
    }
  }
}


# ============================================================================
# OPTION 2: Direct Python Module (For local development)
# ============================================================================
#
# Use this configuration when running Aegis directly on your host machine
# (not in Docker). Claude will spawn the MCP server as a subprocess.
#
# Prerequisites:
#   - Python environment with Aegis dependencies installed
#   - Neo4j and PostgreSQL running locally or accessible
#   - Ollama running locally or accessible
#
# claude_desktop_config.json:

{
  "mcpServers": {
    "aegis-insight": {
      "command": "python",
      "args": ["-m", "aegis_mcp_server"],
      "cwd": "/path/to/aegis-insight",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "aegistrusted",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "aegis",
        "POSTGRES_PASSWORD": "aegistrusted",
        "POSTGRES_DB": "aegis-postgres",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}


# ============================================================================
# OPTION 3: Docker Network (For host.docker.internal scenarios)
# ============================================================================
#
# Use this on macOS/Windows where Docker uses host.docker.internal.
#
# claude_desktop_config.json:

{
  "mcpServers": {
    "aegis-insight": {
      "url": "http://host.docker.internal:8001/mcp",
      "transport": "http"
    }
  }
}


# ============================================================================
# AVAILABLE MCP TOOLS
# ============================================================================
#
# Once configured, Claude will have access to these epistemic analysis tools:
#
# 1. list_domains
#    - Returns available knowledge domains with claim counts
#    - Use to understand what corpora are loaded
#
# 2. get_perspectives
#    - Returns clustered viewpoints on a topic
#    - Shows meta-ratios, source diversity, representative claims
#    - Parameters: topic (required), max_clusters, claims_per_cluster, domain
#
# 3. analyze_topic
#    - Returns suppression, coordination, and anomaly scores
#    - Includes detailed signal breakdowns with specific claims
#    - Parameters: topic (required), max_claims, detail, domain
#
# 4. assess_source
#    - Returns a source's position in the knowledge topology
#    - Shows citation network position, claim type distribution
#    - Parameters: source_identifier (required), detail
#
# 5. get_claim_context
#    - Returns full context for a specific claim
#    - Includes citation chains, temporal/geographic data
#    - Parameters: claim_id (required), include_graph
#
# ============================================================================
# EXAMPLE QUERIES TO TRY
# ============================================================================
#
# After configuring, try these prompts with Claude:
#
# "What domains are available in Aegis?"
#
# "Show me the different perspectives on Thomas Paine's Age of Reason"
#
# "Analyze the topic of Smedley Butler and the Business Plot for 
#  suppression patterns"
#
# "What can you tell me about the epistemic landscape around 
#  electrogravitics research?"
#
# "Assess the source topology for the FBI vault documents"
#
# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
# Connection refused:
#   - Verify Aegis containers are running: docker-compose ps
#   - Check MCP server logs: docker-compose logs aegis-mcp
#   - Ensure port 8001 is not blocked by firewall
#
# Timeout errors:
#   - Complex queries may take 15-30 seconds
#   - Reduce max_claims parameter for faster responses
#   - Check Ollama is running if using semantic search
#
# Empty results:
#   - Verify corpus is loaded: use list_domains first
#   - Try broader search terms
#   - Check Neo4j has data: MATCH (n) RETURN count(n)
#
# Linux Docker networking:
#   - host.docker.internal doesn't work on Linux
#   - Use your host machine's actual IP address
#   - Find it with: hostname -I | awk '{print $1}'
#
# ============================================================================
# SECURITY NOTES
# ============================================================================
#
# - Aegis runs entirely locally - no data leaves your machine
# - Default credentials (aegistrusted) should be changed for production
# - MCP endpoint has no authentication by default
# - Consider firewall rules if exposing to network
#
