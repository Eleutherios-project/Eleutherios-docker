#!/bin/bash
#
# Aegis Insight - Docker Container Entrypoint
# =============================================
#
# This script:
# 1. Waits for database services to be ready
# 2. Seeds demo data on first run
# 3. Starts the API server
#

set -e

echo "=================================================="
echo "  Aegis Insight"
echo "=================================================="
echo ""

# Configuration
NEO4J_HOST="${NEO4J_HOST:-neo4j}"
NEO4J_PORT="${NEO4J_BOLT_PORT:-7687}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
API_PORT="${API_PORT:-8001}"
SEED_ON_FIRST_RUN="${SEED_ON_FIRST_RUN:-true}"

# ============================================================
# Function: Wait for a service to be ready
# ============================================================
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=${4:-30}
    local attempt=1

    echo -n "Waiting for $service_name ($host:$port)..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo " ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo " TIMEOUT!"
    return 1
}

# ============================================================
# Function: Check if Neo4j has data
# ============================================================
neo4j_has_data() {
    local count
    count=$(python3 -c "
from neo4j import GraphDatabase
import os
uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
user = os.environ.get('NEO4J_USER', 'neo4j')
password = os.environ.get('NEO4J_PASSWORD', 'aegistrusted')
driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as session:
    result = session.run('MATCH (n) RETURN count(n) as count')
    print(result.single()['count'])
driver.close()
" 2>/dev/null || echo "0")
    
    [ "$count" -gt 0 ]
}

# ============================================================
# Function: Initialize databases
# ============================================================
initialize_databases() {
    echo ""
    echo "Initializing databases..."
    
    # PostgreSQL: Create required tables/extensions
    echo "Setting up PostgreSQL..."
    python3 -c "
import psycopg2
import os

host = os.environ.get('POSTGRES_HOST', 'postgres')
db = os.environ.get('POSTGRES_DB', 'aegis_insight')
user = os.environ.get('POSTGRES_USER', 'aegis')
password = os.environ.get('POSTGRES_PASSWORD', 'aegis_trusted_2025')

conn = psycopg2.connect(host=host, database=db, user=user, password=password)
cur = conn.cursor()

# Enable pgvector extension
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

# Create claim_embeddings table if not exists
cur.execute('''
CREATE TABLE IF NOT EXISTS claim_embeddings (
    claim_id TEXT PRIMARY KEY,
    claim_text TEXT,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Create topics table if not exists
cur.execute('''
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    document_id TEXT,
    topic_name TEXT,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Create topic_embeddings table if not exists
cur.execute('''
CREATE TABLE IF NOT EXISTS topic_embeddings (
    id SERIAL PRIMARY KEY,
    topic_name TEXT UNIQUE,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
cur.close()
conn.close()
print('PostgreSQL initialized successfully')
" || echo "Warning: PostgreSQL initialization had issues"

    echo "Databases initialized!"
}

# ============================================================
# Function: Seed demo data
# ============================================================
seed_demo_data() {
    if [ -d "/app/demo-data" ]; then
        echo ""
        echo "Checking for demo data..."
        
        if neo4j_has_data; then
            echo "Database already has data, skipping demo import"
        else
            echo "Importing demo data (this may take a few minutes)..."
            python3 /app/scripts/seed_demo_data.py || echo "Warning: Demo data seeding had issues"
        fi
    else
        echo "No demo data directory found, starting with empty database"
    fi
}

# ============================================================
# Main
# ============================================================

# Wait for services
wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT" "PostgreSQL" 60
wait_for_service "$NEO4J_HOST" "$NEO4J_PORT" "Neo4j" 60

# Additional wait for Neo4j to fully initialize
echo "Waiting for Neo4j to complete initialization..."
sleep 5

# Initialize databases
initialize_databases

# Seed demo data on first run
if [ "$SEED_ON_FIRST_RUN" = "true" ]; then
    seed_demo_data
fi

# Start MCP server (if enabled)
MCP_ENABLED=${ENABLE_MCP_SERVER:-true}
if [ "$MCP_ENABLED" = "true" ]; then
    echo "Starting MCP server on port ${MCP_PORT:-8100}..."
    python3 aegis_mcp_server.py &
    sleep 2
    echo "✓ MCP server started"
fi

# Start API server
echo ""
echo "=================================================="
echo "  Starting API Server"
echo "=================================================="
echo ""
echo "  Web Interface:  http://localhost:${API_PORT}"
echo "  API Docs:       http://localhost:${API_PORT}/docs"
echo "  Health Check:   http://localhost:${API_PORT}/api/health"
  if [ "$MCP_ENABLED" = "true" ]; then
    echo "  MCP Server:     http://localhost:${MCP_PORT:-8100}"
  fi
echo ""
echo "=================================================="
echo ""

# Run the API server
exec python3 api_server.py
