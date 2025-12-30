#!/bin/bash
#
# Aegis Insight - Quick Setup
#

set -e

echo ""
echo "=================================================="
echo "  Aegis Insight - Setup"
echo "=================================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running"
    echo "Please start Docker Desktop"
    exit 1
fi

echo "✓ Docker is ready"

# Check Ollama (optional)
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
else
    echo "⚠ Ollama not found (optional - needed for document import)"
fi

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check services
echo ""
echo "Checking services..."
docker-compose ps

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "  Web Interface:  http://localhost:8001"
echo "  Neo4j Browser:  http://localhost:7474"
echo ""
echo "  View logs:      docker-compose logs -f"
echo "  Stop:           docker-compose down"
echo ""
