# Aegis Insight

**Multi-Dimensional Knowledge Graph Analytics with Pattern Recognition and Topology Analysis**

[![CI/CD Pipeline](https://github.com/Eleutherios-project/Eleutherios/actions/workflows/ci.yml/badge.svg)](https://github.com/Eleutherios-project/Eleutherios/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Overview

Aegis Insight is a knowledge graph analytics platform that enables researchers to explore epistemic topology—the structural relationships between claims, sources, and narrative patterns within large document corpora.

Unlike traditional search systems that return documents, Aegis Insight extracts structured claims from unstructured text, builds a queryable knowledge graph, and provides analytical tools to examine the shape and structure of knowledge itself.

### Key Features

- **Multi-Dimensional Extraction**: Seven specialized extractors process documents to identify entities, claims, temporal data, geographic references, citations, emotional content, and authority indicators
- **Local-First Processing**: All LLM inference runs locally via Ollama—no data sent to external APIs
- **Pattern Detection**: Algorithms identify structural patterns in knowledge networks
- **Interactive Visualization**: Real-time, force-directed graph visualization with D3.js
- **OCR Support**: Tesseract integration for scanned documents and historical archives
- **MCP Integration**: Model Context Protocol server for AI assistant integration

## Quick Start

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.com/) installed on host machine
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU recommended for faster processing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Eleutherios-project/Eleutherios.git
cd Eleutherios

# 2. Install required Ollama models
ollama pull mistral-nemo:12b
ollama pull nomic-embed-text

# 3. Run setup (loads ~460MB demo data)
chmod +x setup.sh
./setup.sh

# 4. Start the services
docker-compose up -d

# 5. Open the web interface (wait ~1 min for first-time init)
open http://localhost:8001
```

### Demo Data Included

The setup includes a pre-loaded demo dataset:
- **38,469 claims** extracted from 81 documents
- **17,584 entities** (people, organizations, places, concepts)  
- **43,528 vector embeddings** for semantic search

Try searching: `Smedley Butler`, `Remember the Maine`, or `Thomas Paine`

To start fresh without demo data:
```bash
./setup.sh --no-demo
```

### Verify Installation

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                         │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│  │   Neo4j     │  │ PostgreSQL  │  │    Aegis API Server   │   │
│  │   :7474     │  │   :5432     │  │        :8001          │   │
│  │   :7687     │  │  + pgvector │  │  (FastAPI + Web UI)   │   │
│  └─────────────┘  └─────────────┘  └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ connects to host
                              ▼
                    ┌─────────────────┐
                    │  Ollama (host)  │
                    │     :11434      │
                    └─────────────────┘
```

## Documentation

- [Technical Reference Manual](docs/Aegis_Insight_Technical_Reference.pdf) - Comprehensive documentation
- [Quick Start Guide](docs/QUICK_START.md) - Get up and running fast
- [API Reference](docs/API.md) - REST API documentation
- [MCP Integration](docs/MCP.md) - Model Context Protocol setup

## Usage

### Loading Documents

1. Navigate to the **Data Loading** tab
2. Drag and drop documents (PDF, TXT, DOCX, MD)
3. Click **Start Import**
4. Monitor the three-phase extraction process

### Searching

1. Enter search terms in the search bar
2. Use semantic search to find related claims
3. Click nodes to view details
4. Explore relationships in the graph

### Pattern Detection

1. Select detection mode (Suppression, Coordination, or Anomaly)
2. Enter a topic to analyze
3. Review detection scores and signals
4. Examine affected claims

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://neo4j:7687` |
| `NEO4J_PASSWORD` | Neo4j password | `aegistrusted` |
| `POSTGRES_HOST` | PostgreSQL host | `postgres` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `aegis_trusted_2025` |
| `OLLAMA_HOST` | Ollama API URL | `http://host.docker.internal:11434` |

### Resource Requirements

| Configuration | RAM | Storage | GPU |
|---------------|-----|---------|-----|
| Minimum | 16GB | 50GB | None |
| Recommended | 32GB | 200GB | 8GB VRAM |
| Production | 64GB+ | 500GB+ | 24GB+ VRAM |

## Security

- All containers run as non-root users
- Read-only filesystem where possible
- No data sent to external APIs
- Dependency vulnerability scanning via Dependabot
- Container scanning via Trivy
- Code analysis via CodeQL
- Security Note: Default credentials are used for local deployment. These services are not exposed outside the Docker network. If you expose ports externally, change the passwords in .env.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Neo4j](https://neo4j.com/) - Graph database
- [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) - Vector storage
- [Ollama](https://ollama.com/) - Local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [D3.js](https://d3js.org/) - Graph visualization

---

**Cedrus Strategic LLC** 
