# Aegis Insight - Installation Guide

**Version:** 1.1  
**Last Updated:** January 2025

---

## Quick Start (5 Minutes)

### Prerequisites

You need these installed:
- **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop) (**must be running before you start**)
- **Ollama** - [Download](https://ollama.com/download) (for AI features)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Eleutherios-project/Eleutherios-docker.git
cd Eleutherios-docker

# 2. Start Docker Desktop first, then start services
docker-compose up -d

# 3. Wait for initialization (~2-5 minutes first time)
docker-compose logs -f api
# (Press Ctrl+C when you see "Uvicorn running")

# 4. Open in browser
open http://localhost:8001
```

That's it! The demo data loads automatically on first run.

---

## Detailed Installation

### Step 1: Install Docker Desktop

> **Important:** Docker Desktop must be running before you execute any docker commands.

1. Download Docker Desktop for your OS:
   - [Windows](https://docs.docker.com/desktop/install/windows-install/) (requires WSL2 - see [Windows Setup Guide](WINDOWS_SETUP_GUIDE.md))
   - [Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Linux](https://docs.docker.com/desktop/install/linux-install/)

2. Install and **start Docker Desktop**

3. Verify installation:
   ```bash
   docker --version
   docker-compose --version
   ```

### Step 2: Install Ollama 

Ollama provides local AI processing for document analysis.

1. Download from [ollama.com/download](https://ollama.com/download)

2. Install and start Ollama

3. Pull the required models:
   ```bash
   # Primary extraction model (~7GB)
   ollama pull mistral-nemo:12b
   
   # Embedding model for semantic search (~274MB)
   ollama pull nomic-embed-text
   ```

4. Verify:
   ```bash
   ollama list
   # Should show both mistral-nemo:12b and nomic-embed-text
   ```

**Note:** Aegis Insight works without Ollama for browsing/searching, but document import and LLM-powered queries require it.

### Step 3: Download Aegis Insight

**Option A: Git Clone (Recommended)**
```bash
git clone https://github.com/Eleutherios-project/Eleutherios-docker.git
cd Eleutherios-docker
```

> **For specific versions:** Check out a release branch:
> ```bash
> git checkout release/v1.1
> ```

**Option B: Download ZIP**
1. Go to [github.com/Eleutherios-project/Eleutherios-docker](https://github.com/Eleutherios-project/Eleutherios-docker)
2. Click "Code" → "Download ZIP"
3. Extract to a folder
4. Open terminal in that folder

### Step 4: Start Services

```bash
# Make sure Docker Desktop is running first!

# Start all services
docker-compose up -d

# Watch the startup logs
docker-compose logs -f api

# (Press Ctrl+C when initialization completes)
```

First startup takes 5-10 minutes to:
- Download container images
- Initialize databases
- Load demo data (~38,000 claims)

### Step 5: Access the Interface

Open your browser to: **http://localhost:8001**

You should see the Aegis Insight interface with demo data pre-loaded.

---

## Verification

### Check All Services Running

```bash
docker-compose ps
```

You should see:
| Service | Status |
|---------|--------|
| aegis-neo4j | Up (healthy) |
| aegis-postgres | Up (healthy) |
| aegis-api | Up (healthy) |

### Check Demo Data Loaded

1. Open http://localhost:8001
2. Go to the "Detection" tab
3. Search for "Thomas Paine"
4. Run suppression detection
5. Expected: ~0.83 CRITICAL score

### Test API Health

```bash
curl http://localhost:8001/api/health
```

### Troubleshooting

**"Cannot connect to Docker daemon"**
- Docker Desktop isn't running. Launch it and wait for it to fully start.

**Container won't start?**
```bash
# Check logs for specific service
docker-compose logs neo4j
docker-compose logs postgres
docker-compose logs api
```

**Containers persist after `docker-compose down`?**
```bash
# Stop by name if they were started with different config
docker stop aegis-api aegis-postgres aegis-neo4j
docker rm aegis-api aegis-postgres aegis-neo4j
```

**Can't connect to http://localhost:8001?**
```bash
# Check port isn't in use
lsof -i :8001  # Mac/Linux
netstat -ano | findstr :8001  # Windows

# Check container is running
docker-compose ps
```

**No data showing?**
```bash
# Check Neo4j has data
docker-compose exec neo4j cypher-shell -u neo4j -p aegistrusted \
  "MATCH (n) RETURN count(n)"

# Should return ~60,000+
```

---

## Configuration

### Default Ports

| Service | Port | Purpose |
|---------|------|---------|
| Web UI / API | 8001 | Main interface and REST API |
| MCP Server | 8100 | Claude Desktop integration |
| Neo4j Browser | 7474 | Database admin (optional) |
| Neo4j Bolt | 7687 | Database connection |
| PostgreSQL | 5432 | Embeddings database |

### Changing Ports

Edit `docker-compose.yml`:
```yaml
services:
  api:
    ports:
      - "8080:8001"  # Change 8080 to your preferred port
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

### Connecting to Ollama

**Default (same machine):**
Uses `host.docker.internal` automatically on Windows/Mac/Linux.

**Remote Ollama server:**
Edit `docker-compose.yml`:
```yaml
services:
  api:
    environment:
      - OLLAMA_HOST=http://192.168.1.100:11434
```

---

## Claude Desktop Integration (MCP)

Aegis includes an MCP server for Claude Desktop integration.

### Quick Setup

1. Find your Claude config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. Add the Aegis server (or copy from `examples/claude_desktop_config.json`):
   ```json
   {
     "mcpServers": {
       "aegis-insight": {
         "url": "http://localhost:8001/mcp",
         "transport": "http"
       }
     }
   }
   ```

3. Restart Claude Desktop

4. Try asking Claude:
   - "What domains are available in Aegis?"
   - "Analyze Thomas Paine for suppression patterns"

For detailed configuration options, see [docs/CLAUDE_DESKTOP_MCP_CONFIG.md](docs/CLAUDE_DESKTOP_MCP_CONFIG.md).

---

## Importing Your Own Data

### Quick Import (< 10 PDFs)

1. Open http://localhost:8001
2. Go to "Data Loading" tab
3. Follow the wizard

### Batch Import (10+ PDFs)

1. Copy PDFs to `./data/inbox/`
2. Run import wizard:
   ```bash
   docker-compose exec api python3 /app/aegis_import_wizard.py
   ```

Follow the wizard instructions for GPU parallelism and resuming partially completed jobs.

---

## Stopping and Restarting

### Stop Services
```bash
docker-compose down
```

### Stop and Remove Data
```bash
docker-compose down -v  # -v removes volumes/data
```

### Restart Services
```bash
docker-compose restart
```

---

## Updating

```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose build --no-cache

# Restart
docker-compose down
docker-compose up -d
```

---

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB free
- OS: Windows 10+, macOS 10.15+, Linux

### Recommended
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 200 GB SSD
- GPU: NVIDIA with 8GB+ VRAM (for faster AI processing)

---

## Configuration Options

### Demo Data Loading

By default, Aegis Insight loads demo data on first run. To control this:
```yaml
# In docker-compose.yml
environment:
  - SEED_ON_FIRST_RUN=true   # Load demo data (default)
  - SEED_ON_FIRST_RUN=false  # Start with empty database
```

### Starting Fresh

To completely reset and start with a clean system:
```bash
# Stop and remove all data
docker-compose down -v

# Start fresh (will reload demo data by default)
docker-compose up -d
```

### Data Directories

These directories are mounted from your host for easy access:

- `./data/inbox/` - Place PDF files here for ingestion
- `./data/processed/` - Processed files are moved here
- `./data/calibration_profiles/` - Detection calibration profiles

---

## Getting Help

- **GitHub Issues:** [Report a bug](https://github.com/Eleutherios-project/Eleutherios-docker/issues)
- **Documentation:** See `docs/` folder
- **MCP Integration:** See `docs/CLAUDE_DESKTOP_MCP_CONFIG.md`
- **Windows Guide:** See `WINDOWS_SETUP_GUIDE.md`
- **API Reference:** http://localhost:8001/docs (when running)

---

## License

MIT License - See [LICENSE](LICENSE) file.
