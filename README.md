# Eleutherios - Aegis Insight Engine

**Multi-Dimensional Knowledge Graph Analytics — See how information actually flows.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aegis Insight extracts structured knowledge from documents, builds a queryable graph of claims and entities, and provides analytical tools to detect suppression patterns, coordinated messaging, and manufactured consensus.

**🌐 Website:** [eleutherios.io](https://eleutherios.io)  
**📖 Documentation:** [aegisinsight.net](https://aegisinsight.net)

---

## What It Does

- **Knowledge Topology** — See citation flow, not just content. Understand who references whom and where information originates.
- **Suppression Detection** — Identify when credible voices are systematically marginalized.
- **Coordination Detection** — Detect synchronized messaging through temporal clustering and language analysis.
- **Anomaly Detection** — Find patterns that deviate from expected baselines.
- **Local-First** — Runs on your hardware. No cloud dependency, no data leaving your machine.

---

## Quick Start

### Prerequisites by Platform

| Platform | Requirements |
|----------|-------------|
| **Windows** | WSL2 + Docker Desktop + Ollama ([detailed guide](WINDOWS_SETUP_GUIDE.md)) |
| **Mac** | Docker Desktop + Ollama |
| **Linux** | Docker + Docker Compose + Ollama |

### 1. Install Prerequisites

**All Platforms — Install Ollama:**
```bash
# Download from https://ollama.com/download
# Then pull the required models:
ollama pull mistral-nemo:12b
ollama pull nomic-embed-text
```

**Windows Users:** You must set up WSL2 first. See [Windows Setup Guide](WINDOWS_SETUP_GUIDE.md).

**Mac/Linux — Install Docker:**
- Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- Linux: [Docker Engine](https://docs.docker.com/engine/install/)

### 2. Clone and Start

> **Important:** Make sure Docker Desktop is running before executing these commands.

```bash
# Clone the repository
git clone https://github.com/Eleutherios-project/Eleutherios-docker.git
cd Eleutherios-docker

# Start all services
docker-compose up -d

# Watch startup (Ctrl+C when ready)
docker-compose logs -f api
```

First startup takes 5-10 minutes to:
- Download container images (~2GB)
- Initialize databases
- Load demo data (38K claims, 143K graph records)

### 3. Open in Browser

**http://localhost:8001**

Try searching for: `Smedley Butler`, `Thomas Paine`, `Remember the Maine`

---

## Windows Setup

Windows requires WSL2 (Windows Subsystem for Linux) to run Docker containers efficiently.

### Step 1: Enable WSL2

Open PowerShell as Administrator:
```powershell
wsl --install
```

Restart your computer when prompted. After restart, Ubuntu will launch — create a username and password.

### Step 2: Install Docker Desktop

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. During install, check **"Use WSL 2 instead of Hyper-V"**
3. After install, open Docker Desktop → Settings → Resources → WSL Integration
4. Enable integration for your Ubuntu distro
5. Click "Apply & Restart"

### Step 3: Install NVIDIA GPU Support (Optional, Recommended)

If you have an NVIDIA GPU:

1. Install latest [NVIDIA Windows Driver](https://www.nvidia.com/Download/index.aspx)
2. In Ubuntu (WSL), install the container toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

3. Verify with `nvidia-smi`

### Step 4: Install Ollama in WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull mistral-nemo:12b
ollama pull nomic-embed-text
```

### Step 5: Continue with Quick Start

Now follow the [Clone and Start](#2-clone-and-start) steps above from within your Ubuntu/WSL terminal.

For the complete Windows guide with troubleshooting, see [WINDOWS_SETUP_GUIDE.md](WINDOWS_SETUP_GUIDE.md).

---

## Mac Setup

### Step 1: Install Docker Desktop

Download and install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/).

### Step 2: Install Ollama

Download from [ollama.com/download](https://ollama.com/download) or:

```bash
brew install ollama
```

Start Ollama and pull the models:
```bash
ollama serve &
ollama pull mistral-nemo:12b
ollama pull nomic-embed-text
```

### Step 3: Continue with Quick Start

Follow the [Clone and Start](#2-clone-and-start) steps above.

---

## Linux Setup

### Step 1: Install Docker

Follow the [official Docker installation guide](https://docs.docker.com/engine/install/) for your distro.

### Step 2: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull mistral-nemo:12b
ollama pull nomic-embed-text
```

### Step 3: Continue with Quick Start

Follow the [Clone and Start](#2-clone-and-start) steps above.

---

## Verifying Installation

### Check Services Running

```bash
docker-compose ps
```

Expected output:
```
NAME            STATUS
aegis-neo4j     Up (healthy)
aegis-postgres  Up (healthy)
aegis-api       Up (healthy)
```

### Check Demo Data

```bash
# Node count (should be ~60,000)
docker-compose exec neo4j cypher-shell -u neo4j -p aegistrusted \
  "MATCH (n) RETURN count(n)"

# Relationship count (should be ~83,000)
docker-compose exec neo4j cypher-shell -u neo4j -p aegistrusted \
  "MATCH ()-[r]->() RETURN count(r)"
```

### Test Detection

1. Open http://localhost:8001
2. Go to Detection tab
3. Select "Suppression" mode
4. Search for "Thomas Paine"
5. Should return ~0.83 score (CRITICAL level)

---

## Claude Desktop Integration (MCP)

Aegis includes an MCP (Model Context Protocol) server that allows Claude Desktop to directly query your knowledge graph.

### Quick Setup

1. Copy the example config:
   ```bash
   cat examples/claude_desktop_config.json
   ```

2. Add to your Claude Desktop config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

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
   - "Show me perspectives on the Business Plot"

For detailed configuration and troubleshooting, see [docs/CLAUDE_DESKTOP_MCP_CONFIG.md](docs/CLAUDE_DESKTOP_MCP_CONFIG.md).

---

## Configuration

### Ports

| Service | Port | Purpose |
|---------|------|---------|
| Web UI / API | 8001 | Main interface |
| MCP Server | 8100 | Claude Desktop integration |
| Neo4j Browser | 7474 | Database admin |
| Neo4j Bolt | 7687 | Database protocol |
| PostgreSQL | 5432 | Embeddings storage |

### Environment Variables

Edit `docker-compose.yml`:

```yaml
environment:
  - SEED_ON_FIRST_RUN=true      # Load demo data on first run
  - ENABLE_MCP_SERVER=true      # Start MCP server for AI integration
  - OLLAMA_HOST=http://host.docker.internal:11434  # Ollama location
```

### Data Directories

```
./data/inbox/                 # Place PDFs here for import
./data/processed/             # Processed files move here
./data/calibration_profiles/  # Detection tuning profiles
```

---

## Common Commands

```bash
# Start services (Docker Desktop must be running first!)
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Restart
docker-compose restart

# Full reset (removes all data)
docker-compose down -v
docker-compose up -d
```

---

## Importing Your Own Data

1. Place PDF files in `./data/inbox/`
2. Open http://localhost:8001
3. Go to "Data Loading" tab
4. Follow the wizard

Processing time depends on document count and GPU availability:
- With GPU: ~2-60 minutes per PDF (A6000 class performs faster, 3060 laptop may experience 2x-4x processing times)
- CPU only: significant delays expected and is not recommended

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
- GPU: NVIDIA with 8GB+ VRAM

---

## Troubleshooting

### "Cannot connect to Docker daemon"
Launch Docker Desktop from Start menu (Windows) or Applications (Mac) and wait for it to start.

### Container won't start
```bash
docker-compose logs neo4j
docker-compose logs postgres
docker-compose logs api
```

### Containers persist after docker-compose down
```bash
# Stop by name if started with different config
docker stop aegis-api aegis-postgres aegis-neo4j
docker rm aegis-api aegis-postgres aegis-neo4j
```

### Port already in use
```bash
# Find what's using port 8001
lsof -i :8001  # Mac/Linux
netstat -ano | findstr :8001  # Windows
```

### Ollama not accessible
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start if needed
ollama serve &
```

### Ollama not reachable from containers
```bash
# Test from inside container
docker exec aegis-api curl http://host.docker.internal:11434/api/tags
```

### Windows: GPU not detected in WSL
```powershell
# In PowerShell, restart WSL
wsl --shutdown
wsl
nvidia-smi
```

---

## Architecture

```
+-------------------------------------------------------------+
|                      Your Computer                          |
|  +-------------------------------------------------------+  |
|  |                 Docker Containers                     |  |
|  |  +---------+  +----------+  +---------------------+   |  |
|  |  |  Neo4j  |  |PostgreSQL|  |     Aegis API       |   |  |
|  |  | :7474   |  |  :5432   |  | :8001 (UI + REST)   |   |  |
|  |  | :7687   |  |          |  | :8100 (MCP Server)  |   |  |
|  |  +---------+  +----------+  +---------------------+   |  |
|  +--------------------------|----------------------------+  |
|                             | host.docker.internal          |
|  +--------------------------v----------------------------+  |
|  |               Ollama (host)  :11434                   |  |
|  |        mistral-nemo:12b, nomic-embed-text             |  |
|  +-------------------------------------------------------+  |
|                             |                               |
|                      +------v------+                        |
|                      | NVIDIA GPU  | (optional)             |
|                      +-------------+                        |
+-------------------------------------------------------------+
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Eleutherios-project/Eleutherios-docker/issues)
- **Documentation:** [eleutherios.io](https://eleutherios.io)
- **API Reference:** http://localhost:8001/docs (when running)
- **MCP Config:** See [docs/CLAUDE_DESKTOP_MCP_CONFIG.md](docs/CLAUDE_DESKTOP_MCP_CONFIG.md)

---

## License

MIT License — See [LICENSE](LICENSE) file.

---

*Aegis Insight — Multi-Dimensional Knowledge Graph Analytics*  
*See how information actually flows.*
