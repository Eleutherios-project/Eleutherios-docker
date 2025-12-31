# Eleutherios - Aegis Insight Engine

**Epistemic Defense Infrastructure — See how information actually flows.**

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
| **Windows** | WSL2 + Docker Desktop + Ollama ([detailed guide](#windows-setup)) |
| **Mac** | Docker Desktop + Ollama |
| **Linux** | Docker + Docker Compose + Ollama |

### 1. Install Prerequisites

**All Platforms — Install Ollama:**
```bash
# Download from https://ollama.com/download
# Then pull the required model:
ollama pull mistral-nemo:12b
```

**Windows Users:** You must set up WSL2 first. See [Windows Setup Guide](#windows-setup) below.

**Mac/Linux — Install Docker:**
- Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- Linux: [Docker Engine](https://docs.docker.com/engine/install/)

### 2. Clone and Start

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
- Load demo data (38K claims, 83K relationships)

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

Start Ollama and pull the model:
```bash
ollama serve &
ollama pull mistral-nemo:12b
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
aegis-api       Up
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
4. Search for "Smedley Butler"
5. Should return ~0.78 score (CRITICAL level)

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
./data/inbox/              # Place PDFs here for import
./data/processed/          # Processed files move here
./data/calibration_profiles/  # Detection tuning profiles
```

---

## Common Commands

```bash
# Start services
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
- With GPU: ~2-3 minutes per PDF
- CPU only: ~8-10 minutes per PDF

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

### Container won't start
```bash
docker-compose logs neo4j
docker-compose logs postgres
docker-compose logs api
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

### Windows: "Cannot connect to Docker daemon"
Launch Docker Desktop from Start menu and wait for it to start.

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
┌─────────────────────────────────────────────────────────────┐
│                     Your Computer                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 Docker Containers                      │  │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────────────────┐     │  │
│  │  │  Neo4j  │ │PostgreSQL│ │     Aegis API       │     │  │
│  │  │ :7474   │ │  :5432   │ │ :8001 (UI + REST)   │     │  │
│  │  │ :7687   │ │          │ │ :8100 (MCP Server)  │     │  │
│  │  └─────────┘ └──────────┘ └─────────────────────┘     │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Ollama (host)  :11434                     │  │
│  │              mistral-nemo:12b                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                   │
│                    ┌────┴────┐                              │
│                    │NVIDIA GPU│ (optional, recommended)     │
│                    └─────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Eleutherios-project/Eleutherios-docker/issues)
- **Documentation:** [eleutherios.io](https://eleutherios.io)
- **API Reference:** http://localhost:8001/docs (when running)

---

## License

MIT License — See [LICENSE](LICENSE) file.

---

*Aegis Insight — Epistemic Defense Infrastructure*  
*See how information actually flows.*
