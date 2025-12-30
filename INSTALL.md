# Aegis Insight - Installation Guide

**Version:** 1.0  
**Last Updated:** December 2024

---

## Quick Start (5 Minutes)

### Prerequisites

You need these installed:
- **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop)
- **Ollama** - [Download](https://ollama.com/download) (for AI features)

### Installation

```bash
# 1. Clone or download
git clone https://github.com/Eleutherios-project/Eleutherios.git
cd Eleutherios

# 2. Start services
docker-compose up -d

# 3. Wait for initialization (~2 minutes first time)
docker-compose logs -f api
# (Press Ctrl+C when you see "Uvicorn running")

# 4. Open in browser
open http://localhost:8001
```

That's it! The demo data loads automatically.

---

## Detailed Installation

### Step 1: Install Docker Desktop

1. Download Docker Desktop for your OS:
   - [Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Linux](https://docs.docker.com/desktop/install/linux-install/)

2. Install and start Docker Desktop

3. Verify installation:
   ```bash
   docker --version
   docker-compose --version
   ```

### Step 2: Install Ollama (Optional but Recommended)

Ollama provides local AI processing for document analysis.

1. Download from [ollama.com/download](https://ollama.com/download)

2. Install and start Ollama

3. Pull the required model:
   ```bash
   ollama pull mistral-nemo:12b
   ```

4. Verify:
   ```bash
   ollama list
   ```

**Note:** Aegis Insight works without Ollama, but document import requires it.

### Step 3: Download Aegis Insight

**Option A: Git Clone**
```bash
git clone https://github.com/Eleutherios-project/Eleutherios.git
cd Eleutherios
```

**Option B: Download ZIP**
1. Go to [github.com/Eleutherios-project/Eleutherios](https://github.com/Eleutherios-project/Eleutherios)
2. Click "Code" → "Download ZIP"
3. Extract to a folder
4. Open terminal in that folder

### Step 4: Start Services

```bash
# Start all services
docker-compose up -d

# Watch the startup logs
docker-compose logs -f

# (Press Ctrl+C when initialization completes)
```

First startup takes 2-5 minutes to:
- Download container images
- Initialize databases
- Load demo data

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
| aegis-api | Up |

### Check Demo Data Loaded

1. Open http://localhost:8001
2. Go to the "Detection" tab
3. Enter a search term (try "war" or "prohibition")
4. You should see results

### Troubleshooting

**Container won't start?**
```bash
# Check logs
docker-compose logs neo4j
docker-compose logs postgres
docker-compose logs api

# Common fix: restart Docker Desktop
```

**Can't connect to http://localhost:8001?**
```bash
# Check port isn't in use
lsof -i :8001

# Check container is running
docker-compose ps
```

**No data showing?**
```bash
# Check Neo4j has data
docker-compose exec neo4j cypher-shell -u neo4j -p aegistrusted \
  "MATCH (n) RETURN count(n)"

# Should return a number > 0
```

---

## Configuration

### Default Ports

| Service | Port | Purpose |
|---------|------|---------|
| API | 8001 | Web interface and REST API |
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

**If Ollama runs on the same machine:**
- Linux/Mac: Uses `host.docker.internal` automatically
- Windows: Uses `host.docker.internal` automatically

**If Ollama runs on a different machine:**
Edit `docker-compose.yml`:
```yaml
services:
  api:
    environment:
      - OLLAMA_HOST=http://192.168.1.100:11434
```

---

## Importing Your Own Data

### Quick Import (< 50 PDFs)

1. Open http://localhost:8001
2. Go to "Data Loading" tab
3. Follow the wizard

### Batch Import (50+ PDFs)

1. Copy PDFs to `./data/inbox/`
2. Run import:
   ```bash
   docker-compose exec api python3 /app/run_extraction_pipeline.py \
     --input /app/data/inbox \
     --output /app/data/processed
   ```

See [Data Loading Guide](DATA_LOADING_GUIDE.md) for details.

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

## Getting Help

- **GitHub Issues:** [Report a bug](https://github.com/Eleutherios-project/Eleutherios/issues)
- **Documentation:** See `/docs` folder
- **API Reference:** http://localhost:8001/docs (when running)

---

## License

MIT License - See [LICENSE](LICENSE) file.

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

# Set SEED_ON_FIRST_RUN=false in docker-compose.yml if you want empty databases
# Or leave true to reload demo data

# Start fresh
docker-compose up -d
```

### Data Directories

These directories are mounted from your host for easy access:

- `./data/inbox/` - Place PDF files here for ingestion
- `./data/processed/` - Processed files are moved here
- `./data/calibration_profiles/` - Detection calibration profiles

