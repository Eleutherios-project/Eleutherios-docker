# Aegis Insight - Windows Setup Guide

Complete guide to running Aegis Insight on Windows with WSL2 and GPU acceleration.

**Time Required:** 30-45 minutes  
**Difficulty:** Beginner-friendly

---

## Prerequisites

- Windows 10 (version 2004+) or Windows 11
- NVIDIA GPU (recommended, not required)
- 16GB RAM minimum (32GB recommended)
- 50GB free disk space
- Administrator access

---

## Step 1: Enable WSL2

WSL2 (Windows Subsystem for Linux) lets you run Linux inside Windows.

### 1.1 Open PowerShell as Administrator

1. Press `Win + X`
2. Click **"Windows Terminal (Admin)"** or **"PowerShell (Admin)"**

### 1.2 Install WSL2

```powershell
wsl --install
```

This installs WSL2 with Ubuntu. **Restart your computer when prompted.**

### 1.3 Complete Ubuntu Setup

After restart, Ubuntu will launch automatically. If not, search for "Ubuntu" in Start menu.

1. Wait for installation to complete
2. Create a username (lowercase, no spaces): e.g., `bob`
3. Create a password (you'll need this for `sudo` commands)

### 1.4 Verify WSL2

In PowerShell:
```powershell
wsl --list --verbose
```

You should see Ubuntu with VERSION 2.

---

## Step 2: Install NVIDIA GPU Support (Optional but Recommended)

Skip this section if you don't have an NVIDIA GPU.

### 2.1 Install/Update NVIDIA Windows Driver

1. Go to: https://www.nvidia.com/Download/index.aspx
2. Download and install the latest driver for your GPU
3. Restart Windows

### 2.2 Install NVIDIA Container Toolkit in WSL

Open Ubuntu (WSL) terminal:

```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 2.3 Verify GPU Access

```bash
nvidia-smi
```

You should see your GPU listed. If not, restart WSL:
```powershell
# In PowerShell (not WSL)
wsl --shutdown
wsl
```

---

## Step 3: Install Docker Desktop

### 3.1 Download Docker Desktop

1. Go to: https://www.docker.com/products/docker-desktop/
2. Click **"Download for Windows"**
3. Run the installer

### 3.2 Configure Docker Desktop

During installation:
- ✅ Check **"Use WSL 2 instead of Hyper-V"**
- ✅ Check **"Add shortcut to desktop"**

After installation:
1. **Restart Windows** when prompted
2. Launch Docker Desktop from Start menu
3. Accept the license agreement

### 3.3 Enable WSL Integration

1. Open Docker Desktop
2. Click ⚙️ **Settings** (gear icon)
3. Go to **Resources → WSL Integration**
4. Toggle ON for your Ubuntu distro
5. Click **"Apply & Restart"**

### 3.4 Verify Docker in WSL

Open Ubuntu terminal:
```bash
docker --version
docker-compose --version
```

Both should return version numbers.

Test Docker:
```bash
docker run hello-world
```

---

## Step 4: Install Ollama

Ollama runs the local AI models for extraction and analysis.

### 4.1 Install Ollama

In Ubuntu (WSL) terminal:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4.2 Start Ollama Service

```bash
ollama serve &
```

(Press Enter if the prompt doesn't return immediately)

### 4.3 Download Required Model

```bash
ollama pull mistral-nemo:12b
```

This downloads ~7GB. Takes 5-15 minutes depending on your internet.

### 4.4 Verify Ollama

```bash
ollama list
```

You should see `mistral-nemo:12b` listed.

**Tip:** To make Ollama start automatically, add to your `~/.bashrc`:
```bash
echo 'pgrep -x "ollama" > /dev/null || ollama serve &' >> ~/.bashrc
```

---

## Step 5: Install Aegis Insight

### 5.1 Clone the Repository

```bash
cd ~
git clone https://github.com/Eleutherios-project/Eleutherios.git
cd Eleutherios
```

**Alternative: Manual Download**

If you have the files on a USB drive or downloaded as ZIP:
1. Copy to `C:\Users\<yourname>\Eleutherios`
2. In WSL:
   ```bash
   cd /mnt/c/Users/<yourname>/Eleutherios
   ```

### 5.2 Make Scripts Executable

```bash
chmod +x setup.sh docker-entrypoint.sh
```

### 5.3 Run Setup

```bash
./setup.sh
```

This loads demo data (~460MB) into Docker volumes. Takes 1-2 minutes.

### 5.4 Start Aegis Insight

```bash
docker-compose up -d
```

First run builds the Docker image (~10-15 minutes).

### 5.5 Watch Progress

```bash
docker-compose logs -f api
```

Wait until you see:
```
[INFO] All systems ready!
[INFO] Web UI: http://localhost:8001
```

Press `Ctrl+C` to exit log view (doesn't stop the server).

---

## Step 6: Access Aegis Insight

Open your web browser and go to:

**http://localhost:8001**

### Try These Searches:
- `Smedley Butler`
- `Remember the Maine`
- `Thomas Paine`

### Demo Data Included:
- 38,469 claims from 81 documents
- 17,584 entities (people, places, organizations)
- Full semantic search capabilities
- Detection algorithms ready to use

---

## Common Commands

### Start/Stop Services

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f api

# Restart
docker-compose restart
```

### Check Status

```bash
# Container status
docker-compose ps

# System health
curl http://localhost:8001/api/system/status
```

### Start Ollama (if not running)

```bash
ollama serve &
```

### Complete Reset

```bash
# Remove all data and start fresh
docker-compose down -v
./setup.sh
docker-compose up -d
```

---

## Troubleshooting

### "Cannot connect to Docker daemon"

Docker Desktop isn't running. Launch it from Start menu.

### "Permission denied" errors

```bash
chmod +x setup.sh docker-entrypoint.sh
```

### Ollama not accessible

Start Ollama manually:
```bash
ollama serve &
```

### GPU not detected

1. Update NVIDIA drivers in Windows
2. Restart WSL: `wsl --shutdown` (in PowerShell)
3. Reopen Ubuntu and try `nvidia-smi`

### Slow performance

- Ensure Docker Desktop has enough RAM: Settings → Resources → Memory (8GB minimum)
- Close unnecessary Windows applications
- GPU acceleration helps significantly with LLM operations

### Port 8001 already in use

```bash
# Find what's using the port
sudo lsof -i :8001

# Or change the port in docker-compose.yml:
# ports:
#   - "8002:8001"
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Windows Host                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │                    WSL2 (Ubuntu)                    ││
│  │  ┌─────────────────────────────────────────────────┐││
│  │  │              Docker Containers                  │││
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────────────┐    │││
│  │  │  │  Neo4j  │ │PostgreSQL│ │   Aegis API    │    │││
│  │  │  │ :7474   │ │  :5432  │ │    :8001        │    │││
│  │  │  └─────────┘ └─────────┘ └─────────────────┘    │││
│  │  └─────────────────────────────────────────────────┘││
│  │  ┌─────────────────────────────────────────────────┐││
│  │  │              Ollama (host)  :11434              │││
│  │  │              mistral-nemo:12b                   │││
│  │  └─────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────┘│
│                         │                               │
│                    ┌────┴─────┐                         │
│                    │NVIDIA GPU│ (recommended)           │
│                    └──────────┘                         │
└─────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Explore the Demo Data** - Search, run detection algorithms, examine the graph
2. **Load Your Own Documents** - Use the Data Loading tab to import PDFs
3. **Read the User Manual** - Comprehensive documentation in the repo
4. **Join the Community** - GitHub Issues for questions and feedback

---

## Support

- **GitHub Issues:** https://github.com/Eleutherios-project/Eleutherios/issues
- **Documentation:** See `docs/` folder in repository

---

*Aegis Insight *  
*Cedrus Strategic LLC*
