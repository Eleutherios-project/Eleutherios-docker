#!/usr/bin/env python3
"""
Aegis Insight - Import Wizard
=============================

A user-friendly CLI tool for batch document ingestion into the Aegis knowledge graph.

Features:
- Interactive prompts with sensible defaults
- Pre-flight system checks (Ollama, Neo4j, PostgreSQL, GPU, disk space)
- Duplicate detection with user choice
- Time estimation based on hardware and historical data
- Phase tracking with resume capability
- Progress bars and GPU monitoring
- Automatic cleanup of temp files
- Consolidated logging

Usage:
    python3 aegis_import_wizard.py                          # Interactive mode
    python3 aegis_import_wizard.py --source /path/to/docs   # Specify source
    python3 aegis_import_wizard.py --source /path --yes     # Non-interactive
    python3 aegis_import_wizard.py --dry-run                # Preview only

Author: Aegis Insight Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import signal
import shutil
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
try:
    import questionary
    from questionary import Style
except ImportError:
    print("Error: questionary not installed. Run: pip install questionary")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

# Optional imports for checks
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


# =============================================================================
# Constants & Configuration
# =============================================================================

AEGIS_HOME = Path.home() / ".aegis"
JOBS_DIR = AEGIS_HOME / "jobs"
HISTORY_FILE = AEGIS_HOME / "history.json"
CONFIG_FILE = AEGIS_HOME / "config.json"

# Default settings
DEFAULT_MODEL = "mistral-nemo:12b"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_BATCH_SIZE = 30
DEFAULT_CHECKPOINT_EVERY = 100

# Database connections
# Configuration from aegis_config (supports environment variables)
from aegis_config import (Config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                          POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB)
# LEGACY: NEO4J_URI = "bolt://localhost:7687"
# LEGACY: NEO4J_USER = "neo4j"
# LEGACY: NEO4J_PASSWORD = "aegistrusted"
# LEGACY: 
# LEGACY: POSTGRES_HOST = "localhost"
# LEGACY: POSTGRES_DB = "aegis_insight"
# LEGACY: POSTGRES_USER = "aegis"
# LEGACY: POSTGRES_PASSWORD = "aegis_trusted_2025"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_SECONDARY_URL = "http://localhost:11435"

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.epub'}

# Questionary styling
WIZARD_STYLE = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'bold'),
    ('answer', 'fg:cyan'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
])


# =============================================================================
# Data Classes
# =============================================================================

class Phase(Enum):
    """Import pipeline phases"""
    INIT = "init"
    SCAN = "scan"
    JSONL = "jsonl"
    EXTRACT = "extract"
    GRAPH = "graph"
    EMBED = "embed"
    CLEANUP = "cleanup"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class JobState:
    """Persistent job state for resume capability"""
    job_id: str
    source_dir: str
    created_at: str
    updated_at: str
    phase: str
    phases_completed: List[str]
    files_total: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    chunks_total: int = 0
    chunks_processed: int = 0
    skip_duplicates: bool = True
    duplicates_found: int = 0
    parallelism: int = 4
    model: str = DEFAULT_MODEL
    error_message: str = ""
    
    def save(self, path: Path):
        """Save state to JSON file"""
        self.updated_at = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'JobState':
        """Load state from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PreflightResult:
    """Result of a preflight check"""
    name: str
    passed: bool
    critical: bool
    message: str
    details: str = ""


@dataclass 
class SourceAnalysis:
    """Analysis of source directory"""
    total_files: int
    by_type: Dict[str, int]
    total_size_mb: float
    estimated_chunks: int
    duplicates: List[str]
    new_files: List[Path]


# =============================================================================
# Spinner Animation
# =============================================================================

class Spinner:
    """ASCII spinner for async operations"""
    
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    # Fallback for terminals without unicode
    FRAMES_ASCII = ['/', '-', '\\', '|']
    
    def __init__(self, message: str = ""):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = self.FRAMES
        
    def _spin(self):
        idx = 0
        while self.running:
            frame = self.frames[idx % len(self.frames)]
            print(f"\r{frame} {self.message}", end="", flush=True)
            idx += 1
            time.sleep(0.1)
    
    def start(self, message: str = None):
        if message:
            self.message = message
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
    
    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        print(f"\r{' ' * (len(self.message) + 5)}\r", end="")
        if final_message:
            print(final_message)


# =============================================================================
# Utility Functions
# =============================================================================

def print_header():
    """Print wizard header"""
    # Clear screen
    import os
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                 AEGIS INSIGHT - Import Wizard                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()


def print_section(title: str):
    """Print section header"""
    print()
    print("━" * 70)
    print(f"  {title}")
    print("━" * 70)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins} minute{'s' if mins != 1 else ''}"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def ensure_dirs():
    """Ensure required directories exist"""
    AEGIS_HOME.mkdir(exist_ok=True)
    JOBS_DIR.mkdir(exist_ok=True)


def get_job_dir(job_id: str) -> Path:
    """Get job directory path"""
    return JOBS_DIR / job_id


def run_command(cmd: List[str], cwd: str = None, capture: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


# =============================================================================
# Pre-flight Checks
# =============================================================================

def check_ollama() -> PreflightResult:
    """Check if Ollama is running and model is available"""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m['name'] for m in data.get('models', [])]
            
            if DEFAULT_MODEL in models or f"{DEFAULT_MODEL}:latest" in models:
                return PreflightResult(
                    name="Ollama",
                    passed=True,
                    critical=True,
                    message=f"Running, {DEFAULT_MODEL} loaded",
                    details=f"Available models: {', '.join(models[:5])}"
                )
            else:
                return PreflightResult(
                    name="Ollama",
                    passed=False,
                    critical=True,
                    message=f"Running, but {DEFAULT_MODEL} not found",
                    details=f"Run: ollama pull {DEFAULT_MODEL}"
                )
    except Exception as e:
        return PreflightResult(
            name="Ollama",
            passed=False,
            critical=True,
            message="Not running or not reachable",
            details="Start with: ollama serve"
        )


def check_neo4j() -> PreflightResult:
    """Check Neo4j connection"""
    if not HAS_NEO4J:
        return PreflightResult(
            name="Neo4j",
            passed=False,
            critical=True,
            message="neo4j driver not installed",
            details="Run: pip install neo4j"
        )
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (c:Claim) RETURN count(c) as count")
            count = result.single()['count']
        driver.close()
        return PreflightResult(
            name="Neo4j",
            passed=True,
            critical=True,
            message=f"Connected ({count:,} existing claims)"
        )
    except Exception as e:
        return PreflightResult(
            name="Neo4j",
            passed=False,
            critical=True,
            message="Connection failed",
            details=str(e)[:50]
        )


def check_postgres() -> PreflightResult:
    """Check PostgreSQL connection"""
    if not HAS_PSYCOPG2:
        return PreflightResult(
            name="PostgreSQL",
            passed=False,
            critical=True,
            message="psycopg2 not installed",
            details="Run: pip install psycopg2-binary"
        )
    
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=5
        )
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM claim_embeddings")
            count = cur.fetchone()[0]
        conn.close()
        return PreflightResult(
            name="PostgreSQL",
            passed=True,
            critical=True,
            message=f"Connected ({count:,} embeddings)"
        )
    except Exception as e:
        return PreflightResult(
            name="PostgreSQL",
            passed=False,
            critical=True,
            message="Connection failed",
            details=str(e)[:50]
        )


def check_disk_space(source_dir: Path) -> PreflightResult:
    """Check available disk space"""
    try:
        # Check home directory (where temp files go)
        usage = shutil.disk_usage(AEGIS_HOME)
        free_gb = usage.free / (1024**3)
        
        # Estimate needed space (2x source size for JSONL + overhead)
        source_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
        needed_gb = (source_size * 2.5) / (1024**3)
        
        if free_gb > needed_gb + 5:  # 5GB buffer
            return PreflightResult(
                name="Disk Space",
                passed=True,
                critical=True,
                message=f"{free_gb:.0f} GB free (need ~{needed_gb:.1f} GB)"
            )
        else:
            return PreflightResult(
                name="Disk Space",
                passed=False,
                critical=True,
                message=f"Only {free_gb:.0f} GB free (need ~{needed_gb:.1f} GB)"
            )
    except Exception as e:
        return PreflightResult(
            name="Disk Space",
            passed=True,  # Don't block on check failure
            critical=False,
            message="Could not determine",
            details=str(e)[:50]
        )


def check_gpu() -> PreflightResult:
    """Check GPU status and temperature"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,temperature.gpu,power.limit,power.default_limit',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return PreflightResult(
                name="GPU",
                passed=True,  # Not critical - can run on CPU
                critical=False,
                message="nvidia-smi failed (CPU mode?)"
            )
        
        lines = result.stdout.strip().split('\n')
        gpus = []
        stock_config = False
        high_temp = False
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                name, temp, power_limit, power_default = parts[:4]
                gpus.append({
                    'name': name,
                    'temp': int(float(temp)),
                    'power_limit': float(power_limit),
                    'power_default': float(power_default)
                })
                if float(power_limit) >= float(power_default) - 5:
                    stock_config = True
                if int(float(temp)) > 75:
                    high_temp = True
        
        gpu_count = len(gpus)
        temps = [g['temp'] for g in gpus]
        temp_str = ", ".join(f"{t}°C" for t in temps)
        
        status_parts = [f"{gpu_count}x GPU"]
        if temps:
            status_parts.append(f"@ {temp_str}")
        
        details = ""
        if stock_config:
            details = "Stock power config"
        if high_temp:
            details += " (running warm)" if details else "Running warm"
        
        return PreflightResult(
            name="GPU",
            passed=True,
            critical=False,
            message=" ".join(status_parts),
            details=details
        )
        
    except FileNotFoundError:
        return PreflightResult(
            name="GPU",
            passed=True,
            critical=False,
            message="No NVIDIA GPU detected (CPU mode)"
        )
    except Exception as e:
        return PreflightResult(
            name="GPU",
            passed=True,
            critical=False,
            message="Could not query GPU",
            details=str(e)[:50]
        )


def check_source_dir(source_dir: Path) -> PreflightResult:
    """Check if source directory exists and has files"""
    if not source_dir.exists():
        return PreflightResult(
            name="Source Directory",
            passed=False,
            critical=True,
            message="Does not exist",
            details=str(source_dir)
        )
    
    if not source_dir.is_dir():
        return PreflightResult(
            name="Source Directory",
            passed=False,
            critical=True,
            message="Not a directory",
            details=str(source_dir)
        )
    
    # Count supported files
    files = list(source_dir.rglob('*'))
    supported = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS 
                 and not f.name.startswith('._')]
    
    if not supported:
        return PreflightResult(
            name="Source Directory",
            passed=False,
            critical=True,
            message="No supported files found",
            details=f"Looking for: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    return PreflightResult(
        name="Source Directory",
        passed=True,
        critical=True,
        message=f"{len(supported)} files found"
    )


def run_preflight_checks(source_dir: Path) -> Tuple[List[PreflightResult], bool]:
    """Run all preflight checks and return results"""
    checks = [
        ("Checking Ollama service...", check_ollama),
        ("Checking Neo4j connection...", check_neo4j),
        ("Checking PostgreSQL connection...", check_postgres),
        ("Checking disk space...", lambda: check_disk_space(source_dir)),
        ("Checking GPU status...", check_gpu),
        ("Checking source directory...", lambda: check_source_dir(source_dir)),
    ]
    
    results = []
    spinner = Spinner()
    
    for message, check_func in checks:
        spinner.start(message)
        time.sleep(0.3)  # Brief pause for UX
        result = check_func()
        spinner.stop()
        results.append(result)
        
        # Print result
        if result.passed:
            status = "✓" 
            color = "\033[92m"  # Green
        elif result.critical:
            status = "✗"
            color = "\033[91m"  # Red
        else:
            status = "⚠"
            color = "\033[93m"  # Yellow
        
        reset = "\033[0m"
        print(f"{color}{status}{reset} {result.name:20} {result.message}")
        if result.details and not result.passed:
            print(f"  └─ {result.details}")
    
    # Determine if we can proceed
    can_proceed = all(r.passed for r in results if r.critical)
    
    return results, can_proceed


# =============================================================================
# Source Analysis
# =============================================================================

def analyze_source(source_dir: Path, check_duplicates: bool = True) -> SourceAnalysis:
    """Analyze source directory for files, types, and duplicates"""
    
    files = []
    by_type = {}
    total_size = 0
    
    for f in source_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith('._'):
            files.append(f)
            ext = f.suffix.lower()
            by_type[ext] = by_type.get(ext, 0) + 1
            total_size += f.stat().st_size
    
    # Estimate chunks (rough: 1 chunk per 4KB of text content)
    # PDFs are ~70% overhead, so actual text is ~30% of file size
    # At 1000 words per chunk, ~6KB per chunk on average
    estimated_chunks = max(1, int(total_size * 0.3 / 6000))
    
    # Check for duplicates in Neo4j
    duplicates = []
    new_files = files.copy()
    
    if check_duplicates and HAS_NEO4J:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN d.source_file as sf")
                existing = {r['sf'] for r in result if r['sf']}
            driver.close()
            
            for f in files:
                # Check various path formats
                if str(f) in existing or f.name in existing or str(f.absolute()) in existing:
                    duplicates.append(str(f))
                    new_files.remove(f)
        except Exception:
            pass  # Silently ignore duplicate check failures
    
    return SourceAnalysis(
        total_files=len(files),
        by_type=by_type,
        total_size_mb=total_size / (1024**2),
        estimated_chunks=estimated_chunks,
        duplicates=duplicates,
        new_files=new_files
    )


def get_existing_source_files() -> set:
    """Get set of source files already in Neo4j"""
    if not HAS_NEO4J:
        return set()
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN d.source_file as sf")
            existing = {r['sf'] for r in result if r['sf']}
        driver.close()
        return existing
    except Exception:
        return set()


# =============================================================================
# Time Estimation
# =============================================================================

def load_history() -> Dict:
    """Load historical performance data"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"jobs": [], "avg_chunks_per_second": 2.0}


def save_history(history: Dict):
    """Save historical performance data"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def estimate_time(chunks: int, parallelism: int = 4) -> Tuple[int, int]:
    """Estimate processing time in seconds (min, max)"""
    history = load_history()
    rate = history.get('avg_chunks_per_second', 2.0)
    
    # Adjust for parallelism
    effective_rate = rate * (parallelism / 4)  # Baseline is 4x parallelism
    
    base_time = chunks / effective_rate
    
    # Add overhead for JSONL conversion (~30 sec) and graph building (~60 sec)
    overhead = 90
    
    # Return range (80% to 120% of estimate)
    min_time = int(base_time * 0.8 + overhead)
    max_time = int(base_time * 1.2 + overhead)
    
    return min_time, max_time


# =============================================================================
# GPU Power Management
# =============================================================================

def get_gpu_power_status() -> Tuple[bool, List[Dict]]:
    """
    Check if GPUs are at stock power config.
    Returns (is_stock, gpu_list)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,power.limit,power.default_limit',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return False, []
        
        gpus = []
        is_stock = False
        
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                idx, name, limit, default = parts[:4]
                gpu = {
                    'index': int(idx),
                    'name': name,
                    'power_limit': float(limit),
                    'power_default': float(default),
                    'is_stock': float(limit) >= float(default) - 5
                }
                gpus.append(gpu)
                if gpu['is_stock']:
                    is_stock = True
        
        return is_stock, gpus
        
    except Exception:
        return False, []


def apply_gpu_power_optimization(gpus: List[Dict]) -> bool:
    """Apply power limit reduction to GPUs"""
    success = True
    
    for gpu in gpus:
        if gpu['is_stock']:
            # Reduce to ~77% of default (e.g., 300W -> 230W)
            new_limit = int(gpu['power_default'] * 0.77)
            
            try:
                # Enable persistence mode
                subprocess.run(
                    ['sudo', 'nvidia-smi', '-i', str(gpu['index']), '-pm', '1'],
                    capture_output=True, timeout=10
                )
                
                # Set power limit
                subprocess.run(
                    ['sudo', 'nvidia-smi', '-i', str(gpu['index']), '-pl', str(new_limit)],
                    capture_output=True, timeout=10
                )
                
                print(f"  ✓ GPU {gpu['index']}: {gpu['power_default']:.0f}W → {new_limit}W")
            except Exception as e:
                print(f"  ✗ GPU {gpu['index']}: Failed to set power limit")
                success = False
    
    return success


# =============================================================================
# Pipeline Execution
# =============================================================================

class ImportWizard:
    """Main import wizard class"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state: Optional[JobState] = None
        self.job_dir: Optional[Path] = None
        self.log_file: Optional[Path] = None
        self.interrupted = False
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.interrupted = True
        print("\n\n⚠ Interrupt received. Saving checkpoint...")
        if self.state:
            self.state.save(self.job_dir / "state.json")
            print(f"  Checkpoint saved. Run again to resume.")
        sys.exit(1)
    
    def _log(self, message: str):
        """Write to log file and optionally to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
    
    def run(self):
        """Main wizard flow"""
        ensure_dirs()
        print_header()
        
        # Check for existing incomplete job
        if self._check_resume():
            return
        
        # Get source directory
        source_dir = self._get_source_dir()
        if not source_dir:
            return
        
        # Get job name
        job_id = self._get_job_name(source_dir)
        
        # Set up job directory
        self.job_dir = get_job_dir(job_id)
        self.job_dir.mkdir(exist_ok=True)
        (self.job_dir / "temp").mkdir(exist_ok=True)
        self.log_file = self.job_dir / "import.log"
        
        # Run preflight checks
        print_section("PRE-FLIGHT CHECKS")
        results, can_proceed = run_preflight_checks(source_dir)
        
        if not can_proceed:
            print("\n✗ Pre-flight checks failed. Fix issues above and try again.")
            return
        
        # Analyze source
        print_section("SOURCE ANALYSIS")
        spinner = Spinner("Scanning files...")
        spinner.start()
        analysis = analyze_source(source_dir)
        spinner.stop()
        
        self._print_analysis(analysis)
        
        # Handle duplicates
        files_to_process = analysis.new_files
        skip_duplicates = True
        
        if analysis.duplicates:
            print(f"\n⚠ Found {len(analysis.duplicates)} previously imported files")
            
            if not self.args.yes:
                choice = questionary.select(
                    "What would you like to do?",
                    choices=[
                        f"Skip duplicates (import {len(analysis.new_files)} new files)",
                        "Re-import all (may create duplicate entries)",
                        "Cancel"
                    ],
                    style=WIZARD_STYLE
                ).ask()
                
                if choice is None or "Cancel" in choice:
                    print("Import cancelled.")
                    return
                
                if "Re-import" in choice:
                    skip_duplicates = False
                    files_to_process = [Path(f) for f in 
                                       [str(p) for p in analysis.new_files] + analysis.duplicates]
        
        if not files_to_process:
            print("\n✓ No new files to import. Everything is up to date!")
            return
        
        # Time estimate
        min_time, max_time = estimate_time(analysis.estimated_chunks)
        print(f"\n  Estimated time:     {format_duration(min_time)} - {format_duration(max_time)}")
        
        # GPU power optimization
        is_stock, gpus = get_gpu_power_status()
        if is_stock and max_time > 1800:  # > 30 minutes
            print(f"\n⚠ GPU Power: Running at stock settings")
            print(f"  Long jobs may cause thermal throttling (80°C+)")
            
            if not self.args.yes:
                optimize = questionary.confirm(
                    "Reduce power limit for cooler operation? (~5% slower, 10-15°C cooler)",
                    default=True,
                    style=WIZARD_STYLE
                ).ask()
                
                if optimize:
                    print("\n  Applying power optimization (requires sudo)...")
                    apply_gpu_power_optimization(gpus)
        
        # Get parallelism
        parallelism = self._get_parallelism()
        
        # Get model
        model = self._get_model()
        
        # Dry run check
        if self.args.dry_run:
            print("\n" + "─" * 70)
            print("DRY RUN - No changes will be made")
            print("─" * 70)
            return
        
        # Confirm
        if not self.args.yes:
            print(f"\n  Temp directory:     {self.job_dir / 'temp'}")
            print(f"  Log file:           {self.log_file}")
            
            confirm = questionary.confirm(
                "\nReady to start import?",
                default=True,
                style=WIZARD_STYLE
            ).ask()
            
            if not confirm:
                print("Import cancelled.")
                return
        
        # Create initial state
        self.state = JobState(
            job_id=job_id,
            source_dir=str(source_dir),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            phase=Phase.INIT.value,
            phases_completed=[],
            files_total=len(files_to_process),
            skip_duplicates=skip_duplicates,
            duplicates_found=len(analysis.duplicates),
            parallelism=parallelism,
            model=model
        )
        self.state.save(self.job_dir / "state.json")
        
        # Run pipeline
        try:
            self._run_pipeline(files_to_process)
        except Exception as e:
            self.state.phase = Phase.FAILED.value
            self.state.error_message = str(e)
            self.state.save(self.job_dir / "state.json")
            print(f"\n✗ Import failed: {e}")
            self._log(f"FAILED: {e}")
            return
        
        # Success!
        self._print_summary()
    
    def _check_resume(self) -> bool:
        """Check for incomplete jobs and offer to resume"""
        incomplete_jobs = []
        
        for job_dir in JOBS_DIR.iterdir():
            if job_dir.is_dir():
                state_file = job_dir / "state.json"
                if state_file.exists():
                    try:
                        state = JobState.load(state_file)
                        if state.phase not in [Phase.COMPLETE.value, Phase.FAILED.value]:
                            incomplete_jobs.append((job_dir.name, state))
                    except Exception:
                        pass
        
        if not incomplete_jobs:
            return False
        
        if self.args.yes:
            # In non-interactive mode, resume first incomplete job
            job_id, state = incomplete_jobs[0]
            return self._resume_job(job_id, state)
        
        # Show incomplete jobs
        print("⚠ Found incomplete jobs:\n")
        for job_id, state in incomplete_jobs:
            created = datetime.fromisoformat(state.created_at).strftime("%Y-%m-%d %H:%M")
            progress = ""
            if state.chunks_total > 0:
                pct = state.chunks_processed / state.chunks_total * 100
                progress = f" ({state.chunks_processed}/{state.chunks_total} chunks, {pct:.0f}%)"
            print(f"  • {job_id}")
            print(f"    Started: {created}")
            print(f"    Phase:   {state.phase.upper()}{progress}")
            print(f"    Source:  {state.source_dir}")
            print()
        
        choices = [f"Resume: {job_id}" for job_id, _ in incomplete_jobs]
        choices.extend(["Start new import", "Delete incomplete jobs"])
        
        choice = questionary.select(
            "What would you like to do?",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()
        
        if choice is None:
            return True
        
        if choice.startswith("Resume:"):
            job_id = choice.replace("Resume: ", "")
            state = next(s for j, s in incomplete_jobs if j == job_id)
            return self._resume_job(job_id, state)
        
        if choice == "Delete incomplete jobs":
            for job_id, _ in incomplete_jobs:
                shutil.rmtree(get_job_dir(job_id), ignore_errors=True)
                print(f"  Deleted: {job_id}")
            print()
            return False
        
        return False
    
    def _resume_job(self, job_id: str, state: JobState) -> bool:
        """Resume an incomplete job"""
        print(f"\nResuming job: {job_id}")
        
        self.job_dir = get_job_dir(job_id)
        self.log_file = self.job_dir / "import.log"
        self.state = state
        
        self._log(f"RESUMED from phase {state.phase}")
        
        # Get files to process
        source_dir = Path(state.source_dir)
        if not source_dir.exists():
            print(f"✗ Source directory no longer exists: {source_dir}")
            return True
        
        analysis = analyze_source(source_dir, check_duplicates=state.skip_duplicates)
        files_to_process = analysis.new_files if state.skip_duplicates else \
                          [Path(f) for f in [str(p) for p in analysis.new_files] + analysis.duplicates]
        
        try:
            self._run_pipeline(files_to_process)
        except Exception as e:
            self.state.phase = Phase.FAILED.value
            self.state.error_message = str(e)
            self.state.save(self.job_dir / "state.json")
            print(f"\n✗ Import failed: {e}")
            return True
        
        self._print_summary()
        return True
    
    def _get_source_dir(self) -> Optional[Path]:
        """Get source directory from args or prompt"""
        if self.args.source:
            path = Path(self.args.source).expanduser().resolve()
            if not path.exists():
                print(f"✗ Source directory does not exist: {path}")
                return None
            return path
        
        path_str = questionary.path(
            "Source directory:",
            only_directories=True,
            style=WIZARD_STYLE
        ).ask()
        
        if not path_str:
            return None
        
        return Path(path_str).expanduser().resolve()
    
    def _get_job_name(self, source_dir: Path) -> str:
        """Get job name from args or prompt"""
        if self.args.job:
            return self.args.job
        
        # Suggest name based on source folder
        suggested = source_dir.name.replace(' ', '_').lower()
        
        if self.args.yes:
            return suggested
        
        job_id = questionary.text(
            "Job name (for tracking):",
            default=suggested,
            style=WIZARD_STYLE
        ).ask()
        
        return job_id or suggested
    
    def _get_parallelism(self) -> int:
        """Get parallelism level based on GPU configuration"""
        # Detect GPU count
        gpu_count = 1
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_count = len(result.stdout.strip().split('\n'))
        except Exception:
            pass
        
        # Check if dual Ollama is running
        dual_ollama_available = False
        try:
            import urllib.request
            req = urllib.request.Request(f"{OLLAMA_SECONDARY_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    dual_ollama_available = True
        except:
            pass
        
        # Determine default based on configuration
        if dual_ollama_available and gpu_count >= 2:
            # Dual GPU mode: 2 parallel file extractions (one per GPU)
            default = 2
            mode_desc = f"Dual GPU mode detected ({gpu_count} GPUs, dual Ollama)"
        else:
            # Single GPU mode: batch parallelism
            default = min(4, gpu_count * 2)
            mode_desc = f"Single GPU mode ({gpu_count} GPU{'s' if gpu_count > 1 else ''})"
        
        if self.args.yes:
            print(f"  {mode_desc} → parallelism={default}")
            return default
        
        # Show context in prompt
        result = questionary.text(
            f"{mode_desc}. Parallelism [{default}]:",
            default=str(default),
            style=WIZARD_STYLE
        ).ask()
        
        try:
            return int(result)
        except (ValueError, TypeError):
            return default
    
    def _get_model(self) -> str:
        """Get LLM model"""
        if self.args.yes:
            return DEFAULT_MODEL
        
        # Get available models
        available = [DEFAULT_MODEL]
        try:
            import urllib.request
            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                available = [m['name'] for m in data.get('models', [])]
        except Exception:
            pass
        
        result = questionary.text(
            f"LLM Model [{DEFAULT_MODEL}]:",
            default=DEFAULT_MODEL,
            style=WIZARD_STYLE
        ).ask()
        
        return result or DEFAULT_MODEL
    
    def _print_analysis(self, analysis: SourceAnalysis):
        """Print source analysis results"""
        print(f"  Files found:        {analysis.total_files} total")
        
        for ext, count in sorted(analysis.by_type.items()):
            print(f"    • {ext.upper()[1:]:12} {count}")
        
        print(f"  Total size:         {format_size(int(analysis.total_size_mb * 1024 * 1024))}")
        print(f"  Estimated chunks:   ~{analysis.estimated_chunks}")

    def _run_pipeline(self, files: List[Path]):
        """Run the full import pipeline"""

        completed_phases = self.state.phases_completed

        # Phase 1: JSONL Conversion
        if Phase.JSONL.value not in completed_phases:
            self._run_jsonl_phase(files)

        # Phase 2: Extraction
        if Phase.EXTRACT.value not in completed_phases:
            self._run_extraction_phase()

        # Phase 3: Graph Building - NOW ACTUALLY BUILDS THE GRAPH
        if Phase.GRAPH.value not in completed_phases:
            self._run_graph_phase()

        # Phase 4: Embeddings
        if Phase.EMBED.value not in completed_phases:
            self._run_embedding_phase()

        # Phase 5: Cleanup
        self._run_cleanup_phase()

        # Mark complete
        self.state.phase = Phase.COMPLETE.value
        self.state.save(self.job_dir / "state.json")

    def _run_jsonl_phase(self, files: List[Path]):
        """Convert source files to JSONL"""
        print_section("PHASE 1/4: JSONL CONVERSION")
        self.state.phase = Phase.JSONL.value
        self.state.save(self.job_dir / "state.json")
        self._log("Starting JSONL conversion")
        
        temp_dir = self.job_dir / "temp"
        source_dir = Path(self.state.source_dir)
        
        # Group files by type
        pdfs = [f for f in files if f.suffix.lower() == '.pdf']
        text_files = [f for f in files if f.suffix.lower() in {'.txt', '.md'}]
        docx_files = [f for f in files if f.suffix.lower() == '.docx']
        epub_files = [f for f in files if f.suffix.lower() == '.epub']
        
        total_files = len(files)
        chunks_created = 0
        
        # Process PDFs with parallel converter
        if pdfs:
            print(f"\nConverting {len(pdfs)} PDF files...")
            
            # Find the converter script
            aegis_dir = self._find_aegis_dir()
            if aegis_dir:
                converter = aegis_dir / "mvp_pdf_to_jsonl_parallel.py"
                if converter.exists():
                    cmd = [
                        'python3', str(converter),
                        str(source_dir),
                        str(temp_dir),
                        '--chunk-size', str(DEFAULT_CHUNK_SIZE),
                        '--overlap', str(DEFAULT_OVERLAP),
                        '--workers', '8'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Count created chunks
                        for jsonl_file in temp_dir.glob('*.jsonl'):
                            with open(jsonl_file, 'r') as f:
                                chunks_created += sum(1 for _ in f)
                        print(f"✓ PDFs converted: {chunks_created} chunks created")
                    else:
                        print(f"⚠ PDF conversion had issues: {result.stderr[:100]}")
        
        # Process text files directly
        if text_files:
            print(f"\nConverting {len(text_files)} text files...")
            
            for tf in tqdm(text_files, desc="Text files"):
                try:
                    with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    words = text.split()
                    chunks = []
                    
                    for i in range(0, len(words), DEFAULT_CHUNK_SIZE - DEFAULT_OVERLAP):
                        chunk_words = words[i:i + DEFAULT_CHUNK_SIZE]
                        if len(chunk_words) < 100:
                            continue
                        
                        chunks.append({
                            'content': ' '.join(chunk_words),
                            'sequence_num': len(chunks),
                            'context_metadata': {
                                'source_file': str(tf),
                                'title': tf.stem,
                                'domain': tf.parent.name or 'imported'
                            }
                        })
                    
                    if chunks:
                        output_file = temp_dir / f"{tf.stem}.jsonl"
                        with open(output_file, 'w') as f:
                            for chunk in chunks:
                                f.write(json.dumps(chunk) + '\n')
                        chunks_created += len(chunks)
                        
                except Exception as e:
                    self._log(f"Error processing {tf}: {e}")
        
        # Update state
        self.state.chunks_total = chunks_created
        self.state.files_processed = total_files
        self._mark_phase_complete(Phase.JSONL)
        
        print(f"\n✓ Created {chunks_created} chunks from {total_files} files")
        self._log(f"JSONL complete: {chunks_created} chunks from {total_files} files")



    def _run_extraction_phase(self):
        """Run extraction pipeline on JSONL files with optional dual GPU support"""
        print_section("PHASE 2/4: EXTRACTION")
        self.state.phase = Phase.EXTRACT.value
        self.state.save(self.job_dir / "state.json")
        self._log("Starting extraction")

        temp_dir = self.job_dir / "temp"
        checkpoint_dir = self.job_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        log_dir = self.job_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Find JSONL files
        jsonl_files = sorted(temp_dir.glob('*.jsonl'))

        if not jsonl_files:
            print("⚠  No JSONL files found to process")
            self._mark_phase_complete(Phase.EXTRACT)
            return

        # Find extraction script (prefer v3)
        aegis_dir = self._find_aegis_dir()
        if not aegis_dir:
            print("✗ Could not find Aegis installation directory")
            return

        extraction_script = aegis_dir / "run_extraction_pipeline_v3.py"
        if not extraction_script.exists():
            extraction_script = aegis_dir / "run_extraction_pipeline.py"
            if not extraction_script.exists():
                print(f"✗ Extraction script not found")
                return
            print("  Using legacy extraction pipeline")
        else:
            print("  Using v3 extraction pipeline (with coreference resolution)")

        # Check for dual GPU capability
        dual_gpu = self._check_dual_gpu()

        if dual_gpu:
            print(f"✓ Dual GPU mode detected - using both GPUs")
            self._run_extraction_dual_gpu(jsonl_files, extraction_script, checkpoint_dir, log_dir)
        else:
            print(f"  Single GPU mode - processing {len(jsonl_files)} files")
            self._run_extraction_single_gpu(jsonl_files, extraction_script, checkpoint_dir, log_dir)

        self._mark_phase_complete(Phase.EXTRACT)

    def _run_graph_phase(self):
        """Load extracted data into Neo4j graph"""
        print_section("PHASE 3/4: GRAPH BUILDING")
        self.state.phase = Phase.GRAPH.value
        self.state.save(self.job_dir / "state.json")
        self._log("Starting graph building")

        checkpoint_dir = self.job_dir / "checkpoints"

        # Find all extracted JSONL files
        extracted_files = sorted(checkpoint_dir.glob('*_extracted.jsonl'))

        if not extracted_files:
            print("⚠  No extracted files found to load into graph")
            print("   (This may indicate extraction failed)")
            self._mark_phase_complete(Phase.GRAPH)
            return

        print(f"Loading {len(extracted_files)} extracted files into Neo4j...")

        # Find graph builder
        aegis_dir = self._find_aegis_dir()
        if not aegis_dir:
            print("✗ Could not find Aegis installation directory")
            return

        graph_builder = aegis_dir / "aegis_graph_builder.py"
        if not graph_builder.exists():
            print(f"✗ Graph builder not found: {graph_builder}")
            return

        start_time = time.time()
        loaded = 0
        failed = 0
        total_claims = 0
        total_entities = 0

        for i, extracted_file in enumerate(extracted_files, 1):
            try:
                # Count items in file
                with open(extracted_file, 'r') as f:
                    line_count = sum(1 for _ in f)

                print(f"  [{i}/{len(extracted_files)}] {extracted_file.name} ({line_count} chunks)...", end=' ',
                      flush=True)

                # Run graph builder
                cmd = [
                    'python3', str(graph_builder),
                    str(extracted_file)  # Pass as positional argument
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(aegis_dir),
                    timeout=600  # 10 minute timeout per file
                )

                if result.returncode == 0:
                    loaded += 1
                    # Parse stats from output
                    for line in result.stdout.split('\n'):
                        # New graph builder outputs "Claims created: X"
                        if 'Claims created:' in line:
                            try:
                                num_str = line.split('Claims created:')[1].strip().split()[0]
                                total_claims += int(num_str.replace(',', ''))
                            except (IndexError, ValueError):
                                pass
                        if 'Entities created:' in line:
                            try:
                                num_str = line.split('Entities created:')[1].strip().split()[0]
                                total_entities += int(num_str.replace(',', ''))
                            except (IndexError, ValueError):
                                pass
                    print("✓")
                else:
                    failed += 1
                    error_msg = result.stderr[:100] if result.stderr else 'unknown error'
                    print(f"✗ ({error_msg})")

            except subprocess.TimeoutExpired:
                failed += 1
                print("✗ (timeout)")
            except Exception as e:
                failed += 1
                print(f"✗ ({e})")

        elapsed = time.time() - start_time

        print(f"\n✓ Graph building complete in {format_duration(elapsed)}")
        print(f"  Files loaded: {loaded}/{len(extracted_files)}")
        print(f"  Entities added: {total_entities:,}")
        print(f"  Claims added: {total_claims:,}")
        if failed:
            print(f"  Failed: {failed} files (check individual logs)")

        self._log(
            f"Graph complete: {loaded} files, {total_entities} entities, {total_claims} claims in {format_duration(elapsed)}")
        self._mark_phase_complete(Phase.GRAPH)

    def _check_dual_gpu(self) -> bool:
        """Check if dual GPU Ollama instances are available"""
        import urllib.request

        try:
            # Check primary
            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status != 200:
                    return False

            # Check secondary
            req = urllib.request.Request(f"{OLLAMA_SECONDARY_URL}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status != 200:
                    return False

            return True
        except Exception:
            return False

    def _run_extraction_dual_gpu(self, jsonl_files, extraction_script, checkpoint_dir, log_dir):
        """Run extraction with true dual GPU parallelism - separate processes per GPU"""
        total_files = len(jsonl_files)

        # Sort files by size (largest first) and distribute for balanced workload
        files_with_size = [(f, f.stat().st_size) for f in jsonl_files]
        files_with_size.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute using greedy algorithm (assign to GPU with less total work)
        gpu0_files = []
        gpu1_files = []
        gpu0_size = 0
        gpu1_size = 0
        
        for f, size in files_with_size:
            if gpu0_size <= gpu1_size:
                gpu0_files.append(f)
                gpu0_size += size
            else:
                gpu1_files.append(f)
                gpu1_size += size

        print(f"  GPU 0: {len(gpu0_files)} files ({gpu0_size / 1024 / 1024:.1f} MB)")
        print(f"  GPU 1: {len(gpu1_files)} files ({gpu1_size / 1024 / 1024:.1f} MB)")

        aegis_dir = self._find_aegis_dir()
        start_time = time.time()

        # Create wrapper scripts for each GPU
        gpu0_script = self._create_gpu_wrapper(
            gpu0_files, extraction_script, checkpoint_dir,
            OLLAMA_URL, "0"
        )
        gpu1_script = self._create_gpu_wrapper(
            gpu1_files, extraction_script, checkpoint_dir,
            OLLAMA_SECONDARY_URL, "1"
        )

        # Start both processes
        gpu0_log = log_dir / "gpu0_extraction.log"
        gpu1_log = log_dir / "gpu1_extraction.log"

        env0 = os.environ.copy()
        env0['CUDA_VISIBLE_DEVICES'] = '0'

        env1 = os.environ.copy()
        env1['CUDA_VISIBLE_DEVICES'] = '1'

        with open(gpu0_log, 'w') as log0, open(gpu1_log, 'w') as log1:
            proc0 = subprocess.Popen(
                ['bash', str(gpu0_script)],
                stdout=log0, stderr=subprocess.STDOUT,
                cwd=str(aegis_dir), env=env0
            )
            proc1 = subprocess.Popen(
                ['bash', str(gpu1_script)],
                stdout=log1, stderr=subprocess.STDOUT,
                cwd=str(aegis_dir), env=env1
            )

            print(f"  Started GPU 0 process (PID {proc0.pid})")
            print(f"  Started GPU 1 process (PID {proc1.pid})")
            print()  # Blank line before progress display

            # Monitor both processes with enhanced progress
            last_state_save = time.time()
            while proc0.poll() is None or proc1.poll() is None:
                self._show_extraction_progress(log_dir, gpu0_files, gpu1_files)
                time.sleep(10)  # Check every 10 seconds

                # Update state periodically (every 60s)
                if time.time() - last_state_save > 60:
                    self.state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                    self.state.save(self.job_dir / "state.json")
                    last_state_save = time.time()

            # Clear progress display
            print("\n")

        elapsed = time.time() - start_time

        # Check results
        gpu0_ok = proc0.returncode == 0
        gpu1_ok = proc1.returncode == 0

        print(f"\n\n✓ Extraction complete in {format_duration(elapsed)}")
        print(f"  GPU 0: {'✓' if gpu0_ok else '✗'} ({len(gpu0_files)} files)")
        print(f"  GPU 1: {'✓' if gpu1_ok else '✗'} ({len(gpu1_files)} files)")

        if not gpu0_ok:
            print(f"    Check log: {gpu0_log}")
        if not gpu1_ok:
            print(f"    Check log: {gpu1_log}")

        self._log(f"Dual GPU extraction complete in {format_duration(elapsed)}")

    def _create_gpu_wrapper(self, files, extraction_script, checkpoint_dir, ollama_url, gpu_id):
        """Create a shell script to process files sequentially on one GPU"""
        wrapper_path = self.job_dir / f"extract_gpu{gpu_id}.sh"
        aegis_dir = self._find_aegis_dir()

        with open(wrapper_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {aegis_dir}\n")
            f.write("set -e\n\n")

            for jsonl_file in files:
                job_id = f"{self.state.job_id}_{jsonl_file.stem}"
                f.write(f"echo 'Processing: {jsonl_file.name}'\n")
                f.write(f"python3 {extraction_script} \\\n")
                f.write(f"  --jsonl '{jsonl_file}' \\\n")
                f.write(f"  --ollama-url '{ollama_url}' \\\n")
                f.write(f"  --model '{self.state.model}' \\\n")
                f.write(f"  --job-id '{job_id}' \\\n")
                f.write(f"  --checkpoint-dir '{checkpoint_dir}' \\\n")
                f.write(f"  --skip-graph\n")  # Graph built in separate phase
                f.write(f"echo 'Completed: {jsonl_file.name}'\n\n")

            f.write("echo 'All files processed on GPU {gpu_id}'\n")

        wrapper_path.chmod(0o755)
        return wrapper_path

    def _run_extraction_single_gpu(self, jsonl_files, extraction_script, checkpoint_dir, log_dir):
        """Run extraction on single GPU with parallelism"""
        total_files = len(jsonl_files)
        parallelism = self.state.parallelism
        processed = 0
        failed = 0

        aegis_dir = self._find_aegis_dir()
        start_time = time.time()

        for i in range(0, total_files, parallelism):
            batch = jsonl_files[i:i + parallelism]
            batch_num = i // parallelism + 1
            total_batches = (total_files + parallelism - 1) // parallelism

            print(f"\n=== Batch {batch_num}/{total_batches} ===")

            # Launch parallel jobs
            processes = []
            for jsonl_file in batch:
                job_id = f"{self.state.job_id}_{jsonl_file.stem}"
                log_file = log_dir / f"{jsonl_file.stem}.log"

                cmd = [
                    'python3', str(extraction_script),
                    '--jsonl', str(jsonl_file),
                    '--model', self.state.model,
                    '--job-id', job_id,
                    '--checkpoint-dir', str(checkpoint_dir),
                    '--batch-size', str(DEFAULT_BATCH_SIZE),
                    '--skip-graph'  # Graph built in separate phase
                ]

                # Add ollama-url for v3 pipeline
                if 'v3' in str(extraction_script):
                    cmd.extend(['--ollama-url', OLLAMA_URL])

                with open(log_file, 'w') as lf:
                    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(aegis_dir))
                    processes.append((proc, jsonl_file.stem))
                    print(f"  Started: {jsonl_file.stem}")

            # Wait for batch with progress
            while processes:
                for proc, name in processes[:]:
                    ret = proc.poll()
                    if ret is not None:
                        processes.remove((proc, name))
                        if ret == 0:
                            processed += 1
                            print(f"  ✓ {name}")
                        else:
                            failed += 1
                            print(f"  ✗ {name} (exit code {ret})")

                if processes:
                    self._show_gpu_status()
                    time.sleep(2)

            # Update state
            self.state.chunks_processed = processed
            self.state.save(self.job_dir / "state.json")

        elapsed = time.time() - start_time
        print(f"\n\n✓ Extraction complete in {format_duration(elapsed)}")
        print(f"  Processed: {processed}/{total_files} files")
        if failed:
            print(f"  Failed: {failed} files (check logs)")

        self._log(f"Extraction complete: {processed}/{total_files} files in {format_duration(elapsed)}")

    def _show_extraction_progress(self, log_dir: Path, gpu0_files: list, gpu1_files: list):
        """Show extraction progress by checking log files"""
        try:
            # Check GPU temps
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            gpu_status = ""
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                status_parts = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_id, temp, util = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        status_parts.append(f"GPU{gpu_id}: {temp}C {util}%")
                gpu_status = ' | '.join(status_parts)
            
            # Check progress from logs
            gpu0_done, gpu0_total = self._get_log_progress(log_dir / "gpu0_extraction.log", len(gpu0_files))
            gpu1_done, gpu1_total = self._get_log_progress(log_dir / "gpu1_extraction.log", len(gpu1_files))
            
            # Get current file from log tail
            gpu0_current = self._get_current_file(log_dir / "gpu0_extraction.log")
            gpu1_current = self._get_current_file(log_dir / "gpu1_extraction.log")
            
            # Calculate overall progress
            total_done = gpu0_done + gpu1_done
            total_files = gpu0_total + gpu1_total
            pct = (total_done / total_files * 100) if total_files > 0 else 0
            
            # Build progress bar
            bar_width = 30
            filled = int(bar_width * total_done / total_files) if total_files > 0 else 0
            bar = '#' * filled + '-' * (bar_width - filled)
            
            # Calculate elapsed time
            if not hasattr(self, '_extraction_start_time'):
                self._extraction_start_time = time.time()
            elapsed = time.time() - self._extraction_start_time
            elapsed_str = format_duration(elapsed)
            
            # Estimate remaining time
            if total_done > 0:
                rate = elapsed / total_done
                remaining = rate * (total_files - total_done)
                eta_str = f"ETA: {format_duration(remaining)}"
            else:
                eta_str = "ETA: calculating..."
            
            # Build timestamp
            timestamp = time.strftime("%H:%M:%S")
            
            # Print clear status line (overwrites previous)
            status_line = f"\r  [{timestamp}] [{bar}] {total_done}/{total_files} ({pct:.0f}%) | {elapsed_str} | {eta_str}"
            print(status_line.ljust(100), end='', flush=True)
            
            # Every 30 seconds, print detail lines
            if not hasattr(self, '_last_detail_time'):
                self._last_detail_time = 0
            if time.time() - self._last_detail_time > 30:
                self._last_detail_time = time.time()
                print()  # New line
                print(f"    {gpu_status}")
                if gpu0_current:
                    print(f"    GPU0 ({gpu0_done}/{gpu0_total}): {gpu0_current}")
                if gpu1_current:
                    print(f"    GPU1 ({gpu1_done}/{gpu1_total}): {gpu1_current}")
            
        except Exception:
            pass

    def _get_log_progress(self, log_path: Path, total: int) -> tuple:
        """Count completed files from log"""
        completed = 0
        try:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    content = f.read()
                    completed = content.count('Completed:')
        except:
            pass
        return (completed, total)
    
    def _get_current_file(self, log_path: Path) -> str:
        """Get currently processing file from log"""
        try:
            if log_path.exists():
                # Read last 50 lines efficiently
                with open(log_path, 'rb') as f:
                    f.seek(0, 2)  # End of file
                    size = f.tell()
                    f.seek(max(0, size - 10000))  # Last 10KB
                    lines = f.read().decode('utf-8', errors='ignore').split('\n')
                
                # Find last "Processing:" line
                for line in reversed(lines):
                    if 'Processing:' in line:
                        filename = line.split('Processing:')[-1].strip()
                        # Truncate long names
                        if len(filename) > 45:
                            filename = filename[:42] + "..."
                        return filename
        except:
            pass
        return ""
    
    def _show_gpu_status(self):
        """Show GPU temperature and utilization (legacy, used by single-GPU mode)"""
        """Show GPU temperature and utilization"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                status_parts = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_id, temp, util = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        status_parts.append(f"GPU{gpu_id}: {temp}°C/{util}%")
                print(f"\r  {' | '.join(status_parts)}  ", end='', flush=True)
        except Exception:
            pass

    def _run_embedding_phase(self):
        """Generate embeddings for new claims"""
        print_section("PHASE 3/4: EMBEDDINGS")
        self.state.phase = Phase.EMBED.value
        self.state.save(self.job_dir / "state.json")
        self._log("Starting embedding generation")
        
        # Find embedding script
        aegis_dir = self._find_aegis_dir()
        if not aegis_dir:
            print("⚠ Could not find Aegis directory, skipping embeddings")
            self._mark_phase_complete(Phase.EMBED)
            return
        
        # Check for v2 (incremental) first
        embed_script = aegis_dir / "generate_embeddings.py"
        if not embed_script.exists():
            print("⚠ Embedding script not found, skipping")
            self._mark_phase_complete(Phase.EMBED)
            return
        
        print("Generating embeddings for new claims...")
        
        result = subprocess.run(
            ['python3', str(embed_script)],
            capture_output=True, text=True, cwd=str(aegis_dir)
        )
        
        if result.returncode == 0:
            print("✓ Embeddings generated")
            self._log("Embeddings generated successfully")
        else:
            print(f"⚠ Embedding generation had issues")
            self._log(f"Embedding issues: {result.stderr[:200]}")
        
        self._mark_phase_complete(Phase.EMBED)
    
    def _run_cleanup_phase(self):
        """Clean up temporary files"""
        print_section("PHASE 4/4: CLEANUP")
        self.state.phase = Phase.CLEANUP.value
        self.state.save(self.job_dir / "state.json")
        
        temp_dir = self.job_dir / "temp"
        
        if temp_dir.exists():
            # Count files before cleanup
            file_count = len(list(temp_dir.glob('*')))
            
            # Remove temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"✓ Cleaned up {file_count} temp files")
            self._log(f"Cleanup: removed {file_count} temp files")
        
        self._mark_phase_complete(Phase.CLEANUP)
    
    def _mark_phase_complete(self, phase: Phase):
        """Mark a phase as complete"""
        if phase.value not in self.state.phases_completed:
            self.state.phases_completed.append(phase.value)
        self.state.save(self.job_dir / "state.json")
    
    def _find_aegis_dir(self) -> Optional[Path]:
        """Find the Aegis installation directory"""
        # Check common locations
        candidates = [
            Path("/media/bob/RAID11/DataShare/AegisTrustNet"),
            Path.cwd(),
            Path(__file__).parent,
            Path.home() / "AegisTrustNet",
        ]
        
        for candidate in candidates:
            if candidate.exists() and (candidate / "aegis_extraction_orchestrator.py").exists():
                return candidate
        
        # Try to find it
        for parent in [Path.cwd()] + list(Path.cwd().parents)[:3]:
            if (parent / "aegis_extraction_orchestrator.py").exists():
                return parent
        
        return None
    
    def _print_summary(self):
        """Print final summary"""
        print_section("IMPORT COMPLETE")
        
        # Calculate totals from Neo4j if possible
        entities = 0
        claims = 0
        
        if HAS_NEO4J:
            try:
                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                with driver.session() as session:
                    result = session.run("MATCH (c:Claim) RETURN count(c) as count")
                    claims = result.single()['count']
                    result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                    entities = result.single()['count']
                driver.close()
            except Exception:
                pass
        
        created = datetime.fromisoformat(self.state.created_at)
        elapsed = datetime.now() - created
        
        print(f"  Total time:         {format_duration(elapsed.total_seconds())}")
        print(f"  Files processed:    {self.state.files_total}")
        if self.state.duplicates_found:
            print(f"  Duplicates skipped: {self.state.duplicates_found}")
        print(f"  Chunks processed:   {self.state.chunks_total}")
        
        if entities or claims:
            print(f"  Total entities:     {entities:,}")
            print(f"  Total claims:       {claims:,}")
        
        print(f"\n  Log saved:          {self.log_file}")
        print(f"\n  ✓ Temp files cleaned up")
        
        # Update history for better future estimates
        history = load_history()
        if elapsed.total_seconds() > 0 and self.state.chunks_total > 0:
            rate = self.state.chunks_total / elapsed.total_seconds()
            # Weighted average with history
            old_rate = history.get('avg_chunks_per_second', 2.0)
            history['avg_chunks_per_second'] = (old_rate + rate) / 2
        
        history.setdefault('jobs', []).append({
            'job_id': self.state.job_id,
            'files': self.state.files_total,
            'chunks': self.state.chunks_total,
            'duration_seconds': elapsed.total_seconds(),
            'completed_at': datetime.now().isoformat()
        })
        
        # Keep only last 20 jobs in history
        history['jobs'] = history['jobs'][-20:]
        save_history(history)
        
        self._log(f"COMPLETE: {self.state.files_total} files, {self.state.chunks_total} chunks")
        
        # Offer to open web UI
        if not self.args.yes and not self.args.quiet:
            print()
            open_ui = questionary.confirm(
                "Open web UI to explore?",
                default=False,
                style=WIZARD_STYLE
            ).ask()
            
            if open_ui:
                import webbrowser
                webbrowser.open("http://localhost:8080")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Aegis Insight - Import Wizard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Interactive mode
  %(prog)s --source /path/to/docs       Specify source directory
  %(prog)s --source /path --job mydata  Specify source and job name
  %(prog)s --source /path --yes         Non-interactive (use defaults)
  %(prog)s --dry-run                    Preview without making changes
        """
    )
    
    parser.add_argument('--source', '-s', type=str,
                        help='Source directory containing documents')
    parser.add_argument('--job', '-j', type=str,
                        help='Job name for tracking')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Non-interactive mode (use defaults)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview only, do not process')
    parser.add_argument('--version', '-v', action='version',
                        version='Aegis Import Wizard 1.0.0')
    
    args = parser.parse_args()
    
    # Run wizard
    wizard = ImportWizard(args)
    wizard.run()


if __name__ == "__main__":
    main()
