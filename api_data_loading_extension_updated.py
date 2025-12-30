"""
API Extension for Data Loading Pipeline - UPDATED
Add this to your existing api_server.py

NEW FEATURES:
- File browser endpoint (/list-inbox-files)
- Selected files support
- Progress percentage tracking
- Enhanced stats parsing
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import subprocess
import threading
import time
from pathlib import Path
import logging
import os
import re

logger = logging.getLogger(__name__)

# Create router
data_router = APIRouter(prefix="/api", tags=["data-loading"])

# Global state for tracking load jobs
load_jobs = {}

# Configuration
INBOX_DIR = Path(os.environ.get("INBOX_DIR", "/media/bob/RAID11/DataShare/AegisTrustNet/data/inbox"))
TEMP_IMPORT_DIR = Path("/tmp/aegis_imports")


class LoadRequest(BaseModel):
    """Request model for data loading"""
    type: Optional[str] = None  # 'pdfs' or 'jsonl' for simple mode
    path: Optional[str] = None  # path for simple mode
    selected_files: Optional[List[str]] = None  # NEW: specific files to import
    pdf_dir: Optional[str] = None
    jsonl_dir: Optional[str] = None
    training_script: Optional[str] = None
    output_dir: Optional[str] = "./aegis_processed"
    skip_generation: bool = False
    skip_loading: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "aegistrusted"


class LoadResponse(BaseModel):
    """Response model for data loading"""
    success: bool
    message: str
    job_id: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def stream_output(pipe, job_id: str, stream_type: str):
    """Stream subprocess output in real-time"""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                line = line.rstrip()
                # Log to console
                if stream_type == 'stdout':
                    logger.info(f"[{job_id}] {line}")
                else:
                    logger.error(f"[{job_id}] ERROR: {line}")
                
                # Store in job output
                if job_id in load_jobs:
                    if 'output' not in load_jobs[job_id]:
                        load_jobs[job_id]['output'] = []
                    load_jobs[job_id]['output'].append(line)
                    
                    # Keep only last 1000 lines to prevent memory issues
                    if len(load_jobs[job_id]['output']) > 1000:
                        load_jobs[job_id]['output'] = load_jobs[job_id]['output'][-1000:]
    except Exception as e:
        logger.error(f"Error streaming output for {job_id}: {e}")


def parse_progress_from_output(output_lines: List[str]) -> int:
    """
    Extract progress percentage from output lines
    Looks for patterns like:
    - "Processing file 3 of 10"
    - "Progress: 45%"
    - "Chunk 123/456"
    """
    if not output_lines:
        return 0
    
    # Join last 50 lines for analysis
    recent_output = '\n'.join(output_lines[-50:])
    
    # Look for explicit percentage
    pct_match = re.search(r'(\d+)%', recent_output)
    if pct_match:
        return min(int(pct_match.group(1)), 100)
    
    # Look for "X of Y" or "X/Y" patterns
    ratio_match = re.search(r'(\d+)\s+(?:of|/)\s+(\d+)', recent_output)
    if ratio_match:
        current = int(ratio_match.group(1))
        total = int(ratio_match.group(2))
        if total > 0:
            return min(int((current / total) * 100), 100)
    
    # If job is running but no progress found, return small number
    return 5


def parse_stats_from_output(output_lines: List[str]) -> Dict[str, int]:
    """
    Extract statistics from output
    Looks for patterns like:
    - "Documents Processed: 42"
    - "Entities Created: 1234"
    - "Claims: 567"
    """
    stats = {
        'documents': 0,
        'entities': 0,
        'claims': 0,
        'temporal': 0,
        'geographic': 0
    }
    
    if not output_lines:
        return stats
    
    output_text = '\n'.join(output_lines)
    
    # Document patterns
    doc_patterns = [
        r'Documents?\s+(?:Processed|Loaded|Created):\s*(\d+)',
        r'(\d+)\s+documents?\s+(?:processed|loaded|created)',
        r'Total\s+(?:documents|docs):\s*(\d+)'
    ]
    
    for pattern in doc_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            stats['documents'] = int(match.group(1))
            break
    
    # Entity patterns
    entity_patterns = [
        r'Entities?\s+(?:Extracted|Created):\s*(\d+)',
        r'(\d+)\s+entities?\s+(?:extracted|created)',
        r'Total\s+entities:\s*(\d+)'
    ]
    
    for pattern in entity_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            stats['entities'] = int(match.group(1))
            break
    
    # Claim patterns
    claim_patterns = [
        r'Claims?\s+(?:Extracted|Created):\s*(\d+)',
        r'(\d+)\s+claims?\s+(?:extracted|created)',
        r'Total\s+claims:\s*(\d+)'
    ]
    
    for pattern in claim_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            stats['claims'] = int(match.group(1))
            break
    
    # Temporal patterns
    temporal_patterns = [
        r'Temporal\s+(?:markers?|events?):\s*(\d+)',
        r'(\d+)\s+temporal\s+(?:markers?|events?)'
    ]
    
    for pattern in temporal_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            stats['temporal'] = int(match.group(1))
            break
    
    # Geographic patterns
    geo_patterns = [
        r'Geographic\s+(?:locations?|markers?):\s*(\d+)',
        r'(\d+)\s+geographic\s+(?:locations?|markers?)'
    ]
    
    for pattern in geo_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            stats['geographic'] = int(match.group(1))
            break
    
    return stats


def run_pipeline_background(job_id: str, params: LoadRequest):
    """Run the pipeline in background with real-time output"""
    temp_dir = None
    
    try:
        # Build command based on type using MODERN PIPELINE
        if params.type == 'pdfs':
            # Step 1: Determine source directory
            if params.selected_files:
                # Create temporary directory with only selected files
                temp_dir = TEMP_IMPORT_DIR / job_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                source_dir = Path(params.path)
                
                # Create symlinks to selected files
                for filename in params.selected_files:
                    src = source_dir / filename
                    dst = temp_dir / filename
                    
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src)
                        except Exception as e:
                            logger.warning(f"Could not create symlink for {filename}: {e}")
                            # Fallback: copy file
                            import shutil
                            shutil.copy2(src, dst)
                
                pdf_source = str(temp_dir)
            else:
                # Use full directory
                pdf_source = params.path
            
            # Create output directory for JSONL
            jsonl_output_dir = TEMP_IMPORT_DIR / f"{job_id}_jsonl"
            jsonl_output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_output = jsonl_output_dir / "chunks.jsonl"
            
            # Step 2: Document to JSONL conversion (standard multi-format pipeline)
            cmd = [
                'python3', 'mvp_multiformat_to_jsonl_parallel.py',
                pdf_source,
                str(jsonl_output_dir),
                '--chunk-size', '1000',
                '--overlap', '200',
                '--workers', '1',  # Single worker for wizard imports
                '--formats', 'pdf', 'txt', 'epub', 'docx'  # Support all formats
            ]
            
        elif params.type == 'jsonl':
            # Use extraction pipeline directly
            jsonl_output = Path(params.path)
            cmd = []  # Will be set after PDF conversion completes
        
        # Advanced mode - not currently needed for wizard, but keeping for API compatibility
        else:
            if params.jsonl_dir:
                cmd = ['python3', 'run_extraction_pipeline.py']
                cmd.extend(['--jsonl', params.jsonl_dir])
                cmd.extend(['--job-id', job_id])
            elif params.pdf_dir:
                cmd = ['python3', 'aegis_pdf_import.py']
                cmd.extend(['--pdfs', params.pdf_dir])
                cmd.extend(['--job-id', job_id])
            else:
                load_jobs[job_id] = {
                    'status': 'error',
                    'error': 'No source directory specified',
                    'output': []
                }
                return
        
        # Add Neo4j credentials if provided
        if params.neo4j_uri:
            cmd.extend(['--neo4j-uri', params.neo4j_uri])
        
        if params.neo4j_user:
            cmd.extend(['--neo4j-user', params.neo4j_user])
        
        if params.neo4j_password:
            cmd.extend(['--neo4j-password', params.neo4j_password])
        
        # Update job status
        load_jobs[job_id] = {
            'status': 'running',
            'started': time.time(),
            'output': [],
            'selected_files': params.selected_files or [],
            'file_count': len(params.selected_files) if params.selected_files else 0,
            'phase': 'pdf_conversion' if params.type == 'pdfs' else 'extraction'
        }
        
        # STEP 1: Document to JSONL (if needed)
        if params.type == 'pdfs' and cmd:
            logger.info(f"[{job_id}] STEP 1: Converting documents to JSONL")
            logger.info(f"[{job_id}] Command: {' '.join(cmd)}")
            load_jobs[job_id]['command_step1'] = ' '.join(cmd)
            
            # Run PDF conversion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, job_id, 'stdout')
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, job_id, 'stderr')
            )
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for PDF conversion
            return_code = process.wait(timeout=3600)  # 1 hour for PDF conversion
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            if return_code != 0:
                load_jobs[job_id]['status'] = 'error'
                load_jobs[job_id]['error'] = 'PDF to JSONL conversion failed'
                load_jobs[job_id]['phase'] = 'failed_pdf_conversion'
                logger.error(f"[{job_id}] PDF conversion failed with exit code {return_code}")
                return
            
            # Check if JSONL was created
            if not jsonl_output.exists():
                load_jobs[job_id]['status'] = 'error'
                load_jobs[job_id]['error'] = f'JSONL output not created: {jsonl_output}'
                load_jobs[job_id]['phase'] = 'failed_pdf_conversion'
                logger.error(f"[{job_id}] JSONL file not found: {jsonl_output}")
                return
            
            logger.info(f"[{job_id}] PDF conversion complete: {jsonl_output}")
        
        # STEP 2: Run extraction pipeline
        load_jobs[job_id]['phase'] = 'extraction'
        logger.info(f"[{job_id}] STEP 2: Running extraction pipeline")
        
        extraction_cmd = [
            'python3', 'run_extraction_pipeline.py',
            '--jsonl', str(jsonl_output),
            '--job-id', job_id,
            '--model', 'mistral-nemo:12b',
            '--batch-size', '10',
            '--checkpoint-dir', './checkpoints'
        ]
        
        # Add Neo4j credentials
        if params.neo4j_uri:
            extraction_cmd.extend(['--neo4j-uri', params.neo4j_uri])
        if params.neo4j_user:
            extraction_cmd.extend(['--neo4j-user', params.neo4j_user])
        if params.neo4j_password:
            extraction_cmd.extend(['--neo4j-password', params.neo4j_password])
        
        logger.info(f"[{job_id}] Command: {' '.join(extraction_cmd)}")
        load_jobs[job_id]['command_step2'] = ' '.join(extraction_cmd)
        
        # Run extraction pipeline
        process = subprocess.Popen(
            extraction_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Create threads to stream stdout and stderr
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(process.stdout, job_id, 'stdout')
        )
        stderr_thread = threading.Thread(
            target=stream_output,
            args=(process.stderr, job_id, 'stderr')
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait(timeout=36000)  # 10 hour timeout
        
        # Wait for output threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        # Parse final stats
        stats = parse_stats_from_output(load_jobs[job_id].get('output', []))
        stats['duration'] = int(time.time() - load_jobs[job_id]['started'])
        
        if return_code == 0:
            load_jobs[job_id]['status'] = 'completed'
            load_jobs[job_id]['stats'] = stats
            load_jobs[job_id]['completed'] = time.time()
            logger.info(f"Pipeline job {job_id} completed successfully: {stats}")
        else:
            error_output = '\n'.join([
                line for line in load_jobs[job_id].get('output', [])
                if 'error' in line.lower() or 'exception' in line.lower()
            ])
            load_jobs[job_id]['status'] = 'error'
            load_jobs[job_id]['error'] = error_output or f'Pipeline failed with exit code {return_code}'
            load_jobs[job_id]['completed'] = time.time()
            logger.error(f"Pipeline job {job_id} failed: {load_jobs[job_id]['error']}")
    
    except subprocess.TimeoutExpired:
        load_jobs[job_id]['status'] = 'error'
        load_jobs[job_id]['error'] = 'Pipeline timeout (>10 hours)'
        load_jobs[job_id]['completed'] = time.time()
        logger.error(f"Pipeline job {job_id} timed out")
    
    except Exception as e:
        load_jobs[job_id]['status'] = 'error'
        load_jobs[job_id]['error'] = str(e)
        load_jobs[job_id]['completed'] = time.time()
        logger.error(f"Pipeline job {job_id} error: {e}", exc_info=True)
    
    finally:
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not cleanup temp directory {temp_dir}: {e}")


# ============================================================================
# NEW ENDPOINT: List files in inbox
# ============================================================================

@data_router.get("/list-inbox-files")
async def list_inbox_files():
    """
    List all supported document files in the inbox directory
    Supports: PDF, TXT, EPUB, DOCX
    
    Returns:
        {
            "success": true,
            "files": [
                {
                    "filename": "document.pdf",
                    "path": "/app/data/inbox/document.pdf",
                    "size_mb": 2.45,
                    "type": "pdf"
                }
            ],
            "total_count": 15,
            "total_size_mb": 234.5,
            "inbox_path": "/app/data/inbox",
            "format_counts": {"pdf": 10, "txt": 3, "epub": 2}
        }
    """
    try:
        # Ensure inbox directory exists
        if not INBOX_DIR.exists():
            INBOX_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created inbox directory: {INBOX_DIR}")
        
        files = []
        total_size = 0
        format_counts = {}
        
        # Supported formats
        formats = ['*.pdf', '*.txt', '*.epub', '*.docx', '*.doc']
        
        # List all supported files
        for pattern in formats:
            for doc_file in sorted(INBOX_DIR.glob(pattern)):
                # Skip Mac resource forks
                if doc_file.name.startswith('._'):
                    continue
                
                file_type = doc_file.suffix[1:].lower()  # Remove the dot
                size_bytes = doc_file.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                total_size += size_mb
                
                # Track format counts
                format_counts[file_type] = format_counts.get(file_type, 0) + 1
                
                files.append({
                    "filename": doc_file.name,
                    "path": str(doc_file),
                    "size_mb": round(size_mb, 2),
                    "type": file_type
                })
        
        return {
            "success": True,
            "files": files,
            "total_count": len(files),
            "total_size_mb": round(total_size, 2),
            "inbox_path": str(INBOX_DIR),
            "format_counts": format_counts
        }
    
    except Exception as e:
        logger.error(f"Error listing inbox files: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "files": [],
            "total_count": 0,
            "total_size_mb": 0,
            "inbox_path": str(INBOX_DIR)
        }


# ============================================================================
# EXISTING ENDPOINTS (with enhancements)
# ============================================================================

@data_router.post("/load-pipeline", response_model=LoadResponse)
async def load_pipeline(request: LoadRequest, background_tasks: BackgroundTasks):
    """
    Start data loading pipeline
    
    For simple mode:
    - type: 'pdfs' or 'jsonl'
    - path: directory path
    - selected_files: (OPTIONAL) list of specific filenames to import
    
    For advanced mode:
    - pdf_dir or jsonl_dir
    - optional training_script, skip_generation, skip_loading
    - neo4j connection details
    """
    
    # Validate input
    if request.type:
        # Simple mode
        if not request.path:
            return LoadResponse(
                success=False,
                error="Path is required for simple mode"
            )
    else:
        # Advanced mode
        if not request.pdf_dir and not request.jsonl_dir:
            return LoadResponse(
                success=False,
                error="Either pdf_dir or jsonl_dir is required"
            )
    
    # Generate job ID
    job_id = f"load_{int(time.time())}_{os.getpid()}"
    
    # Start background task
    thread = threading.Thread(
        target=run_pipeline_background,
        args=(job_id, request)
    )
    thread.daemon = True
    thread.start()
    
    return LoadResponse(
        success=True,
        message="Pipeline started",
        job_id=job_id
    )


@data_router.get("/load-status/{job_id}")
async def get_load_status(job_id: str):
    """
    Get status of a loading job with recent output
    
    ENHANCED: Now includes progress percentage and live stats
    """
    
    if job_id not in load_jobs:
        return {
            'success': False,
            'error': 'Job not found'
        }
    
    job = load_jobs[job_id]
    
    response = {
        'success': True,
        'status': job['status'],
        'output': job.get('output', [])[-50:],  # Last 50 lines
        'file_count': job.get('file_count', 0),
        'selected_files': job.get('selected_files', [])
    }
    
    # Calculate duration
    if job['status'] == 'running':
        response['duration'] = int(time.time() - job.get('started', time.time()))
        
        # Calculate progress percentage (NEW)
        response['progress_percent'] = parse_progress_from_output(job.get('output', []))
        
        # Parse current stats (NEW)
        response['stats'] = parse_stats_from_output(job.get('output', []))
        
    elif job['status'] == 'completed':
        response['stats'] = job.get('stats', {})
        response['duration'] = int(job.get('completed', time.time()) - job.get('started', time.time()))
        response['progress_percent'] = 100
        
    elif job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
        response['duration'] = int(job.get('completed', time.time()) - job.get('started', time.time()))
    
    return response


@data_router.get("/load-output/{job_id}")
async def get_load_output(job_id: str, lines: int = 100):
    """Get recent output lines from a job"""
    
    if job_id not in load_jobs:
        return {
            'success': False,
            'error': 'Job not found'
        }
    
    output = load_jobs[job_id].get('output', [])
    
    return {
        'success': True,
        'output': output[-lines:],
        'total_lines': len(output)
    }


@data_router.get("/load-jobs")
async def list_load_jobs():
    """List all loading jobs"""
    # Return simplified view without full output
    jobs_summary = {}
    for job_id, job_data in load_jobs.items():
        jobs_summary[job_id] = {
            'status': job_data.get('status'),
            'started': job_data.get('started'),
            'command': job_data.get('command'),
            'file_count': job_data.get('file_count', 0),
            'duration': int(time.time() - job_data.get('started', time.time())) if job_data.get('status') == 'running' else None
        }
    
    return {
        'success': True,
        'jobs': jobs_summary
    }


@data_router.delete("/load-job/{job_id}")
async def delete_load_job(job_id: str):
    """Delete/clear a completed job from memory"""
    if job_id in load_jobs:
        del load_jobs[job_id]
        return {
            'success': True,
            'message': f'Job {job_id} deleted'
        }
    
    return {
        'success': False,
        'error': 'Job not found'
    }
