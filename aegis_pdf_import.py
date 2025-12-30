#!/usr/bin/env python3
"""
Aegis PDF Import Pipeline
Handles: PDF → JSONL → Extraction → Neo4j

Usage:
    python aegis_pdf_import.py --pdfs /path/to/pdfs --job-id import_001
    python aegis_pdf_import.py --pdfs /path/to/file.pdf --job-id import_002
"""

import argparse
import subprocess
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and stream output"""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"{description} failed with return code {process.returncode}")
            return False
        
        logger.info(f"{description} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"{description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Aegis PDF Import Pipeline')
    parser.add_argument('--pdfs', required=True, help='Path to PDF file or directory')
    parser.add_argument('--job-id', help='Job ID for tracking')
    parser.add_argument('--output-dir', default='./import_output', help='Output directory for JSONL')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--model', default='mistral-nemo:12b', help='LLM model to use')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for extraction')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password', default='aegistrusted', help='Neo4j password')
    
    args = parser.parse_args()
    
    # Generate job ID if not provided
    job_id = args.job_id or f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting PDF import job: {job_id}")
    
    # Validate PDF path
    pdf_path = Path(args.pdfs)
    if not pdf_path.exists():
        logger.error(f"PDF path not found: {pdf_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert PDFs to JSONL
    logger.info("="*70)
    logger.info("STEP 1: Converting PDFs to JSONL")
    logger.info("="*70)
    
    jsonl_output = output_dir / "chunks.jsonl"
    
    pdf_cmd = [
        'python3', 'mvp_pdf_to_jsonl_parallel.py',
        '--input', str(pdf_path),
        '--output', str(jsonl_output),
        '--chunk-size', '1000',
        '--overlap', '200'
    ]
    
    if not run_command(pdf_cmd, "PDF to JSONL conversion"):
        logger.error("Failed to convert PDFs to JSONL")
        return 1
    
    # Check if JSONL was created and has content
    if not jsonl_output.exists():
        logger.error(f"JSONL output not created: {jsonl_output}")
        return 1
    
    # Count chunks
    chunk_count = 0
    with open(jsonl_output, 'r') as f:
        for line in f:
            if line.strip():
                chunk_count += 1
    
    logger.info(f"Created JSONL with {chunk_count} chunks")
    
    if chunk_count == 0:
        logger.error("No chunks in JSONL file")
        return 1
    
    # Step 2: Run extraction pipeline
    logger.info("="*70)
    logger.info("STEP 2: Running extraction pipeline")
    logger.info("="*70)
    
    extract_cmd = [
        'python3', 'run_extraction_pipeline.py',
        '--jsonl', str(jsonl_output),
        '--job-id', job_id,
        '--model', args.model,
        '--batch-size', str(args.batch_size),
        '--checkpoint-dir', args.checkpoint_dir,
        '--neo4j-uri', args.neo4j_uri,
        '--neo4j-user', args.neo4j_user,
        '--neo4j-password', args.neo4j_password
    ]
    
    if not run_command(extract_cmd, "Extraction and graph building"):
        logger.error("Failed to run extraction pipeline")
        return 1
    
    # Success!
    logger.info("="*70)
    logger.info("PDF IMPORT COMPLETE")
    logger.info("="*70)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"JSONL output: {jsonl_output}")
    logger.info(f"Chunks processed: {chunk_count}")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
