#!/usr/bin/env python3
"""
Aegis Insight - Extraction Pipeline v3
======================================

End-to-end pipeline with:
- Coreference resolution (pronouns → entities)
- Entity normalization and clustering
- True dual-GPU parallelism via dual Ollama instances
- Improved graph building

Usage:
    # Single GPU (default)
    python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl
    
    # Dual GPU (requires launch_dual_ollama.sh start)
    python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl --dual-ollama
    
    # Resume from checkpoint
    python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl --resume --job-id previous_job

Author: Aegis Insight Team
Version: 3.0.0
"""

import os
import sys
import json
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import v3 components
from aegis_extraction_orchestrator_v3 import (
    ExtractionConfig,
    EnhancedExtractionOrchestrator,
    DualOllamaOrchestrator
)
from aegis_graph_builder import GraphBuilderV2 as GraphBuilder


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PRIMARY_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_SECONDARY_URL = "http://localhost:11435"
DEFAULT_MODEL = "mistral-nemo:12b"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CHECKPOINT_EVERY = 50


# =============================================================================
# Utility Functions
# =============================================================================

def check_ollama_instance(url: str, timeout: int = 5) -> bool:
    """Check if an Ollama instance is running."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def detect_dual_ollama() -> tuple:
    """
    Auto-detect if dual Ollama is available.
    
    Returns:
        (dual_available, primary_url, secondary_url)
    """
    primary_url = os.environ.get('OLLAMA_PRIMARY_URL', DEFAULT_PRIMARY_URL)
    secondary_url = os.environ.get('OLLAMA_SECONDARY_URL', DEFAULT_SECONDARY_URL)
    
    primary_ok = check_ollama_instance(primary_url)
    secondary_ok = check_ollama_instance(secondary_url)
    
    if primary_ok and secondary_ok:
        return True, primary_url, secondary_url
    elif primary_ok:
        return False, primary_url, None
    else:
        return False, None, None


def load_jsonl(jsonl_path: Path, max_chunks: int = None) -> List[Dict]:
    """
    Load JSONL file and convert to chunk format.
    
    Args:
        jsonl_path: Path to JSONL file
        max_chunks: Optional limit on number of chunks to load
        
    Returns:
        List of chunk dicts ready for processing
    """
    logging.info(f"Loading JSONL from: {jsonl_path}")
    
    chunks = []
    
    with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if max_chunks and idx >= max_chunks:
                break
            
            try:
                record = json.loads(line)
                
                # Handle various JSONL formats
                text = record.get('content') or record.get('text') or record.get('chunk_text', '')
                
                if not text:
                    continue
                
                # Build context from various possible field names
                context = record.get('context_metadata') or record.get('context') or {}
                
                # Ensure required context fields
                if 'source_file' not in context:
                    context['source_file'] = record.get('source_file') or record.get('source') or record.get('metadata', {}).get('source') or str(jsonl_path)
                if 'domain' not in context:
                    context['domain'] = record.get('domain', 'unknown')
                if 'title' not in context:
                    context['title'] = record.get('title', jsonl_path.stem)
                
                chunk = {
                    'chunk_id': f"{jsonl_path.stem}_{idx}",
                    'text': text,
                    'sequence_num': record.get('sequence_num', idx),
                    'context': context
                }
                
                chunks.append(chunk)
                
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {idx}: {e}")
                continue
    
    logging.info(f"Loaded {len(chunks)} chunks")
    return chunks


def save_checkpoint(checkpoint_path: Path, state: Dict):
    """Save checkpoint state."""
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint state if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    jsonl_path: Path,
    job_id: str,
    config: ExtractionConfig,
    use_dual_ollama: bool = False,
    max_chunks: int = None,
    skip_graph: bool = False,
    resume: bool = False,
    checkpoint_dir: Path = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    neo4j_config: Dict = None
) -> Dict:
    """
    Run the full extraction pipeline.
    
    Args:
        jsonl_path: Path to input JSONL file
        job_id: Unique job identifier
        config: Extraction configuration
        use_dual_ollama: Whether to use dual Ollama instances
        max_chunks: Maximum chunks to process (for testing)
        skip_graph: Skip Neo4j graph building
        resume: Resume from checkpoint
        checkpoint_dir: Directory for checkpoints
        checkpoint_every: Save checkpoint every N chunks
        neo4j_config: Neo4j connection settings
        
    Returns:
        Dict with stats and results
    """
    logger = logging.getLogger(__name__)
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path('./checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{job_id}_checkpoint.json"
    output_path = checkpoint_dir / f"{job_id}_extracted.jsonl"
    
    # Load chunks
    chunks = load_jsonl(jsonl_path, max_chunks)
    if not chunks:
        logger.error("No chunks loaded")
        return {'error': 'No chunks loaded'}
    
    # Check for resume
    start_idx = 0
    extracted_data = []
    
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_idx = checkpoint.get('last_processed_idx', 0) + 1
            logger.info(f"Resuming from chunk {start_idx}")
            
            # Load previously extracted data
            if output_path.exists():
                with open(output_path, 'r') as f:
                    for line in f:
                        extracted_data.append(json.loads(line))
    
    # Create orchestrator (always single-GPU mode)
    # Dual-GPU parallelism is handled at the FILE level by run_dual_gpu_extraction.sh
    logger.info("Initializing extraction orchestrator...")
    
    orchestrator = EnhancedExtractionOrchestrator(config=config, logger=logger)
    
    # Process chunks
    logger.info(f"Processing {len(chunks) - start_idx} chunks (starting from {start_idx})...")
    
    start_time = datetime.now()
    
    # Different processing paths for single vs dual GPU
    # NOTE: Dual GPU parallelism is now handled at the FILE level via run_dual_gpu_extraction.sh
    # Each pipeline process uses a single GPU
    
    logger.info("Using single-GPU sequential processing")
    
    # Open output file for appending
    with open(output_path, 'a') as out_file:
        for idx in range(start_idx, len(chunks)):
            chunk = chunks[idx]
            
            try:
                # Process chunk through v3 pipeline
                result = orchestrator.process_chunk(
                    chunk_text=chunk['text'],
                    context=chunk.get('context', {})
                )
                
                # Add chunk metadata
                result['chunk_id'] = chunk['chunk_id']
                result['sequence_num'] = chunk['sequence_num']
                
                # Write to output
                out_file.write(json.dumps(result, default=str) + '\n')
                out_file.flush()
                
                extracted_data.append(result)
                
                # Progress logging
                if (idx + 1) % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (idx - start_idx + 1) / elapsed if elapsed > 0 else 0
                    remaining = len(chunks) - idx - 1
                    eta = remaining / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {idx + 1}/{len(chunks)} chunks | "
                        f"Rate: {rate:.2f}/s | ETA: {eta/60:.1f} min"
                    )
                
                # Checkpoint
                if (idx + 1) % checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, {
                        'job_id': job_id,
                        'last_processed_idx': idx,
                        'total_chunks': len(chunks),
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"Checkpoint saved at chunk {idx + 1}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {e}")
                # Save checkpoint on error
                save_checkpoint(checkpoint_path, {
                    'job_id': job_id,
                    'last_processed_idx': idx - 1,
                    'total_chunks': len(chunks),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                raise
    
    # Get orchestrator stats
    stats = orchestrator.get_stats()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Extraction complete in {elapsed/60:.1f} minutes")
    logger.info(f"Extracted data saved to: {output_path}")
    
    # Build Neo4j graph
    if not skip_graph:
        logger.info("Building Neo4j graph...")
        
        neo4j_config = neo4j_config or {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'aegistrusted'
        }
        
        try:
            builder = GraphBuilder(
                neo4j_uri=neo4j_config['uri'],
                neo4j_user=neo4j_config['user'],
                neo4j_password=neo4j_config['password']
            )
            
            graph_stats = builder.build_graph(extracted_data)
            builder.close()
            
            logger.info("✓ Graph building complete!")
            stats['graph_stats'] = graph_stats
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            logger.warning("Extracted data is saved, graph can be built later")
            stats['graph_error'] = str(e)
    else:
        logger.info("Skipping graph building (--skip-graph flag)")
    
    # Final checkpoint (mark complete)
    save_checkpoint(checkpoint_path, {
        'job_id': job_id,
        'last_processed_idx': len(chunks) - 1,
        'total_chunks': len(chunks),
        'status': 'complete',
        'timestamp': datetime.now().isoformat()
    })
    
    return {
        'job_id': job_id,
        'chunks_processed': len(chunks) - start_idx,
        'total_chunks': len(chunks),
        'output_path': str(output_path),
        'elapsed_seconds': elapsed,
        'stats': stats
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Aegis Insight Extraction Pipeline v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl
  
  # Dual GPU mode (requires launch_dual_ollama.sh start)
  python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl --dual-ollama
  
  # Resume from checkpoint
  python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl --resume --job-id my_job
  
  # Test with limited chunks
  python run_extraction_pipeline_v3.py --jsonl /path/to/file.jsonl --max-chunks 10
        """
    )
    
    # Required arguments
    parser.add_argument('--jsonl', required=True, help='Path to JSONL file')
    
    # Processing options
    parser.add_argument('--max-chunks', type=int, help='Maximum chunks to process')
    parser.add_argument('--job-id', help='Job ID for checkpointing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--skip-graph', action='store_true', help='Skip Neo4j graph building')
    
    # Dual Ollama options
    parser.add_argument('--dual-ollama', action='store_true', 
                        help='Use dual Ollama instances for parallel processing')
    parser.add_argument('--auto-detect-dual', action='store_true', default=True,
                        help='Auto-detect if dual Ollama is available (default: True)')
    parser.add_argument('--ollama-url', default=DEFAULT_PRIMARY_URL, 
                        help=f'Primary Ollama URL (default: {DEFAULT_PRIMARY_URL})')
    parser.add_argument('--secondary-ollama-url', default=DEFAULT_SECONDARY_URL,
                        help=f'Secondary Ollama URL (default: {DEFAULT_SECONDARY_URL})')
    
    # Model options
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'LLM model (default: {DEFAULT_MODEL})')
    
    # Pipeline options
    parser.add_argument('--enable-coreference', action='store_true', default=True,
                        help='Enable coreference resolution (default: True)')
    parser.add_argument('--enable-clustering', action='store_true', default=True,
                        help='Enable entity clustering (default: True)')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                        help='Entity similarity threshold (default: 0.85)')
    
    # Checkpoint options
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint-every', type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help=f'Checkpoint frequency (default: {DEFAULT_CHECKPOINT_EVERY})')
    
    # Neo4j options
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password', default='aegistrusted', help='Neo4j password')
    
    # Logging
    parser.add_argument('--log-level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = f"extraction_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        return 1
    
    # Generate job ID if not provided
    job_id = args.job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Detect dual Ollama - ONLY if not explicitly specified
    use_dual = False  # Default to single GPU
    primary_url = args.ollama_url
    secondary_url = None
    
    # Only auto-detect if using default URL and auto-detect is enabled
    if args.ollama_url == DEFAULT_PRIMARY_URL and args.auto_detect_dual:
        dual_available, detected_primary, detected_secondary = detect_dual_ollama()
        if dual_available:
            logger.info("Auto-detected dual Ollama instances available (but using single for file-level parallelism)")
        elif detected_primary:
            logger.info("Single Ollama instance detected")
            primary_url = detected_primary
        else:
            logger.error("No Ollama instance detected. Is Ollama running?")
            return 1
    else:
        # Explicit URL provided - just verify it works
        if not check_ollama_instance(primary_url):
            logger.error(f"Ollama not responding at {primary_url}")
            return 1
        logger.info(f"Using specified Ollama at {primary_url}")
    
    # Create configuration
    config = ExtractionConfig(
        ollama_url=primary_url,
        secondary_ollama_url=secondary_url if use_dual else None,
        primary_model=args.model,
        enable_coreference=args.enable_coreference,
        enable_entity_clustering=args.enable_clustering,
        entity_similarity_threshold=args.similarity_threshold
    )
    
    # Print configuration
    logger.info("="*70)
    logger.info("AEGIS INSIGHT EXTRACTION PIPELINE v3")
    logger.info("="*70)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Input: {jsonl_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Ollama URL: {primary_url}")
    logger.info(f"Coreference: {'Enabled' if args.enable_coreference else 'Disabled'}")
    logger.info(f"Entity clustering: {'Enabled' if args.enable_clustering else 'Disabled'}")
    logger.info("="*70)
    
    # Run pipeline
    try:
        result = run_pipeline(
            jsonl_path=jsonl_path,
            job_id=job_id,
            config=config,
            use_dual_ollama=use_dual,
            max_chunks=args.max_chunks,
            skip_graph=args.skip_graph,
            resume=args.resume,
            checkpoint_dir=Path(args.checkpoint_dir),
            checkpoint_every=args.checkpoint_every,
            neo4j_config={
                'uri': args.neo4j_uri,
                'user': args.neo4j_user,
                'password': args.neo4j_password
            }
        )
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Job ID: {result['job_id']}")
        logger.info(f"Chunks processed: {result['chunks_processed']}")
        logger.info(f"Time: {result['elapsed_seconds']/60:.1f} minutes")
        logger.info(f"Output: {result['output_path']}")
        
        stats = result.get('stats', {})
        logger.info(f"Entities extracted: {stats.get('entities_extracted', 'N/A')}")
        logger.info(f"Claims extracted: {stats.get('claims_extracted', 'N/A')}")
        logger.info(f"Coreferences resolved: {stats.get('coreferences_resolved', 'N/A')}")
        logger.info(f"Entities clustered: {stats.get('entities_clustered', 'N/A')}")
        
        if 'graph_stats' in stats:
            logger.info(f"Graph nodes created: {stats['graph_stats'].get('nodes_created', 'N/A')}")
        
        logger.info("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
