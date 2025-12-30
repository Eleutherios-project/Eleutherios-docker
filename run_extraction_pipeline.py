#!/usr/bin/env python3
"""
Aegis Insight - Main Extraction Pipeline v2
End-to-end pipeline: JSONL -> Extraction + Embeddings -> Neo4j Graph

NEW in v2:
- Automatic embedding generation (--enable-embeddings flag)
- Support for all 7 extractors (emotion, authority)
- Better progress logging

Usage:
    python run_extraction_pipeline.py --jsonl /path/to/file.jsonl --enable-embeddings
    python run_extraction_pipeline.py --jsonl /path/to/file.jsonl --resume --job-id previous_job
"""

import json
import re
from typing import Dict, List, Optional
import logging
from aegis_config import Config
import argparse
from pathlib import Path
from datetime import datetime

# Import our components
from aegis_extraction_orchestrator_v3 import EnhancedExtractionOrchestrator as ExtractionOrchestrator, ExtractionConfig
from aegis_graph_builder import GraphBuilderV2 as GraphBuilder


def load_jsonl(jsonl_path: Path, max_chunks: int = None) -> List[Dict]:
    """
    Load and convert JSONL file to chunk format.
    
    Handles multiple input formats:
    - New format: {text, metadata: {source, source_type, ...}}
    - Old format: {content, context_metadata: {source_file, ...}}
    
    Returns:
        List of chunk dicts ready for processing
    """
    
    logging.info(f"Loading JSONL from: {jsonl_path}")
    
    chunks = []
    document_source = None  # Track document source for all chunks
    
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_chunks and idx >= max_chunks:
                break
            
            try:
                record = json.loads(line)
                
                # Get text content (try multiple field names)
                text = record.get('content') or record.get('text') or ''
                
                # Build context_metadata from available sources
                context_metadata = {}
                
                # Try old format first (context_metadata)
                if 'context_metadata' in record:
                    context_metadata = record['context_metadata'].copy()
                
                # Try new format (metadata) - this is what the chunker outputs
                if 'metadata' in record:
                    meta = record['metadata']
                    # Map metadata fields to context_metadata fields
                    if 'source' in meta and not context_metadata.get('source_file'):
                        context_metadata['source_file'] = meta['source']
                    if 'source_type' in meta:
                        context_metadata['source_type'] = meta['source_type']
                    if 'doc_id' in meta:
                        context_metadata['doc_id'] = meta['doc_id']
                    if 'chunk_index' in meta:
                        context_metadata['chunk_index'] = meta['chunk_index']
                    if 'total_chunks' in meta:
                        context_metadata['total_chunks'] = meta['total_chunks']
                
                # Track document source from first chunk
                if idx == 0 and context_metadata.get('source_file'):
                    document_source = context_metadata['source_file']
                
                # Fallback: use JSONL filename if still no source_file
                if not context_metadata.get('source_file'):
                    context_metadata['source_file'] = document_source or str(jsonl_path.name)
                
                # Set defaults for missing fields
                if 'domain' not in context_metadata:
                    context_metadata['domain'] = 'general'
                if 'title' not in context_metadata:
                    # Extract title from source_file
                    src = context_metadata.get('source_file', '')
                    title = Path(src).stem if src else jsonl_path.stem
                    # Clean up title (remove extensions, underscores)
                    title = re.sub(r'[_-]', ' ', title)
                    title = re.sub(r'\.(pdf|txt|jsonl|json)$', '', title, flags=re.IGNORECASE)
                    context_metadata['title'] = title.strip()
                
                # Create chunk in expected format
                chunk = {
                    'chunk_id': record.get('metadata', {}).get('doc_id') or f"{jsonl_path.stem}_{idx}",
                    'text': text,
                    'sequence_num': record.get('sequence_num', idx),
                    'context_metadata': context_metadata
                }
                
                chunks.append(chunk)
                
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {idx}: {e}")
                continue
    
    # Log what we found
    if chunks:
        src = chunks[0]['context_metadata'].get('source_file', 'unknown')
        logging.info(f"Loaded {len(chunks)} chunks from source: {src}")
    else:
        logging.info(f"Loaded 0 chunks")
    
    return chunks


def main():
    """Main pipeline execution"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Aegis Insight Extraction Pipeline v2')
    parser.add_argument('--jsonl', required=True, help='Path to JSONL file')
    parser.add_argument('--max-chunks', type=int, help='Maximum number of chunks to process')
    parser.add_argument('--job-id', help='Job ID for checkpointing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--skip-graph', action='store_true', help='Skip Neo4j graph building')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/mvp', help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=30, help='Batch size for processing')
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Checkpoint frequency')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama API URL')
    parser.add_argument('--model', default='mistral-nemo:12b', help='LLM model to use')
    
    # NEW: Embedding parameters
    parser.add_argument('--enable-embeddings', action='store_true', 
                        help='Generate embeddings inline (default: False for backwards compat)')
    parser.add_argument('--postgres-host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--postgres-db', default='aegis_insight', help='PostgreSQL database')
    parser.add_argument('--postgres-user', default='aegis', help='PostgreSQL user')
    parser.add_argument('--postgres-password', default=Config.POSTGRES_PASSWORD, help='PostgreSQL password')
    
    # Neo4j parameters
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password', default=Config.NEO4J_PASSWORD, help='Neo4j password')
    
    # Logging
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    logger.info(f"Starting job: {job_id}")
    
    # Load JSONL
    chunks = load_jsonl(jsonl_path, args.max_chunks)
    if not chunks:
        logger.error("No chunks loaded")
        return 1
    
    # Create PostgreSQL config if embeddings enabled
    postgres_config = None
    if args.enable_embeddings:
        postgres_config = {
            'host': args.postgres_host,
            'database': args.postgres_db,
            'user': args.postgres_user,
            'password': args.postgres_password
        }
        logger.info("Embeddings: ENABLED")
    else:
        logger.info("Embeddings: DISABLED (use --enable-embeddings to enable)")
    
    # Create orchestrator
    logger.info("Initializing extraction orchestrator v2...")
    # Create config for v3 orchestrator
    extraction_config = ExtractionConfig(
        ollama_url=args.ollama_url,
        primary_model=args.model,
        enable_coreference=True,
        enable_entity_clustering=True
    )
    
    orchestrator = ExtractionOrchestrator(
        config=extraction_config,
        logger=logger
    )
    
    # Run extraction using v3 API
    logger.info("Starting extraction process...")
    
    # v3 expects list of chunk texts and document context
    chunk_texts = [chunk.get('text', chunk.get('content', '')) for chunk in chunks]
    
    # Build context from original chunk metadata
    document_context = {
        'job_id': job_id,
        'source_file': chunks[0].get('context_metadata', {}).get('source_file', 'unknown') if chunks else 'unknown',
        'title': chunks[0].get('title', '') if chunks else '',
        'domain': chunks[0].get('domain', 'general') if chunks else 'general'
    }
    
    # Process all chunks - returns list of chunk results with entities/claims
    extracted_data = orchestrator.process_document(chunk_texts, document_context)
    
    result = {'extracted_data': extracted_data, 'stats': orchestrator.get_stats()}
    
    # Save extracted data to JSONL
    output_path = Path(args.checkpoint_dir) / f"{job_id}_extracted.jsonl"
    with open(output_path, 'w') as f:
        for item in result['extracted_data']:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Extracted data saved to: {output_path}")
    
    # Build Neo4j graph (unless skipped)
    if not args.skip_graph:
        logger.info("Building Neo4j graph...")
        try:
            builder = GraphBuilder(
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
                logger=logger
            )
            
            graph_stats = builder.build_graph(result['extracted_data'])
            builder.close()
            
            logger.info("[OK] Graph building complete!")
            
            # Auto-generate embeddings for new claims
            logger.info("Generating embeddings for new claims...")
            try:
                import subprocess
                embed_result = subprocess.run(
                    ['python', 'generate_embeddings.py'],
                    cwd='/media/bob/RAID11/DataShare/AegisTrustNet',
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 min timeout
                )
                if embed_result.returncode == 0:
                    logger.info("[OK] Embeddings generated successfully")
                else:
                    logger.warning(f"Embedding generation had issues: {embed_result.stderr[-500:] if embed_result.stderr else 'unknown'}")
            except Exception as emb_err:
                logger.warning(f"Embedding generation failed: {emb_err}")
                logger.warning("Run 'python generate_embeddings.py' manually to generate embeddings")
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            logger.warning("Extracted data is saved, but graph was not built")
    else:
        logger.info("Skipping graph building (--skip-graph flag)")
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Job ID: {job_id}")
    stats = result.get('stats', {})
    logger.info(f"Chunks processed: {stats.get('chunks_processed', len(result.get('extracted_data', [])))}")
    logger.info(f"Entities extracted: {stats.get('entities_extracted', 0)}")
    logger.info(f"Claims extracted: {stats.get('claims_extracted', 0)}")
    if stats.get('claims_with_temporal'):
        logger.info(f"  - With temporal data: {stats['claims_with_temporal']}")
    if stats.get('claims_with_geographic'):
        logger.info(f"  - With geographic data: {stats['claims_with_geographic']}")
    if stats.get('claims_with_citations'):
        logger.info(f"  - With citations: {stats['claims_with_citations']}")
    if stats.get('claims_with_emotional'):
        logger.info(f"  - With emotional data: {stats['claims_with_emotional']}")
    if stats.get('claims_with_authority'):
        logger.info(f"  - With authority data: {result['stats']['claims_with_authority']}")
    
    # NEW: Show embedding stats
    if args.enable_embeddings and 'embeddings_generated' in result['stats']:
        logger.info(f"Embeddings generated: {result['stats']['embeddings_generated']}")
    
    logger.info(f"Extracted data: {output_path}")
    logger.info("="*70 + "\n")
    
    # Close orchestrator connections
    # orchestrator cleanup not needed for v3
    
    return 0


if __name__ == "__main__":
    exit(main())
