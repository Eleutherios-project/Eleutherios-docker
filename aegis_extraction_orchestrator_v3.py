"""
Aegis Insight - Enhanced Extraction Orchestrator v3
Integrates coreference resolution and entity normalization/clustering

Pipeline:
1. Chunk text
2. Coreference resolution (pronouns → explicit entities)
3. Entity extraction (from resolved text)
4. Entity normalization and clustering
5. Claim extraction (from resolved text)
6. Temporal/Geographic/Citation extraction
7. Build relationships
8. Output JSONL

Key improvements over v2:
- Coreference resolution before claim extraction
- Entity normalization at extraction time
- Entity clustering to prevent fragmentation
- Shortened source paths
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import sys

# Import extractors
from aegis_coreference_resolver import CoreferenceResolver
from aegis_entity_processor import EntityProcessor, NormalizedEntity

# Import existing extractors (these should be in the path)
try:
    from aegis_entity_extractor import EntityExtractor
    from aegis_claim_extractor import ClaimExtractor
    from aegis_temporal_extractor import TemporalExtractor
    from aegis_geographic_extractor import GeographicExtractor
    from aegis_citation_extractor import CitationExtractor
    from aegis_emotion_extractor import AegisEmotionExtractor as EmotionExtractor
    from aegis_authority_domain_analyzer import AuthorityDomainAnalyzer
except ImportError as e:
    print(f"Warning: Could not import extractor: {e}")
    print("Ensure all extractor modules are in the Python path")


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    primary_model: str = "mistral-nemo:12b"  # For most extraction
    large_model: str = "qwen2.5:72b"  # For complex tasks if available
    
    # Processing options
    enable_coreference: bool = True
    enable_entity_clustering: bool = True
    entity_similarity_threshold: float = 0.85
    
    # Quality thresholds
    min_claim_confidence: float = 0.5
    min_entity_confidence: float = 0.5
    coreference_confidence: float = 0.7
    
    # Output options
    source_path_depth: int = 3  # Keep last N path components
    
    # Dual GPU support
    secondary_ollama_url: Optional[str] = None  # e.g., "http://localhost:11435"



# Helper function for document-level date extraction
def extract_section_date(text: str) -> Optional[str]:
    """
    Extract date from section headers in document text.
    
    Looks for patterns like:
    - February 18, 1898
    - March 29, 1898
    - 1898-02-18
    - 02/18/1898
    
    Returns ISO format date string or None.
    """
    import re
    from datetime import datetime
    
    # Pattern 1: Month Day, Year (e.g., "February 18, 1898")
    pattern1 = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        month_name, day, year = match.groups()
        try:
            date_obj = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
            return date_obj.strftime("%Y-%m-%d")
        except:
            pass
    
    # Pattern 2: ISO format (e.g., "1898-02-18")
    pattern2 = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(pattern2, text)
    if match:
        return match.group(0)
    
    # Pattern 3: US format (e.g., "02/18/1898")
    pattern3 = r'(\d{1,2})/(\d{1,2})/(\d{4})'
    match = re.search(pattern3, text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Pattern 4: Just year in context (e.g., "1898" near "February")
    # Skip this for now - too many false positives
    
    return None


class EnhancedExtractionOrchestrator:
    """
    Orchestrates the full extraction pipeline with enhancements.
    """
    
    def __init__(self,
                 config: Optional[ExtractionConfig] = None,
                 neo4j_driver = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize orchestrator.
        
        Args:
            config: Extraction configuration
            neo4j_driver: Optional Neo4j driver for entity clustering
            logger: Optional logger
        """
        self.config = config or ExtractionConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self._init_components(neo4j_driver)
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'claims_extracted': 0,
            'entities_extracted': 0,
            'entities_clustered': 0,
            'coreferences_resolved': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _init_components(self, neo4j_driver):
        """Initialize all extraction components."""
        
        # Coreference resolver
        if self.config.enable_coreference:
            self.coreference_resolver = CoreferenceResolver(
                ollama_url=self.config.ollama_url,
                model=self.config.primary_model,
                confidence_threshold=self.config.coreference_confidence,
                logger=self.logger
            )
        else:
            self.coreference_resolver = None
        
        # Entity processor (normalizer + cluster matcher)
        if self.config.enable_entity_clustering:
            self.entity_processor = EntityProcessor(
                neo4j_driver=neo4j_driver,
                similarity_threshold=self.config.entity_similarity_threshold,
                logger=self.logger
            )
        else:
            self.entity_processor = None
        
        # Standard extractors
        self.entity_extractor = EntityExtractor(
            ollama_url=self.config.ollama_url,
            model=self.config.primary_model,
            logger=self.logger
        )
        
        self.claim_extractor = ClaimExtractor(
            ollama_url=self.config.ollama_url,
            model=self.config.primary_model,
            logger=self.logger
        )
        
        self.temporal_extractor = TemporalExtractor(
            ollama_url=self.config.ollama_url,
            model=self.config.primary_model,
            logger=self.logger
        )
        
        self.geographic_extractor = GeographicExtractor(
            ollama_url=self.config.ollama_url,
            model=self.config.primary_model,
            logger=self.logger
        )
        
        self.citation_extractor = CitationExtractor(
            ollama_url=self.config.ollama_url,
            model=self.config.primary_model,
            logger=self.logger
        )

        # Emotion extractor for coordination detection
        try:
            self.emotion_extractor = EmotionExtractor(
                ollama_url=self.config.ollama_url,
                model=self.config.primary_model,
                logger=self.logger
            )
            self.logger.info("✓ Emotion extractor initialized")
        except Exception as e:
            self.logger.warning(f"Emotion extractor not available: {e}")
            self.emotion_extractor = None
        
        # Authority domain analyzer
        try:
            self.authority_analyzer = AuthorityDomainAnalyzer(
                ollama_url=self.config.ollama_url,
                model=self.config.primary_model,
                logger=self.logger
            )
            self.logger.info("✓ Authority analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Authority analyzer not available: {e}")
            self.authority_analyzer = None
    
    def process_chunk(self, 
                      chunk_text: str, 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single chunk through the full pipeline.
        
        Args:
            chunk_text: The text content of the chunk
            context: Context dict with source_file, title, domain, etc.
            
        Returns:
            Dict with extracted entities, claims, and metadata
        """
        self.stats['chunks_processed'] += 1
        
        result = {
            'chunk_id': self._generate_chunk_id(context, chunk_text),
            'text': chunk_text,
            'context': self._process_context(context),
            'entities': [],
            'claims': [],
            'coreference_metadata': {},
            'processing_timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Step 1: Coreference resolution
            resolved_text = chunk_text
            if self.coreference_resolver:
                resolved_text, coref_meta = self.coreference_resolver.resolve(
                    chunk_text,
                    entities_hint=None,  # Will extract entities first in future optimization
                    context=context
                )
                result['coreference_metadata'] = coref_meta
                if coref_meta.get('resolved'):
                    self.stats['coreferences_resolved'] += coref_meta.get('resolution_count', 1)
            
            # Step 2: Entity extraction (from original text to get all mentions)
            raw_entities = self.entity_extractor.extract(chunk_text, context)
            
            # Step 3: Entity normalization and clustering
            if self.entity_processor:
                processed_entities = []
                for entity in raw_entities:
                    normalized = self.entity_processor.process(
                        raw_name=entity.get('name', ''),
                        entity_type=entity.get('type'),
                        context_text=chunk_text[:200]
                    )
                    
                    # Update entity with canonical name
                    processed_entity = entity.copy()
                    processed_entity['normalized_name'] = normalized.normalized_name
                    processed_entity['canonical_name'] = normalized.canonical_name
                    processed_entity['matched_existing'] = normalized.matched_existing
                    
                    if normalized.matched_existing:
                        self.stats['entities_clustered'] += 1
                    
                    processed_entities.append(processed_entity)
                
                result['entities'] = processed_entities
            else:
                result['entities'] = raw_entities
            
            self.stats['entities_extracted'] += len(result['entities'])
            
            # Step 4: Claim extraction (from RESOLVED text for better entity linking)
            claims = self.claim_extractor.extract(resolved_text, context)
            
            # Step 5: Enrich claims with temporal/geographic/citation data
            # Extract document/section date from chunk text for soft date propagation
            document_date = extract_section_date(chunk_text)
            
            enriched_claims = []
            for claim in claims:
                enriched = self._enrich_claim(claim, context, document_date=document_date)
                enriched_claims.append(enriched)
            
            result['claims'] = enriched_claims
            self.stats['claims_extracted'] += len(enriched_claims)
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}")
            self.stats['errors'] += 1
            result['error'] = str(e)
        
        return result
    
    def _enrich_claim(self, claim: Dict, context: Dict, document_date: Optional[str] = None) -> Dict:
        """Enrich a claim with temporal, geographic, and citation data.
        
        Args:
            claim: The claim dict to enrich
            context: Document context
            document_date: Soft date from section header (used if no hard date in claim)
        """
        
        claim_text = claim.get('claim_text', '')
        
        # Temporal extraction
        try:
            temporal = self.temporal_extractor.extract(claim_text, context)
            claim['temporal_data'] = temporal
            
            # If no hard dates found but we have a document_date, add it as soft date
            if document_date and not temporal.get('absolute_dates'):
                claim['temporal_data']['soft_date'] = {
                    'date': document_date,
                    'source': 'document_section',
                    'confidence': 0.8,
                    'type': 'inferred'
                }
                # Also add to absolute_dates so detectors can find it
                claim['temporal_data']['absolute_dates'].append({
                    'date': document_date,
                    'confidence': 0.8,
                    'type': 'document_context',
                    'context': 'Date from document section header'
                })
        except Exception as e:
            self.logger.debug(f"Temporal extraction failed: {e}")
            temporal_data = {'absolute_dates': [], 'relative_dates': [], 'temporal_markers': []}
            # Still use document_date as fallback
            if document_date:
                temporal_data['soft_date'] = {
                    'date': document_date,
                    'source': 'document_section',
                    'confidence': 0.8,
                    'type': 'inferred'
                }
                temporal_data['absolute_dates'].append({
                    'date': document_date,
                    'confidence': 0.8,
                    'type': 'document_context',
                    'context': 'Date from document section header'
                })
            claim['temporal_data'] = temporal_data
        
        # Geographic extraction
        try:
            geographic = self.geographic_extractor.extract(claim_text, context)
            claim['geographic_data'] = geographic
        except Exception as e:
            self.logger.debug(f"Geographic extraction failed: {e}")
            claim['geographic_data'] = {'locations': [], 'cultural_context': []}
        
        # Citation extraction
        try:
            citation = self.citation_extractor.extract(claim_text, context)
            claim['citation_data'] = citation
        except Exception as e:
            self.logger.debug(f"Citation extraction failed: {e}")
            claim['citation_data'] = {'cites_other_work': False, 'attribution_chain': []}
        
        # Extract emotional content
        if self.emotion_extractor:
            try:
                emotion = self.emotion_extractor.extract(claim_text, context)
                claim['emotional_data'] = emotion
            except Exception as e:
                self.logger.debug(f"Emotion extraction failed: {e}")
                claim['emotional_data'] = {'sentiment': 'neutral', 'intensity': 0.0, 'fear': 0.0, 'anger': 0.0, 'urgency': 0.0}
        else:
            claim['emotional_data'] = {'sentiment': 'neutral', 'intensity': 0.0, 'fear': 0.0, 'anger': 0.0, 'urgency': 0.0}
        
        # Authority domain analysis
        if self.authority_analyzer:
            try:
                # Get entities from result if available (passed from process_chunk)
                entities = context.get('entities', [])
                authority = self.authority_analyzer.analyze(claim_text, entities, context)
                if authority:
                    claim['authority_data'] = authority
            except Exception as e:
                self.logger.debug(f"Authority analysis failed: {e}")
        
        return claim
    
    def _process_context(self, context: Dict) -> Dict:
        """Process context, including shortening source path."""
        
        processed = context.copy()
        
        # Shorten source file path
        if 'source_file' in processed:
            processed['source_file'] = self._shorten_path(
                processed['source_file'],
                self.config.source_path_depth
            )
        
        return processed
    
    def _shorten_path(self, full_path: str, keep_parts: int = 3) -> str:
        """Shorten a file path to last N components."""
        if not full_path:
            return full_path
        
        parts = Path(full_path).parts
        if len(parts) <= keep_parts:
            return full_path
        
        return str(Path(*parts[-keep_parts:]))
    
    def _generate_chunk_id(self, context: Dict, text: str) -> str:
        """Generate unique chunk ID."""
        source = context.get('source_file', 'unknown')
        seq = context.get('sequence_num', 0)
        
        hash_input = f"{source}_{seq}_{text[:100]}"
        hash_val = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        return f"chunk_{hash_val}"
    
    def process_document(self,
                         chunks: List[str],
                         document_context: Dict[str, Any]) -> List[Dict]:
        """
        Process all chunks from a document.
        
        Args:
            chunks: List of chunk texts
            document_context: Document-level context
            
        Returns:
            List of processed chunk results
        """
        results = []
        
        for i, chunk_text in enumerate(chunks):
            context = document_context.copy()
            context['sequence_num'] = i
            context['chunk_number'] = i
            
            result = self.process_chunk(chunk_text, context)
            results.append(result)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        return results
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        stats = self.stats.copy()
        
        # Add component stats
        if self.coreference_resolver:
            stats['coreference_stats'] = self.coreference_resolver.get_stats()
        
        if self.entity_processor:
            stats['entity_processor_stats'] = self.entity_processor.get_stats()
        
        return stats
    
    def write_jsonl(self, results: List[Dict], output_path: str):
        """Write results to JSONL file."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                # Convert any non-serializable objects
                serializable = self._make_serializable(result)
                f.write(json.dumps(serializable) + '\n')
        
        self.logger.info(f"Wrote {len(results)} records to {output_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


# =============================================================================
# DUAL OLLAMA SUPPORT - TRUE CHUNK-LEVEL PARALLELISM
# =============================================================================

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def warmup_ollama(url: str, model: str, logger) -> bool:
    """
    Send a small request to ensure model is loaded on GPU.
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"Warming up Ollama at {url} with model {model}...")
    
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": "Say 'ready' in one word.",
                "stream": False,
                "options": {"num_predict": 5}
            },
            timeout=120  # Model loading can take time
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Ollama at {url} warmed up successfully")
            return True
        else:
            logger.error(f"✗ Ollama at {url} returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to warm up Ollama at {url}: {e}")
        return False


class DualOllamaOrchestrator:
    """
    Orchestrator that uses two Ollama instances for TRUE parallel processing.
    
    Each GPU runs a COMPLETE extraction pipeline on SEPARATE chunks simultaneously.
    This gives ~2x throughput instead of splitting extractors across GPUs.
    
    Architecture:
        GPU 0 (Primary):   Full pipeline for Chunk N
        GPU 1 (Secondary): Full pipeline for Chunk N+1
        
    Both GPUs process different chunks at the same time.
    
    Note: Entity clustering is disabled during parallel extraction to avoid
    GPU memory conflicts. Clustering should happen as a post-processing step.
    """
    
    def __init__(self,
                 config: Optional[ExtractionConfig] = None,
                 neo4j_driver = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config or ExtractionConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate dual mode
        if not self.config.secondary_ollama_url:
            raise ValueError("DualOllamaOrchestrator requires secondary_ollama_url in config")
        
        self.primary_url = self.config.ollama_url
        self.secondary_url = self.config.secondary_ollama_url
        
        self.logger.info("="*60)
        self.logger.info("INITIALIZING DUAL GPU EXTRACTION")
        self.logger.info("="*60)
        
        # CRITICAL: Warm up BOTH Ollama instances BEFORE creating orchestrators
        # This ensures models are loaded on BOTH GPUs
        self.logger.info("Pre-warming both Ollama instances (this loads models on each GPU)...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            warmup_futures = {
                executor.submit(warmup_ollama, self.primary_url, self.config.primary_model, self.logger): "primary",
                executor.submit(warmup_ollama, self.secondary_url, self.config.primary_model, self.logger): "secondary"
            }
            
            warmup_results = {}
            for future in as_completed(warmup_futures):
                name = warmup_futures[future]
                warmup_results[name] = future.result()
        
        if not warmup_results.get('primary'):
            raise RuntimeError(f"Failed to warm up primary Ollama at {self.primary_url}")
        if not warmup_results.get('secondary'):
            raise RuntimeError(f"Failed to warm up secondary Ollama at {self.secondary_url}")
        
        self.logger.info("Both Ollama instances warmed up with models loaded")
        
        # Create TWO complete orchestrators - one per GPU
        self.logger.info("Creating extraction orchestrators...")
        
        # Primary orchestrator (GPU 0)
        primary_config = ExtractionConfig(
            ollama_url=self.primary_url,
            primary_model=self.config.primary_model,
            enable_coreference=self.config.enable_coreference,
            enable_entity_clustering=False,  # Disable - cluster post-extraction
            entity_similarity_threshold=self.config.entity_similarity_threshold,
            source_path_depth=self.config.source_path_depth,
            secondary_ollama_url=None
        )
        self.primary_orchestrator = EnhancedExtractionOrchestrator(
            config=primary_config,
            neo4j_driver=None,
            logger=self.logger
        )
        
        # Secondary orchestrator (GPU 1)
        secondary_config = ExtractionConfig(
            ollama_url=self.secondary_url,
            primary_model=self.config.primary_model,
            enable_coreference=self.config.enable_coreference,
            enable_entity_clustering=False,
            entity_similarity_threshold=self.config.entity_similarity_threshold,
            source_path_depth=self.config.source_path_depth,
            secondary_ollama_url=None
        )
        self.secondary_orchestrator = EnhancedExtractionOrchestrator(
            config=secondary_config,
            neo4j_driver=None,
            logger=self.logger
        )
        
        self.logger.info("="*60)
        self.logger.info("DUAL GPU READY")
        self.logger.info(f"  GPU 0 (Primary):   {self.primary_url}")
        self.logger.info(f"  GPU 1 (Secondary): {self.secondary_url}")
        self.logger.info("="*60)
        
        # Stats tracking
        self.stats = {
            'chunks_processed': 0,
            'primary_chunks': 0,
            'secondary_chunks': 0,
            'parallel_batches': 0,
            'entities_extracted': 0,
            'claims_extracted': 0,
            'coreferences_resolved': 0,
            'entities_clustered': 0,
            'errors': 0
        }
    
    def process_chunk(self, chunk_text: str, context: Dict = None) -> Dict:
        """
        Process a single chunk (uses primary GPU).
        For parallel processing, use process_chunks_parallel().
        """
        result = self.primary_orchestrator.process_chunk(chunk_text, context)
        self._update_stats(result)
        return result
    
    def process_chunks_parallel(self, chunks: List[Dict]) -> List[Dict]:
        """
        Process multiple chunks with true parallelism - 2 chunks at a time.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'context' keys
            
        Returns:
            List of extraction results in order
        """
        results = [None] * len(chunks)  # Pre-allocate to maintain order
        
        self.logger.info(f"Processing {len(chunks)} chunks with dual-GPU parallelism (2 at a time)...")
        
        def process_on_gpu(orchestrator, idx: int, chunk: Dict, gpu_name: str) -> tuple:
            """Process chunk on specified GPU."""
            start = time.time()
            try:
                result = orchestrator.process_chunk(
                    chunk['text'],
                    chunk.get('context', {})
                )
                elapsed = time.time() - start
                self.logger.info(f"  {gpu_name} completed chunk {idx} in {elapsed:.1f}s")
                return idx, result, gpu_name
            except Exception as e:
                self.logger.error(f"{gpu_name} error on chunk {idx}: {e}")
                return idx, {'error': str(e)}, gpu_name
        
        # Process in pairs - submit both, wait for both, then next pair
        for batch_start in range(0, len(chunks), 2):
            batch_num = batch_start // 2 + 1
            total_batches = (len(chunks) + 1) // 2
            self.logger.info(f"Batch {batch_num}/{total_batches}: chunks {batch_start} and {batch_start+1 if batch_start+1 < len(chunks) else 'N/A'}")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                # Submit chunk to primary GPU
                idx1 = batch_start
                futures[executor.submit(
                    process_on_gpu, 
                    self.primary_orchestrator, 
                    idx1, 
                    chunks[idx1], 
                    "GPU0"
                )] = idx1
                
                # Submit next chunk to secondary GPU (if exists)
                if batch_start + 1 < len(chunks):
                    idx2 = batch_start + 1
                    futures[executor.submit(
                        process_on_gpu, 
                        self.secondary_orchestrator, 
                        idx2, 
                        chunks[idx2], 
                        "GPU1"
                    )] = idx2
                
                # Wait for this pair to complete
                for future in as_completed(futures):
                    idx, result, gpu_name = future.result()
                    results[idx] = result
                    
                    if gpu_name == "GPU0":
                        self.stats['primary_chunks'] += 1
                    else:
                        self.stats['secondary_chunks'] += 1
                    
                    self._update_stats(result)
            
            self.stats['parallel_batches'] += 1
        
        self.logger.info(f"All {len(chunks)} chunks processed")
        self.logger.info(f"  GPU0 processed: {self.stats['primary_chunks']} chunks")
        self.logger.info(f"  GPU1 processed: {self.stats['secondary_chunks']} chunks")
        
        return results
    
    def _update_stats(self, result: Dict):
        """Update aggregate statistics from a single result."""
        self.stats['chunks_processed'] += 1
        
        if 'error' in result:
            self.stats['errors'] += 1
            return
        
        self.stats['entities_extracted'] += len(result.get('entities', []))
        self.stats['claims_extracted'] += len(result.get('claims', []))
        
        if result.get('coreference_applied'):
            self.stats['coreferences_resolved'] += result.get('coreference_count', 0)
        
        # Count clustered entities
        for entity in result.get('entities', []):
            if entity.get('matched_to'):
                self.stats['entities_clustered'] += 1
    
    def get_stats(self) -> Dict:
        """Return aggregate statistics."""
        stats = self.stats.copy()
        
        # Add sub-orchestrator stats
        stats['primary_stats'] = self.primary_orchestrator.get_stats()
        stats['secondary_stats'] = self.secondary_orchestrator.get_stats()
        
        return stats


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Extraction Pipeline v3")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    parser.add_argument("--dual", action="store_true", help="Enable dual Ollama mode")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.input and args.output:
        process_file(args.input, args.output, dual_mode=args.dual)
    else:
        parser.print_help()


def run_tests():
    """Run integration tests."""
    print("="*70)
    print("ENHANCED EXTRACTION ORCHESTRATOR v3 - TESTS")
    print("="*70)
    
    # Test configuration
    config = ExtractionConfig(
        enable_coreference=True,
        enable_entity_clustering=True
    )
    
    orchestrator = EnhancedExtractionOrchestrator(config=config)
    
    # Test chunk
    test_chunk = """
    Gen Butler testified before Congress about the Business Plot. 
    He accused the bankers of orchestrating a fascist coup attempt.
    They denied everything, calling it a publicity stunt.
    The committee investigated but took no action.
    """
    
    test_context = {
        'source_file': '/media/bob/RAID11/DataShare/demo_corpus/butler/testimony.pdf',
        'title': 'Butler Congressional Testimony',
        'domain': 'political_history'
    }
    
    print("\nProcessing test chunk...")
    result = orchestrator.process_chunk(test_chunk, test_context)
    
    print(f"\nResults:")
    print(f"  Chunk ID: {result['chunk_id']}")
    print(f"  Source (shortened): {result['context'].get('source_file')}")
    print(f"  Entities: {len(result['entities'])}")
    for e in result['entities'][:3]:
        canonical = e.get('canonical_name', e.get('name'))
        matched = "✓ matched" if e.get('matched_existing') else "new"
        print(f"    - {e.get('name')} → {canonical} ({matched})")
    
    print(f"  Claims: {len(result['claims'])}")
    for c in result['claims'][:3]:
        print(f"    - [{c.get('claim_type')}] {c.get('claim_text', '')[:60]}...")
    
    print(f"  Coreference resolved: {result['coreference_metadata'].get('resolved', False)}")
    
    print(f"\nStats: {orchestrator.get_stats()}")
    
    print("\n✅ Test complete")


def process_file(input_path: str, output_path: str, dual_mode: bool = False):
    """Process a JSONL file through the pipeline."""
    
    config = ExtractionConfig()
    if dual_mode:
        config.secondary_ollama_url = "http://localhost:11435"
    
    orchestrator = DualOllamaOrchestrator(config=config) if dual_mode else EnhancedExtractionOrchestrator(config=config)
    
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            chunk_text = data.get('text', '')
            context = data.get('context', {})
            
            result = orchestrator.process_chunk(chunk_text, context)
            results.append(result)
    
    orchestrator.write_jsonl(results, output_path)
    print(f"Stats: {orchestrator.get_stats()}")


if __name__ == "__main__":
    main()
