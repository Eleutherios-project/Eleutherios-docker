"""
Aegis Insight - Extraction Orchestrator v2
Coordinates all extractors + embedding generation in one pass
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import all extractors
from aegis_entity_extractor import EntityExtractor
from aegis_claim_extractor import ClaimExtractor
from aegis_temporal_extractor import TemporalExtractor
from aegis_geographic_extractor import GeographicExtractor
from aegis_citation_extractor import CitationExtractor
from aegis_emotion_extractor import AegisEmotionExtractor as EmotionExtractor
from aegis_authority_domain_analyzer import AuthorityDomainAnalyzer

# Optional: Coreference resolution for better entity linking
try:
    from aegis_coreference_resolver import CoreferenceResolver
    HAS_COREFERENCE = True
except ImportError:
    HAS_COREFERENCE = False
    CoreferenceResolver = None




# ============================================================================
# Helper: Extract dates from document section headers
# ============================================================================

def extract_section_date(text: str):
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
    
    return None


class ExtractionOrchestrator:
    """
    Orchestrates all extraction processes + embedding generation
    Features:
    - Coordinates 7 specialized extractors
    - Automatic embedding generation for claims
    - Checkpoint/resume capability
    - Progress tracking
    - Error handling with retries
    - Dual GPU support (automatic detection)
    """
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "mistral-nemo:12b",
                 checkpoint_dir: str = "./checkpoints/mvp",
                 batch_size: int = 30,
                 checkpoint_every: int = 100,
                 enable_checkpointing: bool = True,
                 resume_from_checkpoint: bool = True,
                 max_retries: int = 3,
                 enable_embeddings: bool = True,
                 postgres_config: Dict = None,
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.enable_checkpointing = enable_checkpointing
        self.resume_from_checkpoint = resume_from_checkpoint
        self.max_retries = max_retries
        self.enable_embeddings = enable_embeddings
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractors
        self.logger.info("Initializing extractors...")
        self.entity_extractor = EntityExtractor(ollama_url, model, logger)
        self.claim_extractor = ClaimExtractor(ollama_url, model, logger)
        self.temporal_extractor = TemporalExtractor(ollama_url, model, logger)
        self.geographic_extractor = GeographicExtractor(ollama_url, model, logger)
        self.citation_extractor = CitationExtractor(ollama_url, model, logger)
        self.emotion_extractor = EmotionExtractor(ollama_url, model, logger)
        self.authority_analyzer = AuthorityDomainAnalyzer(ollama_url, model, logger)
        
        # Initialize embedding model if enabled
        self.embedding_model = None
        self.pg_conn = None
        
        if enable_embeddings:
            try:
                self.logger.info("Initializing embedding model...")
                from sentence_transformers import SentenceTransformer
                import psycopg2
                
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                self.logger.info(f"✓ Embedding model loaded (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
                
                # PostgreSQL connection
                pg_config = postgres_config or {
                    'host': 'localhost',
                    'database': 'aegis_insight',
                    'user': 'aegis',
                    'password': 'aegis_trusted_2025'
                }
                
                self.pg_conn = psycopg2.connect(**pg_config)
                self._init_embedding_table()
                self.logger.info("✓ PostgreSQL connected")
                
            except ImportError as e:
                self.logger.warning(f"Embedding dependencies not available: {e}")
                self.logger.warning("   Install with: pip install sentence-transformers psycopg2-binary")
                self.enable_embeddings = False
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize embeddings: {e}")
                self.enable_embeddings = False
        
        # Stats tracking
        self.stats = {
            'chunks_processed': 0,
            'chunks_failed': 0,
            'entities_extracted': 0,
            'claims_extracted': 0,
            'claims_with_temporal': 0,
            'claims_with_geographic': 0,
            'claims_with_citations': 0,
            'claims_with_emotional': 0,
            'claims_with_soft_date': 0,
            'coreferences_resolved': 0,
            'claims_with_authority': 0,
            'embeddings_generated': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        
        self.logger.info("✓ Orchestrator initialized")
    
    def _init_embedding_table(self):
        """Initialize PostgreSQL table for embeddings"""
        
        with self.pg_conn.cursor() as cur:
            # Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS claim_embeddings (
                    claim_id TEXT PRIMARY KEY,
                    claim_text TEXT NOT NULL,
                    embedding vector(384),
                    confidence FLOAT,
                    claim_type TEXT,
                    source_file TEXT,
                    chunk_id TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS claim_embeddings_idx 
                ON claim_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            self.pg_conn.commit()
    
    def _generate_embeddings(self, claims: List[Dict], chunk_id: str):
        """
        Generate and store embeddings for claims
        Called inline during extraction - no separate pass needed
        """
        
        if not self.enable_embeddings or not claims or not self.embedding_model:
            return
        
        try:
            # Extract texts
            texts = [claim.get('claim_text', '') for claim in claims]
            
            # Generate embeddings (fast - ~0.01 sec/claim)
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Store in PostgreSQL with transaction handling
            try:
                with self.pg_conn.cursor() as cur:
                    for i, claim in enumerate(claims):
                        claim_id = claim.get('claim_id', f"{chunk_id}_claim_{i}")
                        
                        cur.execute("""
                            INSERT INTO claim_embeddings 
                                (claim_id, claim_text, embedding, confidence, claim_type, source_file, chunk_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (claim_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                confidence = EXCLUDED.confidence,
                                claim_type = EXCLUDED.claim_type
                        """, (
                            claim_id,
                            claim.get('claim_text', ''),
                            embeddings[i].tolist(),
                            float(claim.get('confidence', 0.5)),
                            claim.get('claim_type', 'CONTEXTUAL'),
                            claim.get('source_file', 'unknown'),
                            chunk_id
                        ))
                    
                    self.pg_conn.commit()
                
                self.stats['embeddings_generated'] += len(claims)
                
            except Exception as db_error:
                # Rollback on any database error
                self.pg_conn.rollback()
                self.logger.warning(f"Database error for chunk {chunk_id}: {db_error}")
            
        except Exception as e:
            self.logger.warning(f"Embedding generation failed for chunk {chunk_id}: {e}")
    
    def process_chunks(self, 
                      chunks: List[Dict],
                      job_id: str = None) -> Dict:
        """
        Process chunks through all extractors + generate embeddings
        
        Args:
            chunks: List of chunk dicts with text, chunk_id, context_metadata
            job_id: Optional job identifier for checkpointing
            
        Returns:
            Dict with extracted_data and stats
        """
        
        if not chunks:
            self.logger.warning("No chunks to process")
            return {'extracted_data': [], 'stats': self.stats}
        
        # Generate job ID if not provided
        if not job_id:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting extraction job: {job_id}")
        self.logger.info(f"Total chunks: {len(chunks)}")
        self.logger.info(f"Embeddings: {'ENABLED' if self.enable_embeddings else 'DISABLED'}")
        
        # Check for checkpoint
        start_idx = 0
        extracted_data = []
        
        if self.enable_checkpointing and self.resume_from_checkpoint:
            checkpoint = self._load_checkpoint(job_id)
            if checkpoint:
                start_idx = checkpoint['last_idx'] + 1
                extracted_data = checkpoint['data']
                self.stats = checkpoint['stats']
                self.logger.info(f"✓ Resumed from checkpoint at chunk {start_idx}")
        
        # Start timer
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Process chunks
        for idx in range(start_idx, len(chunks)):
            try:
                result = self._process_single_chunk(chunks[idx], idx, len(chunks))
                extracted_data.append(result)
                
                self.stats['chunks_processed'] += 1
                
                # Checkpoint
                if self.enable_checkpointing and (idx + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(job_id, idx, extracted_data)
                
            except KeyboardInterrupt:
                self.logger.warning("Interrupted by user")
                if self.enable_checkpointing:
                    self._save_checkpoint(job_id, idx - 1, extracted_data)
                raise
            except Exception as e:
                self.logger.error(f"Failed to process chunk {idx}: {e}")
                self.stats['chunks_failed'] += 1
                self.stats['errors'].append({
                    'chunk_idx': idx,
                    'error': str(e)
                })
        
        # End timer
        self.stats['end_time'] = datetime.now().isoformat()
        
        # Final checkpoint
        if self.enable_checkpointing:
            self._save_checkpoint(job_id, len(chunks) - 1, extracted_data)
        
        self.logger.info("Extraction complete!")
        self._print_stats()
        
        return {
            'job_id': job_id,
            'extracted_data': extracted_data,
            'stats': self.stats
        }
    
    def _process_single_chunk(self, chunk: Dict, idx: int, total: int) -> Dict:
        """Process a single chunk through all extractors + generate embeddings"""
        
        chunk_text = chunk['text']
        chunk_id = chunk['chunk_id']
        
        # Extract document/section date from chunk text (for soft date propagation)
        document_date = extract_section_date(chunk_text)
        
        # Run coreference resolution if available (resolves pronouns to entities)
        resolved_text = chunk_text
        if self.coreference_resolver:
            try:
                resolved_text, coref_meta = self.coreference_resolver.resolve(chunk_text)
                if coref_meta.get('resolved'):
                    self.stats['coreferences_resolved'] += coref_meta.get('resolution_count', 1)
            except Exception as e:
                self.logger.debug(f"Coreference resolution skipped: {e}")
                resolved_text = chunk_text
        
        # Build extraction context
        extraction_context = {
            'domain': chunk.get('context_metadata', {}).get('domain', 'general'),
            'title': chunk.get('context_metadata', {}).get('title', 'Untitled'),
            'source_file': chunk.get('context_metadata', {}).get('source_file', 'unknown'),
            'chunk_number': chunk.get('chunk_number', 0),
            'min_entity_confidence': 0.5,
            'min_claim_confidence': 0.5
        }
        
        result = {
            'chunk_id': chunk_id,
            'chunk_text': chunk_text,  # Preserve original text
            'sequence_num': chunk.get('sequence_num', idx),  # Preserve sequence
            'context': extraction_context,  # Preserve context for graph builder
            'entities': [],
            'claims': []
        }
        
        # Extract entities (from resolved text if coreference ran)
        entities = self._extract_with_retry(
            self.entity_extractor.extract,
            resolved_text,
            extraction_context,
            'entities'
        )
        if entities:
            result['entities'] = entities
            self.stats['entities_extracted'] += len(entities)
        else:
            result['entities'] = []
        
        # Extract claims (from resolved text if coreference ran)
        claims = self._extract_with_retry(
            self.claim_extractor.extract,
            resolved_text,
            extraction_context,
            'claims'
        )
        if claims:
            result['claims'] = claims
            self.stats['claims_extracted'] += len(claims)
        else:
            result['claims'] = []
        
        # For each claim, extract dimensional data
        if not claims:
            claims = []
        
        for claim in claims:
            claim_text = claim.get('claim_text', '')
            
            if not claim_text:
                continue
            
            # Temporal data
            temporal = self._extract_with_retry(
                self.temporal_extractor.extract,
                claim_text,
                extraction_context,
                'temporal'
            )
            if temporal:
                claim['temporal_data'] = temporal
                self.stats['claims_with_temporal'] += 1
                
                # If no hard dates found but we have document_date, add as soft date
                if document_date and not temporal.get('absolute_dates'):
                    if 'absolute_dates' not in claim['temporal_data']:
                        claim['temporal_data']['absolute_dates'] = []
                    claim['temporal_data']['absolute_dates'].append({
                        'date': document_date,
                        'confidence': 0.8,
                        'type': 'document_context',
                        'context': 'Date from document section header'
                    })
                    claim['temporal_data']['soft_date'] = {
                        'date': document_date,
                        'source': 'document_section',
                        'confidence': 0.8
                    }
                    self.stats['claims_with_soft_date'] += 1
            elif document_date:
                # No temporal data extracted, but we have document date
                claim['temporal_data'] = {
                    'absolute_dates': [{
                        'date': document_date,
                        'confidence': 0.8,
                        'type': 'document_context',
                        'context': 'Date from document section header'
                    }],
                    'relative_dates': [],
                    'temporal_markers': [],
                    'soft_date': {
                        'date': document_date,
                        'source': 'document_section',
                        'confidence': 0.8
                    }
                }
                self.stats['claims_with_soft_date'] += 1
            
            # Geographic data
            geographic = self._extract_with_retry(
                self.geographic_extractor.extract,
                claim_text,
                extraction_context,
                'geographic'
            )
            if geographic:
                claim['geographic_data'] = geographic
                self.stats['claims_with_geographic'] += 1
            
            # Citation data
            citation = self._extract_with_retry(
                self.citation_extractor.extract,
                claim_text,
                extraction_context,
                'citation'
            )
            if citation:
                claim['citation_data'] = citation
                self.stats['claims_with_citations'] += 1
            
            # Emotional content
            emotional = self._extract_with_retry(
                self.emotion_extractor.extract,
                claim_text,
                extraction_context,
                'emotional'
            )
            if emotional:
                claim['emotional_data'] = emotional
                self.stats['claims_with_emotional'] += 1
            
            # Authority domain analysis
            try:
                authority = self.authority_analyzer.analyze(claim_text, entities, extraction_context)
                if authority:
                    claim['authority_data'] = authority
                    self.stats['claims_with_authority'] += 1
            except Exception as e:
                self.logger.warning(f"Authority analysis failed: {e}")
        
        # Generate embeddings for claims (inline - no separate pass!)
        if claims:
            self._generate_embeddings(claims, chunk_id)
        
        # Progress update
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            # Calculate elapsed time safely
            if self.stats["start_time"] and isinstance(self.stats["start_time"], str):
                start_dt = datetime.fromisoformat(self.stats["start_time"])
                elapsed = time.time() - time.mktime(start_dt.timetuple())
            else:
                elapsed = 0
            
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - (idx + 1)) / rate if rate > 0 else 0
            
            self.logger.info(
                f"Progress: {idx + 1}/{total} chunks "
                f"({(idx+1)/total*100:.1f}%) | "
                f"Rate: {rate:.2f} chunks/sec | "
                f"ETA: {eta/60:.1f} min"
            )
        
        return result
    
    def _extract_with_retry(self, 
                           extractor_func,
                           text: str,
                           context: Dict,
                           extractor_name: str = 'unknown') -> Optional[Dict]:
        """
        Call an extractor function with retry logic
        Fixed to handle argument passing correctly
        """
        
        for attempt in range(self.max_retries):
            try:
                # Simple, direct call - let Python handle method binding
                result = extractor_func(text, context)
                return result
            except TypeError as e:
                # If we get a signature mismatch, log details and return None
                if "positional argument" in str(e):
                    self.logger.error(
                        f"{extractor_name} extraction failed: signature mismatch - {e}"
                    )
                    self.logger.error(f"This usually means the extractor needs updating")
                    return None
                else:
                    # Other TypeError, treat as regular error
                    if attempt == self.max_retries - 1:
                        self.logger.error(
                            f"{extractor_name} extraction failed after {self.max_retries} attempts: {e}"
                        )
                        return None
                    else:
                        self.logger.warning(
                            f"{extractor_name} extraction attempt {attempt + 1} failed, retrying..."
                        )
                        time.sleep(1)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        f"{extractor_name} extraction failed after {self.max_retries} attempts: {e}"
                    )
                    return None
                else:
                    self.logger.warning(
                        f"{extractor_name} extraction attempt {attempt + 1} failed, retrying..."
                    )
                    time.sleep(1)
        
        return None
    
    def _save_checkpoint(self, job_id: str, last_idx: int, extracted_data: List[Dict]):
        """Save checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / f"{job_id}_checkpoint.json"
        data_path = self.checkpoint_dir / f"{job_id}_data.jsonl"
        
        checkpoint = {
            'job_id': job_id,
            'last_idx': last_idx,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        with open(data_path, 'w') as f:
            for item in extracted_data:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"✓ Checkpoint saved at chunk {last_idx}")
    
    def _load_checkpoint(self, job_id: str) -> Optional[Dict]:
        """Load checkpoint if exists"""
        
        checkpoint_path = self.checkpoint_dir / f"{job_id}_checkpoint.json"
        data_path = self.checkpoint_dir / f"{job_id}_data.jsonl"
        
        if not checkpoint_path.exists() or not data_path.exists():
            return None
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        checkpoint['data'] = data
        return checkpoint
    
    def _print_stats(self):
        """Print extraction statistics"""
        
        self.logger.info("="*70)
        self.logger.info("EXTRACTION STATISTICS")
        self.logger.info("="*70)
        self.logger.info(f"Chunks processed: {self.stats['chunks_processed']}")
        self.logger.info(f"Chunks failed: {self.stats['chunks_failed']}")
        self.logger.info(f"Entities extracted: {self.stats['entities_extracted']}")
        self.logger.info(f"Claims extracted: {self.stats['claims_extracted']}")
        self.logger.info(f"  ├─ With temporal data: {self.stats['claims_with_temporal']}")
        self.logger.info(f"  ├─ With geographic data: {self.stats['claims_with_geographic']}")
        self.logger.info(f"  ├─ With citations: {self.stats['claims_with_citations']}")
        self.logger.info(f"  ├─ With emotional data: {self.stats['claims_with_emotional']}")
        self.logger.info(f"  ├─ With soft dates: {self.stats.get('claims_with_soft_date', 0)}")
        self.logger.info(f"  └─ Coreferences resolved: {self.stats.get('coreferences_resolved', 0)}")
        self.logger.info(f"  └─ With authority analysis: {self.stats['claims_with_authority']}")
        
        if self.enable_embeddings:
            self.logger.info(f"Embeddings generated: {self.stats['embeddings_generated']}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            duration = (end - start).total_seconds()
            rate = self.stats['chunks_processed'] / duration if duration > 0 else 0
            self.logger.info(f"Duration: {duration/60:.1f} minutes")
            self.logger.info(f"Rate: {rate:.2f} chunks/sec")
        
        self.logger.info("="*70)
    
    def close(self):
        """Close connections"""
        if self.pg_conn:
            self.pg_conn.close()
            self.logger.info("✓ PostgreSQL connection closed")


# Convenience function
def create_orchestrator(enable_embeddings: bool = True, **kwargs):
    """
    Create orchestrator with sensible defaults
    
    Args:
        enable_embeddings: Whether to generate embeddings inline (default: True)
        **kwargs: Additional orchestrator parameters
    """
    
    defaults = {
        'model': 'mistral-nemo:12b',
        'batch_size': 30,
        'checkpoint_every': 100,
        'enable_checkpointing': True,
        'max_retries': 3
    }
    
    defaults.update(kwargs)
    defaults['enable_embeddings'] = enable_embeddings
    
    return ExtractionOrchestrator(**defaults)
