#!/usr/bin/env python3
"""
Advanced JSONL Corpus Loader for AegisTrustNet with Enhanced Claim Extraction

Version: 2.0 - LLM-Enhanced
Date: November 3, 2025

Loads pre-processed training data from JSONL files into Neo4j with:
- LLM-based claim extraction (Ollama integration)
- Multiple extraction modes (regex, SpaCy, LLM small/large)
- Hierarchical source attribution
- Domain classification
- Enhanced entity and claim extraction

Usage:
    # Basic loading with LLM extraction
    python load_jsonl_corpus.py --jsonl file.jsonl --mode llm_large
    
    # With specific model
    python load_jsonl_corpus.py --jsonl file.jsonl --mode llm_large --model qwen2.5:72b
    
    # Fast regex mode for testing
    python load_jsonl_corpus.py --jsonl file.jsonl --mode none --max-records 100
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from neo4j import GraphDatabase
import re
import sys

# Import enhanced claim extractor
try:
    from enhanced_claim_extractor import ClaimExtractor, ExtractionConfig, ExtractionMode, OllamaClient
except ImportError:
    # If running from different directory, try to import from same dir as this script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from enhanced_claim_extractor import ClaimExtractor, ExtractionConfig, ExtractionMode, OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadStats:
    """Track loading statistics"""
    documents_processed: int = 0
    chunks_processed: int = 0
    entities_created: int = 0
    claims_created: int = 0
    relationships_created: int = 0
    errors: int = 0
    start_time: datetime = None
    
    # Claim type breakdown
    claims_primary: int = 0
    claims_secondary: int = 0
    claims_meta: int = 0
    claims_contextual: int = 0
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def duration(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    def summary(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    LOADING COMPLETE                           ║
╚══════════════════════════════════════════════════════════════╝

Documents Processed:    {self.documents_processed:,}
Chunks Processed:       {self.chunks_processed:,}
Entities Created:       {self.entities_created:,}
Claims Created:         {self.claims_created:,}
  ├─ PRIMARY:           {self.claims_primary:,}
  ├─ SECONDARY:         {self.claims_secondary:,}
  ├─ META:              {self.claims_meta:,}
  └─ CONTEXTUAL:        {self.claims_contextual:,}
Relationships Created:  {self.relationships_created:,}
Errors:                 {self.errors:,}
Duration:               {self.duration():.2f} seconds
Processing Rate:        {self.chunks_processed/self.duration():.2f} chunks/sec
Avg Claims/Chunk:       {self.claims_created/max(1,self.chunks_processed):.2f}
"""


class EntityExtractor:
    """Extract entities from text content"""
    
    # Enhanced entity patterns
    PERSON_PATTERNS = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
        r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. Jane Doe
        r'\bProf\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Prof. Robert Brown
        r'\b[A-Z]\. [A-Z][a-z]+\b',  # J. Smith
        r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # John Q. Public
    ]
    
    ORGANIZATION_PATTERNS = [
        r'\b(?:CDC|NIH|FDA|WHO|EPA|NASA|NSF|DARPA|IEEE|USMC|USAF)\b',
        r'\bCompanyA\b|\bCompanyB\b|\bCompanyC\b',
        r'\b[A-Z][a-z]+ (?:University|Institute|Laboratory|Corp|Corporation)\b',
        r'\bDepartment of [A-Z][a-z]+\b',
        r'\b(?:Marine Corps|Air Force|Navy|Army)\b',
    ]
    
    CONCEPT_KEYWORDS = [
        # Military/Strategy
        'OODA loop', 'Boyd cycle', 'doctrine', 'strategy', 'tactics',
        # Psychology/Neuroscience
        'resilience', 'stress', 'ptsd', 'depression', 'anxiety', 'trauma',
        'neuroscience', 'brain', 'cortex', 'neural', 'cognitive',
        # Archaeology/History
        'civilization', 'archaeology', 'ancient', 'artifact',
        # Consciousness/Philosophy
        'consciousness', 'awareness', 'quantum', 'reality',
        # Medical/Biological
        'vaccine', 'clinical trial', 'adverse event', 'immune', 'inflammation',
        'gene', 'protein', 'cell', 'molecule', 'receptor',
        # General research
        'study', 'research', 'theory', 'hypothesis', 'mechanism'
    ]
    
    @staticmethod
    def extract_entities(text: str, confidence_boost: float = 0.0) -> List[Dict[str, Any]]:
        """Extract entities from text with enhanced NLP"""
        entities = []
        
        # Extract authors from academic format
        author_match = re.search(r'Authors?\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if author_match:
            author_text = author_match.group(1)
            authors = re.findall(r'[A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+', author_text)
            for author in authors[:10]:
                entities.append({
                    'name': author.strip(),
                    'type': 'Person',
                    'confidence': 0.9 + confidence_boost
                })
        
        # Extract journal names
        journal_match = re.search(r'Journal\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if journal_match:
            journal = journal_match.group(1).strip()
            if len(journal) > 5:
                entities.append({
                    'name': journal,
                    'type': 'Organization',
                    'confidence': 0.9 + confidence_boost
                })
        
        # Extract people
        for pattern in EntityExtractor.PERSON_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group()
                # Skip false positives
                if name not in ['The University', 'New York', 'United States', 'Marine Corps']:
                    entities.append({
                        'name': name,
                        'type': 'Person',
                        'confidence': 0.7 + confidence_boost
                    })
        
        # Extract organizations
        for pattern in EntityExtractor.ORGANIZATION_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append({
                    'name': match.group(),
                    'type': 'Organization',
                    'confidence': 0.8 + confidence_boost
                })
        
        # Extract concepts
        text_lower = text.lower()
        for keyword in EntityExtractor.CONCEPT_KEYWORDS:
            if keyword.lower() in text_lower:
                # Find capitalized version
                matches = re.finditer(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'name': match.group(),
                        'type': 'Concept',
                        'confidence': 0.7 + confidence_boost
                    })
                    break  # Only add once
        
        # Extract title case phrases (potential concepts)
        concept_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b'
        for match in re.finditer(concept_pattern, text):
            phrase = match.group()
            if any(kw in phrase.lower() for kw in ['loop', 'cycle', 'theory', 'model', 'framework']):
                entities.append({
                    'name': phrase,
                    'type': 'Concept',
                    'confidence': 0.6 + confidence_boost
                })
        
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities


class Neo4jLoader:
    """Load JSONL data into Neo4j knowledge graph"""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "aegistrusted",
                 claim_extractor: Optional[ClaimExtractor] = None):
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.stats = LoadStats()
        self.claim_extractor = claim_extractor or ClaimExtractor()
        
        # Test connection
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        with self.driver.session() as session:
            indexes = [
                "CREATE INDEX doc_source IF NOT EXISTS FOR (d:Document) ON (d.source_file)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX claim_text IF NOT EXISTS FOR (c:Claim) ON (c.claim_text)",
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (ch:Chunk) ON (ch.chunk_id)",
            ]
            
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
        
        logger.info("Indexes created/verified")
    
    def load_document(self, metadata: Dict[str, Any]) -> str:
        """Create or merge Document node"""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (d:Document {source_file: $source_file})
                ON CREATE SET
                    d.title = $title,
                    d.author = $author,
                    d.domain = $domain,
                    d.created_at = datetime()
                ON MATCH SET
                    d.last_updated = datetime()
                RETURN d.source_file as source_file
            """, {
                'source_file': metadata.get('source_file', 'unknown'),
                'title': metadata.get('title', 'Untitled'),
                'author': metadata.get('author', 'Unknown'),
                'domain': metadata.get('domain', 'unknown')
            })
            
            record = result.single()
            return record['source_file'] if record else metadata.get('source_file', 'unknown')
    
    def load_chunk(self, chunk_data: Dict[str, Any], doc_source: str) -> str:
        """Create Chunk node and link to Document"""
        chunk_id = f"{doc_source}_{chunk_data.get('sequence_num', 0)}"
        
        with self.driver.session() as session:
            session.run("""
                CREATE (ch:Chunk {
                    chunk_id: $chunk_id,
                    text: $text,
                    sequence_num: $sequence_num,
                    confidence_score: $confidence_score,
                    domain: $domain,
                    created_at: datetime()
                })
                WITH ch
                MATCH (d:Document {source_file: $doc_source})
                MERGE (d)-[:CONTAINS_CHUNK]->(ch)
            """, {
                'chunk_id': chunk_id,
                'text': chunk_data['text'],
                'sequence_num': chunk_data.get('sequence_num', 0),
                'confidence_score': chunk_data.get('confidence_score', 0.5),
                'domain': chunk_data.get('domain', 'unknown'),
                'doc_source': doc_source
            })
            
            self.stats.chunks_processed += 1
            self.stats.relationships_created += 1
            
            return chunk_id
    
    def load_entity(self, entity: Dict[str, Any], chunk_id: str):
        """Create Entity node and link to Chunk"""
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET
                    e.confidence = $confidence,
                    e.created_at = datetime()
                ON MATCH SET
                    e.confidence = CASE 
                        WHEN $confidence > e.confidence THEN $confidence 
                        ELSE e.confidence 
                    END
                WITH e
                MATCH (ch:Chunk {chunk_id: $chunk_id})
                MERGE (ch)-[:MENTIONS]->(e)
            """, {
                'name': entity['name'],
                'type': entity['type'],
                'confidence': entity.get('confidence', 0.5),
                'chunk_id': chunk_id
            })
            
            self.stats.entities_created += 1
            self.stats.relationships_created += 1
    
    def load_claim(self, claim: Dict[str, Any], chunk_id: str):
        """Create Claim node with enhanced attributes and link to Chunk"""
        claim_type = claim.get('claim_type', 'CONTEXTUAL')
        
        with self.driver.session() as session:
            session.run("""
                CREATE (c:Claim {
                    claim_text: $text,
                    claim_type: $claim_type,
                    trust_score: $confidence,
                    domain: $domain,
                    source_file: $source_file,
                    extraction_method: $extraction_method,
                    created_at: datetime()
                })
                SET c.subject = $subject,
                    c.relation = $relation,
                    c.object = $object,
                    c.significance = $significance
                WITH c
                MATCH (ch:Chunk {chunk_id: $chunk_id})
                MERGE (ch)-[:CONTAINS_CLAIM]->(c)
            """, {
                'text': claim.get('text', claim.get('full_text', '')),
                'claim_type': claim_type,
                'confidence': claim.get('confidence', 0.5),
                'domain': claim.get('domain', 'unknown'),
                'source_file': claim.get('source_file', ''),
                'extraction_method': claim.get('extraction_method', 'unknown'),
                'subject': claim.get('subject', ''),
                'relation': claim.get('relation', ''),
                'object': claim.get('object', ''),
                'significance': claim.get('significance', 0.5),
                'chunk_id': chunk_id
            })
            
            self.stats.claims_created += 1
            self.stats.relationships_created += 1
            
            # Track by type
            if claim_type == 'PRIMARY':
                self.stats.claims_primary += 1
            elif claim_type == 'SECONDARY':
                self.stats.claims_secondary += 1
            elif claim_type == 'META':
                self.stats.claims_meta += 1
            else:
                self.stats.claims_contextual += 1
    
    def process_jsonl_record(self, record: Dict[str, Any]):
        """Process a single JSONL record with enhanced extraction"""
        try:
            # Extract metadata
            metadata = record.get('context_metadata', {})
            source_file = metadata.get('source_file') or record.get('source_file', 'unknown')
            metadata['source_file'] = source_file
            
            # Load document
            doc_id = self.load_document(metadata)
            self.stats.documents_processed += 1
            
            # Get text content
            text_content = record.get('content') or record.get('text', '')
            
            # Load chunk
            chunk_data = {
                'text': text_content,
                'sequence_num': record.get('sequence_num', 0),
                'confidence_score': metadata.get('confidence_score', 0.5),
                'domain': metadata.get('domain', 'unknown'),
            }
            chunk_id = self.load_chunk(chunk_data, doc_id)
            
            # Extract and load entities
            entities = EntityExtractor.extract_entities(
                text_content,
                confidence_boost=metadata.get('confidence_score', 0.0) * 0.2
            )
            
            for entity in entities[:20]:
                self.load_entity(entity, chunk_id)
            
            # Extract and load claims using configured extractor
            claims = self.claim_extractor.extract_claims(text_content, metadata)
            for claim in claims:
                self.load_claim(claim, chunk_id)
            
            # Progress logging
            if self.stats.chunks_processed % 10 == 0:
                logger.info(f"Processed {self.stats.chunks_processed} chunks, "
                          f"{self.stats.entities_created} entities, "
                          f"{self.stats.claims_created} claims "
                          f"({self.stats.chunks_processed/self.stats.duration():.2f} chunks/sec)")
        
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            self.stats.errors += 1


def load_jsonl_file(loader: Neo4jLoader, filepath: Path, max_records: Optional[int] = None):
    """Load a single JSONL file"""
    logger.info(f"Loading {filepath.name}...")
    
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if max_records and count >= max_records:
                break
            
            try:
                record = json.loads(line.strip())
                loader.process_jsonl_record(record)
                count += 1
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {filepath.name}: {e}")
                loader.stats.errors += 1
            except Exception as e:
                logger.error(f"Error processing line in {filepath.name}: {e}")
                loader.stats.errors += 1
    
    logger.info(f"Completed {filepath.name}: {count} records processed")


def load_directory(loader: Neo4jLoader, directory: Path, pattern: str = "*.jsonl", 
                   max_files: Optional[int] = None):
    """Load all JSONL files from a directory"""
    files = sorted(directory.glob(pattern))
    
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
    
    for i, filepath in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing {filepath.name}")
        load_jsonl_file(loader, filepath)


def main():
    parser = argparse.ArgumentParser(
        description='Load JSONL training data into AegisTrustNet Neo4j database with LLM-enhanced claim extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use LLM extraction (70B model)
  python load_jsonl_corpus.py --jsonl file.jsonl --mode llm_large
  
  # Use specific model
  python load_jsonl_corpus.py --jsonl file.jsonl --mode llm_large --model qwen2.5:72b
  
  # Fast regex mode for testing
  python load_jsonl_corpus.py --jsonl file.jsonl --mode none --max-records 100
  
  # Load directory with LLM extraction
  python load_jsonl_corpus.py --source /path/to/jsonl_dir --mode llm_small
        """
    )
    
    # Input options
    parser.add_argument('--jsonl', type=str, help='Path to single JSONL file')
    parser.add_argument('--source', type=str, help='Path to directory containing JSONL files')
    parser.add_argument('--pattern', type=str, default='*.jsonl', help='File pattern (default: *.jsonl)')
    parser.add_argument('--max-records', type=int, help='Max records per file (for testing)')
    parser.add_argument('--max-files', type=int, help='Max files to process (for testing)')
    
    # Extraction mode options
    parser.add_argument('--mode', type=str, 
                       choices=['none', 'spacy', 'llm_small', 'llm_large'],
                       default='llm_large',
                       help='Extraction mode (default: llm_large)')
    parser.add_argument('--model', type=str, help='Specific Ollama model name (optional)')
    parser.add_argument('--system-prompt', type=str, help='Custom system prompt file (optional)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence for claims (default: 0.6)')
    parser.add_argument('--timeout', type=int, default=120,
                       help='LLM timeout in seconds (default: 120)')
    
    # Neo4j connection
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--neo4j-password', type=str, default='aegistrusted',
                       help='Neo4j password (default: aegistrusted)')
    
    args = parser.parse_args()
    
    if not args.jsonl and not args.source:
        parser.error("Must specify either --jsonl or --source")
    
    # Load custom system prompt if provided
    system_prompt = None
    if args.system_prompt:
        try:
            with open(args.system_prompt, 'r') as f:
                system_prompt = f.read()
            logger.info(f"Loaded custom system prompt from {args.system_prompt}")
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}")
    
    # Check Ollama availability if using LLM mode
    if args.mode in ['llm_small', 'llm_large']:
        ollama = OllamaClient()
        if not ollama.is_available():
            logger.error("Ollama is not available!")
            logger.error("Please start Ollama: ollama serve")
            logger.error("Or use --mode none for regex extraction")
            return 1
        
        models = ollama.list_models()
        logger.info(f"Ollama is running with {len(models)} models available")
        if args.model:
            model_names = [m['name'] for m in models]
            if args.model not in model_names:
                logger.warning(f"Model '{args.model}' not found. Available: {', '.join(model_names)}")
    
    # Initialize claim extractor
    extraction_config = ExtractionConfig(
        mode=ExtractionMode(args.mode),
        model_name=args.model,
        system_prompt=system_prompt,
        confidence_threshold=args.confidence_threshold,
        timeout_seconds=args.timeout
    )
    
    claim_extractor = ClaimExtractor(extraction_config)
    logger.info(f"Claim extraction mode: {args.mode}")
    if args.mode in ['llm_small', 'llm_large']:
        logger.info(f"Using model: {claim_extractor.config.model_name}")
    
    # Initialize loader
    loader = Neo4jLoader(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        claim_extractor=claim_extractor
    )
    
    try:
        # Create indexes
        logger.info("Creating/verifying indexes...")
        loader.create_indexes()
        
        # Load data
        if args.jsonl:
            filepath = Path(args.jsonl)
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return 1
            load_jsonl_file(loader, filepath, max_records=args.max_records)
        
        elif args.source:
            directory = Path(args.source)
            if not directory.exists():
                logger.error(f"Directory not found: {directory}")
                return 1
            load_directory(loader, directory, pattern=args.pattern, max_files=args.max_files)
        
        # Print summary
        print(loader.stats.summary())
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        print(loader.stats.summary())
        return 1
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        loader.close()


if __name__ == '__main__':
    exit(main())
