#!/usr/bin/env python3
"""
AegisTrustNet - PDF Corpus Loader
Loads PDFs from a directory, extracts claims and entities, and populates Neo4j

Usage:
    python load_corpus.py --source /path/to/pdfs --category research
    python load_corpus.py --source /path/to/pdfs --batch-size 10 --skip-existing
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from neo4j import GraphDatabase
from dotenv import load_dotenv

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")
    PDF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('corpus-loader')

# Load environment
load_dotenv()

# Neo4j connection
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")


class CorpusLoader:
    """Load PDF documents into AegisTrustNet knowledge graph"""
    
    def __init__(self, category: str = "general", batch_size: int = 10):
        self.category = category
        self.batch_size = batch_size
        self.stats = {
            "pdfs_processed": 0,
            "pdfs_failed": 0,
            "entities_created": 0,
            "claims_created": 0,
            "documents_created": 0
        }
        
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info(f"✅ Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from a PDF file"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not available")
            return None
        
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction - looks for capitalized phrases
        In production, use spaCy or similar NLP library
        """
        entities = set()
        
        # Find capitalized words/phrases (crude but works)
        # Matches sequences of capitalized words
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, text)
        
        # Filter out common words and short matches
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'They', 
                     'It', 'He', 'She', 'We', 'You', 'I', 'A', 'An'}
        
        for match in matches:
            if len(match) > 3 and match not in stop_words:
                entities.add(match)
        
        # Limit to reasonable number
        return list(entities)[:20]
    
    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential claims from text
        Uses simple heuristics - in production use proper NLP
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Claim indicators
        indicators = [
            'shows', 'demonstrates', 'proves', 'indicates', 'suggests',
            'reveals', 'confirms', 'establishes', 'argues', 'claims',
            'proposes', 'asserts', 'maintains', 'contends', 'states',
            'evidence', 'research', 'study', 'findings', 'analysis'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short or very long sentences
            if len(sentence) < 20 or len(sentence) > 500:
                continue
            
            # Check if sentence contains claim indicators
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in indicators):
                # Extract entities mentioned in this claim
                entities = self.extract_entities(sentence)
                
                claims.append({
                    'text': sentence,
                    'entities': entities,
                    'confidence': 0.6  # Base confidence
                })
        
        # Limit number of claims per document
        return claims[:30]
    
    def load_document(self, pdf_path: str, skip_existing: bool = True) -> bool:
        """Load a single PDF document into Neo4j"""
        
        filename = os.path.basename(pdf_path)
        doc_id = f"doc-{abs(hash(pdf_path)) % (10 ** 10)}"
        
        try:
            with self.driver.session() as session:
                # Check if already processed
                if skip_existing:
                    result = session.run(
                        "MATCH (d:Document {path: $path}) RETURN d.id",
                        {"path": pdf_path}
                    )
                    if result.single():
                        logger.info(f"⏭️  Skipping (already loaded): {filename}")
                        return True
                
                # Extract text
                logger.info(f"📄 Processing: {filename}")
                text = self.extract_text_from_pdf(pdf_path)
                
                if not text or len(text) < 100:
                    logger.warning(f"⚠️  Insufficient text extracted from {filename}")
                    self.stats['pdfs_failed'] += 1
                    return False
                
                # Extract entities and claims
                entities = self.extract_entities(text)
                claims = self.extract_claims(text)
                
                logger.info(f"   Found {len(entities)} entities, {len(claims)} claims")
                
                # Create document node
                session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.path = $path,
                        d.filename = $filename,
                        d.category = $category,
                        d.loaded_date = datetime(),
                        d.text_length = $text_length,
                        d.num_claims = $num_claims,
                        d.num_entities = $num_entities
                """, {
                    'id': doc_id,
                    'path': pdf_path,
                    'filename': filename,
                    'category': self.category,
                    'text_length': len(text),
                    'num_claims': len(claims),
                    'num_entities': len(entities)
                })
                
                self.stats['documents_created'] += 1
                
                # Create entities
                for entity_name in entities:
                    entity_id = f"entity-{abs(hash(entity_name)) % (10 ** 10)}"
                    
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET 
                            e.id = $id,
                            e.type = 'EXTRACTED',
                            e.category = $category,
                            e.trust_score = 0.5,
                            e.frequency = 1
                        ON MATCH SET
                            e.frequency = e.frequency + 1
                        
                        WITH e
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[r:MENTIONS]->(e)
                        ON CREATE SET r.count = 1
                        ON MATCH SET r.count = r.count + 1
                    """, {
                        'name': entity_name,
                        'id': entity_id,
                        'category': self.category,
                        'doc_id': doc_id
                    })
                    
                    self.stats['entities_created'] += 1
                
                # Create claims
                for claim_data in claims:
                    claim_id = f"claim-{abs(hash(claim_data['text'])) % (10 ** 10)}"
                    
                    session.run("""
                        MERGE (c:Claim {id: $id})
                        ON CREATE SET
                            c.claim_text = $text,
                            c.category = $category,
                            c.trust_score = $confidence,
                            c.source = 'extracted'
                        
                        WITH c
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[r:CONTAINS]->(c)
                    """, {
                        'id': claim_id,
                        'text': claim_data['text'],
                        'category': self.category,
                        'confidence': claim_data['confidence'],
                        'doc_id': doc_id
                    })
                    
                    # Link claim to entities
                    for entity_name in claim_data['entities']:
                        session.run("""
                            MATCH (c:Claim {id: $claim_id})
                            MATCH (e:Entity {name: $entity_name})
                            MERGE (c)-[r:MENTIONS]->(e)
                            ON CREATE SET r.weight = 1
                            ON MATCH SET r.weight = r.weight + 1
                        """, {
                            'claim_id': claim_id,
                            'entity_name': entity_name
                        })
                    
                    self.stats['claims_created'] += 1
                
                self.stats['pdfs_processed'] += 1
                logger.info(f"✅ Loaded: {filename}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error loading {filename}: {e}")
            self.stats['pdfs_failed'] += 1
            return False
    
    def load_directory(self, source_dir: str, skip_existing: bool = True, 
                      recursive: bool = True) -> Dict[str, int]:
        """Load all PDFs from a directory"""
        
        source_path = Path(source_dir)
        
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return self.stats
        
        # Find all PDFs
        if recursive:
            pdf_files = list(source_path.rglob("*.pdf")) + list(source_path.rglob("*.PDF"))
        else:
            pdf_files = list(source_path.glob("*.pdf")) + list(source_path.glob("*.PDF"))
        
        total_pdfs = len(pdf_files)
        logger.info(f"📚 Found {total_pdfs} PDF files in {source_dir}")
        
        if total_pdfs == 0:
            logger.warning("No PDF files found!")
            return self.stats
        
        # Process in batches
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{total_pdfs}] Processing batch...")
            self.load_document(str(pdf_path), skip_existing=skip_existing)
            
            # Progress update every 10 files
            if i % 10 == 0:
                self.print_stats()
        
        return self.stats
    
    def print_stats(self):
        """Print loading statistics"""
        print("\n" + "="*60)
        print("📊 Loading Statistics")
        print("="*60)
        print(f"PDFs Processed:     {self.stats['pdfs_processed']}")
        print(f"PDFs Failed:        {self.stats['pdfs_failed']}")
        print(f"Documents Created:  {self.stats['documents_created']}")
        print(f"Entities Created:   {self.stats['entities_created']}")
        print(f"Claims Created:     {self.stats['claims_created']}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Load PDF corpus into AegisTrustNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all PDFs from a directory
  python load_corpus.py --source /path/to/pdfs --category research
  
  # Load with specific settings
  python load_corpus.py --source /path/to/pdfs --category academic --batch-size 20
  
  # Skip already loaded files
  python load_corpus.py --source /path/to/pdfs --skip-existing
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        required=True,
        help='Source directory containing PDF files'
    )
    
    parser.add_argument(
        '--category', '-c',
        default='general',
        help='Category tag for these documents (default: general)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=10,
        help='Number of PDFs to process in each batch (default: 10)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip PDFs that are already loaded'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not PDF_AVAILABLE:
        print("\n❌ PyPDF2 is required but not installed.")
        print("Install it with: pip install PyPDF2")
        return 1
    
    print("\n" + "="*60)
    print("🚀 AegisTrustNet - PDF Corpus Loader")
    print("="*60)
    print(f"Source:     {args.source}")
    print(f"Category:   {args.category}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Skip Existing: {args.skip_existing}")
    print("="*60 + "\n")
    
    # Create loader
    loader = CorpusLoader(
        category=args.category,
        batch_size=args.batch_size
    )
    
    try:
        # Load corpus
        stats = loader.load_directory(
            args.source,
            skip_existing=args.skip_existing,
            recursive=not args.no_recursive
        )
        
        # Final stats
        print("\n" + "="*60)
        print("✅ Loading Complete!")
        print("="*60)
        loader.print_stats()
        
        if stats['pdfs_processed'] > 0:
            print("Next steps:")
            print("  1. Restart your API server if it's running")
            print("  2. Open http://localhost:8001/")
            print("  3. Search for entities or topics from your PDFs")
            print("  4. Try the Graph View to see relationships")
            print()
            return 0
        else:
            print("⚠️  No PDFs were successfully processed")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Loading interrupted by user")
        loader.print_stats()
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        loader.close()


if __name__ == "__main__":
    sys.exit(main())
