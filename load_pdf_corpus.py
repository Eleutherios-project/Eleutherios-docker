#!/usr/bin/env python3
"""
PDF Corpus Loader for AegisTrustNet

For PDFs that haven't been pre-processed into JSONL format.
This provides basic extraction and loading capabilities.

For best results, use the JSONL loader with pre-processed data.

Usage:
    python load_pdf_corpus.py --pdf /path/to/file.pdf --category academic
    python load_pdf_corpus.py --source /path/to/pdfs --category research
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not installed. Install with: pip install PyPDF2")

from load_jsonl_corpus import Neo4jLoader, LoadStats, EntityExtractor, ClaimExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extract text and metadata from PDF files"""
    
    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract text from PDF"""
        if not PDF_AVAILABLE:
            raise RuntimeError("PyPDF2 not installed")
        
        text = []
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {pdf_path.name}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path.name}: {e}")
            return ""
        
        return "\n\n".join(text)
    
    @staticmethod
    def extract_metadata(pdf_path: Path) -> Dict:
        """Extract PDF metadata"""
        metadata = {
            'source_file': str(pdf_path),
            'title': pdf_path.stem.replace('_', ' ').replace('-', ' '),
            'category': 'unknown',
            'domain': 'unknown',
            'confidence_score': 0.5,
        }
        
        if not PDF_AVAILABLE:
            return metadata
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                info = pdf.metadata
                
                if info:
                    if info.get('/Title'):
                        metadata['title'] = info['/Title']
                    if info.get('/Author'):
                        metadata['author'] = info['/Author']
                    if info.get('/Subject'):
                        metadata['subject'] = info['/Subject']
        except Exception as e:
            logger.warning(f"Could not extract metadata from {pdf_path.name}: {e}")
        
        return metadata
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        sequence_num = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'sequence_num': sequence_num,
                    'length': current_length
                })
                
                # Keep overlap sentences
                overlap_text_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_text_length + len(s) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_text_length += len(s)
                
                current_chunk = overlap_sentences
                current_length = overlap_text_length
                sequence_num += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'sequence_num': sequence_num,
                'length': current_length
            })
        
        return chunks


def process_pdf(loader: Neo4jLoader, pdf_path: Path, category: str = 'unknown'):
    """Process a single PDF file"""
    logger.info(f"Processing {pdf_path.name}...")
    
    if not PDF_AVAILABLE:
        logger.error("PyPDF2 not installed. Cannot process PDFs.")
        return
    
    # Extract text and metadata
    text = PDFProcessor.extract_text(pdf_path)
    if not text or len(text) < 100:
        logger.warning(f"Insufficient text extracted from {pdf_path.name}")
        return
    
    metadata = PDFProcessor.extract_metadata(pdf_path)
    metadata['category'] = category
    
    # Infer domain from filename or content
    filename_lower = pdf_path.name.lower()
    if any(keyword in filename_lower for keyword in ['academic', 'journal', 'paper']):
        metadata['domain'] = 'academic_scholarly'
        metadata['confidence_score'] = 0.7
    elif any(keyword in filename_lower for keyword in ['spiritual', 'consciousness', 'meditation']):
        metadata['domain'] = 'contemplative_spiritual'
        metadata['confidence_score'] = 0.6
    elif any(keyword in filename_lower for keyword in ['technical', 'manual', 'guide']):
        metadata['domain'] = 'technical_implementation'
        metadata['confidence_score'] = 0.7
    
    # Create document node
    doc_id = loader.load_document(metadata)
    loader.stats.documents_processed += 1
    
    # Chunk the text
    chunks = PDFProcessor.chunk_text(text)
    logger.info(f"  Split into {len(chunks)} chunks")
    
    # Process each chunk
    for chunk_data in chunks:
        # Create JSONL-like record
        record = {
            'text': chunk_data['text'],
            'sequence_num': chunk_data['sequence_num'],
            'context_metadata': metadata
        }
        
        try:
            loader.process_jsonl_record(record)
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_data['sequence_num']}: {e}")
            loader.stats.errors += 1
    
    logger.info(f"Completed {pdf_path.name}")


def process_directory(loader: Neo4jLoader, directory: Path, category: str = 'unknown',
                     pattern: str = '*.pdf', max_files: Optional[int] = None):
    """Process all PDF files in a directory"""
    files = sorted(directory.glob(pattern))
    
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Found {len(files)} PDF files in {directory}")
    
    for i, pdf_path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing {pdf_path.name}")
        process_pdf(loader, pdf_path, category)


def main():
    parser = argparse.ArgumentParser(
        description='Load PDF files directly into AegisTrustNet Neo4j database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load single PDF
  python load_pdf_corpus.py --pdf /path/to/document.pdf --category academic
  
  # Load directory of PDFs
  python load_pdf_corpus.py --source /path/to/pdfs --category research
  
  # Limit number of files (for testing)
  python load_pdf_corpus.py --source /path/to/pdfs --max-files 5

Note: For best performance, use load_jsonl_corpus.py with pre-processed JSONL data.
This PDF loader is provided for convenience but has simpler extraction capabilities.
        """
    )
    
    parser.add_argument('--pdf', type=str,
                       help='Path to single PDF file')
    parser.add_argument('--source', type=str,
                       help='Path to directory containing PDF files')
    parser.add_argument('--category', type=str, default='unknown',
                       help='Category tag (academic, research, technical, etc.)')
    parser.add_argument('--pattern', type=str, default='*.pdf',
                       help='File pattern to match (default: *.pdf)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum files to process (for testing)')
    parser.add_argument('--chunk-size', type=int, default=2000,
                       help='Chunk size in characters (default: 2000)')
    parser.add_argument('--overlap', type=int, default=200,
                       help='Overlap between chunks (default: 200)')
    
    # Neo4j connection
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--neo4j-password', type=str, default='aegistrusted',
                       help='Neo4j password (default: aegistrusted)')
    
    args = parser.parse_args()
    
    if not PDF_AVAILABLE:
        logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        return 1
    
    if not args.pdf and not args.source:
        parser.error("Must specify either --pdf or --source")
    
    # Initialize loader
    loader = Neo4jLoader(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password
    )
    
    try:
        # Create indexes
        logger.info("Creating/verifying indexes...")
        loader.create_indexes()
        
        # Load data
        if args.pdf:
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                logger.error(f"File not found: {pdf_path}")
                return 1
            process_pdf(loader, pdf_path, args.category)
        
        elif args.source:
            directory = Path(args.source)
            if not directory.exists():
                logger.error(f"Directory not found: {directory}")
                return 1
            process_directory(loader, directory, args.category, args.pattern, args.max_files)
        
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
