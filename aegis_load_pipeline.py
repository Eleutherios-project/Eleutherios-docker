#!/usr/bin/env python3
"""
AegisTrustNet End-to-End Data Pipeline

Integrates:
1. PDF categorization
2. Training data generation (JSONL creation)
3. Knowledge graph loading

Usage:
    # Simple: Just point to PDFs
    python3 aegis_load_pipeline.py --pdfs /path/to/pdfs
    
    # With categories
    python3 aegis_load_pipeline.py --pdfs /path/to/pdfs --categories academic research
    
    # Skip training data generation (if you already have JSONL)
    python3 aegis_load_pipeline.py --jsonl /path/to/jsonl_dir --skip-generation
    
    # Full control
    python3 aegis_load_pipeline.py \
        --pdfs /path/to/pdfs \
        --output-dir ./processed_data \
        --training-script /path/to/your/training_pipeline.py \
        --load-graph
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AegisPipeline:
    """End-to-end pipeline for loading PDFs into AegisTrustNet"""
    
    # Standard category mappings
    CATEGORIES = {
        'academic': 'academic_scholarly',
        'research': 'research_investigative',
        'spiritual': 'contemplative_spiritual',
        'wisdom': 'practical_wisdom',
        'strategic': 'strategic_analytical',
        'technical': 'technical_implementation'
    }
    
    def __init__(self, 
                 pdf_dir: Optional[Path] = None,
                 jsonl_dir: Optional[Path] = None,
                 output_dir: Path = Path('./aegis_processed'),
                 training_script: Optional[Path] = None):
        
        self.pdf_dir = pdf_dir
        self.jsonl_dir = jsonl_dir
        self.output_dir = output_dir
        self.training_script = training_script
        
        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_output = self.output_dir / 'jsonl'
        self.jsonl_output.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def discover_pdfs(self, category_auto_detect: bool = True) -> dict:
        """Discover and optionally categorize PDFs"""
        if not self.pdf_dir:
            return {}
        
        pdf_files = list(self.pdf_dir.rglob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        if not category_auto_detect:
            return {'uncategorized': pdf_files}
        
        # Try to auto-categorize based on directory structure
        categorized = {}
        for pdf in pdf_files:
            # Check if PDF is in a recognized category folder
            category = None
            for part in pdf.parts:
                part_lower = part.lower()
                if 'academic' in part_lower or 'scholar' in part_lower:
                    category = 'academic_scholarly'
                elif 'research' in part_lower or 'investigat' in part_lower:
                    category = 'research_investigative'
                elif 'spiritual' in part_lower or 'contemplat' in part_lower:
                    category = 'contemplative_spiritual'
                elif 'wisdom' in part_lower or 'practical' in part_lower:
                    category = 'practical_wisdom'
                elif 'strategic' in part_lower or 'analytical' in part_lower:
                    category = 'strategic_analytical'
                elif 'technical' in part_lower or 'implement' in part_lower:
                    category = 'technical_implementation'
                
                if category:
                    break
            
            if not category:
                category = 'uncategorized'
            
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(pdf)
        
        logger.info("PDF Categorization:")
        for cat, files in categorized.items():
            logger.info(f"  {cat}: {len(files)} files")
        
        return categorized
    
    def generate_training_data(self, categorized_pdfs: dict) -> Path:
        """Generate JSONL training data from PDFs"""
        logger.info("=" * 70)
        logger.info("STEP 1: Generating Training Data")
        logger.info("=" * 70)
        
        if not self.training_script or not self.training_script.exists():
            logger.warning("No training script provided, using simple extraction")
            return self._simple_pdf_to_jsonl(categorized_pdfs)
        
        # Use user's training script
        logger.info(f"Using training script: {self.training_script}")
        
        # Call the training script for each category
        for category, pdf_files in categorized_pdfs.items():
            if not pdf_files:
                continue
            
            logger.info(f"Processing {len(pdf_files)} files in category: {category}")
            
            # Create temporary directory for this category
            cat_dir = self.output_dir / 'temp_pdfs' / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy PDFs to temp directory
            for pdf in pdf_files:
                shutil.copy2(pdf, cat_dir / pdf.name)
            
            # Run training script
            # Adjust this based on your actual training script's interface
            try:
                result = subprocess.run([
                    'python3', str(self.training_script),
                    '--input', str(cat_dir),
                    '--output', str(self.jsonl_output / category),
                    '--category', category
                ], capture_output=True, text=True, timeout=3600)
                
                if result.returncode != 0:
                    logger.error(f"Training script failed for {category}")
                    logger.error(result.stderr)
                else:
                    logger.info(f"✓ Generated JSONL for {category}")
            
            except subprocess.TimeoutExpired:
                logger.error(f"Training script timeout for {category}")
            except Exception as e:
                logger.error(f"Error running training script: {e}")
        
        return self.jsonl_output
    
    def _simple_pdf_to_jsonl(self, categorized_pdfs: dict) -> Path:
        """Simple PDF to JSONL conversion (fallback)"""
        logger.info("Using simple PDF extraction")
        
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 not installed. Cannot extract PDFs.")
            logger.error("Install with: pip install PyPDF2")
            sys.exit(1)
        
        for category, pdf_files in categorized_pdfs.items():
            if not pdf_files:
                continue
            
            cat_output = self.jsonl_output / category
            cat_output.mkdir(parents=True, exist_ok=True)
            
            output_file = cat_output / f"{category}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as out:
                for pdf_path in pdf_files:
                    try:
                        with open(pdf_path, 'rb') as pdf_file:
                            pdf = PyPDF2.PdfReader(pdf_file)
                            
                            text = []
                            for page in pdf.pages:
                                text.append(page.extract_text())
                            
                            full_text = '\n\n'.join(text)
                            
                            # Create JSONL record
                            record = {
                                'content': full_text,
                                'context_metadata': {
                                    'context_type': category,
                                    'domain': category,
                                    'confidence_score': 0.7,
                                    'source_file': str(pdf_path)
                                },
                                'source_file': str(pdf_path),
                                'extraction_method': 'simple_pdf'
                            }
                            
                            out.write(json.dumps(record) + '\n')
                    
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path.name}: {e}")
            
            logger.info(f"✓ Created {output_file}")
        
        return self.jsonl_output
    
    def load_to_neo4j(self, jsonl_dir: Path, categories: Optional[List[str]] = None):
        """Load JSONL data into Neo4j knowledge graph"""
        logger.info("=" * 70)
        logger.info("STEP 2: Loading into Knowledge Graph")
        logger.info("=" * 70)
        
        # Find load_jsonl_corpus.py
        loader_script = Path(__file__).parent / 'load_jsonl_corpus.py'
        if not loader_script.exists():
            logger.error(f"Loader script not found: {loader_script}")
            logger.error("Make sure load_jsonl_corpus.py is in the same directory")
            sys.exit(1)
        
        # Get all category directories
        if categories:
            cat_dirs = [jsonl_dir / cat for cat in categories if (jsonl_dir / cat).exists()]
        else:
            cat_dirs = [d for d in jsonl_dir.iterdir() if d.is_dir()]
        
        if not cat_dirs:
            # Try loading JSONL files directly from the directory
            jsonl_files = list(jsonl_dir.glob('*.jsonl'))
            if jsonl_files:
                logger.info(f"Found {len(jsonl_files)} JSONL files in {jsonl_dir}")
                result = subprocess.run([
                    'python3', str(loader_script),
                    '--source', str(jsonl_dir),
                    '--category', 'general'
                ])
                return result.returncode == 0
            else:
                logger.error(f"No JSONL files or category directories found in {jsonl_dir}")
                return False
        
        # Load each category
        total_success = True
        for cat_dir in cat_dirs:
            category = cat_dir.name
            logger.info(f"\nLoading category: {category}")
            
            result = subprocess.run([
                'python3', str(loader_script),
                '--source', str(cat_dir),
                '--category', category.split('_')[0]  # Use first part of category name
            ])
            
            if result.returncode != 0:
                logger.error(f"Failed to load {category}")
                total_success = False
            else:
                logger.info(f"✓ Loaded {category}")
        
        return total_success
    
    def run(self, 
            skip_generation: bool = False,
            skip_loading: bool = False,
            categories: Optional[List[str]] = None):
        """Run the complete pipeline"""
        
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║" + " " * 18 + "AEGISTRUSTNET DATA PIPELINE" + " " * 23 + "║")
        logger.info("╚" + "═" * 68 + "╝")
        logger.info("Starting pipeline execution!")
        
        # Determine what to do
        if self.jsonl_dir:
            # User provided JSONL directory directly
            jsonl_location = self.jsonl_dir
            skip_generation = True
        elif self.pdf_dir and not skip_generation:
            # Generate JSONL from PDFs
            categorized = self.discover_pdfs()
            jsonl_location = self.generate_training_data(categorized)
        else:
            logger.error("Must provide either --pdfs or --jsonl directory")
            return False
        
        # Load to Neo4j
        if not skip_loading:
            success = self.load_to_neo4j(jsonl_location, categories)
            
            if success:
                logger.info("")
                logger.info("╔" + "═" * 68 + "╗")
                logger.info("║" + " " * 25 + "PIPELINE COMPLETE!" + " " * 26 + "║")
                logger.info("╚" + "═" * 68 + "╝")
                logger.info("")
                logger.info("Next steps:")
                logger.info("  1. Open web UI: http://localhost:8082")
                logger.info("  2. Search for topics from your documents")
                logger.info("  3. Explore the knowledge graph!")
            else:
                logger.error("Pipeline completed with errors")
                return False
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='AegisTrustNet End-to-End Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple: Load PDFs from a directory
  python3 aegis_load_pipeline.py --pdfs /path/to/pdfs
  
  # With your training script
  python3 aegis_load_pipeline.py \
    --pdfs /path/to/pdfs \
    --training-script /path/to/training_pipeline4.py
  
  # Just load existing JSONL (skip PDF processing)
  python3 aegis_load_pipeline.py \
    --jsonl /path/to/jsonl_dir
  
  # Generate JSONL but don't load to Neo4j yet
  python3 aegis_load_pipeline.py \
    --pdfs /path/to/pdfs \
    --skip-loading
  
  # Load specific categories only
  python3 aegis_load_pipeline.py \
    --jsonl /path/to/jsonl_dir \
    --categories academic research
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pdfs', type=str,
                            help='Directory containing PDF files')
    input_group.add_argument('--jsonl', type=str,
                            help='Directory containing JSONL files (skip PDF processing)')
    
    # Processing options
    parser.add_argument('--output-dir', type=str, default='./aegis_processed',
                       help='Output directory for processed data (default: ./aegis_processed)')
    parser.add_argument('--training-script', type=str,
                       help='Path to your training data generation script')
    
    # Control options
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip JSONL generation (use with --jsonl)')
    parser.add_argument('--skip-loading', action='store_true',
                       help='Skip Neo4j loading (only generate JSONL)')
    parser.add_argument('--categories', nargs='+',
                       help='Specific categories to load (e.g., academic research)')
    
    # Neo4j options
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--neo4j-password', type=str, default='aegistrusted',
                       help='Neo4j password (default: aegistrusted)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = AegisPipeline(
        pdf_dir=Path(args.pdfs) if args.pdfs else None,
        jsonl_dir=Path(args.jsonl) if args.jsonl else None,
        output_dir=Path(args.output_dir),
        training_script=Path(args.training_script) if args.training_script else None
    )
    
    # Run pipeline
    success = pipeline.run(
        skip_generation=args.skip_generation,
        skip_loading=args.skip_loading,
        categories=args.categories
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
