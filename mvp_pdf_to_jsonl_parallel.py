#!/usr/bin/env python3
"""
MVP PDF to JSONL Converter - PARALLEL VERSION
Uses multiprocessing to convert multiple PDFs simultaneously
"""

import json
import pypdf
import re
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# OCR support for scanned PDFs
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR not available. Install pdf2image and pytesseract for scanned PDF support.")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract complete text from PDF with OCR fallback for scanned documents"""
    try:
        text_blocks = []
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text_blocks.append(text)
        
        full_text = '\n\n'.join(text_blocks)
        
        # Check text quality - garbage PDFs have mostly non-printable chars
        printable_ratio = sum(1 for c in full_text if c.isprintable() or c.isspace()) / max(len(full_text), 1)
        
        # Trigger OCR if too short OR too much garbage (< 50% printable)
        if (len(full_text.strip()) < 100 or printable_ratio < 0.5) and OCR_AVAILABLE:
            print(f"  Low quality text ({printable_ratio:.0%} printable), trying OCR: {Path(pdf_path).name}")
            full_text = extract_text_with_ocr(pdf_path)
        
        return full_text
    except Exception as e:
        print(f"  Error extracting {pdf_path}: {e}")
        return ""

def extract_text_with_ocr(pdf_path: str, dpi: int = 200) -> str:
    """Extract text from scanned PDF using OCR"""
    try:
        text_blocks = []
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=dpi)
        
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            if text.strip():
                # Clean up OCR artifacts
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text_blocks.append(text)
            
            # Progress for large PDFs
            if (i + 1) % 10 == 0:
                print(f"    OCR progress: {i+1}/{len(images)} pages")
        
        print(f"  OCR complete: {len(text_blocks)} pages with text")
        return '\n\n'.join(text_blocks)
    except Exception as e:
        print(f"  OCR error: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks by words"""
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:  # Skip tiny chunks
            break
        chunks.append(' '.join(chunk_words))
        i += (chunk_size - overlap)
    
    return chunks

def process_single_pdf(pdf_path: Path, output_dir: Path, chunk_size: int, overlap: int):
    """Process a single PDF - designed for parallel execution"""
    try:
        # Extract text
        text = extract_text_from_pdf(str(pdf_path))
        if not text:
            return (pdf_path.name, 0, "No text extracted")
        
        # Create chunks
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            return (pdf_path.name, 0, "No chunks created")
        
        # Write JSONL
        output_path = output_dir / f"{pdf_path.stem}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                entry = {
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "source_type": "pdf",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "doc_id": f"{pdf_path.stem}_{i:04d}"
                    }
                }
                f.write(json.dumps(entry) + '\n')
        
        return (pdf_path.name, len(chunks), "success")
        
    except Exception as e:
        return (pdf_path.name, 0, f"Error: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MVP PDFs to JSONL (Parallel)')
    parser.add_argument('input_dir', help='Directory containing PDFs')
    parser.add_argument('output_dir', help='Output directory for JSONL files')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Words per chunk')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap in words')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDFs (skip Mac resource forks)
    pdf_files = [f for f in input_dir.glob('*.pdf') if not f.name.startswith('._')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return
    
    # Determine number of workers
    num_workers = args.workers or max(1, cpu_count() - 1)
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Using {num_workers} parallel workers")
    print("=" * 80)
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_pdf, 
        output_dir=output_dir, 
        chunk_size=args.chunk_size, 
        overlap=args.overlap
    )
    
    # Process in parallel with progress bar
    total_chunks = 0
    successful = 0
    failed = 0
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, pdf_files),
            total=len(pdf_files),
            desc="Converting PDFs",
            unit="file"
        ))
    
    # Summarize results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("-" * 80)
    
    for filename, chunk_count, status in results:
        if status == "success":
            total_chunks += chunk_count
            successful += 1
            print(f"✓ {filename}: {chunk_count} chunks")
        else:
            failed += 1
            print(f"✗ {filename}: {status}")
    
    print("=" * 80)
    print(f"✅ Successfully converted: {successful}/{len(pdf_files)} PDFs")
    if failed > 0:
        print(f"⚠️  Failed: {failed} PDFs")
    print(f"📝 Total chunks created: {total_chunks}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
