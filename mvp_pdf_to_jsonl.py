#!/usr/bin/env python3
"""
MVP PDF to JSONL Converter
Simplified version for Aegis MVP extraction
"""

import json
import pypdf
import re
from pathlib import Path
from tqdm import tqdm

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract complete text from PDF"""
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
        return '\n\n'.join(text_blocks)
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
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

def process_pdf(pdf_path: Path, output_path: Path, chunk_size: int = 1000, overlap: int = 200):
    """Convert single PDF to JSONL"""
    
    # Extract text
    text = extract_text_from_pdf(str(pdf_path))
    if not text:
        return 0
    
    # Create chunks
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Write JSONL
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
    
    return len(chunks)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MVP PDFs to JSONL')
    parser.add_argument('input_dir', help='Directory containing PDFs')
    parser.add_argument('output_dir', help='Output directory for JSONL files')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Words per chunk')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap in words')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDFs (skip Mac resource forks)
    pdf_files = [f for f in input_dir.glob('*.pdf') if not f.name.startswith('._')]
    
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 80)
    
    total_chunks = 0
    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        output_path = output_dir / f"{pdf_path.stem}.jsonl"
        chunks = process_pdf(pdf_path, output_path, args.chunk_size, args.overlap)
        total_chunks += chunks
        print(f"  {pdf_path.name}: {chunks} chunks")
    
    print("=" * 80)
    print(f"✅ Converted {len(pdf_files)} PDFs to {total_chunks} chunks")
    print(f"   Output: {output_dir}")

if __name__ == '__main__':
    main()
