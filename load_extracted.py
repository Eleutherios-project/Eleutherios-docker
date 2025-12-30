#!/usr/bin/env python3
"""Load extracted JSONL files into Neo4j"""
import json
import sys
from pathlib import Path
from aegis_graph_builder import GraphBuilderV2 as GraphBuilder

# Connect
builder = GraphBuilder(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="aegistrusted"
)

checkpoint_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
files = list(checkpoint_dir.glob("*_extracted.jsonl"))

print(f"Found {len(files)} extracted files")

total_chunks = 0
total_claims = 0
total_entities = 0

for jsonl_file in files:
    print(f"Loading: {jsonl_file.name}...")
    
    chunks = []
    with open(jsonl_file) as f:
        for line in f:
            chunks.append(json.loads(line))
    
    if chunks:
        stats = builder.build_graph(chunks)
        total_chunks += stats.get('chunks', 0)
        total_claims += stats.get('claims', 0)
        total_entities += stats.get('entities', 0)
        print(f"  ✓ {len(chunks)} chunks, {stats.get('claims', 0)} claims")

print(f"\n{'='*50}")
print(f"TOTAL LOADED:")
print(f"  Chunks:   {total_chunks}")
print(f"  Claims:   {total_claims}")
print(f"  Entities: {total_entities}")
print(f"{'='*50}")

builder.close()
