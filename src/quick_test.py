#!/usr/bin/env python3
import sys
sys.path.insert(0, '/media/bob/RAID11/DataShare/AegisTrustNet/src')

from neo4j import GraphDatabase
import os

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("Testing query with search_query parameter...")

with driver.session() as session:
    result = session.run("""
        MATCH (claim:Claim)
        WHERE toLower(claim.claim_text) CONTAINS toLower($search_query)
          AND claim.trust_score >= $min_trust
        WITH claim LIMIT $max_nodes
        
        MATCH (chunk:Chunk)-[r1:CONTAINS_CLAIM]->(claim)
        OPTIONAL MATCH (chunk)-[r2:MENTIONS]->(entity:Entity)
        
        RETURN 
            collect(DISTINCT claim) as claims,
            collect(DISTINCT entity) as entities,
            collect(DISTINCT r1) as claim_rels,
            collect(DISTINCT r2) as entity_rels
    """,
        search_query="meditation",
        min_trust=0.3,
        max_nodes=10
    )
    
    record = result.single()
    
    print(f"Claims: {len(record['claims'])}")
    print(f"Entities: {len(record['entities'])}")
    print(f"Claim rels: {len(record['claim_rels'])}")
    print(f"Entity rels: {len(record['entity_rels'])}")
    
    claim_rels = record.get('claim_rels') or []
    entity_rels = record.get('entity_rels') or []
    
    print(f"\nAfter .get() or []:")
    print(f"  claim_rels length: {len(claim_rels)}")
    print(f"  entity_rels length: {len(entity_rels)}")
    
    if len(claim_rels) > 0:
        print("\n✓ Query returns data!")
        print("\nThe problem must be in how the API code processes the results.")
        print("Let me check the exact API code...")
    else:
        print("\n❌ Query itself returns 0 relationships")
        print("This is a data or query issue, not a code issue")

driver.close()
