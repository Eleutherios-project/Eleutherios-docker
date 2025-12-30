#!/usr/bin/env python3
from neo4j import GraphDatabase
import os

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Use the EXACT same query string as the failing API call
query_text = "qi gong and meditation"
min_trust = 0.3
max_nodes = 50

print(f"Testing with: '{query_text}'")
print(f"min_trust: {min_trust}, max_nodes: {max_nodes}\n")

with driver.session() as session:
    result = session.run("""
        // Find Claims matching the query
        MATCH (claim:Claim)
        WHERE toLower(claim.claim_text) CONTAINS toLower($search_query)
          AND claim.trust_score >= $min_trust
        WITH claim LIMIT $max_nodes
        
        // Get Chunks containing these Claims
        MATCH (chunk:Chunk)-[r1:CONTAINS_CLAIM]->(claim)
        
        // Get Entities mentioned in those Chunks
        OPTIONAL MATCH (chunk)-[r2:MENTIONS]->(entity:Entity)
        
        // Get Documents containing the Chunks
        OPTIONAL MATCH (doc:Document)-[r3:CONTAINS_CHUNK]->(chunk)
        
        RETURN 
            collect(DISTINCT claim) as claims,
            collect(DISTINCT entity) as entities,
            collect(DISTINCT chunk) as chunks,
            collect(DISTINCT doc) as documents,
            collect(DISTINCT r1) as claim_rels,
            collect(DISTINCT r2) as entity_rels,
            collect(DISTINCT r3) as doc_rels
    """,
        search_query=query_text,
        min_trust=min_trust,
        max_nodes=max_nodes
    )
    
    record = result.single()
    
    print("Results:")
    print(f"  claims: {len(record['claims'])}")
    print(f"  entities: {len(record['entities'])}")
    print(f"  chunks: {len(record['chunks'])}")
    print(f"  claim_rels: {len(record['claim_rels'])}")
    print(f"  entity_rels: {len(record['entity_rels'])}")
    
    if len(record['claim_rels']) == 0:
        print("\n❌ This search returns 0 relationships!")
        print("\nTrying just 'meditation'...")
        
        result2 = session.run("""
            MATCH (claim:Claim)
            WHERE toLower(claim.claim_text) CONTAINS toLower($search_query)
              AND claim.trust_score >= $min_trust
            RETURN count(claim) as count
        """,
            search_query="meditation",
            min_trust=0.3
        )
        count1 = result2.single()['count']
        
        result3 = session.run("""
            MATCH (claim:Claim)
            WHERE toLower(claim.claim_text) CONTAINS toLower($search_query)
              AND claim.trust_score >= $min_trust
            RETURN count(claim) as count
        """,
            search_query="qi gong and meditation",
            min_trust=0.3
        )
        count2 = result3.single()['count']
        
        print(f"\n  'meditation' finds: {count1} claims")
        print(f"  'qi gong and meditation' finds: {count2} claims")
        
        if count2 == 0:
            print("\n💡 The search string doesn't match any claims!")
            print("   Claims must contain ALL words: 'qi', 'gong', 'and', 'meditation'")
    else:
        print(f"\n✓ Found {len(record['claim_rels'])} claim relationships!")

driver.close()
