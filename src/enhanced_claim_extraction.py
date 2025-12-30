#!/usr/bin/env python3
"""
AegisTrustNet - Enhanced Claim Extraction Module
This module extracts claims from documents and text for the trust network.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('aegis-claim-extraction')

# Load environment variables
load_dotenv()

# Constants and configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

def extract_claims_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract claims from text using simple pattern matching
    
    Args:
        text: The text to extract claims from
        
    Returns:
        List of extracted claims
    """
    logger.info(f"Extracting claims from text ({len(text)} characters)")
    
    # Simple demo implementation
    # In a real implementation, this would use more sophisticated NLP
    claims = []
    sentences = text.split('.')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Simple heuristic for detecting claims
        claim_indicators = [
            "is ", "are ", "was ", "were ", "will be ", "can ", "could ", "should ",
            "must ", "might ", "may ", "has ", "have ", "had ", "shows ", "suggests ",
            "indicates ", "proves ", "demonstrates ", "reveals ", "states that ", "claims that "
        ]
        
        # Check if sentence might contain a claim
        if any(indicator in sentence.lower() for indicator in claim_indicators):
            # Identify potential entities in the claim
            words = sentence.split()
            entities = []
            
            for i, word in enumerate(words):
                # Crude entity extraction - look for capitalized words
                if word and word[0].isupper() and len(word) > 3 and word.lower() not in ["this", "that", "there", "these", "those", "they", "their"]:
                    entity = word.strip(",.;:!?\"'()[]{}")
                    if entity and len(entity) > 3:
                        entities.append(entity)
            
            if len(entities) > 0:
                claims.append({
                    "text": sentence,
                    "entities": entities,
                    "confidence": min(0.5 + (len(entities) * 0.1), 0.9)  # More entities = higher confidence
                })
    
    logger.info(f"Extracted {len(claims)} potential claims")
    return claims

def store_claims_in_neo4j(claims: List[Dict[str, Any]]) -> bool:
    """
    Store extracted claims in the Neo4j graph
    
    Args:
        claims: List of extracted claims
        
    Returns:
        Success flag
    """
    if not claims:
        logger.info("No claims to store")
        return True
        
    logger.info(f"Storing {len(claims)} claims in Neo4j")
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600
        )
        
        with driver.session() as session:
            for claim in claims:
                # Create a unique ID for the claim
                claim_text = claim["text"]
                claim_hash = str(abs(hash(claim_text)) % (10 ** 10))
                
                # Store the claim
                query = """
                MERGE (c:Claim {claim_text: $claim_text})
                ON CREATE SET 
                    c.id = $id,
                    c.first_seen = datetime(),
                    c.confidence = $confidence,
                    c.trust_score = 0.5,
                    c.source = 'auto-extracted'
                ON MATCH SET 
                    c.confidence = CASE 
                        WHEN $confidence > c.confidence THEN $confidence
                        ELSE c.confidence
                    END
                
                RETURN c.id as id
                """
                
                result = session.run(query, {
                    "claim_text": claim_text,
                    "id": f"claim-{claim_hash}",
                    "confidence": claim["confidence"]
                })
                
                claim_id = result.single()["id"]
                
                # Link claim to entities
                for entity_name in claim["entities"]:
                    entity_query = """
                    MATCH (c:Claim {id: $claim_id})
                    
                    MERGE (e:Entity {name: $entity_name})
                    ON CREATE SET 
                        e.id = $entity_id,
                        e.type = 'EXTRACTED',
                        e.trust_score = 0.5
                    
                    MERGE (c)-[r:MENTIONS]->(e)
                    ON CREATE SET r.weight = 1
                    ON MATCH SET r.weight = r.weight + 1
                    """
                    
                    entity_hash = str(abs(hash(entity_name)) % (10 ** 10))
                    
                    session.run(entity_query, {
                        "claim_id": claim_id,
                        "entity_name": entity_name,
                        "entity_id": f"entity-{entity_hash}"
                    })
        
        logger.info(f"Successfully stored {len(claims)} claims in Neo4j")
        return True
        
    except Exception as e:
        logger.error(f"Error storing claims in Neo4j: {str(e)}")
        return False
    finally:
        if driver:
            driver.close()

def process_document_claims():
    """
    Process documents in the database to extract claims
    
    Returns:
        Success flag
    """
    logger.info("Processing documents for claim extraction")
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600
        )
        
        with driver.session() as session:
            # Find documents that haven't been processed for claims
            result = session.run("""
            MATCH (d:Document)
            WHERE NOT exists(d.claims_extracted) OR d.claims_extracted = false
            RETURN d.id as id, d.title as title
            LIMIT 10
            """)
            
            documents = [(record["id"], record["title"]) for record in result]
            
            if not documents:
                logger.info("No documents requiring claim extraction")
                return True
                
            logger.info(f"Found {len(documents)} documents for claim extraction")
            
            for doc_id, doc_title in documents:
                # Get document text - in a real system, this would access the actual document
                # Here we'll just generate a placeholder text
                doc_text = f"The document '{doc_title}' contains various claims about entities in our knowledge graph. " + \
                           f"It suggests that ancient civilizations had advanced knowledge of astronomy and mathematics. " + \
                           f"Archaeological evidence indicates complex engineering in prehistoric monuments."
                
                # Extract claims
                claims = extract_claims_from_text(doc_text)
                
                # Store claims
                success = store_claims_in_neo4j(claims)
                
                if success:
                    # Mark document as processed
                    session.run("""
                    MATCH (d:Document {id: $doc_id})
                    SET d.claims_extracted = true,
                        d.claims_extraction_date = datetime()
                    """, {"doc_id": doc_id})
                    
                    logger.info(f"Processed document: {doc_title}")
                else:
                    logger.error(f"Failed to process document: {doc_title}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in document claim processing: {str(e)}")
        return False
    finally:
        if driver:
            driver.close()

def run_claim_extraction():
    """
    Run the complete claim extraction process
    
    Returns:
        Success flag
    """
    logger.info("Starting claim extraction process")
    
    # Process documents for claims
    doc_success = process_document_claims()
    
    if doc_success:
        logger.info("Claim extraction completed successfully")
    else:
        logger.warning("Claim extraction completed with errors")
    
    return doc_success

if __name__ == "__main__":
    # Run claim extraction process when script is executed directly
    success = run_claim_extraction()
    print(f"Claim extraction {'succeeded' if success else 'failed'}")