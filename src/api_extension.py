#!/usr/bin/env python3
"""
AegisTrustNet - FastAPI Extension
This module extends the Pattern RAG FastAPI service with endpoints for
the Truth Network Overlay functionality.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trust algorithm
from src.trust_algo import process_user_trust_preferences, TrustPropagationAlgorithm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aegis-trustnet-api")

# Constants and configuration
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

# Create API router
router = APIRouter(prefix="/trust", tags=["trust-network"])

# Database connection
def get_neo4j_driver():
    """Create and return a Neo4j driver"""
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600
    )
    try:
        yield driver
    finally:
        driver.close()

# Pydantic models for API requests and responses
class TrustPreference(BaseModel):
    source: str = Field(..., description="Name of the source")
    trust_score: float = Field(..., description="Trust score (-1.0 to 1.0)", ge=-1.0, le=1.0)

class UserTrustPreferences(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: List[TrustPreference] = Field(..., description="List of trust preferences")

class TrustNetwork(BaseModel):
    id: str = Field(..., description="Network identifier")
    name: str = Field(..., description="Network name")
    size: int = Field(..., description="Number of nodes in network")
    description: Optional[str] = Field(None, description="Network description")

class TrustSource(BaseModel):
    id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Source name")
    trust_score: float = Field(..., description="Trust score (0.0 to 1.0)")
    source_type: Optional[str] = Field(None, description="Type of source")
    description: Optional[str] = Field(None, description="Source description")

class ClaimAssessment(BaseModel):
    claim_id: str = Field(..., description="Claim identifier")
    claim_text: str = Field(..., description="Text of the claim")
    trust_score: float = Field(..., description="Trust score (0.0 to 1.0)")
    supporting_sources: List[Dict[str, Any]] = Field(..., description="Sources supporting the claim")
    contradicting_sources: List[Dict[str, Any]] = Field(..., description="Sources contradicting the claim")
    confidence: float = Field(..., description="Confidence in assessment (0.0 to 1.0)")

class ClaimWithEvidence(BaseModel):
    claim_id: str = Field(..., description="Claim identifier")
    claim_text: str = Field(..., description="Text of the claim")
    trust_score: float = Field(..., description="Trust score (0.0 to 1.0)")
    evidence: List[Dict[str, Any]] = Field(..., description="Evidence for/against the claim")
    related_claims: List[Dict[str, Any]] = Field(..., description="Related claims")

class MultiPerspectiveResponse(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    perspectives: List[Dict[str, Any]] = Field(..., description="Different perspectives on the question")
    consensus: Optional[str] = Field(None, description="Areas of consensus across perspectives")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used in response")
    confidence: float = Field(..., description="Overall confidence score")

# API Endpoints
@router.get("/networks", response_model=List[TrustNetwork])
async def get_trust_networks(driver=Depends(get_neo4j_driver)):
    """Get available trust networks"""
    logger.info("Request to get trust networks")
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (tn:TrustNetwork)
                OPTIONAL MATCH (n)-[:BELONGS_TO]->(tn)
                WITH tn, count(n) as node_count
                RETURN tn.id as id, tn.name as name, tn.description as description, node_count as size
            """)
            
            networks = [
                {"id": record["id"], "name": record["name"], 
                 "description": record["description"], "size": record["size"]}
                for record in result
            ]
            
            return networks
    except Exception as e:
        logger.error(f"Error fetching trust networks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/networks/{network_id}", response_model=TrustNetwork)
async def get_trust_network(network_id: str, driver=Depends(get_neo4j_driver)):
    """Get details of a specific trust network"""
    logger.info(f"Request to get trust network: {network_id}")
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (tn:TrustNetwork {id: $network_id})
                OPTIONAL MATCH (n)-[:BELONGS_TO]->(tn)
                WITH tn, count(n) as node_count
                RETURN tn.id as id, tn.name as name, tn.description as description, node_count as size
            """, {"network_id": network_id})
            
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Trust network {network_id} not found")
                
            return {
                "id": record["id"],
                "name": record["name"],
                "description": record["description"],
                "size": record["size"]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching trust network {network_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/sources", response_model=List[TrustSource])
async def get_trust_sources(
    limit: int = 20, 
    offset: int = 0,
    min_trust: float = 0.0,
    driver=Depends(get_neo4j_driver)
):
    """Get sources with their trust metrics"""
    logger.info(f"Request to get trust sources (limit={limit}, offset={offset}, min_trust={min_trust})")
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Source)
                WHERE s.trust_score >= $min_trust
                RETURN s.id as id, s.name as name, s.trust_score as trust_score,
                       s.source_type as source_type, s.description as description
                ORDER BY s.trust_score DESC
                SKIP $offset LIMIT $limit
            """, {"limit": limit, "offset": offset, "min_trust": min_trust})
            
            sources = [
                {"id": record["id"], "name": record["name"], 
                 "trust_score": record["trust_score"], 
                 "source_type": record["source_type"],
                 "description": record["description"]}
                for record in result
            ]
            
            return sources
    except Exception as e:
        logger.error(f"Error fetching trust sources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/sources/{source_id}", response_model=TrustSource)
async def get_trust_source(source_id: str, driver=Depends(get_neo4j_driver)):
    """Get trust metrics for a specific source"""
    logger.info(f"Request to get trust source: {source_id}")
    
    try:
        with driver.session() as session:
            # Try to find by ID first
            result = session.run("""
                MATCH (s:Source)
                WHERE s.id = $source_id OR s.name = $source_id
                RETURN s.id as id, s.name as name, s.trust_score as trust_score,
                       s.source_type as source_type, s.description as description
            """, {"source_id": source_id})
            
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
                
            return {
                "id": record["id"],
                "name": record["name"],
                "trust_score": record["trust_score"] if record["trust_score"] is not None else 0.0,
                "source_type": record["source_type"],
                "description": record["description"]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching trust source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/claims/{claim_id}", response_model=ClaimWithEvidence)
async def get_claim_assessment(claim_id: str, driver=Depends(get_neo4j_driver)):
    """Get trust assessment for a specific claim"""
    logger.info(f"Request to get claim assessment: {claim_id}")
    
    try:
        with driver.session() as session:
            # Get the claim details
            claim_result = session.run("""
                MATCH (c:Claim)
                WHERE c.id = $claim_id OR c.claim_text CONTAINS $claim_id
                RETURN c.id as id, c.claim_text as claim_text, 
                       coalesce(c.trust_score, 0.0) as trust_score
            """, {"claim_id": claim_id})
            
            claim_record = claim_result.single()
            if not claim_record:
                raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
            
            # Get evidence for/against the claim
            evidence_result = session.run("""
                MATCH (c:Claim {id: $claim_id})<-[r]-(s)
                WHERE type(r) IN ['SUPPORTS', 'CONTRADICTS', 'MENTIONS']
                RETURN type(r) as relation_type, s.name as source_name,
                       labels(s) as source_types, coalesce(s.trust_score, 0.0) as source_trust,
                       coalesce(r.weight, 1.0) as weight
                ORDER BY source_trust DESC
            """, {"claim_id": claim_record["id"]})
            
            evidence = [
                {
                    "source_name": record["source_name"],
                    "source_type": record["source_types"][0] if record["source_types"] else "Unknown",
                    "source_trust": record["source_trust"],
                    "relation": record["relation_type"],
                    "weight": record["weight"]
                }
                for record in evidence_result
            ]
            
            # Get related claims
            related_result = session.run("""
                MATCH (c:Claim {id: $claim_id})-[:RELATED_TO|SUPPORTS|CONTRADICTS]-(related:Claim)
                RETURN related.id as id, related.claim_text as claim_text,
                       coalesce(related.trust_score, 0.0) as trust_score
                LIMIT 5
            """, {"claim_id": claim_record["id"]})
            
            related_claims = [
                {
                    "id": record["id"],
                    "claim_text": record["claim_text"],
                    "trust_score": record["trust_score"]
                }
                for record in related_result
            ]
            
            return {
                "claim_id": claim_record["id"],
                "claim_text": claim_record["claim_text"],
                "trust_score": claim_record["trust_score"],
                "evidence": evidence,
                "related_claims": related_claims
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching claim assessment {claim_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/user/preferences", status_code=202)
async def set_user_trust_preferences(preferences: UserTrustPreferences):
    """Set user trust preferences and compute personalized trust scores"""
    logger.info(f"Request to set trust preferences for user: {preferences.user_id}")
    
    # Convert to the format expected by the trust algorithm
    prefs_dict = {
        item.source: item.trust_score for item in preferences.preferences
    }
    
    # Process asynchronously
    try:
        # This would normally be done in a background task queue,
        # but we'll simulate that here with a simple function call
        # that will be executed in the main thread
        success = process_user_trust_preferences(preferences.user_id, prefs_dict)
        
        if success:
            return {"message": "Trust preferences processed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process trust preferences")
    except Exception as e:
        logger.error(f"Error processing trust preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/chat/completions/verified")
async def verified_chat_completions(request: Request):
    """Get pattern-findings with trust verification"""
    body = await request.json()
    logger.info(f"Request for verified completions: {body.get('messages', [])[:-1]}")
    
    # Extract the query from the messages
    messages = body.get("messages", [])
    query = next((msg["content"] for msg in reversed(messages) if msg.get("role") == "user"), None)
    
    if not query:
        raise HTTPException(status_code=400, detail="No valid query found in messages")
    
    try:
        # This endpoint integrates with the Pattern RAG system
        # We'll need to add trust verification to the existing pattern finding logic
        
        # 1. Get the user's trust preferences if available
        user_id = body.get("user", "anonymous")
        
        # 2. Query Neo4j for relevant claims related to the query
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Find claims that might be relevant to the query
            claims_result = session.run("""
                CALL db.index.fulltext.queryNodes("entity_names", $query) 
                YIELD node, score
                WHERE node:Claim
                RETURN node.id as id, node.claim_text as text, 
                       coalesce(node.trust_score, 0.0) as trust_score,
                       score as relevance
                ORDER BY relevance DESC
                LIMIT 5
            """, {"query": query})
            
            relevant_claims = [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "trust_score": record["trust_score"],
                    "relevance": record["relevance"]
                }
                for record in claims_result
            ]
            
            # Get contrasting perspectives based on community detection
            perspectives_result = session.run("""
                MATCH (c:Community)
                WHERE c.size > 10
                WITH c
                MATCH (e:Entity)-[:BELONGS_TO]->(c)
                WHERE e.trust_score > 0.3
                WITH c, e
                ORDER BY e.trust_score DESC
                LIMIT 5
                WITH c, collect(e.name) as top_entities
                RETURN c.id as community_id, c.name as community_name, 
                       top_entities, coalesce(c.description, 'Community based on graph structure') as description
                LIMIT 3
            """)
            
            perspectives = [
                {
                    "community_id": record["community_id"],
                    "name": record["community_name"],
                    "key_entities": record["top_entities"],
                    "description": record["description"]
                }
                for record in perspectives_result
            ]
        
        driver.close()
        
        # 3. Forward to the Pattern RAG with enhanced context
        # This would call the pattern_rag_service with added trust context
        # For now, we'll return a simplified response
        
        # The actual implementation would integrate with the pattern finding logic
        # and return a MultiPerspectiveResponse
        
        return {
            "query": query,
            "trust_context": {
                "relevant_claims": relevant_claims,
                "perspectives": perspectives
            },
            "message": "This endpoint will integrate with the Pattern RAG system to provide trust-verified responses"
        }
    except Exception as e:
        logger.error(f"Error in verified chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
