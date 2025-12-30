#!/usr/bin/env python3
"""
AegisTrustNet - Multi-Perspective Response Service
This module provides trust-enhanced responses that incorporate trusted sources,
highlight significant patterns, and contrast competing viewpoints when appropriate.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('aegis-multi-perspective')

# Load environment variables
load_dotenv()

# Constants and configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")
PATTERN_RAG_DIR = os.environ.get("PATTERN_RAG_DIR", "")
MIN_TRUST_SCORE = float(os.environ.get("MIN_TRUST_SCORE", "0.3"))

class MultiPerspectiveService:
    def __init__(self, user_id: Optional[str] = None):
        """
        Initialize the multi-perspective service with Neo4j connection and user context
        
        Args:
            user_id: Optional user identifier for personalized trust preferences
        """
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600
            )
            logger.info(f"Connected to Neo4j at {NEO4J_URI}")
            
            self.user_id = user_id
            self.user_trust_preferences = self._load_user_trust_preferences() if user_id else {}
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def _load_user_trust_preferences(self) -> Dict[str, float]:
        """Load user trust preferences from the database"""
        if not self.user_id:
            return {}
            
        preferences = {}
        
        with self.driver.session() as session:
            result = session.run("""
            MATCH (u:User {id: $user_id})-[r:TRUSTS]->(s:Source)
            RETURN s.name as source, r.score as trust_score
            """, {"user_id": self.user_id})
            
            for record in result:
                preferences[record["source"]] = record["trust_score"]
                
        logger.info(f"Loaded {len(preferences)} trust preferences for user {self.user_id}")
        return preferences

    def get_relevant_claims(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find claims relevant to the user's query
        
        Args:
            query: The user's query text
            limit: Maximum number of claims to return
            
        Returns:
            List of relevant claims with trust scores
        """
        claims = []
        
        try:
            with self.driver.session() as session:
                # Skip fulltext search completely
                # Use direct claim text matching
                words = [w.strip(".,?!:;()[]{}\"'") for w in query.split() if len(w) > 3]
                
                if words:
                    logger.info("Using direct claim text search")
                    
                    # Create parameters for each word
                    word_params = {
                        f"word{i}": words[i].lower() for i in range(min(5, len(words)))
                    }
                    word_params["limit"] = limit
                    word_params["min_trust"] = MIN_TRUST_SCORE
                    
                    # Build a query with multiple CONTAINS clauses
                    where_clauses = [f"toLower(c.claim_text) CONTAINS $word{i}" for i in range(min(5, len(words)))]
                    where_clause = " OR ".join(where_clauses)
                    
                    query = f"""
                    MATCH (c:Claim)
                    WHERE {where_clause}
                    RETURN c.id as id, c.claim_text as text, 
                           coalesce(c.trust_score, 0.5) as trust_score, 
                           1.0 as relevance
                    LIMIT $limit
                    """
                    
                    result = session.run(query, word_params)
                    
                    for record in result:
                        claim = {
                            "id": record["id"],
                            "text": record["text"],
                            "trust_score": record["trust_score"],
                            "relevance": record["relevance"]
                        }
                        
                        # Get supporting and contradicting evidence
                        evidence = self._get_claim_evidence(session, claim["id"])
                        claim.update(evidence)
                        
                        claims.append(claim)
                else:
                    # If no meaningful words, just get some sample claims
                    logger.info("Using sample claims search")
                    
                    query = """
                    MATCH (c:Claim)
                    RETURN c.id as id, c.claim_text as text, 
                           coalesce(c.trust_score, 0.5) as trust_score, 
                           0.5 as relevance
                    ORDER BY c.trust_score DESC
                    LIMIT $limit
                    """
                    
                    result = session.run(query, {"limit": limit})
                    
                    for record in result:
                        claim = {
                            "id": record["id"],
                            "text": record["text"],
                            "trust_score": record["trust_score"],
                            "relevance": record["relevance"]
                        }
                        
                        # Get supporting and contradicting evidence
                        evidence = self._get_claim_evidence(session, claim["id"])
                        claim.update(evidence)
                        
                        claims.append(claim)
        except Exception as e:
            logger.error(f"Error getting relevant claims: {str(e)}")
            
            # Create a dummy claim if no claims found
            if not claims:
                claims.append({
                    "id": "no-match",
                    "text": f"No specific information found about {query}",
                    "trust_score": 0.3,
                    "relevance": 0.1,
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "entities": []
                })
                
        logger.info(f"Found {len(claims)} relevant claims for query: {query}")
        return claims

    def _get_claim_evidence(self, session, claim_id: str) -> Dict[str, Any]:
        """Get supporting and contradicting evidence for a claim"""
        try:
            result = session.run("""
            MATCH (c:Claim {id: $claim_id})
            
            // Get supporting claims/sources
            OPTIONAL MATCH (c)<-[rs:SUPPORTS]-(supporting)
            WITH c, collect({
                id: supporting.id,
                name: CASE WHEN supporting:Source THEN supporting.name 
                            WHEN supporting:Claim THEN supporting.claim_text
                            ELSE 'Unknown' END,
                type: labels(supporting)[0],
                trust_score: coalesce(supporting.trust_score, 0.5)
            }) as supporting_evidence
            
            // Get contradicting claims/sources
            OPTIONAL MATCH (c)<-[rc:CONTRADICTS]-(contradicting)
            WITH c, supporting_evidence, collect({
                id: contradicting.id,
                name: CASE WHEN contradicting:Source THEN contradicting.name 
                           WHEN contradicting:Claim THEN contradicting.claim_text
                           ELSE 'Unknown' END,
                type: labels(contradicting)[0],
                trust_score: coalesce(contradicting.trust_score, 0.5)
            }) as contradicting_evidence
            
            // Get entities mentioned in claim
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH c, supporting_evidence, contradicting_evidence, 
                 collect({name: e.name, type: e.type}) as entities
            
            RETURN supporting_evidence, contradicting_evidence, entities
            """, {"claim_id": claim_id})
            
            record = result.single()
            if not record:
                return {
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "entities": []
                }
            
            return {
                "supporting_evidence": record["supporting_evidence"],
                "contradicting_evidence": record["contradicting_evidence"],
                "entities": record["entities"]
            }
        except Exception as e:
            logger.error(f"Error getting claim evidence: {str(e)}")
            return {
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "entities": []
            }

    def get_relevant_patterns(self, query: str, entities: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find patterns relevant to the query and mentioned entities
        
        Args:
            query: The user's query text
            entities: List of entity names mentioned in the query or relevant claims
            limit: Maximum number of patterns to return
            
        Returns:
            List of relevant patterns
        """
        patterns = []
        
        if not entities:
            return patterns
            
        with self.driver.session() as session:
            # Find patterns that involve the entities
            entity_params = {f"entity{i}": entities[i] for i in range(len(entities))}
            entity_where_clauses = [f"e.name = $entity{i}" for i in range(len(entities))]
            entity_where = " OR ".join(entity_where_clauses)
            
            query = f"""
            MATCH (p:Pattern)-[:INVOLVES]->(e:Entity)
            WHERE {entity_where}
            WITH p, count(DISTINCT e) as entity_match_count
            ORDER BY entity_match_count DESC
            LIMIT $limit
            
            RETURN p.id as id, p.type as type, p.data as data, 
                   p.discovered as discovered, entity_match_count
            """
            
            params = {"limit": limit}
            params.update(entity_params)
            
            result = session.run(query, params)
            
            for record in result:
                pattern_data = json.loads(record["data"])
                pattern = {
                    "id": record["id"],
                    "type": record["type"],
                    "discovered": record["discovered"],
                    "entity_match_count": record["entity_match_count"],
                    "data": pattern_data
                }
                
                patterns.append(pattern)
                
        logger.info(f"Found {len(patterns)} relevant patterns")
        return patterns

    def get_community_perspectives(self, query: str, entities: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find different community perspectives on the query topic
        
        Args:
            query: The user's query text
            entities: List of entity names mentioned in the query or relevant claims
            limit: Maximum number of perspectives to return
            
        Returns:
            List of community perspectives
        """
        perspectives = []
        
        if not entities:
            return perspectives
            
        with self.driver.session() as session:
            # Find communities that have strong opinions on these entities
            entity_params = {f"entity{i}": entities[i] for i in range(len(entities))}
            entity_where_clauses = [f"e.name = $entity{i}" for i in range(len(entities))]
            entity_where = " OR ".join(entity_where_clauses)
            
            query = f"""
            MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
            WHERE {entity_where}
            WITH c, count(DISTINCT e) as entity_count
            WHERE c.size > 10 AND entity_count > 0
            
            // Find top entities in this community
            MATCH (top:Entity)-[:BELONGS_TO]->(c)
            WHERE top.trust_score > 0.5
            WITH c, entity_count, top
            ORDER BY top.trust_score DESC
            LIMIT 10
            
            WITH c, entity_count, collect({{name: top.name, trust_score: top.trust_score}}) as top_entities
            
            // Find claims supported by this community
            OPTIONAL MATCH (claim:Claim)<-[:SUPPORTS]-(source:Source)-[:BELONGS_TO]->(c)
            WHERE claim.trust_score > 0.4
            WITH c, entity_count, top_entities, 
                 collect(DISTINCT {{id: claim.id, text: claim.claim_text, trust: claim.trust_score}}) as supported_claims
            
            RETURN c.id as id, c.name as name, entity_count,
                   coalesce(c.description, 'Community based on network analysis') as description,
                   top_entities, supported_claims
            ORDER BY entity_count DESC
            LIMIT $limit
            """
            
            params = {"limit": limit}
            params.update(entity_params)
            
            result = session.run(query, params)
            
            for record in result:
                perspective = {
                    "community_id": record["id"],
                    "name": record["name"],
                    "description": record["description"],
                    "relevance": record["entity_count"],
                    "key_entities": record["top_entities"],
                    "supported_claims": record["supported_claims"]
                }
                
                perspectives.append(perspective)
                
        logger.info(f"Found {len(perspectives)} community perspectives")
        return perspectives

    def generate_consensus_view(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a consensus view based on the most trusted claims
        
        Args:
            claims: List of relevant claims with evidence
            
        Returns:
            Dictionary with consensus information
        """
        if not claims:
            return {"exists": False, "message": "No relevant information found"}
        
        # Filter for high-trust claims (above 0.7)
        high_trust_claims = [c for c in claims if c["trust_score"] > 0.7]
        
        if not high_trust_claims:
            return {"exists": False, "message": "No high-trust claims available for consensus"}
        
        # Find areas of agreement
        consensus_claims = []
        all_entities = set()
        
        for claim in high_trust_claims:
            # A claim with strong support and limited contradiction is part of consensus
            if (len(claim["supporting_evidence"]) > len(claim["contradicting_evidence"])) and \
               any(e["trust_score"] > 0.7 for e in claim["supporting_evidence"]):
                consensus_claims.append(claim)
                
                # Collect entities
                for entity in claim.get("entities", []):
                    all_entities.add(entity["name"])
        
        if not consensus_claims:
            return {"exists": False, "message": "No consensus found among high-trust claims"}
        
        # Extract key points of consensus
        consensus = {
            "exists": True,
            "claims": consensus_claims,
            "entities": list(all_entities),
            "strength": len(consensus_claims) / max(1, len(high_trust_claims))
        }
        
        return consensus

    def generate_multi_perspective_response(self, query: str, original_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive response that includes multiple perspectives
        
        Args:
            query: The user's query
            original_response: Optional response from another system to enhance
            
        Returns:
            Dictionary with the enhanced response
        """
        start_time = time.time()
        logger.info(f"Generating multi-perspective response for: {query}")
        
        # Step 1: Extract key entities from the query
        entities = self._extract_entities_from_query(query)
        logger.info(f"Extracted entities: {entities}")
        
        # Step 2: Get relevant claims
        claims = self.get_relevant_claims(query)
        
        # Extract all entities from claims
        claim_entities = []
        for claim in claims:
            for entity in claim.get("entities", []):
                if entity["name"] not in entities and entity["name"] not in claim_entities:
                    claim_entities.append(entity["name"])
        
        # Combine query entities and claim entities
        all_entities = entities + claim_entities[:10]  # Limit to avoid too many
        
        # Step 3: Get relevant patterns
        patterns = self.get_relevant_patterns(query, all_entities)
        
        # Step 4: Get community perspectives
        perspectives = self.get_community_perspectives(query, all_entities)
        
        # Step 5: Generate consensus view
        consensus = self.generate_consensus_view(claims)
        
        # Step 6: Calculate overall confidence
        if claims:
            avg_trust = sum(claim["trust_score"] for claim in claims) / len(claims)
            evidence_ratio = sum(len(claim["supporting_evidence"]) for claim in claims) / max(1, sum(len(claim["contradicting_evidence"]) for claim in claims))
            confidence = min(1.0, avg_trust * (0.5 + 0.5 * min(1.0, evidence_ratio / 5)))
        else:
            confidence = 0.3  # Low confidence if no claims found
        
        # Step 7: Construct the response
        response = {
            "query": query,
            "original_answer": original_response,
            "claims": claims,
            "patterns": patterns,
            "perspectives": perspectives,
            "consensus": consensus,
            "confidence": confidence,
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Generated multi-perspective response with {len(claims)} claims, {len(patterns)} patterns, and {len(perspectives)} perspectives")
        return response

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entity names from the query"""
        entities = []
        
        try:
            with self.driver.session() as session:
                # Skip fulltext search completely
                # Use direct entity name matching
                words = [w.strip(".,?!:;()[]{}\"'") for w in query.split() if len(w) > 3]
                
                if words:
                    # Try to match entity names directly
                    word_params = {f"word{i}": words[i].lower() for i in range(min(5, len(words)))}
                    word_params["limit"] = 10
                    
                    # Build a Cypher query with multiple CONTAINS clauses
                    where_clauses = [f"toLower(e.name) CONTAINS $word{i}" for i in range(min(5, len(words)))]
                    where_clause = " OR ".join(where_clauses)
                    
                    query = f"""
                    MATCH (e:Entity)
                    WHERE {where_clause}
                    RETURN e.name as name, 1.0 as score
                    LIMIT $limit
                    """
                    
                    result = session.run(query, word_params)
                    for record in result:
                        entities.append(record["name"])
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            
            # Provide some default entities based on common words in query
            for word in query.split():
                if len(word) > 4 and word[0].isupper():
                    entities.append(word.strip(".,?!:;()[]{}\"'"))
        
        return entities

    def build_llm_prompt(self, multi_perspective_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a prompt for an LLM to generate a natural language response
        from the multi-perspective data
        
        Args:
            multi_perspective_data: The multi-perspective response data
            
        Returns:
            Dictionary with prompt information for the LLM
        """
        query = multi_perspective_data["query"]
        claims = multi_perspective_data.get("claims", [])
        patterns = multi_perspective_data.get("patterns", [])
        perspectives = multi_perspective_data.get("perspectives", [])
        consensus = multi_perspective_data.get("consensus", {"exists": False})
        confidence = multi_perspective_data.get("confidence", 0.5)
        
        # Build a structured prompt for the LLM
        prompt = {
            "role": "system",
            "content": f"""You are a research assistant with access to a knowledge graph of interconnected information.
I've analyzed the query: "{query}"

Based on my trust network analysis, here's what I found:
"""
        }
        
        # Add claims section
        if claims:
            prompt["content"] += "\n## Relevant Information\n"
            for i, claim in enumerate(claims[:5]):  # Limit to top 5
                trust_level = "High" if claim["trust_score"] > 0.7 else "Moderate" if claim["trust_score"] > 0.4 else "Low"
                prompt["content"] += f"\n{i+1}. {claim['text']} (Trust: {trust_level})"
                
                # Add supporting evidence for high-trust claims
                if claim["trust_score"] > 0.6 and claim["supporting_evidence"]:
                    prompt["content"] += "\n   Supporting evidence:"
                    for evidence in claim["supporting_evidence"][:3]:
                        prompt["content"] += f"\n   - {evidence['name']}"
        
        # Add patterns section
        if patterns:
            prompt["content"] += "\n\n## Significant Patterns\n"
            for pattern in patterns[:3]:  # Limit to top 3
                pattern_type = pattern["type"].replace("_", " ").title()
                
                if pattern["type"] == "correlation":
                    data = pattern["data"]
                    prompt["content"] += f"\n- Found {pattern_type}: {data['entity1']} and {data['entity2']} frequently appear together (confidence: {data['lift']:.2f})"
                
                elif pattern["type"] == "causal":
                    data = pattern["data"]
                    prompt["content"] += f"\n- Found {pattern_type}: {data['cause']} may lead to {data['effect']} (confidence: {data['confidence']:.2f})"
                
                elif pattern["type"] == "higher_order_effect":
                    data = pattern["data"]
                    intermediates = " → ".join(data["intermediate_factors"])
                    prompt["content"] += f"\n- Found {pattern_type}: {data['initial_cause']} → {intermediates} → {data['final_effect']}"
                
                elif pattern["type"] == "anomalous_claim":
                    data = pattern["data"]
                    prompt["content"] += f"\n- Found {pattern_type}: '{data['claim_text']}' is unusual but has significant support"
                
                else:
                    prompt["content"] += f"\n- Found {pattern_type} involving {', '.join(all_entities)}"
        
        # Add perspectives section if there are meaningful differences
        if len(perspectives) > 1:
            prompt["content"] += "\n\n## Different Perspectives\n"
            for perspective in perspectives:
                prompt["content"] += f"\n{perspective['name']}:"
                if perspective['supported_claims']:
                    claims_sample = perspective['supported_claims'][:2]
                    for claim in claims_sample:
                        prompt["content"] += f"\n- {claim['text']}"
                else:
                    key_entities = [e["name"] for e in perspective.get("key_entities", [])][:3]
                    prompt["content"] += f"\n- Key entities: {', '.join(key_entities)}"
        
        # Add consensus section if it exists
        if consensus["exists"]:
            prompt["content"] += "\n\n## Areas of Consensus\n"
            for claim in consensus.get("claims", [])[:3]:
                prompt["content"] += f"\n- {claim['text']}"
        
        # Add confidence level
        confidence_level = "High" if confidence > 0.7 else "Moderate" if confidence > 0.4 else "Low"
        prompt["content"] += f"\n\nOverall confidence in this response: {confidence_level}"
        
        # Add instructions for the LLM
        prompt["content"] += """

Please provide a comprehensive, well-structured response to the query that:
1. Focuses on the most trustworthy information
2. Highlights significant patterns and connections
3. Mentions different perspectives ONLY when they are grounded in trusted sources
4. Clearly states the level of confidence in different parts of the response
5. Maintains a balanced, neutral tone while focusing on PATTERNS rather than opinions

DO NOT include any statements referring to "different perspectives" or "according to the trust network" or similar meta-commentary. 
Write as if you are directly answering the query with the information provided.
"""
        
        return prompt

def process_user_query(user_id: Optional[str], query: str, original_response: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a user query to generate a multi-perspective response
    
    Args:
        user_id: Optional user identifier
        query: The user's query
        original_response: Optional original response to enhance
        
    Returns:
        Multi-perspective response
    """
    service = MultiPerspectiveService(user_id)
    try:
        # Generate multi-perspective data
        response_data = service.generate_multi_perspective_response(query, original_response)
        
        # Build LLM prompt
        prompt = service.build_llm_prompt(response_data)
        response_data["prompt"] = prompt
        
        return response_data
    finally:
        service.close()

if __name__ == "__main__":
    # Simple test
    test_query = "What evidence connects ancient civilizations to advanced technology?"
    response = process_user_query(None, test_query)
    print(json.dumps(response["prompt"], indent=2))
