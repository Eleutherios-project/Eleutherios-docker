#!/usr/bin/env python3
"""
Aegis Insight - Pattern Search with LLM Synthesis

Combines Neo4j graph search + embedding similarity + LLM synthesis
to generate comprehensive summaries of knowledge patterns.

FIXED VERSION - Handles None values in confidence/similarity
"""

import logging
import os
import json
from typing import List, Dict, Optional
from neo4j import GraphDatabase
import psycopg2
from sentence_transformers import SentenceTransformer
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternSearchLLM:
    """
    Pattern search that combines graph topology and semantic similarity,
    then synthesizes results using an LLM
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "aegistrusted",
        postgres_host: str = os.environ.get("POSTGRES_HOST", "localhost"),
        postgres_db: str = "aegis_insight",
        postgres_user: str = "aegis",
        postgres_password: str = "aegis_trusted_2025",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mistral-nemo:12b"
    ):
        """Initialize connections"""
        
        # Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        
        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=postgres_host,
            database=postgres_db,
            user=postgres_user,
            password=postgres_password
        )
        
        # Embedding model (same as used for generation)
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        
        # Ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        logger.info("Pattern search initialized")
    
    def search_neo4j(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search Neo4j for claims and entities matching the query
        """
        
        cypher = """
        // Find matching claims
        MATCH (c:Claim)
        WHERE toLower(c.claim_text) CONTAINS toLower($search_query)
        WITH c, c.confidence as relevance
        ORDER BY relevance DESC
        LIMIT $limit
        
        // Get related entities
        OPTIONAL MATCH (chunk:Chunk)-[:CONTAINS_CLAIM]->(c)
        OPTIONAL MATCH (chunk)-[:MENTIONS]->(e:Entity)
        
        RETURN 
            elementId(c) as id,
            c.claim_text as text,
            c.confidence as confidence,
            c.claim_type as claim_type,
            c.source_file as source,
            collect(DISTINCT e.name)[0..5] as entities
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, search_query=query, limit=limit)
            claims = [dict(record) for record in result]
        
        logger.info(f"Neo4j found {len(claims)} matching claims")
        return claims


    def search_entity_aliases(self, query: str, limit: int = 100) -> List[Dict]:
        """
        Search for claims via Entity relationships with SAME_AS traversal.
        This finds claims that mention entity variants (e.g., "Gen Butler" -> "Smedley Butler")
        """
        
        cypher = """
        // Find entities matching the query
        MATCH (target:Entity)
        WHERE toLower(target.name) CONTAINS toLower($search_query)
        
        // Traverse SAME_AS to find all aliases (up to 2 hops)
        WITH target
        OPTIONAL MATCH (target)-[:SAME_AS*0..2]-(related:Entity)
        WITH COLLECT(DISTINCT target) + COLLECT(DISTINCT related) AS entities
        UNWIND entities AS e
        
        // Find claims via Chunk->MENTIONS->Entity path
        MATCH (chunk:Chunk)-[:MENTIONS]->(e)
        MATCH (chunk)-[:CONTAINS_CLAIM]->(c:Claim)
        
        WITH DISTINCT c, e.name as matched_entity
        
        RETURN 
            elementId(c) as id,
            c.claim_text as text,
            c.confidence as confidence,
            c.claim_type as claim_type,
            c.source_file as source,
            collect(DISTINCT matched_entity)[0..5] as entities,
            'entity_alias' as search_method
        LIMIT $limit
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, search_query=query, limit=limit)
            claims = [dict(record) for record in result]
        
        logger.info(f"Entity alias search found {len(claims)} claims via SAME_AS traversal")
        return claims


    def search_embeddings(self, query: str, limit: int = 20, min_similarity: float = 0.3) -> List[Dict]:
        """
        Search embeddings for semantically similar claims
        """

        # Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # Search PostgreSQL
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    claim_id,
                    claim_text,
                    confidence,
                    claim_type,
                    source_file,
                    1 - (embedding <=> %s::vector) as similarity
                FROM claim_embeddings
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                query_embedding.tolist(),
                query_embedding.tolist(),
                min_similarity,
                query_embedding.tolist(),
                limit
            ))

            results = cur.fetchall()

        if not results:
            return []

        # Get claim IDs (these are string hashes like 'claim_2d86e4c4d950')
        claim_ids = [row[0] for row in results]

        # Get element IDs from Neo4j using claim_id property
        element_id_map = {}
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Claim)
                WHERE c.claim_id IN $claim_ids
                RETURN c.claim_id as claim_id, elementId(c) as element_id
            """, claim_ids=claim_ids)
            for record in result:
                element_id_map[record['claim_id']] = record['element_id']

        claims = []
        for row in results:
            claim_id = row[0]  # This is a string like 'claim_2d86e4c4d950'
            element_id = element_id_map.get(claim_id, claim_id)  # Use element ID if found
            claims.append({
                'id': element_id,
                'text': row[1],
                'confidence': row[2],
                'claim_type': row[3],
                'source': row[4],
                'similarity': row[5],
                'entities': []
            })

        logger.info(f"Embedding search found {len(claims)} similar claims")
        return claims
    
    def merge_results(self, neo4j_results: List[Dict], embedding_results: List[Dict]) -> List[Dict]:
        """
        Merge and deduplicate results from both sources
        Prioritize by: similarity score > confidence > claim type (PRIMARY > SECONDARY > META > CONTEXTUAL)
        """
        
        # Create lookup by ID
        merged = {}
        
        # Add Neo4j results (prioritize graph connections)
        for claim in neo4j_results:
            claim_id = str(claim['id'])
            claim['search_source'] = 'graph'
            claim['similarity'] = claim.get('confidence', 0.5)  # Use confidence as proxy
            merged[claim_id] = claim
        
        # Add embedding results (add or enhance)
        for claim in embedding_results:
            claim_id = str(claim['id'])
            if claim_id in merged:
                # Already have it from graph, add similarity score
                merged[claim_id]['similarity'] = max(
                    merged[claim_id].get('similarity', 0),
                    claim['similarity']
                )
                merged[claim_id]['search_source'] = 'both'
            else:
                # New result from embeddings
                claim['search_source'] = 'semantic'
                merged[claim_id] = claim
        
        # Convert to list and sort by relevance
        results = list(merged.values())
        
        # Sort by: similarity score, then confidence, then claim type
        claim_type_priority = {'PRIMARY': 3, 'SECONDARY': 2, 'META': 1, 'CONTEXTUAL': 0}
        results.sort(
            key=lambda x: (
                x.get('similarity', 0),
                x.get('confidence', 0),
                claim_type_priority.get(x.get('claim_type', 'CONTEXTUAL'), 0)
            ),
            reverse=True
        )
        
        logger.info(f"Merged to {len(results)} unique claims")
        return results

    def _format_context_for_llm(self, claims: List[Dict], query: str) -> str:
        """
        Format claims into context for LLM synthesis
        
        FIXED: Handles None values for confidence and similarity
        """

        if not claims:
            return f"No relevant claims found for query: {query}"

        context = f"Query: {query}\n\n"
        context += f"Found {len(claims)} relevant claims:\n\n"

        for i, claim in enumerate(claims[:20], 1):  # Limit to top 20
            # Safely get values with defaults - CRITICAL FIX: Handle None
            claim_text = claim.get('text', 'No text available')
            claim_type = claim.get('claim_type', 'UNKNOWN')
            confidence = claim.get('confidence') or 0.0  # ← FIX: Returns 0.0 if None
            similarity = claim.get('similarity') or 0.0  # ← FIX: Returns 0.0 if None
            source = claim.get('source', 'Unknown source')
            entities = claim.get('entities', [])

            # Format safely
            context += f"[{i}] TYPE: {claim_type}\n"
            context += f"    TEXT: {claim_text}\n"
            context += f"    CONFIDENCE: {confidence:.2f}\n"
            context += f"    SIMILARITY: {similarity:.2f}\n"
            context += f"    SOURCE: {source}\n"

            if entities:
                context += f"    ENTITIES: {', '.join(str(e) for e in entities)}\n"

            context += "\n"

        return context

    def synthesize_with_llm(self, query: str, context: str) -> Dict:
        """
        Send context to LLM for synthesis
        """
        
        prompt = f"""You are an expert research analyst examining claims from a knowledge graph about: {query}

The knowledge graph contains different types of claims:
- PRIMARY: Original research, foundational theories, first-hand observations
- SECONDARY: Derived findings, applications, interpretations
- META: Claims about other claims, institutional responses, dismissals
- CONTEXTUAL: Background information, supporting facts

Below is the context from the knowledge graph. Analyze it and provide:

1. **Key Findings**: What are the main substantive claims? (2-3 paragraphs)
2. **Trust Assessment**: What is the confidence level of the evidence? Note any PRIMARY research vs META dismissals.
3. **Narrative Patterns**: Note any conflicting viewpoints, contested claims, or asymmetries in how different sources treat this topic. Do NOT make conclusions about suppression or coordination - that requires separate detection analysis.
4. **Summary**: A concise synthesis (1 paragraph)

Be objective. Cite evidence. Note where claims conflict.

CONTEXT:
{context}

Provide your analysis:"""
        
        logger.info(f"Sending to {self.ollama_model} for synthesis...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temp for more factual synthesis
                        "num_predict": 2000  # Allow 1-2 pages
                    }
                },
                timeout=180  # 3 minute timeout for longer synthesis
            )
            
            if response.status_code == 200:
                result = response.json()
                synthesis = result.get('response', '')
                logger.info(f"LLM synthesis complete ({len(synthesis)} chars)")
                return {
                    'success': True,
                    'synthesis': synthesis,
                    'model': self.ollama_model
                }
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return {
                    'success': False,
                    'error': f"Ollama returned status {response.status_code}"
                }
        
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return {
                'success': False,
                'error': "LLM synthesis timed out after 3 minutes"
            }
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search(self, query: str, limit: int = 100) -> Dict:
        """
        Main search method - combines everything
        Now includes entity alias search via SAME_AS traversal
        """
        
        logger.info(f"Pattern search: '{query}'")
        
        # 1. Search Neo4j (text match)
        neo4j_results = self.search_neo4j(query, limit=limit)
        
        # 2. Search embeddings (semantic similarity)
        embedding_results = self.search_embeddings(query, limit=limit, min_similarity=0.3)
        
        # 3. Search entity aliases (SAME_AS traversal) - NEW
        entity_results = self.search_entity_aliases(query, limit=limit)
        logger.info(f"Entity alias search added {len(entity_results)} claims")
        
        # 4. Merge all results (deduplicate by ID)
        merged_claims = self.merge_results(neo4j_results, embedding_results)
        
        # Add entity results (deduplicate)
        existing_ids = {c.get('id') for c in merged_claims}
        for claim in entity_results:
            if claim.get('id') not in existing_ids:
                merged_claims.append(claim)
                existing_ids.add(claim.get('id'))
        
        logger.info(f"Total merged claims after entity search: {len(merged_claims)}")
        
        if not merged_claims:
            return {
                'query': query,
                'synthesis': f"No results found for '{query}'. Try broader search terms.",
                'claims': [],
                'total_claims': 0,
                'synthesis_success': False
            }
        
        # 4. Format context - FIXED: Now calls _format_context_for_llm correctly
        context = self._format_context_for_llm(merged_claims, query)
        
        # 5. LLM synthesis
        synthesis_result = self.synthesize_with_llm(query, context)
        
        # 6. Return complete result
        return {
            'query': query,
            'total_claims': len(merged_claims),
            'claims': merged_claims[:limit],  # Return top 20 for display
            'synthesis': synthesis_result.get('synthesis', ''),
            'synthesis_success': synthesis_result.get('success', False),
            'synthesis_error': synthesis_result.get('error'),
            'model_used': self.ollama_model,
            'search_stats': {
                'neo4j_matches': len(neo4j_results),
                'embedding_matches': len(embedding_results),
                'merged_total': len(merged_claims)
            }
        }
    
    def close(self):
        """Close connections"""
        self.neo4j_driver.close()
        self.pg_conn.close()


# Test function
if __name__ == "__main__":
    
    logger.info("=" * 70)
    logger.info("Testing Pattern Search with LLM")
    logger.info("=" * 70)
    
    searcher = PatternSearchLLM()
    
    try:
        # Test query
        test_query = "Smedley Butler testimony"
        logger.info(f"\nTest Query: {test_query}")
        logger.info("=" * 70)
        
        result = searcher.search(test_query)
        
        logger.info(f"\nResults:")
        logger.info(f"  Total claims found: {result['total_claims']}")
        logger.info(f"  Neo4j matches: {result['search_stats']['neo4j_matches']}")
        logger.info(f"  Embedding matches: {result['search_stats']['embedding_matches']}")
        logger.info(f"  Synthesis success: {result['synthesis_success']}")
        
        if result['synthesis_success']:
            logger.info(f"\nSYNTHESIS:\n{result['synthesis']}")
        else:
            logger.error(f"\nSynthesis failed: {result.get('synthesis_error')}")
        
        logger.info(f"\nTop 5 claims:")
        for i, claim in enumerate(result['claims'][:5], 1):
            conf = claim.get('confidence') or 0.0
            sim = claim.get('similarity') or 0.0
            logger.info(
                f"{i}. [{claim['claim_type']}] {claim['text'][:80]}... "
                f"(conf: {conf:.2f}, sim: {sim:.2f})"
            )
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        searcher.close()
    
    logger.info("\n✅ Done!")
