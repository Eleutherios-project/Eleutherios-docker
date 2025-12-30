import os
"""
Shared topic expansion utilities for Aegis detectors.
Provides embedding-based topic expansion for semantic search.
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Cache model to avoid reloading
_embedding_model = None
_pg_connection_params = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "database": "aegis_insight", 
    "user": "aegis",
    "password": "aegis_trusted_2025"
}

def get_embedding_model():
    """Get or create cached embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    return _embedding_model


def expand_topics(pattern: str, threshold: float = 0.40, limit: int = 20) -> List[str]:
    """
    Expand a search pattern to semantically similar topic names.
    
    Args:
        pattern: Search term to expand
        threshold: Minimum similarity (0.0-1.0), default 0.40
        limit: Maximum topics to return
        
    Returns:
        List of topic names matching the pattern
    """
    try:
        import psycopg2
        
        model = get_embedding_model()
        query_emb = model.encode(pattern).tolist()
        
        conn = psycopg2.connect(**_pg_connection_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT topic_name, 1 - (embedding <=> %s::vector) as similarity
            FROM topic_embeddings
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_emb, query_emb, threshold, limit))
        
        topics = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if topics:
            logger.info(f"Topic expansion: '{pattern}' -> {len(topics)} topics: {topics[:5]}{'...' if len(topics) > 5 else ''}")
        
        return topics
        
    except Exception as e:
        logger.warning(f"Topic expansion failed: {e}")
        return []


def expand_topics_with_scores(pattern: str, threshold: float = 0.40, limit: int = 20) -> List[Tuple[str, float]]:
    """
    Expand a search pattern, returning topics with similarity scores.
    
    Returns:
        List of (topic_name, similarity_score) tuples
    """
    try:
        import psycopg2
        
        model = get_embedding_model()
        query_emb = model.encode(pattern).tolist()
        
        conn = psycopg2.connect(**_pg_connection_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT topic_name, 1 - (embedding <=> %s::vector) as similarity
            FROM topic_embeddings
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_emb, query_emb, threshold, limit))
        
        results = [(row[0], round(row[1], 3)) for row in cursor.fetchall()]
        conn.close()
        
        return results
        
    except Exception as e:
        logger.warning(f"Topic expansion failed: {e}")
        return []


def get_claims_via_topics(neo4j_session, pattern: str, 
                          threshold: float = 0.40,
                          limit: int = 500,
                          require_geographic: bool = False,
                          require_temporal: bool = False) -> List[dict]:
    """
    Get claims by expanding pattern to topics, then traversing the graph.
    
    This is the main entry point for detectors to get semantically relevant claims.
    
    Args:
        neo4j_session: Active Neo4j session
        pattern: Search pattern to expand
        threshold: Topic similarity threshold
        limit: Maximum claims to return
        require_geographic: Only return claims with geographic data
        require_temporal: Only return claims with temporal data
        
    Returns:
        List of claim dictionaries
    """
    import json
    
    # Get expanded topics
    expanded_topics = expand_topics(pattern, threshold)
    
    if not expanded_topics:
        logger.info(f"No topics found for '{pattern}', using text fallback")
        return []
    
    # Build query with optional filters
    where_clauses = ["c.claim_text IS NOT NULL"]
    
    if require_geographic:
        where_clauses.append("c.geographic_data IS NOT NULL AND c.geographic_data <> '{}' AND c.geographic_data <> ''")
    
    if require_temporal:
        where_clauses.append("c.temporal_data IS NOT NULL AND c.temporal_data <> '{}' AND c.temporal_data <> ''")
    
    where_clause = " AND ".join(where_clauses)
    
    query = f"""
        MATCH (t:Topic)
        WHERE t.name IN $topic_names
        MATCH (t)<-[:ABOUT]-(d:Document)
        MATCH (c:Claim) WHERE c.source_file = d.source_file
        WHERE {where_clause}
        RETURN DISTINCT
               c.claim_id as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.source_file as source_file,
               c.geographic_data as geographic_data,
               c.temporal_data as temporal_data,
               c.emotional_data as emotional_data,
               c.authority_data as authority_data,
               d.source_file as doc_source,
               elementId(c) as element_id
        ORDER BY c.confidence DESC
        LIMIT $limit
    """
    
    claims = []
    try:
        result = neo4j_session.run(query, topic_names=expanded_topics, limit=limit)
        
        for record in result:
            claim = dict(record)
            
            # Use doc_source as fallback for source_file
            if not claim.get('source_file'):
                claim['source_file'] = claim.get('doc_source')
            
            # Parse geographic_data JSON
            if claim.get('geographic_data'):
                try:
                    if isinstance(claim['geographic_data'], str):
                        claim['geographic_data'] = json.loads(claim['geographic_data'])
                except:
                    claim['geographic_data'] = {}
            
            # Parse temporal_data JSON
            if claim.get('temporal_data'):
                try:
                    if isinstance(claim['temporal_data'], str):
                        claim['temporal_data'] = json.loads(claim['temporal_data'])
                except:
                    claim['temporal_data'] = {}
            
            claims.append(claim)
        
        logger.info(f"Topic traversal found {len(claims)} claims for '{pattern}'")
        
    except Exception as e:
        logger.error(f"Topic traversal query failed: {e}")
    
    return claims


def get_claims_via_topics_hybrid(neo4j_session, pattern: str, 
                                  threshold: float = 0.40,
                                  limit: int = 500,
                                  require_text_match: bool = True) -> list:
    """
    Get claims by topic expansion with optional text filtering.
    
    When require_text_match=True (default), claims must ALSO contain
    at least one significant word from the pattern. This prevents
    overly broad topic matches from returning irrelevant results.
    """
    import json
    import re
    
    # Get expanded topics
    expanded_topics = expand_topics(pattern, threshold)
    
    if not expanded_topics:
        logger.info(f"No topics found for '{pattern}'")
        return []
    
    # Extract significant words (3+ chars, not common words)
    stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was', 'were'}
    pattern_words = [w.lower() for w in re.findall(r'\w+', pattern) 
                     if len(w) >= 3 and w.lower() not in stop_words]
    
    # Build query - topic traversal with optional text filter
    if require_text_match and pattern_words:
        # Use case-insensitive regex with word boundaries
        # Neo4j regex: (?i) for case-insensitive, \\b for word boundary
        text_conditions = " OR ".join([f"c.claim_text =~ '(?i).*\\\\b{w}\\\\b.*'" for w in pattern_words])
        query = f"""
            MATCH (t:Topic)
            WHERE t.name IN $topic_names
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (c:Claim) WHERE c.source_file = d.source_file
            AND ({text_conditions})
            RETURN DISTINCT
                   c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.geographic_data as geographic_data,
                   c.temporal_data as temporal_data,
               c.emotional_data as emotional_data,
               c.authority_data as authority_data,
                   d.source_file as doc_source
            ORDER BY c.confidence DESC
            LIMIT $limit
        """
    else:
        query = """
            MATCH (t:Topic)
            WHERE t.name IN $topic_names
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (c:Claim) WHERE c.source_file = d.source_file
            RETURN DISTINCT
                   c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.geographic_data as geographic_data,
                   c.temporal_data as temporal_data,
               c.emotional_data as emotional_data,
               c.authority_data as authority_data,
                   d.source_file as doc_source
            ORDER BY c.confidence DESC
            LIMIT $limit
        """
    
    claims = []
    try:
        result = neo4j_session.run(query, topic_names=expanded_topics, limit=limit)
        
        for record in result:
            claim = dict(record)
            if not claim.get('source_file'):
                claim['source_file'] = claim.get('doc_source')
            
            # Parse JSON fields
            for field in ['geographic_data', 'temporal_data']:
                if claim.get(field):
                    try:
                        if isinstance(claim[field], str):
                            claim[field] = json.loads(claim[field])
                    except:
                        claim[field] = {}
            
            claims.append(claim)
        
        logger.info(f"Hybrid topic search found {len(claims)} claims for '{pattern}' (text_match={require_text_match})")
        
    except Exception as e:
        logger.error(f"Hybrid topic search failed: {e}")
    
    return claims
