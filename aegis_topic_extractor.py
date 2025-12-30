#!/usr/bin/env python3
"""
Aegis Topic Extractor
Post-extraction topic summarization with embedding-based query expansion.

Extracts subject-matter topics from documents after claim extraction is complete,
enabling Topic-based traversal for detection queries.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import ollama
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TopicExtractor:
    """
    Extracts and manages document topics with embedding support.
    
    Usage:
        extractor = TopicExtractor(neo4j_driver, pg_conn)
        topics = extractor.extract_topics_for_document("doc_123")
    """
    
    def __init__(self, neo4j_driver, pg_conn, ollama_model: str = "mistral-nemo:12b"):
        """
        Initialize the topic extractor.
        
        Args:
            neo4j_driver: Neo4j driver instance
            pg_conn: PostgreSQL connection (psycopg2)
            ollama_model: LLM model for topic extraction
        """
        self.driver = neo4j_driver
        self.pg_conn = pg_conn
        self.llm_model = ollama_model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        logger.info(f"TopicExtractor initialized with model: {ollama_model}")
    
    def extract_topics_for_document(self, doc_id: str) -> Dict:
        """
        Main entry point - extract topics for a document after extraction complete.
        
        Args:
            doc_id: Document identifier (doc_id or source_file fragment)
            
        Returns:
            Dict with 'topics' list or 'primary'/'secondary' lists for large docs
        """
        with self.driver.session() as session:
            # Get document metadata
            doc_info = self._get_document_info(session, doc_id)
            chunk_count = doc_info['chunk_count']
            
            if chunk_count == 0:
                logger.warning(f"No chunks found for document: {doc_id}")
                return {"topics": []}
            
            logger.info(f"Extracting topics for {doc_info['source_file']} ({chunk_count} chunks)")
            
            # Get extracted data using resolved source_file
            source_file = doc_info['source_file']
            claims = self._get_sample_claims(session, source_file, limit=30)
            entities = self._get_top_entities(session, source_file, limit=20)
            
            if not claims:
                logger.warning(f"No claims found for document: {doc_id}")
                return {"topics": []}
            
            # Determine structure based on document size
            is_hierarchical = chunk_count >= 50
            
            # Generate topics via LLM
            topics = self._llm_extract_topics(
                source_file=doc_info['source_file'],
                chunk_count=chunk_count,
                claims=claims,
                entities=entities,
                hierarchical=is_hierarchical
            )
            
            # Create Topic nodes with embeddings and link to document
            self._create_topic_links(session, doc_info['source_file'], topics, is_hierarchical)
            
            topic_count = len(topics.get('topics', [])) + \
                         len(topics.get('primary', [])) + \
                         len(topics.get('secondary', []))
            logger.info(f"Created {topic_count} topics for {doc_info['source_file']}")
            
            return topics
    
    def _get_document_info(self, session, doc_id: str) -> Dict:
        """Get document metadata and chunk count."""
        result = session.run("""
            MATCH (d:Document)
            WHERE d.source_file CONTAINS $doc_id
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            WITH d, count(c) as chunk_count
            WHERE chunk_count > 0
            WITH d, chunk_count
            ORDER BY chunk_count DESC
            RETURN d.source_file as source_file, chunk_count
            LIMIT 1
        """, doc_id=doc_id)
        record = result.single()
        if record:
            return dict(record)
        return {'source_file': doc_id, 'doc_id': doc_id, 'chunk_count': 0}
    
    def _get_sample_claims(self, session, doc_id: str, limit: int = 30) -> List[str]:
        """Get representative claims from document, sorted by confidence."""
        result = session.run("""
            MATCH (d:Document)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WHERE d.source_file = $doc_id
            RETURN c.claim_text as text, c.confidence as confidence
            ORDER BY c.confidence DESC
            LIMIT $limit
        """, doc_id=doc_id, limit=limit)
        return [r['text'] for r in result if r['text']]
    
    def _get_top_entities(self, session, doc_id: str, limit: int = 20) -> List[str]:
        """Get most frequently mentioned entities in document."""
        result = session.run("""
            MATCH (d:Document)-[:CONTAINS]->(ch:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE d.source_file = $doc_id
            RETURN e.name as name, count(*) as mentions
            ORDER BY mentions DESC
            LIMIT $limit
        """, doc_id=doc_id, limit=limit)
        return [r['name'] for r in result if r['name']]
    
    def _llm_extract_topics(self, source_file: str, chunk_count: int,
                           claims: List[str], entities: List[str],
                           hierarchical: bool) -> Dict:
        """Use LLM to extract topics from document content."""
        
        # Determine topic counts and structure based on document size
        if chunk_count < 10:
            topic_instruction = "Return 5-10 topics."
            format_spec = '{"topics": ["topic1", "topic2", ...]}'
        elif chunk_count < 50:
            topic_instruction = "Return 10-15 topics."
            format_spec = '{"topics": ["topic1", "topic2", ...]}'
        else:
            topic_instruction = """Return topics in two tiers:
- primary: 8-12 main subjects (what the document is fundamentally about)
- secondary: 10-15 supporting topics (specific concepts, places, people discussed)"""
            format_spec = '{"primary": ["topic1", "topic2", ...], "secondary": ["topic1", "topic2", ...]}'
        
        # Format claims for prompt
        claims_text = "\n".join(f"- {c}" for c in claims[:25])
        entities_text = ", ".join(entities[:15]) if entities else "None identified"
        
        prompt = f"""Extract SEARCHABLE topics from this document.

Document: {source_file}
Sections: {chunk_count}

Entities mentioned: {entities_text}

Sample claims:
{claims_text}

Extract TWO types of topics:

TYPE 1 - SUBJECT MATTER (what is being discussed):
- Objects/structures: e.g., "temple", "monument", "artifact"  
- Phenomena/concepts: e.g., "climate change", "migration", "trade"
- Fields: e.g., "geology", "astronomy" (but only if central to content)

TYPE 2 - NAMED ENTITIES (specific instances):
- Places: specific locations mentioned
- Events: specific named events
- Time periods: specific eras mentioned

RULES:
- Lowercase all terms
- One concept per topic (no compound terms)
- Include BOTH general subject terms AND specific names
- Skip author names and meta-terms like "book", "research", "study"

{topic_instruction}

Return ONLY valid JSON:
{format_spec}"""

        try:
            response = ollama.generate(model=self.llm_model, prompt=prompt)
            text = response['response'].strip()
            
            # Extract JSON from response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                logger.debug(f"LLM extracted topics: {parsed}")
                return parsed
        except Exception as e:
            logger.error(f"LLM topic extraction failed: {e}")
        
        return {"topics": []}
    
    def _normalize_topic(self, raw: str) -> str:
        """
        Light normalization - preserve specificity.
        Only handles case and obvious plurals.
        """
        if not raw:
            return ""
        
        normalized = raw.lower().strip()
        
        # Remove leading/trailing punctuation
        normalized = normalized.strip('.,;:!?"\'')
        
        # Handle obvious plurals only
        if normalized.endswith('ies') and len(normalized) > 4:
            normalized = normalized[:-3] + 'y'  # "theories" → "theory"
        elif normalized.endswith('es') and len(normalized) > 4:
            # Don't change "processes" → "process" etc., too aggressive
            pass
        elif normalized.endswith('s') and not normalized.endswith('ss') and len(normalized) > 3:
            # Simple plural: "pyramids" → "pyramid"
            normalized = normalized[:-1]
        
        return normalized
    
    def _create_topic_with_embedding(self, session, name: str, category: str = None) -> bool:
        """
        Create Topic node in Neo4j and store embedding in PostgreSQL.
        
        Returns True if new topic created, False if already existed.
        """
        if not name:
            return False
        
        # Create/merge in Neo4j
        result = session.run("""
            MERGE (t:Topic {name: $name})
            ON CREATE SET t.category = $category, t.created_at = datetime()
            ON MATCH SET t.category = COALESCE(t.category, $category)
            RETURN t.name as name, t.category as category
        """, name=name, category=category)
        
        # Store embedding in PostgreSQL
        try:
            embedding = self.embedding_model.encode(name).tolist()
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                INSERT INTO topic_embeddings (topic_name, embedding)
                VALUES (%s, %s)
                ON CONFLICT (topic_name) DO NOTHING
            """, (name, embedding))
            self.pg_conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store topic embedding for '{name}': {e}")
            return False
    
    def _create_topic_links(self, session, doc_id: str, topics: Dict, hierarchical: bool):
        """Create Topic nodes and link to document."""
        
        if hierarchical and 'primary' in topics:
            # Large document with hierarchical topics
            for topic in topics.get('primary', []):
                normalized = self._normalize_topic(topic)
                if normalized:
                    self._create_topic_with_embedding(session, normalized, 'primary')
                    session.run("""
                        MATCH (t:Topic {name: $name})
                        MATCH (d:Document {source_file: $doc_id})
                        MERGE (d)-[:ABOUT]->(t)
                    """, name=normalized, doc_id=doc_id)
            
            for topic in topics.get('secondary', []):
                normalized = self._normalize_topic(topic)
                if normalized:
                    self._create_topic_with_embedding(session, normalized, 'secondary')
                    session.run("""
                        MATCH (t:Topic {name: $name})
                        MATCH (d:Document {source_file: $doc_id})
                        MERGE (d)-[:ABOUT]->(t)
                    """, name=normalized, doc_id=doc_id)
        else:
            # Small/medium document with flat topic list
            for topic in topics.get('topics', []):
                normalized = self._normalize_topic(topic)
                if normalized:
                    self._create_topic_with_embedding(session, normalized, None)
                    session.run("""
                        MATCH (t:Topic {name: $name})
                        MATCH (d:Document {source_file: $doc_id})
                        MERGE (d)-[:ABOUT]->(t)
                    """, name=normalized, doc_id=doc_id)


def get_topics_for_query(query: str, pg_conn, model=None, threshold: float = 0.80) -> List[Tuple[str, float]]:
    """
    Query-time expansion - find Topics similar to search query using embeddings.
    
    Call this from detectors before Topic traversal to expand the query
    to semantically similar topics.
    
    Args:
        query: Search query string
        pg_conn: PostgreSQL connection
        model: SentenceTransformer model (loads default if None)
        threshold: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        List of (topic_name, similarity_score) tuples, sorted by similarity
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    
    query_embedding = model.encode(query).tolist()
    
    try:
        cursor = pg_conn.cursor()
        cursor.execute("""
            SELECT topic_name, 1 - (embedding <=> %s::vector) as similarity
            FROM topic_embeddings
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT 20
        """, (query_embedding, query_embedding, threshold))
        
        results = [(row[0], round(row[1], 3)) for row in cursor.fetchall()]
        return results
    except Exception as e:
        logger.error(f"Topic query expansion failed: {e}")
        return []


def get_topic_names_for_query(query: str, pg_conn, model=None, threshold: float = 0.80) -> List[str]:
    """
    Convenience wrapper - returns just topic names without scores.
    """
    results = get_topics_for_query(query, pg_conn, model, threshold)
    return [name for name, score in results]


# CLI for testing
if __name__ == "__main__":
    import argparse
    import psycopg2
    from neo4j import GraphDatabase
    
    parser = argparse.ArgumentParser(description="Extract topics for a document")
    parser.add_argument("doc_id", help="Document ID or source file name")
    parser.add_argument("--model", default="mistral-nemo:12b", help="Ollama model")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Connect to databases
    neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "aegistrusted"))
    pg_conn = psycopg2.connect(
        host="localhost",
        database="aegis_insight",
        user="aegis",
        password="aegis_trusted_2025"
    )
    
    try:
        extractor = TopicExtractor(neo4j_driver, pg_conn, ollama_model=args.model)
        topics = extractor.extract_topics_for_document(args.doc_id)
        
        print("\n" + "="*60)
        print("EXTRACTED TOPICS")
        print("="*60)
        
        if 'primary' in topics:
            print("\nPrimary Topics:")
            for t in topics.get('primary', []):
                print(f"  • {t}")
            print("\nSecondary Topics:")
            for t in topics.get('secondary', []):
                print(f"  • {t}")
        else:
            print("\nTopics:")
            for t in topics.get('topics', []):
                print(f"  • {t}")
        
        print("="*60)
    finally:
        neo4j_driver.close()
        pg_conn.close()
