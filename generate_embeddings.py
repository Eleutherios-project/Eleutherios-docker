#!/usr/bin/env python3
"""
Generate embeddings for NEW claims in Neo4j and store in PostgreSQL

This script:
1. Checks which claims already have embeddings in PostgreSQL
2. Fetches only NEW claims from Neo4j (not already in PostgreSQL)
3. Generates embeddings using sentence-transformers
4. Stores embeddings in PostgreSQL with pgvector
5. Creates index for fast similarity search

v2 Changes:
- Only processes claims that don't already have embeddings (incremental mode)
- Adds --force flag to regenerate all embeddings if needed
- Adds --batch-size flag for tuning
- Better progress reporting
"""

import logging
import time
import argparse
from typing import List, Dict, Set
import psycopg2
from psycopg2.extras import execute_batch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from aegis_config (supports environment variables)
from aegis_config import (Config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                          POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB)
# Configuration
# LEGACY: NEO4J_URI = "bolt://localhost:7687"
# LEGACY: NEO4J_USER = "neo4j"
# LEGACY: NEO4J_PASSWORD = "aegistrusted"
# LEGACY: 
# LEGACY: POSTGRES_HOST = "localhost"
# LEGACY: POSTGRES_DB = "aegis_insight"
# LEGACY: POSTGRES_USER = "aegis"
# LEGACY: POSTGRES_PASSWORD = "aegis_trusted_2025"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast


class EmbeddingGenerator:
    """Generate and store embeddings for claims"""
    
    def __init__(self, batch_size: int = 100):
        """Initialize connections and model"""
        
        self.batch_size = batch_size
        
        # Neo4j connection
        logger.info("Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        # PostgreSQL connection
        logger.info("Connecting to PostgreSQL...")
        self.pg_conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Initialize database schema
        self._init_database()
    
    def _init_database(self):
        """Create tables and indexes if they don't exist"""
        
        with self.pg_conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create embeddings table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS claim_embeddings (
                    claim_id TEXT PRIMARY KEY,
                    claim_text TEXT NOT NULL,
                    embedding vector({self.embedding_dim}),
                    confidence FLOAT,
                    claim_type TEXT,
                    source_file TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create index for similarity search (if not exists)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS claim_embeddings_idx 
                ON claim_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            self.pg_conn.commit()
        
        logger.info("Database schema ready")
    
    def get_existing_claim_ids(self) -> Set[str]:
        """Get set of claim IDs that already have embeddings"""
        
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT claim_id FROM claim_embeddings")
            existing = {row[0] for row in cur.fetchall()}
        
        return existing
    
    def get_claims_from_neo4j(self, exclude_ids: Set[str] = None) -> List[Dict]:
        """Fetch claims from Neo4j, optionally excluding IDs that already exist"""
        
        logger.info("Fetching claims from Neo4j...")
        
        query = """
        MATCH (c:Claim)
        RETURN 
            c.claim_id as claim_id,
            c.claim_text as claim_text,
            c.confidence as confidence,
            c.claim_type as claim_type,
            c.source_file as source_file
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(query)
            all_claims = [dict(record) for record in result]
        
        logger.info(f"Found {len(all_claims)} total claims in Neo4j")
        
        # Filter out claims that already have embeddings
        if exclude_ids:
            claims = [c for c in all_claims if c['claim_id'] not in exclude_ids]
            logger.info(f"After excluding existing: {len(claims)} new claims to process")
        else:
            claims = all_claims
        
        return claims
    
    def generate_embeddings_batch(self, claims: List[Dict]) -> List[np.ndarray]:
        """Generate embeddings for a batch of claims"""
        
        texts = [claim['claim_text'] or '' for claim in claims]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def store_embeddings(self, claims: List[Dict], embeddings: List[np.ndarray]):
        """Store embeddings in PostgreSQL"""
        
        data = [
            (
                claim['claim_id'],
                claim['claim_text'] or '',
                embeddings[i].tolist(),
                float(claim.get('confidence') or 0.5),
                claim.get('claim_type') or 'CONTEXTUAL',
                claim.get('source_file') or 'unknown'
            )
            for i, claim in enumerate(claims)
        ]
        
        with self.pg_conn.cursor() as cur:
            execute_batch(
                cur,
                """
                INSERT INTO claim_embeddings 
                    (claim_id, claim_text, embedding, confidence, claim_type, source_file)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (claim_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    confidence = EXCLUDED.confidence,
                    claim_type = EXCLUDED.claim_type
                """,
                data,
                page_size=100
            )
            self.pg_conn.commit()
    
    def get_stats(self) -> Dict:
        """Get current embedding statistics"""
        
        stats = {}
        
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM claim_embeddings")
            stats['postgres_count'] = cur.fetchone()[0]
        
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (c:Claim) RETURN count(c) as count")
            stats['neo4j_count'] = result.single()['count']
        
        stats['missing'] = stats['neo4j_count'] - stats['postgres_count']
        
        return stats
    
    def process_incremental(self):
        """Process only NEW claims (incremental mode)"""
        
        start_time = time.time()
        
        # Get existing claim IDs from PostgreSQL
        logger.info("Checking existing embeddings...")
        existing_ids = self.get_existing_claim_ids()
        logger.info(f"Found {len(existing_ids)} existing embeddings in PostgreSQL")
        
        # Get only new claims from Neo4j
        claims = self.get_claims_from_neo4j(exclude_ids=existing_ids)
        total_claims = len(claims)
        
        if total_claims == 0:
            logger.info("✅ No new claims to process - embeddings are up to date!")
            return 0
        
        logger.info(f"Processing {total_claims} NEW claims in batches of {self.batch_size}...")
        
        # Process in batches
        processed = 0
        for i in range(0, total_claims, self.batch_size):
            batch = claims[i:i + self.batch_size]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(batch)
            
            # Store in PostgreSQL
            self.store_embeddings(batch, embeddings)
            
            processed += len(batch)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_claims - processed) / rate if rate > 0 else 0
            
            logger.info(
                f"Processed {processed}/{total_claims} new claims "
                f"({100*processed/total_claims:.1f}%) | "
                f"Rate: {rate:.1f}/sec | "
                f"ETA: {eta:.0f}s"
            )
        
        total_time = time.time() - start_time
        logger.info(f"✅ Complete! Processed {total_claims} new claims in {total_time:.1f}s")
        
        return total_claims
    
    def process_all(self):
        """Process ALL claims (regenerate everything)"""
        
        start_time = time.time()
        
        # Clear existing embeddings
        logger.warning("Force mode: Clearing all existing embeddings...")
        with self.pg_conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE claim_embeddings")
            self.pg_conn.commit()
        
        # Get all claims from Neo4j
        claims = self.get_claims_from_neo4j(exclude_ids=None)
        total_claims = len(claims)
        
        if total_claims == 0:
            logger.warning("No claims found in Neo4j!")
            return 0
        
        logger.info(f"Processing {total_claims} claims in batches of {self.batch_size}...")
        
        # Process in batches
        processed = 0
        for i in range(0, total_claims, self.batch_size):
            batch = claims[i:i + self.batch_size]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(batch)
            
            # Store in PostgreSQL
            self.store_embeddings(batch, embeddings)
            
            processed += len(batch)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_claims - processed) / rate if rate > 0 else 0
            
            logger.info(
                f"Processed {processed}/{total_claims} claims "
                f"({100*processed/total_claims:.1f}%) | "
                f"Rate: {rate:.1f}/sec | "
                f"ETA: {eta/60:.1f}min"
            )
        
        total_time = time.time() - start_time
        logger.info(f"✅ Complete! Processed {total_claims} claims in {total_time/60:.1f} minutes")
        
        return total_claims
    
    def test_similarity_search(self, query: str, top_k: int = 5):
        """Test similarity search"""
        
        logger.info(f"\nTesting search: '{query}'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    claim_id,
                    claim_text,
                    confidence,
                    claim_type,
                    1 - (embedding <=> %s::vector) as similarity
                FROM claim_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            results = cur.fetchall()
        
        for i, (cid, text, conf, ctype, sim) in enumerate(results, 1):
            text_preview = (text[:80] + '...') if len(text) > 80 else text
            logger.info(f"  {i}. [{ctype}] {text_preview} (sim: {sim:.3f})")
    
    def close(self):
        """Close connections"""
        self.neo4j_driver.close()
        self.pg_conn.close()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Generate embeddings for Aegis claims')
    parser.add_argument('--force', action='store_true', 
                        help='Regenerate ALL embeddings (default: incremental)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics only, do not process')
    parser.add_argument('--test', type=str, default=None,
                        help='Test similarity search with given query')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Aegis Insight - Embedding Generation v2")
    logger.info("=" * 60)
    
    generator = EmbeddingGenerator(batch_size=args.batch_size)
    
    try:
        # Show stats
        stats = generator.get_stats()
        logger.info(f"Neo4j claims:      {stats['neo4j_count']:,}")
        logger.info(f"PostgreSQL embeds: {stats['postgres_count']:,}")
        logger.info(f"Missing:           {stats['missing']:,}")
        
        if args.stats:
            # Stats only mode
            pass
        elif args.test:
            # Test mode
            generator.test_similarity_search(args.test, top_k=5)
        elif args.force:
            # Force regenerate all
            generator.process_all()
        else:
            # Incremental mode (default)
            generator.process_incremental()
        
        # Show final stats
        final_stats = generator.get_stats()
        logger.info(f"\nFinal: {final_stats['postgres_count']:,} embeddings stored")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        generator.close()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
