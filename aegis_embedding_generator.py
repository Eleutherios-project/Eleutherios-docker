"""
Aegis Insight - Embedding Generator
Generates embeddings for all chunks/claims using sentence-transformers
Stores in PostgreSQL with pgvector for similarity search
"""

from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
import torch
import numpy as np
from typing import List, Dict, Optional
import logging
import time
from pathlib import Path

class EmbeddingGenerator:
    """
    Generate embeddings for text chunks and claims
    Priority: Run first to enable deduplication and matching
    """
    
    def __init__(self, 
                 config: Dict,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Embedding configuration
        embed_config = config['embedding']
        self.model_name = embed_config['model_name']
        self.dimensions = embed_config['dimensions']
        self.batch_size = embed_config['batch_size']
        
        # GPU configuration
        gpu_config = config['gpu']
        self.device = embed_config['device']
        
        # Check if specified GPU is available
        if 'cuda' in self.device:
            gpu_num = int(self.device.split(':')[1])
            if gpu_num >= torch.cuda.device_count():
                self.logger.warning(f"GPU {gpu_num} not available, falling back to cuda:0")
                self.device = 'cuda:0'
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            self.device = 'cpu'
        
        # Load model
        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.logger.info(f"Target device: {self.device}")
        
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        self.model = self.model.to(self.device)
        
        self.logger.info(f"✓ Model loaded on {self.device}")
        self.logger.info(f"✓ Embedding dimensions: {self.dimensions}")
        
        # PostgreSQL connection
        pg_config = config['databases']['postgresql']
        self.pg_conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['user'],
            password=pg_config['password']
        )
        self.logger.info(f"✓ Connected to PostgreSQL: {pg_config['database']}")
        
        # Stats
        self.stats = {
            'total_generated': 0,
            'total_stored': 0,
            'total_time': 0.0,
            'avg_batch_time': 0.0
        }
    
    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings (batch_size, dimensions)
        """
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=len(texts),  # Process entire batch at once
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
        
        return embeddings
    
    def store_embeddings(self,
                        embedding_ids: List[str],
                        claim_ids: List[str],
                        chunk_ids: List[str],
                        embeddings: np.ndarray):
        """
        Store embeddings in PostgreSQL
        
        Args:
            embedding_ids: Unique IDs for embeddings
            claim_ids: Associated claim IDs
            chunk_ids: Associated chunk IDs
            embeddings: numpy array of embeddings
        """
        
        cursor = self.pg_conn.cursor()
        
        # Prepare data
        data = [
            (
                eid,
                cid if cid else chid,  # Use chunk_id if no claim_id
                chid,
                emb.tolist(),
                self.model_name
            )
            for eid, cid, chid, emb in zip(
                embedding_ids, claim_ids, chunk_ids, embeddings
            )
        ]
        
        # Batch insert with UPSERT
        execute_values(
            cursor,
            """
            INSERT INTO claim_embeddings 
                (embedding_id, claim_id, chunk_id, embedding, model_version)
            VALUES %s
            ON CONFLICT (embedding_id) DO UPDATE
            SET 
                embedding = EXCLUDED.embedding,
                created_at = NOW()
            """,
            data,
            template="(%s, %s, %s, %s::vector, %s)"
        )
        
        self.pg_conn.commit()
        cursor.close()
        
        self.stats['total_stored'] += len(data)
    
    def find_similar(self, 
                    query_vector: np.ndarray,
                    top_k: int = 100,
                    min_similarity: float = 0.75) -> List[Dict]:
        """
        Find similar embeddings using PostgreSQL function
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of {claim_id, chunk_id, similarity} dicts
        """
        
        cursor = self.pg_conn.cursor()
        
        # Call PostgreSQL similarity function
        cursor.execute(
            """
            SELECT claim_id, chunk_id, similarity
            FROM find_similar_claims(%s::vector, %s, %s)
            """,
            (query_vector.tolist(), top_k, min_similarity)
        )
        
        results = [
            {
                'claim_id': row[0],
                'chunk_id': row[1],
                'similarity': row[2]
            }
            for row in cursor.fetchall()
        ]
        
        cursor.close()
        
        return results
    
    def process_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Process all chunks to generate embeddings
        
        Args:
            chunks: List of chunk dicts with 'chunk_id' and 'text'
            
        Returns:
            Statistics dict
        """
        
        total = len(chunks)
        processed = 0
        start_time = time.time()
        batch_times = []
        
        self.logger.info("="*60)
        self.logger.info("STAGE 1: EMBEDDING GENERATION")
        self.logger.info("="*60)
        self.logger.info(f"\nGenerating embeddings for {total} chunks...")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info("")
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch_start = time.time()
            batch = chunks[i:i + self.batch_size]
            
            # Extract texts (use claim_text if available, else chunk text)
            texts = []
            for c in batch:
                # Priority: claim_text > text > empty string
                text = c.get('claim_text', c.get('text', ''))
                if not text:
                    self.logger.warning(f"Empty text for chunk {c.get('chunk_id')}")
                    text = ""  # Will generate zero vector
                texts.append(text)
            
            # Generate embeddings
            embeddings = self.generate_batch(texts)
            
            # Prepare IDs
            embedding_ids = [c['chunk_id'] for c in batch]
            claim_ids = [c.get('claim_id', None) for c in batch]
            chunk_ids = [c['chunk_id'] for c in batch]
            
            # Store in database
            self.store_embeddings(embedding_ids, claim_ids, chunk_ids, embeddings)
            
            # Update stats
            processed += len(batch)
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Calculate metrics
            pct = (processed / total) * 100
            elapsed = time.time() - start_time
            avg_time = np.mean(batch_times)
            chunks_per_sec = len(batch) / batch_time
            remaining_batches = (total - processed) / self.batch_size
            eta = remaining_batches * avg_time
            
            # Progress update
            self.logger.info(
                f"Embeddings: {processed}/{total} ({pct:.1f}%) | "
                f"Speed: {chunks_per_sec:.1f} chunks/sec | "
                f"Batch: {batch_time:.2f}s | "
                f"ETA: {self._format_time(eta)}"
            )
        
        # Final stats
        total_time = time.time() - start_time
        avg_rate = total / total_time if total_time > 0 else 0
        
        self.stats['total_generated'] = total
        self.stats['total_time'] = total_time
        self.stats['avg_batch_time'] = np.mean(batch_times)
        
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("✓ EMBEDDING GENERATION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total embeddings: {total}")
        self.logger.info(f"Total time: {self._format_time(total_time)}")
        self.logger.info(f"Average rate: {avg_rate:.2f} embeddings/sec")
        self.logger.info(f"Model: {self.model_name} ({self.dimensions}-dim)")
        self.logger.info("")
        
        return {
            'total_processed': total,
            'total_time': total_time,
            'avg_rate': avg_rate,
            'model': self.model_name,
            'dimensions': self.dimensions,
            'device': str(self.device)
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        if seconds < 0:
            return "calculating..."
        
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"
    
    def close(self):
        """Close database connection"""
        if self.pg_conn:
            self.pg_conn.close()
            self.logger.info("PostgreSQL connection closed")

if __name__ == "__main__":
    # Test embedding generator
    import sys
    sys.path.append('/home/claude')
    from aegis_config import CONFIG
    
    # Create test chunks
    test_chunks = [
        {
            'chunk_id': 'test_001',
            'text': 'Ancient civilizations possessed advanced astronomical knowledge.',
            'claim_text': 'Ancient civilizations had sophisticated understanding of celestial mechanics.'
        },
        {
            'chunk_id': 'test_002',
            'text': 'The pyramids demonstrate remarkable engineering precision.',
            'claim_text': 'Pyramid construction shows advanced mathematical knowledge.'
        }
    ]
    
    # Initialize generator
    generator = EmbeddingGenerator(CONFIG)
    
    # Process test chunks
    stats = generator.process_chunks(test_chunks)
    
    print("\n✓ Test complete")
    print(f"Generated {stats['total_processed']} embeddings")
    
    # Test similarity search
    print("\nTesting similarity search...")
    query_text = "ancient astronomy knowledge"
    query_embedding = generator.generate_batch([query_text])[0]
    
    similar = generator.find_similar(query_embedding, top_k=5, min_similarity=0.5)
    
    print(f"Found {len(similar)} similar chunks:")
    for result in similar:
        print(f"  - {result['chunk_id']}: {result['similarity']:.3f}")
    
    generator.close()
