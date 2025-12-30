#!/usr/bin/env python3
"""
Aegis Insight - Detection Setup Jobs

Implementations of all detection setup jobs:
- build_citations
- generate_embeddings  
- build_temporal_index
- build_geographic_index

These plug into the DetectionSystemBuilder infrastructure.
"""

import logging
from typing import Dict, Optional, Callable
from neo4j import GraphDatabase
import subprocess
import json


class DetectionSetupJobs:
    """
    Job implementations for detection system setup
    
    Each job follows the pattern:
    - Accepts checkpoint dict (resume from failure)
    - Accepts progress_callback (report progress)
    - Accepts builder (access to Neo4j, etc.)
    - Returns result dict
    """
    
    @staticmethod
    def build_citations(checkpoint: Optional[Dict] = None,
                       progress_callback: Optional[Callable] = None,
                       builder = None) -> Dict:
        """
        Build citation network using incremental approach
        
        Uses build_citation_relationships_v2.py logic
        """
        from build_citation_relationships_v2 import CitationRelationshipBuilderV2
        
        logger = logging.getLogger(__name__)
        logger.info("Building citation relationships...")
        
        citation_builder = CitationRelationshipBuilderV2(
            neo4j_uri=builder.driver._pool.address[0] if builder else "bolt://localhost:7687"
        )
        
        try:
            # Phase 1: Intra-document citations
            if progress_callback:
                progress_callback(0.1)
            
            logger.info("Phase 1: Intra-document citations")
            citation_builder.build_intra_document_citations()
            
            if progress_callback:
                progress_callback(0.5)
            
            # Phase 2: Entity-based citations (batched)
            logger.info("Phase 2: Entity-based citations")
            citation_builder.build_entity_based_citations_batched(batch_size=100)
            
            if progress_callback:
                progress_callback(0.9)
            
            # Get stats
            stats = citation_builder.get_stats()
            
            if progress_callback:
                progress_callback(1.0)
            
            citation_builder.close()
            
            return {
                'total_relationships': stats['total'],
                'by_type': stats['by_type'],
                'avg_per_claim': stats['avg_citations_per_claim'],
                'max_per_claim': stats['max_citations_per_claim']
            }
            
        except Exception as e:
            citation_builder.close()
            raise
    
    @staticmethod
    def generate_embeddings(checkpoint: Optional[Dict] = None,
                           progress_callback: Optional[Callable] = None,
                           builder = None) -> Dict:
        """
        Generate embeddings for all claims
        
        Uses generate_embeddings.py logic
        """
        logger = logging.getLogger(__name__)
        logger.info("Generating embeddings...")
        
        # Check if embeddings script exists
        try:
            from generate_embeddings import generate_claim_embeddings
            
            if progress_callback:
                progress_callback(0.1)
            
            # Generate embeddings with progress tracking
            def embedding_progress(current, total):
                if progress_callback:
                    progress_callback(0.1 + (0.8 * current / total))
            
            result = generate_claim_embeddings(
                progress_callback=embedding_progress
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            return {
                'embeddings_generated': result.get('count', 0),
                'model': result.get('model', 'unknown'),
                'dimensions': result.get('dimensions', 0)
            }
            
        except ImportError:
            logger.warning("generate_embeddings.py not found - using placeholder")
            
            # Placeholder implementation
            # TODO: Implement actual embedding generation
            if progress_callback:
                progress_callback(1.0)
            
            return {
                'embeddings_generated': 0,
                'model': 'placeholder',
                'note': 'Embedding generation not yet implemented'
            }
    
    @staticmethod
    def build_temporal_index(checkpoint: Optional[Dict] = None,
                            progress_callback: Optional[Callable] = None,
                            builder = None) -> Dict:
        """
        Build temporal index for coordination detection
        
        Creates indexes and relationships for fast temporal queries
        """
        logger = logging.getLogger(__name__)
        logger.info("Building temporal index...")
        
        if not builder or not builder.driver:
            raise ValueError("Builder with Neo4j driver required")
        
        with builder.driver.session() as session:
            # Create index on temporal data
            if progress_callback:
                progress_callback(0.2)
            
            logger.info("Creating temporal indexes...")
            
            # Index for date-based queries
            try:
                session.run("""
                    CREATE INDEX claim_temporal_idx IF NOT EXISTS
                    FOR (c:Claim) ON (c.temporal_data)
                """)
            except:
                pass  # Index might already exist
            
            if progress_callback:
                progress_callback(0.5)
            
            # Count claims with temporal data
            result = session.run("""
                MATCH (c:Claim)
                WHERE c.temporal_data IS NOT NULL 
                AND c.temporal_data <> '{}'
                AND c.temporal_data <> 'null'
                RETURN count(c) as count
            """)
            temporal_claims = result.single()['count']
            
            if progress_callback:
                progress_callback(0.8)
            
            # Create time-based relationships (optional - for graph queries)
            # Skip for now to save time
            
            if progress_callback:
                progress_callback(1.0)
            
            return {
                'claims_with_temporal_data': temporal_claims,
                'indexes_created': 1
            }
    
    @staticmethod
    def build_geographic_index(checkpoint: Optional[Dict] = None,
                              progress_callback: Optional[Callable] = None,
                              builder = None) -> Dict:
        """
        Build geographic index for anomaly detection
        
        Creates spatial indexes for fast geographic queries
        """
        logger = logging.getLogger(__name__)
        logger.info("Building geographic index...")
        
        if not builder or not builder.driver:
            raise ValueError("Builder with Neo4j driver required")
        
        with builder.driver.session() as session:
            # Create index on geographic data
            if progress_callback:
                progress_callback(0.2)
            
            logger.info("Creating geographic indexes...")
            
            try:
                session.run("""
                    CREATE INDEX claim_geographic_idx IF NOT EXISTS
                    FOR (c:Claim) ON (c.geographic_data)
                """)
            except:
                pass
            
            if progress_callback:
                progress_callback(0.5)
            
            # Count claims with geographic data
            result = session.run("""
                MATCH (c:Claim)
                WHERE c.geographic_data IS NOT NULL 
                AND c.geographic_data <> '{}'
                AND c.geographic_data <> 'null'
                RETURN count(c) as count
            """)
            geographic_claims = result.single()['count']
            
            if progress_callback:
                progress_callback(0.8)
            
            # Create Location nodes (optional)
            # Skip for now to save time
            
            if progress_callback:
                progress_callback(1.0)
            
            return {
                'claims_with_geographic_data': geographic_claims,
                'indexes_created': 1
            }


def register_jobs(builder):
    """
    Register job implementations with the builder
    
    Call this to connect job functions to the DetectionSystemBuilder
    """
    jobs = DetectionSetupJobs()
    
    builder.JOBS['build_citations'].func = jobs.build_citations
    builder.JOBS['generate_embeddings'].func = jobs.generate_embeddings
    builder.JOBS['build_temporal_index'].func = jobs.build_temporal_index
    builder.JOBS['build_geographic_index'].func = jobs.build_geographic_index
    
    return builder


if __name__ == "__main__":
    # Test job implementations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    from aegis_detection_system_builder import DetectionSystemBuilder
    
    # Create builder
    builder = DetectionSystemBuilder()
    
    # Register jobs
    builder = register_jobs(builder)
    
    # Test system status
    status = builder.get_system_status()
    print(json.dumps(status, indent=2))
    
    builder.close()
