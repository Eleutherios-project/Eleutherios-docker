#!/usr/bin/env python3
"""
AegisTrustNet - Trust Propagation Algorithm
This module implements the trust propagation algorithms for the Truth Network Overlay.
It calculates trust scores based on network structure, citations, and user preferences.

The algorithm is CPU-optimized for a single machine environment with 32 cores.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from neo4j import GraphDatabase
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
from functools import partial
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('aegis-trustnet-algo')

# Load environment variables
load_dotenv()

# Constants and configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

# Algorithm parameters
DAMPING_FACTOR = 0.85          # How much trust propagates through the network (like PageRank)
DISTRUST_FACTOR = 0.5          # How much negative relationships affect trust scores
ITERATIONS_MAX = 20            # Maximum iterations for convergence
CONVERGENCE_THRESHOLD = 1e-6   # Convergence threshold for trust score changes
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")  # Cache for matrix computations

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

class TrustPropagationAlgorithm:
    def __init__(self, batch_mode=False):
        """Initialize the trust propagation algorithm with Neo4j connection"""
        self.batch_mode = batch_mode
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600
            )
            logger.info(f"Connected to Neo4j at {NEO4J_URI}")
            
            # CPU cores for parallel processing
            self.num_cores = min(30, mp.cpu_count())  # Use up to 30 cores, leaving some for system
            logger.info(f"Using {self.num_cores} CPU cores for trust computations")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_network_size(self):
        """Get the size of the trust network for memory planning"""
        with self.driver.session() as session:
            # Count nodes in network
            node_count = session.run("""
                MATCH (n) 
                WHERE n:Source OR n:Entity OR n:Claim
                RETURN count(n) as node_count
            """).single()["node_count"]
            
            # Count relationships in network
            rel_count = session.run("""
                MATCH ()-[r]->() 
                WHERE type(r) IN ['CITES', 'SUPPORTS', 'CONTRADICTS', 'CLAIMS', 'RELATED_TO', 'MENTIONS']
                RETURN count(r) as rel_count
            """).single()["rel_count"]
            
            return node_count, rel_count

    def estimate_memory_usage(self, node_count, rel_count):
        """Estimate memory usage for matrix computations"""
        # Each node requires:
        # - Node ID mapping: ~16 bytes
        # - Trust score: ~8 bytes
        # - Other metadata: ~32 bytes
        node_memory = node_count * 56
        
        # Each relationship requires:
        # - Source/target indices: ~16 bytes
        # - Weight value: ~8 bytes
        # - Other metadata: ~8 bytes
        rel_memory = rel_count * 32
        
        # Sparse matrix overhead: ~20% of relationship memory
        matrix_overhead = rel_memory * 0.2
        
        # Temporary computation data: ~3x the matrix size
        temp_data = (rel_memory + matrix_overhead) * 3
        
        # Total estimated memory in gigabytes
        total_memory_gb = (node_memory + rel_memory + matrix_overhead + temp_data) / (1024**3)
        
        return total_memory_gb

    def fetch_network_data(self):
        """Fetch network data from Neo4j and prepare for trust computation"""
        logger.info("Fetching network data from Neo4j...")
        
        # Get network size to estimate memory requirements
        node_count, rel_count = self.get_network_size()
        logger.info(f"Network size: {node_count} nodes, {rel_count} relationships")
        
        # Estimate memory usage
        memory_gb = self.estimate_memory_usage(node_count, rel_count)
        logger.info(f"Estimated memory requirement: {memory_gb:.2f} GB")
        
        # Check if system has enough memory
        available_memory = 150.0  # GB, from your machine specifications
        if memory_gb > available_memory * 0.8:
            logger.warning(f"Memory requirement ({memory_gb:.2f} GB) approaches available memory ({available_memory} GB)")
            logger.warning("Will use disk-based processing for large matrices")
            use_disk_cache = True
        else:
            use_disk_cache = False
        
        # Fetch nodes with a single query for efficiency
        with self.driver.session() as session:
            # Fetch all nodes in the trust network
            nodes_query = """
            MATCH (n) 
            WHERE n:Source OR n:Entity OR n:Claim
            RETURN id(n) as neo4j_id, 
                   CASE 
                     WHEN n:Source THEN coalesce(n.name, '') 
                     WHEN n:Entity THEN coalesce(n.name, '')
                     WHEN n:Claim THEN coalesce(n.claim_text, '')
                     ELSE 'Unknown'
                   END as name,
                   labels(n) as types,
                   CASE
                     WHEN n:Source THEN coalesce(n.initial_trust, 0.0)
                     ELSE 0.0
                   END as initial_trust
            """
            
            nodes_result = session.run(nodes_query)
            
            # Process nodes into a DataFrame for easier handling
            nodes_data = []
            for record in nodes_result:
                nodes_data.append({
                    'neo4j_id': record['neo4j_id'],
                    'name': record['name'],
                    'types': record['types'],
                    'initial_trust': record['initial_trust']
                })
            
            nodes_df = pd.DataFrame(nodes_data)
            
            # Create a mapping from Neo4j IDs to matrix indices
            id_to_index = {row['neo4j_id']: i for i, row in enumerate(nodes_data)}
            index_to_id = {i: row['neo4j_id'] for i, row in enumerate(nodes_data)}
            
            # Initialize trust scores from initial values
            trust_scores = np.zeros(len(nodes_df))
            for i, row in enumerate(nodes_data):
                trust_scores[i] = row['initial_trust']
            
            logger.info(f"Loaded {len(nodes_df)} nodes for trust computation")
            
            # Fetch all relationships in batches to manage memory
            rels_data = []
            batch_size = 100000
            skip = 0
            
            while True:
                # Fetch relationships batch
                rels_query = f"""
                MATCH (source)-[r]->(target)
                WHERE type(r) IN ['CITES', 'SUPPORTS', 'CONTRADICTS', 'CLAIMS', 'RELATED_TO', 'MENTIONS']
                RETURN id(source) as source_id,
                       id(target) as target_id,
                       type(r) as rel_type,
                       CASE
                         WHEN type(r) = 'CONTRADICTS' THEN -1.0 * coalesce(r.weight, 1.0) * {DISTRUST_FACTOR}
                         WHEN type(r) = 'SUPPORTS' THEN coalesce(r.weight, 1.0)
                         ELSE coalesce(r.weight, 1.0) * 0.5
                       END as weight
                SKIP {skip} LIMIT {batch_size}
                """
                
                batch_result = session.run(rels_query)
                batch_data = list(batch_result)
                
                if not batch_data:
                    break
                
                for record in batch_data:
                    # Skip relationships that involve nodes not in our node list
                    if record['source_id'] not in id_to_index or record['target_id'] not in id_to_index:
                        continue
                    
                    rels_data.append({
                        'source_idx': id_to_index[record['source_id']],
                        'target_idx': id_to_index[record['target_id']],
                        'rel_type': record['rel_type'],
                        'weight': record['weight']
                    })
                
                skip += batch_size
                logger.info(f"Loaded {len(rels_data)} relationships so far...")
                
                if len(batch_data) < batch_size:
                    break
            
            logger.info(f"Loaded {len(rels_data)} relationships for trust computation")
            
            # Convert to DataFrame for easier handling
            rels_df = pd.DataFrame(rels_data)
            
            return {
                'nodes_df': nodes_df,
                'rels_df': rels_df,
                'id_to_index': id_to_index,
                'index_to_id': index_to_id,
                'trust_scores': trust_scores,
                'use_disk_cache': use_disk_cache
            }
    def process_relationships_chunk(chunk_df):
        """Process a chunk of relationships (function moved outside of method for pickling)"""
        chunk_results = []
        for _, row in chunk_df.iterrows():
            source_idx = row['source_idx']
            target_idx = row['target_idx']
            weight = row['weight']
            chunk_results.append((source_idx, target_idx, weight))
        return chunk_results

class TrustPropagationAlgorithm:
    def compute_trust_scores(self, network_data, user_trust_seeds=None):
        """
        Compute trust scores using a PageRank-like algorithm with positive and negative relations
        
        Args:
            network_data: Dictionary with network data from fetch_network_data
            user_trust_seeds: Dictionary mapping node IDs to user-defined trust scores
        
        Returns:
            Dictionary mapping node IDs to computed trust scores
        """
        logger.info("Computing trust scores...")
        start_time = time.time()
        
        nodes_df = network_data['nodes_df']
        rels_df = network_data['rels_df']
        id_to_index = network_data['id_to_index']
        index_to_id = network_data['index_to_id']
        initial_trust = network_data['trust_scores']
        use_disk_cache = network_data['use_disk_cache']
        
        n_nodes = len(nodes_df)
        
        # Apply user trust seeds if provided
        if user_trust_seeds:
            for node_id, trust_score in user_trust_seeds.items():
                if node_id in id_to_index:
                    node_idx = id_to_index[node_id]
                    initial_trust[node_idx] = trust_score
        
        # Prepare the adjacency matrix 
        logger.info(f"Building adjacency matrix for {n_nodes} nodes and {len(rels_df)} relationships...")
        
        # Use LIL format for efficient construction, then convert to CSR for computation
        adj_matrix = lil_matrix((n_nodes, n_nodes), dtype=np.float32)
        
   
        
        # Determine chunk size based on relationship count
        chunk_size = min(500000, max(10000, len(rels_df) // self.num_cores))
        rel_chunks = [rels_df.iloc[i:i+chunk_size] for i in range(0, len(rels_df), chunk_size)]
        
        # Process chunks in parallel
        if len(rel_chunks) > 1:
            logger.info(f"Processing relationships in {len(rel_chunks)} parallel chunks...")
            with mp.Pool(processes=self.num_cores) as pool:
                all_results = pool.map(process_relationships_chunk, rel_chunks)
                
                # Combine results
                for chunk_results in all_results:
                    for source_idx, target_idx, weight in chunk_results:
                        adj_matrix[source_idx, target_idx] = weight
        else:
            # Single-thread processing for small datasets
            for _, row in rels_df.iterrows():
                source_idx = row['source_idx']
                target_idx = row['target_idx']
                weight = row['weight']
                adj_matrix[source_idx, target_idx] = weight
        
        # Convert to CSR format for efficient computation
        adj_matrix = adj_matrix.tocsr()
        
        # Normalize the matrix column-wise (like in PageRank)
        logger.info("Normalizing adjacency matrix...")
        
        # Column sums for normalization
        col_sums = np.array(adj_matrix.sum(axis=0)).flatten()
        
        # Replace zeros with 1 to avoid division by zero
        col_sums[col_sums == 0] = 1
        
        # Normalize columns
        for j in range(n_nodes):
            adj_matrix.data[adj_matrix.indptr[j]:adj_matrix.indptr[j+1]] /= col_sums[j]
        
        # Initialize trust scores
        trust_scores = initial_trust.copy()
        
        # Iterative computation, similar to PageRank
        logger.info("Starting iterative trust computation...")
        
        prev_scores = trust_scores.copy()
        converged = False
        
        for iteration in range(ITERATIONS_MAX):
            # Trust propagation step
            # Formula: trust = (1-d) * initial_trust + d * (A * trust)
            #   where A is the normalized adjacency matrix and d is the damping factor
            
            # Matrix multiplication step (most computationally intensive)
            propagated = DAMPING_FACTOR * (adj_matrix.T @ trust_scores)
            
            # Combine with initial trust
            trust_scores = (1 - DAMPING_FACTOR) * initial_trust + propagated
            
            # Check convergence
            diff = np.linalg.norm(trust_scores - prev_scores)
            logger.info(f"Iteration {iteration+1}: convergence metric = {diff:.6f}")
            
            if diff < CONVERGENCE_THRESHOLD:
                logger.info(f"Converged after {iteration+1} iterations")
                converged = True
                break
                
            prev_scores = trust_scores.copy()
        
        if not converged:
            logger.warning(f"Did not fully converge after {ITERATIONS_MAX} iterations")
        
        # Normalize scores to [0,1] range
        min_score = trust_scores.min()
        max_score = trust_scores.max()
        
        if max_score > min_score:
            normalized_scores = (trust_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(trust_scores)
        
        # Create result dictionary mapping Neo4j IDs to trust scores
        trust_result = {}
        for idx, score in enumerate(normalized_scores):
            neo4j_id = index_to_id[idx]
            trust_result[neo4j_id] = float(score)
        
        elapsed = time.time() - start_time
        logger.info(f"Trust computation completed in {elapsed:.2f} seconds")
        
        return trust_result

    def store_trust_scores(self, trust_scores):
        """Store computed trust scores back in Neo4j"""
        logger.info("Storing trust scores in Neo4j...")
        start_time = time.time()
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        all_node_ids = list(trust_scores.keys())
        total_nodes = len(all_node_ids)
        
        with self.driver.session() as session:
            for i in range(0, total_nodes, batch_size):
                batch = all_node_ids[i:i+batch_size]
                
                # Prepare batch parameters
                params = {
                    'batch': [{'id': node_id, 'score': trust_scores[node_id]} for node_id in batch],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update nodes in batch
                query = """
                UNWIND $batch as node
                MATCH (n) WHERE id(n) = node.id
                SET n.trust_score = node.score,
                    n.trust_updated = datetime($timestamp)
                """
                
                session.run(query, params)
                logger.info(f"Updated {min(i+batch_size, total_nodes)}/{total_nodes} nodes")
        
        elapsed = time.time() - start_time
        logger.info(f"Trust scores stored in Neo4j in {elapsed:.2f} seconds")

    def run_full_trust_computation(self, user_trust_seeds=None):
        """Run a complete trust computation cycle"""
        try:
            logger.info("Starting full trust computation cycle...")
            
            # 1. Fetch network data
            network_data = self.fetch_network_data()
            
            # 2. Compute trust scores
            trust_scores = self.compute_trust_scores(network_data, user_trust_seeds)
            
            # 3. Store results back to Neo4j
            self.store_trust_scores(trust_scores)
            
            # 4. Compute community detection if batch mode
            if self.batch_mode:
                self.detect_trust_communities()
            
            logger.info("Trust computation cycle completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error in trust computation: {str(e)}")
            traceback.print_exc()
            return False

    def detect_trust_communities(self):
        """
        Detect communities in the trust network
        This identifies distinct 'trust islands' or networks that might
        represent different worldviews or narrative clusters
        """
        logger.info("Detecting trust communities...")
        
        with self.driver.session() as session:
            # Use Neo4j's graph algorithms for community detection
            # Note: This requires the Graph Data Science (GDS) library to be installed
            try:
                # First check if GDS is available
                gds_check = session.run("CALL gds.list() YIELD name RETURN count(*) > 0 as has_gds").single()
                
                if not gds_check or not gds_check.get("has_gds", False):
                    logger.warning("Graph Data Science library not available - skipping community detection")
                    return
                
                # Create a graph projection for community detection
                # Focus on positive relationships (SUPPORTS, CITES with positive sentiment)
                proj_query = """
                CALL gds.graph.project(
                    'trust_communities',
                    ['Source', 'Entity', 'Claim'],
                    {
                        SUPPORTS: {orientation: 'UNDIRECTED'},
                        CLAIMS: {orientation: 'UNDIRECTED'},
                        CITES: {
                            orientation: 'UNDIRECTED',
                            properties: {
                                weight: {
                                    property: 'weight',
                                    defaultValue: 1.0
                                }
                            }
                        },
                        RELATED_TO: {
                            orientation: 'UNDIRECTED',
                            properties: {
                                weight: {
                                    property: 'weight',
                                    defaultValue: 0.5
                                }
                            }
                        }
                    },
                    {relationshipProperties: ['weight']}
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """
                
                proj_result = session.run(proj_query)
                for record in proj_result:
                    logger.info(f"Created graph projection with {record['nodeCount']} nodes and {record['relationshipCount']} relationships")
                
                # Run Louvain algorithm for community detection
                community_query = """
                CALL gds.louvain.write(
                    'trust_communities',
                    {
                        relationshipWeightProperty: 'weight',
                        writeProperty: 'community_id',
                        includeIntermediateCommunities: true,
                        intermediateCommunitiesWriteProperty: 'communities'
                    }
                )
                YIELD communityCount, modularity, modularities
                RETURN communityCount, modularity, modularities
                """
                
                community_result = session.run(community_query)
                for record in community_result:
                    logger.info(f"Detected {record['communityCount']} communities with modularity {record['modularity']}")
                
                # Clean up the projection
                session.run("CALL gds.graph.drop('trust_communities')")
                
                # Create community nodes to represent each detected community
                create_community_nodes = """
                MATCH (n)
                WHERE n.community_id IS NOT NULL
                WITH DISTINCT n.community_id AS community_id, count(n) AS node_count
                
                MERGE (c:Community {id: community_id})
                ON CREATE SET 
                    c.size = node_count,
                    c.detected = datetime()
                ON MATCH SET
                    c.size = node_count,
                    c.updated = datetime()
                
                WITH c
                
                MATCH (n) 
                WHERE n.community_id = c.id
                MERGE (n)-[:BELONGS_TO]->(c)
                """
                
                session.run(create_community_nodes)
                logger.info("Created community nodes and relationships")
                
                # Analyze and label communities
                analyze_communities = """
                MATCH (c:Community)
                WHERE c.name IS NULL
                
                MATCH (n)-[:BELONGS_TO]->(c)
                WITH c, n.name AS name, count(*) AS cnt
                ORDER BY cnt DESC
                LIMIT 10
                
                WITH c, collect(name) AS top_names
                SET c.top_members = top_names,
                    c.name = 'Community-' + c.id
                
                RETURN c.id, c.name, c.size, c.top_members
                """
                
                analyze_result = session.run(analyze_communities)
                for record in analyze_result:
                    logger.info(f"Community {record['c.name']} with {record['c.size']} members. Top members: {record['c.top_members']}")
                
            except Exception as e:
                logger.error(f"Error in community detection: {str(e)}")
                logger.error(traceback.format_exc())

def run_scheduled_computation():
    """Run a scheduled computation of trust scores"""
    algo = TrustPropagationAlgorithm()
    try:
        # Run the full computation cycle
        algo.run_full_trust_computation()
    finally:
        algo.close()

def process_user_trust_preferences(user_id, preferences):
    """
    Process a user's trust preferences and compute personalized trust scores
    
    Args:
        user_id: User identifier
        preferences: Dict mapping source names to trust scores (-1.0 to 1.0)
    
    Returns:
        Success flag
    """
    algo = TrustPropagationAlgorithm(batch_mode=False)
    try:
        # Convert source names to Neo4j node IDs
        user_trust_seeds = {}
        
        with algo.driver.session() as session:
            for source_name, trust_score in preferences.items():
                query = """
                MATCH (s:Source {name: $name})
                RETURN id(s) as node_id
                """
                
                result = session.run(query, {"name": source_name})
                record = result.single()
                
                if record:
                    node_id = record["node_id"]
                    user_trust_seeds[node_id] = trust_score
                    
                    # Also store user preference in database
                    save_pref = """
                    MATCH (s:Source {name: $name})
                    MERGE (u:User {id: $user_id})
                    MERGE (u)-[r:TRUSTS]->(s)
                    SET r.score = $score,
                        r.timestamp = datetime()
                    """
                    
                    session.run(save_pref, {
                        "name": source_name,
                        "user_id": user_id,
                        "score": trust_score
                    })
        
        # Run computation with user's trust seeds
        return algo.run_full_trust_computation(user_trust_seeds)
    finally:
        algo.close()

if __name__ == "__main__":
    # When run directly, perform a full trust computation cycle
    run_scheduled_computation()
