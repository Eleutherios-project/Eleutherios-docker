#!/usr/bin/env python3
"""
AegisTrustNet - Pattern Discovery Service
This module analyzes the trust network to discover significant patterns,
second and third-order effects, and meaningful correlations.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('aegis-pattern-discovery')

# Load environment variables
load_dotenv()

# Constants and configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
MIN_CORRELATION = float(os.environ.get("MIN_CORRELATION", "0.3"))

class PatternDiscoveryService:
    def __init__(self):
        """Initialize the pattern discovery service with Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600
            )
            logger.info(f"Connected to Neo4j at {NEO4J_URI}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def discover_correlation_patterns(self, min_frequency: int = 3) -> List[Dict[str, Any]]:
        """
        Discover correlation patterns between entities and events
        Only returns statistically significant correlations with sufficient support
        """
        start_time = time.time()
        logger.info(f"Discovering correlation patterns (min_frequency: {min_frequency})...")
        
        patterns = []
        
        with self.driver.session() as session:
            # Find entity pairs that co-occur frequently
            cooccurrence_query = """
            MATCH (d:Document)-[:MENTIONS]->(e1:Entity)
            MATCH (d)-[:MENTIONS]->(e2:Entity)
            WHERE e1 <> e2 AND e1.name < e2.name
            WITH e1.name as entity1, e2.name as entity2, count(d) as cooccurrence_count
            WHERE cooccurrence_count >= $min_frequency
            
            // Get individual occurrence counts
            MATCH (d1:Document)-[:MENTIONS]->(e1:Entity {name: entity1})
            WITH entity1, entity2, cooccurrence_count, count(d1) as count1
            
            MATCH (d2:Document)-[:MENTIONS]->(e2:Entity {name: entity2})
            WITH entity1, entity2, cooccurrence_count, count1, count(d2) as count2
            
            // Calculate statistical measures
            WITH entity1, entity2, cooccurrence_count, count1, count2,
                 cooccurrence_count * 1.0 / count1 as confidence1,
                 cooccurrence_count * 1.0 / count2 as confidence2,
                 cooccurrence_count * 1.0 / count1 * count2 as lift
            WHERE confidence1 >= $confidence_threshold OR confidence2 >= $confidence_threshold
            
            RETURN entity1, entity2, cooccurrence_count, count1, count2, 
                   confidence1, confidence2, lift
            ORDER BY lift DESC
            LIMIT 500
            """
            
            result = session.run(cooccurrence_query, {
                "min_frequency": min_frequency,
                "confidence_threshold": CONFIDENCE_THRESHOLD
            })
            
            for record in result:
                pattern = {
                    "type": "correlation",
                    "entity1": record["entity1"],
                    "entity2": record["entity2"],
                    "cooccurrences": record["cooccurrence_count"],
                    "entity1_occurrences": record["count1"],
                    "entity2_occurrences": record["count2"],
                    "confidence1": record["confidence1"],
                    "confidence2": record["confidence2"],
                    "lift": record["lift"],
                    "significance": self._calculate_pattern_significance(
                        record["cooccurrence_count"], 
                        record["count1"], 
                        record["count2"],
                        # Assuming total document count is around 1000 - should be fetched dynamically
                        1000
                    )
                }
                
                # Only add significant patterns
                if pattern["significance"] > 0.5:
                    patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} significant correlation patterns")
        
        elapsed = time.time() - start_time
        logger.info(f"Correlation pattern discovery completed in {elapsed:.2f} seconds")
        return patterns

    def discover_causal_patterns(self, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """
        Discover potential causal relationships between entities and events
        Uses temporal sequence and causal language indicators
        """
        start_time = time.time()
        logger.info(f"Discovering causal patterns (min_confidence: {min_confidence})...")
        
        patterns = []
        
        with self.driver.session() as session:
            # Find causal claims
            causal_claims_query = """
            MATCH (c:Claim)
            WHERE c.claim_type = 'causal'
            MATCH (c)-[:MENTIONS]->(e1:Entity)
            MATCH (c)-[:MENTIONS]->(e2:Entity)
            WHERE e1 <> e2
            
            // Look for causal language indicators
            WITH c, e1.name as cause, e2.name as effect, c.claim_text as claim_text
            
            // Identify causal direction using keyword analysis
            WHERE (
                // Cause -> Effect patterns
                (
                    claim_text CONTAINS (cause + " cause") OR
                    claim_text CONTAINS (cause + " lead") OR
                    claim_text CONTAINS (cause + " result") OR
                    claim_text CONTAINS (cause + " drive") OR
                    claim_text CONTAINS (cause + " influence") OR
                    claim_text CONTAINS (cause + " affect") OR
                    claim_text CONTAINS (cause + " impact") OR
                    claim_text CONTAINS ("because of " + cause) OR
                    claim_text CONTAINS ("due to " + cause)
                ) AND (
                    claim_text CONTAINS effect
                )
            ) OR (
                // Effect <- Cause patterns
                (
                    claim_text CONTAINS effect AND
                    claim_text CONTAINS (effect + " because") OR
                    claim_text CONTAINS (effect + " due to") OR
                    claim_text CONTAINS (effect + " result of") OR
                    claim_text CONTAINS (effect + " caused by") OR
                    claim_text CONTAINS (effect + " influenced by") OR
                    claim_text CONTAINS (effect + " affected by") OR
                    claim_text CONTAINS (effect + " from " + cause)
                )
            )
            
            WITH cause, effect, count(c) as claim_count, collect(c.id) as claim_ids
            
            // Calculate basic confidence based on claim count and claim trust scores
            MATCH (claim:Claim)
            WHERE claim.id IN claim_ids
            WITH cause, effect, claim_count, avg(coalesce(claim.trust_score, 0.5)) as avg_trust
            
            WHERE claim_count >= 2 AND avg_trust >= $min_confidence
            
            RETURN cause, effect, claim_count, avg_trust as confidence
            ORDER BY confidence DESC, claim_count DESC
            LIMIT 100
            """
            
            result = session.run(causal_claims_query, {
                "min_confidence": min_confidence
            })
            
            for record in result:
                pattern = {
                    "type": "causal",
                    "cause": record["cause"],
                    "effect": record["effect"],
                    "claim_count": record["claim_count"],
                    "confidence": record["confidence"]
                }
                
                patterns.append(pattern)
            
            # Also look for temporal sequences that might indicate causality
            temporal_sequence_query = """
            MATCH (d1:Document)-[:MENTIONS]->(e1:Entity)
            MATCH (d2:Document)-[:MENTIONS]->(e2:Entity)
            WHERE e1 <> e2
                  AND d1.indexed_date < d2.indexed_date
                  AND duration.between(d1.indexed_date, d2.indexed_date).days <= 90
            
            WITH e1.name as first_event, e2.name as second_event, 
                 count(DISTINCT d1) as count1, count(DISTINCT d2) as count2
            
            WHERE count1 >= 3 AND count2 >= 3
            
            // Count how many times the sequence appears in correct order
            MATCH (d3:Document)-[:MENTIONS]->(e3:Entity {name: first_event})
            MATCH (d4:Document)-[:MENTIONS]->(e4:Entity {name: second_event})
            WHERE d3.indexed_date < d4.indexed_date
                  AND duration.between(d3.indexed_date, d4.indexed_date).days <= 90
            
            WITH first_event, second_event, count1, count2, count(DISTINCT d3) as sequence_count
            
            // Calculate temporal confidence
            WITH first_event, second_event, count1, count2, sequence_count,
                 1.0 * sequence_count / count1 as temporal_confidence
            
            WHERE temporal_confidence >= $min_confidence
            
            RETURN first_event, second_event, count1, count2, sequence_count, temporal_confidence
            ORDER BY temporal_confidence DESC
            LIMIT 50
            """
            
            temporal_result = session.run(temporal_sequence_query, {
                "min_confidence": min_confidence
            })
            
            for record in temporal_result:
                pattern = {
                    "type": "temporal_sequence",
                    "first_event": record["first_event"],
                    "second_event": record["second_event"],
                    "first_event_count": record["count1"],
                    "second_event_count": record["count2"],
                    "sequence_count": record["sequence_count"],
                    "temporal_confidence": record["temporal_confidence"]
                }
                
                patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} potential causal patterns")
        
        elapsed = time.time() - start_time
        logger.info(f"Causal pattern discovery completed in {elapsed:.2f} seconds")
        return patterns

    def discover_network_influence_patterns(self) -> List[Dict[str, Any]]:
        """
        Discover influence patterns in the entity network
        Identifies key influencers and how influence propagates
        """
        start_time = time.time()
        logger.info("Discovering network influence patterns...")
        
        patterns = []
        
        with self.driver.session() as session:
            # Get a subgraph of entities and their relationships
            graph_query = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE type(r) IN ['SUPPORTS', 'CONTRADICTS', 'RELATED_TO', 'CITES', 'MENTIONS']
            WITH e1, e2, r, type(r) as relation_type, coalesce(r.weight, 1.0) as weight
            
            RETURN e1.name as source, e2.name as target, 
                   relation_type, weight, 
                   coalesce(e1.trust_score, 0.5) as source_trust,
                   coalesce(e2.trust_score, 0.5) as target_trust
            LIMIT 10000
            """
            
            result = session.run(graph_query)
            
            # Build a NetworkX directed graph
            G = nx.DiGraph()
            
            for record in result:
                source = record["source"]
                target = record["target"]
                relation_type = record["relation_type"]
                weight = record["weight"]
                source_trust = record["source_trust"]
                target_trust = record["target_trust"]
                
                # Add nodes if they don't exist
                if not G.has_node(source):
                    G.add_node(source, trust_score=source_trust)
                
                if not G.has_node(target):
                    G.add_node(target, trust_score=target_trust)
                
                # Add edge with attributes
                edge_weight = weight
                if relation_type == "CONTRADICTS":
                    edge_weight = -weight  # Negative weight for contradictions
                
                G.add_edge(source, target, weight=edge_weight, relation_type=relation_type)
            
            # Calculate network centrality measures
            if len(G) > 0:
                # PageRank centrality
                pagerank = nx.pagerank(G, weight='weight')
                
                # Betweenness centrality (identifies bridge nodes)
                betweenness = nx.betweenness_centrality(G, weight='weight')
                
                # Eigenvector centrality (identifies influential nodes)
                eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                
                # Find key influence patterns
                for node in G.nodes():
                    # Only include significant nodes
                    if pagerank[node] > 0.01 or betweenness[node] > 0.05 or eigenvector[node] > 0.05:
                        # Find direct influence paths
                        influence_paths = []
                        
                        for neighbor in G.successors(node):
                            edge_data = G.get_edge_data(node, neighbor)
                            if edge_data['weight'] > 0.5:  # Significant positive influence
                                influence_paths.append({
                                    "target": neighbor,
                                    "relationship": edge_data['relation_type'],
                                    "strength": edge_data['weight']
                                })
                        
                        pattern = {
                            "type": "influence",
                            "entity": node,
                            "pagerank": pagerank[node],
                            "betweenness": betweenness[node],
                            "eigenvector": eigenvector[node],
                            "influence_score": (pagerank[node] + betweenness[node] + eigenvector[node]) / 3,
                            "influence_paths": influence_paths
                        }
                        
                        patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} influence patterns")
        
        elapsed = time.time() - start_time
        logger.info(f"Network influence pattern discovery completed in {elapsed:.2f} seconds")
        return patterns

    def discover_higher_order_effects(self) -> List[Dict[str, Any]]:
        """
        Discover second and third-order effects in the network
        These are indirect consequences that propagate through multiple steps
        """
        start_time = time.time()
        logger.info("Discovering higher-order effects...")
        
        patterns = []
        
        with self.driver.session() as session:
            # Look for multi-step causal chains (2nd and 3rd order effects)
            chains_query = """
            // Find chains of causal relationships
            MATCH path = (e1:Entity)-[:CAUSES|INFLUENCES|SUPPORTS*2..3]->(e3:Entity)
            WHERE e1 <> e3
            
            // Ensure this is a true chain without shortcuts
            WHERE NOT (e1)-[:CAUSES|INFLUENCES|SUPPORTS]->(e3)
            
            // Extract entities and relationships in the path
            UNWIND nodes(path) as node
            WITH path, collect(node.name) as entities
            
            UNWIND relationships(path) as rel
            WITH path, entities, collect(type(rel)) as relationship_types
            
            RETURN entities, relationship_types, length(path) as chain_length, id(path) as path_id
            LIMIT 100
            """
            
            result = session.run(chains_query)
            
            for record in result:
                entities = record["entities"]
                relationship_types = record["relationship_types"]
                chain_length = record["chain_length"]
                
                # Only include if we have a complete chain
                if len(entities) >= chain_length + 1:
                    pattern = {
                        "type": "higher_order_effect",
                        "order": chain_length,
                        "initial_cause": entities[0],
                        "intermediate_factors": entities[1:-1],
                        "final_effect": entities[-1],
                        "relationship_chain": relationship_types
                    }
                    
                    patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} higher-order effect patterns")
            
            # Also look for cascading effects (many effects from one cause)
            cascade_query = """
            MATCH (e1:Entity)
            
            // First order effects
            MATCH (e1)-[r1:CAUSES|INFLUENCES|SUPPORTS]->(e2:Entity)
            
            // Second order effects (effects of the effects)
            MATCH (e2)-[r2:CAUSES|INFLUENCES|SUPPORTS]->(e3:Entity)
            WHERE e1 <> e3
            
            WITH e1, count(DISTINCT e2) as first_order_count, 
                 count(DISTINCT e3) as second_order_count
                 
            WHERE first_order_count >= 3 AND second_order_count >= 5
            
            RETURN e1.name as cause, first_order_count, second_order_count,
                   first_order_count * second_order_count as cascade_magnitude
            ORDER BY cascade_magnitude DESC
            LIMIT 20
            """
            
            cascade_result = session.run(cascade_query)
            
            for record in cascade_result:
                pattern = {
                    "type": "cascade_effect",
                    "cause": record["cause"],
                    "first_order_effects": record["first_order_count"],
                    "second_order_effects": record["second_order_count"],
                    "cascade_magnitude": record["cascade_magnitude"]
                }
                
                patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} cascade effect patterns")
        
        elapsed = time.time() - start_time
        logger.info(f"Higher-order effect discovery completed in {elapsed:.2f} seconds")
        return patterns

    def discover_anomalous_patterns(self) -> List[Dict[str, Any]]:
        """
        Discover anomalous or unexpected patterns in the data
        These are connections that defy conventional wisdom or exist in high-trust sources
        but contradict common belief
        """
        start_time = time.time()
        logger.info("Discovering anomalous patterns...")
        
        patterns = []
        
        with self.driver.session() as session:
            # Find claims with high trust but low consensus
            anomalous_claims_query = """
            MATCH (c:Claim)
            WHERE c.trust_score > 0.7  // High trust
            
            // Find contradicting claims
            MATCH (c)-[r:CONTRADICTS]-(c2:Claim)
            WHERE c2.trust_score > 0.5  // Also from somewhat trusted sources
            
            // Calculate controversy score
            WITH c, count(r) as contradiction_count
            WHERE contradiction_count > 0
            
            RETURN c.id as claim_id, c.claim_text as claim_text, 
                   c.trust_score as trust_score,
                   contradiction_count,
                   c.trust_score * contradiction_count as anomaly_score
            ORDER BY anomaly_score DESC
            LIMIT 50
            """
            
            result = session.run(anomalous_claims_query)
            
            for record in result:
                pattern = {
                    "type": "anomalous_claim",
                    "claim_id": record["claim_id"],
                    "claim_text": record["claim_text"],
                    "trust_score": record["trust_score"],
                    "contradiction_count": record["contradiction_count"],
                    "anomaly_score": record["anomaly_score"]
                }
                
                patterns.append(pattern)
            
            # Find unexpected connections between entities
            unexpected_connections_query = """
            MATCH (e1:Entity)-[r:RELATED_TO|SUPPORTS|CONTRADICTS]->(e2:Entity)
            WHERE e1.trust_score > 0.6 AND e2.trust_score > 0.6
                  AND e1.type <> e2.type  // Different entity types
            
            // Get community information
            MATCH (e1)-[:BELONGS_TO]->(c1:Community)
            MATCH (e2)-[:BELONGS_TO]->(c2:Community)
            WHERE c1 <> c2  // Different communities
            
            RETURN e1.name as entity1, e1.type as type1,
                   e2.name as entity2, e2.type as type2,
                   type(r) as relationship_type,
                   c1.name as community1, c2.name as community2
            LIMIT 30
            """
            
            unexpected_result = session.run(unexpected_connections_query)
            
            for record in unexpected_result:
                pattern = {
                    "type": "unexpected_connection",
                    "entity1": record["entity1"],
                    "entity1_type": record["type1"],
                    "entity2": record["entity2"],
                    "entity2_type": record["type2"],
                    "relationship": record["relationship_type"],
                    "community1": record["community1"],
                    "community2": record["community2"]
                }
                
                patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} anomalous patterns")
        
        elapsed = time.time() - start_time
        logger.info(f"Anomalous pattern discovery completed in {elapsed:.2f} seconds")
        return patterns

    def _calculate_pattern_significance(self, cooccurrences, count1, count2, total_documents):
        """Calculate statistical significance of a pattern using chi-square test"""
        # Handle edge cases
        if count1 == 0 or count2 == 0 or total_documents == 0:
            return 0
            
        # Expected co-occurrence under independence assumption
        expected = (count1 * count2) / total_documents
        
        # If expected is too small, pattern might not be statistically valid
        if expected < 5:
            return 0.1  # Low significance
            
        # Chi-square value
        chi_square = ((cooccurrences - expected) ** 2) / expected
        
        # Convert chi-square to a significance value from 0 to 1
        # Higher values mean more significant patterns
        # This is a simplified approach; normally, you'd look up the p-value
        significance = min(1.0, chi_square / 20)  # Normalize to 0-1 range
        
        return significance

    def store_discovered_patterns(self, patterns):
        """Store discovered patterns in the database for future reference"""
        if not patterns:
            logger.info("No patterns to store")
            return
            
        stored_count = 0
        start_time = time.time()
            
        with self.driver.session() as session:
            # Create pattern nodes
            for pattern in patterns:
                pattern_id = f"pattern-{hash(str(pattern)) & 0xffffffff}"
                pattern_type = pattern.get("type", "unknown")
                
                # Base query for creating pattern node
                query = """
                CREATE (p:Pattern {
                    id: $pattern_id,
                    type: $pattern_type,
                    discovered: datetime(),
                    data: $pattern_data
                })
                RETURN p.id
                """
                
                try:
                    session.run(query, {
                        "pattern_id": pattern_id,
                        "pattern_type": pattern_type,
                        "pattern_data": json.dumps(pattern)
                    })
                    stored_count += 1
                    
                    # Create relationships between pattern and entities
                    self._link_pattern_to_entities(session, pattern_id, pattern)
                except Exception as e:
                    logger.error(f"Error storing pattern {pattern_id}: {str(e)}")
            
        elapsed = time.time() - start_time
        logger.info(f"Stored {stored_count}/{len(patterns)} patterns in {elapsed:.2f} seconds")

    def _link_pattern_to_entities(self, session, pattern_id, pattern):
        """Create relationships between a pattern and related entities"""
        # Extract entities based on pattern type
        entities = []
        
        if pattern["type"] == "correlation":
            entities = [pattern["entity1"], pattern["entity2"]]
        elif pattern["type"] == "causal":
            entities = [pattern["cause"], pattern["effect"]]
        elif pattern["type"] == "temporal_sequence":
            entities = [pattern["first_event"], pattern["second_event"]]
        elif pattern["type"] == "influence":
            entities = [pattern["entity"]]
            # Add influenced entities
            for path in pattern.get("influence_paths", []):
                entities.append(path["target"])
        elif pattern["type"] == "higher_order_effect":
            entities = [pattern["initial_cause"], pattern["final_effect"]]
            entities.extend(pattern.get("intermediate_factors", []))
        elif pattern["type"] == "cascade_effect":
            entities = [pattern["cause"]]
        elif pattern["type"] == "anomalous_claim":
            # For claims, we need to find the entities mentioned in the claim
            claim_id = pattern["claim_id"]
            result = session.run("""
            MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
            RETURN e.name as entity
            """, {"claim_id": claim_id})
            
            entities = [record["entity"] for record in result]
        elif pattern["type"] == "unexpected_connection":
            entities = [pattern["entity1"], pattern["entity2"]]
        
        # Create relationships for each entity
        for entity in entities:
            if entity:
                try:
                    session.run("""
                    MATCH (p:Pattern {id: $pattern_id})
                    MATCH (e:Entity {name: $entity})
                    MERGE (p)-[r:INVOLVES]->(e)
                    """, {"pattern_id": pattern_id, "entity": entity})
                except Exception as e:
                    logger.warning(f"Error linking pattern {pattern_id} to entity {entity}: {str(e)}")

    def run_pattern_discovery(self):
        """Run a complete pattern discovery cycle"""
        try:
            all_patterns = []
            
            # Run all pattern discovery methods
            correlation_patterns = self.discover_correlation_patterns()
            all_patterns.extend(correlation_patterns)
            
            causal_patterns = self.discover_causal_patterns()
            all_patterns.extend(causal_patterns)
            
            influence_patterns = self.discover_network_influence_patterns()
            all_patterns.extend(influence_patterns)
            
            higher_order_patterns = self.discover_higher_order_effects()
            all_patterns.extend(higher_order_patterns)
            
            anomalous_patterns = self.discover_anomalous_patterns()
            all_patterns.extend(anomalous_patterns)
            
            # Store all discovered patterns
            self.store_discovered_patterns(all_patterns)
            
            logger.info(f"Pattern discovery completed: found {len(all_patterns)} patterns")
            return True
        except Exception as e:
            logger.error(f"Error in pattern discovery: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.close()

def run_pattern_discovery():
    """Run the pattern discovery process"""
    discovery = PatternDiscoveryService()
    return discovery.run_pattern_discovery()

if __name__ == "__main__":
    run_pattern_discovery()
