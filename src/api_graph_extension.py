#!/usr/bin/env python3
"""
AegisTrustNet - Graph Visualization API Extension (FIXED)
Provides endpoints for returning graph data for visualization

FIXES APPLIED:
1. Corrected Cypher query to find Claims matching search terms
2. Proper traversal of Chunk -> Claim and Chunk -> Entity relationships
3. Creates connected network showing how claims relate to entities
4. Fixed node ID generation to ensure connections work
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis-graph-api")

# Load environment variables
load_dotenv()

# Neo4j connection
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

# Create router
graph_router = APIRouter(prefix="/graph", tags=["graph-visualization"])

class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # 'source', 'claim', 'entity', 'person', 'document'
    size: float = 10.0
    trust_score: float = 0.5
    metadata: Dict[str, Any] = {}

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str  # 'cites', 'supports', 'opposes', 'mentions', 'influences'
    weight: float = 1.0
    label: Optional[str] = None

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = {}

def get_driver():
    """Create Neo4j driver"""
    return GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600
    )

@graph_router.get("/query/{query_text}", response_model=GraphData)
async def get_graph_for_query(
    query_text: str,
    max_nodes: int = Query(50, ge=10, le=200),
    min_trust: float = Query(0.3, ge=0.0, le=1.0),
    depth: int = Query(2, ge=1, le=3),
    claim_ids: str = Query(None, description="Comma-separated claim IDs to include with edges")
):
    """
    Get graph data for a specific query
    
    FIXED: Now properly queries Claims and builds connected network
    
    Args:
        query_text: The search query
        max_nodes: Maximum number of nodes to return
        min_trust: Minimum trust score threshold
        depth: Depth of relationships to traverse
    """
    logger.info(f"Getting graph data for query: {query_text}")
    
    # Parse claim_ids if provided
    extra_claim_ids = []
    if claim_ids:
        extra_claim_ids = [cid.strip() for cid in claim_ids.split(',') if cid.strip()]
        logger.info(f"Will also fetch {len(extra_claim_ids)} additional claims by ID")
    
    driver = get_driver()
    
    try:
        with driver.session() as session:
            # FIXED: Query with relevance scoring and entity filtering
            # Split search into significant terms (filter out short words)
            search_terms = [t.strip().lower() for t in query_text.split() if len(t.strip()) > 2]
            
            if not search_terms:
                # No valid search terms
                return GraphData(nodes=[], edges=[], metadata={"query": query_text, "error": "No valid search terms"})
            
            # Build relevance scoring cases
            relevance_cases = []
            params = {"min_trust": min_trust, "max_nodes": max_nodes}
            
            for i, term in enumerate(search_terms):
                params[f"term{i}"] = term
                # Weight longer/more specific terms higher
                weight = 2 if len(term) > 5 else 1
                relevance_cases.append(f"CASE WHEN toLower(claim.claim_text) CONTAINS $term{i} THEN {weight} ELSE 0 END")
            
            relevance_expression = " + ".join(relevance_cases)

            result = session.run(f"""
                // Find Claims with relevance scoring
                MATCH (claim:Claim)
                WHERE claim.confidence >= $min_trust
                WITH claim, ({relevance_expression}) as relevance_score
                WHERE relevance_score > 0
                WITH claim, relevance_score
                ORDER BY relevance_score DESC, claim.confidence DESC
                LIMIT $max_nodes
                
                // Get Chunks containing these Claims
                MATCH (chunk:Chunk)-[r1:CONTAINS_CLAIM]->(claim)
                
                // Get Entities mentioned in those Chunks
                OPTIONAL MATCH (chunk)-[r2:MENTIONS]->(entity:Entity)
                
                // Get Documents containing the Chunks
                OPTIONAL MATCH (doc:Document)-[r3:CONTAINS_CHUNK]->(chunk)
                
                // Return with relevance scores
                WITH 
                    collect(DISTINCT claim) as claims,
                    collect(DISTINCT {{entity: entity, chunk: chunk}}) as entity_mentions,
                    collect(DISTINCT chunk) as chunks,
                    collect(DISTINCT doc) as documents,
                    collect(DISTINCT r1) as claim_rels
                
                // Filter entities: only those mentioned in 2+ chunks OR with high relevance
                UNWIND entity_mentions as em
                WITH claims, chunks, documents, claim_rels,
                     em.entity as entity,
                     collect(DISTINCT em.chunk) as entity_chunks
                WHERE size(entity_chunks) >= 2  // Entity appears in 2+ relevant chunks
                   OR entity.name IN [{", ".join([f"$term{i}" for i in range(len(search_terms))])}]  // Or matches search term
                
                WITH claims, chunks, documents, claim_rels,
                     collect(DISTINCT entity) as filtered_entities
                
                // Get relationships for filtered entities only
                UNWIND chunks as chunk
                OPTIONAL MATCH (chunk)-[r2:MENTIONS]->(entity:Entity)
                WHERE entity IN filtered_entities
                
                RETURN 
                    claims,
                    filtered_entities as entities,
                    chunks,
                    documents,
                    claim_rels,
                    collect(DISTINCT r2) as entity_rels,
                    [] as doc_rels
            """, **params)
            
            record = result.single()
            
            if not record:
                # Return empty graph
                return GraphData(nodes=[], edges=[], metadata={"query": query_text})
            
            # FIXED: Process nodes with proper ID generation
            nodes = []
            node_ids = set()
            
            # Add Claim nodes
            for claim in record['claims']:
                if claim:
                    node_id = f"claim-{claim.element_id}"
                    if node_id not in node_ids:
                        claim_text = claim.get('claim_text', claim.get('text', ''))
                        nodes.append(GraphNode(
                            id=node_id,
                            label=claim_text[:50] + "..." if len(claim_text) > 50 else claim_text,
                            type="claim",
                            size=20.0,
                            trust_score=claim.get('trust_score', 0.5),
                            metadata={
                                "full_text": claim_text,
                                "claim_type": claim.get('claim_type', 'unknown'),
                                "domain": claim.get('domain', 'unknown'),
                                "subject": claim.get('subject', ''),
                                "relation": claim.get('relation', ''),
                                "object": claim.get('object', ''),
                                "significance": claim.get('significance', 0.0),
                                "source_file": claim.get('source_file', ''),
                                "temporal_data": claim.get('temporal_data', ''),
                                "geographic_data": claim.get('geographic_data', ''),
                                "citation_data": claim.get('citation_data', ''),
                                "confidence": claim.get('confidence', 0.5)
                            }
                        ))
                        node_ids.add(node_id)
            
            # Add Entity nodes
            for entity in record['entities']:
                if entity:
                    node_id = f"entity-{entity.element_id}"
                    if node_id not in node_ids:
                        nodes.append(GraphNode(
                            id=node_id,
                            label=entity['name'],
                            type="entity",
                            size=15.0,
                            trust_score=entity.get('trust_score', 0.8),
                            metadata={
                                "entity_type": entity.get('type', 'unknown'),
                                "description": entity.get('description', '')
                            }
                        ))
                        node_ids.add(node_id)
            
            # Initialize chunk mappings for edge creation
            chunk_claims = {}  # chunk_element_id -> [claim_ids]
            chunk_entities = {}  # chunk_element_id -> [entity_ids]
            
            # Fetch additional claims by ID if provided
            if extra_claim_ids:
                logger.info(f"Fetching {len(extra_claim_ids)} additional claims by ID")
                for claim_id in extra_claim_ids:
                    # Extract the element_id from claim-4:xxx:yyy format
                    element_id = claim_id.replace('claim-', '') if claim_id.startswith('claim-') else claim_id
                    
                    extra_result = session.run("""
                        MATCH (c:Claim)
                        WHERE elementId(c) = $element_id OR c.claim_id = $claim_id_raw
                        OPTIONAL MATCH (chunk:Chunk)-[r1:CONTAINS_CLAIM]->(c)
                        OPTIONAL MATCH (chunk)-[r2:MENTIONS]->(entity:Entity)
                        RETURN c as claim, 
                               collect(DISTINCT entity) as entities,
                               collect(DISTINCT {chunk: chunk, rel: r1}) as chunk_rels,
                               collect(DISTINCT {chunk: chunk, entity: entity, rel: r2}) as entity_rels
                    """, element_id=element_id, claim_id_raw=claim_id.replace('claim-', '').split(':')[-1] if ':' in claim_id else claim_id)
                    
                    extra_record = extra_result.single()
                    if extra_record and extra_record['claim']:
                        claim = extra_record['claim']
                        node_id = f"claim-{claim.element_id}"
                        if node_id not in node_ids:
                            claim_text = claim.get('claim_text', claim.get('text', ''))
                            nodes.append(GraphNode(
                                id=node_id,
                                label=claim_text[:50] + "..." if len(claim_text) > 50 else claim_text,
                                type="claim",
                                size=20.0,
                                trust_score=claim.get('confidence', 0.5),
                                metadata={
                                    "full_text": claim_text,
                                    "claim_type": claim.get('claim_type', 'unknown'),
                                    "source_file": claim.get('source_file', ''),
                                    "temporal_data": claim.get('temporal_data', ''),
                                    "geographic_data": claim.get('geographic_data', ''),
                                    "citation_data": claim.get('citation_data', ''),
                                    "confidence": claim.get('confidence', 0.5),
                                    "from_detection": True
                                }
                            ))
                            node_ids.add(node_id)
                            
                        # Add entities for this claim
                        for entity in extra_record['entities']:
                            if entity:
                                entity_node_id = f"entity-{entity.element_id}"
                                if entity_node_id not in node_ids:
                                    nodes.append(GraphNode(
                                        id=entity_node_id,
                                        label=entity['name'],
                                        type="entity",
                                        size=15.0,
                                        trust_score=0.8,
                                        metadata={"entity_type": entity.get('type', 'unknown')}
                                    ))
                                    node_ids.add(entity_node_id)
                        
                        # Track chunk relationships for edge creation
                        for cr in extra_record['chunk_rels']:
                            if cr['chunk']:
                                chunk_id = cr['chunk'].element_id
                                if chunk_id not in chunk_claims:
                                    chunk_claims[chunk_id] = []
                                if node_id not in chunk_claims[chunk_id]:
                                    chunk_claims[chunk_id].append(node_id)
                        
                        for er in extra_record['entity_rels']:
                            if er['chunk'] and er['entity']:
                                chunk_id = er['chunk'].element_id
                                entity_node_id = f"entity-{er['entity'].element_id}"
                                if chunk_id not in chunk_entities:
                                    chunk_entities[chunk_id] = []
                                if entity_node_id not in chunk_entities[chunk_id]:
                                    chunk_entities[chunk_id].append(entity_node_id)
                
                logger.info(f"After extra claims: {len(nodes)} nodes total")
            
            # FIXED: Create edges properly connecting claims to entities
            edges = []
            
            # Map claims to chunks - DON'T check 'if rel' because empty relationships are falsy!
            claim_rels = record.get('claim_rels') or []
            logger.info(f"Processing {len(claim_rels)} claim relationships")
            for i, rel in enumerate(claim_rels):
                try:
                    chunk_element_id = rel.start_node.element_id
                    claim_id = f"claim-{rel.end_node.element_id}"
                    if chunk_element_id not in chunk_claims:
                        chunk_claims[chunk_element_id] = []
                    chunk_claims[chunk_element_id].append(claim_id)
                    if i < 3:
                        logger.info(f"  Mapped claim {i}: chunk={chunk_element_id[:30]}... -> claim={claim_id[:30]}...")
                except (AttributeError, TypeError) as e:
                    logger.error(f"  Error processing claim rel {i}: {e}")
                    continue
            
            # Map entities to chunks - DON'T check 'if rel' because empty relationships are falsy!
            entity_rels = record.get('entity_rels') or []
            logger.info(f"Processing {len(entity_rels)} entity relationships")
            for i, rel in enumerate(entity_rels):
                try:
                    chunk_element_id = rel.start_node.element_id
                    entity_id = f"entity-{rel.end_node.element_id}"
                    if chunk_element_id not in chunk_entities:
                        chunk_entities[chunk_element_id] = []
                    chunk_entities[chunk_element_id].append(entity_id)
                    if i < 3:
                        logger.info(f"  Mapped entity {i}: chunk={chunk_element_id[:30]}... -> entity={entity_id[:30]}...")
                except (AttributeError, TypeError) as e:
                    logger.error(f"  Error processing entity rel {i}: {e}")
                    continue
            
            # Log for debugging
            logger.info(f"Chunks with claims: {len(chunk_claims)}, with entities: {len(chunk_entities)}")
            
            # Create edges between claims and entities that share chunks
            for chunk_id in chunk_claims:
                claims_in_chunk = chunk_claims.get(chunk_id, [])
                entities_in_chunk = chunk_entities.get(chunk_id, [])
                
                # Connect each claim to each entity in the same chunk
                for claim_id in claims_in_chunk:
                    for entity_id in entities_in_chunk:
                        if claim_id in node_ids and entity_id in node_ids:
                            edges.append(GraphEdge(
                                source=claim_id,
                                target=entity_id,
                                type="mentions",
                                weight=0.7,
                                label="mentions"
                            ))
            
            # Also connect entities to other entities that co-occur in claims
            for chunk_id in chunk_entities:
                entities_in_chunk = chunk_entities.get(chunk_id, [])
                # Create edges between entities in same chunk
                for i, entity1 in enumerate(entities_in_chunk):
                    for entity2 in entities_in_chunk[i+1:]:
                        if entity1 in node_ids and entity2 in node_ids:
                            edges.append(GraphEdge(
                                source=entity1,
                                target=entity2,
                                type="co-occurs",
                                weight=0.4,
                                label="related"
                            ))
            
            return GraphData(
                nodes=nodes,
                edges=edges,
                metadata={
                    "query": query_text,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "claims_found": len([n for n in nodes if n.type == "claim"]),
                    "entities_found": len([n for n in nodes if n.type == "entity"]),
                    "min_trust": min_trust
                }
            )
            
    except Exception as e:
        logger.error(f"Error getting graph data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        driver.close()

@graph_router.get("/entity/{entity_id}", response_model=GraphData)
async def get_entity_graph(
    entity_id: str,
    max_nodes: int = Query(30, ge=10, le=100),
    depth: int = Query(1, ge=1, le=2)
):
    """
    Get the network graph for a specific entity
    
    Args:
        entity_id: The entity ID
        max_nodes: Maximum nodes to return
        depth: Relationship depth to traverse
    """
    logger.info(f"Getting entity graph for: {entity_id}")
    
    driver = get_driver()
    
    try:
        with driver.session() as session:
            # Get entity and its network
            result = session.run("""
                MATCH (center)
                WHERE center.id = $entity_id OR center.name = $entity_id
                
                // Get relationships up to specified depth
                OPTIONAL MATCH path = (center)-[*1..$depth]-(related)
                
                WITH center, 
                     collect(DISTINCT related) as related_nodes,
                     [r in relationships(path) | r] as all_rels
                
                RETURN 
                    center,
                    related_nodes[0..$max_nodes] as related_nodes,
                    all_rels
            """, {
                "entity_id": entity_id,
                "depth": depth,
                "max_nodes": max_nodes
            })
            
            record = result.single()
            
            if not record or not record["center"]:
                raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
            
            # Process nodes
            nodes = []
            node_ids = set()
            
            # Add center node
            center = record["center"]
            center_id = center.get("id", str(center.id))
            nodes.append(GraphNode(
                id=center_id,
                label=center.get("name") or center.get("title", "Unknown"),
                type="center",
                size=25.0,
                trust_score=center.get("trust_score", 0.5),
                metadata={
                    "description": center.get("description", ""),
                    "is_center": True
                }
            ))
            node_ids.add(center_id)
            
            # Add related nodes
            for node in record["related_nodes"] or []:
                node_id = node.get("id", str(node.id))
                if node_id not in node_ids:
                    nodes.append(GraphNode(
                        id=node_id,
                        label=node.get("name") or node.get("title", "Unknown"),
                        type=list(node.labels)[0].lower() if node.labels else "related",
                        size=12.0 + (node.get("trust_score", 0.5) * 8),
                        trust_score=node.get("trust_score", 0.5),
                        metadata={"description": node.get("description", "")}
                    ))
                    node_ids.add(node_id)
            
            # Process relationships
            edges = []
            seen_edges = set()
            
            for rel in record["all_rels"] or []:
                source_id = rel.start_node.get("id", str(rel.start_node.id))
                target_id = rel.end_node.get("id", str(rel.end_node.id))
                edge_key = f"{source_id}-{target_id}-{rel.type}"
                
                if source_id in node_ids and target_id in node_ids and edge_key not in seen_edges:
                    edges.append(GraphEdge(
                        source=source_id,
                        target=target_id,
                        type=rel.type.lower(),
                        weight=rel.get("weight", 1.0),
                        label=rel.type
                    ))
                    seen_edges.add(edge_key)
            
            return GraphData(
                nodes=nodes,
                edges=edges,
                metadata={
                    "center_entity": entity_id,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "depth": depth
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        driver.close()

@graph_router.get("/clusters", response_model=Dict[str, Any])
async def get_network_clusters(
    min_cluster_size: int = Query(3, ge=2, le=20),
    max_clusters: int = Query(10, ge=1, le=50)
):
    """
    Get major clusters/communities in the trust network
    """
    logger.info("Getting network clusters")
    
    driver = get_driver()
    
    try:
        with driver.session() as session:
            # Simple community detection using node degree
            result = session.run("""
                MATCH (n)-[r]-(m)
                WITH n, count(r) as degree
                WHERE degree >= $min_cluster_size
                ORDER BY degree DESC
                LIMIT $max_clusters
                
                MATCH (n)-[r]-(m)
                
                RETURN 
                    n.id as center_id,
                    n.name as center_name,
                    n.trust_score as trust_score,
                    degree,
                    collect(DISTINCT {id: m.id, name: m.name}) as connected_nodes
            """, {
                "min_cluster_size": min_cluster_size,
                "max_clusters": max_clusters
            })
            
            clusters = []
            for record in result:
                clusters.append({
                    "center_id": record["center_id"],
                    "center_name": record["center_name"],
                    "trust_score": record["trust_score"],
                    "size": record["degree"],
                    "connected_nodes": record["connected_nodes"]
                })
            
            return {
                "clusters": clusters,
                "count": len(clusters)
            }
            
    except Exception as e:
        logger.error(f"Error getting clusters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        driver.close()

@graph_router.get("/health")
async def graph_health():
    """Health check for graph API"""
    return {
        "status": "healthy",
        "component": "Graph Visualization API (FIXED)"
    }
