"""
Data Management API Endpoints for Aegis Insight Admin Panel

Provides endpoints for:
- Listing and managing source files
- Browsing, filtering, and searching claims
- Soft-excluding claims from detection (reversible)
- Hard-deleting claims and sources (permanent)

Author: Aegis Development Team
Created: 2025-12-07
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
from datetime import datetime

# Database connections
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Database configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "aegis_insight")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "aegis")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "aegis_trusted_2025")

# Create router
data_management_router = APIRouter(prefix="/api/admin/data", tags=["data-management"])


# =====================================================
# Pydantic Models
# =====================================================

class ClaimIdsRequest(BaseModel):
    claim_ids: List[str]

class SourceDeleteRequest(BaseModel):
    source_file: str
    
class ClaimSearchRequest(BaseModel):
    search: Optional[str] = None
    source: Optional[str] = None
    claim_type: Optional[str] = None  # PRIMARY, META, SECONDARY, CONTEXTUAL
    status: Optional[str] = "active"  # active, excluded, all
    limit: int = 50
    offset: int = 0


# =====================================================
# Database Helpers
# =====================================================

def get_neo4j_driver():
    """Get Neo4j driver connection"""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_postgres_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )


# =====================================================
# Source Management Endpoints
# =====================================================

@data_management_router.get("/sources")
async def list_sources(
    search: Optional[str] = Query(None, description="Filter sources by filename")
):
    """
    List all source files with claim counts and statistics.
    
    Returns:
        sources: List of source files with metadata
        total_sources: Total number of unique sources
        total_claims: Total claims across all sources
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Build query with optional search filter
            if search:
                query = """
                    MATCH (c:Claim)
                    WHERE c.source_file CONTAINS $search
                    WITH c.source_file AS source,
                         count(c) AS claim_count,
                         sum(CASE WHEN c.excluded = true THEN 1 ELSE 0 END) AS excluded_count,
                         min(c.created_at) AS date_added,
                         collect(DISTINCT c.claim_type) AS claim_types
                    RETURN source, claim_count, excluded_count, date_added, claim_types
                    ORDER BY claim_count DESC
                """
                result = session.run(query, search=search)
            else:
                query = """
                    MATCH (c:Claim)
                    WITH c.source_file AS source,
                         count(c) AS claim_count,
                         sum(CASE WHEN c.excluded = true THEN 1 ELSE 0 END) AS excluded_count,
                         min(c.created_at) AS date_added,
                         collect(DISTINCT c.claim_type) AS claim_types
                    RETURN source, claim_count, excluded_count, date_added, claim_types
                    ORDER BY claim_count DESC
                """
                result = session.run(query)
            
            sources = []
            total_claims = 0
            for record in result:
                source_file = record["source"] or "unknown"
                claim_count = record["claim_count"]
                total_claims += claim_count
                
                sources.append({
                    "source_file": source_file,
                    "display_name": os.path.basename(source_file) if source_file else "unknown",
                    "claim_count": claim_count,
                    "excluded_count": record["excluded_count"] or 0,
                    "active_count": claim_count - (record["excluded_count"] or 0),
                    "date_added": record["date_added"],
                    "claim_types": record["claim_types"] or []
                })
        
        driver.close()
        
        return {
            "success": True,
            "sources": sources,
            "total_sources": len(sources),
            "total_claims": total_claims
        }
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@data_management_router.delete("/sources")
async def delete_source(request: SourceDeleteRequest):
    """
    Permanently delete all claims from a source file.
    
    This removes:
    - All Claim nodes from Neo4j
    - All associated embeddings from PostgreSQL
    - Orphaned Chunk nodes (if no claims remain)
    
    WARNING: This action cannot be undone!
    """
    source_file = request.source_file
    
    try:
        driver = get_neo4j_driver()
        deleted_claims = 0
        deleted_chunks = 0
        claim_ids = []
        
        with driver.session() as session:
            # First, get all claim_ids for PostgreSQL cleanup
            result = session.run("""
                MATCH (c:Claim)
                WHERE c.source_file CONTAINS $source_file
                RETURN c.claim_id AS claim_id
            """, source_file=source_file)
            
            claim_ids = [record["claim_id"] for record in result if record["claim_id"]]
            
            # Delete claims from Neo4j
            result = session.run("""
                MATCH (c:Claim)
                WHERE c.source_file CONTAINS $source_file
                WITH c, c.claim_id AS cid
                DETACH DELETE c
                RETURN count(*) AS deleted
            """, source_file=source_file)
            
            deleted_claims = result.single()["deleted"]
            
            # Delete orphaned chunks (chunks with no remaining claims)
            result = session.run("""
                MATCH (ch:Chunk)
                WHERE ch.source_file CONTAINS $source_file
                AND NOT (ch)-[:CONTAINS_CLAIM]->()
                WITH ch
                DETACH DELETE ch
                RETURN count(*) AS deleted
            """, source_file=source_file)
            
            deleted_chunks = result.single()["deleted"]
            
            # Delete Topic relationships for this document
            result = session.run("""
                MATCH (d:Document)-[r:ABOUT]->(t:Topic)
                WHERE d.source_file CONTAINS $source_file
                DELETE r
                RETURN count(*) AS deleted
            """, source_file=source_file)
            deleted_topic_rels = result.single()["deleted"]
            
            # Find and delete orphaned Topics (no documents linked)
            result = session.run("""
                MATCH (t:Topic)
                WHERE NOT (t)<-[:ABOUT]-(:Document)
                WITH t, t.name AS topic_name
                DETACH DELETE t
                RETURN collect(topic_name) AS orphaned_topics
            """)
            orphaned_topics = result.single()["orphaned_topics"] or []
        
        driver.close()
        
        # Delete orphaned topic embeddings from PostgreSQL
        deleted_topic_embeddings = 0
        if orphaned_topics:
            try:
                conn = get_postgres_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM topic_embeddings 
                        WHERE topic_name = ANY(%s)
                    """, (orphaned_topics,))
                    deleted_topic_embeddings = cur.rowcount
                conn.commit()
                conn.close()
            except Exception as pg_err:
                logger.warning(f"Topic embeddings cleanup error (non-fatal): {pg_err}")
        
        # Delete from PostgreSQL
        deleted_embeddings = 0
        if claim_ids:
            try:
                conn = get_postgres_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM claim_embeddings 
                        WHERE claim_id = ANY(%s)
                    """, (claim_ids,))
                    deleted_embeddings = cur.rowcount
                conn.commit()
                conn.close()
            except Exception as pg_err:
                logger.warning(f"PostgreSQL cleanup error (non-fatal): {pg_err}")
        
        logger.info(f"Deleted source '{source_file}': {deleted_claims} claims, {deleted_chunks} chunks, {deleted_embeddings} embeddings, {len(orphaned_topics)} orphaned topics")
        
        return {
            "success": True,
            "source_file": source_file,
            "deleted": {
                "claims": deleted_claims,
                "chunks": deleted_chunks,
                "embeddings": deleted_embeddings,
                "topic_relationships": deleted_topic_rels,
                "orphaned_topics": len(orphaned_topics),
                "topic_embeddings": deleted_topic_embeddings
            }
        }
        
    except Exception as e:
        logger.error(f"Error deleting source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Claims Management Endpoints
# =====================================================

@data_management_router.get("/claims")
async def list_claims(
    source: Optional[str] = Query(None, description="Filter by source file"),
    claim_type: Optional[str] = Query(None, description="Filter by claim type (PRIMARY, META, etc)"),
    status: Optional[str] = Query("active", description="Filter by status: active, excluded, all"),
    search: Optional[str] = Query(None, description="Search claim text"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    List claims with filtering and pagination.
    
    Filters:
    - source: Filter by source file (partial match)
    - claim_type: PRIMARY, SECONDARY, META, CONTEXTUAL
    - status: active (default), excluded, all
    - search: Full-text search on claim text
    
    Returns paginated results with total count.
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Build dynamic WHERE clauses
            where_clauses = []
            params = {"limit": limit, "offset": offset}
            
            if source:
                where_clauses.append("c.source_file CONTAINS $source")
                params["source"] = source
            
            if claim_type:
                where_clauses.append("c.claim_type = $claim_type")
                params["claim_type"] = claim_type.upper()
            
            if status == "active":
                where_clauses.append("(c.excluded IS NULL OR c.excluded = false)")
            elif status == "excluded":
                where_clauses.append("c.excluded = true")
            # "all" = no status filter
            
            if search:
                where_clauses.append("toLower(c.claim_text) CONTAINS toLower($search)")
                params["search"] = search
            
            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            # Count total matching claims
            count_query = f"""
                MATCH (c:Claim)
                {where_clause}
                RETURN count(c) AS total
            """
            total = session.run(count_query, **params).single()["total"]
            
            # Get paginated claims
            query = f"""
                MATCH (c:Claim)
                {where_clause}
                RETURN c.claim_id AS claim_id,
                       elementId(c) AS element_id,
                       c.claim_text AS claim_text,
                       c.claim_type AS claim_type,
                       c.source_file AS source_file,
                       c.excluded AS excluded,
                       c.created_at AS created_at
                ORDER BY c.source_file, c.claim_id
                SKIP $offset
                LIMIT $limit
            """
            
            result = session.run(query, **params)
            
            claims = []
            for record in result:
                claims.append({
                    "claim_id": record["claim_id"],
                    "element_id": record["element_id"],
                    "claim_text": record["claim_text"],
                    "claim_type": record["claim_type"],
                    "source_file": record["source_file"],
                    "display_source": os.path.basename(record["source_file"]) if record["source_file"] else "unknown",
                    "excluded": record["excluded"] or False,
                    "status": "excluded" if record["excluded"] else "active",
                    "created_at": record["created_at"]
                })
        
        driver.close()
        
        return {
            "success": True,
            "claims": claims,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(claims) < total
        }
        
    except Exception as e:
        logger.error(f"Error listing claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@data_management_router.get("/claims/{claim_id}")
async def get_claim_detail(claim_id: str):
    """
    Get detailed information about a specific claim.
    
    Returns:
    - Full claim text and metadata
    - Related entities
    - Temporal and geographic data
    - Source chunk information
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Get claim with related data
            result = session.run("""
                MATCH (c:Claim {claim_id: $claim_id})
                OPTIONAL MATCH (ch:Chunk)-[:CONTAINS_CLAIM]->(c)
                OPTIONAL MATCH (ch)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (c)-[:HAS_TEMPORAL]->(t:TemporalMarker)
                OPTIONAL MATCH (c)-[:HAS_LOCATION]->(g:GeoLocation)
                WITH c, ch, 
                     collect(DISTINCT {name: e.name, type: e.entity_type}) AS entities,
                     collect(DISTINCT {date: t.date_text, type: t.temporal_type}) AS temporal,
                     collect(DISTINCT {location: g.location_name, lat: g.latitude, lon: g.longitude}) AS geographic
                RETURN c.claim_id AS claim_id,
                       elementId(c) AS element_id,
                       c.claim_text AS claim_text,
                       c.claim_type AS claim_type,
                       c.source_file AS source_file,
                       c.excluded AS excluded,
                       c.created_at AS created_at,
                       c.confidence AS confidence,
                       ch.chunk_id AS chunk_id,
                       ch.text AS chunk_text,
                       entities,
                       temporal,
                       geographic
            """, claim_id=claim_id)
            
            record = result.single()
            
            if not record:
                raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
            
            # Filter out empty entity/temporal/geo entries
            entities = [e for e in record["entities"] if e.get("name")]
            temporal = [t for t in record["temporal"] if t.get("date")]
            geographic = [g for g in record["geographic"] if g.get("location")]
            
            claim_detail = {
                "claim_id": record["claim_id"],
                "element_id": record["element_id"],
                "claim_text": record["claim_text"],
                "claim_type": record["claim_type"],
                "source_file": record["source_file"],
                "display_source": os.path.basename(record["source_file"]) if record["source_file"] else "unknown",
                "excluded": record["excluded"] or False,
                "status": "excluded" if record["excluded"] else "active",
                "created_at": record["created_at"],
                "confidence": record["confidence"],
                "chunk_id": record["chunk_id"],
                "chunk_text": record["chunk_text"][:500] if record["chunk_text"] else None,
                "entities": entities,
                "temporal": temporal,
                "geographic": geographic
            }
        
        driver.close()
        
        return {
            "success": True,
            "claim": claim_detail
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting claim detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Claim Status Management (Exclude/Restore)
# =====================================================

@data_management_router.post("/claims/exclude")
async def exclude_claims(request: ClaimIdsRequest):
    """
    Soft-exclude claims from detection analysis.
    
    Excluded claims:
    - Are skipped during suppression/coordination/anomaly detection
    - Remain in the database and can be restored
    - Are visible in Data Management with 'excluded' status
    
    This is reversible via the /claims/restore endpoint.
    """
    claim_ids = request.claim_ids
    
    if not claim_ids:
        raise HTTPException(status_code=400, detail="No claim_ids provided")
    
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("""
                UNWIND $claim_ids AS cid
                MATCH (c:Claim {claim_id: cid})
                SET c.excluded = true,
                    c.excluded_at = datetime()
                RETURN count(c) AS excluded_count
            """, claim_ids=claim_ids)
            
            excluded_count = result.single()["excluded_count"]
        
        driver.close()
        
        logger.info(f"Excluded {excluded_count} claims: {claim_ids[:5]}...")
        
        return {
            "success": True,
            "excluded_count": excluded_count,
            "claim_ids": claim_ids
        }
        
    except Exception as e:
        logger.error(f"Error excluding claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@data_management_router.post("/claims/restore")
async def restore_claims(request: ClaimIdsRequest):
    """
    Restore previously excluded claims to active status.
    
    Restored claims will be included in future detection analysis.
    """
    claim_ids = request.claim_ids
    
    if not claim_ids:
        raise HTTPException(status_code=400, detail="No claim_ids provided")
    
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("""
                UNWIND $claim_ids AS cid
                MATCH (c:Claim {claim_id: cid})
                SET c.excluded = false,
                    c.restored_at = datetime()
                RETURN count(c) AS restored_count
            """, claim_ids=claim_ids)
            
            restored_count = result.single()["restored_count"]
        
        driver.close()
        
        logger.info(f"Restored {restored_count} claims: {claim_ids[:5]}...")
        
        return {
            "success": True,
            "restored_count": restored_count,
            "claim_ids": claim_ids
        }
        
    except Exception as e:
        logger.error(f"Error restoring claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@data_management_router.delete("/claims")
async def delete_claims(request: ClaimIdsRequest):
    """
    Permanently delete claims from the database.
    
    This removes:
    - Claim nodes from Neo4j (with all relationships)
    - Associated embeddings from PostgreSQL
    
    WARNING: This action cannot be undone!
    """
    claim_ids = request.claim_ids
    
    if not claim_ids:
        raise HTTPException(status_code=400, detail="No claim_ids provided")
    
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("""
                UNWIND $claim_ids AS cid
                MATCH (c:Claim {claim_id: cid})
                DETACH DELETE c
                RETURN count(*) AS deleted_count
            """, claim_ids=claim_ids)
            
            neo4j_deleted = result.single()["deleted_count"]
        
        driver.close()
        
        # Delete from PostgreSQL
        pg_deleted = 0
        try:
            conn = get_postgres_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM claim_embeddings 
                    WHERE claim_id = ANY(%s)
                """, (claim_ids,))
                pg_deleted = cur.rowcount
            conn.commit()
            conn.close()
        except Exception as pg_err:
            logger.warning(f"PostgreSQL cleanup error (non-fatal): {pg_err}")
        
        logger.info(f"Deleted {neo4j_deleted} claims from Neo4j, {pg_deleted} embeddings from PostgreSQL")
        
        return {
            "success": True,
            "deleted": {
                "neo4j_claims": neo4j_deleted,
                "postgres_embeddings": pg_deleted
            },
            "claim_ids": claim_ids
        }
        
    except Exception as e:
        logger.error(f"Error deleting claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Statistics Endpoint
# =====================================================

@data_management_router.get("/stats")
async def get_data_stats():
    """
    Get overall data statistics for the Data Management dashboard.
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Claim)
                WITH count(c) AS total_claims,
                     sum(CASE WHEN c.excluded = true THEN 1 ELSE 0 END) AS excluded_claims,
                     sum(CASE WHEN c.claim_type = 'PRIMARY' THEN 1 ELSE 0 END) AS primary_claims,
                     sum(CASE WHEN c.claim_type = 'META' THEN 1 ELSE 0 END) AS meta_claims,
                     sum(CASE WHEN c.claim_type = 'SECONDARY' THEN 1 ELSE 0 END) AS secondary_claims,
                     sum(CASE WHEN c.claim_type = 'CONTEXTUAL' THEN 1 ELSE 0 END) AS contextual_claims
                MATCH (s:Claim)
                WITH total_claims, excluded_claims, primary_claims, meta_claims, secondary_claims, contextual_claims,
                     count(DISTINCT s.source_file) AS total_sources
                OPTIONAL MATCH (ch:Chunk)
                WITH total_claims, excluded_claims, primary_claims, meta_claims, secondary_claims, contextual_claims,
                     total_sources, count(ch) AS total_chunks
                OPTIONAL MATCH (e:Entity)
                RETURN total_claims, excluded_claims, primary_claims, meta_claims, secondary_claims, contextual_claims,
                       total_sources, total_chunks, count(e) AS total_entities
            """)
            
            record = result.single()
            
            stats = {
                "total_claims": record["total_claims"],
                "active_claims": record["total_claims"] - (record["excluded_claims"] or 0),
                "excluded_claims": record["excluded_claims"] or 0,
                "total_sources": record["total_sources"],
                "total_chunks": record["total_chunks"],
                "total_entities": record["total_entities"],
                "claims_by_type": {
                    "PRIMARY": record["primary_claims"] or 0,
                    "META": record["meta_claims"] or 0,
                    "SECONDARY": record["secondary_claims"] or 0,
                    "CONTEXTUAL": record["contextual_claims"] or 0
                }
            }
        
        driver.close()
        
        # Get embedding count from PostgreSQL
        try:
            conn = get_postgres_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM claim_embeddings")
                stats["total_embeddings"] = cur.fetchone()[0]
            conn.close()
        except Exception as pg_err:
            logger.warning(f"Could not get embedding count: {pg_err}")
            stats["total_embeddings"] = None
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Claim Types Reference
# =====================================================

@data_management_router.get("/claim-types")
async def get_claim_types():
    """
    Get list of claim types present in the database.
    Useful for populating filter dropdowns.
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Claim)
                WHERE c.claim_type IS NOT NULL
                RETURN DISTINCT c.claim_type AS claim_type, count(c) AS count
                ORDER BY count DESC
            """)
            
            claim_types = [
                {"type": record["claim_type"], "count": record["count"]}
                for record in result
            ]
        
        driver.close()
        
        return {
            "success": True,
            "claim_types": claim_types
        }
        
    except Exception as e:
        logger.error(f"Error getting claim types: {e}")
        raise HTTPException(status_code=500, detail=str(e))
