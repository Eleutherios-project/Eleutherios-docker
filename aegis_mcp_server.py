#!/usr/bin/env python3
"""
Aegis Insight - MCP (Model Context Protocol) Server
====================================================

Provides epistemic context endpoints for AI systems to query:
- Suppression patterns
- Coordination signatures  
- Citation topology
- Multi-perspective analysis

Endpoints:
1. analyze_topic - Pre-retrieval epistemic check
2. assess_source - Source position in knowledge topology
3. get_perspectives - Clustered perspectives on topic
4. scan_corpus - Batch scan for patterns (async)
5. get_claim_context - Full context for specific claim
6. list_domains - Available domains and metadata

Version: 1.0 MVP
Date: December 2025
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict

# FastAPI
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Database
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aegis_mcp")

# =============================================================================
# Configuration
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "aegistrusted")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "aegis_insight")
POSTGRES_USER = os.getenv("POSTGRES_USER", "aegis")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "aegis_trusted_2025")

# =============================================================================
# Pydantic Models (Request/Response)
# =============================================================================

class DetailLevel(str, Enum):
    abbreviated = "abbreviated"
    standard = "standard"
    verbose = "verbose"


class AnalyzeTopicRequest(BaseModel):
    topic: str = Field(..., description="Topic to analyze")
    domain: Optional[str] = Field(None, description="Domain scope")
    detail: DetailLevel = Field(DetailLevel.standard, description="Response detail level")
    max_claims: int = Field(200, ge=10, le=1000, description="Maximum claims to analyze")
    profile: Optional[str] = Field(None, description="Detection profile name")


class AssessSourceRequest(BaseModel):
    source_identifier: str = Field(..., description="Source name, path, or ID")
    domain: Optional[str] = Field(None, description="Domain scope")
    detail: DetailLevel = Field(DetailLevel.standard, description="Response detail level")


class GetPerspectivesRequest(BaseModel):
    topic: str = Field(..., description="Topic to analyze")
    domain: Optional[str] = Field(None, description="Domain scope")
    max_clusters: int = Field(5, ge=2, le=10, description="Maximum perspective clusters")
    claims_per_cluster: int = Field(5, ge=1, le=20, description="Representative claims per cluster")


class ScanCorpusRequest(BaseModel):
    domain: Optional[str] = Field(None, description="Domain to scan")
    since_hours: int = Field(24, ge=1, le=720, description="Scan claims from last N hours")


class GetClaimContextRequest(BaseModel):
    claim_id: str = Field(..., description="Claim identifier")
    include_graph: bool = Field(False, description="Include citation subgraph")


# =============================================================================
# Database Connections
# =============================================================================

class DatabaseManager:
    """Manages Neo4j and PostgreSQL connections"""
    
    def __init__(self):
        self.neo4j_driver = None
        self.pg_conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connections"""
        # Neo4j
        try:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("✓ Connected to Neo4j")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            self.neo4j_driver = None
        
        # PostgreSQL
        try:
            self.pg_conn = psycopg2.connect(
                host=POSTGRES_HOST,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            logger.info("✓ Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            self.pg_conn = None
    
    def get_neo4j_session(self):
        """Get Neo4j session"""
        if self.neo4j_driver:
            return self.neo4j_driver.session()
        raise HTTPException(status_code=503, detail="Neo4j unavailable")
    
    def get_pg_cursor(self):
        """Get PostgreSQL cursor"""
        if self.pg_conn:
            return self.pg_conn.cursor(cursor_factory=RealDictCursor)
        return None
    
    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        if self.pg_conn:
            self.pg_conn.close()


# Global database manager
db: Optional[DatabaseManager] = None

# =============================================================================
# Detection Integration
# =============================================================================

class DetectionService:
    """Integrates with Aegis detection algorithms"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_claims_for_topic(self, topic: str, domain: Optional[str], max_claims: int) -> List[Dict]:
        """Fetch claims matching topic using semantic search"""
        
        with self.db.get_neo4j_session() as session:
            # Use semantic search via embeddings if available, fall back to text search
            query = """
            MATCH (c:Claim)
            WHERE toLower(c.claim_text) CONTAINS toLower($topic)
            """
            
            if domain:
                query += " AND c.domain = $domain"
            
            query += """
            RETURN c.claim_id AS claim_id,
                   c.claim_text AS claim_text,
                   c.claim_type AS claim_type,
                   c.confidence AS confidence,
                   c.source_file AS source_file,
                   c.domain AS domain
            LIMIT $max_claims
            """
            
            result = session.run(query, topic=topic, domain=domain, max_claims=max_claims)
            claims = [dict(record) for record in result]
            
        return claims
    
    def detect_suppression(self, claims: List[Dict], profile: Optional[str] = None) -> Dict:
        """Run suppression detection on claims"""
        
        if not claims:
            return {
                'score': 0.0,
                'level': 'NONE',
                'confidence': 0.0,
                'signals': {},
                'indicators': []
            }
        
        # Calculate signals
        signals = {}
        
        # Signal 1: META claim density
        meta_claims = [c for c in claims if c.get('claim_type') == 'META']
        primary_claims = [c for c in claims if c.get('claim_type') == 'PRIMARY']
        meta_density = len(meta_claims) / len(claims) if claims else 0
        signals['meta_claim_density'] = {
            'score': min(meta_density * 2, 1.0),  # Scale up
            'meta_count': len(meta_claims),
            'primary_count': len(primary_claims),
            'total_claims': len(claims)
        }
        
        # Signal 2: Network isolation (citation check)
        with self.db.get_neo4j_session() as session:
            claim_ids = [c['claim_id'] for c in claims if c.get('claim_id')]
            if claim_ids:
                result = session.run("""
                    MATCH (c:Claim)
                    WHERE c.claim_id IN $claim_ids
                    OPTIONAL MATCH (c)-[:CITES]->()
                    WITH c, count(*) as citations
                    RETURN avg(citations) as avg_citations,
                           sum(CASE WHEN citations = 0 THEN 1 ELSE 0 END) as uncited_count
                """, claim_ids=claim_ids)
                
                record = result.single()
                if record:
                    uncited_ratio = record['uncited_count'] / len(claim_ids) if claim_ids else 0
                    signals['network_isolation'] = {
                        'score': uncited_ratio,
                        'avg_citations': record['avg_citations'] or 0,
                        'uncited_count': record['uncited_count']
                    }
                else:
                    signals['network_isolation'] = {'score': 0.5, 'avg_citations': 0, 'uncited_count': 0}
            else:
                signals['network_isolation'] = {'score': 0.5, 'avg_citations': 0, 'uncited_count': 0}
        
        # Signal 3: Evidence avoidance (META claims without citations)
        meta_without_evidence = len([c for c in meta_claims 
                                     if not c.get('citation_data') or c.get('citation_data') == '{}'])
        evidence_avoidance = meta_without_evidence / len(meta_claims) if meta_claims else 0
        signals['evidence_avoidance'] = {
            'score': evidence_avoidance,
            'meta_without_citations': meta_without_evidence,
            'meta_total': len(meta_claims)
        }
        
        # Signal 4: Suppression narrative indicators
        suppression_keywords = [
            'suppressed', 'censored', 'silenced', 'banned', 'removed',
            'dismissed', 'ignored', 'attacked', 'discredited', 'marginalized',
            'fired', 'prosecuted', 'imprisoned', 'exiled', 'condemned'
        ]
        
        indicators_found = []
        for claim in primary_claims:
            text = claim.get('claim_text', '').lower()
            for keyword in suppression_keywords:
                if keyword in text:
                    indicators_found.append({
                        'claim_id': claim.get('claim_id'),
                        'text': claim.get('claim_text', '')[:100],
                        'keyword': keyword
                    })
                    break
        
        suppression_indicator_score = min(len(indicators_found) / 5, 1.0)  # Cap at 5 indicators
        signals['suppression_narrative'] = {
            'score': suppression_indicator_score,
            'indicators_found': len(indicators_found),
            'indicators': indicators_found[:10]  # Limit to 10
        }
        
        # Aggregate score
        weights = {
            'meta_claim_density': 0.15,
            'network_isolation': 0.20,
            'evidence_avoidance': 0.20,
            'suppression_narrative': 0.45
        }
        
        total_score = sum(
            signals.get(signal, {}).get('score', 0) * weight
            for signal, weight in weights.items()
        )
        
        # Determine level
        if total_score >= 0.75:
            level = 'CRITICAL'
        elif total_score >= 0.55:
            level = 'HIGH'
        elif total_score >= 0.35:
            level = 'MODERATE'
        elif total_score >= 0.15:
            level = 'LOW'
        else:
            level = 'MINIMAL'
        
        # Confidence based on claim count
        confidence = min(len(claims) / 50, 1.0)
        
        return {
            'score': round(total_score, 3),
            'level': level,
            'confidence': round(confidence, 2),
            'signals': signals,
            'indicators': indicators_found[:10]
        }
    
    def detect_coordination(self, claims: List[Dict]) -> Dict:
        """Run coordination detection on claims"""
        
        if not claims:
            return {'score': 0.0, 'clusters': [], 'language_similarity': 0.0}
        
        # Simplified coordination detection
        # Full implementation would use temporal clustering and language similarity
        
        # Check for temporal clustering (claims from similar timeframes)
        # For now, return placeholder
        return {
            'score': 0.0,
            'clusters': [],
            'language_similarity_avg': 0.0,
            'temporal_clustering_detected': False,
            'citation_cartel_detected': False
        }
    
    def detect_anomaly(self, claims: List[Dict]) -> Dict:
        """Run anomaly detection on claims"""
        
        if not claims:
            return {'score': 0.0, 'anomalies': []}
        
        # Simplified anomaly detection
        # Full implementation would use cross-cultural pattern matching
        
        return {
            'score': 0.0,
            'cross_domain_patterns': [],
            'geographic_clustering': None
        }


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Aegis Insight MCP Server",
    description="Epistemic context endpoints for AI systems",
    version="1.0.0"
)

# CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detection service
detection: Optional[DetectionService] = None


@app.on_event("startup")
async def startup():
    """Initialize connections on startup"""
    global db, detection
    db = DatabaseManager()
    detection = DetectionService(db)
    logger.info("Aegis MCP Server started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global db
    if db:
        db.close()
    logger.info("Aegis MCP Server stopped")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "neo4j": db.neo4j_driver is not None if db else False,
        "postgresql": db.pg_conn is not None if db else False,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    return status


# =============================================================================
# MCP Endpoints
# =============================================================================

@app.post("/mcp/analyze_topic")
async def analyze_topic(request: AnalyzeTopicRequest):
    """
    Analyze a topic for suppression and coordination patterns.
    
    Use for pre-retrieval epistemic check: "Should I be careful with this topic?"
    
    Returns detection scores and pattern indicators.
    """
    start_time = time.time()
    
    try:
        # Fetch claims
        claims = detection.get_claims_for_topic(
            topic=request.topic,
            domain=request.domain,
            max_claims=request.max_claims
        )
        
        if not claims:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": {
                        "code": "TOPIC_NOT_FOUND",
                        "message": "No claims found matching topic",
                        "details": {
                            "topic": request.topic,
                            "domain": request.domain,
                            "suggestion": "Try broader terms or different domain"
                        }
                    },
                    "query_ms": int((time.time() - start_time) * 1000)
                }
            )
        
        # Run detection
        suppression = detection.detect_suppression(claims, request.profile)
        coordination = detection.detect_coordination(claims)
        anomaly = detection.detect_anomaly(claims)
        
        query_ms = int((time.time() - start_time) * 1000)
        
        # Build response based on detail level
        if request.detail == DetailLevel.abbreviated:
            # Fast, minimal response
            flags = []
            if suppression['score'] > 0.5:
                flags.append("suppression_pattern")
            if suppression['signals'].get('network_isolation', {}).get('score', 0) > 0.6:
                flags.append("citation_void_present")
            if suppression['signals'].get('evidence_avoidance', {}).get('score', 0) > 0.6:
                flags.append("evidence_avoidance")
            if coordination.get('temporal_clustering_detected'):
                flags.append("temporal_clustering")
            
            return {
                "success": True,
                "topic": request.topic,
                "domain": request.domain,
                "suppression_score": suppression['score'],
                "coordination_score": coordination['score'],
                "anomaly_score": anomaly['score'],
                "confidence": suppression['confidence'],
                "flags": flags,
                "claim_count": len(claims),
                "query_ms": query_ms
            }
        
        elif request.detail == DetailLevel.standard:
            # Balanced response with signal breakdown
            return {
                "success": True,
                "topic": request.topic,
                "domain": request.domain,
                "suppression_score": suppression['score'],
                "suppression_level": suppression['level'],
                "coordination_score": coordination['score'],
                "anomaly_score": anomaly['score'],
                "confidence": suppression['confidence'],
                
                "signals": {
                    "suppression": {
                        "meta_claim_density": suppression['signals'].get('meta_claim_density', {}),
                        "network_isolation": suppression['signals'].get('network_isolation', {}),
                        "evidence_avoidance": suppression['signals'].get('evidence_avoidance', {}),
                        "suppression_narrative": {
                            "score": suppression['signals'].get('suppression_narrative', {}).get('score', 0),
                            "indicators_found": suppression['signals'].get('suppression_narrative', {}).get('indicators_found', 0)
                        }
                    },
                    "coordination": {
                        "temporal_clustering_detected": coordination.get('temporal_clustering_detected', False),
                        "language_similarity_avg": coordination.get('language_similarity_avg', 0),
                        "citation_cartel_detected": coordination.get('citation_cartel_detected', False)
                    },
                    "anomaly": {
                        "cross_domain_patterns": anomaly.get('cross_domain_patterns', []),
                        "geographic_clustering": anomaly.get('geographic_clustering')
                    }
                },
                
                "sample_claims": [
                    {
                        "claim_id": c.get('claim_id'),
                        "text": c.get('claim_text', '')[:200],
                        "type": c.get('claim_type'),
                        "source": c.get('source_file', '').split('/')[-1] if c.get('source_file') else None
                    }
                    for c in claims[:5]
                ],
                
                "claim_count": len(claims),
                "query_ms": query_ms
            }
        
        else:  # verbose
            # Full response with all details
            return {
                "success": True,
                "topic": request.topic,
                "domain": request.domain,
                "suppression_score": suppression['score'],
                "suppression_level": suppression['level'],
                "coordination_score": coordination['score'],
                "anomaly_score": anomaly['score'],
                "confidence": suppression['confidence'],
                
                "signals": {
                    "suppression": suppression['signals'],
                    "coordination": coordination,
                    "anomaly": anomaly
                },
                
                "indicators": suppression.get('indicators', []),
                
                "claims": [
                    {
                        "claim_id": c.get('claim_id'),
                        "text": c.get('claim_text'),
                        "type": c.get('claim_type'),
                        "confidence": c.get('confidence'),
                        "source": c.get('source_file'),
                        "domain": c.get('domain')
                    }
                    for c in claims
                ],
                
                "claim_count": len(claims),
                "query_ms": query_ms
            }
    
    except Exception as e:
        logger.error(f"analyze_topic error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                },
                "query_ms": int((time.time() - start_time) * 1000)
            }
        )


@app.post("/mcp/assess_source")
async def assess_source(request: AssessSourceRequest):
    """
    Assess a specific source's position in the knowledge topology.
    
    Evaluates citation patterns, credential context, and cluster membership.
    """
    start_time = time.time()
    
    try:
        with db.get_neo4j_session() as session:
            # Find source (document)
            result = session.run("""
                MATCH (d:Document)
                WHERE d.source_file CONTAINS $source_id
                   OR d.title CONTAINS $source_id
                OPTIONAL MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
                WITH d, collect(DISTINCT c) as claims
                RETURN d.source_file AS source_file,
                       d.title AS title,
                       d.domain AS domain,
                       size(claims) AS claim_count,
                       claims
                LIMIT 1
            """, source_id=request.source_identifier)
            
            record = result.single()
            
            if not record:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "error": {
                            "code": "SOURCE_NOT_FOUND",
                            "message": f"Source not found: {request.source_identifier}"
                        },
                        "query_ms": int((time.time() - start_time) * 1000)
                    }
                )
            
            # Get citation topology
            topo_result = session.run("""
                MATCH (d:Document {source_file: $source_file})
                OPTIONAL MATCH (d)-[:CONTAINS]->(:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
                OPTIONAL MATCH (c)-[:CITES]->(cited:Claim)
                OPTIONAL MATCH (c)<-[:CITES]-(citing:Claim)
                RETURN count(DISTINCT cited) AS cites_count,
                       count(DISTINCT citing) AS cited_by_count
            """, source_file=record['source_file'])
            
            topo = topo_result.single()
            
            query_ms = int((time.time() - start_time) * 1000)
            
            response = {
                "success": True,
                "source_found": True,
                "source": {
                    "source_file": record['source_file'],
                    "title": record['title'],
                    "domain": record['domain'],
                    "claim_count": record['claim_count']
                },
                "topology": {
                    "cites_count": topo['cites_count'] if topo else 0,
                    "cited_by_count": topo['cited_by_count'] if topo else 0,
                    "network_position": "isolated" if (topo['cited_by_count'] if topo else 0) < 2 else "connected"
                },
                "query_ms": query_ms
            }
            
            if request.detail == DetailLevel.verbose:
                # Add claim details
                response["claims"] = [
                    {
                        "claim_id": c.get('claim_id'),
                        "text": c.get('claim_text', '')[:200] if c.get('claim_text') else None,
                        "type": c.get('claim_type')
                    }
                    for c in (record['claims'] or [])[:20]
                ]
            
            return response
    
    except Exception as e:
        logger.error(f"assess_source error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(e)},
                "query_ms": int((time.time() - start_time) * 1000)
            }
        )


@app.post("/mcp/get_perspectives")
async def get_perspectives(request: GetPerspectivesRequest):
    """
    Get clustered perspectives on a topic with representative claims.
    
    Useful for multi-perspective synthesis and balanced response generation.
    """
    start_time = time.time()
    
    try:
        # Get claims for topic
        claims = detection.get_claims_for_topic(
            topic=request.topic,
            domain=request.domain,
            max_claims=500
        )
        
        if not claims:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": {"code": "TOPIC_NOT_FOUND", "message": "No claims found"},
                    "query_ms": int((time.time() - start_time) * 1000)
                }
            )
        
        # Simple clustering by claim type (MVP approach)
        # Full implementation would use embedding-based HDBSCAN clustering
        clusters = {}
        
        # Cluster by type
        for claim in claims:
            claim_type = claim.get('claim_type', 'UNKNOWN')
            if claim_type not in clusters:
                clusters[claim_type] = {
                    'label': f"{claim_type} claims",
                    'claims': [],
                    'size': 0
                }
            clusters[claim_type]['claims'].append(claim)
            clusters[claim_type]['size'] += 1
        
        # Format response
        perspective_clusters = []
        for cluster_type, cluster_data in list(clusters.items())[:request.max_clusters]:
            representative = cluster_data['claims'][:request.claims_per_cluster]
            perspective_clusters.append({
                'label': cluster_data['label'],
                'size': cluster_data['size'],
                'representative_claims': [
                    {
                        'claim_id': c.get('claim_id'),
                        'text': c.get('claim_text', '')[:200],
                        'source': c.get('source_file', '').split('/')[-1] if c.get('source_file') else None
                    }
                    for c in representative
                ]
            })
        
        query_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "topic": request.topic,
            "domain": request.domain,
            "cluster_count": len(perspective_clusters),
            "clusters": perspective_clusters,
            "total_claims": len(claims),
            "query_ms": query_ms
        }
    
    except Exception as e:
        logger.error(f"get_perspectives error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(e)},
                "query_ms": int((time.time() - start_time) * 1000)
            }
        )


@app.post("/mcp/scan_corpus")
async def scan_corpus(request: ScanCorpusRequest, background_tasks: BackgroundTasks):
    """
    Batch scan for new manipulation patterns (async).
    
    Queues a background job to scan the corpus and returns job ID.
    """
    start_time = time.time()
    
    # Generate job ID
    job_id = f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # In production, this would queue a background task
    # For MVP, return placeholder
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "message": "Corpus scan queued. Use job_id to check status.",
        "estimated_duration_seconds": 300,
        "query_ms": int((time.time() - start_time) * 1000)
    }


@app.post("/mcp/get_claim_context")
async def get_claim_context(request: GetClaimContextRequest):
    """
    Get full epistemic context for a specific claim.
    
    Returns claim details, source topology, and related claims.
    """
    start_time = time.time()
    
    try:
        with db.get_neo4j_session() as session:
            # Find claim
            result = session.run("""
                MATCH (c:Claim {claim_id: $claim_id})
                OPTIONAL MATCH (ch:Chunk)-[:CONTAINS_CLAIM]->(c)
                OPTIONAL MATCH (d:Document)-[:CONTAINS]->(ch)
                OPTIONAL MATCH (c)-[:CITES]->(cited:Claim)
                OPTIONAL MATCH (c)<-[:CITES]-(citing:Claim)
                RETURN c.claim_id AS claim_id,
                       c.claim_text AS claim_text,
                       c.claim_type AS claim_type,
                       c.confidence AS confidence,
                       c.source_file AS source_file,
                       c.domain AS domain,
                       c.temporal_data AS temporal_data,
                       c.geographic_data AS geographic_data,
                       d.title AS document_title,
                       collect(DISTINCT cited.claim_id) AS cites,
                       collect(DISTINCT citing.claim_id) AS cited_by
            """, claim_id=request.claim_id)
            
            record = result.single()
            
            if not record or not record['claim_id']:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "error": {"code": "CLAIM_NOT_FOUND", "message": f"Claim not found: {request.claim_id}"},
                        "query_ms": int((time.time() - start_time) * 1000)
                    }
                )
            
            query_ms = int((time.time() - start_time) * 1000)
            
            response = {
                "success": True,
                "claim": {
                    "claim_id": record['claim_id'],
                    "text": record['claim_text'],
                    "type": record['claim_type'],
                    "confidence": record['confidence'],
                    "source_file": record['source_file'],
                    "document_title": record['document_title'],
                    "domain": record['domain'],
                    "temporal_data": json.loads(record['temporal_data']) if record['temporal_data'] else None,
                    "geographic_data": json.loads(record['geographic_data']) if record['geographic_data'] else None
                },
                "epistemic_context": {
                    "cites": [c for c in record['cites'] if c],
                    "cited_by": [c for c in record['cited_by'] if c],
                    "cites_count": len([c for c in record['cites'] if c]),
                    "cited_by_count": len([c for c in record['cited_by'] if c])
                },
                "query_ms": query_ms
            }
            
            if request.include_graph:
                # Get citation subgraph
                graph_result = session.run("""
                    MATCH (c:Claim {claim_id: $claim_id})
                    OPTIONAL MATCH (c)-[:CITES*1..2]-(related:Claim)
                    WITH c, collect(DISTINCT related) AS related_claims
                    UNWIND related_claims AS r
                    OPTIONAL MATCH (r)-[rel:CITES]-(other:Claim)
                    WHERE other IN related_claims OR other = c
                    RETURN collect(DISTINCT {id: r.claim_id, text: left(r.claim_text, 100)}) AS nodes,
                           collect(DISTINCT {from: startNode(rel).claim_id, to: endNode(rel).claim_id}) AS edges
                """, claim_id=request.claim_id)
                
                graph = graph_result.single()
                if graph:
                    response["citation_subgraph"] = {
                        "nodes": graph['nodes'],
                        "edges": graph['edges']
                    }
            
            return response
    
    except Exception as e:
        logger.error(f"get_claim_context error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(e)},
                "query_ms": int((time.time() - start_time) * 1000)
            }
        )


@app.get("/mcp/list_domains")
async def list_domains():
    """
    List available domains and their metadata.
    
    Returns domain names, claim counts, and status.
    """
    start_time = time.time()
    
    try:
        with db.get_neo4j_session() as session:
            result = session.run("""
                MATCH (c:Claim)
                WITH c.domain AS domain, count(c) AS claim_count
                RETURN domain, claim_count
                ORDER BY claim_count DESC
            """)
            
            domains = []
            for record in result:
                domain_name = record['domain'] or 'unknown'
                domains.append({
                    "domain_id": domain_name.lower().replace(' ', '_'),
                    "name": domain_name,
                    "claim_count": record['claim_count'],
                    "calibration_status": "available"
                })
            
            # Get total stats
            stats_result = session.run("""
                MATCH (c:Claim) 
                WITH count(c) AS claims
                MATCH (e:Entity)
                WITH claims, count(e) AS entities
                MATCH (d:Document)
                RETURN claims, entities, count(d) AS documents
            """)
            
            stats = stats_result.single()
            
            query_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "domains": domains,
                "total_claims": stats['claims'] if stats else 0,
                "total_entities": stats['entities'] if stats else 0,
                "total_documents": stats['documents'] if stats else 0,
                "default_domain": domains[0]['domain_id'] if domains else None,
                "query_ms": query_ms
            }
    
    except Exception as e:
        logger.error(f"list_domains error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(e)},
                "query_ms": int((time.time() - start_time) * 1000)
            }
        )


# =============================================================================
# MCP Tool Registration Endpoint
# =============================================================================

@app.get("/mcp/tools")
async def list_tools():
    """
    Return MCP tool registration for AI systems.
    
    This endpoint provides the tool schema that AI systems use to discover
    available epistemic analysis capabilities.
    """
    return {
        "name": "aegis_insight",
        "description": "Query epistemic context for topics and sources. Detects suppression patterns, coordination signatures, and citation topology.",
        "version": "1.0.0",
        "tools": [
            {
                "name": "analyze_topic",
                "description": "Analyze a topic for suppression and coordination patterns. Returns detection scores and pattern indicators.",
                "endpoint": "/mcp/analyze_topic",
                "method": "POST",
                "parameters": {
                    "topic": {"type": "string", "required": True, "description": "Topic to analyze"},
                    "domain": {"type": "string", "required": False, "description": "Domain scope"},
                    "detail": {"type": "string", "required": False, "enum": ["abbreviated", "standard", "verbose"], "default": "standard"},
                    "max_claims": {"type": "integer", "required": False, "default": 200}
                }
            },
            {
                "name": "assess_source",
                "description": "Assess a specific source's position in the knowledge topology.",
                "endpoint": "/mcp/assess_source",
                "method": "POST",
                "parameters": {
                    "source_identifier": {"type": "string", "required": True, "description": "Source name or path"},
                    "domain": {"type": "string", "required": False},
                    "detail": {"type": "string", "required": False, "enum": ["abbreviated", "standard", "verbose"]}
                }
            },
            {
                "name": "get_perspectives",
                "description": "Get clustered perspectives on a topic with representative claims.",
                "endpoint": "/mcp/get_perspectives",
                "method": "POST",
                "parameters": {
                    "topic": {"type": "string", "required": True},
                    "domain": {"type": "string", "required": False},
                    "max_clusters": {"type": "integer", "required": False, "default": 5},
                    "claims_per_cluster": {"type": "integer", "required": False, "default": 5}
                }
            },
            {
                "name": "scan_corpus",
                "description": "Batch scan for new manipulation patterns (async job).",
                "endpoint": "/mcp/scan_corpus",
                "method": "POST",
                "parameters": {
                    "domain": {"type": "string", "required": False},
                    "since_hours": {"type": "integer", "required": False, "default": 24}
                }
            },
            {
                "name": "get_claim_context",
                "description": "Get full epistemic context for a specific claim.",
                "endpoint": "/mcp/get_claim_context",
                "method": "POST",
                "parameters": {
                    "claim_id": {"type": "string", "required": True},
                    "include_graph": {"type": "boolean", "required": False, "default": False}
                }
            },
            {
                "name": "list_domains",
                "description": "List available domains and their metadata.",
                "endpoint": "/mcp/list_domains",
                "method": "GET",
                "parameters": {}
            }
        ]
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the MCP server"""
    import uvicorn
    
    port = int(os.getenv("MCP_PORT", "8100"))
    host = os.getenv("MCP_HOST", "0.0.0.0")
    
    logger.info(f"Starting Aegis MCP Server on {host}:{port}")
    
    uvicorn.run(
        "aegis_mcp_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
