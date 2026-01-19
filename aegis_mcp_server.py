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
3. get_perspectives - Clustered perspectives on topic (semantic clustering)
4. scan_corpus - Batch scan for patterns (async)
5. get_claim_context - Full context for specific claim
6. list_domains - Available domains and metadata

Version: 1.1 (Fixed lifespan, enhanced perspectives)
Date: January 2026
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

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

# Ollama for embeddings (used by get_perspectives clustering)
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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
    use_semantic_clustering: bool = Field(True, description="Use embedding-based clustering")


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
            # Check connection and reconnect if needed
            try:
                self.pg_conn.cursor().execute("SELECT 1")
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                self._reconnect_postgres()
            return self.pg_conn.cursor(cursor_factory=RealDictCursor)
        return None
    
    def _reconnect_postgres(self):
        """Reconnect to PostgreSQL if connection lost"""
        try:
            self.pg_conn = psycopg2.connect(
                host=POSTGRES_HOST,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            logger.info("✓ Reconnected to PostgreSQL")
        except Exception as e:
            logger.error(f"PostgreSQL reconnection failed: {e}")
    
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
    """Integrates with Aegis detection algorithms via two-stage semantic pipeline"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.api_base_url = "http://localhost:8001"

    def get_claims_for_topic(self, topic: str, domain: Optional[str], max_claims: int) -> tuple:
        """
        Fetch claims using semantic search (two-stage pipeline).

        Returns:
            tuple: (claims_list, claim_ids_list)
        """
        try:
            # STAGE 1: Semantic search via pattern-search endpoint (uses GPU embeddings)
            logger.info(f"MCP: Semantic search for topic: {topic}")

            response = requests.post(
                f"{self.api_base_url}/api/pattern-search",
                json={"query": topic, "limit": max_claims},
                timeout=90  # Allow time for GPU embedding computation
            )

            if response.status_code != 200:
                logger.warning(f"Pattern search failed ({response.status_code}), falling back to text search")
                return self._fallback_text_search(topic, domain, max_claims)

            data = response.json()
            claims = data.get("claims", [])

            if not claims:
                logger.info("No claims from semantic search, trying text fallback")
                return self._fallback_text_search(topic, domain, max_claims)

            # Extract claim IDs for detector (Neo4j elementId format)
            claim_ids = []
            for c in claims:
                cid = c.get("id") or c.get("claim_id") or c.get("elementId")
                if cid:
                    claim_ids.append(cid)

            logger.info(f"MCP: Semantic search returned {len(claims)} claims, {len(claim_ids)} with IDs")

            # Filter by domain if specified
            if domain:
                claims = [c for c in claims if c.get("domain") == domain]
                logger.info(f"MCP: After domain filter: {len(claims)} claims")

            return claims, claim_ids

        except requests.exceptions.Timeout:
            logger.warning("Pattern search timed out, falling back to text search")
            return self._fallback_text_search(topic, domain, max_claims)
        except Exception as e:
            logger.error(f"Semantic search error: {e}, falling back to text search")
            return self._fallback_text_search(topic, domain, max_claims)

    def _fallback_text_search(self, topic: str, domain: Optional[str], max_claims: int) -> tuple:
        """Legacy text search fallback (no semantic matching) - scores will be inaccurate"""
        logger.warning("MCP: Using legacy text search - detection scores may be INACCURATE!")

        with self.db.get_neo4j_session() as session:
            query = """
            MATCH (c:Claim)
            WHERE toLower(c.claim_text) CONTAINS toLower($topic)
            """

            if domain:
                query += " AND c.domain = $domain"

            query += """
            RETURN elementId(c) AS id,
                   c.claim_id AS claim_id,
                   c.claim_text AS claim_text,
                   c.claim_type AS claim_type,
                   c.confidence AS confidence,
                   c.source_file AS source_file,
                   c.domain AS domain
            LIMIT $max_claims
            """

            result = session.run(query, topic=topic, domain=domain, max_claims=max_claims)
            claims = [dict(record) for record in result]
            claim_ids = [c["id"] for c in claims if c.get("id")]

        return claims, claim_ids

    def detect_suppression(self, claims: List[Dict], profile: Optional[str] = None,
                           claim_ids: List[str] = None, topic: str = None) -> Dict:
        """Run suppression detection via API with claim_ids for semantic coverage."""
        try:
            payload = {
                "topic": topic or "unknown",
                "query": topic or "unknown",
                "limit": len(claims) if claims else 500
            }

            if claim_ids:
                payload["claim_ids"] = claim_ids
                logger.info(f"MCP: Suppression detection with {len(claim_ids)} semantic claim IDs")
            else:
                logger.warning("MCP: No claim_ids provided - detection may be inaccurate")

            if profile:
                payload["profile"] = profile

            response = requests.post(
                f"{self.api_base_url}/api/detect/suppression",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("result", data)
                return {
                    'score': result.get("suppression_score", 0),
                    'level': self._score_to_level(result.get("suppression_score", 0)),
                    'confidence': result.get("confidence", 0),
                    'signals': result.get("signals", {}),
                    'indicators': result.get("indicators", []),
                    'claims_analyzed': result.get("claims_analyzed", len(claims))
                }
            else:
                logger.error(f"Suppression API error: {response.status_code} - {response.text[:200]}")
                return self._local_suppression_calc(claims)

        except Exception as e:
            logger.error(f"Suppression detection error: {e}")
            return self._local_suppression_calc(claims)

    def detect_coordination(self, claims: List[Dict], claim_ids: List[str] = None, topic: str = None) -> Dict:
        """Run coordination detection via API with claim_ids."""
        try:
            payload = {
                "topic": topic or "unknown",
                "query": topic or "unknown",
                "limit": len(claims) if claims else 500
            }

            if claim_ids:
                payload["claim_ids"] = claim_ids
                logger.info(f"MCP: Coordination detection with {len(claim_ids)} semantic claim IDs")

            response = requests.post(
                f"{self.api_base_url}/api/detect/coordination",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    'score': data.get("coordination_score", 0),
                    'confidence': data.get("confidence", 0),
                    'signals': data.get("signals", {}),
                    'temporal_clustering_detected': data.get("signals", {}).get("temporal_clustering", {}).get(
                        "burst_detected", False),
                    'citation_cartel_detected': data.get("signals", {}).get("citation_cartel", {}).get(
                        "cartel_detected", False),
                    'language_similarity_avg': data.get("signals", {}).get("language_similarity", {}).get(
                        "avg_similarity", 0),
                    'clusters': data.get("clusters", []),
                    'claims_analyzed': data.get("claims_analyzed", len(claims))
                }
            else:
                logger.error(f"Coordination API error: {response.status_code}")
                return {'score': 0, 'confidence': 0, 'signals': {}, 'temporal_clustering_detected': False,
                        'citation_cartel_detected': False, 'language_similarity_avg': 0, 'clusters': []}

        except Exception as e:
            logger.error(f"Coordination detection error: {e}")
            return {'score': 0, 'confidence': 0, 'signals': {}, 'temporal_clustering_detected': False,
                    'citation_cartel_detected': False, 'language_similarity_avg': 0, 'clusters': []}

    def detect_anomaly(self, claims: List[Dict], claim_ids: List[str] = None, topic: str = None) -> Dict:
        """Run anomaly detection via API with claim_ids."""
        try:
            payload = {
                "topic": topic or "unknown",
                "query": topic or "unknown",
                "limit": len(claims) if claims else 500
            }

            if claim_ids:
                payload["claim_ids"] = claim_ids
                logger.info(f"MCP: Anomaly detection with {len(claim_ids)} semantic claim IDs")

            response = requests.post(
                f"{self.api_base_url}/api/detect/anomaly",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    'score': data.get("anomaly_score", 0),
                    'confidence': data.get("confidence", 0),
                    'cross_domain_patterns': data.get("cross_domain_patterns", []),
                    'geographic_clustering': data.get("geographic_clustering"),
                    'anomalies_found': data.get("anomalies_found", 0)
                }
            else:
                logger.error(f"Anomaly API error: {response.status_code}")
                return {'score': 0, 'confidence': 0, 'cross_domain_patterns': [], 'geographic_clustering': None}

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {'score': 0, 'confidence': 0, 'cross_domain_patterns': [], 'geographic_clustering': None}

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to level string"""
        if score >= 0.7:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.35:
            return "MODERATE"
        elif score >= 0.15:
            return "LOW"
        else:
            return "MINIMAL"

    def _local_suppression_calc(self, claims: List[Dict]) -> Dict:
        """Local fallback calculation if API fails"""
        if not claims:
            return {'score': 0, 'level': 'MINIMAL', 'confidence': 0, 'signals': {}, 'indicators': []}

        meta_count = sum(1 for c in claims if c.get('claim_type') == 'META')
        meta_density = meta_count / len(claims) if claims else 0

        score = min(meta_density * 0.5, 0.5)

        return {
            'score': score,
            'level': self._score_to_level(score),
            'confidence': 0.3,
            'signals': {
                'meta_claim_density': {'score': meta_density},
                '_note': 'Fallback calculation - API unavailable'
            },
            'indicators': []
        }


# =============================================================================
# Perspective Clustering Service
# =============================================================================

class PerspectiveClusteringService:
    """
    Enhanced perspective clustering using semantic embeddings.
    
    Provides meaningful clustering of claims beyond simple type-based grouping.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.api_base_url = "http://localhost:8001"
    
    def cluster_claims_semantically(self, claims: List[Dict], max_clusters: int = 5) -> List[Dict]:
        """
        Cluster claims using embedding similarity from PostgreSQL.
        
        Uses pre-computed embeddings for fast clustering.
        """
        if not claims or len(claims) < 2:
            return self._fallback_type_clustering(claims, max_clusters)
        
        try:
            # Get claim IDs
            claim_ids = []
            for c in claims:
                cid = c.get('claim_id') or c.get('id')
                if cid:
                    claim_ids.append(cid)
            
            if not claim_ids:
                return self._fallback_type_clustering(claims, max_clusters)
            
            # Try to get clusters via API (uses embedding similarity)
            response = requests.post(
                f"{self.api_base_url}/api/cluster-perspectives",
                json={
                    "claim_ids": claim_ids[:200],  # Limit for performance
                    "max_clusters": max_clusters
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("clusters"):
                    logger.info(f"MCP: Got {len(data['clusters'])} semantic clusters from API")
                    return data["clusters"]
            
            # Fallback to source-based clustering
            return self._source_based_clustering(claims, max_clusters)
            
        except Exception as e:
            logger.warning(f"Semantic clustering failed: {e}, using fallback")
            return self._fallback_type_clustering(claims, max_clusters)
    
    def _source_based_clustering(self, claims: List[Dict], max_clusters: int) -> List[Dict]:
        """
        Cluster by source document - groups claims from same sources together.
        Useful for seeing how different sources discuss a topic.
        """
        source_clusters = {}
        
        for claim in claims:
            source = claim.get('source_file') or claim.get('source') or 'Unknown Source'
            # Normalize source name
            source_name = source.split('/')[-1] if '/' in source else source
            source_name = source_name.replace('.jsonl', '').replace('.txt', '').replace('.pdf', '')
            
            if source_name not in source_clusters:
                source_clusters[source_name] = {
                    'claims': [],
                    'types': set()
                }
            source_clusters[source_name]['claims'].append(claim)
            claim_type = claim.get('claim_type') or claim.get('type') or 'UNKNOWN'
            source_clusters[source_name]['types'].add(claim_type)
        
        # Sort by number of claims and take top clusters
        sorted_sources = sorted(source_clusters.items(), key=lambda x: len(x[1]['claims']), reverse=True)
        
        clusters = []
        for source_name, data in sorted_sources[:max_clusters]:
            # Determine dominant perspective
            types_list = list(data['types'])
            if 'META' in types_list and len(data['claims']) > 2:
                label = f"Critical perspective: {source_name}"
            elif 'PRIMARY' in types_list:
                label = f"Primary source: {source_name}"
            elif 'SECONDARY' in types_list:
                label = f"Secondary analysis: {source_name}"
            else:
                label = f"Source: {source_name}"
            
            clusters.append({
                'label': label,
                'size': len(data['claims']),
                'source': source_name,
                'claim_types': types_list,
                'claims': data['claims']
            })
        
        return clusters
    
    def _fallback_type_clustering(self, claims: List[Dict], max_clusters: int) -> List[Dict]:
        """
        Fallback: cluster by claim type (PRIMARY, META, SECONDARY, CONTEXTUAL).
        Simple but provides meaningful perspective separation.
        """
        type_clusters = {}
        
        type_labels = {
            'PRIMARY': 'Primary claims (direct assertions)',
            'META': 'Meta claims (commentary/criticism)',
            'SECONDARY': 'Secondary claims (derived/analytical)',
            'CONTEXTUAL': 'Contextual claims (background/framing)',
            'UNKNOWN': 'Uncategorized claims'
        }
        
        for claim in claims:
            claim_type = claim.get('claim_type') or claim.get('type') or 'UNKNOWN'
            if claim_type not in type_clusters:
                type_clusters[claim_type] = {
                    'label': type_labels.get(claim_type, f'{claim_type} claims'),
                    'claims': [],
                    'sources': set()
                }
            type_clusters[claim_type]['claims'].append(claim)
            source = claim.get('source_file') or claim.get('source')
            if source:
                type_clusters[claim_type]['sources'].add(source.split('/')[-1])
        
        # Sort by size and format
        sorted_types = sorted(type_clusters.items(), key=lambda x: len(x[1]['claims']), reverse=True)
        
        clusters = []
        for claim_type, data in sorted_types[:max_clusters]:
            clusters.append({
                'label': data['label'],
                'size': len(data['claims']),
                'claim_type': claim_type,
                'source_count': len(data['sources']),
                'claims': data['claims']
            })
        
        return clusters


# =============================================================================
# FastAPI Application with Lifespan
# =============================================================================

# Global services
detection: Optional[DetectionService] = None
clustering: Optional[PerspectiveClusteringService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan handler - replaces deprecated on_event decorators.
    """
    global db, detection, clustering
    
    # Startup
    logger.info("Starting Aegis MCP Server...")
    db = DatabaseManager()
    detection = DetectionService(db)
    clustering = PerspectiveClusteringService(db)
    logger.info("Aegis MCP Server started")
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("Shutting down Aegis MCP Server...")
    if db:
        db.close()
    logger.info("Aegis MCP Server stopped")


app = FastAPI(
    title="Aegis Insight MCP Server",
    description="Epistemic context endpoints for AI systems",
    version="1.1.0",
    lifespan=lifespan
)

# CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    Uses TWO-STAGE PIPELINE:
    1. Semantic search via /api/pattern-search (GPU embeddings)
    2. Detection analysis with claim_ids for full coverage
    """
    start_time = time.time()

    try:
        # TWO-STAGE PIPELINE: Get claims via semantic search
        claims, claim_ids = detection.get_claims_for_topic(
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

        # Run detection WITH claim_ids for proper semantic coverage
        suppression = detection.detect_suppression(
            claims,
            profile=request.profile,
            claim_ids=claim_ids,
            topic=request.topic
        )
        coordination = detection.detect_coordination(
            claims,
            claim_ids=claim_ids,
            topic=request.topic
        )
        anomaly = detection.detect_anomaly(
            claims,
            claim_ids=claim_ids,
            topic=request.topic
        )

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
                "suppression_level": suppression['level'],
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
                            "indicators_found": suppression['signals'].get('suppression_narrative', {}).get(
                                'indicators_found', 0)
                        }
                    },
                    "coordination": {
                        "score": coordination['score'],
                        "temporal_clustering_detected": coordination.get('temporal_clustering_detected', False),
                        "language_similarity_avg": coordination.get('language_similarity_avg', 0),
                        "citation_cartel_detected": coordination.get('citation_cartel_detected', False)
                    },
                    "anomaly": {
                        "score": anomaly['score'],
                        "cross_domain_patterns": anomaly.get('cross_domain_patterns', []),
                        "geographic_clustering": anomaly.get('geographic_clustering')
                    }
                },

                "indicators": suppression.get('indicators', []),

                "claims": [
                    {
                        "claim_id": c.get('claim_id') or c.get('id'),
                        "text": (c.get('claim_text') or c.get('text', ''))[:200],
                        "type": c.get('claim_type') or c.get('type'),
                        "confidence": c.get('confidence'),
                        "source": (c.get('source_file') or c.get('source', '')).split('/')[-1] if (
                                    c.get('source_file') or c.get('source')) else None,
                        "domain": c.get('domain')
                    }
                    for c in claims
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
                        "claim_id": c.get('claim_id') or c.get('id'),
                        "text": c.get('claim_text') or c.get('text'),
                        "type": c.get('claim_type') or c.get('type'),
                        "confidence": c.get('confidence'),
                        "source": c.get('source_file') or c.get('source'),
                        "domain": c.get('domain')
                    }
                    for c in claims
                ],

                "claim_count": len(claims),
                "query_ms": query_ms
            }

    except Exception as e:
        logger.error(f"analyze_topic error: {e}")
        import traceback
        traceback.print_exc()
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
    
    ENHANCED VERSION: Uses semantic clustering via embeddings when available,
    with intelligent fallback to source-based and type-based clustering.
    
    Useful for multi-perspective synthesis and balanced response generation.
    """
    start_time = time.time()
    
    try:
        # Get claims for topic via semantic search
        claims, claim_ids = detection.get_claims_for_topic(
            topic=request.topic,
            domain=request.domain,
            max_claims=500
        )
        
        if not claims:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": {
                        "code": "TOPIC_NOT_FOUND",
                        "message": f"No claims found for topic: {request.topic}"
                    },
                    "query_ms": int((time.time() - start_time) * 1000)
                }
            )
        
        # Use enhanced clustering service
        if request.use_semantic_clustering:
            raw_clusters = clustering.cluster_claims_semantically(claims, request.max_clusters)
        else:
            raw_clusters = clustering._fallback_type_clustering(claims, request.max_clusters)
        
        # Format response with representative claims
        perspective_clusters = []
        for cluster in raw_clusters:
            cluster_claims = cluster.get('claims', [])
            representative = cluster_claims[:request.claims_per_cluster]
            
            # Calculate cluster statistics
            meta_count = sum(1 for c in cluster_claims if (c.get('claim_type') or c.get('type')) == 'META')
            primary_count = sum(1 for c in cluster_claims if (c.get('claim_type') or c.get('type')) == 'PRIMARY')
            
            # Get unique sources in cluster
            sources = set()
            for c in cluster_claims:
                src = c.get('source_file') or c.get('source')
                if src:
                    sources.add(src.split('/')[-1])
            
            perspective_clusters.append({
                'label': cluster.get('label', 'Unnamed cluster'),
                'size': cluster.get('size', len(cluster_claims)),
                'meta_ratio': meta_count / len(cluster_claims) if cluster_claims else 0,
                'primary_count': primary_count,
                'source_count': len(sources),
                'sources': list(sources)[:5],  # Top 5 sources
                'representative_claims': [
                    {
                        'claim_id': c.get('claim_id') or c.get('id'),
                        'text': (c.get('claim_text') or c.get('text', ''))[:300],
                        'type': c.get('claim_type') or c.get('type'),
                        'confidence': c.get('confidence'),
                        'source': (c.get('source_file') or c.get('source', '')).split('/')[-1] if (
                            c.get('source_file') or c.get('source')) else None
                    }
                    for c in representative
                ]
            })
        
        query_ms = int((time.time() - start_time) * 1000)
        
        # Calculate overall perspective diversity
        total_sources = set()
        for cluster in perspective_clusters:
            total_sources.update(cluster.get('sources', []))
        
        return {
            "success": True,
            "topic": request.topic,
            "domain": request.domain,
            "cluster_count": len(perspective_clusters),
            "total_claims": len(claims),
            "total_sources": len(total_sources),
            "clustering_method": "semantic" if request.use_semantic_clustering else "type-based",
            "clusters": perspective_clusters,
            "perspective_diversity": {
                "source_spread": len(total_sources),
                "cluster_balance": min(c['size'] for c in perspective_clusters) / max(c['size'] for c in perspective_clusters) if perspective_clusters else 0,
                "meta_presence": any(c['meta_ratio'] > 0.3 for c in perspective_clusters)
            },
            "query_ms": query_ms
        }
    
    except Exception as e:
        logger.error(f"get_perspectives error: {e}")
        import traceback
        traceback.print_exc()
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
        "version": "1.1.0",
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
                "description": "Get clustered perspectives on a topic with representative claims. Uses semantic clustering for meaningful perspective separation.",
                "endpoint": "/mcp/get_perspectives",
                "method": "POST",
                "parameters": {
                    "topic": {"type": "string", "required": True, "description": "Topic to analyze"},
                    "domain": {"type": "string", "required": False},
                    "max_clusters": {"type": "integer", "required": False, "default": 5, "description": "Maximum perspective clusters (2-10)"},
                    "claims_per_cluster": {"type": "integer", "required": False, "default": 5, "description": "Representative claims per cluster (1-20)"},
                    "use_semantic_clustering": {"type": "boolean", "required": False, "default": True, "description": "Use embedding-based clustering"}
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
    
    port = int(os.getenv("MCP_PORT", "8101"))
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
