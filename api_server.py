#!/usr/bin/env python3
"""
AegisTrustNet - Standalone API Server (Fixed)
This version properly serves static files from the /web/ directory
"""

import os
import logging
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional
from src.api_graph_extension import graph_router
from api_data_loading_extension import data_router
from api_ollama_extension import register_ollama_routes
from pattern_search_fixed import PatternSearchFixed
from pattern_search_llm import PatternSearchLLM
from aegis_detection_system_builder import DetectionSystemBuilder
from aegis_detection_setup_jobs import register_jobs
from aegis_suppression_detector import SuppressionDetector
from aegis_coordination_detector import CoordinationDetector
from aegis_anomaly_detector import AnomalyDetector
from aegis_detection_config import get_config, update_config, reset_config
from aegis_data_quality_checker import DataQualityChecker
from calibration_api_endpoints import calibration_router
from data_management_api import data_management_router



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('aegis-api-server')

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Load environment variables
config_env = os.path.join(BASE_DIR, "config", ".env")
if os.path.exists(config_env):
    load_dotenv(config_env)
else:
    load_dotenv()  # Load from current directory if config/.env doesn't exist


# === Request Models ===
class DetectionSetupRequest(BaseModel):
    """Request model for detection setup endpoint"""
    detector: Optional[str] = None
    all: Optional[bool] = False


def main():
    # Create FastAPI app
    app = FastAPI(
        title="AegisTrustNet API", 
        description="Truth Network Overlay with Multi-Perspective Analysis",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )

    # Initialize detection builder
    detection_builder = DetectionSystemBuilder()
    detection_builder = register_jobs(detection_builder)

    # Initialize pattern searcher with LLM
    global pattern_searcher_llm
    try:
        pattern_searcher_llm = PatternSearchLLM(
            neo4j_uri=os.environ.get("NEO4J_URI", os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
            neo4j_user="neo4j",
            neo4j_password="aegistrusted",  # Update if different
            postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
            postgres_db="aegis_insight",
            postgres_user="aegis",
            postgres_password="aegis_trusted_2025",  # Update if different
            ollama_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model="mistral-nemo:12b"  # or "qwen2.5:72b" for higher quality
        )
        logger.info("✓ Pattern searcher with LLM initialized successfully")
    except Exception as e:
        logger.error(f"Could not initialize pattern searcher: {str(e)}")
        pattern_searcher_llm = None

    # Import routers
    try:
        from src.api_extension import router as trust_router
        app.include_router(trust_router)
        logger.info("✓ Trust network router registered successfully")
    except ImportError as e:
        logger.error(f"✗ Could not import trust network router: {str(e)}")
    
    try:
        from src.api_integration import enhanced_router
        app.include_router(enhanced_router)
        app.include_router(graph_router)
        app.include_router(data_router)
        register_ollama_routes(app)
        app.include_router(calibration_router)
        app.include_router(data_management_router)
        logger.info("✓ Enhanced API, Calibration, Data Management and graph routers registered successfully")

        # Initialize pattern searcher
        try:
            global pattern_searcher
            pattern_searcher = PatternSearchFixed(
                neo4j_uri=os.environ.get("NEO4J_URI", os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
                neo4j_user="neo4j",
                neo4j_password="aegistrusted"
            )
            logger.info("Pattern searcher initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize pattern searcher: {str(e)}")
            pattern_searcher = None
    except ImportError as e:
        logger.error(f"Could not import enhanced API router: {str(e)}")
    
    # Root endpoint - redirect to index.html
    @app.get("/")
    async def root():
        """Serve the main web interface"""
        index_path = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {
            "status": "online",
            "service": "AegisTrustNet API",
            "message": "Web interface not found. Please ensure /web/index.html exists.",
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "trust": "/trust",
                "enhanced": "/enhanced",
                "graph": "/graph"
            }
        }

    @app.get("/api/health")
    async def health_check():
        """
        Health check endpoint for container orchestration.

        Returns:
            - status: 'healthy' if all systems nominal
            - timestamp: current UTC time
            - services: individual service status
        """
        services = {
            "api": True,
            "neo4j": False,
            "postgres": False
        }

        # Check Neo4j
        try:
            from neo4j import GraphDatabase
            neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
            neo4j_password = os.environ.get("NEO4J_PASSWORD", "aegistrusted")
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            services["neo4j"] = True
        except Exception as e:
            services["neo4j_error"] = str(e)

        # Check PostgreSQL
        try:
            import psycopg2
            pg_host = os.environ.get("POSTGRES_HOST", "localhost")
            pg_db = os.environ.get("POSTGRES_DB", "aegis_insight")
            pg_user = os.environ.get("POSTGRES_USER", "aegis")
            pg_password = os.environ.get("POSTGRES_PASSWORD", "aegis_trusted_2025")
            conn = psycopg2.connect(
                host=pg_host,
                database=pg_db,
                user=pg_user,
                password=pg_password,
                connect_timeout=5
            )
            conn.close()
            services["postgres"] = True
        except Exception as e:
            services["postgres_error"] = str(e)

        # Overall health
        all_healthy = services["api"] and services["neo4j"] and services["postgres"]

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": services
        }

    @app.get("/api/health/simple")
    async def health_check_simple():
        """
        Simple health check for Docker HEALTHCHECK directive.
        Returns 200 if API is responding, regardless of backend status.
        """
        return {"status": "ok"}

    # Health check endpoint
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "AegisTrustNet API",
            "web_directory": WEB_DIR,
            "web_exists": os.path.exists(WEB_DIR)
        }
    
    # Pattern search endpoint
    @app.post("/api/pattern-search")
    async def pattern_search_endpoint(request: dict):
        """
        Pattern search with LLM synthesis

        Request body: {"query": "search terms"}
        Returns: {
            "query": str,
            "synthesis": str,  # LLM-generated summary
            "claims": [...],   # Top matching claims
            "total_claims": int,
            "search_stats": {...}
        }
        """
        if pattern_searcher_llm is None:
            return {
                "error": "Pattern searcher not initialized",
                "query": request.get('query', ''),
                "synthesis": "Pattern search is currently unavailable.",
                "claims": [],
                "total_claims": 0
            }

        query = request.get('query', '')
        limit = request.get('limit', 100)
        if not query:
            return {
                "error": "No query provided",
                "query": "",
                "synthesis": "Please provide a search query.",
                "claims": [],
                "total_claims": 0
            }

        try:
            result = pattern_searcher_llm.search(query, limit=limit)
            return result
        except Exception as e:
            logger.error(f"Pattern search error: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "query": query,
                "synthesis": f"Error during search: {str(e)}",
                "claims": [],
                "total_claims": 0
            }
    
    # Graph visualization endpoint
    @app.get("/api/graph/entity/{entity_name}")
    async def get_entity_graph(entity_name: str):
        """Return graph structure for visualization"""
        try:
            from neo4j import GraphDatabase
            
            neo4j_uri = os.environ.get("NEO4J_URI", os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
            neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
            neo4j_password = os.environ.get("NEO4J_PASSWORD", "aegistrusted")
            
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            
            query = """
            MATCH (center:Entity {name: $entity})
            OPTIONAL MATCH (center)<-[:MENTIONS]-(chunk)-[:CONTAINS_CLAIM]->(claim:Claim)
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(related:Entity)
            WHERE related <> center
            
            RETURN 
                center,
                collect(DISTINCT claim)[0..20] as claims,
                collect(DISTINCT related)[0..10] as related_entities
            LIMIT 1
            """
            
            with driver.session() as session:
                result = session.run(query, entity=entity_name).single()
                
                if not result:
                    return {"nodes": [], "edges": []}
                
                nodes = []
                edges = []
                
                # Center entity
                center = result['center']
                center_id = f"entity-{center.id}"
                nodes.append({
                    'id': center_id,
                    'label': center['name'],
                    'type': 'Entity',
                    'size': 20,
                    'trust_score': 0.8
                })
                
                # Claims
                for claim in result['claims']:
                    if claim:
                        claim_id = f"claim-{claim.id}"
                        nodes.append({
                            'id': claim_id,
                            'label': claim.get('claim_text', '')[:50] + '...',
                            'type': 'Claim',
                            'size': 12,
                            'trust_score': claim.get('confidence', 0.5)
                        })
                        edges.append({
                            'source': center_id,
                            'target': claim_id,
                            'type': 'MENTIONS',
                            'weight': 1.0
                        })
                
                # Related entities
                for related in result['related_entities']:
                    if related:
                        related_id = f"entity-{related.id}"
                        nodes.append({
                            'id': related_id,
                            'label': related['name'],
                            'type': 'Entity',
                            'size': 10,
                            'trust_score': 0.7
                        })
                        edges.append({
                            'source': center_id,
                            'target': related_id,
                            'type': 'RELATED',
                            'weight': 0.5
                        })
            
            driver.close()
            return {'nodes': nodes, 'edges': edges}
            
        except Exception as e:
            logger.error(f"Graph error: {str(e)}")
            return {"nodes": [], "edges": [], "error": str(e)}

    # === System Status Endpoint ===
    @app.get("/api/system/status")
    async def get_system_status():
        """Get comprehensive system status"""
        try:
            from neo4j import GraphDatabase

            # Connect to Neo4j
            driver = GraphDatabase.driver(
                os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                auth=("neo4j", "aegistrusted")
            )

            with driver.session() as session:
                # Get claims count
                result = session.run("MATCH (c:Claim) RETURN count(c) as count")
                claims_count = result.single()['count']

                # Get citations count
                result = session.run("""
                    MATCH ()-[r:CITES]->() 
                    RETURN count(r) as count
                """)
                citations_count = result.single()['count']

                # Get embeddings count from PostgreSQL (where they're actually stored)
                import psycopg2
                try:
                    pg_conn = psycopg2.connect(
                        host=os.environ.get("POSTGRES_HOST", "localhost"),
                        database="aegis_insight",
                        user="aegis",
                        password="aegis_trusted_2025"
                    )
                    pg_cursor = pg_conn.cursor()
                    pg_cursor.execute("SELECT COUNT(*) FROM claim_embeddings")
                    embeddings_count = pg_cursor.fetchone()[0]
                    pg_cursor.close()
                    pg_conn.close()
                except Exception as e:
                    logger.warning(f"Could not query PostgreSQL embeddings: {e}")
                    embeddings_count = 0

                # Get geographic data count
                result = session.run("""
                    MATCH (c:Claim)
                    WHERE c.geographic_data IS NOT NULL 
                      AND c.geographic_data <> '{}'
                      AND c.geographic_data <> ''
                    RETURN count(c) as count
                """)
                geo_claims_count = result.single()['count']

            driver.close()

            # Build detector status
            detectors_ready = {
                'suppression': {
                    'ready': citations_count > 0,
                    'missing': [] if citations_count > 0 else ['build_citations']
                },
                'coordination': {
                    'ready': embeddings_count > 0,
                    'missing': [] if embeddings_count > 0 else ['generate_embeddings', 'build_temporal_index']
                },
                'anomaly': {
                    'ready': embeddings_count > 0 and geo_claims_count > 0,
                    'missing': ['generate_embeddings', 'build_geographic_index'] if embeddings_count == 0 else [
                        'build_geographic_index'] if geo_claims_count == 0 else []
                }
            }

            # Check Neo4j connection
            neo4j_connected = True  # If we got here, it's connected

            # Check Ollama
            ollama_available = False
            try:
                import requests
                response = requests.get(f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/tags", timeout=2)
                ollama_available = response.status_code == 200
            except:
                pass

            status = {
                'claims_count': claims_count,
                'citations_count': citations_count,
                'embeddings_count': embeddings_count,
                'geographic_claims_count': geo_claims_count,
                'detectors_ready': detectors_ready,
                'neo4j_connected': neo4j_connected,
                'ollama_available': ollama_available,
                'db_size': f"{(claims_count / 1000):.1f} MB",  # Rough estimate
                'last_ingestion': None,
                'jobs': {}
            }

            return JSONResponse(content=status)

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            # Return empty structure so UI doesn't break
            return JSONResponse(content={
                'claims_count': 0,
                'citations_count': 0,
                'embeddings_count': 0,
                'geographic_claims_count': 0,
                'detectors_ready': {
                    'suppression': {'ready': False, 'missing': ['build_citations']},
                    'coordination': {'ready': False, 'missing': ['generate_embeddings']},
                    'anomaly': {'ready': False, 'missing': ['generate_embeddings', 'build_geographic_index']}
                },
                'neo4j_connected': False,
                'ollama_available': False,
                'db_size': 'Unknown',
                'last_ingestion': None,
                'jobs': {}
            })

    # === Detection Setup Endpoint ===
    @app.post("/api/admin/detection-setup")
    async def run_detection_setup(
            request: DetectionSetupRequest,
            background_tasks: BackgroundTasks
    ):
        """
        Start detection setup job(s) in background

        Args:
            detector: Specific detector to setup, or None for all
            all: If True, setup all detectors

        Returns:
            - success: Boolean
            - job_ids: List of started job IDs
            - message: Status message
        """
        try:
            job_ids = []

            if request.all or request.detector is None:
                # Setup all detectors
                logger.info("Starting setup for all detectors")

                # Run in background
                background_tasks.add_task(
                    detection_builder.run_detection_setup,
                    for_detector=None,
                    skip_completed=False  # FORCE RE-RUN
                )

                # Get job IDs that will run
                job_ids = ['build_citations', 'generate_embeddings',
                           'build_temporal_index', 'build_geographic_index']

                return {
                    "success": True,
                    "job_ids": job_ids,
                    "message": "All detector setups started. This will take ~30-60 minutes."
                }
            else:
                # Setup specific detector
                detector = request.detector.lower()

                if detector not in ['suppression', 'coordination', 'anomaly']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid detector: {detector}. Must be 'suppression', 'coordination', or 'anomaly'"
                    )

                logger.info(f"Starting setup for {detector} detector")

                # Run in background
                background_tasks.add_task(
                    detection_builder.run_detection_setup,
                    for_detector=detector,
                    skip_completed=False  # FORCE RE-RUN
                )

                # Get required jobs for this detector
                requirements = detection_builder.get_detector_requirements(detector)
                job_ids = requirements.get('missing', [])

                return {
                    "success": True,
                    "job_ids": job_ids,
                    "message": f"{detector.capitalize()} detector setup started"
                }

        except Exception as e:
            logger.error(f"Failed to start detection setup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # === Clear Checkpoints Endpoint ===
    @app.post("/api/admin/clear-checkpoints")
    async def clear_checkpoints():
        """
        Clear all job checkpoints

        Forces jobs to restart from beginning on next run
        """
        try:
            detection_builder.clear_all_checkpoints()

            return {
                "success": True,
                "message": "All checkpoints cleared"
            }
        except Exception as e:
            logger.error(f"Failed to clear checkpoints: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # === View Logs Endpoint ===
    @app.get("/api/admin/logs")
    async def view_logs(
            lines: int = 100,
            level: str = "INFO"
    ):
        """
        Get recent log entries

        Args:
            lines: Number of recent lines to return (default: 100)
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        """
        try:
            # Try to get log file path from detection builder
            try:
                log_file = detection_builder.get_log_file_path()
            except AttributeError:
                # Fallback to default location
                log_file = "/tmp/aegis_detection.log"

            if not os.path.exists(log_file):
                return {
                    "logs": [],
                    "message": "No log file found"
                }

            # Read last N lines
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]

            # Filter by level
            filtered = [
                line for line in recent_lines
                if level.upper() in line
            ]

            return {
                "logs": filtered,
                "total_lines": len(filtered)
            }

        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ============================================================================
    # DETECTION ENDPOINTS
    # ============================================================================

    @app.post("/api/detect/suppression")
    async def detect_suppression(request: Request):
        """
        Detect suppression patterns in knowledge claims with calibration profile support
        """
        try:
            body = await request.json()
            query = body.get("query") or body.get("topic")  # Accept both
            profile = body.get("profile",
                               "state_suppression.json")  # Default to state_suppression for historical figures
            claim_ids = body.get("claim_ids")

            if not query:
                return {"success": False, "error": "Query is required"}

            logger.info(f"Suppression detection for: {query} with profile: {profile}")
            logger.info(f"claim_ids received: {type(claim_ids)} - {len(claim_ids) if claim_ids else None}")

            # Load profile path
            profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration_profiles")
            profile_path = os.path.join(profile_dir, profile)

            if not os.path.exists(profile_path):
                print(f"Profile not found: {profile_path}, using built-in defaults")
                profile_path = None

            # Run suppression detector with profile
            from aegis_suppression_detector_v2 import SuppressionDetector
            detector = SuppressionDetector(profile_path=profile_path)
            print(f"DEBUG: Calling detect_suppression with query={query}, NO claim_ids")
            result = detector.detect_suppression(query, claim_ids=None)  # FORCE no claim_ids  # Use Topic traversal + prioritized sampling
            detector.close()

            # Ensure result has success flag
            if isinstance(result, dict):
                if 'success' not in result:
                    result['success'] = True
                if 'topic' not in result and 'query' not in result:
                    result['topic'] = query
                result['profile_used'] = profile
            else:
                result = {
                    "success": True,
                    "topic": query,
                    "profile_used": profile,
                    "result": result
                }

            print(f"Suppression detection complete: score={result.get('suppression_score', 'N/A')}")
            return {"success": True, "result": result}

        except Exception as e:
            print(f"Suppression detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "topic": query if 'query' in locals() else "unknown"
            }

    @app.post("/api/detect/coordination")
    async def detect_coordination(request: Request):
        """
        Detect coordinated messaging campaigns

        Returns data with IDs matching graph node format (claim-{elementId})
        """
        try:
            body = await request.json()
            query = body.get("query")
            claim_ids_raw = body.get("claim_ids", [])
            limit = body.get("limit", 500)
            options = body.get("options", {})

            if not query:
                return {"success": False, "error": "Query is required"}

            print(f"=" * 60)
            print(f"Coordination detection for: {query}")
            print(f"  claim_ids from pattern search: {len(claim_ids_raw) if claim_ids_raw else 0}")

            # Run coordination detector
            from aegis_coordination_detector import CoordinationDetector
            detector = CoordinationDetector()
            result = detector.detect_coordination(query, claim_ids=None, limit=limit)

            # === Build coordinated_claims with GRAPH-COMPATIBLE IDs ===
            coordinated_claims = []
            clusters = []
            coordinated_pairs = []

            try:
                # Use the same topic utils that the detector uses
                from aegis_topic_utils import get_claims_via_topics_hybrid

                with detector.driver.session() as session:
                    # Get claims using the same hybrid search as the detector
                    print(f"  Using get_claims_via_topics_hybrid...")
                    topic_claims = get_claims_via_topics_hybrid(
                        session,
                        query,
                        limit=min(limit, 100),
                        require_text_match=True  # CORRECT parameter name
                    )
                    print(f"  Topic search found {len(topic_claims)} claims")

                    # Now get elementIds for these claims to match graph format
                    if topic_claims:
                        # Extract claim_ids from topic search results
                        found_claim_ids = [c.get('claim_id') for c in topic_claims if c.get('claim_id')]
                        print(f"  Found {len(found_claim_ids)} claim_ids to convert")

                        if found_claim_ids:
                            # Query to get elementIds for these claims
                            element_query = """
                            MATCH (c:Claim)
                            WHERE c.claim_id IN $claim_ids
                            RETURN elementId(c) as element_id,
                                   c.claim_id as claim_id,
                                   c.claim_text as claim_text,
                                   c.source_file as source_file,
                                   c.temporal_data as temporal_data
                            LIMIT $limit
                            """

                            elem_result = session.run(
                                element_query,
                                claim_ids=found_claim_ids[:100],
                                limit=100
                            )
                            records = list(elem_result)
                            print(f"  Got {len(records)} claims with elementIds")

                            for record in records:
                                element_id = record['element_id']
                                graph_id = f"claim-{element_id}"

                                claim_data = {
                                    'id': graph_id,
                                    'claim_id': graph_id,
                                    'neo4j_claim_id': record['claim_id'],
                                    'claim_text': record['claim_text'][:150] if record['claim_text'] else '',
                                    'source_file': record['source_file']
                                }

                                # Extract publication date
                                temporal_data = record.get('temporal_data')
                                if temporal_data:
                                    try:
                                        import json
                                        temporal = json.loads(temporal_data) if isinstance(temporal_data,
                                                                                           str) else temporal_data
                                        if temporal and isinstance(temporal, dict):
                                            dates = temporal.get('dates') or temporal.get('absolute_dates') or []
                                            if dates and len(dates) > 0:
                                                first_date = dates[0]
                                                date_val = first_date.get('date', '') if isinstance(first_date,
                                                                                                    dict) else str(
                                                    first_date)
                                                if date_val:
                                                    claim_data['publication_date'] = str(date_val)[:10]
                                    except:
                                        pass

                                coordinated_claims.append(claim_data)

                    print(f"  Built {len(coordinated_claims)} claims for graph highlighting")
                    if coordinated_claims:
                        print(f"  Sample graph ID: {coordinated_claims[0].get('id')}")

                    # Build clusters from temporal clustering signal
                    temporal_signal = result.get('signals', {}).get('temporal_clustering', {})
                    if temporal_signal.get('burst_detected') and coordinated_claims:
                        burst_size = min(
                            temporal_signal.get('burst_size', len(coordinated_claims)),
                            len(coordinated_claims)
                        )
                        burst_claims = [c['id'] for c in coordinated_claims[:burst_size]]
                        clusters = [{
                            'claim_ids': burst_claims,
                            'temporal_range': temporal_signal.get('interpretation', 'Temporal burst detected'),
                            'size': len(burst_claims)
                        }]
                        print(f"  Created cluster with {len(burst_claims)} claims")

                    # Build coordinated_pairs for visual connections
                    coord_score = result.get('coordination_score', 0)
                    if len(coordinated_claims) >= 2 and coord_score > 0.3:
                        for i in range(min(len(coordinated_claims) - 1, 20)):
                            coordinated_pairs.append({
                                'source_id': coordinated_claims[i]['id'],
                                'target_id': coordinated_claims[i + 1]['id']
                            })
                        print(f"  Created {len(coordinated_pairs)} coordinated pairs")

            except Exception as e:
                print(f"ERROR building highlighting data: {e}")
                import traceback
                traceback.print_exc()

            detector.close()

            # Add highlighting data to result
            result['coordinated_claims'] = coordinated_claims
            result['clusters'] = clusters
            result['coordinated_pairs'] = coordinated_pairs

            # Ensure result has success flag
            if isinstance(result, dict):
                if 'success' not in result:
                    result['success'] = True
                if 'topic' not in result:
                    result['topic'] = query

            print(f"=" * 60)
            print(f"RESULT: score={result.get('coordination_score', 'N/A')}, "
                  f"claims={len(coordinated_claims)}, clusters={len(clusters)}")
            print(f"=" * 60)

            return result

        except Exception as e:
            print(f"Coordination detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "topic": query if 'query' in locals() else "unknown"
            }

    @app.post("/api/detect/anomaly")
    async def detect_anomaly(request: Request):
        """
        Detect cross-cultural anomalies and patterns
        """
        try:
            body = await request.json()
            query = body.get("query")  # Uses 'query' for consistency
            claim_ids = body.get("claim_ids")
            limit = body.get("limit", 500)  # User-specified limit
            options = body.get("options", {})

            if not query:
                return {"success": False, "error": "Query is required"}

            print(f"Anomaly detection for: {query} (limit={limit})")

            # Run anomaly detector
            from aegis_anomaly_detector import AnomalyDetector
            detector = AnomalyDetector()
            result = detector.detect_anomaly(query, claim_ids=claim_ids, limit=limit)
            detector.close()

            # Ensure result has success flag
            if isinstance(result, dict):
                if 'success' not in result:
                    result['success'] = True
                # Add query to result if not present
                if 'pattern' not in result and 'query' not in result:
                    result['pattern'] = query
            else:
                # If result is not a dict, wrap it
                result = {
                    "success": True,
                    "pattern": query,
                    "result": result
                }

            print(f"Anomaly detection complete: score={result.get('anomaly_score', 'N/A')}")
            return result

        except Exception as e:
            print(f"Anomaly detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "pattern": query if 'query' in locals() else "unknown"
            }

    @app.get("/api/config/detection")
    async def get_detection_config():
        """Get current detection configuration"""
        try:
            config = get_config()
            return config.to_dict()
        except Exception as e:
            logger.error(f"Failed to get detection config: {e}")
            return {"error": str(e)}, 500

    @app.post("/api/config/detection")
    async def update_detection_config(updates: dict):
        """Update detection configuration"""
        try:
            success, error = update_config(**updates)
            if success:
                config = get_config()
                logger.info(f"Detection config updated: {updates}")
                return {"status": "success", "config": config.to_dict()}
            else:
                logger.warning(f"Failed to update detection config: {error}")
                return {"status": "error", "message": error}, 400
        except Exception as e:
            logger.error(f"Error updating detection config: {e}")
            return {"status": "error", "message": str(e)}, 500

    @app.post("/api/config/detection/reset")
    async def reset_detection_config():
        """Reset detection configuration to defaults"""
        try:
            config = reset_config()
            logger.info("Detection config reset to defaults")
            return {"status": "success", "config": config.to_dict()}
        except Exception as e:
            logger.error(f"Failed to reset detection config: {e}")
            return {"status": "error", "message": str(e)}, 500

    @app.get("/api/config/detection/validate")
    async def validate_detection_config():
        """Validate current detection configuration"""
        try:
            config = get_config()
            is_valid, errors = config.validate()
            return {"valid": is_valid, "errors": errors}
        except Exception as e:
            logger.error(f"Failed to validate config: {e}")
            return {"valid": False, "errors": [str(e)]}, 500

    @app.post("/api/detect/quality-check")
    async def check_data_quality(request: Request):
        """
        Check if dataset is sufficient for detection analysis.

        Should be called BEFORE running detection to warn users
        about data limitations.

        Body:
            {
                "topic": "historical analysis",
                "detection_type": "suppression"  // or "coordination", "anomaly", "all"
            }

        Returns:
            {
                "topic": "historical analysis",
                "detection_type": "suppression",
                "sufficient": true/false,
                "warnings": [...],
                "metrics": {...},
                "recommendations": [...]
            }
        """
        try:
            body = await request.json()
            topic = body.get("topic")
            detection_type = body.get("detection_type", "all")

            if not topic:
                return {"error": "Topic is required"}, 400

            checker = DataQualityChecker()

            try:
                if detection_type == "suppression":
                    result = checker.assess_for_suppression(topic)
                elif detection_type == "coordination":
                    result = checker.assess_for_coordination(topic)
                elif detection_type == "anomaly":
                    result = checker.assess_for_anomaly(topic)
                elif detection_type == "all":
                    result = checker.assess_all(topic)
                else:
                    return {"error": f"Invalid detection_type: {detection_type}"}, 400

                return {
                    "topic": topic,
                    "detection_type": detection_type,
                    **result
                }
            finally:
                checker.close()

        except Exception as e:
            logger.error(f"Quality check error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}, 500

    # === Shutdown Handler ===
    @app.on_event("shutdown")
    async def shutdown():
        """Clean up detection builder on shutdown"""
        detection_builder.close()

    # Get API configuration from environment
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8001))
    
    # Log startup info
    logger.info("=" * 60)
    logger.info(f"🚀 Starting AegisTrustNet API Server")
    logger.info(f"   Web Interface: http://localhost:{port}/")
    logger.info(f"   API Docs: http://localhost:{port}/docs")
    logger.info("=" * 60)
    
    # Mount static files LAST - after all API routes
    # CRITICAL: This must be last or it intercepts API calls!
    if os.path.exists(WEB_DIR):
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
        logger.info(f"✓ Static files mounted from: {WEB_DIR}")
    else:
        logger.error(f"✗ Web directory not found: {WEB_DIR}")
    
    # Run the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
