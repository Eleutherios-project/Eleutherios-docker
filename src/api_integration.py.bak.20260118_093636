#!/usr/bin/env python3
"""
AegisTrustNet - API Integration Module
Connects the multi-perspective service with the Pattern RAG API
"""

import os
import sys
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Import the multi-perspective service
try:
    from src.multi_perspective_service import process_user_query
    MULTI_PERSPECTIVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import multi_perspective_service: {e}")
    MULTI_PERSPECTIVE_AVAILABLE = False

# Import claim extraction
try:
    from src.enhanced_claim_extraction import run_claim_extraction
    CLAIM_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import enhanced_claim_extraction: {e}")
    CLAIM_EXTRACTION_AVAILABLE = False

# Import pattern discovery
try:
    from src.pattern_discovery import run_pattern_discovery
    PATTERN_DISCOVERY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pattern_discovery: {e}")
    PATTERN_DISCOVERY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aegis-api-integration")

# Load environment variables
load_dotenv()

# Create API router
enhanced_router = APIRouter(prefix="/enhanced", tags=["enhanced-responses"])

# Optional dependencies for Pattern RAG integration
try:
    # Import from Pattern RAG if available
    pattern_rag_dir = os.environ.get("PATTERN_RAG_DIR", "")
    if pattern_rag_dir not in sys.path:
        sys.path.append(pattern_rag_dir)
        
    # Try importing the pattern_rag_service
    try:
        from pattern_rag_service_test_ver import process_query as pattern_rag_process_query
        PATTERN_RAG_AVAILABLE = True
        logger.info("Successfully imported Pattern RAG components")
    except ImportError:
        # Try alternative function name if available
        try:
            from pattern_rag_service_test_ver import query as pattern_rag_process_query
            PATTERN_RAG_AVAILABLE = True
            logger.info("Successfully imported Pattern RAG using 'query' function")
        except ImportError:
            # Create a simple mock function
            def pattern_rag_process_query(query_text):
                logger.info("Using Ollama for LLM responses")
                import requests
                try:
                    response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': 'mistral-nemo:12b',
                            'prompt': f"Answer this question with factual information: {query_text}",
                            'stream': False
                        },
                        timeout=60  # Increased to 60 seconds
                    )
                    if response.status_code == 200:
                        return {
                            "query": query_text,
                            "answer": response.json().get('response', '')
                        }
                    else:
                        return {
                            "query": query_text,
                            "answer": f"Ollama returned status {response.status_code}"
                        }
                except Exception as e:
                    logger.error(f"Ollama error: {str(e)}")
                    return {
                        "query": query_text,
                        "answer": f"Error connecting to Ollama: {str(e)}"
                    }
            PATTERN_RAG_AVAILABLE = True
            logger.info("Using mock Pattern RAG function")
except Exception as e:
    logger.warning(f"Could not import Pattern RAG components: {str(e)}")
    PATTERN_RAG_AVAILABLE = False

# Pydantic models for API
class EnhancedQueryRequest(BaseModel):
    query: str = Field(..., description="The user query")
    user_id: Optional[str] = Field(None, description="Optional user ID for personalized responses")
    include_details: bool = Field(False, description="Whether to include detailed analysis in response")
    trust_threshold: Optional[float] = Field(0.4, description="Minimum trust threshold (0.0-1.0)", ge=0.0, le=1.0)

class EnhancedQueryResponse(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence in the answer (0.0-1.0)")
    sources: List[Dict[str, Any]] = Field([], description="Sources used in the response")
    patterns: Optional[List[Dict[str, Any]]] = Field(None, description="Patterns discovered in the data")
    perspectives: Optional[List[Dict[str, Any]]] = Field(None, description="Different perspectives on the question")
    processing_time: float = Field(..., description="Total processing time in seconds")

class TrustPreference(BaseModel):
    source: str = Field(..., description="Name of the source")
    trust_score: float = Field(..., description="Trust score (-1.0 to 1.0)", ge=-1.0, le=1.0)

class UserTrustPreferences(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: List[TrustPreference] = Field(..., description="List of trust preferences")

@enhanced_router.post("/query", response_model=None)
async def enhanced_query(request: EnhancedQueryRequest):
    """
    Process a query with enhanced trust network analysis
    """
    start_time = time.time()
    logger.info(f"Enhanced query received: {request.query}")
    
    try:
        # Step 1: Get original response from Pattern RAG if available
        original_response = None
        pattern_rag_data = None
        
        if PATTERN_RAG_AVAILABLE:
            try:
                logger.info("Getting response from Pattern RAG")
                pattern_rag_data = pattern_rag_process_query(request.query)
                original_response = pattern_rag_data.get("answer", "")
                logger.info("Received Pattern RAG response")
            except Exception as e:
                logger.error(f"Error getting Pattern RAG response: {str(e)}")
        
        # Step 2: Run claim extraction if needed
        # This would typically be done on a regular schedule, not per-request
        # But we'll include it here for completeness
        if request.include_details and CLAIM_EXTRACTION_AVAILABLE:
            try:
                logger.info("Running claim extraction")
                run_claim_extraction()
            except Exception as e:
                logger.error(f"Error in claim extraction: {str(e)}")
        
        # Step 3: Get multi-perspective response
        multi_perspective_data = {}
        try:
            if MULTI_PERSPECTIVE_AVAILABLE:
                logger.info("Getting multi-perspective response")
                multi_perspective_data = process_user_query(
                    request.user_id, 
                    request.query, 
                    original_response
                )
            else:
                logger.warning("Multi-perspective service not available")
                # Create a basic response
                multi_perspective_data = {
                    "query": request.query,
                    "original_answer": original_response,
                    "claims": [],
                    "patterns": [],
                    "perspectives": [],
                    "confidence": 0.5
                }
        except Exception as e:
            logger.error(f"Error in multi-perspective service: {str(e)}")
            # Create a basic response if there's an error
            multi_perspective_data = {
                "query": request.query,
                "original_answer": original_response,
                "claims": [],
                "patterns": [],
                "perspectives": [],
                "confidence": 0.5,
                "error": str(e)
            }
        
        # Step 4: Format the response
        # Use the LLM prompt as the answer for now
        prompt = multi_perspective_data.get("prompt", {}).get("content", "")
        
        # In a real implementation, you would send this prompt to an LLM
        # For now, we'll just use a placeholder or the original response
        enhanced_answer = original_response or "This would be the enhanced response after sending the prompt to an LLM."
        
        # If no original response, use a mock answer
        if not enhanced_answer:
            enhanced_answer = f"Based on the trust network analysis, the answer to '{request.query}' involves multiple perspectives and considerations. [This is a simulated response as the full system integration is still in progress]"
        
        # Create the response object to match exactly what the UI expects
        response = {
            "query": request.query,
            "answer": enhanced_answer,
            "confidence": multi_perspective_data.get("confidence", 0.5),
            "sources": []
        }
        
        # Add sources if available
        claims = multi_perspective_data.get("claims", [])
        if claims:
            response["sources"] = [
                {
                    "id": claim.get("id", ""),
                    "text": claim.get("text", ""),
                    "trust_score": claim.get("trust_score", 0.5)
                }
                for claim in claims[:5]  # Limit to top 5
            ]
        
        # Add patterns if available and requested
        if request.include_details and "patterns" in multi_perspective_data:
            response["patterns"] = []
            patterns = multi_perspective_data.get("patterns", [])
            for pattern in patterns:
                response["patterns"].append({
                    "type": pattern.get("type", "correlation"),
                    "data": pattern.get("data", {})
                })
        
        # Add sources if available
        claims = multi_perspective_data.get("claims", [])
        if claims:
            response["sources"] = [
                {"id": claim.get("id"), "text": claim.get("text"), "trust_score": claim.get("trust_score")}
                for claim in claims[:5]  # Limit to top 5
            ]
        
        # Add processing time
        response["processing_time"] = time.time() - start_time
        
        # Add detailed information if requested
        if request.include_details:
            response["patterns"] = multi_perspective_data.get("patterns", [])
            response["perspectives"] = multi_perspective_data.get("perspectives", [])
            response["prompt"] = prompt
        
        logger.info(f"Enhanced query processed in {response['processing_time']:.2f} seconds")
        return response
        
    except Exception as e:
        logger.error(f"Error processing enhanced query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a useful error response instead of throwing an HTTP exception
        return {
            "query": request.query,
            "answer": f"We encountered an error processing your query. {str(e)}",
            "error": str(e),
            "confidence": 0.0,
            "sources": [],
            "processing_time": time.time() - start_time
        }

@enhanced_router.post("/preferences", status_code=202)
async def set_preferences(preferences: UserTrustPreferences):
    """
    Set user trust preferences
    """
    logger.info(f"Received trust preferences for user: {preferences.user_id}")
    
    # Store the preferences (mock implementation)
    prefs = [{"source": p.source, "trust_score": p.trust_score} for p in preferences.preferences]
    
    return {
        "status": "success",
        "message": f"Stored {len(prefs)} preferences for user {preferences.user_id}"
    }

@enhanced_router.post("/run_pattern_discovery")
async def trigger_pattern_discovery():
    """
    Manually trigger the pattern discovery process
    """
    if not PATTERN_DISCOVERY_AVAILABLE:
        return {"status": "error", "message": "Pattern discovery component not available"}
        
    try:
        logger.info("Running pattern discovery")
        success = run_pattern_discovery()
        
        if success:
            return {"status": "success", "message": "Pattern discovery completed successfully"}
        else:
            return {"status": "error", "message": "Pattern discovery encountered errors"}
    except Exception as e:
        logger.error(f"Error in pattern discovery: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@enhanced_router.get("/")
async def enhanced_root():
    """Root endpoint for the enhanced API"""
    return {
        "status": "online",
        "component": "AegisTrustNet Enhanced API",
        "capabilities": {
            "multi_perspective": MULTI_PERSPECTIVE_AVAILABLE,
            "claim_extraction": CLAIM_EXTRACTION_AVAILABLE,
            "pattern_discovery": PATTERN_DISCOVERY_AVAILABLE,
            "pattern_rag": PATTERN_RAG_AVAILABLE
        }
    }

@enhanced_router.get("/debug")
async def debug_connection():
    """Debug endpoint to verify UI-to-API connectivity"""
    return {
        "status": "connected",
        "timestamp": time.time(),
        "server_info": {
            "cors_enabled": True,
            "multi_perspective_available": MULTI_PERSPECTIVE_AVAILABLE,
            "pattern_rag_available": PATTERN_RAG_AVAILABLE,
            "claim_extraction_available": CLAIM_EXTRACTION_AVAILABLE,
            "pattern_discovery_available": PATTERN_DISCOVERY_AVAILABLE
        }
    }

@enhanced_router.post("/echo")
async def echo_request(request: Request):
    """Echo the request body to debug UI-to-API communication"""
    body = await request.json()
    return {
        "status": "echo",
        "received": body,
        "timestamp": time.time()
    }

# Scheduler task for regular pattern discovery
async def scheduled_pattern_discovery():
    """
    Run pattern discovery on a regular schedule
    """
    while True:
        try:
            logger.info("Running scheduled pattern discovery")
            if PATTERN_DISCOVERY_AVAILABLE:
                run_pattern_discovery()
                logger.info("Scheduled pattern discovery completed")
            else:
                logger.warning("Pattern discovery component not available")
        except Exception as e:
            logger.error(f"Error in scheduled pattern discovery: {str(e)}")
        
        # Wait for 24 hours
        await asyncio.sleep(24 * 60 * 60)

def start_scheduler():
    """Start the background scheduler"""
    loop = asyncio.get_event_loop()
    task = loop.create_task(scheduled_pattern_discovery())
    return task
