"""
API Extensions for AegisTrustNet - Ollama Integration & Enhanced Claim Extraction

Add these routes to your existing api_server.py or api_integration.py

Version: 2.0
Date: November 3, 2025
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import sys

# Import enhanced claim extractor
try:
    from enhanced_claim_extractor import (
        ClaimExtractor, ExtractionConfig, ExtractionMode, 
        OllamaClient, DEFAULT_SYSTEM_PROMPT
    )
except ImportError:
    # Try from script directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from enhanced_claim_extractor import (
        ClaimExtractor, ExtractionConfig, ExtractionMode,
        OllamaClient, DEFAULT_SYSTEM_PROMPT
    )

logger = logging.getLogger(__name__)

# Create router
ollama_router = APIRouter(prefix="/api/ollama", tags=["ollama"])


# Request/Response Models
class ModelInfo(BaseModel):
    """Ollama model information"""
    name: str
    size: Optional[str] = None
    parameters: Optional[str] = None
    recommended: bool = False
    suitable_for: List[str] = []


class OllamaStatusResponse(BaseModel):
    """Ollama service status"""
    available: bool
    models_count: int
    models: List[ModelInfo]
    recommended_model: Optional[str] = None


class ClaimExtractionRequest(BaseModel):
    """Request for claim extraction"""
    text: str
    mode: str = "llm_large"  # none, spacy, llm_small, llm_large
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = {}
    confidence_threshold: float = 0.6


class ClaimExtractionResponse(BaseModel):
    """Response from claim extraction"""
    success: bool
    claims: List[Dict[str, Any]]
    stats: Dict[str, Any]
    error: Optional[str] = None


class SystemPromptResponse(BaseModel):
    """System prompt information"""
    default_prompt: str
    domain_specific_prompts: Dict[str, str]


@ollama_router.get("/status", response_model=OllamaStatusResponse)
async def get_ollama_status():
    """
    Check Ollama status and list available models
    """
    ollama = OllamaClient()
    
    if not ollama.is_available():
        return OllamaStatusResponse(
            available=False,
            models_count=0,
            models=[]
        )
    
    # Get models
    raw_models = ollama.list_models()
    
    # Enrich with metadata
    models = []
    for m in raw_models:
        name = m.get('name', '')
        
        # Determine suitability
        suitable_for = []
        recommended = False
        
        if any(x in name.lower() for x in ['70b', '72b']):
            suitable_for = ['llm_large']
            recommended = True
            parameters = "70B+"
        elif any(x in name.lower() for x in ['13b', '14b']):
            suitable_for = ['llm_small', 'llm_large']
            parameters = "13B"
        elif any(x in name.lower() for x in ['7b', '8b']):
            suitable_for = ['llm_small']
            parameters = "7-8B"
        else:
            suitable_for = ['llm_small']
            parameters = "Unknown"
        
        models.append(ModelInfo(
            name=name,
            size=m.get('size', 'Unknown'),
            parameters=parameters,
            recommended=recommended,
            suitable_for=suitable_for
        ))
    
    # Get recommended model
    recommended = ollama.recommend_model(ExtractionMode.LLM_LARGE)
    
    return OllamaStatusResponse(
        available=True,
        models_count=len(models),
        models=models,
        recommended_model=recommended
    )


@ollama_router.post("/extract", response_model=ClaimExtractionResponse)
async def extract_claims(request: ClaimExtractionRequest):
    """
    Extract claims from text using specified mode
    
    Supports:
    - mode='none': Fast regex patterns (legacy)
    - mode='spacy': SpaCy NLP extraction
    - mode='llm_small': 7-13B parameter models
    - mode='llm_large': 70B+ parameter models (highest quality)
    """
    
    try:
        # Validate mode
        if request.mode not in ['none', 'spacy', 'llm_small', 'llm_large']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}"
            )
        
        # Create extraction config
        config = ExtractionConfig(
            mode=ExtractionMode(request.mode),
            model_name=request.model,
            system_prompt=request.system_prompt,
            confidence_threshold=request.confidence_threshold
        )
        
        # Initialize extractor
        extractor = ClaimExtractor(config)
        
        # Extract claims
        claims = extractor.extract_claims(request.text, request.metadata)
        
        # Compute stats
        stats = {
            'total_claims': len(claims),
            'mode': request.mode,
            'model': config.model_name,
            'claim_types': {
                'PRIMARY': len([c for c in claims if c.get('claim_type') == 'PRIMARY']),
                'SECONDARY': len([c for c in claims if c.get('claim_type') == 'SECONDARY']),
                'META': len([c for c in claims if c.get('claim_type') == 'META']),
                'CONTEXTUAL': len([c for c in claims if c.get('claim_type') == 'CONTEXTUAL'])
            },
            'avg_confidence': sum(c.get('confidence', 0) for c in claims) / max(len(claims), 1)
        }
        
        return ClaimExtractionResponse(
            success=True,
            claims=claims,
            stats=stats
        )
    
    except Exception as e:
        logger.error(f"Claim extraction error: {e}")
        return ClaimExtractionResponse(
            success=False,
            claims=[],
            stats={},
            error=str(e)
        )


@ollama_router.get("/system-prompts", response_model=SystemPromptResponse)
async def get_system_prompts():
    """
    Get default and domain-specific system prompts
    """
    
    domain_prompts = {
        'strategic_analytical': """Focus on:
- Strategic concepts and their origins
- Influence on doctrine and practice  
- Institutional vs practitioner adoption
- Criticism and reception patterns
- Network topology of idea propagation""",
        
        'research_investigative': """Focus on:
- Evidence claims and their sources
- Mainstream vs alternative interpretations
- Dismissal and suppression patterns
- Independent corroboration
- Attribution chains (who cited whom)""",
        
        'academic_scholarly': """Focus on:
- Research findings and methodologies
- Citation relationships
- Theoretical developments
- Peer review and validation
- Paradigm shifts and controversies""",
        
        'contemplative_spiritual': """Focus on:
- Philosophical claims and traditions
- Consciousness and awareness concepts
- Cross-tradition connections
- Modern scientific parallels
- Experiential vs theoretical knowledge""",
        
        'practical_wisdom': """Focus on:
- Actionable principles and heuristics
- Real-world applications
- Evidence-based insights
- Decision-making frameworks
- Practical vs theoretical knowledge""",
        
        'technical_implementation': """Focus on:
- Technical specifications and standards
- Implementation patterns
- Best practices and anti-patterns
- Standards and protocols
- Design decisions and trade-offs"""
    }
    
    return SystemPromptResponse(
        default_prompt=DEFAULT_SYSTEM_PROMPT,
        domain_specific_prompts=domain_prompts
    )


@ollama_router.post("/test-extraction")
async def test_extraction(request: ClaimExtractionRequest):
    """
    Test claim extraction on sample text
    Returns detailed breakdown for quality assessment
    """
    
    # Run extraction
    response = await extract_claims(request)
    
    if not response.success:
        return response
    
    # Add detailed breakdown for testing
    breakdown = {
        'success': response.success,
        'claims_by_type': {},
        'sample_claims': {},
        'confidence_distribution': {
            'high (>0.8)': 0,
            'medium (0.6-0.8)': 0,
            'low (<0.6)': 0
        },
        'stats': response.stats
    }
    
    # Group claims by type
    for claim in response.claims:
        claim_type = claim.get('claim_type', 'CONTEXTUAL')
        if claim_type not in breakdown['claims_by_type']:
            breakdown['claims_by_type'][claim_type] = []
        breakdown['claims_by_type'][claim_type].append(claim)
        
        # Confidence distribution
        conf = claim.get('confidence', 0)
        if conf > 0.8:
            breakdown['confidence_distribution']['high (>0.8)'] += 1
        elif conf >= 0.6:
            breakdown['confidence_distribution']['medium (0.6-0.8)'] += 1
        else:
            breakdown['confidence_distribution']['low (<0.6)'] += 1
    
    # Get sample claims from each type
    for claim_type, claims_list in breakdown['claims_by_type'].items():
        if claims_list:
            # Get highest confidence claim of this type
            best_claim = max(claims_list, key=lambda c: c.get('confidence', 0))
            breakdown['sample_claims'][claim_type] = {
                'text': best_claim.get('text', ''),
                'subject': best_claim.get('subject', ''),
                'relation': best_claim.get('relation', ''),
                'object': best_claim.get('object', ''),
                'confidence': best_claim.get('confidence', 0),
                'significance': best_claim.get('significance', 0)
            }
    
    return breakdown


# Health check endpoint
@ollama_router.get("/health")
async def health_check():
    """
    Health check for Ollama integration
    """
    ollama = OllamaClient()
    
    return {
        'status': 'healthy' if ollama.is_available() else 'degraded',
        'ollama_available': ollama.is_available(),
        'extraction_modes': {
            'none': 'available',
            'spacy': 'available',  # May require model download
            'llm_small': 'available' if ollama.is_available() else 'unavailable',
            'llm_large': 'available' if ollama.is_available() else 'unavailable'
        }
    }


# Function to register these routes with your FastAPI app
def register_ollama_routes(app):
    """
    Register Ollama routes with FastAPI app
    
    Usage:
        from api_ollama_extension import register_ollama_routes
        
        app = FastAPI()
        register_ollama_routes(app)
    """
    app.include_router(ollama_router)
    logger.info("Ollama API routes registered")


# Example integration snippet
"""
To integrate into your existing API:

1. Add to your api_server.py or api_integration.py:

from api_ollama_extension import register_ollama_routes

app = FastAPI()

# ... your existing routes ...

# Add Ollama routes
register_ollama_routes(app)

2. Or manually include the router:

from api_ollama_extension import ollama_router

app.include_router(ollama_router)

3. Test the endpoints:

curl http://localhost:8001/api/ollama/status
curl http://localhost:8001/api/ollama/health

4. Extract claims:

curl -X POST http://localhost:8001/api/ollama/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Boyd developed the OODA loop...",
    "mode": "llm_large",
    "metadata": {"domain": "strategic_analytical"}
  }'
"""
