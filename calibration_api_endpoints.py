"""
Aegis Insight - Calibration API Endpoints

Add these endpoints to your main FastAPI app.

Provides:
- Profile CRUD (list, get, create, update, delete)
- Detection with profile selection
- Profile validation

Author: Aegis Insight Team
Date: November 2025
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
import logging
from datetime import datetime

# Import the new detector
from aegis_suppression_detector_v2 import (
    SuppressionDetector, 
    get_profile_directory,
    list_profiles,
    load_profile,
    save_profile
)

logger = logging.getLogger(__name__)

# Create router - add this to your main app with: app.include_router(calibration_router)
calibration_router = APIRouter(prefix="/api", tags=["calibration"])

# Configuration - adjust these paths for your environment
DATA_DIR = os.environ.get('AEGIS_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
PROFILE_DIR = os.path.join(DATA_DIR, 'calibration_profiles')

# Ensure profile directory exists
os.makedirs(PROFILE_DIR, exist_ok=True)


# === Pydantic Models ===

class ProfileMetadata(BaseModel):
    name: str
    description: str = ""
    author: str = "user"
    tags: List[str] = []


class ScoringConfig(BaseModel):
    mode: str = "goldfinger"
    happenstance_max: int = 1
    coincidence_max: int = 2
    enemy_action_base: float = 0.50
    logarithmic_factor: float = 0.20
    max_score: float = 0.95


class PatternCategory(BaseModel):
    include: List[str] = []
    exclude: List[str] = []


class SemanticPatterns(BaseModel):
    match_mode: str = "semantic"
    match_threshold: float = 0.75
    suppression_experiences: PatternCategory = PatternCategory()
    institutional_actions: PatternCategory = PatternCategory()
    dismissal_language: PatternCategory = PatternCategory()
    suppression_of_record: PatternCategory = PatternCategory()


class SignalWeights(BaseModel):
    suppression_narrative: float = 0.40
    meta_claim_density: float = 0.15
    network_isolation: float = 0.10
    evidence_avoidance: float = 0.20
    authority_mismatch: float = 0.15


class Thresholds(BaseModel):
    min_claims_for_analysis: int = 5
    score_levels: Dict[str, float] = {
        "critical": 0.80,
        "high": 0.60,
        "moderate": 0.40,
        "low": 0.20
    }


class CalibrationProfile(BaseModel):
    schema_version: str = "2.0"
    metadata: ProfileMetadata
    scoring: ScoringConfig = ScoringConfig()
    semantic_patterns: SemanticPatterns = SemanticPatterns()
    signal_weights: SignalWeights = SignalWeights()
    thresholds: Thresholds = Thresholds()


class ProfileCreateRequest(BaseModel):
    profile: CalibrationProfile
    filename: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    profile: CalibrationProfile


class DetectionRequest(BaseModel):
    topic: str
    claim_ids: Optional[List[str]] = None
    profile: str = "default.json"
    domain: Optional[str] = None
    limit: int = 500


# === Profile Management Endpoints ===

@calibration_router.get("/profiles")
async def api_list_profiles():
    """
    List all available calibration profiles
    
    Returns:
        profiles: List of profile summaries with filename, name, description, tags
    """
    try:
        profiles = []
        
        if not os.path.exists(PROFILE_DIR):
            return {"profiles": [], "profile_dir": PROFILE_DIR}
        
        for filename in os.listdir(PROFILE_DIR):
            if filename.endswith('.json'):
                path = os.path.join(PROFILE_DIR, filename)
                try:
                    with open(path, 'r') as f:
                        profile = json.load(f)
                    
                    metadata = profile.get('metadata', {})
                    profiles.append({
                        'filename': filename,
                        'name': metadata.get('name', filename),
                        'description': metadata.get('description', ''),
                        'tags': metadata.get('tags', []),
                        'author': metadata.get('author', 'unknown'),
                        'schema_version': profile.get('schema_version', '1.0')
                    })
                except Exception as e:
                    logger.warning(f"Failed to read profile {filename}: {e}")
        
        # Sort by name
        profiles.sort(key=lambda x: x['name'])
        
        return {
            "profiles": profiles,
            "profile_dir": PROFILE_DIR,
            "count": len(profiles)
        }
        
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.get("/profiles/{filename}")
async def api_get_profile(filename: str):
    """
    Get a specific calibration profile
    
    Args:
        filename: Profile filename (e.g., 'default.json')
        
    Returns:
        Full profile JSON
    """
    try:
        path = os.path.join(PROFILE_DIR, filename)
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Profile not found: {filename}")
        
        with open(path, 'r') as f:
            profile = json.load(f)
        
        return {
            "filename": filename,
            "profile": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.post("/profiles")
async def api_create_profile(request: ProfileCreateRequest):
    """
    Create a new calibration profile
    
    Args:
        profile: CalibrationProfile object
        filename: Optional filename (auto-generated from name if not provided)
        
    Returns:
        Created profile with filename
    """
    try:
        profile_dict = request.profile.dict()
        
        # Generate filename if not provided
        if request.filename:
            filename = request.filename
            if not filename.endswith('.json'):
                filename += '.json'
        else:
            name = profile_dict['metadata']['name']
            filename = name.lower().replace(' ', '_').replace('-', '_') + '.json'
        
        path = os.path.join(PROFILE_DIR, filename)
        
        # Check if exists
        if os.path.exists(path):
            raise HTTPException(
                status_code=409, 
                detail=f"Profile already exists: {filename}. Use PUT to update."
            )
        
        # Add timestamps
        now = datetime.utcnow().isoformat() + 'Z'
        profile_dict['metadata']['created'] = now
        profile_dict['metadata']['modified'] = now
        
        # Validate and normalize weights
        weights = profile_dict.get('signal_weights', {})
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize
            for key in weights:
                weights[key] = round(weights[key] / weight_sum, 3)
            profile_dict['signal_weights'] = weights
        
        # Save
        with open(path, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        logger.info(f"Created profile: {filename}")
        
        return {
            "success": True,
            "filename": filename,
            "profile": profile_dict,
            "message": f"Profile '{profile_dict['metadata']['name']}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.put("/profiles/{filename}")
async def api_update_profile(filename: str, request: ProfileUpdateRequest):
    """
    Update an existing calibration profile
    
    Args:
        filename: Profile filename to update
        profile: Updated CalibrationProfile object
        
    Returns:
        Updated profile
    """
    try:
        path = os.path.join(PROFILE_DIR, filename)
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Profile not found: {filename}")
        
        profile_dict = request.profile.dict()
        
        # Preserve created timestamp, update modified
        try:
            with open(path, 'r') as f:
                old_profile = json.load(f)
            created = old_profile.get('metadata', {}).get('created')
            if created:
                profile_dict['metadata']['created'] = created
        except:
            pass
        
        profile_dict['metadata']['modified'] = datetime.utcnow().isoformat() + 'Z'
        
        # Normalize weights
        weights = profile_dict.get('signal_weights', {})
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            for key in weights:
                weights[key] = round(weights[key] / weight_sum, 3)
            profile_dict['signal_weights'] = weights
        
        # Save
        with open(path, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        logger.info(f"Updated profile: {filename}")
        
        return {
            "success": True,
            "filename": filename,
            "profile": profile_dict,
            "message": f"Profile updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.delete("/profiles/{filename}")
async def api_delete_profile(filename: str):
    """
    Delete a calibration profile
    
    Built-in profiles cannot be deleted.
    
    Args:
        filename: Profile filename to delete
    """
    try:
        # Protect built-in profiles
        protected = ['default.json', 'modern_fact_checker.json', 
                    'academic_gatekeeping.json', 'state_suppression.json',
                    'ideological_subversion.json']
        
        if filename in protected:
            raise HTTPException(
                status_code=403, 
                detail=f"Cannot delete built-in profile: {filename}"
            )
        
        path = os.path.join(PROFILE_DIR, filename)
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Profile not found: {filename}")
        
        os.remove(path)
        
        logger.info(f"Deleted profile: {filename}")
        
        return {
            "success": True,
            "deleted": filename,
            "message": f"Profile deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete profile {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.post("/profiles/{filename}/clone")
async def api_clone_profile(filename: str, new_name: str = Query(...)):
    """
    Clone an existing profile with a new name
    
    Args:
        filename: Source profile filename
        new_name: Name for the new profile
        
    Returns:
        New profile
    """
    try:
        path = os.path.join(PROFILE_DIR, filename)
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Profile not found: {filename}")
        
        with open(path, 'r') as f:
            profile = json.load(f)
        
        # Update metadata
        profile['metadata']['name'] = new_name
        profile['metadata']['based_on'] = filename
        profile['metadata']['author'] = 'user'
        
        now = datetime.utcnow().isoformat() + 'Z'
        profile['metadata']['created'] = now
        profile['metadata']['modified'] = now
        
        # Generate new filename
        new_filename = new_name.lower().replace(' ', '_').replace('-', '_') + '.json'
        new_path = os.path.join(PROFILE_DIR, new_filename)
        
        if os.path.exists(new_path):
            raise HTTPException(
                status_code=409,
                detail=f"Profile already exists: {new_filename}"
            )
        
        with open(new_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Cloned {filename} to {new_filename}")
        
        return {
            "success": True,
            "filename": new_filename,
            "profile": profile,
            "cloned_from": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clone profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Detection Endpoints ===

@calibration_router.post("/detect/suppression")
async def api_detect_suppression(request: DetectionRequest):
    """
    Run suppression detection with specified profile
    
    Args:
        topic: Topic/query to analyze
        profile: Profile filename to use (default: 'default.json')
        domain: Optional domain filter
        limit: Max claims to analyze
        
    Returns:
        Full detection result with signals and interpretation
    """
    try:
        # Load profile
        profile_path = os.path.join(PROFILE_DIR, request.profile)
        
        if not os.path.exists(profile_path):
            # Fall back to default
            logger.warning(f"Profile not found: {request.profile}, using default")
            profile_path = os.path.join(PROFILE_DIR, 'default.json')
            
            if not os.path.exists(profile_path):
                # No profiles at all - use built-in default
                profile_path = None
        
        # Initialize detector
        detector = SuppressionDetector(profile_path=profile_path)
        
        # Run detection
        result = detector.detect_suppression(
            topic=request.topic,
            domain=request.domain,
            claim_ids=None  # Force Topic Traversal for better coverage
        )
        
        detector.close()
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.post("/detect/suppression/with-profile")
async def api_detect_with_inline_profile(
    topic: str,
    profile: CalibrationProfile,
    domain: Optional[str] = None
):
    """
    Run suppression detection with an inline profile (not saved)
    
    Useful for testing profile changes before saving.
    
    Args:
        topic: Topic/query to analyze
        profile: Full profile object
        domain: Optional domain filter
        
    Returns:
        Detection result
    """
    try:
        profile_dict = profile.dict()
        
        # Initialize detector with inline profile
        detector = SuppressionDetector(profile_dict=profile_dict)
        
        result = detector.detect_suppression(
            topic=topic,
            domain=domain
        )
        
        detector.close()
        
        return {
            "success": True,
            "result": result,
            "profile_used": profile_dict.get('metadata', {}).get('name', 'inline')
        }
        
    except Exception as e:
        logger.error(f"Detection with inline profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@calibration_router.get("/detect/profiles/recommend")
async def api_recommend_profile(topic: str):
    """
    Recommend a profile based on topic keywords
    
    Simple heuristic matching - returns suggested profile.
    
    Args:
        topic: Topic to analyze
        
    Returns:
        Recommended profile filename and reason
    """
    topic_lower = topic.lower()
    
    # Simple keyword matching
    if any(kw in topic_lower for kw in ['paine', 'butler', 'revolution', 'whistleblower', 
                                         'sedition', 'treason', 'government']):
        return {
            "recommended": "state_suppression.json",
            "reason": "Topic appears related to state/institutional suppression",
            "confidence": 0.7
        }
    
    if any(kw in topic_lower for kw in ['medical', 'health', 'treatment', 'clinical',
                                         'fact-check', 'misinformation', 'deplatform']):
        return {
            "recommended": "modern_fact_checker.json",
            "reason": "Topic appears related to modern medical/platform suppression",
            "confidence": 0.7
        }
    
    if any(kw in topic_lower for kw in ['archaeology', 'academic', 'science', 'research',
                                         'peer-review', 'tenure', 'paradigm']):
        return {
            "recommended": "academic_gatekeeping.json",
            "reason": "Topic appears related to academic/scientific gatekeeping",
            "confidence": 0.7
        }
    
    if any(kw in topic_lower for kw in ['bezmenov', 'subversion', 'communist', 'infiltration',
                                         'institutional', 'ideological']):
        return {
            "recommended": "ideological_subversion.json",
            "reason": "Topic appears related to ideological subversion patterns",
            "confidence": 0.7
        }
    
    return {
        "recommended": "default.json",
        "reason": "No specific pattern detected, using balanced default",
        "confidence": 0.5
    }


# === Validation Endpoints ===

@calibration_router.post("/profiles/validate")
async def api_validate_profile(profile: CalibrationProfile):
    """
    Validate a profile without saving it
    
    Checks:
    - Weight normalization
    - Pattern validity
    - Threshold consistency
    
    Returns:
        Validation result with warnings
    """
    warnings = []
    errors = []
    
    profile_dict = profile.dict()
    
    # Check weights sum
    weights = profile_dict.get('signal_weights', {})
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        warnings.append(f"Weights sum to {weight_sum:.3f}, will be normalized to 1.0")
    
    # Check for empty patterns
    patterns = profile_dict.get('semantic_patterns', {})
    for category in ['suppression_experiences', 'institutional_actions', 
                     'dismissal_language', 'suppression_of_record']:
        cat_patterns = patterns.get(category, {})
        include = cat_patterns.get('include', [])
        if not include:
            warnings.append(f"No patterns defined for {category}")
    
    # Check thresholds
    score_levels = profile_dict.get('thresholds', {}).get('score_levels', {})
    if score_levels:
        if score_levels.get('low', 0) >= score_levels.get('moderate', 0.4):
            errors.append("Threshold 'low' should be less than 'moderate'")
        if score_levels.get('moderate', 0.4) >= score_levels.get('high', 0.6):
            errors.append("Threshold 'moderate' should be less than 'high'")
        if score_levels.get('high', 0.6) >= score_levels.get('critical', 0.8):
            errors.append("Threshold 'high' should be less than 'critical'")
    
    # Check scoring config
    scoring = profile_dict.get('scoring', {})
    if scoring.get('happenstance_max', 1) >= scoring.get('coincidence_max', 2):
        errors.append("happenstance_max should be less than coincidence_max")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_weights": {k: round(v/weight_sum, 3) for k, v in weights.items()} if weight_sum > 0 else weights
    }


# === Helper function to integrate with existing app ===

def register_calibration_routes(app):
    """
    Register calibration routes with an existing FastAPI app
    
    Usage:
        from calibration_api_endpoints import register_calibration_routes
        register_calibration_routes(app)
    """
    app.include_router(calibration_router)
    logger.info("Registered calibration API routes")
