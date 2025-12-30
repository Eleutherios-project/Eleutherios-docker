#!/usr/bin/env python3
"""
Aegis Insight - Suppression Pattern Detector v2

MAJOR CHANGES FROM v1:
- NEW PRIMARY SIGNAL: Suppression Narrative Detection
  Looks for suppression indicators IN PRIMARY claims, not just META attacks
  
- GOLDFINGER SCORING: Non-linear threshold-based scoring
  1 indicator = happenstance (0.10)
  2 indicators = coincidence (0.20)
  3+ indicators = enemy action (0.50+ logarithmic growth)
  
- CALIBRATION PROFILES: JSON-based configuration
  Users can tune patterns and weights for their domain

- SEMANTIC PATTERN MATCHING: Embedding-based similarity
  "imprisoned" matches "jailed", "incarcerated" etc.

Author: Aegis Insight Team
Date: November 2025
Version: 2.0
"""

import json
import math
import logging
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

# For semantic matching
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Falling back to keyword matching.")

# For LLM-based directionality validation
try:
    import requests
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral-nemo:12b')

# =====================================================
# VALIDATION HELPERS (Fix false positives in detection)
# =====================================================

# Negation patterns that invalidate suppression indicators
NEGATION_PATTERNS = [
    'not ', 'never ', "wasn't ", "weren't ", "didn't ", "doesn't ", "don't ",
    "no ", "hardly ", "barely ", "rarely ", "seldom ", "without ",
    "not necessarily", "no evidence of", "never actually", "wasn't really"
]

def is_negated(claim_text: str, pattern: str) -> bool:
    """
    Check if a pattern match is negated in the claim.
    Returns True if the pattern appears to be negated.
    
    Example: "The criticism was not necessarily malicious"
             Pattern "malicious" is negated by "not necessarily"
    """
    claim_lower = claim_text.lower()
    pattern_lower = pattern.lower()
    
    pattern_pos = claim_lower.find(pattern_lower)
    if pattern_pos == -1:
        return False
    
    # Look for negation in the 50 chars before the pattern
    prefix_start = max(0, pattern_pos - 50)
    prefix = claim_lower[prefix_start:pattern_pos]
    
    for neg in NEGATION_PATTERNS:
        if neg in prefix:
            return True
    
    return False

def topic_in_claim(claim_text: str, topic: str) -> bool:
    """
    Check if the topic (or any word from multi-word topic) appears in the claim.
    
    Example: topic="Mark Twain", claim="Gerhardt ruined the bust" -> False
             topic="Mark Twain", claim="Mark Twain was imprisoned" -> True
             topic="Smedley Butler", claim="Butler was called crazy" -> True
    """
    if not topic:
        return True  # No topic filter = allow all
    
    claim_lower = claim_text.lower()
    topic_lower = topic.lower()
    
    # Check exact topic match first
    if topic_lower in claim_lower:
        return True
    
    # Check individual words (for "Smedley Butler" matching "Butler")
    topic_words = topic_lower.split()
    for word in topic_words:
        # Skip very short words and common words
        if len(word) >= 4 and word in claim_lower:
            return True
    
    return False

def pattern_near_topic(claim_text: str, topic: str, pattern: str, max_distance: int = 12) -> bool:
    """
    Check if the matched pattern is within max_distance words of the topic.
    
    This filters out false positives like:
    - "Gerhardt ruined the bust he made of Mark Twain" 
      → "ruined" is 7 words away from "Twain" but topic isn't the subject
    
    Example: 
      claim="Butler was called crazy by Murphy"
      topic="Butler", pattern="crazy" → distance=2 words → True
      
      claim="Gerhardt ruined the bust he made of Mark Twain"  
      topic="Twain", pattern="ruined" → distance=7 words → borderline
      
    Args:
        claim_text: The full claim text
        topic: The search topic (e.g., "Mark Twain")
        pattern: The matched suppression pattern (e.g., "ruined")
        max_distance: Maximum word distance allowed (default 12)
    
    Returns:
        True if pattern is near topic, False otherwise
    """
    if not topic or not pattern:
        return True
    
    claim_lower = claim_text.lower()
    pattern_lower = pattern.lower()
    topic_lower = topic.lower()
    
    # Find pattern position
    pattern_pos = claim_lower.find(pattern_lower)
    if pattern_pos == -1:
        # Pattern not found as substring - this is a semantic match
        # For semantic matches, be more strict: check if ANY topic word 
        # appears in first half of claim (topic should be subject)
        words = claim_lower.split()
        first_half = ' '.join(words[:len(words)//2 + 2])
        
        topic_words = topic_lower.split()
        topic_in_first_half = any(
            tw in first_half for tw in topic_words if len(tw) >= 4
        )
        
        if not topic_in_first_half:
            return False  # Topic not prominent in claim
        
        return True  # Topic is in first half, allow semantic match
    
    # Find closest topic word position
    topic_words = topic_lower.split()
    min_distance = float('inf')
    
    for topic_word in topic_words:
        if len(topic_word) < 4:
            continue
        
        topic_pos = claim_lower.find(topic_word)
        if topic_pos == -1:
            continue
        
        # Count words between pattern and topic
        if pattern_pos < topic_pos:
            between = claim_lower[pattern_pos:topic_pos]
        else:
            between = claim_lower[topic_pos:pattern_pos]
        
        word_distance = len(between.split())
        min_distance = min(min_distance, word_distance)
    
    if min_distance == float('inf'):
        return True  # Topic not found, allow (shouldn't happen with topic_in_claim check)
    
    return min_distance <= max_distance


def validate_indicator_directionality(topic: str, claim_text: str, pattern: str, 
                                       category: str, timeout: float = 10.0) -> dict:
    """
    Use LLM to validate whether the topic entity is the TARGET of suppression
    or the one DOING the suppression/attack.
    
    Returns:
        {
            'valid': True/False,
            'direction': 'target' | 'attacker' | 'unclear',
            'reason': str
        }
    """
    if not LLM_AVAILABLE:
        return {'valid': True, 'direction': 'unclear', 'reason': 'LLM not available'}
    
    prompt = f"""Analyze this claim about "{topic}":

Claim: "{claim_text}"
Matched pattern: "{pattern}" ({category})

Question: In this claim, is "{topic}" the TARGET/VICTIM of the action described by "{pattern}", 
or is "{topic}" the one DOING/CAUSING that action to someone else?

Respond with ONLY one of these exact words:
- TARGET (if {topic} is receiving/suffering the action)
- ATTACKER (if {topic} is doing the action to others)
- UNCLEAR (if the relationship is ambiguous)

Your answer:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 20
                }
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip().upper()
            
            # Parse the answer
            if 'TARGET' in answer:
                return {'valid': True, 'direction': 'target', 'reason': 'LLM confirmed target'}
            elif 'ATTACKER' in answer:
                return {'valid': False, 'direction': 'attacker', 'reason': 'LLM: topic is attacker, not victim'}
            else:
                # Unclear - give benefit of doubt
                return {'valid': True, 'direction': 'unclear', 'reason': 'LLM: direction unclear'}
        else:
            return {'valid': True, 'direction': 'error', 'reason': f'LLM error: {response.status_code}'}
            
    except requests.exceptions.Timeout:
        return {'valid': True, 'direction': 'timeout', 'reason': 'LLM timeout - keeping indicator'}
    except Exception as e:
        return {'valid': True, 'direction': 'error', 'reason': f'LLM error: {str(e)}'}


class SuppressionNarrativeDetector:
    """
    Detects suppression patterns by finding accumulation of
    suppression indicators in PRIMARY claims.
    
    Uses Goldfinger scoring:
    - 1 indicator = happenstance
    - 2 indicators = coincidence  
    - 3+ indicators = enemy action (logarithmic growth)
    
    Four pattern categories:
    - suppression_experiences: Personal consequences (imprisoned, vilified, poverty)
    - institutional_actions: Formal actions (charged with, fired, deplatformed)
    - dismissal_language: Delegitimizing language (debunked, seditious, pseudoscience)
    - suppression_of_record: Historical erasure (omitted from textbooks, forgotten)
    """
    
    def __init__(self, profile: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize narrative detector with calibration profile
        
        Args:
            profile: Calibration profile dict (loaded from JSON)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.profile = profile
        self.scoring = profile.get('scoring', {})
        self.patterns = profile.get('semantic_patterns', {})
        
        # Load embedding model for semantic matching
        if SEMANTIC_AVAILABLE:
            self.logger.info("Loading embedding model for semantic pattern matching...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            self._embed_patterns()
        else:
            self.model = None
            self.pattern_embeddings = {}
        
        # LLM-based semantic matching (higher fidelity fallback)
        self.use_llm_semantic = self.patterns.get('use_llm_semantic', True)
        self.ollama_url = self.patterns.get('ollama_url', 'http://localhost:11434')
        self.llm_model = self.patterns.get('llm_model', 'mistral-nemo:12b')
        self.llm_semantic_cache = {}  # Cache LLM results to avoid re-processing
    
    def _embed_patterns(self):
        """Pre-compute embeddings for all pattern categories"""
        self.pattern_embeddings = {}
        
        for category in ['suppression_experiences', 'institutional_actions', 
                         'dismissal_language', 'suppression_of_record']:
            patterns = self.patterns.get(category, {})
            include = patterns.get('include', [])
            exclude = patterns.get('exclude', [])
            
            if include:
                self.logger.debug(f"Embedding {len(include)} patterns for {category}")
                self.pattern_embeddings[category] = {
                    'include': self.model.encode(include),
                    'include_texts': include,
                    'exclude': self.model.encode(exclude) if exclude else None,
                    'exclude_texts': exclude
                }
            else:
                self.pattern_embeddings[category] = {
                    'include': None,
                    'include_texts': [],
                    'exclude': None,
                    'exclude_texts': []
                }
    
    def _llm_semantic_match(self, claim_text: str, category: str) -> Dict:
        """
        Use LLM to determine if claim indicates suppression pattern.
        Higher fidelity than embedding similarity for nuanced language.
        
        Args:
            claim_text: The claim to analyze
            category: The pattern category to check
            
        Returns:
            Dict with matches, pattern_type, key_phrase, confidence
        """
        if not LLM_AVAILABLE:
            return {'matches': False, 'confidence': 0}
        
        # Check cache first
        cache_key = f"{claim_text[:100]}:{category}"
        if cache_key in self.llm_semantic_cache:
            return self.llm_semantic_cache[cache_key]
        
        # Category-specific prompts
        category_descriptions = {
            'suppression_experiences': 'personal consequences like imprisonment, exile, poverty, vilification, being silenced, fired, or career destruction',
            'institutional_actions': 'formal institutional actions like charges, indictments, convictions, bans, censorship, deplatforming, or official condemnation',
            'dismissal_language': 'delegitimizing language like being called crazy, radical, dangerous, seditious, a conspiracy theorist, or pseudoscientific',
            'suppression_of_record': 'historical erasure like being left out of history, omitted from textbooks, forgotten, memory suppressed, documents destroyed, or cover-ups'
        }
        
        category_desc = category_descriptions.get(category, 'suppression or censorship')
        
        prompt = f"""Analyze if this claim describes {category_desc}.

CLAIM: "{claim_text}"

Does this claim describe someone experiencing {category_desc}?

Reply ONLY with JSON (no other text):
{{"matches": true/false, "key_phrase": "the specific phrase indicating this pattern or null", "confidence": 0.0-1.0}}"""

        try:
            resp = requests.post(
                f'{self.ollama_url}/api/generate',
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=15
            )
            response_text = resp.json().get('response', '')
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                result['match_type'] = 'llm_semantic'
                self.llm_semantic_cache[cache_key] = result
                return result
        except Exception as e:
            self.logger.debug(f"LLM semantic match error: {e}")
        
        result = {'matches': False, 'confidence': 0, 'key_phrase': None}
        self.llm_semantic_cache[cache_key] = result
        return result

    def detect(self, claims: List[Dict], topic: str = None) -> Dict:
        """
        Detect suppression narrative in a set of claims
        
        Args:
            claims: List of claim dicts with claim_text, claim_type, etc.
            topic: Optional topic for LLM directionality validation
            
        Returns:
            {
                'suppression_score': float (0.0-1.0),
                'indicators_found': int,
                'indicator_details': [...],
                'scoring_mode': str,
                'interpretation': str,
                'category_breakdown': {...}
            }
        """
        
        # Find all suppression indicators in PRIMARY claims
        indicators = self._find_indicators(claims, topic)
        
        # Apply Goldfinger scoring
        score = self._calculate_score(len(indicators))
        
        # Categorize indicators
        category_breakdown = defaultdict(int)
        for ind in indicators:
            category_breakdown[ind['category']] += 1
        
        # Generate interpretation
        interpretation = self._interpret(score, indicators, dict(category_breakdown))
        
        return {
            'score': round(score, 3),
            'indicators_found': len(indicators),
            'indicator_details': indicators[:20],  # Limit for display
            'scoring_mode': self.scoring.get('mode', 'goldfinger'),
            'interpretation': interpretation,
            'category_breakdown': dict(category_breakdown)
        }
    
    def _find_indicators(self, claims: List[Dict], topic: str = None) -> List[Dict]:
        """Find all suppression indicators in claims with directionality validation"""
        
        raw_indicators = []
        threshold = self.patterns.get('match_threshold', 0.80)  # Raised from 0.75 to reduce false positives
        match_mode = self.patterns.get('match_mode', 'semantic')
        
        for claim in claims:
            # INCLUDE ALL CLAIM TYPES for suppression narrative detection
            # Historical suppression is often documented in CONTEXTUAL claims
            # (biographical facts like "was imprisoned", "died in poverty")
            # The LLM directionality check will filter out irrelevant matches
            
            claim_text = claim.get('claim_text', '')
            if not claim_text:
                continue
            
            # Skip excluded claims (soft-deleted via Data Management)
            if claim.get('excluded', False):
                continue
            
            # FIX 1: Topic must appear in claim to be considered
            # Filters out unrelated claims like "Gerhardt ruined the bust" for topic "Mark Twain"
            if topic and not claim.get("from_topic_traversal") and not topic_in_claim(claim_text, topic):
                continue
            
            # Get embedding for semantic matching
            if SEMANTIC_AVAILABLE and self.model and match_mode in ['semantic', 'broad']:
                claim_embedding = self.model.encode([claim_text])[0]
            else:
                claim_embedding = None
            
            # Check each pattern category
            for category in ['suppression_experiences', 'institutional_actions',
                           'dismissal_language', 'suppression_of_record']:
                
                match_result = self._check_pattern_match(
                    claim_text,
                    claim_embedding, 
                    category, 
                    threshold,
                    match_mode
                )
                
                if match_result['matches']:
                    # FIX 2: Check for negation before counting as indicator
                    # Filters out "not malicious", "never suppressed", etc.
                    matched_pattern = match_result['best_match']
                    if matched_pattern and is_negated(claim_text, matched_pattern):
                        continue  # Skip negated patterns
                    
                    # FIX 4: Proximity check - pattern should be near topic
                    # Filters out "Gerhardt ruined the bust he made of Mark Twain"
                    # where "ruined" is far from the topic entity
                    if matched_pattern and topic and not pattern_near_topic(claim_text, topic, matched_pattern, max_distance=5):
                        continue  # Skip - pattern too far from topic
                    
                    raw_indicators.append({
                        'claim_id': claim.get('claim_id', 'unknown'),
                        'element_id': claim.get('element_id'),  # Neo4j element ID for graph matching
                        'claim_text': claim_text[:300],  # Truncate for display
                        'category': category,
                        'similarity': match_result['similarity'],
                        'matched_pattern': matched_pattern,
                        'match_mode': match_mode
                    })
                    break  # Count each claim only once (strongest category)
        
        # === LLM DIRECTIONALITY VALIDATION ===
        # Only validate if we have a topic and LLM is available
        if topic and LLM_AVAILABLE and raw_indicators:
            validated_indicators = []
            rejected_count = 0
            
            for ind in raw_indicators:
                validation = validate_indicator_directionality(
                    topic=topic,
                    claim_text=ind['claim_text'],
                    pattern=ind['matched_pattern'],
                    category=ind['category']
                )
                
                if validation['valid']:
                    ind['direction_validated'] = True
                    ind['direction'] = validation['direction']
                    validated_indicators.append(ind)
                else:
                    # Log rejected indicators for debugging
                    self.logger.debug(
                        f"Rejected indicator: {ind['matched_pattern']} - {validation['reason']}"
                    )
                    rejected_count += 1
            
            if rejected_count > 0:
                self.logger.info(
                    f"LLM directionality: kept {len(validated_indicators)}, "
                    f"rejected {rejected_count} (topic was attacker, not victim)"
                )
            
            return validated_indicators
        else:
            # No LLM validation - return all matches
            return raw_indicators
    
    def _check_pattern_match(self, 
                             claim_text: str,
                             claim_embedding: Optional[np.ndarray],
                             category: str, 
                             threshold: float,
                             match_mode: str) -> Dict:
        """
        Check if claim matches patterns in category
        
        Uses HYBRID approach:
        1. First try keyword matching (most reliable)
        2. Then try semantic matching if keywords don't match
        
        Modes:
        - keyword: Keyword substring match only
        - semantic: Keyword first, then semantic fallback
        - hybrid: Same as semantic (default)
        """
        
        embeddings = self.pattern_embeddings.get(category, {})
        include_emb = embeddings.get('include')
        include_texts = embeddings.get('include_texts', [])
        exclude_emb = embeddings.get('exclude')
        exclude_texts = embeddings.get('exclude_texts', [])
        
        if not include_texts:
            return {'matches': False, 'similarity': 0.0, 'best_match': None}
        
        claim_lower = claim_text.lower()
        
        # === STEP 1: KEYWORD MATCHING (Primary - Most Reliable) ===
        # Check exclude patterns first
        for exclude_pattern in exclude_texts:
            if exclude_pattern.lower() in claim_lower:
                return {'matches': False, 'similarity': 0.0, 'best_match': None}
        
        # Check include patterns with keyword matching
        for include_pattern in include_texts:
            if include_pattern.lower() in claim_lower:
                return {
                    'matches': True,
                    'similarity': 1.0,  # Exact keyword match
                    'best_match': include_pattern,
                    'match_type': 'keyword'
                }
        
        # === STEP 2: LLM SEMANTIC MATCHING (High Fidelity Fallback) ===
        # Pre-filter: Only call LLM if embedding has some similarity (saves ~80% of LLM calls)
        should_try_llm = False
        if self.use_llm_semantic and LLM_AVAILABLE and claim_embedding is not None and include_emb is not None:
            quick_sims = cosine_similarity([claim_embedding], include_emb)[0]
            if float(np.max(quick_sims)) >= 0.30:  # Only try LLM if some embedding similarity
                should_try_llm = True
        
        if should_try_llm:
            llm_result = self._llm_semantic_match(claim_text, category)
            if llm_result.get('matches') and llm_result.get('confidence', 0) >= 0.7:
                return {
                    'matches': True,
                    'similarity': llm_result.get('confidence', 0.85),
                    'best_match': llm_result.get('key_phrase', 'LLM detected'),
                    'match_type': 'llm_semantic'
                }
        
        # === STEP 3: EMBEDDING MATCHING (Lower priority fallback) ===
        if match_mode in ['semantic', 'hybrid', 'broad'] and claim_embedding is not None and include_emb is not None:
            # Adjust threshold based on mode
            if match_mode == 'broad':
                semantic_threshold = 0.55  # Lower for broad
            else:
                semantic_threshold = threshold  # Use profile threshold
            
            # Calculate similarities to include patterns
            include_sims = cosine_similarity([claim_embedding], include_emb)[0]
            max_include_idx = int(np.argmax(include_sims))
            max_include_sim = float(include_sims[max_include_idx])
            
            if max_include_sim >= semantic_threshold:
                # Check exclude patterns semantically
                if exclude_emb is not None and len(exclude_emb) > 0:
                    exclude_sims = cosine_similarity([claim_embedding], exclude_emb)[0]
                    max_exclude_sim = float(np.max(exclude_sims))
                    
                    if max_exclude_sim >= semantic_threshold:
                        # Excluded pattern matches - don't count
                        return {'matches': False, 'similarity': max_include_sim, 'best_match': None}
                
                # Get the matched pattern text
                best_match = include_texts[max_include_idx] if max_include_idx < len(include_texts) else None
                
                return {
                    'matches': True,
                    'similarity': max_include_sim,
                    'best_match': best_match,
                    'match_type': 'semantic'
                }
        
        # No match found
        return {'matches': False, 'similarity': 0.0, 'best_match': None}
    
    def _calculate_score(self, count: int) -> float:
        """
        Apply Goldfinger scoring
        
        Formula:
        - count <= happenstance_max: 0.10 * count
        - count <= coincidence_max: 0.20
        - count > coincidence_max: base + log(excess) * factor
        """
        
        if count == 0:
            return 0.0
        
        mode = self.scoring.get('mode', 'goldfinger')
        
        if mode == 'linear':
            # Simple linear scoring (legacy)
            return min(count * 0.1, 0.95)
        
        # Goldfinger scoring
        happenstance_max = self.scoring.get('happenstance_max', 1)
        coincidence_max = self.scoring.get('coincidence_max', 2)
        enemy_action_base = self.scoring.get('enemy_action_base', 0.50)
        log_factor = self.scoring.get('logarithmic_factor', 0.20)
        max_score = self.scoring.get('max_score', 0.95)
        
        if count <= happenstance_max:
            # Happenstance: low score
            return 0.10 * count
        
        elif count <= coincidence_max:
            # Coincidence: slightly higher
            return 0.20
        
        else:
            # Enemy action: base + logarithmic growth
            excess = count - coincidence_max + 1
            score = enemy_action_base + math.log(excess) * log_factor
            return min(score, max_score)
    
    def _interpret(self, score: float, indicators: List[Dict], categories: Dict) -> str:
        """Generate human-readable interpretation"""
        
        thresholds = self.profile.get('thresholds', {}).get('score_levels', {})
        
        if score >= thresholds.get('critical', 0.80):
            level = "CRITICAL"
        elif score >= thresholds.get('high', 0.60):
            level = "HIGH"
        elif score >= thresholds.get('moderate', 0.40):
            level = "MODERATE"
        elif score >= thresholds.get('low', 0.20):
            level = "LOW"
        else:
            level = "MINIMAL"
        
        # Category summary
        category_parts = []
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            readable_cat = cat.replace('_', ' ')
            category_parts.append(f"{count} {readable_cat}")
        
        category_summary = ", ".join(category_parts) if category_parts else "none"
        
        return f"{level} suppression pattern detected. Found {len(indicators)} indicators: {category_summary}."


class SuppressionDetector:
    """
    Main suppression detector with calibration profile support
    
    Combines:
    - Suppression Narrative Detection (primary signal, Goldfinger scored)
    - META Claim Density
    - Network Isolation  
    - Evidence Avoidance
    - Authority Mismatch
    
    Uses calibration profiles for customization per domain.
    """
    
    # Default weights (used if profile doesn't specify)
    DEFAULT_WEIGHTS = {
        'suppression_narrative': 0.40,
        'meta_claim_density': 0.15,
        'network_isolation': 0.10,
        'evidence_avoidance': 0.20,
        'authority_mismatch': 0.15
    }
    
    # Default profile path
    DEFAULT_PROFILE_DIR = 'calibration_profiles'
    
    def __init__(self,
                 neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "aegistrusted",
                 profile_path: Optional[str] = None,
                 profile_dict: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize suppression detector
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            profile_path: Path to calibration profile JSON file
            profile_dict: Calibration profile as dict (alternative to file)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Load calibration profile
        if profile_dict:
            self.profile = profile_dict
            self.logger.info(f"Using provided profile: {profile_dict.get('metadata', {}).get('name', 'unnamed')}")
        elif profile_path:
            self.profile = self._load_profile(profile_path)
        else:
            self.profile = self._default_profile()
        
        self.weights = self.profile.get('signal_weights', self.DEFAULT_WEIGHTS)
        self.thresholds = self.profile.get('thresholds', {})
        
        # Normalize weights
        self._normalize_weights()
        
        # Initialize narrative detector
        self.narrative_detector = SuppressionNarrativeDetector(self.profile, self.logger)
        
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            self.driver.verify_connectivity()
            self.logger.info("✓ Connected to Neo4j")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _load_profile(self, path: str) -> Dict:
        """Load calibration profile from JSON file"""
        try:
            with open(path, 'r') as f:
                profile = json.load(f)
            
            self.logger.info(f"Loaded profile: {profile.get('metadata', {}).get('name', path)}")
            return profile
            
        except FileNotFoundError:
            self.logger.warning(f"Profile not found: {path}. Using default.")
            return self._default_profile()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in profile {path}: {e}")
            return self._default_profile()
    
    def _default_profile(self) -> Dict:
        """Return default profile - loads State Suppression as default for demo data"""
        # Try to load state_suppression.json as default
        # Look in common locations
        possible_paths = [
            Path(__file__).parent / 'data' / 'calibration_profiles' / 'state_suppression.json',
            Path('data/calibration_profiles/state_suppression.json'),
            Path('/media/bob/RAID11/DataShare/AegisTrustNet/data/calibration_profiles/state_suppression.json'),
        ]
        
        for default_profile_path in possible_paths:
            if default_profile_path.exists():
                try:
                    with open(default_profile_path) as f:
                        profile = json.load(f)
                        self.logger.info(f"Loaded default profile: State/Institutional Suppression")
                        return profile
                except Exception as e:
                    self.logger.warning(f"Could not load default profile from {default_profile_path}: {e}")
        
        # Fallback to hardcoded default
        return {
            'schema_version': '2.0',
            'metadata': {
                'name': 'Default - Balanced',
                'description': 'General purpose detection'
            },
            'scoring': {
                'mode': 'goldfinger',
                'happenstance_max': 1,
                'coincidence_max': 2,
                'enemy_action_base': 0.50,
                'logarithmic_factor': 0.20,
                'max_score': 0.95
            },
            'semantic_patterns': {
                'match_mode': 'semantic',
                'match_threshold': 0.75,
                'suppression_experiences': {
                    'include': [
                        'imprisoned', 'jailed', 'arrested',
                        'vilified', 'smeared', 'attacked',
                        'exiled', 'banished', 'fled',
                        'poverty', 'destitute', 'abandoned', 'forgotten',
                        'censored', 'banned', 'suppressed',
                        'career destroyed', 'reputation ruined',
                        'died alone', 'few mourners'
                    ],
                    'exclude': []
                },
                'institutional_actions': {
                    'include': [
                        'charged with', 'accused of', 'indicted',
                        'convicted', 'sentenced', 'condemned',
                        'fired', 'dismissed', 'tenure denied',
                        'funding cut', 'deplatformed'
                    ],
                    'exclude': []
                },
                'dismissal_language': {
                    'include': [
                        'debunked', 'discredited', 'conspiracy theory',
                        'misinformation', 'pseudoscience', 'fringe',
                        'no evidence', 'baseless', 'unfounded'
                    ],
                    'exclude': []
                },
                'suppression_of_record': {
                    'include': [
                        'rarely mentioned', 'omitted from', 'forgotten',
                        'ignored by', 'written out of', 'largely unknown'
                    ],
                    'exclude': []
                }
            },
            'signal_weights': self.DEFAULT_WEIGHTS.copy(),
            'thresholds': {
                'min_claims_for_analysis': 5,
                'score_levels': {
                    'critical': 0.80,
                    'high': 0.60,
                    'moderate': 0.40,
                    'low': 0.20
                }
            }
        }
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            self.logger.warning(f"Weights sum to {weight_sum:.2f}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] = self.weights[key] / weight_sum
    
    def set_profile(self, profile: Dict):
        """
        Set calibration profile at runtime
        
        Args:
            profile: Calibration profile dict
        """
        self.profile = profile
        self.weights = profile.get('signal_weights', self.DEFAULT_WEIGHTS)
        self.thresholds = profile.get('thresholds', {})
        self._normalize_weights()
        
        # Reinitialize narrative detector with new profile
        self.narrative_detector = SuppressionNarrativeDetector(profile, self.logger)
        
        self.logger.info(f"Profile updated: {profile.get('metadata', {}).get('name', 'unnamed')}")
    
    def detect_suppression(self, 
                          topic: str,
                          domain: Optional[str] = None,
                          claim_ids: Optional[List[str]] = None,
                          claims: Optional[List[Dict]] = None) -> Dict:
        """
        Detect suppression patterns for a given topic
        
        Args:
            topic: Topic to analyze (used for claim search)
            domain: Optional domain filter
            claims: Optional pre-fetched claims (skips database query)
            
        Returns:
            {
                'topic': str,
                'suppression_score': 0.0-1.0,
                'level': str (CRITICAL/HIGH/MODERATE/LOW/MINIMAL),
                'confidence': 0.0-1.0,
                'signals': {...},
                'interpretation': str,
                'claim_counts': {...},
                'claims_analyzed': int,
                'profile_used': str
            }
        """
        
        self.logger.info(f"Analyzing suppression patterns for topic: {topic}")
        self.logger.info(f"Using profile: {self.profile.get('metadata', {}).get('name', 'default')}")
        
        # Get claims - priority: provided claims > claim_ids > topic search
        if claims is None:
            with self.driver.session() as session:
                if claim_ids and len(claim_ids) > 0:
                    claims = self._get_claims_by_ids(session, claim_ids)
                    self.logger.info(f"Using {len(claims)} claims from semantic search")
                else:
                    claims = self._get_claims_for_topic(session, topic, domain)
                    self.logger.info(f"Using {len(claims)} claims from text search (legacy)")
        
        if not claims:
            self.logger.warning(f"No claims found for topic: {topic}")
            return self._empty_result(topic)
        
        min_claims = self.thresholds.get('min_claims_for_analysis', 5)
        if len(claims) < min_claims:
            self.logger.warning(f"Only {len(claims)} claims found, below minimum {min_claims}")
        
        self.logger.info(f"Found {len(claims)} claims to analyze")
        
        # Calculate each signal
        signals = {}
        
        # PRIMARY SIGNAL: Suppression Narrative (Goldfinger scored)
        # Pass topic for LLM directionality validation
        narrative_result = self.narrative_detector.detect(claims, topic=topic)
        signals['suppression_narrative'] = {
            'score': narrative_result['score'],
            'indicators_found': narrative_result['indicators_found'],
            'category_breakdown': narrative_result['category_breakdown'],
            'details': narrative_result['indicator_details'],
            'interpretation': narrative_result['interpretation']
        }
        
        # SECONDARY SIGNALS
        signals['meta_claim_density'] = self._calculate_meta_density(claims)
        signals['network_isolation'] = self._calculate_network_isolation(claims)
        signals['evidence_avoidance'] = self._calculate_evidence_avoidance(claims)
        signals['authority_mismatch'] = self._calculate_authority_mismatch(claims)
        
        # Aggregate using weights - exclude non-functional signals from average
        total_score = 0.0
        total_weight = 0.0
        for signal_name, weight in self.weights.items():
            signal_data = signals.get(signal_name, {})
            signal_score = signal_data.get('score', 0.0)
            
            # Skip signals that are non-functional (simplified implementation with 0 score)
            if signal_data.get('note') and signal_score == 0:
                self.logger.debug(f"Excluding non-functional signal: {signal_name}")
                continue
            
            total_score += signal_score * weight
            total_weight += weight
        
        # Normalize by actual weights used (so non-functional signals don't drag down average)
        if total_weight > 0 and total_weight < 1.0:
            total_score = total_score / total_weight
        
        # Determine level
        score_levels = self.thresholds.get('score_levels', {})
        if total_score >= score_levels.get('critical', 0.80):
            level = "CRITICAL"
        elif total_score >= score_levels.get('high', 0.60):
            level = "HIGH"
        elif total_score >= score_levels.get('moderate', 0.40):
            level = "MODERATE"
        elif total_score >= score_levels.get('low', 0.20):
            level = "LOW"
        else:
            level = "MINIMAL"
        
        # Calculate confidence
        confidence = self._calculate_confidence(signals, len(claims))
        
        # Generate interpretation
        interpretation = self._interpret_score(total_score, level, signals)
        
        # Count claim types
        claim_counts = self._count_claim_types(claims)
        
        # Build affected_claims list for graph highlighting
        # These are the claims that matched suppression indicators
        # Include both claim_id (our ID) and graph_id (for D3 visualization matching)
        affected_claims = []
        narrative_details = signals.get('suppression_narrative', {}).get('details', [])
        for indicator in narrative_details:
            element_id = indicator.get('element_id')
            # Build graph-compatible ID: "claim-{element_id}"
            graph_id = f"claim-{element_id}" if element_id else None
            
            affected_claims.append({
                'id': indicator.get('claim_id'),
                'claim_id': indicator.get('claim_id'),
                'graph_id': graph_id,  # ID used by graph renderer
                'claim_text': indicator.get('claim_text'),
                'suppression_score': 1.0,  # Matched indicator = high score
                'category': indicator.get('category'),
                'pattern': indicator.get('matched_pattern')
            })
        
        return {
            'topic': topic,
            'suppression_score': round(total_score, 3),
            'level': level,
            'confidence': round(confidence, 3),
            'signals': signals,
            'interpretation': interpretation,
            'claim_counts': claim_counts,
            'claims_analyzed': len(claims),
            'profile_used': self.profile.get('metadata', {}).get('name', 'default'),
            'affected_claims': affected_claims  # For graph highlighting
        }
    
    # High-precision suppression terms (empirically validated)
    HIGH_PRECISION_TERMS = [
        'suppressed', 'censored', 'banned', 'objected', 'withheld',
        'forbidden', 'unpublished', 'silenced', 'blacklisted', 'redacted',
        'classified', 'concealed', 'prohibited'
    ]

    def _get_claims_for_topic(self, 
                              session,
                              topic: str,
                              domain: Optional[str] = None,
                              limit: int = 500) -> List[Dict]:
        """
        Two-stage prioritized claim retrieval using Topic node traversal.
        
        Stage 1: Fetch high-salience claims (META type OR high-precision terms)
        Stage 2: Fill remaining slots with random sample
        
        Uses Topic node traversal (not Entity matching) to capture ALL claims
        from documents about the topic, including those using pronouns.
        
        Empirical basis: 12.2% META ratio for high-precision terms vs 6.3% for
        noisy terms = 94% lift in signal quality.
        
        Args:
            session: Neo4j session
            topic: Topic name to search
            domain: Optional domain filter
            limit: Maximum claims to return
            
        Returns:
            List of claim dicts, prioritized by suppression salience
        """
        
        # Normalize topic: handle hyphen/space variations
        normalized_topic = topic.replace('-', ' ').replace('  ', ' ')
        topic_words = [w.strip() for w in normalized_topic.split() if len(w.strip()) > 2]
        
        self.logger.info(f"=== PATCHED V2 === Fetching claims for topic '{topic}' (words: {topic_words}) using Topic traversal + prioritized sampling...")
        
        # Stage 1: Priority claims via Topic traversal (META + high-precision terms)
        priority_query = """
            MATCH (t:Topic)
            WHERE ALL(word IN $topic_words WHERE toLower(t.name) CONTAINS toLower(word))
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WITH c, d,
                 CASE WHEN c.claim_type = 'META' THEN 2 ELSE 0 END +
                 CASE WHEN any(term IN $hp_terms WHERE toLower(c.claim_text) CONTAINS term) THEN 3 ELSE 0 END
                 AS priority
            WHERE priority > 0
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.temporal_data as temporal_data,
                   c.geographic_data as geographic_data,
                   c.citation_data as citation_data,
                   d.source_file as doc_source,
                   elementId(c) as element_id,
                   priority
            ORDER BY priority DESC
            LIMIT $limit
        """
        
        claims = {}  # Use dict to deduplicate by element_id
        
        try:
            result = session.run(
                priority_query,
                topic_words=topic_words,
                hp_terms=self.HIGH_PRECISION_TERMS,
                limit=limit
            )
            
            for record in result:
                claim = dict(record)
                element_id = claim.get('element_id')
                if element_id and element_id not in claims:
                    # Use doc_source if source_file is missing
                    if not claim.get('source_file'):
                        claim['source_file'] = claim.get('doc_source')
                    # Parse JSON fields
                    for field in ['temporal_data', 'geographic_data', 'citation_data']:
                        if claim.get(field):
                            try:
                                claim[field] = json.loads(claim[field]) if isinstance(claim[field], str) else claim[field]
                            except:
                                claim[field] = {}
                    claim["from_topic_traversal"] = True
                    claims[element_id] = claim
            
            self.logger.info(f"Stage 1 (priority): Found {len(claims)} high-salience claims")
            
        except Exception as e:
            self.logger.warning(f"Topic traversal failed: {e}, falling back to text search")
            return self._get_claims_for_topic_text_fallback(session, topic, domain, limit)
        
        # If we have enough, return
        if len(claims) >= limit:
            return list(claims.values())[:limit]
        
        # Stage 2: Fill with random sample via Topic traversal
        remaining = limit - len(claims)
        seen_ids = list(claims.keys())
        
        random_query = """
            MATCH (t:Topic)
            WHERE ALL(word IN $topic_words WHERE toLower(t.name) CONTAINS toLower(word))
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WHERE NOT elementId(c) IN $seen_ids
            WITH c, d, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.temporal_data as temporal_data,
                   c.geographic_data as geographic_data,
                   c.citation_data as citation_data,
                   d.source_file as doc_source,
                   elementId(c) as element_id,
                   0 as priority
        """
        
        try:
            result = session.run(
                random_query,
                topic_words=topic_words,
                seen_ids=seen_ids,
                limit=remaining
            )
            
            random_count = 0
            for record in result:
                claim = dict(record)
                element_id = claim.get('element_id')
                if element_id and element_id not in claims:
                    if not claim.get('source_file'):
                        claim['source_file'] = claim.get('doc_source')
                    for field in ['temporal_data', 'geographic_data', 'citation_data']:
                        if claim.get(field):
                            try:
                                claim[field] = json.loads(claim[field]) if isinstance(claim[field], str) else claim[field]
                            except:
                                claim[field] = {}
                    claim["from_topic_traversal"] = True
                    claims[element_id] = claim
                    random_count += 1
            
            self.logger.info(f"Stage 2 (random fill): Added {random_count} claims")
            
        except Exception as e:
            self.logger.warning(f"Random fill failed: {e}")
        
        # If Topic traversal returned nothing, try text search fallback
        if len(claims) == 0:
            self.logger.info("No claims via Topic traversal, trying text search fallback...")
            return self._get_claims_for_topic_text_fallback(session, topic, domain, limit)
        
        self.logger.info(f"Total: {len(claims)} claims for analysis")
        return list(claims.values())

    def _get_claims_for_topic_text_fallback(self, 
                                            session,
                                            topic: str,
                                            domain: Optional[str] = None,
                                            limit: int = 500) -> List[Dict]:
        """
        Fallback: Text search when no Topic node exists.
        Still uses two-stage prioritized approach.
        """
        
        self.logger.info(f"Text search fallback for '{topic}'...")
        
        # Stage 1: Priority claims with topic in text
        priority_query = """
            MATCH (c:Claim)
            WHERE toLower(c.claim_text) CONTAINS toLower($topic)
            WITH c,
                 CASE WHEN c.claim_type = 'META' THEN 2 ELSE 0 END +
                 CASE WHEN any(term IN $hp_terms WHERE toLower(c.claim_text) CONTAINS term) THEN 3 ELSE 0 END
                 AS priority
            WHERE priority > 0
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.temporal_data as temporal_data,
                   c.geographic_data as geographic_data,
                   c.citation_data as citation_data,
                   elementId(c) as element_id,
                   priority
            ORDER BY priority DESC
            LIMIT $limit
        """
        
        claims = {}
        
        try:
            result = session.run(
                priority_query,
                topic=topic,
                hp_terms=self.HIGH_PRECISION_TERMS,
                limit=limit
            )
            
            for record in result:
                claim = dict(record)
                element_id = claim.get('element_id')
                if element_id and element_id not in claims:
                    for field in ['temporal_data', 'geographic_data', 'citation_data']:
                        if claim.get(field):
                            try:
                                claim[field] = json.loads(claim[field]) if isinstance(claim[field], str) else claim[field]
                            except:
                                claim[field] = {}
                    claim["from_topic_traversal"] = True
                    claims[element_id] = claim
            
            self.logger.info(f"Text fallback Stage 1: {len(claims)} priority claims")
            
        except Exception as e:
            self.logger.warning(f"Priority text search failed: {e}")
        
        if len(claims) >= limit:
            return list(claims.values())[:limit]
        
        # Stage 2: Random text search
        remaining = limit - len(claims)
        seen_ids = list(claims.keys())
        
        random_query = """
            MATCH (c:Claim)
            WHERE toLower(c.claim_text) CONTAINS toLower($topic)
            AND NOT elementId(c) IN $seen_ids
            WITH c, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.temporal_data as temporal_data,
                   c.geographic_data as geographic_data,
                   c.citation_data as citation_data,
                   elementId(c) as element_id,
                   0 as priority
        """
        
        try:
            result = session.run(
                random_query,
                topic=topic,
                seen_ids=seen_ids,
                limit=remaining
            )
            
            for record in result:
                claim = dict(record)
                element_id = claim.get('element_id')
                if element_id and element_id not in claims:
                    for field in ['temporal_data', 'geographic_data', 'citation_data']:
                        if claim.get(field):
                            try:
                                claim[field] = json.loads(claim[field]) if isinstance(claim[field], str) else claim[field]
                            except:
                                claim[field] = {}
                    claim["from_topic_traversal"] = True
                    claims[element_id] = claim
            
            self.logger.info(f"Text fallback Stage 2: total {len(claims)} claims")
            
        except Exception as e:
            self.logger.warning(f"Random text search failed: {e}")
        
        return list(claims.values())


    def _get_claims_by_ids(self, session, claim_ids: List[str]) -> List[Dict]:
        """Retrieve claims by their element IDs (from semantic search)"""
        
        if not claim_ids:
            return []
        
        query = """
        MATCH (c:Claim)
        WHERE elementId(c) IN $claim_ids
        RETURN c.claim_id as claim_id,
               elementId(c) as element_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.temporal_data as temporal_data,
               c.geographic_data as geographic_data,
               c.citation_data as citation_data,
               c.source_file as source_file
        """
        
        result = session.run(query, claim_ids=claim_ids)
        
        claims = []
        for record in result:
            claim = dict(record)
            
            # Parse JSON fields
            for field in ['temporal_data', 'geographic_data', 'citation_data']:
                if claim.get(field):
                    try:
                        claim[field] = json.loads(claim[field])
                    except:
                        claim[field] = {}
            
            claims.append(claim)
        
        return claims
    
    def _calculate_meta_density(self, claims: List[Dict]) -> Dict:
        """Calculate META claim density signal"""
        
        meta_count = 0
        primary_count = 0
        
        for claim in claims:
            claim_type = claim.get('claim_type', '').upper()
            if claim_type == 'META':
                meta_count += 1
            elif claim_type == 'PRIMARY':
                primary_count += 1
        
        total = meta_count + primary_count
        if total == 0:
            return {
                'score': 0.0,
                'meta_count': 0,
                'primary_count': 0,
                'meta_ratio': 0.0,
                'interpretation': 'No PRIMARY or META claims found'
            }
        
        meta_ratio = meta_count / total
        
        # Score: High META ratio = potential suppression
        # But cap the score - META claims alone aren't definitive
        score = min(meta_ratio * 1.5, 0.8)
        
        return {
            'score': round(score, 3),
            'meta_count': meta_count,
            'primary_count': primary_count,
            'meta_ratio': round(meta_ratio, 3),
            'interpretation': self._interpret_meta_density(meta_ratio, meta_count, primary_count)
        }
    
    def _calculate_network_isolation(self, claims: List[Dict]) -> Dict:
        """Calculate network isolation signal (citation asymmetry)"""
        
        # Count claims with citation data
        with_citations = 0
        citation_count = 0
        
        for claim in claims:
            citation_data = claim.get('citation_data', {})
            if citation_data:
                citations = citation_data.get('citations', [])
                if citations:
                    with_citations += 1
                    citation_count += len(citations)
        
        if len(claims) == 0:
            return {
                'score': 0.0,
                'claims_with_citations': 0,
                'total_citations': 0,
                'interpretation': 'No claims to analyze'
            }
        
        citation_ratio = with_citations / len(claims)
        
        # Low citation ratio = potential isolation
        # Invert: high isolation = low citations
        isolation_score = 1.0 - citation_ratio
        
        return {
            'score': round(isolation_score * 0.5, 3),  # Cap contribution
            'claims_with_citations': with_citations,
            'total_citations': citation_count,
            'citation_ratio': round(citation_ratio, 3),
            'interpretation': f"{with_citations}/{len(claims)} claims have citations"
        }
    
    def _calculate_evidence_avoidance(self, claims: List[Dict]) -> Dict:
        """Calculate evidence avoidance signal (META claims without citations)"""
        
        meta_claims = [c for c in claims if c.get('claim_type', '').upper() == 'META']
        
        if not meta_claims:
            return {
                'score': 0.0,
                'meta_without_evidence': 0,
                'meta_with_evidence': 0,
                'interpretation': 'No META claims to analyze'
            }
        
        without_evidence = 0
        with_evidence = 0
        
        for claim in meta_claims:
            citation_data = claim.get('citation_data', {})
            citations = citation_data.get('citations', []) if citation_data else []
            
            if citations:
                with_evidence += 1
            else:
                without_evidence += 1
        
        total = without_evidence + with_evidence
        avoidance_ratio = without_evidence / total if total > 0 else 0
        
        # High avoidance = dismissing without evidence
        score = avoidance_ratio
        
        return {
            'score': round(score, 3),
            'meta_without_evidence': without_evidence,
            'meta_with_evidence': with_evidence,
            'avoidance_ratio': round(avoidance_ratio, 3),
            'interpretation': self._interpret_evidence_avoidance(with_evidence, without_evidence, avoidance_ratio)
        }
    
    def _calculate_authority_mismatch(self, claims: List[Dict]) -> Dict:
        """Calculate authority mismatch signal (simplified)"""
        
        # This is a simplified version
        # Full implementation would use authority_domain_analyzer
        
        mismatch_keywords = [
            'actor', 'celebrity', 'politician', 'journalist', 'blogger',
            'influencer', 'commentator', 'pundit'
        ]
        
        mismatch_count = 0
        
        for claim in claims:
            claim_text = claim.get('claim_text', '').lower()
            for keyword in mismatch_keywords:
                if keyword in claim_text:
                    mismatch_count += 1
                    break
        
        if len(claims) == 0:
            return {
                'score': 0.0,
                'mismatch_count': 0,
                'interpretation': 'No claims to analyze',
                'note': 'Using simplified keyword detection'
            }
        
        mismatch_ratio = mismatch_count / len(claims)
        score = min(mismatch_ratio * 2, 0.8)  # Cap contribution
        
        return {
            'score': round(score, 3),
            'mismatch_count': mismatch_count,
            'mismatch_ratio': round(mismatch_ratio, 3),
            'interpretation': self._interpret_authority_mismatch(mismatch_count, mismatch_ratio),
            'note': 'Using simplified keyword detection'
        }
    
    def _calculate_confidence(self, signals: Dict, claim_count: int) -> float:
        """Calculate confidence in result"""
        
        # Factors:
        # 1. Number of claims (more = more confident)
        # 2. Number of signals with data
        # 3. Strength of narrative signal
        
        # Claim count factor
        min_claims = self.thresholds.get('min_claims_for_analysis', 5)
        claim_factor = min(claim_count / (min_claims * 4), 1.0)
        
        # Signal data factor
        signals_with_data = sum(1 for s in signals.values() if s.get('score', 0) > 0)
        signal_factor = signals_with_data / len(self.weights)
        
        # Narrative strength factor
        narrative_indicators = signals.get('suppression_narrative', {}).get('indicators_found', 0)
        narrative_factor = min(narrative_indicators / 5, 1.0)
        
        # Weighted combination
        confidence = (claim_factor * 0.3) + (signal_factor * 0.3) + (narrative_factor * 0.4)
        
        return confidence
    
    def _count_claim_types(self, claims: List[Dict]) -> Dict:
        """Count claims by type"""
        
        counts = {
            'total': len(claims),
            'primary': 0,
            'secondary': 0,
            'meta': 0,
            'contextual': 0
        }
        
        for claim in claims:
            claim_type = claim.get('claim_type', '').upper()
            if claim_type == 'PRIMARY':
                counts['primary'] += 1
            elif claim_type == 'SECONDARY':
                counts['secondary'] += 1
            elif claim_type == 'META':
                counts['meta'] += 1
            elif claim_type == 'CONTEXTUAL':
                counts['contextual'] += 1
        
        return counts
    
    def _interpret_score(self, score: float, level: str, signals: Dict) -> str:
        """Generate human-readable interpretation"""
        
        # Get narrative summary
        narrative = signals.get('suppression_narrative', {})
        narrative_interp = narrative.get('interpretation', '')
        
        # Get top contributing signals
        signal_contributions = []
        for name, weight in self.weights.items():
            sig_score = signals.get(name, {}).get('score', 0.0)
            contribution = sig_score * weight
            signal_contributions.append((name, contribution, sig_score))
        
        signal_contributions.sort(key=lambda x: x[1], reverse=True)
        top_signals = signal_contributions[:2]
        
        top_signal_names = [s[0].replace('_', ' ').title() for s in top_signals]
        
        interpretation = f"{level} suppression pattern (score: {score:.2f}). "
        interpretation += f"Top signals: {', '.join(top_signal_names)}. "
        interpretation += narrative_interp
        
        return interpretation
    
    # Interpretation helpers
    
    def _interpret_meta_density(self, ratio: float, meta: int, primary: int) -> str:
        if ratio > 0.5:
            return f"High META density ({ratio:.1%}): {meta} dismissals vs {primary} substantive claims"
        elif ratio > 0.3:
            return f"Moderate META density ({ratio:.1%})"
        else:
            return f"Low META density ({ratio:.1%})"
    
    def _interpret_evidence_avoidance(self, with_cites: int, without: int, ratio: float) -> str:
        if ratio > 0.7:
            return f"High evidence avoidance: {without} META claims without citations"
        elif ratio > 0.4:
            return f"Moderate evidence avoidance: {without} uncited dismissals"
        else:
            return f"Low evidence avoidance: Most META claims have citations"
    
    def _interpret_authority_mismatch(self, count: int, ratio: float) -> str:
        if ratio > 0.3:
            return f"Significant authority mismatch: {count} claims from non-experts"
        elif ratio > 0.1:
            return f"Some authority concerns: {count} potential mismatches"
        else:
            return f"Authority alignment appears normal"
    
    def _empty_result(self, topic: str = "unknown") -> Dict:
        """Return empty result when no data available"""
        return {
            'topic': topic,
            'suppression_score': 0.0,
            'level': 'MINIMAL',
            'confidence': 0.0,
            'signals': {},
            'interpretation': 'No data available for analysis',
            'claim_counts': {
                'total': 0,
                'primary': 0,
                'secondary': 0,
                'meta': 0,
                'contextual': 0
            },
            'claims_analyzed': 0,
            'profile_used': self.profile.get('metadata', {}).get('name', 'default')
        }
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")


# === Profile Management Functions ===

def get_profile_directory(base_dir: str = '.') -> str:
    """Get or create calibration profiles directory"""
    profile_dir = os.path.join(base_dir, 'calibration_profiles')
    os.makedirs(profile_dir, exist_ok=True)
    return profile_dir


def list_profiles(profile_dir: str) -> List[Dict]:
    """List all available calibration profiles"""
    profiles = []
    
    for filename in os.listdir(profile_dir):
        if filename.endswith('.json'):
            path = os.path.join(profile_dir, filename)
            try:
                with open(path, 'r') as f:
                    profile = json.load(f)
                profiles.append({
                    'filename': filename,
                    'name': profile.get('metadata', {}).get('name', filename),
                    'description': profile.get('metadata', {}).get('description', ''),
                    'tags': profile.get('metadata', {}).get('tags', [])
                })
            except:
                pass
    
    return profiles


def load_profile(profile_dir: str, filename: str) -> Dict:
    """Load a specific calibration profile"""
    path = os.path.join(profile_dir, filename)
    with open(path, 'r') as f:
        return json.load(f)


def save_profile(profile_dir: str, profile: Dict, filename: Optional[str] = None) -> str:
    """Save a calibration profile"""
    if filename is None:
        name = profile.get('metadata', {}).get('name', 'custom')
        filename = name.lower().replace(' ', '_').replace('-', '_') + '.json'
    
    path = os.path.join(profile_dir, filename)
    
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    return filename


# === Main / Test ===

def main():
    """Test the suppression detector with different profiles"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*80)
    print("AEGIS INSIGHT - SUPPRESSION DETECTOR v2")
    print("Testing Goldfinger Scoring with Narrative Detection")
    print("="*80)
    
    # Initialize detector with default profile
    detector = SuppressionDetector()
    
    # Test topics
    test_topics = [
        "Thomas Paine",
        "Smedley Butler",
        "Tesla"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {topic}")
        print('='*80)
        
        result = detector.detect_suppression(topic)
        
        print(f"\nSuppression Score: {result['suppression_score']:.3f} ({result['level']})")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Claims Analyzed: {result['claims_analyzed']}")
        print(f"Profile Used: {result['profile_used']}")
        
        print(f"\nInterpretation: {result['interpretation']}")
        
        print(f"\nClaim Counts:")
        for claim_type, count in result['claim_counts'].items():
            print(f"  {claim_type}: {count}")
        
        print(f"\nSignal Breakdown:")
        for signal_name, signal_data in result['signals'].items():
            score = signal_data.get('score', 0.0)
            weight = detector.weights.get(signal_name, 0.0)
            contribution = score * weight
            print(f"\n  {signal_name}:")
            print(f"    Score: {score:.3f} × {weight:.2f} weight = {contribution:.3f}")
            if 'indicators_found' in signal_data:
                print(f"    Indicators: {signal_data['indicators_found']}")
            if 'interpretation' in signal_data:
                print(f"    {signal_data['interpretation']}")
    
    detector.close()
    
    print(f"\n{'='*80}")
    print("DETECTION COMPLETE")
    print('='*80)


if __name__ == "__main__":
    main()
