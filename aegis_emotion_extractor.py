#!/usr/bin/env python3
"""
Aegis Insight - Emotional Content Extractor

Extracts emotional manipulation signals from claims to support
detection of psyops and coordinated influence campaigns.

Based on Chase Hughes' FATE/FEAR frameworks:
- Emotional appeals vs factual content
- Fear/anger/outrage triggering
- Sentiment manipulation patterns

Author: Aegis Insight Team
Date: November 2025
"""

import json
import logging
from typing import Dict, List, Optional
import requests


class AegisEmotionExtractor:
    """
    Extracts emotional content and manipulation signals from claim text
    
    Detects:
    - Overall sentiment (fear/anger/hope/neutral)
    - Emotional intensity (0.0-1.0)
    - Fact-to-emotion ratio (higher = more factual)
    - Specific emotional triggers (fear keywords, urgency language)
    - Manipulation indicators (appeals to emotion vs evidence)
    """
    
    # Emotional trigger keywords (simplified - LLM will do deeper analysis)
    FEAR_KEYWORDS = [
        'crisis', 'disaster', 'emergency', 'unprecedented', 'catastrophic',
        'collapse', 'destroy', 'threaten', 'danger', 'risk', 'fatal',
        'deadly', 'terrifying', 'shocking', 'horrific', 'panic'
    ]
    
    ANGER_KEYWORDS = [
        'outrage', 'outrageous', 'unacceptable', 'corrupt', 'fraud',
        'scandal', 'betrayal', 'lie', 'liar', 'attack', 'assault',
        'abuse', 'violation', 'criminal', 'traitor'
    ]
    
    URGENCY_KEYWORDS = [
        'now', 'immediately', 'urgent', 'act now', 'don\'t wait',
        'limited time', 'before it\'s too late', 'right now',
        'critical', 'must', 'need to'
    ]
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "mistral-nemo:12b",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize emotion extractor
        
        Args:
            ollama_url: Ollama API endpoint
            model: Model to use for extraction
            logger: Optional logger instance
        """
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Emotion extractor initialized with model: {model}")
    
    def extract(self, 
                claim_text: str,
                context: Optional[Dict] = None) -> Dict:
        """
        Extract emotional content from claim text
        
        Args:
            claim_text: The claim text to analyze
            context: Optional context (not currently used)
            
        Returns:
            Dict with emotional analysis:
            {
                "primary_sentiment": "fear|anger|hope|neutral",
                "emotional_intensity": 0.0-1.0,
                "fact_to_emotion_ratio": 0.0-1.0,
                "emotional_triggers": ["keyword1", "keyword2"],
                "manipulation_indicators": {
                    "appeals_to_fear": true/false,
                    "appeals_to_anger": true/false,
                    "urgency_without_evidence": true/false,
                    "loaded_language": true/false
                },
                "extraction_confidence": 0.0-1.0
            }
        """
        
        if not claim_text or len(claim_text.strip()) < 10:
            return self._empty_result()
        
        try:
            # Quick keyword scan first
            keywords_found = self._scan_keywords(claim_text.lower())
            
            # LLM-based emotional analysis
            prompt = self._build_prompt(claim_text)
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent analysis
                        "num_predict": 400
                    }
                },
                timeout=180
            )
            
            if response.status_code != 200:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return self._empty_result()
            
            result_text = response.json().get('response', '').strip()
            
            # Parse JSON response
            result = self._parse_response(result_text, keywords_found)
            
            return result
            
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout extracting emotion from claim")
            return self._empty_result()
        except Exception as e:
            self.logger.error(f"Error extracting emotion: {e}")
            return self._empty_result()
    
    def _scan_keywords(self, text_lower: str) -> Dict[str, List[str]]:
        """Quick keyword scan for emotional triggers"""
        
        found_fear = [kw for kw in self.FEAR_KEYWORDS if kw in text_lower]
        found_anger = [kw for kw in self.ANGER_KEYWORDS if kw in text_lower]
        found_urgency = [kw for kw in self.URGENCY_KEYWORDS if kw in text_lower]
        
        return {
            'fear': found_fear,
            'anger': found_anger,
            'urgency': found_urgency
        }
    
    def _build_prompt(self, claim_text: str) -> str:
        """Build prompt for emotional analysis"""
        
        return f"""Analyze the EMOTIONAL CONTENT of this claim. Identify manipulation tactics.

CLAIM TEXT:
{claim_text}

Analyze:
1. PRIMARY SENTIMENT: Is this primarily fear-based, anger-based, hope-based, or neutral/factual?
2. EMOTIONAL INTENSITY: How strong is the emotional appeal? (0.0 = purely factual, 1.0 = extreme emotion)
3. FACT-TO-EMOTION RATIO: What % is factual evidence vs emotional appeal? (0.0 = all emotion, 1.0 = all facts)
4. MANIPULATION INDICATORS:
   - Does it appeal to fear without clear evidence?
   - Does it appeal to anger or outrage?
   - Does it create urgency without justification?
   - Does it use loaded/inflammatory language?

Return ONLY valid JSON (no markdown, no explanation):
{{
  "primary_sentiment": "fear" | "anger" | "hope" | "neutral",
  "emotional_intensity": 0.75,
  "fact_to_emotion_ratio": 0.3,
  "manipulation_indicators": {{
    "appeals_to_fear": true,
    "appeals_to_anger": false,
    "urgency_without_evidence": true,
    "loaded_language": true
  }},
  "reasoning": "Brief 1-sentence explanation"
}}"""
    
    def _parse_response(self, 
                       response_text: str, 
                       keywords: Dict[str, List[str]]) -> Dict:
        """Parse LLM response and combine with keyword analysis"""
        
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            
            # Validate and normalize
            result = {
                "primary_sentiment": parsed.get('primary_sentiment', 'neutral'),
                "emotional_intensity": float(parsed.get('emotional_intensity', 0.0)),
                "fact_to_emotion_ratio": float(parsed.get('fact_to_emotion_ratio', 0.5)),
                "emotional_triggers": keywords['fear'] + keywords['anger'] + keywords['urgency'],
                "manipulation_indicators": parsed.get('manipulation_indicators', {}),
                "extraction_confidence": 0.85  # High confidence for successful parse
            }
            
            # Add trigger counts
            result['trigger_counts'] = {
                'fear_keywords': len(keywords['fear']),
                'anger_keywords': len(keywords['anger']),
                'urgency_keywords': len(keywords['urgency'])
            }
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse emotion JSON: {e}")
            
            # Fallback: Use keyword analysis only
            return self._fallback_analysis(keywords)
    
    def _fallback_analysis(self, keywords: Dict[str, List[str]]) -> Dict:
        """Fallback analysis using only keywords if LLM parse fails"""
        
        # Determine primary sentiment from keyword counts
        fear_count = len(keywords['fear'])
        anger_count = len(keywords['anger'])
        
        if fear_count > anger_count and fear_count > 0:
            primary = 'fear'
        elif anger_count > 0:
            primary = 'anger'
        else:
            primary = 'neutral'
        
        # Estimate intensity from keyword density
        total_triggers = fear_count + anger_count + len(keywords['urgency'])
        intensity = min(1.0, total_triggers * 0.2)  # Cap at 1.0
        
        return {
            "primary_sentiment": primary,
            "emotional_intensity": intensity,
            "fact_to_emotion_ratio": 0.5,  # Unknown, assume balanced
            "emotional_triggers": keywords['fear'] + keywords['anger'] + keywords['urgency'],
            "manipulation_indicators": {
                "appeals_to_fear": fear_count > 0,
                "appeals_to_anger": anger_count > 0,
                "urgency_without_evidence": len(keywords['urgency']) > 0,
                "loaded_language": total_triggers > 2
            },
            "trigger_counts": {
                'fear_keywords': fear_count,
                'anger_keywords': anger_count,
                'urgency_keywords': len(keywords['urgency'])
            },
            "extraction_confidence": 0.4  # Lower confidence for fallback
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "primary_sentiment": "neutral",
            "emotional_intensity": 0.0,
            "fact_to_emotion_ratio": 0.5,
            "emotional_triggers": [],
            "manipulation_indicators": {
                "appeals_to_fear": False,
                "appeals_to_anger": False,
                "urgency_without_evidence": False,
                "loaded_language": False
            },
            "trigger_counts": {
                'fear_keywords': 0,
                'anger_keywords': 0,
                'urgency_keywords': 0
            },
            "extraction_confidence": 0.0
        }


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = AegisEmotionExtractor()
    
    # Test cases
    test_claims = [
        # High emotion, fear-based
        "UNPRECEDENTED CRISIS: Scientists warn of CATASTROPHIC collapse unless we act NOW!",
        
        # High emotion, anger-based  
        "OUTRAGEOUS fraud exposed! Corrupt officials caught in shocking scandal!",
        
        # Factual, neutral
        "The study involved 500 participants over 12 months, measuring vitamin D levels.",
        
        # Mixed: some facts, some emotion
        "Research shows concerning trends in youth mental health, though experts debate the causes."
    ]
    
    print("Testing Emotion Extractor...\n")
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {claim[:60]}...")
        print(f"{'='*70}")
        
        result = extractor.extract(claim)
        
        print(f"Sentiment: {result['primary_sentiment']}")
        print(f"Intensity: {result['emotional_intensity']:.2f}")
        print(f"Fact/Emotion: {result['fact_to_emotion_ratio']:.2f}")
        print(f"Triggers found: {len(result['emotional_triggers'])}")
        print(f"Manipulation indicators: {result['manipulation_indicators']}")
        print(f"Confidence: {result['extraction_confidence']:.2f}")
