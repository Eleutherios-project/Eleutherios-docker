#!/usr/bin/env python3
"""
Enhanced Claim Extractor for AegisTrustNet
Supports multiple extraction modes: regex, spacy, and LLM-based extraction via Ollama

Version: 2.0
Date: November 3, 2025
"""

import re
import json
import logging
import requests
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExtractionMode(Enum):
    """Available extraction modes"""
    NONE = "none"  # Legacy regex patterns (fast, low quality)
    SPACY = "spacy"  # SpaCy NLP (medium quality)
    LLM_SMALL = "llm_small"  # 7-13B models (high quality)
    LLM_LARGE = "llm_large"  # 70B+ models (maximum quality)


@dataclass
class ExtractionConfig:
    """Configuration for claim extraction"""
    mode: ExtractionMode = ExtractionMode.LLM_LARGE
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    confidence_threshold: float = 0.6
    max_claims_per_chunk: int = 20
    timeout_seconds: int = 300  # Increased from 120 to 300 (5 minutes) for 70B models
    progress_callback: Optional[Callable] = None


DEFAULT_SYSTEM_PROMPT = """You are an expert knowledge graph architect analyzing text to extract factual claims for epistemological network analysis.

Your task is to identify and extract claims in these categories:

1. PRIMARY: Original ideas, creations, developments, discoveries
   - Format: "X developed Y", "X wrote Y", "X proposed Y"
   - Example: "Boyd developed the OODA loop"

2. SECONDARY: Influence, adoption, application, reach
   - Format: "X influenced Y", "Y adopted X", "X applied to Y"
   - Example: "OODA loop influenced Marine Corps doctrine"

3. META: Commentary, criticism, reception, suppression
   - Format: "X dismissed Y", "Y criticized X", "X rejected by Y"
   - Example: "Academic strategists dismissed Boyd's work"
   - CRITICAL: These indicate suppression patterns

4. CONTEXTUAL: Background facts, biographical details
   - Format: "X served at Y", "X lived in Y", "X studied Y"
   - Example: "Boyd served at Nellis AFB"

For each claim, provide:
{
  "subject": "entity name",
  "relation": "verb/action",
  "object": "entity name",
  "claim_type": "PRIMARY|SECONDARY|META|CONTEXTUAL",
  "significance": 0.0-1.0,
  "full_text": "complete sentence",
  "confidence": 0.0-1.0
}

Pay special attention to:
- Attribution chains (X influenced Y who influenced Z)
- Suppression patterns (dismissed, rejected, criticized)
- Adoption patterns (who accepted vs who rejected)
- Network topology (institutional vs practitioner adoption)

Exclude:
- Questions
- Metadata (DOI, PMID, etc.)
- Table/Figure references
- Trivial biographical details

Output ONLY valid JSON. Format: {"claims": [...]}
No markdown, no code fences, no explanation."""


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []
    
    def recommend_model(self, mode: ExtractionMode) -> Optional[str]:
        """Recommend best available model for extraction mode"""
        models = self.list_models()
        if not models:
            return None
        
        model_names = [m['name'] for m in models]
        
        if mode == ExtractionMode.LLM_LARGE:
            # Prefer 70B+ models
            for name in model_names:
                if any(x in name.lower() for x in ['70b', '72b', 'qwen2.5:72', 'llama3.1:70']):
                    return name
        
        elif mode == ExtractionMode.LLM_SMALL:
            # Prefer 7-13B models
            for name in model_names:
                if any(x in name.lower() for x in ['7b', '13b', 'mistral', 'llama3:8']):
                    return name
        
        # Fallback to first available
        return model_names[0] if model_names else None
    
    def generate(self, 
                 prompt: str,
                 model: str,
                 system_prompt: str,
                 timeout: int = 120) -> Optional[str]:
        """Generate completion from Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 2000
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
        
        except requests.Timeout:
            logger.error(f"Ollama request timeout after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return None


class ClaimExtractor:
    """Extract factual claims using configurable methods"""
    
    CLAIM_PATTERNS = [
        r'(?:shows?|demonstrates?|proves?|indicates?|suggests?|reveals?) that .{20,200}[.!]',
        r'(?:according to|based on|research shows|studies indicate) .{20,200}[.!]',
        r'(?:found|discovered|observed|concluded) that .{20,200}[.!]',
        r'(?:is|are|was|were) (?:significantly|dramatically|substantially) .{20,150}[.!]',
        r'(?:can|could|may|might) (?:result in|lead to|cause) .{20,150}[.!]',
        r'(?:effective|ineffective) (?:in|for|at) .{20,150}[.!]',
        r'(?:associated with|correlated with|linked to) .{20,150}[.!]',
    ]
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize extractor with configuration"""
        self.config = config or ExtractionConfig()
        self.ollama = OllamaClient(self.config.ollama_url)
        self.nlp = None
        
        # Initialize SpaCy if needed
        if self.config.mode == ExtractionMode.SPACY:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_trf")
                logger.info("SpaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load SpaCy: {e}. Falling back to regex")
                self.config.mode = ExtractionMode.NONE
        
        # Auto-detect LLM model if needed
        if self.config.mode in [ExtractionMode.LLM_SMALL, ExtractionMode.LLM_LARGE]:
            if not self.ollama.is_available():
                logger.warning("Ollama not available, falling back to regex")
                self.config.mode = ExtractionMode.NONE
            elif not self.config.model_name:
                self.config.model_name = self.ollama.recommend_model(self.config.mode)
                if self.config.model_name:
                    logger.info(f"Auto-selected model: {self.config.model_name}")
                else:
                    logger.warning("No suitable model found, falling back to regex")
                    self.config.mode = ExtractionMode.NONE
        
        logger.info(f"ClaimExtractor initialized: mode={self.config.mode.value}")
    
    def extract_claims(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract claims using configured method"""
        
        if self.config.mode == ExtractionMode.NONE:
            return self._extract_regex(text, metadata)
        elif self.config.mode == ExtractionMode.SPACY:
            return self._extract_spacy(text, metadata)
        elif self.config.mode in [ExtractionMode.LLM_SMALL, ExtractionMode.LLM_LARGE]:
            return self._extract_llm(text, metadata)
        
        return []
    
    def _extract_regex(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy regex-based extraction"""
        claims = []
        sentences = re.split(r'[.!?]+', text)
        
        for pattern in self.CLAIM_PATTERNS:
            for sentence in sentences:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    claim_text = match.group().strip()
                    if len(claim_text) >= 30:
                        claims.append({
                            'text': claim_text,
                            'confidence': 0.5,
                            'domain': metadata.get('domain', 'unknown'),
                            'source_file': metadata.get('source_file', ''),
                            'extraction_method': 'regex',
                            'claim_type': 'CONTEXTUAL'
                        })
        
        return claims[:self.config.max_claims_per_chunk]
    
    def _extract_spacy(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """SpaCy-based extraction"""
        if not self.nlp:
            return self._extract_regex(text, metadata)
        
        claims = []
        
        try:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                root = sent.root
                if root.pos_ != 'VERB':
                    continue
                
                subject = None
                obj = None
                
                for child in root.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        subject = self._get_phrase(child)
                    if child.dep_ in ['dobj', 'attr', 'prep']:
                        obj = self._get_phrase(child)
                
                if subject and obj:
                    claims.append({
                        'text': sent.text.strip(),
                        'subject': subject,
                        'relation': root.lemma_,
                        'object': obj,
                        'confidence': 0.7,
                        'domain': metadata.get('domain', 'unknown'),
                        'source_file': metadata.get('source_file', ''),
                        'extraction_method': 'spacy',
                        'claim_type': 'CONTEXTUAL'
                    })
        
        except Exception as e:
            logger.error(f"SpaCy extraction error: {e}")
            return self._extract_regex(text, metadata)
        
        return claims[:self.config.max_claims_per_chunk]
    
    def _get_phrase(self, token) -> str:
        """Get full phrase from dependency tree"""
        return ' '.join([child.text for child in token.subtree])
    
    def _extract_llm(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM-based extraction via Ollama with retry logic"""
        
        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        
        # Add domain-specific guidance
        domain = metadata.get('domain', 'unknown')
        domain_guidance = self._get_domain_guidance(domain)
        
        user_prompt = f"""{domain_guidance}

TEXT TO ANALYZE:
{text[:4000]}

Extract all claims as JSON."""
        
        # Retry logic - try up to 3 times with increasing timeout
        max_retries = 3
        base_timeout = self.config.timeout_seconds
        
        for attempt in range(max_retries):
            # Increase timeout on each retry (2x, 3x, 4x)
            current_timeout = base_timeout * (attempt + 2)
            
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} with timeout {current_timeout}s")
            
            # Call Ollama
            response_text = self.ollama.generate(
                prompt=user_prompt,
                model=self.config.model_name,
                system_prompt=system_prompt,
                timeout=current_timeout
            )
            
            if response_text:
                break  # Success!
            
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
        
        if not response_text:
            # CRITICAL: Do NOT fall back to regex in LLM mode
            logger.error(f"Ollama failed after {max_retries} attempts - SKIPPING chunk to maintain quality")
            logger.error("Consider increasing --timeout or checking Ollama/GPU status")
            return []  # Return empty rather than low-quality regex results
        
        # Parse JSON response
        claims = self._parse_llm_response(response_text, metadata)
        
        # Filter by confidence
        claims = [c for c in claims if c.get('confidence', 0) >= self.config.confidence_threshold]
        
        return claims[:self.config.max_claims_per_chunk]
    
    def _get_domain_guidance(self, domain: str) -> str:
        """Get domain-specific extraction guidance"""
        guidance = {
            'strategic_analytical': """Focus on:
- Strategic concepts and their origins
- Influence on doctrine and practice
- Institutional vs practitioner adoption
- Criticism and reception patterns""",
            
            'research_investigative': """Focus on:
- Evidence claims and their sources
- Mainstream vs alternative interpretations
- Dismissal and suppression patterns
- Independent corroboration""",
            
            'academic_scholarly': """Focus on:
- Research findings and methodologies
- Citation relationships
- Theoretical developments
- Peer review and validation""",
            
            'contemplative_spiritual': """Focus on:
- Philosophical claims and traditions
- Consciousness and awareness concepts
- Cross-tradition connections
- Modern scientific parallels""",
            
            'practical_wisdom': """Focus on:
- Actionable principles
- Real-world applications
- Evidence-based insights
- Decision-making frameworks""",
            
            'technical_implementation': """Focus on:
- Technical specifications
- Implementation patterns
- Best practices
- Standards and protocols"""
        }
        
        return guidance.get(domain, "Extract all significant factual claims.")
    
    def _parse_llm_response(self, response_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse and validate LLM JSON response with robust error handling"""
        
        # Clean response - remove markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        
        # Remove any leading/trailing whitespace and BOM
        response_text = response_text.strip().lstrip('\ufeff')
        
        # Try to extract JSON if it's embedded in text
        if not response_text.startswith('{') and not response_text.startswith('['):
            # Try to find JSON in the response
            import re
            json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
        # Attempt to fix common JSON errors
        response_text = self._repair_json(response_text)
        
        try:
            data = json.loads(response_text)
            
            if isinstance(data, dict) and 'claims' in data:
                claims_list = data['claims']
            elif isinstance(data, list):
                claims_list = data
            else:
                logger.error(f"Unexpected JSON structure: {type(data)}")
                return []
            
            # Validate and enhance each claim
            validated_claims = []
            for claim in claims_list:
                if not isinstance(claim, dict):
                    continue
                
                # Required fields with fallbacks
                full_text = claim.get('full_text') or claim.get('text') or claim.get('claim_text')
                if not full_text:
                    continue  # Skip claims without text
                
                confidence = claim.get('confidence', 0.7)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.7
                
                # Add metadata
                claim['domain'] = metadata.get('domain', 'unknown')
                claim['source_file'] = metadata.get('source_file', '')
                claim['extraction_method'] = 'llm'
                
                # Normalize claim_type
                claim_type = str(claim.get('claim_type', 'CONTEXTUAL')).upper()
                if claim_type not in ['PRIMARY', 'SECONDARY', 'META', 'CONTEXTUAL']:
                    claim_type = 'CONTEXTUAL'
                claim['claim_type'] = claim_type
                
                # Ensure text field exists
                claim['text'] = full_text
                claim['confidence'] = confidence
                
                # Clean up any string fields with encoding issues
                for key in ['text', 'subject', 'relation', 'object']:
                    if key in claim and isinstance(claim[key], str):
                        claim[key] = claim[key].encode('utf-8', errors='ignore').decode('utf-8')
                
                validated_claims.append(claim)
            
            if not validated_claims:
                logger.warning("LLM returned valid JSON but no valid claims extracted")
            
            return validated_claims
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Problematic JSON (first 500 chars): {response_text[:500]}")
            
            # Try more aggressive repair
            repaired = self._aggressive_json_repair(response_text)
            if repaired:
                try:
                    data = json.loads(repaired)
                    logger.info("Successfully recovered claims after aggressive repair")
                    return self._parse_llm_response(repaired, metadata)
                except:
                    pass
            
            return []  # Return empty rather than crash
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _repair_json(self, text: str) -> str:
        """Apply common JSON repairs"""
        # Fix unescaped quotes in strings (common issue)
        # This is a simplified version - more sophisticated repair could be added
        
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Ensure proper string termination for common patterns
        # Fix "text": "some text that got cut off
        text = re.sub(r'"(\w+)":\s*"([^"]*?)$', r'"\1": "\2"', text, flags=re.MULTILINE)
        
        return text
    
    def _aggressive_json_repair(self, text: str) -> Optional[str]:
        """Attempt aggressive JSON repair for badly malformed responses"""
        try:
            # Try to extract just the claims array
            import re
            
            # Look for claims array pattern
            claims_match = re.search(r'"claims"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if claims_match:
                claims_text = claims_match.group(1)
                
                # Try to extract individual claim objects
                # Look for patterns like {"subject": "...", ...}
                claim_objects = re.findall(r'\{[^}]*?"full_text"[^}]*?\}', claims_text)
                
                if claim_objects:
                    # Reconstruct claims array
                    repaired = '{"claims": [' + ','.join(claim_objects) + ']}'
                    # Validate it's proper JSON
                    json.loads(repaired)
                    return repaired
            
            return None
        except:
            return None


# Convenience function for quick extraction
def extract_claims(text: str, 
                   metadata: Dict[str, Any],
                   mode: str = "llm_large",
                   model: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Quick claim extraction with sensible defaults
    
    Args:
        text: Text to extract claims from
        metadata: Document metadata (domain, source_file, etc.)
        mode: 'none', 'spacy', 'llm_small', or 'llm_large'
        model: Specific Ollama model name (optional)
    
    Returns:
        List of extracted claims
    """
    config = ExtractionConfig(
        mode=ExtractionMode(mode),
        model_name=model
    )
    extractor = ClaimExtractor(config)
    return extractor.extract_claims(text, metadata)
