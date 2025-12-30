"""
Aegis Insight - Claim Extractor
Extracts and classifies claims with nested attribution chains
Handles PRIMARY, SECONDARY, META, and CONTEXTUAL claim types
"""

import json
import logging
import requests
from typing import Dict, List, Optional
import uuid

class ClaimExtractor:
    """
    Extract and classify claims using specialized LLM prompts
    Supports nested attribution (matryoshka pattern)
    """
    
    CLAIM_TYPES = {
        "PRIMARY": "Original ideas, foundational theories, first-order claims",
        "SECONDARY": "Derived claims, applications, interpretations of primary claims",
        "META": "Claims about other claims, institutional responses, dismissals",
        "CONTEXTUAL": "Background facts, supporting information, historical context"
    }
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:72b",
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
    
    def extract(self, chunk_text: str, context: Dict) -> List[Dict]:
        """
        Extract typed claims from chunk text
        
        Args:
            chunk_text: Text to extract from
            context: Dict with domain, title, chunk_id, etc.
            
        Returns:
            List of claim dicts with claim_text, claim_type, confidence, attribution
        """
        
        prompt = self._build_prompt(chunk_text, context)
        
        try:
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse JSON
            claims = self._parse_response(response)
            
            # Filter by quality
            min_confidence = context.get('min_claim_confidence', 0.5)
            min_length = context.get('min_claim_length', 20)
            
            valid_claims = []
            for claim in claims:
                claim_text = claim.get('claim_text', '')
                confidence = claim.get('confidence', 0)
                
                if len(claim_text) >= min_length and confidence >= min_confidence:
                    # Add unique ID
                    claim['claim_id'] = self._generate_claim_id(context.get('chunk_id'), claim_text)
                    valid_claims.append(claim)
            
            return valid_claims
            
        except Exception as e:
            self.logger.error(f"Claim extraction failed: {e}")
            return []
    
    def _generate_claim_id(self, chunk_id: str, claim_text: str) -> str:
        """Generate unique claim ID"""
        # Use hash of chunk_id + claim_text for reproducibility
        import hashlib
        combined = f"{chunk_id}_{claim_text[:100]}"
        hash_val = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"claim_{hash_val}"
    
    def _build_prompt(self, chunk_text: str, context: Dict) -> str:
        """Build specialized prompt for claim extraction"""
        
        domain = context.get('domain', 'unknown')
        title = context.get('title', 'unknown')
        
        prompt = f"""Extract ALL factual claims from this text and classify them by type.

Text: {chunk_text}

Context:
- Domain: {domain}
- Document: {title}

Claim Types:
1. PRIMARY - Original ideas, foundational theories, first discoveries
   Example: "Boyd developed the OODA loop decision-making framework"

2. SECONDARY - Derived claims, applications, interpretations
   Example: "OODA loop has been applied to business strategy"

3. META - Claims ABOUT other claims, institutional responses, dismissals
   Example: "Mainstream academics initially dismissed Boyd's work"
   Example: "Critics labeled the theory as pseudoscience"

4. CONTEXTUAL - Background facts, supporting information, dates
   Example: "Boyd was a fighter pilot in the Korean War"

Instructions:
- Extract every claim (factual statement) you find
- Each claim should be ONE complete sentence
- Classify each claim by type (PRIMARY/SECONDARY/META/CONTEXTUAL)
- Assign confidence based on clarity and explicitness (0.0-1.0)
- Extract attribution chain for nested sources (e.g., "According to X in Y's book...")
- Include supporting evidence if mentioned

Return ONLY valid JSON array:
[
  {{
    "claim_text": "complete factual statement",
    "claim_type": "PRIMARY|SECONDARY|META|CONTEXTUAL",
    "confidence": 0.85,
    "supporting_evidence": ["evidence if mentioned"],
    "attribution_chain": [
      {{
        "level": 1,
        "source_type": "book|paper|document|data|testimony",
        "source_name": "if mentioned",
        "author": "if mentioned",
        "confidence": 0.9
      }}
    ]
  }}
]

If no claims found, return: []
"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 3000  # Claims can be longer
                }
            },
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API returned {response.status_code}")
        
        result = response.json()
        return result.get('response', '')
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse JSON response, handling common issues"""
        
        # Clean response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        try:
            claims = json.loads(response)
            
            if not isinstance(claims, list):
                self.logger.warning(f"Response is not a list: {type(claims)}")
                return []
            
            # Validate and normalize
            valid_claims = []
            for claim in claims:
                if isinstance(claim, dict) and 'claim_text' in claim:
                    # Ensure required fields
                    if 'claim_type' not in claim:
                        claim['claim_type'] = 'CONTEXTUAL'  # Default
                    if 'confidence' not in claim:
                        claim['confidence'] = 0.7
                    if 'supporting_evidence' not in claim:
                        claim['supporting_evidence'] = []
                    if 'attribution_chain' not in claim:
                        claim['attribution_chain'] = []
                    
                    # Validate claim type
                    if claim['claim_type'] not in self.CLAIM_TYPES:
                        claim['claim_type'] = 'CONTEXTUAL'
                    
                    valid_claims.append(claim)
            
            return valid_claims
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            self.logger.debug(f"Response was: {response[:200]}")
            
            # Attempt recovery
            recovered = self._attempt_json_recovery(response)
            if recovered:
                self.logger.info(f"Recovered {len(recovered)} claims from malformed JSON")
                return recovered
            
            return []
    
    def _attempt_json_recovery(self, response: str) -> List[Dict]:
        """Try to recover claims from malformed JSON"""
        
        claims = []
        
        import re
        
        # Strategy 1: Multiple consecutive JSON arrays (most common LLM error)
        # Find ALL [ ] array pairs and merge them
        try:
            all_arrays = []
            pos = 0
            while pos < len(response):
                start = response.find('[', pos)
                if start < 0:
                    break
                depth = 0
                end = start
                for i, c in enumerate(response[start:], start):
                    if c == '[': depth += 1
                    elif c == ']': depth -= 1
                    if depth == 0:
                        end = i
                        break
                if end > start:
                    try:
                        parsed = json.loads(response[start:end+1])
                        if isinstance(parsed, list):
                            all_arrays.extend(parsed)
                    except:
                        pass
                    pos = end + 1
                else:
                    pos = start + 1
            
            if all_arrays:
                return self._validate_claims(all_arrays)
        except:
            pass
        
        # Strategy 2: JSON Lines (one object per line)
        try:
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') or line.startswith('['):
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, list):
                            claims.extend(parsed)
                        elif isinstance(parsed, dict) and 'claim_text' in parsed:
                            claims.append(parsed)
                    except:
                        pass
            
            if claims:
                return self._validate_claims(claims)
        except:
            pass
        
        # Strategy 3: Regex extraction (last resort)
        pattern = r'"claim_text"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, response)
        
        for claim_text in matches:
            if len(claim_text) >= 20:  # Min length
                claims.append({
                    'claim_text': claim_text,
                    'claim_type': 'CONTEXTUAL',
                    'confidence': 0.5,  # Lower confidence for recovered
                    'supporting_evidence': [],
                    'attribution_chain': []
                })
        
        return self._validate_claims(claims) if claims else []

    def _validate_claims(self, claims: List[Dict]) -> List[Dict]:
        """Validate and normalize recovered claims"""
        valid = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            if 'claim_text' not in claim or len(claim.get('claim_text', '')) < 10:
                continue
            # Normalize
            claim.setdefault('claim_type', 'CONTEXTUAL')
            claim.setdefault('confidence', 0.7)
            claim.setdefault('supporting_evidence', [])
            claim.setdefault('attribution_chain', [])
            if claim['claim_type'] not in self.CLAIM_TYPES:
                claim['claim_type'] = 'CONTEXTUAL'
            valid.append(claim)
        return valid

if __name__ == "__main__":
    # Test claim extractor
    logging.basicConfig(level=logging.INFO)
    
    extractor = ClaimExtractor()
    
    test_text = """
    According to Document ABC-123, submitted to authorities, the clinical trial 
    data showed 1,223 fatalities within the first 90 days. The document, analyzed 
    in 'Historical Papers' by a researcher, reveals previously undisclosed adverse 
    event frequencies. Mainstream media outlets dismissed these findings as misinformation.
    """
    
    context = {
        'domain': 'research_investigative',
        'title': 'Test Document',
        'chunk_id': 'test_001',
        'min_claim_confidence': 0.5,
        'min_claim_length': 20
    }
    
    claims = extractor.extract(test_text, context)
    
    print(f"\nExtracted {len(claims)} claims:")
    for c in claims:
        print(f"\n  Type: {c['claim_type']}")
        print(f"  Claim: {c['claim_text'][:100]}...")
        print(f"  Confidence: {c['confidence']:.2f}")
        if c['attribution_chain']:
            print(f"  Attribution: {len(c['attribution_chain'])} levels")
