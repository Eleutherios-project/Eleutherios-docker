"""
Aegis Insight - Citation Extractor
Extracts citation relationships and builds attribution chains
Handles nested sources (matryoshka doll pattern)
Uses embeddings for matching to existing claims
"""

import json
import logging
import requests
from typing import Dict, List, Optional

class CitationExtractor:
    """
    Extract citations and build attribution chains
    Detects relationships: CITES, SUPPORTS, CONTRADICTS
    """
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:72b",
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
    
    def extract(self, claim_text: str, context: Dict) -> Dict:
        """
        Extract citations and attribution chains from claim text
        
        Args:
            claim_text: Claim text to analyze
            context: Dict with domain, title, etc.
            
        Returns:
            Dict with cites_other_work, attribution_chain, relationship_type
        """
        
        prompt = self._build_prompt(claim_text, context)
        
        try:
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse JSON
            citation_data = self._parse_response(response)
            
            return citation_data
            
        except Exception as e:
            self.logger.error(f"Citation extraction failed: {e}")
            return {
                'cites_other_work': False,
                'attribution_chain': [],
                'relationship_type': None,
                'confidence': 0.0
            }
    
    def _build_prompt(self, claim_text: str, context: Dict) -> str:
        """Build specialized prompt for citation extraction"""
        
        domain = context.get('domain', 'unknown')
        title = context.get('title', 'unknown')
        
        prompt = f"""Analyze this claim for references to other work or sources.

Claim: {claim_text}

Context:
- Domain: {domain}
- Document: {title}

Instructions:
- Does this claim cite, reference, or build upon other work?
- Extract the full attribution chain (nested sources)
- Determine relationship type (CITES, SUPPORTS, CONTRADICTS)
- Track matryoshka pattern: "According to X in Y's book about Z's research..."

Return ONLY valid JSON:
{{
  "cites_other_work": true|false,
  "attribution_chain": [
    {{
      "level": 1,
      "source_type": "book|paper|document|data|testimony|media",
      "source_name": "Name of source",
      "author": "if known",
      "confidence": 0.90,
      "quote_or_paraphrase": "relevant excerpt if quoted"
    }},
    {{
      "level": 2,
      "source_type": "document",
      "source_name": "Nested source (source within source)",
      "author": "if known",
      "confidence": 0.85
    }}
  ],
  "relationship_type": "CITES|SUPPORTS|CONTRADICTS|null",
  "confidence": 0.85
}}

Relationship types:
- CITES: Neutral reference to other work
- SUPPORTS: Claim supports/validates other work
- CONTRADICTS: Claim opposes/refutes other work

Track nested attribution carefully. Example:
According to Smith's analysis of Document ABC-123 from the archives...
Level 1: Wolf's book
Level 2: Document ABC-123
Level 3: FDA (regulatory source)

If no citations, return:
{{
  "cites_other_work": false,
  "attribution_chain": [],
  "relationship_type": null,
  "confidence": 0.0
}}
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
                    "num_predict": 2000
                }
            },
            timeout=90
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API returned {response.status_code}")
        
        result = response.json()
        return result.get('response', '')
    
    def _parse_response(self, response: str) -> Dict:
        """Parse JSON response"""
        
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
            citation_data = json.loads(response)
            
            # Ensure all required keys
            if 'cites_other_work' not in citation_data:
                citation_data['cites_other_work'] = False
            if 'attribution_chain' not in citation_data:
                citation_data['attribution_chain'] = []
            if 'relationship_type' not in citation_data:
                citation_data['relationship_type'] = None
            if 'confidence' not in citation_data:
                citation_data['confidence'] = 0.5
            
            # Validate attribution chain depth
            max_depth = 3  # Configurable
            if len(citation_data['attribution_chain']) > max_depth:
                self.logger.warning(f"Attribution chain too deep ({len(citation_data['attribution_chain'])}), truncating to {max_depth}")
                citation_data['attribution_chain'] = citation_data['attribution_chain'][:max_depth]
            
            return citation_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            self.logger.debug(f"Response was: {response[:200]}")
            
            return {
                'cites_other_work': False,
                'attribution_chain': [],
                'relationship_type': None,
                'confidence': 0.0
            }

if __name__ == "__main__":
    # Test citation extractor
    logging.basicConfig(level=logging.INFO)
    
    extractor = CitationExtractor()
    
    test_text = """
    According to a researcher's analysis in 'Historical Papers', Document ABC-123 
    submitted to the FDA showed 1,223 fatalities. This contradicts mainstream media 
    reports that claimed the treatments were completely safe.
    """
    
    context = {
        'domain': 'research_investigative',
        'title': 'Test Document'
    }
    
    citation = extractor.extract(test_text, context)
    
    print(f"\nExtracted citation data:")
    print(f"  Cites other work: {citation['cites_other_work']}")
    print(f"  Relationship type: {citation['relationship_type']}")
    print(f"  Confidence: {citation['confidence']:.2f}")
    print(f"  Attribution chain: {len(citation['attribution_chain'])} levels")
    
    for i, attr in enumerate(citation['attribution_chain'], 1):
        print(f"    Level {i}: {attr['source_name']} ({attr['source_type']})")
