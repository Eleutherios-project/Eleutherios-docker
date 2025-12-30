"""
Aegis Insight - Temporal Extractor
Extracts temporal references: absolute dates, relative periods, temporal markers
Handles BCE dates and historical periods
"""

import json
import logging
import requests
from typing import Dict, Optional

class TemporalExtractor:
    """
    Extract temporal data using specialized LLM prompts
    Supports absolute dates, relative periods, and temporal markers
    """
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "mistral-nemo:12b",
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
    
    def extract(self, claim_text: str, context: Dict) -> Dict:
        """
        Extract temporal data from claim text
        
        Args:
            claim_text: Claim text to extract from
            context: Dict with domain, title, etc.
            
        Returns:
            Dict with absolute_dates, relative_dates, temporal_markers
        """
        
        prompt = self._build_prompt(claim_text, context)
        
        try:
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse JSON
            temporal = self._parse_response(response)
            
            return temporal
            
        except Exception as e:
            self.logger.error(f"Temporal extraction failed: {e}")
            return {
                'absolute_dates': [],
                'relative_dates': [],
                'temporal_markers': []
            }
    
    def _build_prompt(self, claim_text: str, context: Dict) -> str:
        """Build specialized prompt for temporal extraction"""
        
        title = context.get('title', 'unknown')
        
        prompt = f"""Extract ALL temporal references from this text.

Text: {claim_text}

Context: {title}

Instructions:
- Find ALL dates in ANY format including:
  * Full dates: January 15, 1945 or 1945-01-15
  * Abbreviated: 7/10/45, 12-25-34, 1/5/1945
  * Month-year: July 1934, Dec. 1945
  * Year only: 1945, '45
  * Decades: 1930s, the thirties
  * Ranges: 1795-1822, 1930-1945
- Convert abbreviated years intelligently (45 = 1945, 34 = 1934 for 20th century context)
- Extract historical periods (Bronze Age, Medieval, etc.)
- Identify temporal markers (durations, sequences, relative times)
- Handle BCE dates properly
- Assign confidence based on clarity (higher for explicit dates)

Return ONLY valid JSON:
{{
  "absolute_dates": [
    {{
      "date": "YYYY-MM-DD or YYYY or 'YYYY BCE'",
      "confidence": 0.95,
      "type": "historical|modern|prehistoric",
      "context": "what this date refers to"
    }}
  ],
  "relative_dates": [
    {{
      "period": "Bronze Age|Ice Age|Medieval Period|etc.",
      "confidence": 0.85,
      "reference": "what this period refers to"
    }}
  ],
  "temporal_markers": [
    {{
      "text": "3000 years ago|during the war|etc.",
      "type": "duration|period|sequence",
      "confidence": 0.80
    }}
  ]
}}

If no temporal references, return:
{{
  "absolute_dates": [],
  "relative_dates": [],
  "temporal_markers": []
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
                    "num_predict": 1500
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
            temporal = json.loads(response)
            
            # Ensure all required keys
            if 'absolute_dates' not in temporal:
                temporal['absolute_dates'] = []
            if 'relative_dates' not in temporal:
                temporal['relative_dates'] = []
            if 'temporal_markers' not in temporal:
                temporal['temporal_markers'] = []
            
            return temporal
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            self.logger.debug(f"Response was: {response[:200]}")
            
            # Attempt recovery: find first valid JSON object
            try:
                start = response.find('{')
                if start >= 0:
                    depth = 0
                    for i, c in enumerate(response[start:], start):
                        if c == '{': depth += 1
                        elif c == '}': depth -= 1
                        if depth == 0:
                            temporal = json.loads(response[start:i+1])
                            temporal.setdefault('absolute_dates', [])
                            temporal.setdefault('relative_dates', [])
                            temporal.setdefault('temporal_markers', [])
                            self.logger.info("Recovered temporal data from malformed JSON")
                            return temporal
            except:
                pass
            
            return {
                'absolute_dates': [],
                'relative_dates': [],
                'temporal_markers': []
            }

if __name__ == "__main__":
    # Test temporal extractor
    logging.basicConfig(level=logging.INFO)
    
    extractor = TemporalExtractor()
    
    test_text = """
    The Bronze Age collapse occurred around 1200 BCE, devastating civilizations 
    across the Mediterranean. Recent evidence from 2023 suggests climate change 
    played a major role. The catastrophe lasted approximately 50 years.
    """
    
    context = {
        'domain': 'research_investigative',
        'title': 'Test Document'
    }
    
    temporal = extractor.extract(test_text, context)
    
    print(f"\nExtracted temporal data:")
    print(f"  Absolute dates: {len(temporal['absolute_dates'])}")
    for d in temporal['absolute_dates']:
        print(f"    - {d['date']} ({d['type']})")
    
    print(f"  Relative dates: {len(temporal['relative_dates'])}")
    for p in temporal['relative_dates']:
        print(f"    - {p['period']}")
    
    print(f"  Temporal markers: {len(temporal['temporal_markers'])}")
    for m in temporal['temporal_markers']:
        print(f"    - {m['text']} ({m['type']})")
