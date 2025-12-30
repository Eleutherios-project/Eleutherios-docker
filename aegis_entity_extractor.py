"""
Aegis Insight - Entity Extractor
Extracts entities (People, Organizations, Concepts, Locations, Events) using Qwen 70B
"""

import json
import logging
import requests
from typing import Dict, List, Optional

class EntityExtractor:
    """
    Extract entities using specialized LLM prompts
    Focus on high-quality extraction with confidence scores
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:72b",
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        # Test connection
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.debug(f"✓ Connected to Ollama at {ollama_url}")
            else:
                self.logger.warning(f"Ollama connection issue: {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama: {e}")
    
    def extract(self, chunk_text: str, context: Dict) -> List[Dict]:
        """
        Extract entities from chunk text
        
        Args:
            chunk_text: Text to extract from
            context: Dict with domain, title, etc.
            
        Returns:
            List of entity dicts with name, type, confidence, aliases
        """
        
        prompt = self._build_prompt(chunk_text, context)
        
        try:
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse JSON
            entities = self._parse_response(response)
            
            # Filter by confidence
            min_confidence = context.get('min_entity_confidence', 0.5)
            entities = [e for e in entities if e.get('confidence', 0) >= min_confidence]
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _build_prompt(self, chunk_text: str, context: Dict) -> str:
        """Build specialized prompt for entity extraction"""
        
        domain = context.get('domain', 'unknown')
        title = context.get('title', 'unknown')
        
        prompt = f"""Extract ALL entities from this text. Be thorough and precise.

Text: {chunk_text}

Context:
- Domain: {domain}
- Document: {title}

Entity Types:
1. Person - Individual human beings (authors, researchers, historical figures)
2. Organization - Companies, institutions, agencies, universities
3. Concept - Ideas, theories, frameworks, methodologies (multi-word, capitalized)
4. Location - Places, geographic features, countries, cities
5. Event - Named historical events, conferences, incidents

Instructions:
- Extract every entity you find
- Use canonical forms (e.g., "United States" not "U.S.")
- Include confidence score (0.0-1.0) based on clarity
- List alternate names in aliases if known
- For concepts, only include multi-word capitalized terms (e.g., "Machine Learning", "Bronze Age")
- For people, use full names when available

Return ONLY valid JSON array (no markdown, no explanation):
[
  {{
    "name": "canonical name",
    "type": "Person|Organization|Concept|Location|Event",
    "confidence": 0.95,
    "aliases": ["alternate name 1", "alternate name 2"]
  }}
]

If no entities found, return: []
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
                    "temperature": 0.1,  # Low for consistency
                    "top_p": 0.9,
                    "num_predict": 2000
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
        
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        # Try parsing
        try:
            entities = json.loads(response)
            
            # Ensure it's a list
            if not isinstance(entities, list):
                self.logger.warning(f"Response is not a list: {type(entities)}")
                return []
            
            # Validate structure
            valid_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                    # Ensure required fields
                    if 'confidence' not in entity:
                        entity['confidence'] = 0.8  # Default
                    if 'aliases' not in entity:
                        entity['aliases'] = []
                    
                    valid_entities.append(entity)
            
            return valid_entities
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            self.logger.debug(f"Response was: {response[:200]}")
            
            # Attempt to recover partial JSON
            recovered = self._attempt_json_recovery(response)
            if recovered:
                self.logger.info(f"Recovered {len(recovered)} entities from malformed JSON")
                return recovered
            
            return []
    
    def _attempt_json_recovery(self, response: str) -> List[Dict]:
        """Try to recover entities from malformed JSON"""
        
        entities = []
        
        import re
        
        # Strategy 1: Multiple consecutive JSON arrays (most common LLM error)
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
                # Validate entities
                for e in all_arrays:
                    if isinstance(e, dict) and 'name' in e:
                        e.setdefault('type', 'UNKNOWN')
                        e.setdefault('confidence', 0.7)
                        e.setdefault('aliases', [])
                        entities.append(e)
                if entities:
                    return entities
        except:
            pass
        
        # Strategy 2: Regex pattern matching (fallback)
        pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"[^}]*\}'
        matches = re.finditer(pattern, response)
        
        for match in matches:
            name = match.group(1)
            entity_type = match.group(2)
            
            entities.append({
                'name': name,
                'type': entity_type,
                'confidence': 0.6,  # Lower confidence for recovered
                'aliases': []
            })
        
        return entities

if __name__ == "__main__":
    # Test entity extractor
    logging.basicConfig(level=logging.INFO)
    
    extractor = EntityExtractor()
    
    test_text = """
    Dr. Anthony Fauci, Director of NIAID, stated that the CDC recommends 
    treatment protocols. However, alternative researchers like Dr. Smith 
    have raised concerns about myocarditis risks.
    """
    
    context = {
        'domain': 'research_investigative',
        'title': 'Test Document',
        'min_entity_confidence': 0.5
    }
    
    entities = extractor.extract(test_text, context)
    
    print(f"\nExtracted {len(entities)} entities:")
    for e in entities:
        print(f"  - {e['name']} ({e['type']}) [confidence: {e['confidence']:.2f}]")
