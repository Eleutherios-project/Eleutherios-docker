"""
Aegis Insight - Geographic Extractor
Extracts locations with coordinates and cultural context
Uses LLM for extraction + geopy for geocoding
"""

import json
import logging
import requests
from typing import Dict, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

class GeographicExtractor:
    """
    Extract geographic references and cultural context
    Includes geocoding for coordinates
    """
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "mistral-nemo:12b",
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="aegis-insight-v1")
        self.geocode_cache = {}  # Simple cache to avoid repeated lookups
    
    def extract(self, claim_text: str, context: Dict) -> Dict:
        """
        Extract geographic and cultural data from claim text
        
        Args:
            claim_text: Claim text to extract from
            context: Dict with domain, title, etc.
            
        Returns:
            Dict with locations and cultural_context
        """
        
        prompt = self._build_prompt(claim_text, context)
        
        try:
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse JSON
            geo_data = self._parse_response(response)
            
            # Geocode locations
            for loc in geo_data.get('locations', []):
                coords = self._geocode_location(loc['name'])
                if coords:
                    loc['coordinates'] = coords
            
            return geo_data
            
        except Exception as e:
            self.logger.error(f"Geographic extraction failed: {e}")
            return {
                'locations': [],
                'cultural_context': []
            }
    
    def _build_prompt(self, claim_text: str, context: Dict) -> str:
        """Build specialized prompt for geographic extraction"""
        
        prompt = f"""Extract ALL geographic references and cultural contexts from this text.

Text: {claim_text}

Instructions:
- Find all locations (cities, countries, landmarks, water bodies, etc.)
- Extract cultural contexts (civilizations, peoples, traditions)
- Use canonical names (e.g., "Lake Titicaca" not "Lake Titikaka")
- Assign confidence based on clarity

Return ONLY valid JSON:
{{
  "locations": [
    {{
      "name": "Lake Titicaca",
      "type": "water_body|city|country|landmark|region|mountain",
      "country": "Peru/Bolivia",
      "confidence": 0.95,
      "context": "what this location relates to in the text"
    }}
  ],
  "cultural_context": [
    {{
      "culture": "Inca|Ancient Egyptian|Maya|etc.",
      "confidence": 0.90,
      "period": "if known (e.g., 'Pre-Columbian')",
      "context": "cultural relevance to the text"
    }}
  ]
}}

If no geographic or cultural references, return:
{{
  "locations": [],
  "cultural_context": []
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
            geo_data = json.loads(response)
            
            # Ensure all required keys
            if 'locations' not in geo_data:
                geo_data['locations'] = []
            if 'cultural_context' not in geo_data:
                geo_data['cultural_context'] = []
            
            return geo_data
            
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
                            geo_data = json.loads(response[start:i+1])
                            geo_data.setdefault('locations', [])
                            geo_data.setdefault('cultural_context', [])
                            self.logger.info("Recovered geographic data from malformed JSON")
                            return geo_data
            except:
                pass
            
            return {
                'locations': [],
                'cultural_context': []
            }
    
    def _geocode_location(self, location_name: str) -> Optional[Dict]:
        """
        Get coordinates for location name using geopy
        
        Args:
            location_name: Name of location
            
        Returns:
            Dict with lat, lon, formatted_address or None
        """
        
        # Check cache
        if location_name in self.geocode_cache:
            return self.geocode_cache[location_name]
        
        try:
            # Rate limit to be nice to Nominatim
            time.sleep(1)
            
            location = self.geocoder.geocode(location_name, timeout=5)
            
            if location:
                coords = {
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'formatted': location.address
                }
                
                # Cache result
                self.geocode_cache[location_name] = coords
                
                return coords
            else:
                self.logger.debug(f"No geocoding results for: {location_name}")
                return None
                
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            self.logger.warning(f"Geocoding failed for {location_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Geocoding error for {location_name}: {e}")
            return None

if __name__ == "__main__":
    # Test geographic extractor
    logging.basicConfig(level=logging.INFO)
    
    extractor = GeographicExtractor()
    
    test_text = """
    The Inca civilization at Lake Titicaca developed sophisticated reed boat 
    construction techniques around 1200 CE. Similar methods were found in 
    ancient Egypt near Lake Tana, despite no documented contact between 
    the Pre-Columbian Americas and ancient North Africa.
    """
    
    context = {
        'domain': 'research_investigative',
        'title': 'Test Document'
    }
    
    geo_data = extractor.extract(test_text, context)
    
    print(f"\nExtracted geographic data:")
    print(f"  Locations: {len(geo_data['locations'])}")
    for loc in geo_data['locations']:
        print(f"    - {loc['name']} ({loc['type']})")
        if 'coordinates' in loc:
            print(f"      Coords: {loc['coordinates']['lat']:.2f}, {loc['coordinates']['lon']:.2f}")
    
    print(f"  Cultural context: {len(geo_data['cultural_context'])}")
    for cult in geo_data['cultural_context']:
        print(f"    - {cult['culture']} ({cult.get('period', 'unknown period')})")
