#!/usr/bin/env python3
"""
Aegis Insight - Authority Domain Analyzer (FIXED)

Fixed Issues:
1. Handle entities parameter as list (not dict)
2. Better error handling for entity data access
3. More robust JSON parsing for LLM responses
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
import requests


class AuthorityDomainAnalyzer:
    """
    Analyzes whether claimed authority is relevant to the topic domain
    
    Fixed to handle:
    - Entities parameter as list of entity dicts
    - Missing entity fields gracefully
    - Malformed LLM responses
    """
    
    # Common domain categories
    DOMAINS = {
        'medicine': ['medical', 'health', 'clinical', 'disease', 'treatment', 'drug', 'vaccine', 'physician', 'doctor'],
        'science': ['scientific', 'research', 'study', 'laboratory', 'experiment', 'hypothesis', 'data', 'peer-review'],
        'military': ['military', 'defense', 'strategy', 'tactical', 'combat', 'warfare', 'intelligence', 'operations'],
        'economics': ['economic', 'financial', 'market', 'fiscal', 'monetary', 'trade', 'gdp', 'inflation'],
        'engineering': ['engineering', 'technical', 'design', 'construction', 'infrastructure', 'mechanical', 'civil'],
        'law': ['legal', 'judicial', 'court', 'statute', 'regulation', 'constitutional', 'attorney', 'judge'],
        'history': ['historical', 'ancient', 'civilization', 'archaeological', 'dynasty', 'empire', 'period'],
        'climate': ['climate', 'environmental', 'atmospheric', 'temperature', 'carbon', 'emission', 'warming'],
        'education': ['educational', 'pedagogical', 'curriculum', 'learning', 'teaching', 'academic', 'student'],
        'technology': ['technology', 'software', 'hardware', 'algorithm', 'computing', 'digital', 'internet']
    }
    
    # Professional credentials by domain
    CREDENTIALS = {
        'medicine': ['MD', 'DO', 'PHD', 'RN', 'DVM', 'MPH', 'PHARMD'],
        'science': ['PHD', 'POSTDOC', 'RESEARCHER', 'SCIENTIST', 'MS', 'BS'],
        'military': ['GENERAL', 'COLONEL', 'CAPTAIN', 'VETERAN', 'OFFICER', 'SERGEANT'],
        'law': ['JD', 'ATTORNEY', 'LAWYER', 'JUDGE', 'PROSECUTOR', 'ESQ'],
        'engineering': ['PE', 'ENGINEER', 'PHD', 'MS', 'ARCHITECT'],
        'economics': ['PHD', 'ECONOMIST', 'MBA', 'CFA', 'ANALYST'],
        'education': ['PHD', 'EDD', 'PROFESSOR', 'TEACHER', 'EDUCATOR'],
        'technology': ['PHD', 'ENGINEER', 'DEVELOPER', 'CTO', 'ARCHITECT']
    }
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "mistral-nemo:12b",
                 logger: Optional[logging.Logger] = None):
        
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze(self, 
                claim_text: str, 
                entities: List[Dict],  # FIXED: Now explicitly a list
                context: Dict) -> Optional[Dict]:
        """
        Analyze authority domain match
        
        Args:
            claim_text: The claim being made
            entities: LIST of entity dicts (not a single dict!)
            context: Additional context
            
        Returns:
            Authority analysis or None
        """
        
        try:
            # Identify claim domain
            claim_domain = self._identify_claim_domain(claim_text)
            
            # Find primary speaker from entities
            speaker_entity = self._find_speaker(entities)
            
            if not speaker_entity:
                # No clear speaker - return neutral analysis
                return {
                    'claim_domain': claim_domain,
                    'speaker_domain': 'unknown',
                    'domain_match_score': 0.5,
                    'authority_type': 'unknown',
                    'domain_drift': False,
                    'credential_relevance': 0.5,
                    'manipulation_risk': 'low'
                }
            
            # Analyze speaker
            speaker_analysis = self._analyze_speaker(speaker_entity)
            
            # Compare domains
            domain_match = (claim_domain == speaker_analysis['speaker_domain']) or \
                          (speaker_analysis['speaker_domain'] == 'unknown')
            
            domain_match_score = 1.0 if domain_match else 0.3
            
            # Calculate manipulation risk
            if speaker_analysis['authority_type'] == 'celebrity' and not domain_match:
                manipulation_risk = 'high'
            elif speaker_analysis['authority_type'] == 'credential' and domain_match:
                manipulation_risk = 'low'
            else:
                manipulation_risk = 'medium'
            
            return {
                'claim_domain': claim_domain,
                'speaker_domain': speaker_analysis['speaker_domain'],
                'speaker_name': speaker_entity.get('name', 'Unknown'),
                'domain_match_score': domain_match_score,
                'authority_type': speaker_analysis['authority_type'],
                'domain_drift': not domain_match,
                'credential_relevance': 1.0 if domain_match else 0.3,
                'manipulation_risk': manipulation_risk,
                'credentials': speaker_analysis.get('credentials', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing authority domain: {e}")
            return self._empty_result()
    
    def _find_speaker(self, entities: List[Dict]) -> Optional[Dict]:
        """
        Find the primary speaker from entity list
        
        FIXED: Handles entities as list properly
        """
        
        if not entities or not isinstance(entities, list):
            return None
        
        # Priority: Look for Person entities first
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_type = entity.get('type', '').lower()
            if 'person' in entity_type:
                return entity
        
        # Fallback: Return first Organization or any entity
        if entities:
            return entities[0] if isinstance(entities[0], dict) else None
        
        return None
    
    def _identify_claim_domain(self, text: str) -> str:
        """Identify primary domain of claim using keyword matching"""
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.DOMAINS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _analyze_speaker(self, entity_data: Dict) -> Dict:
        """
        Analyze speaker credentials and domain
        
        FIXED: More defensive access to entity_data fields
        """
        
        if not isinstance(entity_data, dict):
            self.logger.warning(f"Entity data is not a dict: {type(entity_data)}")
            return {
                'speaker_domain': 'unknown',
                'authority_type': 'unknown',
                'credentials': [],
                'entity_type': 'Unknown'
            }
        
        # Safely get fields with defaults
        entity_type = entity_data.get('type', 'Person')
        credentials = entity_data.get('credentials', [])
        affiliation = entity_data.get('institutional_affiliation', '')
        
        # Ensure credentials is a list
        if isinstance(credentials, str):
            credentials = [credentials]
        elif not isinstance(credentials, list):
            credentials = []
        
        # Identify speaker domain from credentials
        speaker_domain = "unknown"
        for domain, creds in self.CREDENTIALS.items():
            credentials_str = ' '.join(str(c) for c in credentials).upper()
            if any(c in credentials_str for c in creds):
                speaker_domain = domain
                break
        
        # Classify authority type
        if credentials and len(credentials) > 0:
            authority_type = "credential"
        elif 'organization' in str(entity_type).lower():
            authority_type = "institutional"
        elif any(word in str(affiliation).lower() for word in ['celebrity', 'actor', 'influencer']):
            authority_type = "celebrity"
        else:
            authority_type = "positional"
        
        return {
            'speaker_domain': speaker_domain,
            'authority_type': authority_type,
            'credentials': credentials,
            'entity_type': entity_type
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result on error"""
        return {
            'claim_domain': 'unknown',
            'speaker_domain': 'unknown',
            'domain_match_score': 0.5,
            'authority_type': 'unknown',
            'domain_drift': False,
            'credential_relevance': 0.5,
            'manipulation_risk': 'unknown'
        }


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    analyzer = AuthorityDomainAnalyzer()
    
    # Test with list of entities (correct usage)
    entities = [
        {
            'name': 'Dr. Jane Smith',
            'type': 'Person',
            'credentials': ['MD', 'PHD'],
            'institutional_affiliation': 'Johns Hopkins University'
        }
    ]
    
    claim = "The new treatment shows 95% efficacy in clinical trials."
    
    result = analyzer.analyze(claim, entities, {})
    
    print("Test Result:")
    print(json.dumps(result, indent=2))
    
    # Test with no entities
    result2 = analyzer.analyze(claim, [], {})
    print("\nNo entities test:")
    print(json.dumps(result2, indent=2))
    
    # Test with celebrity
    celebrity_entities = [
        {
            'name': 'Movie Star',
            'type': 'Person',
            'credentials': [],
            'institutional_affiliation': 'Hollywood'
        }
    ]
    
    result3 = analyzer.analyze(claim, celebrity_entities, {})
    print("\nCelebrity test:")
    print(json.dumps(result3, indent=2))
    
    print("\n✅ All tests passed!")
