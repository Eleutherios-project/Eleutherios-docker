import os
#!/usr/bin/env python3
"""
Aegis Insight - Cross-Cultural Anomaly Detector

Detects identical complex patterns in isolated cultural contexts:
1. Geographic Isolation - Similar patterns >5000 miles apart
2. Temporal Overlap - Same knowledge in different eras
3. Cultural Isolation - No documented contact between cultures
4. Pattern Complexity - Too complex to arise independently
5. Similarity Score - Semantic similarity of descriptions

Author: Aegis Insight Team
Date: November 2025
"""

import json
import logging
import math
from aegis_config import Config
import psycopg2
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from neo4j import GraphDatabase
from collections import defaultdict, Counter
from datetime import datetime
from aegis_detection_config import get_config
from aegis_topic_utils import expand_topics, get_claims_via_topics_hybrid



class AnomalyDetector:
    """
    Detects cross-cultural anomaly patterns in knowledge graphs
    
    Implements 5 detection signals:
    - Geographic Isolation (0.25 weight)
    - Temporal Overlap (0.15 weight)
    - Cultural Isolation (0.25 weight)
    - Pattern Complexity (0.20 weight)
    - Similarity Score (0.15 weight)
    
    Returns anomaly score 0.0-1.0 with signal breakdown
    """
    
    # Signal weights
    WEIGHTS = {
        'geographic_isolation': 0.25,
        'temporal_overlap': 0.15,
        'cultural_isolation': 0.25,
        'pattern_complexity': 0.20,
        'similarity_score': 0.15
    }
    
    # Thresholds
    HIGH_DISTANCE_MILES = 5000  # Geographic isolation threshold
    SIMILARITY_THRESHOLD = 0.85  # Embedding similarity for anomaly detection
    
    # Known cultural contacts (for cultural isolation scoring)
    # Format: {(culture1, culture2): contact_level}
    # contact_level: 0.0 = no contact, 1.0 = extensive contact
    KNOWN_CONTACTS = {
        ('greek', 'roman'): 0.9,
        ('egyptian', 'greek'): 0.7,
        ('mesopotamian', 'egyptian'): 0.6,
        ('chinese', 'mongolian'): 0.8,
        ('indian', 'chinese'): 0.5,
        ('persian', 'greek'): 0.6,
        ('roman', 'egyptian'): 0.7,
        # Pre-Columbian Americas = isolated
        ('mesoamerican', 'egyptian'): 0.0,
        ('andean', 'egyptian'): 0.0,
        ('mesoamerican', 'mesopotamian'): 0.0,
        ('andean', 'mesopotamian'): 0.0,
        ('polynesian', 'andean'): 0.1,  # Some contact possible
    }
    
    # Pattern complexity keywords
    COMPLEXITY_INDICATORS = {
        'simple': [
            'fire', 'tool', 'shelter', 'clothing', 'pottery',
            'agriculture', 'domestication', 'hunting', 'gathering'
        ],
        'moderate': [
            'metallurgy', 'writing', 'astronomy', 'navigation',
            'architecture', 'irrigation', 'calendar', 'mathematics'
        ],
        'complex': [
            'pyramid', 'megalith', 'precision', 'astronomical alignment',
            'advanced metallurgy', 'complex calendar', 'sophisticated mathematics',
            'precision engineering', 'acoustic properties', 'magnetic alignment'
        ],
        'very_complex': [
            'seven-element narrative', 'identical ritual', 'advanced astronomy',
            'precision stone cutting', 'anti-seismic', 'acoustic resonance',
            'megalithic precision', 'astronomical alignment precision'
        ]
    }
    
    # Complexity scores
    COMPLEXITY_SCORES = {
        'simple': 0.2,
        'moderate': 0.5,
        'complex': 0.8,
        'very_complex': 0.95
    }
    
    def __init__(self,
                 neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize anomaly detector"""
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Neo4j connection
        # Use config fallback if password not provided
        neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            self.driver.verify_connectivity()
            self.logger.info("✓ Connected to Neo4j")
            self.config = get_config()
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        # Connect to PostgreSQL for embeddings
        try:
            self.pg_conn = psycopg2.connect(
                host="localhost",
                database="aegis_insight",
                user="aegis",
                password="aegis_trusted_2025"
            )
            self.logger.info("✓ Connected to PostgreSQL for embeddings")
        except Exception as e:
            self.logger.warning(f"PostgreSQL connection failed (embeddings unavailable): {e}")
            self.pg_conn = None

    def detect_anomaly(self,
                       pattern: str,
                       claim_ids: Optional[List[int]] = None,
                       domain: Optional[str] = None,
                       limit: int = 500) -> Dict:
        """
        Detect cross-cultural anomaly patterns for a given topic/pattern
        
        Args:
                    pattern: Pattern to analyze for anomalies
                    claim_ids: Optional list of claim IDs from semantic search.
                              If provided, analyzes only these claims.
                              If None, falls back to text search (legacy behavior).
                    domain: Optional domain filter
            
        Returns:
            Dict with anomaly analysis
        """
        
        self.logger.info(f"Analyzing anomaly patterns for: {pattern}")
        
        min_distance = self.HIGH_DISTANCE_MILES

        with self.driver.session() as session:
            # Use Topic traversal (same as suppression detector)
            claims = self._get_claims_for_topic(session, pattern, domain, limit=limit)
            self.logger.info(f"Analyzing {len(claims)} claims from Topic traversal")
            
            if not claims:
                self.logger.warning(f"No geographic claims found for pattern: {pattern}")
                return self._empty_result(pattern)
            
            self.logger.info(f"Found {len(claims)} claims with geographic data")
            
            # Group claims by location
            location_groups = self._group_claims_by_location(claims)
            
            if len(location_groups) < 2:  # Need at least 2 locations for comparison
                self.logger.warning(
                    f"Only {len(location_groups)} locations found, need at least 2"
                )
                return self._empty_result(pattern)
            
            self.logger.info(f"Grouped into {len(location_groups)} distinct locations")
            
            # Find anomalous pairs (high isolation + high similarity)
            anomalies = self._find_anomalous_pairs(
                session, location_groups, min_distance
            )
            
            if not anomalies:
                self.logger.info("No anomalous patterns detected")
                return self._no_anomaly_result(pattern, len(location_groups))
            
            self.logger.info(f"Found {len(anomalies)} anomalous pairs")
            
            # Calculate aggregate anomaly score
            anomaly_score, confidence = self._calculate_aggregate_score(anomalies)
            
            # Generate interpretation
            interpretation = self._interpret_score(anomaly_score, len(anomalies))
            
            # Get the top anomaly for detailed reporting
            top_anomaly = max(anomalies, key=lambda x: x['anomaly_score'])
            
            # Classify severity
            if anomaly_score >= 0.8:
                severity = "CRITICAL"
            elif anomaly_score >= 0.6:
                severity = "HIGH"
            elif anomaly_score >= 0.4:
                severity = "MODERATE"
            elif anomaly_score >= 0.2:
                severity = "LOW"
            else:
                severity = "MINIMAL"
            
            # Collect representative claims from top anomalous locations
            # Scale claim collection based on user's limit
            num_anomalies = min(len(anomalies), max(10, limit // 5))  # More anomalies for higher limits
            claims_per_location = max(3, limit // 20)  # More claims per location for higher limits
            
            anomalous_claims = []
            seen_claims = set()
            for anomaly in anomalies[:num_anomalies]:
                loc1, loc2 = anomaly.get('location1', ''), anomaly.get('location2', '')
                for loc in [loc1, loc2]:
                    for item in location_groups.get(loc, [])[:claims_per_location]:
                        claim = item.get('claim', item)  # Unwrap from location_groups structure
                        claim_id = claim.get('claim_id') or claim.get('element_id')
                        if claim_id and claim_id not in seen_claims:
                            seen_claims.add(claim_id)
                            anomalous_claims.append({
                                'id': claim_id,
                                'claim_text': claim.get('claim_text', ''),
                                'location': loc,
                                'source_file': claim.get('source_file', '')
                            })
                        # Stop if we have enough
                        if len(anomalous_claims) >= limit:
                            break
                    if len(anomalous_claims) >= limit:
                        break
                if len(anomalous_claims) >= limit:
                    break
            
            return {
                'pattern': pattern,
                'anomaly_score': round(anomaly_score, 3),
                'severity': severity,
                'confidence': round(confidence, 3),
                'anomalies_found': len(anomalies),
                'locations_analyzed': len(location_groups),
                'top_anomaly': top_anomaly,
                'all_anomalies': anomalies,
                'anomalous_claims': anomalous_claims[:limit],
                'interpretation': interpretation,
                'claims_analyzed': len(claims)
            }


    def _get_claims_for_topic(self, 
                              session,
                              pattern: str,
                              domain: Optional[str] = None,
                              limit: int = 500) -> List[Dict]:
        """
        Retrieve claims using Topic node traversal.
        
        Uses Topic node traversal (not text CONTAINS) to capture ALL claims
        from documents about the topic.
        
        Args:
            session: Neo4j session
            pattern: Topic/pattern to search
            domain: Optional domain filter
            limit: Maximum claims to return
            
        Returns:
            List of claim dicts with geographic data
        """
        
        self.logger.info(f"Fetching claims for pattern '{pattern}' using Topic traversal...")
        
        # Try embedding-based topic expansion first
        expanded_topics = expand_topics(pattern, threshold=0.35)
        
        if expanded_topics:
            self.logger.info(f"Topic expansion found {len(expanded_topics)} topics: {expanded_topics[:5]}...")
            
            # Use hybrid search with text filtering for better precision
            hybrid_claims = get_claims_via_topics_hybrid(session, pattern, threshold=0.35, limit=limit, require_text_match=True)
            
            if hybrid_claims:
                self.logger.info(f"Hybrid search found {len(hybrid_claims)} precise claims")
                # Filter for geographic data and return
                geo_claims = []
                for claim in hybrid_claims:
                    geo = claim.get('geographic_data', {})
                    if isinstance(geo, str):
                        try:
                            geo = json.loads(geo)
                        except:
                            geo = {}
                    if geo.get('locations'):
                        claim['geographic_data'] = geo
                        geo_claims.append(claim)
                if geo_claims:
                    self.logger.info(f"Found {len(geo_claims)} claims with geographic data")
                    return geo_claims[:limit]
            
            # Fallback to broad topic search if hybrid returns nothing
            topic_words = None
            query = """
            MATCH (t:Topic)
            WHERE t.name IN $topic_names
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WHERE c.geographic_data IS NOT NULL
              AND c.geographic_data <> '{}'
              AND c.geographic_data <> ''
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.geographic_data as geographic_data,
                   c.temporal_data as temporal_data,
                   d.source_file as doc_source,
                   elementId(c) as element_id
            ORDER BY c.confidence DESC
            LIMIT $limit
            """
        else:
            # Fallback to word matching
            normalized_topic = pattern.replace('-', ' ').replace('  ', ' ')
            topic_words = [w.strip() for w in normalized_topic.split() if len(w.strip()) > 2]
            query = """
            MATCH (t:Topic)
            WHERE ALL(word IN $topic_words WHERE toLower(t.name) CONTAINS toLower(word))
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WHERE c.geographic_data IS NOT NULL
              AND c.geographic_data <> '{}'
              AND c.geographic_data <> ''
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.geographic_data as geographic_data,
                   c.temporal_data as temporal_data,
                   d.source_file as doc_source,
                   elementId(c) as element_id
            ORDER BY c.confidence DESC
            LIMIT $limit
            """
        
        claims = []
        
        try:
            if expanded_topics:
                result = session.run(query, topic_names=expanded_topics, limit=limit)
            else:
                result = session.run(query, topic_words=topic_words, limit=limit)
            
            for record in result:
                claim = dict(record)
                if not claim.get('source_file'):
                    claim['source_file'] = claim.get('doc_source')
                # Parse geographic_data JSON
                if claim.get('geographic_data'):
                    try:
                        claim['geographic_data'] = json.loads(claim['geographic_data']) if isinstance(claim['geographic_data'], str) else claim['geographic_data']
                    except:
                        claim['geographic_data'] = {}
                # Parse temporal_data JSON
                if claim.get('temporal_data'):
                    try:
                        claim['temporal_data'] = json.loads(claim['temporal_data']) if isinstance(claim['temporal_data'], str) else claim['temporal_data']
                    except:
                        claim['temporal_data'] = {}
                # Only include if we have valid location data
                geo_data = claim.get('geographic_data', {})
                if geo_data.get('locations'):
                    claims.append(claim)
            
            self.logger.info(f"Topic traversal found {len(claims)} claims with geographic data")
            
        except Exception as e:
            self.logger.warning(f"Topic traversal failed: {e}, falling back to text search")
            return self._get_claims_by_text(session, pattern, domain, limit)
        
        # Fallback if no results
        if len(claims) == 0:
            self.logger.info("No claims via Topic traversal, trying text search fallback...")
            return self._get_claims_by_text(session, pattern, domain, limit)
        
        return claims


    def _get_claims_by_text(self, session, pattern: str, domain: Optional[str], limit: int = 500) -> List[Dict]:
        """
        LEGACY METHOD: Retrieve claims using text search

        Uses simple CONTAINS matching. Less accurate than semantic search.
        """
        
        query = """
        MATCH (c:Claim)
        WHERE toLower(c.claim_text) CONTAINS toLower($pattern)
          AND c.geographic_data IS NOT NULL
          AND c.geographic_data <> '{}'
          AND c.geographic_data <> ''
        RETURN c.claim_id as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.source_file as source_file,
               c.geographic_data as geographic_data,
               c.temporal_data as temporal_data,
               c.created_at as created_at
               ORDER BY c.confidence DESC
               LIMIT $limit
        """
        
        result = session.run(query, pattern=pattern, limit=limit)
        
        claims = []
        for record in result:
            claim = dict(record)
            
            # Parse geographic_data JSON
            if claim.get('geographic_data'):
                try:
                    claim['geographic_data'] = json.loads(claim['geographic_data'])
                except:
                    claim['geographic_data'] = {}
            
            # Parse temporal_data JSON
            if claim.get('temporal_data'):
                try:
                    claim['temporal_data'] = json.loads(claim['temporal_data'])
                except:
                    claim['temporal_data'] = {}
            
            # Only include if we have valid location data
            geo_data = claim.get('geographic_data', {})
            if geo_data.get('locations'):
                claims.append(claim)
        
        return claims

    def _get_claims_by_ids(self, session, claim_ids: List[int]) -> List[Dict]:
        """
        NEW METHOD: Fetch specific claims by their Neo4j IDs

        Used when claim IDs come from semantic search (pattern search).

        Args:
            session: Neo4j session
            claim_ids: List of Neo4j node IDs (integers)

        Returns:
            List of claim dicts with all required fields
        """

        query = """
        MATCH (c:Claim)
        WHERE elementId(c) IN $claim_ids

        RETURN elementId(c) as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.geographic_data as geographic_data,
               c.source_file as source_file
               ORDER BY c.confidence DESC
               LIMIT $limit
        """

        result = session.run(query, claim_ids=claim_ids, limit=self.config.max_claims_analyzed)

        claims = []
        for record in result:
            claim = dict(record)

            # Parse JSON field
            if claim.get('geographic_data'):
                try:
                    claim['geographic_data'] = json.loads(claim['geographic_data']) if isinstance(
                        claim['geographic_data'], str) else claim['geographic_data']
                except:
                    claim['geographic_data'] = {}

            claims.append(claim)

        return claims
    
    def _group_claims_by_location(self, claims: List[Dict]) -> Dict[str, List[Dict]]:
        """Group claims by geographic location"""
        
        location_groups = defaultdict(list)
        
        for claim in claims:
            geo_data = claim.get('geographic_data', {})
            locations = geo_data.get('locations', [])
            
            for location in locations:
                # Use primary location name as key
                location_name = location.get('name', '') or location.get('location', '')
                if location_name:
                    location_groups[location_name].append({
                        'claim': claim,
                        'location': location
                    })
        
        return dict(location_groups)
    
    def _find_anomalous_pairs(self,
                             session,
                             location_groups: Dict[str, List[Dict]],
                             min_distance: float) -> List[Dict]:
        """Find pairs of locations with anomalous patterns"""
        
        anomalies = []
        locations = list(location_groups.keys())
        
        # Compare each pair of locations
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                loc1 = locations[i]
                loc2 = locations[j]
                
                group1 = location_groups[loc1]
                group2 = location_groups[loc2]
                
                # Calculate signals for this pair
                signals = self._calculate_pair_signals(
                    session, group1, group2, min_distance
                )
                
                # Calculate anomaly score for this pair
                pair_score = self._aggregate_pair_signals(signals)
                self.logger.info(f"Pair {loc1} <-> {loc2}: score={pair_score:.3f}")
                
                # If anomalous (score > 0.6), add to results
                if pair_score > 0.3:  # Lowered from 0.6 for testing
                    anomalies.append({
                        'location1': loc1,
                        'location2': loc2,
                        'anomaly_score': round(pair_score, 3),
                        'signals': signals,
                        'claims_at_loc1': len(group1),
                        'claims_at_loc2': len(group2)
                    })
        
        return anomalies
    
    def _calculate_pair_signals(self,
                               session,
                               group1: List[Dict],
                               group2: List[Dict],
                               min_distance: float) -> Dict:
        """Calculate all signals for a location pair"""
        
        signals = {}
        
        # Signal 1: Geographic Isolation
        signals['geographic_isolation'] = self._calculate_geographic_isolation(
            group1, group2, min_distance
        )
        
        # Signal 2: Temporal Overlap
        signals['temporal_overlap'] = self._calculate_temporal_overlap(
            group1, group2
        )
        
        # Signal 3: Cultural Isolation
        signals['cultural_isolation'] = self._calculate_cultural_isolation(
            group1, group2
        )
        
        # Signal 4: Pattern Complexity
        signals['pattern_complexity'] = self._calculate_pattern_complexity(
            group1, group2
        )
        
        # Signal 5: Similarity Score (requires embeddings)
        signals['similarity_score'] = self._calculate_similarity_score(
            session, group1, group2
        )
        
        return signals
    
    def _calculate_geographic_isolation(self,
                                       group1: List[Dict],
                                       group2: List[Dict],
                                       min_distance: float) -> Dict:
        """
        Calculate geographic isolation signal
        
        High score = locations far apart with no geographic connection
        """
        
        # Get coordinates from first claim in each group
        loc1 = group1[0]['location']
        loc2 = group2[0]['location']
        
        coords1 = loc1.get('coordinates', {})
        coords2 = loc2.get('coordinates', {})
        
        if not (coords1 and coords2):
            return {
                'score': 0.0,
                'isolated': False,
                'interpretation': 'Missing coordinate data'
            }
        
        lat1 = coords1.get('latitude')
        lon1 = coords1.get('longitude')
        lat2 = coords2.get('latitude')
        lon2 = coords2.get('longitude')
        
        if None in [lat1, lon1, lat2, lon2]:
            return {
                'score': 0.0,
                'isolated': False,
                'interpretation': 'Incomplete coordinate data'
            }
        
        # Calculate distance using Haversine formula
        distance_km = self._haversine_distance(lat1, lon1, lat2, lon2)
        distance_miles = distance_km * 0.621371
        
        # Score based on distance
        if distance_miles >= min_distance:
            # Scale score: 5000 miles = 0.7, 8000+ miles = 1.0
            score = min(0.7 + (distance_miles - min_distance) / 10000, 1.0)
            isolated = True
        else:
            score = distance_miles / min_distance * 0.5
            isolated = False
        
        interpretation = self._interpret_geographic_isolation(
            distance_miles, isolated
        )
        
        return {
            'score': round(score, 3),
            'isolated': isolated,
            'distance_miles': round(distance_miles, 1),
            'distance_km': round(distance_km, 1),
            'interpretation': interpretation
        }
    
    def _haversine_distance(self,
                           lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth in kilometers"""
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon / 2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _calculate_temporal_overlap(self,
                                   group1: List[Dict],
                                   group2: List[Dict]) -> Dict:
        """
        Calculate temporal overlap signal
        
        High score = cultures existed at same time
        """
        
        # Extract time periods from both groups
        dates1 = self._extract_time_periods(group1)
        dates2 = self._extract_time_periods(group2)
        
        if not (dates1 and dates2):
            return {
                'score': 0.5,  # Assume possible overlap if no data
                'overlap_detected': True,
                'interpretation': 'Insufficient temporal data - assuming overlap possible'
            }
        
        # Check for overlap
        overlap = self._check_temporal_overlap(dates1, dates2)
        
        if overlap:
            # Overlap increases anomaly (simultaneous independent invention)
            score = 0.8
            interpretation = f"Temporal overlap detected - patterns existed simultaneously"
        else:
            # No overlap = possible transmission, less anomalous
            score = 0.3
            interpretation = f"No temporal overlap - sequential development possible"
        
        return {
            'score': round(score, 3),
            'overlap_detected': overlap,
            'period1': dates1.get('range', 'Unknown'),
            'period2': dates2.get('range', 'Unknown'),
            'interpretation': interpretation
        }
    
    def _extract_time_periods(self, group: List[Dict]) -> Optional[Dict]:
        """Extract time period information from claim group"""
        
        all_dates = []
        
        for item in group:
            claim = item['claim']
            temporal = claim.get('temporal_data', {})
            
            if temporal.get('dates'):
                for date_obj in temporal['dates']:
                    date_str = date_obj.get('date')
                    if date_str:
                        try:
                            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            all_dates.append(date)
                        except:
                            pass
        
        if not all_dates:
            return None
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        return {
            'min': min_date,
            'max': max_date,
            'range': f"{min_date.year} - {max_date.year}"
        }
    
    def _check_temporal_overlap(self, dates1: Dict, dates2: Dict) -> bool:
        """Check if two time periods overlap"""
        
        # Periods overlap if one starts before the other ends
        overlap = (dates1['min'] <= dates2['max'] and 
                  dates2['min'] <= dates1['max'])
        
        return overlap
    
    def _calculate_cultural_isolation(self,
                                     group1: List[Dict],
                                     group2: List[Dict]) -> Dict:
        """
        Calculate cultural isolation signal
        
        High score = no documented cultural contact
        """
        
        # Try to identify cultures from location names or claim text
        culture1 = self._identify_culture(group1)
        culture2 = self._identify_culture(group2)
        
        # Check known contacts
        contact_level = self._get_contact_level(culture1, culture2)
        
        # Invert: no contact = high isolation score
        isolation_score = 1.0 - contact_level
        
        isolated = isolation_score > 0.7
        
        interpretation = self._interpret_cultural_isolation(
            culture1, culture2, contact_level, isolated
        )
        
        return {
            'score': round(isolation_score, 3),
            'isolated': isolated,
            'culture1': culture1,
            'culture2': culture2,
            'documented_contact': round(contact_level, 2),
            'interpretation': interpretation
        }
    
    def _identify_culture(self, group: List[Dict]) -> str:
        """Identify culture from location and claim data"""
        
        # Get location name
        location = group[0]['location'].get('location', '').lower()
        
        # Simple keyword matching
        culture_keywords = {
            'egyptian': ['egypt', 'cairo', 'nile', 'pharaoh'],
            'greek': ['greece', 'athens', 'sparta', 'delphi'],
            'roman': ['rome', 'italy', 'roman'],
            'mesopotamian': ['mesopotamia', 'babylon', 'sumerian', 'iraq'],
            'chinese': ['china', 'chinese', 'beijing', 'yellow river'],
            'indian': ['india', 'indian', 'indus', 'ganges'],
            'mesoamerican': ['mexico', 'maya', 'aztec', 'olmec', 'teotihuacan'],
            'andean': ['peru', 'inca', 'bolivia', 'andes', 'titicaca'],
            'polynesian': ['hawaii', 'tahiti', 'samoa', 'easter island'],
            'persian': ['persia', 'iran', 'persian']
        }
        
        for culture, keywords in culture_keywords.items():
            if any(keyword in location for keyword in keywords):
                return culture
        
        # Check claim text if location not identified
        for item in group[:3]:  # Check first 3 claims
            claim_text = item['claim'].get('claim_text', '').lower()
            for culture, keywords in culture_keywords.items():
                if any(keyword in claim_text for keyword in keywords):
                    return culture
        
        return 'unknown'
    
    def _get_contact_level(self, culture1: str, culture2: str) -> float:
        """Get documented contact level between cultures"""
        
        # Check both orderings
        key1 = (culture1, culture2)
        key2 = (culture2, culture1)
        
        if key1 in self.KNOWN_CONTACTS:
            return self.KNOWN_CONTACTS[key1]
        elif key2 in self.KNOWN_CONTACTS:
            return self.KNOWN_CONTACTS[key2]
        else:
            # Unknown cultures or no documented contact
            # Be conservative: assume some contact possible
            return 0.3
    
    def _calculate_pattern_complexity(self,
                                     group1: List[Dict],
                                     group2: List[Dict]) -> Dict:
        """
        Calculate pattern complexity signal
        
        High score = pattern too complex to arise independently
        """
        
        # Analyze claim text for complexity indicators
        all_claims = [item['claim'].get('claim_text', '') 
                     for item in group1 + group2]
        
        combined_text = ' '.join(all_claims).lower()
        
        # Check complexity levels
        complexity_level = 'simple'
        max_score = 0.0
        
        for level, keywords in self.COMPLEXITY_INDICATORS.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches > 0:
                score = self.COMPLEXITY_SCORES[level]
                if score > max_score:
                    max_score = score
                    complexity_level = level
        
        complex = max_score >= 0.8
        
        interpretation = self._interpret_pattern_complexity(
            complexity_level, complex
        )
        
        return {
            'score': round(max_score, 3),
            'complex': complex,
            'complexity_level': complexity_level,
            'interpretation': interpretation
        }
    
    def _calculate_similarity_score(self,
                                   session,
                                   group1: List[Dict],
                                   group2: List[Dict]) -> Dict:
        """
        Calculate semantic similarity between claim groups
        
        Requires embeddings to be available
        """
        
        # Get claim IDs
        claim_ids1 = [item['claim']['claim_id'] for item in group1]
        claim_ids2 = [item['claim']['claim_id'] for item in group2]
        
        # Query for embedding similarities
        try:
            similarities = self._get_embedding_similarities(
                session, claim_ids1, claim_ids2
            )
            
            if not similarities:
                return {
                    'score': 0.5,  # Neutral if no embeddings
                    'similar': False,
                    'avg_similarity': 0.0,
                    'interpretation': 'Embeddings not available',
                    'note': 'Run generate_embeddings to enable this signal'
                }
            
            # Calculate average similarity
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            
            # High similarity = anomalous
            if avg_similarity >= self.SIMILARITY_THRESHOLD:
                score = 0.95
                similar = True
            elif avg_similarity >= 0.75:
                score = 0.8
                similar = True
            elif avg_similarity >= 0.65:
                score = 0.6
                similar = False
            else:
                score = avg_similarity * 0.5
                similar = False
            
            interpretation = self._interpret_similarity(
                avg_similarity, max_similarity, similar
            )
            
            return {
                'score': round(score, 3),
                'similar': similar,
                'avg_similarity': round(avg_similarity, 3),
                'max_similarity': round(max_similarity, 3),
                'comparisons': len(similarities),
                'interpretation': interpretation
            }
            
        except Exception as e:
            self.logger.warning(f"Could not calculate similarities: {e}")
            return {
                'score': 0.5,
                'similar': False,
                'avg_similarity': 0.0,
                'interpretation': 'Similarity calculation failed',
                'note': 'Embeddings may not be available'
            }

    def _get_embedding_similarities(self,
                                    session,
                                    claim_ids1: List[str],
                                    claim_ids2: List[str]) -> List[float]:
        """
        Query PostgreSQL pgvector for embedding similarities between claim groups.
        """
        if not self.pg_conn:
            self.logger.debug("No PostgreSQL connection - skipping embedding similarity")
            return []

        if not claim_ids1 or not claim_ids2:
            return []

        try:
            cur = self.pg_conn.cursor()

            # Get embeddings for group 1
            cur.execute("""
                            SELECT claim_id, embedding
                            FROM claim_embeddings 
                            WHERE claim_id = ANY(%s)
                        """, (claim_ids1,))
            rows1 = cur.fetchall()

            if not rows1:
                return []

            # Get embeddings for group 2
            cur.execute("""
                            SELECT claim_id, embedding
                            FROM claim_embeddings 
                            WHERE claim_id = ANY(%s)
                        """, (claim_ids2,))
            rows2 = cur.fetchall()

            if not rows2:
                return []

            # Parse embeddings from pgvector string format to numpy arrays
            # pgvector returns strings like '[0.016238958,0.11214772,...]'
            def parse_embedding(emb):
                if emb is None:
                    return None
                if isinstance(emb, str):
                    # Parse JSON array string to list, then to numpy
                    return np.array(json.loads(emb))
                else:
                    # Already a list/array
                    return np.array(emb)
            
            # Calculate cosine similarities using numpy
            similarities = []

            for id1, emb1 in rows1:
                vec1 = parse_embedding(emb1)
                if vec1 is None:
                    continue
                norm1 = np.linalg.norm(vec1)
                if norm1 == 0:
                    continue

                for id2, emb2 in rows2:
                    vec2 = parse_embedding(emb2)
                    if vec2 is None:
                        continue
                    norm2 = np.linalg.norm(vec2)
                    if norm2 == 0:
                        continue

                    # Cosine similarity = dot product / (norm1 * norm2)
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    similarities.append(float(similarity))

            return similarities

        except Exception as e:
            self.logger.error(f"Error calculating embedding similarities: {e}")
            return []
    
    def _aggregate_pair_signals(self, signals: Dict) -> float:
        """Aggregate signals into anomaly score for a location pair"""
        
        score = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.WEIGHTS.items():
            signal = signals.get(signal_name, {})
            signal_score = signal.get('score', 0.0)
            
            score += signal_score * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            score = score / total_weight
        
        return score
    
    def _calculate_aggregate_score(self, anomalies: List[Dict]) -> Tuple[float, float]:
        """Calculate aggregate anomaly score across all pairs"""
        
        if not anomalies:
            return 0.0, 0.0
        
        # Average of top anomalies (weighted toward highest scores)
        scores = sorted([a['anomaly_score'] for a in anomalies], reverse=True)
        
        # Weight: 50% top score, 30% second, 20% rest
        if len(scores) == 1:
            aggregate = scores[0]
        elif len(scores) == 2:
            aggregate = scores[0] * 0.6 + scores[1] * 0.4
        else:
            top = scores[0] * 0.5
            second = scores[1] * 0.3
            rest = sum(scores[2:]) / len(scores[2:]) * 0.2
            aggregate = top + second + rest
        
        # Confidence based on number of anomalies found
        if len(anomalies) >= 5:
            confidence = 0.95
        elif len(anomalies) >= 3:
            confidence = 0.85
        elif len(anomalies) >= 2:
            confidence = 0.75
        else:
            confidence = 0.6
        
        return aggregate, confidence
    
    def _interpret_score(self, score: float, num_anomalies: int) -> str:
        """Generate interpretation of anomaly score"""
        
        if score >= 0.85:
            level = "STRONG cross-cultural anomaly detected"
        elif score >= 0.70:
            level = "SIGNIFICANT anomaly pattern detected"
        elif score >= 0.55:
            level = "MODERATE anomaly signals present"
        else:
            level = "WEAK anomaly indicators"
        
        interpretation = f"{level}. "
        interpretation += f"Found {num_anomalies} anomalous location pair(s). "
        
        if score >= 0.75:
            interpretation += "Pattern suggests common ancient source or lost cultural contact."
        elif score >= 0.60:
            interpretation += "Pattern warrants further investigation."
        else:
            interpretation += "Pattern may be coincidental or result of indirect transmission."
        
        return interpretation
    
    # Interpretation helpers
    
    def _interpret_geographic_isolation(self, distance: float, isolated: bool) -> str:
        if isolated:
            return f"High geographic isolation: {distance:.0f} miles apart"
        else:
            return f"Moderate distance: {distance:.0f} miles - contact possible"
    
    def _interpret_cultural_isolation(self, 
                                      culture1: str, 
                                      culture2: str,
                                      contact: float,
                                      isolated: bool) -> str:
        if isolated:
            return f"High cultural isolation: {culture1}-{culture2}, no documented contact ({contact:.0%})"
        else:
            return f"Known cultural contact: {culture1}-{culture2} ({contact:.0%})"
    
    def _interpret_pattern_complexity(self, level: str, complex: bool) -> str:
        if complex:
            return f"High complexity: {level} pattern - unlikely independent invention"
        else:
            return f"Moderate complexity: {level} pattern - independent invention possible"
    
    def _interpret_similarity(self, avg: float, max_val: float, similar: bool) -> str:
        if similar:
            return f"High similarity: avg {avg:.2f}, max {max_val:.2f} - nearly identical patterns"
        else:
            return f"Moderate similarity: avg {avg:.2f} - some commonality but distinct"
    
    def _empty_result(self, pattern: str) -> Dict:
        """Return empty result when no data available"""
        return {
            'pattern': pattern,
            'anomaly_score': 0.0,
            'severity': 'MINIMAL',
            'confidence': 0.0,
            'anomalies_found': 0,
            'locations_analyzed': 0,
            'top_anomaly': None,
            'all_anomalies': [],
            'interpretation': 'No geographic data available for analysis'
        }
    
    def _no_anomaly_result(self, pattern: str, locations: int) -> Dict:
        """Return result when no anomalies detected"""
        return {
            'pattern': pattern,
            'anomaly_score': 0.0,
            'severity': 'MINIMAL',
            'confidence': 0.8,
            'anomalies_found': 0,
            'locations_analyzed': locations,
            'top_anomaly': None,
            'all_anomalies': [],
            'interpretation': f'Analyzed {locations} locations - no significant anomalies detected'
        }

    def close(self):
        """Close database connections"""
        if self.driver:
            self.driver.close()
        if self.pg_conn:
            self.pg_conn.close()

def main():
    """Test anomaly detector"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    detector = AnomalyDetector()
    
    # Test patterns
    test_patterns = [
        "pyramid construction",
        "flood narrative",
        "reed boat",
        "astronomical alignment"
    ]
    
    for pattern in test_patterns:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {pattern}")
        print('='*80)
        
        result = detector.detect_anomaly(pattern)
        
        print(f"\nAnomaly Score: {result['anomaly_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Anomalies Found: {result['anomalies_found']}")
        print(f"Locations Analyzed: {result['locations_analyzed']}")
        print(f"\nInterpretation: {result['interpretation']}")
        
        if result['top_anomaly']:
            top = result['top_anomaly']
            print(f"\nTop Anomaly:")
            print(f"  {top['location1']} ↔ {top['location2']}")
            print(f"  Score: {top['anomaly_score']:.3f}")
            
            print(f"\n  Signal Breakdown:")
            for signal_name, signal_data in top['signals'].items():
                print(f"    {signal_name}:")
                print(f"      Score: {signal_data.get('score', 0.0):.3f}")
                print(f"      {signal_data.get('interpretation', 'N/A')}")
    
    detector.close()
    
    print(f"\n{'='*80}")
    print("DETECTION COMPLETE")
    print('='*80)


if __name__ == "__main__":
    main()
