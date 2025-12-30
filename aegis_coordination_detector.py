import os
#!/usr/bin/env python3
"""
Aegis Insight - Coordination Pattern Detector (Fixed)

CHANGES:
- Actually uses claim_ids when provided (instead of ignoring them)
- Falls back to Topic traversal when no claim_ids provided
- Handles both elementId and claim_id string formats

Detects manufactured consensus through synchronized messaging:
1. Temporal Clustering - Claims appearing in narrow time windows
2. Language Similarity - Near-identical wording across sources
3. Citation Cartel - Hub-spoke network patterns
4. Synchronized Emotional Triggers - Coordinated fear/anger appeals
5. Source Centralization - Many sources tracing to single origin

Author: Aegis Insight Team
Date: December 2025
"""

import json
from aegis_config import Config
from aegis_topic_utils import expand_topics, get_claims_via_topics_hybrid
import logging
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
import psycopg2
from psycopg2.extras import RealDictCursor
from aegis_detection_config import get_config

class CoordinationDetector:
    """
    Detects coordination patterns in knowledge graphs
    
    Implements 5 detection signals:
    - Temporal Clustering (0.25 weight)
    - Language Similarity (0.25 weight)
    - Citation Cartel (0.20 weight)
    - Synchronized Emotional Triggers (0.15 weight)
    - Source Centralization (0.15 weight)
    
    Returns coordination score 0.0-1.0 with signal breakdown
    """
    
    # Signal weights
    WEIGHTS = {
        'temporal_clustering': 0.25,
        'language_similarity': 0.25,
        'citation_cartel': 0.20,
        'synchronized_emotion': 0.15,
        'source_centralization': 0.15
    }
    
    # Thresholds
    BURST_THRESHOLD_DAYS = 14  # Claims within 14 days = potential burst
    SIMILARITY_THRESHOLD = 0.85  # Embedding similarity > 0.85 = likely coordination
    
    # Emotion keywords for synchronization detection
    FEAR_KEYWORDS = [
        'crisis', 'disaster', 'emergency', 'catastrophic', 'deadly',
        'dangerous', 'threat', 'unprecedented', 'shocking'
    ]
    
    ANGER_KEYWORDS = [
        'outrage', 'outrageous', 'scandal', 'fraud', 'betrayal',
        'criminal', 'abuse', 'violation', 'attack'
    ]
    
    URGENCY_KEYWORDS = [
        'now', 'immediately', 'urgent', 'act now', 'critical',
        'must', 'before it\'s too late'
    ]
    
    def __init__(self,
                 neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = None,
                 pg_host: str = os.environ.get("POSTGRES_HOST", "localhost"),
                 pg_database: str = "aegis_insight",
                 pg_user: str = "aegis",
                 pg_password: str = "aegis_trusted_2025",
                 logger: Optional[logging.Logger] = None):
        """Initialize coordination detector"""
        
        # PostgreSQL connection for embeddings
        self.pg_config = {
            'host': pg_host,
            'database': pg_database,
            'user': pg_user,
            'password': pg_password
        }
        
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

    def detect_coordination(self,
                            topic: str,
                            claim_ids: Optional[List[str]] = None,
                            domain: Optional[str] = None,
                            time_range: Optional[Tuple[str, str]] = None,
                            limit: int = 500) -> Dict:
        """
        Detect coordination patterns for a given topic
        
        Args:
            topic: Topic to analyze
            claim_ids: Optional list of claim IDs from pattern search.
                      If provided, analyzes only these claims.
                      If None, falls back to Topic traversal.
            domain: Optional domain filter
            time_range: Optional (start_date, end_date) tuple
            limit: Maximum claims to analyze
            
        Returns:
            Dict with coordination analysis
        """
        
        self.logger.info(f"Analyzing coordination patterns for topic: {topic}")
        if claim_ids:
            self.logger.info(f"Received {len(claim_ids)} claim_ids from pattern search")

        with self.driver.session() as session:
            # PREFER claim_ids if provided (from pattern search)
            if claim_ids and len(claim_ids) > 0:
                claims = self._get_claims_by_ids(session, claim_ids)
                self.logger.info(f"Analyzing {len(claims)} claims from {len(claim_ids)} provided claim_ids")
            else:
                # Fallback to Topic traversal
                claims = self._get_claims_for_topic(session, topic, domain, limit=limit)
                self.logger.info(f"Analyzing {len(claims)} claims from Topic traversal")
            
            if not claims:
                self.logger.warning(f"No claims found for topic: {topic}")
                return self._empty_result(topic)
            
            self.logger.info(f"Found {len(claims)} claims to analyze")
            
            # Calculate each signal
            signals = {}
            
            # Signal 1: Temporal Clustering
            signals['temporal_clustering'] = self._calculate_temporal_clustering(claims)
            
            # Signal 2: Language Similarity (requires embeddings)
            signals['language_similarity'] = self._calculate_language_similarity(
                session, claims
            )
            
            # Signal 3: Citation Cartel
            signals['citation_cartel'] = self._calculate_citation_cartel(
                session, claims
            )
            
            # Signal 4: Synchronized Emotional Triggers
            signals['synchronized_emotion'] = self._calculate_synchronized_emotion(claims)
            
            # Signal 5: Source Centralization
            signals['source_centralization'] = self._calculate_source_centralization(
                session, claims
            )
            
            # Aggregate signals
            coordination_score, confidence = self._aggregate_signals(signals)
            
            # Generate interpretation
            interpretation = self._interpret_score(coordination_score, signals)
            
            # Classify severity
            if coordination_score >= 0.8:
                severity = "CRITICAL"
            elif coordination_score >= 0.6:
                severity = "HIGH"
            elif coordination_score >= 0.4:
                severity = "MODERATE"
            elif coordination_score >= 0.2:
                severity = "LOW"
            else:
                severity = "MINIMAL"
            
            # Collect representative claims for UI display
            # Prioritize claims with temporal data (useful for coordination analysis)
            coordinated_claims = []
            for claim in sorted(claims, key=lambda c: 1 if c.get('temporal_data') else 0, reverse=True)[:limit]:
                coordinated_claims.append({
                    'id': claim.get('claim_id') or claim.get('element_id'),
                    'claim_text': claim.get('claim_text', ''),
                    'source_file': claim.get('source_file', ''),
                    'temporal_data': claim.get('temporal_data')
                })
            
            return {
                'topic': topic,
                'coordination_score': round(coordination_score, 3),
                'severity': severity,
                'confidence': round(confidence, 3),
                'signals': signals,
                'interpretation': interpretation,
                'claims_analyzed': len(claims),
                'claim_ids_received': len(claim_ids) if claim_ids else 0,
                'coordinated_claims': coordinated_claims
            }

    def _get_claims_by_ids(self, session, claim_ids: List[str]) -> List[Dict]:
        """
        Retrieve specific claims by their IDs from pattern search.
        
        Handles both:
        - Element IDs (Neo4j 5.x): "4:61206b3e-bcf2-4158-b011-62db60fba4ff:0"
        - Claim IDs (string): "claim_29501469f5d1"
        """
        
        if not claim_ids:
            return []
        
        # Limit to avoid memory issues
        max_claims = 500
        if len(claim_ids) > max_claims:
            self.logger.warning(f"Limiting from {len(claim_ids)} to {max_claims} claims")
            claim_ids = claim_ids[:max_claims]
        
        # Query that handles both ID formats
        query = """
        MATCH (c:Claim)
        WHERE elementId(c) IN $claim_ids OR c.claim_id IN $claim_ids
        RETURN elementId(c) as element_id,
               c.claim_id as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.source_file as source_file,
               c.temporal_data as temporal_data,
               c.emotional_data as emotional_data,
               c.created_at as created_at
        """
        
        result = session.run(query, claim_ids=claim_ids)
        
        claims = []
        for record in result:
            claim = dict(record)
            
            # Parse temporal_data JSON if present
            if claim.get('temporal_data'):
                try:
                    claim['temporal_data'] = json.loads(claim['temporal_data'])
                except:
                    claim['temporal_data'] = {}
            
            # Parse emotional_data JSON if present
            if claim.get('emotional_data'):
                try:
                    claim['emotional_data'] = json.loads(claim['emotional_data'])
                except:
                    claim['emotional_data'] = {}
            
            claims.append(claim)
        
        self.logger.info(f"Retrieved {len(claims)} claims from {len(claim_ids)} IDs")
        return claims

    def _get_claims_for_topic(self, 
                              session,
                              topic: str,
                              domain: Optional[str] = None,
                              limit: int = 500) -> List[Dict]:
        """
        Retrieve claims using Topic node traversal.
        
        Uses Topic node traversal (not text CONTAINS) to capture ALL claims
        from documents about the topic, including those using pronouns.
        
        Args:
            session: Neo4j session
            topic: Topic name to search
            domain: Optional domain filter
            limit: Maximum claims to return
            
        Returns:
            List of claim dicts
        """
        
        self.logger.info(f"Fetching claims for topic '{topic}' using Topic traversal...")
        
        # Try embedding-based topic expansion first
        expanded_topics = expand_topics(topic, threshold=0.35)
        
        if expanded_topics:
            self.logger.info(f"Expanded topics: {expanded_topics}")
            
            # Get claims through Topic node traversal
            claims = get_claims_via_topics_hybrid(
                session,
                topic,  # Fixed: pass string not list
                limit=limit
            )
            
            if claims:
                self.logger.info(f"Found {len(claims)} claims via Topic traversal")
                return claims
        
        # Fallback to direct text search if no Topic nodes found
        self.logger.info("Falling back to direct text search...")
        
        query = """
        MATCH (c:Claim)
        WHERE toLower(c.claim_text) CONTAINS toLower($topic)
        """
        
        if domain:
            query += " AND c.domain = $domain"
        
        query += f"""
        RETURN elementId(c) as element_id,
               c.claim_id as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.source_file as source_file,
               c.temporal_data as temporal_data,
               c.emotional_data as emotional_data,
               c.created_at as created_at
        LIMIT {limit}
        """
        
        result = session.run(query, topic=topic, domain=domain)
        
        claims = []
        for record in result:
            claim = dict(record)
            
            if claim.get('temporal_data'):
                try:
                    claim['temporal_data'] = json.loads(claim['temporal_data'])
                except:
                    claim['temporal_data'] = {}
            
            if claim.get('emotional_data'):
                try:
                    claim['emotional_data'] = json.loads(claim['emotional_data'])
                except:
                    claim['emotional_data'] = {}
            
            claims.append(claim)
        
        return claims
    
    def _calculate_temporal_clustering(self, claims: List[Dict]) -> Dict:
        """
        Calculate temporal clustering signal
        
        Detects if many claims appeared in a narrow time window
        """
        
        # Extract dates from temporal_data
        dates = []
        for claim in claims:
            temporal = claim.get('temporal_data', {})
            
            # Handle absolute_dates array format
            if temporal.get('absolute_dates'):
                for date_obj in temporal['absolute_dates']:
                    date_str = date_obj.get('date')
                    if date_str:
                        try:
                            # Try various date formats
                            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y']:
                                try:
                                    date = datetime.strptime(date_str.split('T')[0], fmt)
                                    dates.append(date)
                                    break
                                except ValueError:
                                    continue
                        except:
                            pass
            
            # Also check legacy 'dates' format
            elif temporal.get('dates'):
                for date_obj in temporal['dates']:
                    date_str = date_obj.get('date')
                    if date_str:
                        try:
                            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            dates.append(date)
                        except:
                            pass
        
        if len(dates) < 3:
            return {
                'score': 0.0,
                'burst_detected': False,
                'dates_analyzed': len(dates),
                'interpretation': f'Insufficient temporal data (only {len(dates)} dates found)'
            }
        
        # Sort dates
        dates.sort()
        
        # Find largest cluster within threshold
        max_cluster_size = 0
        max_cluster_window = None
        
        for i, start_date in enumerate(dates):
            end_date = start_date + timedelta(days=self.BURST_THRESHOLD_DAYS)
            cluster = [d for d in dates if start_date <= d <= end_date]
            
            if len(cluster) > max_cluster_size:
                max_cluster_size = len(cluster)
                max_cluster_window = (start_date, end_date)
        
        # Calculate score
        cluster_ratio = max_cluster_size / len(dates)
        
        # Score based on cluster size AND ratio
        # High ratio is a strong signal even with fewer claims
        if cluster_ratio >= 0.8:
            # Very high concentration - likely coordinated regardless of count
            if max_cluster_size >= 20:
                score = 0.9
            elif max_cluster_size >= 10:
                score = 0.8
            elif max_cluster_size >= 5:
                score = 0.7
            else:
                score = 0.5
        elif cluster_ratio >= 0.6:
            # High concentration
            if max_cluster_size >= 30:
                score = 0.8
            elif max_cluster_size >= 15:
                score = 0.6
            elif max_cluster_size >= 5:
                score = 0.5
            else:
                score = 0.3
        elif max_cluster_size >= 50 and cluster_ratio > 0.3:
            score = 0.9
        elif max_cluster_size >= 25 and cluster_ratio > 0.3:
            score = 0.8  # 25+ claims at 30%+ = very coordinated
        elif max_cluster_size >= 15 and cluster_ratio > 0.25:
            score = 0.7  # 15+ claims at 25%+ = coordinated
        elif max_cluster_size >= 10 and cluster_ratio > 0.2:
            score = 0.6
        elif max_cluster_size >= 5:
            score = 0.4 + (cluster_ratio * 0.5)  # Scale with ratio
        else:
            score = 0.2 * cluster_ratio
        
        score = min(score, 1.0)  # Cap at 1.0
        burst_detected = score >= 0.5
        
        interpretation = self._interpret_temporal_clustering(
            max_cluster_size, len(dates), max_cluster_window, burst_detected
        )
        
        return {
            'score': round(score, 3),
            'burst_detected': burst_detected,
            'cluster_size': max_cluster_size,
            'total_dates': len(dates),
            'cluster_ratio': round(cluster_ratio, 3),
            'window_start': max_cluster_window[0].isoformat() if max_cluster_window else None,
            'window_end': max_cluster_window[1].isoformat() if max_cluster_window else None,
            'interpretation': interpretation
        }
    
    def _calculate_language_similarity(self,
                                       session,
                                       claims: List[Dict]) -> Dict:
        """
        Calculate language similarity signal using pgvector embeddings
        
        High similarity across DIFFERENT sources suggests copy-paste coordination.
        Same-source similarity is expected and ignored.
        """
        
        if len(claims) < 2:
            return {
                'score': 0,
                'avg_similarity': 0,
                'high_similarity_pairs': 0,
                'interpretation': 'Need at least 2 claims for similarity analysis'
            }
        
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(**self.pg_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get claim IDs
            claim_ids = [c.get('claim_id') for c in claims if c.get('claim_id')]
            
            if len(claim_ids) < 2:
                conn.close()
                return {
                    'score': 0,
                    'avg_similarity': 0,
                    'high_similarity_pairs': 0,
                    'interpretation': 'Insufficient claim IDs for similarity analysis'
                }
            
            # Calculate pairwise similarities using pgvector
            # Only compare claims from DIFFERENT sources
            query = """
                WITH claim_pairs AS (
                    SELECT 
                        a.claim_id as id_a,
                        b.claim_id as id_b,
                        a.source_file as source_a,
                        b.source_file as source_b,
                        1 - (a.embedding <=> b.embedding) as similarity
                    FROM claim_embeddings a
                    CROSS JOIN claim_embeddings b
                    WHERE a.claim_id = ANY(%s)
                      AND b.claim_id = ANY(%s)
                      AND a.claim_id < b.claim_id
                      AND a.source_file != b.source_file
                )
                SELECT 
                    COUNT(*) as total_pairs,
                    COUNT(CASE WHEN similarity > 0.85 THEN 1 END) as high_similarity,
                    COUNT(CASE WHEN similarity > 0.70 THEN 1 END) as medium_similarity,
                    COALESCE(AVG(similarity), 0) as avg_similarity,
                    COALESCE(MAX(similarity), 0) as max_similarity
                FROM claim_pairs
            """
            
            cur.execute(query, (claim_ids, claim_ids))
            result = cur.fetchone()
            conn.close()
            
            if not result or result['total_pairs'] == 0:
                return {
                    'score': 0,
                    'avg_similarity': 0,
                    'high_similarity_pairs': 0,
                    'interpretation': 'No cross-source pairs found for comparison'
                }
            
            total_pairs = result['total_pairs']
            high_sim_pairs = result['high_similarity'] or 0
            medium_sim_pairs = result['medium_similarity'] or 0
            avg_similarity = float(result['avg_similarity'] or 0)
            max_similarity = float(result['max_similarity'] or 0)
            
            # Calculate score using ABSOLUTE thresholds
            # 78 near-identical phrases across different sources = coordination!
            # Don't dilute by dividing by all possible pairs
            
            # Thresholds based on absolute counts
            if high_sim_pairs >= 50:
                score = 0.9  # Strong coordination signal
            elif high_sim_pairs >= 20:
                score = 0.7
            elif high_sim_pairs >= 10:
                score = 0.5
            elif high_sim_pairs >= 5:
                score = 0.3
            elif medium_sim_pairs >= 50:
                score = 0.5
            elif medium_sim_pairs >= 20:
                score = 0.3
            else:
                # Fall back to ratio-based for small datasets
                score = min(avg_similarity, 0.2)
            
            # Boost if max similarity is very high (near-verbatim copying)
            if max_similarity > 0.95:
                score = min(score + 0.1, 1.0)
            
            if high_sim_pairs >= 20:
                interpretation = f"COORDINATED language: {high_sim_pairs} near-identical phrases across sources"
            elif high_sim_pairs >= 5:
                interpretation = f"Suspicious similarity: {high_sim_pairs} highly similar phrases across sources"
            elif medium_sim_pairs >= 20:
                interpretation = f"Moderate coordination: {medium_sim_pairs} similar phrases across sources"
            else:
                interpretation = f"Normal variation: avg similarity {avg_similarity:.1%}"
            
            return {
                'score': round(score, 3),
                'avg_similarity': round(avg_similarity, 3),
                'max_similarity': round(max_similarity, 3),
                'high_similarity_pairs': high_sim_pairs,
                'medium_similarity_pairs': medium_sim_pairs,
                'total_pairs': total_pairs,
                'interpretation': interpretation
            }
            
        except Exception as e:
            self.logger.warning(f"Language similarity calculation failed: {e}")
            return {
                'score': 0,
                'avg_similarity': 0,
                'high_similarity_pairs': 0,
                'interpretation': f'Error calculating similarity: {str(e)[:50]}'
            }
    
    def _calculate_citation_cartel(self,
                                   session,
                                   claims: List[Dict]) -> Dict:
        """
        Calculate citation cartel signal
        
        Detects hub-spoke citation patterns
        """
        
        # Get claim IDs
        claim_ids = [c.get('claim_id') for c in claims if c.get('claim_id')]
        
        if not claim_ids:
            return {
                'score': 0.0,
                'cartel_detected': False,
                'interpretation': 'No claim IDs available for citation analysis'
            }
        
        # Query citation patterns
        query = """
        MATCH (c:Claim)-[:CITES]->(cited:Claim)
        WHERE c.claim_id IN $claim_ids
        RETURN c.claim_id as citing_claim,
               cited.claim_id as cited_claim,
               cited.source_file as cited_source
        """
        
        result = session.run(query, claim_ids=claim_ids)
        citations = list(result)
        
        if len(citations) < 3:
            return {
                'score': 0.0,
                'cartel_detected': False,
                'total_citations': len(citations),
                'interpretation': 'Insufficient citation data'
            }
        
        # Count citations to each source
        cited_sources = Counter(r['cited_source'] for r in citations if r.get('cited_source'))
        
        if not cited_sources:
            return {
                'score': 0.0,
                'cartel_detected': False,
                'interpretation': 'No citation targets found'
            }
        
        most_cited_source, most_cited_count = cited_sources.most_common(1)[0]
        total_citations = len(citations)
        concentration_ratio = most_cited_count / total_citations
        
        if concentration_ratio > 0.6:
            score = 0.9
        elif concentration_ratio > 0.4:
            score = 0.6
        elif concentration_ratio > 0.2:
            score = 0.3
        else:
            score = 0.1
        
        cartel_detected = concentration_ratio > 0.4
        
        return {
            'score': round(score, 3),
            'cartel_detected': cartel_detected,
            'most_cited_source': most_cited_source,
            'most_cited_count': most_cited_count,
            'total_citations': total_citations,
            'concentration_ratio': round(concentration_ratio, 3),
            'interpretation': self._interpret_citation_cartel(most_cited_count, total_citations, cartel_detected)
        }
    
    def _calculate_synchronized_emotion(self, claims: List[Dict]) -> Dict:
        """
        Calculate synchronized emotional triggers signal
        
        Uses LLM-extracted emotional_data when available, falls back to keywords.
        
        Emotional data structure:
        {
            "primary_sentiment": "fear" | "anger" | "neutral",
            "emotional_intensity": 0.0-1.0,
            "manipulation_indicators": {
                "appeals_to_fear": true/false,
                "appeals_to_anger": true/false,
                "urgency_without_evidence": true/false
            }
        }
        """
        
        fear_claims = []
        anger_claims = []
        urgency_claims = []
        high_intensity_claims = []
        
        for claim in claims:
            text = claim.get('claim_text', '').lower()
            emotional_data = claim.get('emotional_data', {})
            
            # Parse JSON if it's a string
            if isinstance(emotional_data, str):
                try:
                    emotional_data = json.loads(emotional_data)
                except:
                    emotional_data = {}
            
            # Check if we have extracted emotional data
            has_emotional_data = bool(emotional_data and emotional_data.get('primary_sentiment'))
            
            if has_emotional_data:
                # Use LLM-extracted emotional analysis
                sentiment = emotional_data.get('primary_sentiment', '').lower()
                intensity = emotional_data.get('emotional_intensity', 0)
                indicators = emotional_data.get('manipulation_indicators', {})
                
                # Track high intensity claims
                if intensity >= 0.7:
                    high_intensity_claims.append(claim)
                
                # Check for fear
                if (sentiment == 'fear' or 
                    indicators.get('appeals_to_fear', False)):
                    fear_claims.append(claim)
                
                # Check for anger
                if (sentiment == 'anger' or 
                    indicators.get('appeals_to_anger', False)):
                    anger_claims.append(claim)
                
                # Check for urgency
                if indicators.get('urgency_without_evidence', False):
                    urgency_claims.append(claim)
                    
            else:
                # Fallback to keyword matching if no emotional_data
                if any(kw in text for kw in self.FEAR_KEYWORDS):
                    fear_claims.append(claim)
                
                if any(kw in text for kw in self.ANGER_KEYWORDS):
                    anger_claims.append(claim)
                
                if any(kw in text for kw in self.URGENCY_KEYWORDS):
                    urgency_claims.append(claim)
        
        total = len(claims) if claims else 1
        
        fear_ratio = len(fear_claims) / total
        anger_ratio = len(anger_claims) / total
        urgency_ratio = len(urgency_claims) / total
        intensity_ratio = len(high_intensity_claims) / total
        
        # Score: High usage of emotional manipulation = coordination
        # Take the MAX of strong signals, don't average them down
        # Fear + Urgency together is yellow journalism signature
        fear_urgency_combo = max(fear_ratio, urgency_ratio) + min(fear_ratio, urgency_ratio) * 0.5
        emotion_score = max(
            fear_urgency_combo,  # Fear+urgency combo
            fear_ratio * 1.2,   # Strong fear alone
            anger_ratio * 1.3,  # Anger is very strong signal
            (fear_ratio * 0.35 + anger_ratio * 0.35 + urgency_ratio * 0.2 + intensity_ratio * 0.1)  # Original weighted
        )
        
        # Boost if multiple emotion types present (coordinated multi-vector attack)
        types_present = sum([fear_ratio > 0.2, anger_ratio > 0.2, urgency_ratio > 0.2])
        if types_present >= 2:
            emotion_score *= 1.5  # Increased from 1.3 - multiple emotions = stronger signal
        
        # Boost if high intensity across claims
        if intensity_ratio > 0.5:
            emotion_score *= 1.2
        
        score = min(emotion_score, 1.0)
        
        synchronized = score > 0.4  # Lowered from 0.6 - extraction is more accurate
        
        interpretation = self._interpret_synchronized_emotion(
            fear_ratio, anger_ratio, urgency_ratio, synchronized
        )
        
        return {
            'score': round(score, 3),
            'synchronized': synchronized,
            'fear_ratio': round(fear_ratio, 3),
            'anger_ratio': round(anger_ratio, 3),
            'urgency_ratio': round(urgency_ratio, 3),
            'fear_claims': len(fear_claims),
            'anger_claims': len(anger_claims),
            'urgency_claims': len(urgency_claims),
            'high_intensity_claims': len(high_intensity_claims),
            'interpretation': interpretation
        }
    
    def _calculate_source_centralization(self,
                                        session,
                                        claims: List[Dict]) -> Dict:
        """
        Calculate source centralization signal
        
        Detects if many sources trace back to single origin
        """
        
        # Count unique sources
        sources = [c.get('source_file') for c in claims if c.get('source_file')]
        source_counts = Counter(sources)
        
        if len(source_counts) == 0:
            return {
                'score': 0.0,
                'centralized': False,
                'interpretation': 'No source data available'
            }
        
        # Calculate centralization
        total_claims = len(sources)
        unique_sources = len(source_counts)
        
        # Most common source
        most_common_source, most_common_count = source_counts.most_common(1)[0]
        
        # Centralization metrics
        concentration_ratio = most_common_count / total_claims
        source_diversity = unique_sources / total_claims
        
        # Score: High concentration in few sources = centralization
        if concentration_ratio > 0.5:
            score = 0.8
        elif concentration_ratio > 0.3:
            score = 0.6
        elif source_diversity < 0.2:  # Few unique sources
            score = 0.5
        else:
            score = concentration_ratio * 0.5
        
        centralized = concentration_ratio > 0.4 or source_diversity < 0.25
        
        interpretation = self._interpret_source_centralization(
            unique_sources, total_claims, most_common_count, centralized
        )
        
        return {
            'score': round(score, 3),
            'centralized': centralized,
            'unique_sources': unique_sources,
            'total_claims': total_claims,
            'concentration_ratio': round(concentration_ratio, 3),
            'top_source': most_common_source,
            'top_source_count': most_common_count,
            'interpretation': interpretation
        }
    
    def _aggregate_signals(self, signals: Dict) -> Tuple[float, float]:
        """Aggregate signals into coordination score"""
        
        score = 0.0
        total_weight = 0.0
        signals_with_data = 0
        
        for signal_name, weight in self.WEIGHTS.items():
            signal = signals.get(signal_name, {})
            signal_score = signal.get('score', 0.0)
            interpretation = signal.get('interpretation', '')
            
            # Skip signals with insufficient data
            if 'Insufficient' in interpretation or 'No data' in interpretation:
                continue
            
            score += signal_score * weight
            total_weight += weight
            
            # Count signals with real data
            if signal_score > 0.0:
                signals_with_data += 1
        
        # Normalize by actual weights used
        if total_weight > 0:
            score = score / total_weight
        
        # Confidence based on data availability
        confidence = signals_with_data / len(self.WEIGHTS) if signals_with_data > 0 else 0.5
        
        return score, confidence
    
    def _interpret_score(self, score: float, signals: Dict) -> str:
        """Generate interpretation of coordination score"""
        
        if score >= 0.8:
            level = "STRONG coordination pattern detected"
        elif score >= 0.6:
            level = "MODERATE coordination pattern detected"
        elif score >= 0.4:
            level = "WEAK coordination signals present"
        elif score >= 0.2:
            level = "MINIMAL coordination indicators"
        else:
            level = "NO significant coordination detected"
        
        # Identify strongest signals
        signal_scores = {
            name: data.get('score', 0.0)
            for name, data in signals.items()
        }
        
        strongest = sorted(signal_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        signal_names = [name.replace('_', ' ').title() for name, _ in strongest]
        
        interpretation = f"{level}. "
        interpretation += f"Strongest signals: {', '.join(signal_names)}."
        
        return interpretation
    
    # Interpretation helpers
    
    def _interpret_temporal_clustering(self, cluster_size, total, window, burst):
        if burst:
            return f"Temporal burst detected: {cluster_size} claims in {self.BURST_THRESHOLD_DAYS} days ({cluster_size/total:.1%} of total)"
        else:
            return f"No significant burst: {cluster_size} claims in window ({cluster_size/total:.1%})"
    
    def _interpret_citation_cartel(self, count, total, cartel):
        if cartel:
            return f"Citation cartel pattern: {count}/{total} citations concentrated"
        else:
            return f"Normal citation pattern: {count}/{total} to most cited"
    
    def _interpret_synchronized_emotion(self, fear, anger, urgency, sync):
        if sync:
            return f"Synchronized emotional triggers: Fear {fear:.1%}, Anger {anger:.1%}, Urgency {urgency:.1%}"
        else:
            return f"Normal emotional distribution: Fear {fear:.1%}, Anger {anger:.1%}, Urgency {urgency:.1%}"
    
    def _interpret_source_centralization(self, unique, total, top_count, central):
        if central:
            return f"Centralized sources: {unique} sources for {total} claims, top source has {top_count} ({top_count/total:.1%})"
        else:
            return f"Diverse sources: {unique} sources for {total} claims"
    
    def _empty_result(self, topic: str) -> Dict:
        """Return empty result when no data available"""
        return {
            'topic': topic,
            'coordination_score': 0.0,
            'severity': 'NONE',
            'confidence': 0.0,
            'signals': {},
            'interpretation': 'No data available for analysis',
            'claims_analyzed': 0,
            'claim_ids_received': 0,
            'coordinated_claims': []
        }
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")


def main():
    """Test coordination detector"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    detector = CoordinationDetector()
    
    # Test topics
    test_topics = [
        "Spanish American War",
        "Butler",
        "treatment"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {topic}")
        print('='*80)
        
        result = detector.detect_coordination(topic)
        
        print(f"\nCoordination Score: {result['coordination_score']:.3f}")
        print(f"Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Claims Analyzed: {result['claims_analyzed']}")
        print(f"\nInterpretation: {result['interpretation']}")
        
        print(f"\nSignal Breakdown:")
        for signal_name, signal_data in result['signals'].items():
            print(f"\n  {signal_name}:")
            print(f"    Score: {signal_data.get('score', 0.0):.3f}")
            print(f"    {signal_data.get('interpretation', 'N/A')}")
    
    detector.close()
    
    print(f"\n{'='*80}")
    print("DETECTION COMPLETE")
    print('='*80)


if __name__ == "__main__":
    main()
