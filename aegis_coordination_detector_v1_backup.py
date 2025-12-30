#!/usr/bin/env python3
"""
Aegis Insight - Coordination Pattern Detector

Detects manufactured consensus through synchronized messaging:
1. Temporal Clustering - Claims appearing in narrow time windows
2. Language Similarity - Near-identical wording across sources
3. Citation Cartel - Hub-spoke network patterns
4. Synchronized Emotional Triggers - Coordinated fear/anger appeals
5. Source Centralization - Many sources tracing to single origin

Author: Aegis Insight Team
Date: November 2025
"""

import json
from aegis_topic_utils import expand_topics, get_claims_via_topics_hybrid
import logging
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
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
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "aegistrusted",
                 logger: Optional[logging.Logger] = None):
        """Initialize coordination detector"""
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Neo4j connection
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
                            claim_ids: Optional[List[int]] = None,
                            domain: Optional[str] = None,
                            time_range: Optional[Tuple[str, str]] = None,
                            limit: int = 500) -> Dict:
        """
        Detect coordination patterns for a given topic
        
        Args:
                    topic: Topic to analyze
                    claim_ids: Optional list of claim IDs from semantic search.
                              If provided, analyzes only these claims.
                              If None, falls back to text search (legacy behavior).
                    domain: Optional domain filter
                    time_range: Optional (start_date, end_date) tuple
            
        Returns:
            Dict with coordination analysis
        """
        
        self.logger.info(f"Analyzing coordination patterns for topic: {topic}")

        with self.driver.session() as session:
            # Use Topic traversal (same as suppression detector)
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
                'coordinated_claims': coordinated_claims
            }


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
            self.logger.info(f"Topic expansion found {len(expanded_topics)} topics: {expanded_topics[:5]}...")
            
            # Use hybrid search with text filtering for better precision
            from aegis_topic_utils import get_claims_via_topics_hybrid
            hybrid_claims = get_claims_via_topics_hybrid(session, topic, threshold=0.35, limit=limit)
            
            if hybrid_claims:
                self.logger.info(f"Hybrid search found {len(hybrid_claims)} precise claims")
                # Convert to dict format expected by rest of method
                claims = {}
                for claim in hybrid_claims:
                    element_id = claim.get('claim_id', str(id(claim)))
                    if element_id not in claims:
                        claims[element_id] = claim
                return list(claims.values())[:limit]
            
            # Fallback to broad topic search if hybrid returns nothing
            topic_words = None
            query = """
            MATCH (t:Topic)
            WHERE t.name IN $topic_names
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.temporal_data as temporal_data,
                   c.geographic_data as geographic_data,
                   d.source_file as doc_source,
                   elementId(c) as element_id
            ORDER BY c.confidence DESC
            LIMIT $limit
            """
        else:
            # Fallback to word matching
            normalized_topic = topic.replace('-', ' ').replace('  ', ' ')
            topic_words = [w.strip() for w in normalized_topic.split() if len(w.strip()) > 2]
            query = """
            MATCH (t:Topic)
            WHERE ALL(word IN $topic_words WHERE toLower(t.name) CONTAINS toLower(word))
            MATCH (t)<-[:ABOUT]-(d:Document)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   c.temporal_data as temporal_data,
                   c.geographic_data as geographic_data,
                   d.source_file as doc_source,
                   elementId(c) as element_id
            ORDER BY c.confidence DESC
            LIMIT $limit
            """
        
        claims = {}
        
        try:
            if expanded_topics:
                result = session.run(query, topic_names=expanded_topics, limit=limit)
            else:
                result = session.run(query, topic_words=topic_words, limit=limit)
            
            for record in result:
                claim = dict(record)
                element_id = claim.get('element_id')
                if element_id and element_id not in claims:
                    if not claim.get('source_file'):
                        claim['source_file'] = claim.get('doc_source')
                    # Parse temporal_data JSON
                    if claim.get('temporal_data'):
                        try:
                            claim['temporal_data'] = json.loads(claim['temporal_data']) if isinstance(claim['temporal_data'], str) else claim['temporal_data']
                        except:
                            claim['temporal_data'] = {}
                    claim['from_topic_traversal'] = True
                    claims[element_id] = claim
            
            self.logger.info(f"Topic traversal found {len(claims)} claims")
            
        except Exception as e:
            self.logger.warning(f"Topic traversal failed: {e}, falling back to text search")
            return self._get_claims_by_text(session, topic, domain, limit)
        
        # Fallback if no results
        if len(claims) == 0:
            self.logger.info("No claims via Topic traversal, trying text search fallback...")
            return self._get_claims_by_text(session, topic, domain, limit)
        
        return list(claims.values())


    def _get_claims_by_text(self,
                            session,
                            topic: str,
                            domain: Optional[str],
                            limit: int = 500) -> List[Dict]:
        """
        LEGACY METHOD: Retrieve claims using text search

        Uses simple CONTAINS matching. Less accurate than semantic search.
        """
        
        query = """
        MATCH (c:Claim)
        WHERE toLower(c.claim_text) CONTAINS toLower($topic)
        """
        
        if domain:
            query += " AND c.domain = $domain"
        
        query += """
        RETURN c.claim_id as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.source_file as source_file,
               c.temporal_data as temporal_data,
               c.created_at as created_at
               ORDER BY c.confidence DESC
               LIMIT $limit
        """
        
        result = session.run(query, topic=topic, domain=domain, limit=limit)

        
        claims = []
        for record in result:
            claim = dict(record)
            
            # Parse temporal_data JSON
            if claim.get('temporal_data'):
                try:
                    claim['temporal_data'] = json.loads(claim['temporal_data'])
                except:
                    claim['temporal_data'] = {}
            
            claims.append(claim)
        
        return claims

    def _get_claims_by_ids(self,
                           session,
                           claim_ids: List[int]) -> List[Dict]:
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
               c.source_file as source_file,
               c.temporal_data as temporal_data,
               c.created_at as created_at
               ORDER BY c.confidence DESC
               LIMIT $limit
        """

        result = session.run(query, claim_ids=claim_ids, limit=self.config.max_claims_analyzed)

        claims = []
        for record in result:
            claim = dict(record)

            # Parse JSON field
            if claim.get('temporal_data'):
                try:
                    claim['temporal_data'] = json.loads(claim['temporal_data']) if isinstance(claim['temporal_data'],
                                                                                              str) else claim[
                        'temporal_data']
                except:
                    claim['temporal_data'] = {}

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
            temporal = claim.get('temporal_data') or {}
            
            # Try to get date from temporal_data (schema: absolute_dates)
            if temporal.get('absolute_dates'):
                for date_obj in temporal['absolute_dates']:
                    date_str = date_obj.get('date')
                    if date_str:
                        try:
                            # Handle year-only (e.g., "1876") or full ISO format
                            if len(date_str) == 4 and date_str.isdigit():
                                date = datetime(int(date_str), 6, 15)  # Mid-year estimate
                            elif 'BCE' in date_str:
                                continue  # Skip BCE dates for clustering
                            else:
                                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            dates.append(date)
                        except:
                            pass
        
        if len(dates) < 3:
            return {
                'score': 0.0,
                'burst_detected': False,
                'dates_analyzed': len(dates),
                'interpretation': 'Insufficient temporal data'
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
        
        # Score based on cluster size and ratio
        if max_cluster_size >= 50 and cluster_ratio > 0.5:
            score = 0.9
        elif max_cluster_size >= 30 and cluster_ratio > 0.4:
            score = 0.7
        elif max_cluster_size >= 20 and cluster_ratio > 0.3:
            score = 0.5
        elif max_cluster_size >= 10 and cluster_ratio > 0.2:
            score = 0.3
        else:
            score = cluster_ratio * 0.5  # Lower scores for smaller clusters
        
        burst_detected = max_cluster_size >= 20
        
        interpretation = self._interpret_temporal_clustering(
            max_cluster_size, len(dates), max_cluster_window, burst_detected
        )
        
        return {
            'score': round(score, 3),
            'burst_detected': burst_detected,
            'cluster_size': max_cluster_size,
            'total_claims': len(dates),
            'cluster_ratio': round(cluster_ratio, 3),
            'window_start': max_cluster_window[0].isoformat() if max_cluster_window else None,
            'window_end': max_cluster_window[1].isoformat() if max_cluster_window else None,
            'interpretation': interpretation
        }
    
    def _calculate_language_similarity(self,
                                      session,
                                      claims: List[Dict]) -> Dict:
        """
        Calculate language similarity signal using embeddings
        
        High similarity across sources suggests copy-paste coordination
        """
        
        # Note: This requires embeddings and PostgreSQL integration
        # For now, return simplified version
        
        # TODO: Query PostgreSQL for embeddings
        # TODO: Calculate pairwise cosine similarity
        # TODO: Identify clusters of similar claims
        
        # Placeholder implementation
        return {
            'score': 0.0,
            'avg_similarity': 0.0,
            'high_similarity_pairs': 0,
            'interpretation': 'Language similarity analysis requires embeddings (not yet integrated)',
            'note': 'Feature pending - requires PostgreSQL embeddings integration'
        }
    
    def _calculate_citation_cartel(self,
                                  session,
                                  claims: List[Dict]) -> Dict:
        """
        Calculate citation cartel signal
        
        Detects hub-spoke patterns where core group cites each other
        but ignores outsiders
        """
        
        claim_ids = [c['claim_id'] for c in claims]
        
        if not claim_ids:
            return {
                'score': 0.0,
                'cartel_detected': False,
                'interpretation': 'No claims to analyze'
            }
        
        # Get citation network
        query = """
        MATCH (c1:Claim)-[r:CITES]->(c2:Claim)
        WHERE c1.claim_id IN $claim_ids
        RETURN c1.claim_id as source,
               c2.claim_id as target,
               c2.claim_id IN $claim_ids as is_internal
        """
        
        result = session.run(query, claim_ids=claim_ids)
        
        # Build citation graph
        internal_citations = defaultdict(int)
        external_citations = defaultdict(int)
        
        for record in result:
            source = record['source']
            is_internal = record['is_internal']
            
            if is_internal:
                internal_citations[source] += 1
            else:
                external_citations[source] += 1
        
        # Calculate cartel metrics
        if not internal_citations and not external_citations:
            return {
                'score': 0.0,
                'cartel_detected': False,
                'interpretation': 'No citation data available'
            }
        
        # Average internal vs external citations
        avg_internal = (sum(internal_citations.values()) / len(internal_citations)
                       if internal_citations else 0)
        avg_external = (sum(external_citations.values()) / len(external_citations)
                       if external_citations else 0)
        
        # Cartel score: high internal, low external = cartel pattern
        if avg_internal > 0:
            cartel_ratio = avg_internal / (avg_internal + avg_external) if (avg_internal + avg_external) > 0 else 0
        else:
            cartel_ratio = 0
        
        # Detection threshold
        cartel_detected = (
            avg_internal >= 5 and  # Core group cites each other frequently
            cartel_ratio > 0.7     # Mostly internal citations
        )
        
        score = cartel_ratio if cartel_detected else cartel_ratio * 0.5
        
        interpretation = self._interpret_citation_cartel(
            avg_internal, avg_external, cartel_detected
        )
        
        return {
            'score': round(score, 3),
            'cartel_detected': cartel_detected,
            'avg_internal_citations': round(avg_internal, 2),
            'avg_external_citations': round(avg_external, 2),
            'cartel_ratio': round(cartel_ratio, 3),
            'interpretation': interpretation
        }
    
    def _calculate_synchronized_emotion(self, claims: List[Dict]) -> Dict:
        """
        Calculate synchronized emotional triggers signal
        
        Detects coordinated use of fear/anger/urgency keywords
        """
        
        # Analyze emotional keywords across claims
        fear_claims = []
        anger_claims = []
        urgency_claims = []
        
        for claim in claims:
            text = claim.get('claim_text', '').lower()
            
            # Check for emotion keywords
            has_fear = any(keyword in text for keyword in self.FEAR_KEYWORDS)
            has_anger = any(keyword in text for keyword in self.ANGER_KEYWORDS)
            has_urgency = any(keyword in text for keyword in self.URGENCY_KEYWORDS)
            
            if has_fear:
                fear_claims.append(claim)
            if has_anger:
                anger_claims.append(claim)
            if has_urgency:
                urgency_claims.append(claim)
        
        total = len(claims)
        fear_ratio = len(fear_claims) / total if total > 0 else 0
        anger_ratio = len(anger_claims) / total if total > 0 else 0
        urgency_ratio = len(urgency_claims) / total if total > 0 else 0
        
        # Score: High usage of emotional keywords = coordination
        # Multiple types of emotion = stronger signal
        emotion_score = (fear_ratio + anger_ratio + urgency_ratio) / 3
        
        # Boost if multiple types present
        types_present = sum([fear_ratio > 0.2, anger_ratio > 0.2, urgency_ratio > 0.2])
        if types_present >= 2:
            emotion_score *= 1.3
        
        score = min(emotion_score, 1.0)
        
        synchronized = score > 0.6
        
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
            
            score += signal_score * weight
            total_weight += weight
            
            # Count signals with real data
            if signal_score > 0.0 and not signal.get('note'):
                signals_with_data += 1
        
        # Normalize
        if total_weight > 0:
            score = score / total_weight
        
        # Confidence based on data availability
        confidence = signals_with_data / len(self.WEIGHTS)
        
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
    
    def _interpret_citation_cartel(self, internal, external, cartel):
        if cartel:
            return f"Citation cartel detected: avg {internal:.1f} internal vs {external:.1f} external citations"
        else:
            return f"Normal citation pattern: {internal:.1f} internal, {external:.1f} external"
    
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
            'confidence': 0.0,
            'signals': {},
            'interpretation': 'No data available for analysis',
            'claims_analyzed': 0
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
        "COVID",
        "climate",
        "vaccine"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {topic}")
        print('='*80)
        
        result = detector.detect_coordination(topic)
        
        print(f"\nCoordination Score: {result['coordination_score']:.3f}")
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
