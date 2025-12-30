import os
#!/usr/bin/env python3
"""
Aegis Insight - Suppression Pattern Detector (UPDATED)

CHANGES from v1.0:
- Added claim_ids parameter to detect_suppression()
- Added _get_claims_by_ids() method for semantic search integration
- Maintains backwards compatibility with text search fallback

Detects patterns of systematic suppression by analyzing:
1. META Claim Density - Ratio of dismissal to substantive claims
2. Network Isolation - Citation asymmetry patterns
3. Evidence Avoidance - Dismissals without technical rebuttals
4. Emotional Manipulation - Fact vs emotion ratio
5. Authority Domain Mismatch - Speakers outside their expertise

Author: Aegis Insight Team
Date: November 2025
Version: 1.1
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase
from collections import defaultdict
from aegis_config import Config
from aegis_detection_config import get_config

class SuppressionDetector:
    """
    Detects suppression patterns in knowledge graphs
    
    Implements 5 detection signals:
    - META Claim Density (0.15 weight)
    - Network Isolation (0.15 weight)
    - Evidence Avoidance (0.10 weight)
    - Emotional Manipulation (0.30 weight)
    - Authority Domain Mismatch (0.30 weight)
    
    Returns suppression score 0.0-1.0 with signal breakdown
    """
    
    # Signal weights for aggregation
    WEIGHTS = {
        'meta_claim_density': 0.15,
        'network_isolation': 0.15,
        'evidence_avoidance': 0.10,
        'emotional_manipulation': 0.30,
        'authority_mismatch': 0.30
    }
    
    # Dismissal keywords for META detection
    DISMISSAL_KEYWORDS = [
        'pseudoscience', 'conspiracy', 'debunked', 'misinformation',
        'disinformation', 'fake news', 'quack', 'fringe', 'discredited',
        'baseless', 'unfounded', 'myth', 'hoax', 'false claim'
    ]
    
    def __init__(self,
                 neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize suppression detector
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Connect to Neo4j
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

    def detect_suppression(self,
                           topic: str = None,
                           query: str = None,  # Alias for topic
                           domain: Optional[str] = None,
                           time_range: Optional[Tuple[str, str]] = None,
                           claim_ids: Optional[List[str]] = None) -> Dict:
        """
        Detect suppression patterns for a given topic
        
        Args:
            topic: Topic to analyze (used for claim search or labeling)
            claim_ids: Optional list of claim IDs from semantic search.
                      If provided, analyzes only these claims.
                      If None, falls back to text search (legacy behavior).
            domain: Optional domain filter (used only with text search)
            time_range: Optional (start_date, end_date) tuple
            
        Returns:
            Dict with suppression analysis:
            {
                "suppression_score": 0.0-1.0,
                "confidence": 0.0-1.0,
                "signals": {
                    "meta_claim_density": {...},
                    "network_isolation": {...},
                    "evidence_avoidance": {...},
                    "emotional_manipulation": {...},
                    "authority_mismatch": {...}
                },
                "interpretation": str,
                "claim_counts": {
                    "total": int,
                    "primary": int,
                    "secondary": int,
                    "meta": int,
                    "contextual": int
                }
            }
        """
        
        self.logger.info(f"Analyzing suppression patterns for topic: {topic}")
        # Handle query alias (API sends 'query', code expects 'topic')
        if query and not topic:
            topic = query

        if not topic:
            return {'success': False, 'error': 'Query is required'}
        
        with self.driver.session() as session:
            # NEW: Use claim_ids if provided (from semantic search)
            if claim_ids and len(claim_ids) > 0:
                claims = self._get_claims_by_ids(session, claim_ids)
                self.logger.info(f"Analyzing {len(claims)} claims from provided IDs (semantic search)")
            else:
                # Legacy: text search fallback
                claims = self._get_claims_by_text(session, topic, domain)
                self.logger.info(f"Analyzing {len(claims)} claims from text search (legacy mode)")
            
            if not claims:
                self.logger.warning(f"No claims found for topic: {topic}")
                return self._empty_result(topic)
            
            self.logger.info(f"Found {len(claims)} claims to analyze")
            
            # Calculate each signal
            signals = {}
            
            # Signal 1: META Claim Density
            signals['meta_claim_density'] = self._calculate_meta_density(claims)
            
            # Signal 2: Network Isolation
            signals['network_isolation'] = self._calculate_network_isolation(
                session, claims
            )
            
            # Signal 3: Evidence Avoidance
            signals['evidence_avoidance'] = self._calculate_evidence_avoidance(claims)
            
            # Signal 4: Emotional Manipulation
            signals['emotional_manipulation'] = self._calculate_emotional_manipulation(claims)
            
            # Signal 5: Authority Domain Mismatch
            signals['authority_mismatch'] = self._calculate_authority_mismatch(claims)
            
            # Aggregate signals into suppression score
            suppression_score, confidence = self._aggregate_signals(signals)
            
            # Build result
            result = {
                'topic': topic,
                'suppression_score': round(suppression_score, 3),
                'confidence': round(confidence, 3),
                'claims_analyzed': len(claims),
                'signals': signals,
                'interpretation': self._interpret_score(suppression_score, signals),
                'claim_counts': self._count_claim_types(claims)
            }
            
            return result
    
    def _get_claims_by_ids(self, session, claim_ids: List[int]) -> List[Dict]:
        """
        NEW METHOD: Fetch specific claims by their Neo4j IDs
        
        This is used when claim IDs come from semantic search (pattern search).
        
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
               c.temporal_data as temporal_data,
               c.geographic_data as geographic_data,
               c.citation_data as citation_data
        ORDER BY c.confidence DESC
        LIMIT $limit
        """
        
        result = session.run(query, claim_ids=claim_ids, limit=self.config.max_claims_analyzed)
        
        claims = []
        for record in result:
            claim = dict(record)
            
            # Parse JSON fields
            for field in ['temporal_data', 'geographic_data', 'citation_data']:
                if claim.get(field):
                    try:
                        claim[field] = json.loads(claim[field]) if isinstance(claim[field], str) else claim[field]
                    except:
                        claim[field] = {}
            
            claims.append(claim)
        
        return claims
    
    def _get_claims_by_text(self, session, topic: str, domain: Optional[str] = None) -> List[Dict]:
        """
        RENAMED from _get_claims_for_topic: Legacy text search method
        
        Uses simple CONTAINS text matching. Less accurate than semantic search.
        
        Args:
            session: Neo4j session
            topic: Topic to search for
            domain: Optional domain filter
            
        Returns:
            List of claim dicts
        """
        
        query = """
        MATCH (c:Claim)
        WHERE toLower(c.claim_text) CONTAINS toLower($topic)
        """
        
        if domain:
            query += " AND c.domain = $domain"
        
        query += """
        RETURN elementId(c) as claim_id,
               c.claim_text as claim_text,
               c.claim_type as claim_type,
               c.confidence as confidence,
               c.temporal_data as temporal_data,
               c.geographic_data as geographic_data,
               c.citation_data as citation_data
               ORDER BY c.confidence DESC
               LIMIT $limit
        """
        
        result = session.run(query, topic=topic, domain=domain, limit=self.config.max_claims_analyzed)
        
        claims = []
        for record in result:
            claim = dict(record)
            
            # Parse JSON fields
            for field in ['temporal_data', 'geographic_data', 'citation_data']:
                if claim.get(field):
                    try:
                        claim[field] = json.loads(claim[field]) if isinstance(claim[field], str) else claim[field]
                    except:
                        claim[field] = {}
            
            claims.append(claim)
        
        return claims
    

    # High-precision suppression terms (empirically validated)
    HIGH_PRECISION_TERMS = [
        'suppressed', 'censored', 'banned', 'objected', 'withheld',
        'forbidden', 'unpublished', 'silenced', 'blacklisted', 'redacted',
        'classified', 'concealed', 'prohibited'
    ]
    
    def _get_claims_for_topic(self, session, topic: str, limit: int = 500) -> List[Dict]:
        """
        Two-stage prioritized claim retrieval.
        
        Stage 1: Fetch high-salience claims (META type OR high-precision terms)
        Stage 2: Fill remaining slots with random sample
        
        This ensures suppression-relevant claims bubble to the top regardless
        of corpus size. Empirically validated: 12.2% META ratio for high-precision
        vs 6.3% for noisy terms.
        
        Args:
            session: Neo4j session
            topic: Topic name to search
            limit: Maximum claims to return
            
        Returns:
            List of claim dicts, prioritized by suppression salience
        """
        
        # Stage 1: Priority claims (META + high-precision terms)
        self.logger.debug(f"Stage 1: Fetching priority claims for topic '{topic}'...")
        
        priority_query = """
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
            MATCH (d:Document)-[:ABOUT]->(t)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WITH c, d,
                 CASE WHEN c.claim_type = 'META' THEN 2 ELSE 0 END +
                 CASE WHEN any(term IN $hp_terms WHERE toLower(c.claim_text) CONTAINS term) THEN 3 ELSE 0 END
                 AS priority
            WHERE priority > 0
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   d.source_file as doc_source,
                   elementId(c) as element_id,
                   priority
            ORDER BY priority DESC
            LIMIT $limit
        """
        
        result = session.run(
            priority_query,
            topic=topic,
            hp_terms=self.HIGH_PRECISION_TERMS,
            limit=limit
        )
        
        priority_claims = []
        seen_ids = set()
        
        for record in result:
            claim = {
                'claim_id': record['claim_id'],
                'claim_text': record['claim_text'],
                'claim_type': record['claim_type'],
                'confidence': record['confidence'],
                'source_file': record['source_file'] or record['doc_source'],
                'element_id': record['element_id'],
                'priority': record['priority']
            }
            if claim['claim_id'] not in seen_ids:
                priority_claims.append(claim)
                seen_ids.add(claim['claim_id'])
        
        self.logger.info(f"Stage 1: Found {len(priority_claims)} priority claims")
        
        # If we have enough priority claims, return them
        if len(priority_claims) >= limit:
            return priority_claims[:limit]
        
        # Stage 2: Fill remaining slots with random sample
        remaining = limit - len(priority_claims)
        self.logger.debug(f"Stage 2: Filling {remaining} slots with random claims...")
        
        random_query = """
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
            MATCH (d:Document)-[:ABOUT]->(t)
            MATCH (d)-[:CONTAINS]->(ch:Chunk)-[:CONTAINS_CLAIM]->(c:Claim)
            WHERE NOT c.claim_id IN $seen_ids
            WITH c, d, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   d.source_file as doc_source,
                   elementId(c) as element_id,
                   0 as priority
        """
        
        result = session.run(
            random_query,
            topic=topic,
            seen_ids=list(seen_ids),
            limit=remaining
        )
        
        random_claims = []
        for record in result:
            claim = {
                'claim_id': record['claim_id'],
                'claim_text': record['claim_text'],
                'claim_type': record['claim_type'],
                'confidence': record['confidence'],
                'source_file': record['source_file'] or record['doc_source'],
                'element_id': record['element_id'],
                'priority': 0
            }
            if claim['claim_id'] not in seen_ids:
                random_claims.append(claim)
                seen_ids.add(claim['claim_id'])
        
        self.logger.info(f"Stage 2: Added {len(random_claims)} random claims")
        
        all_claims = priority_claims + random_claims
        self.logger.info(f"Total: {len(all_claims)} claims for analysis")
        
        return all_claims
    
    def _get_claims_for_topic_fallback(self, session, topic: str, limit: int = 500) -> List[Dict]:
        """
        Fallback: Text search when no Topic node exists.
        Uses same two-stage approach but searches claim_text directly.
        """
        
        self.logger.debug(f"Fallback: Text search for '{topic}'...")
        
        # Stage 1: Priority claims with topic in text
        priority_query = """
            MATCH (c:Claim)
            WHERE toLower(c.claim_text) CONTAINS toLower($topic)
            WITH c,
                 CASE WHEN c.claim_type = 'META' THEN 2 ELSE 0 END +
                 CASE WHEN any(term IN $hp_terms WHERE toLower(c.claim_text) CONTAINS term) THEN 3 ELSE 0 END
                 AS priority
            WHERE priority > 0
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   elementId(c) as element_id,
                   priority
            ORDER BY priority DESC
            LIMIT $limit
        """
        
        result = session.run(
            priority_query,
            topic=topic,
            hp_terms=self.HIGH_PRECISION_TERMS,
            limit=limit
        )
        
        priority_claims = []
        seen_ids = set()
        
        for record in result:
            claim = {
                'claim_id': record['claim_id'],
                'claim_text': record['claim_text'],
                'claim_type': record['claim_type'],
                'confidence': record['confidence'],
                'source_file': record['source_file'],
                'element_id': record['element_id'],
                'priority': record['priority']
            }
            if claim['claim_id'] not in seen_ids:
                priority_claims.append(claim)
                seen_ids.add(claim['claim_id'])
        
        if len(priority_claims) >= limit:
            return priority_claims[:limit]
        
        # Stage 2: Random with topic mention
        remaining = limit - len(priority_claims)
        
        random_query = """
            MATCH (c:Claim)
            WHERE toLower(c.claim_text) CONTAINS toLower($topic)
            AND NOT c.claim_id IN $seen_ids
            WITH c, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN c.claim_id as claim_id,
                   c.claim_text as claim_text,
                   c.claim_type as claim_type,
                   c.confidence as confidence,
                   c.source_file as source_file,
                   elementId(c) as element_id,
                   0 as priority
        """
        
        result = session.run(
            random_query,
            topic=topic,
            seen_ids=list(seen_ids),
            limit=remaining
        )
        
        for record in result:
            claim = {
                'claim_id': record['claim_id'],
                'claim_text': record['claim_text'],
                'claim_type': record['claim_type'],
                'confidence': record['confidence'],
                'source_file': record['source_file'],
                'element_id': record['element_id'],
                'priority': 0
            }
            if claim['claim_id'] not in seen_ids:
                priority_claims.append(claim)
                seen_ids.add(claim['claim_id'])
        
        return priority_claims

    def _calculate_meta_density(self, claims: List[Dict]) -> Dict:
        """
        Calculate META claim density signal
        
        High META density = lots of dismissal without substantive engagement
        
        Returns:
            {
                "score": 0.0-1.0,
                "meta_ratio": float,
                "dismissal_count": int,
                "primary_count": int,
                "interpretation": str
            }
        """
        
        # Count claim types
        meta_count = sum(1 for c in claims if c.get('claim_type') == 'META')
        primary_count = sum(1 for c in claims if c.get('claim_type') == 'PRIMARY')
        
        total = len(claims)
        meta_ratio = meta_count / total if total > 0 else 0.0
        
        # Count dismissal keywords in META claims
        dismissal_count = 0
        for claim in claims:
            if claim.get('claim_type') == 'META':
                text = claim.get('claim_text', '').lower()
                if any(keyword in text for keyword in self.DISMISSAL_KEYWORDS):
                    dismissal_count += 1
        
        # Score: Higher META ratio + more dismissals = higher suppression
        if meta_count == 0:
            score = 0.0
            interpretation = f"No META claims found ({primary_count} primary claims)"
        else:
            dismissal_ratio = dismissal_count / meta_count
            # Weighted combination: 60% META ratio, 40% dismissal ratio
            score = (meta_ratio * 0.6) + (dismissal_ratio * 0.4)
            
            interpretation = (f"High META density ({meta_ratio:.1%}) with {dismissal_count} "
                            f"dismissals vs {primary_count} substantive claims")
        
        return {
            'score': round(score, 3),
            'meta_ratio': round(meta_ratio, 3),
            'meta_count': meta_count,
            'primary_count': primary_count,
            'dismissal_count': dismissal_count,
            'interpretation': interpretation
        }
    
    def _calculate_network_isolation(self, session, claims: List[Dict]) -> Dict:
        """
        Calculate network isolation signal
        
        High isolation = alternative sources cite mainstream but aren't cited back
        
        Returns:
            {
                "score": 0.0-1.0,
                "internal_citations": int,
                "external_citations": int,
                "citation_asymmetry": float,
                "interpretation": str
            }
        """
        
        # Get claim IDs
        claim_ids = [c['claim_id'] for c in claims]
        
        if not claim_ids:
            return {
                'score': 0.0,
                'internal_citations': 0,
                'external_citations': 0,
                'citation_asymmetry': 0.0,
                'interpretation': 'No claims to analyze'
            }
        
        # Count citations
        query = """
        MATCH (c:Claim)
        WHERE elementId(c) IN $claim_ids
        
        // Internal citations (within our claim set)
        OPTIONAL MATCH (c)-[:CITES]->(internal:Claim)
        WHERE elementId(internal) IN $claim_ids
        
        // External citations (outside our claim set)
        OPTIONAL MATCH (c)-[:CITES]->(external:Claim)
        WHERE NOT elementId(external) IN $claim_ids
        
        RETURN 
            count(DISTINCT internal) as internal_citations,
            count(DISTINCT external) as external_citations
        """
        
        result = session.run(query, claim_ids=claim_ids)
        record = result.single()
        
        internal = record['internal_citations'] or 0
        external = record['external_citations'] or 0
        total = internal + external
        
        if total == 0:
            citation_asymmetry = 0.0
            score = 0.0
            interpretation = "No citations found"
        else:
            # High external citations relative to internal = high isolation
            citation_asymmetry = external / total
            score = citation_asymmetry
            
            interpretation = (f"{'High isolation' if score > 0.7 else 'Normal citation pattern'}: "
                            f"{external} outbound citations vs {internal} internal")
        
        return {
            'score': round(score, 3),
            'internal_citations': internal,
            'external_citations': external,
            'citation_asymmetry': round(citation_asymmetry, 3),
            'interpretation': interpretation
        }
    
    def _calculate_evidence_avoidance(self, claims: List[Dict]) -> Dict:
        """
        Calculate evidence avoidance signal
        
        High avoidance = META claims that provide no counter-evidence
        
        Returns:
            {
                "score": 0.0-1.0,
                "meta_with_citations": int,
                "meta_without_citations": int,
                "avoidance_ratio": float,
                "interpretation": str
            }
        """
        
        # Get META claims
        meta_claims = [c for c in claims if c.get('claim_type') == 'META']
        
        if not meta_claims:
            return {
                'score': 0.0,
                'meta_with_citations': 0,
                'meta_without_citations': 0,
                'avoidance_ratio': 0.0,
                'interpretation': 'No META claims to analyze'
            }
        
        # Count META claims with/without citations
        meta_with_citations = 0
        meta_without_citations = 0
        
        for claim in meta_claims:
            citation_data = claim.get('citation_data', {})
            has_citations = bool(citation_data.get('citations'))
            
            if has_citations:
                meta_with_citations += 1
            else:
                meta_without_citations += 1
        
        # Score: Higher ratio of META without citations = higher avoidance
        avoidance_ratio = meta_without_citations / len(meta_claims)
        score = avoidance_ratio
        
        interpretation = (f"{'Strong evidence avoidance' if score > 0.7 else 'Some evidence provided'}: "
                         f"{meta_without_citations} META claims with no citations")
        
        return {
            'score': round(score, 3),
            'meta_with_citations': meta_with_citations,
            'meta_without_citations': meta_without_citations,
            'avoidance_ratio': round(avoidance_ratio, 3),
            'interpretation': interpretation
        }
    
    def _calculate_emotional_manipulation(self, claims: List[Dict]) -> Dict:
        """
        Calculate emotional manipulation signal (SIMPLIFIED VERSION)
        
        Uses keyword detection for fear/anger appeals
        Future: Integrate with emotion_data from extractors
        
        Returns:
            {
                "score": 0.0-1.0,
                "emotional_claim_count": int,
                "emotional_ratio": float,
                "interpretation": str,
                "note": str
            }
        """
        
        # Fear/anger keywords
        fear_keywords = [
            'dangerous', 'deadly', 'fatal', 'catastrophic', 'disaster',
            'crisis', 'emergency', 'threat', 'risk', 'hazard'
        ]
        
        anger_keywords = [
            'outrage', 'scandal', 'fraud', 'betrayal', 'corrupt',
            'deceptive', 'lies', 'coverup', 'manipulation'
        ]
        
        emotional_keywords = fear_keywords + anger_keywords
        
        # Count claims with emotional language
        emotional_count = 0
        for claim in claims:
            text = claim.get('claim_text', '').lower()
            if any(keyword in text for keyword in emotional_keywords):
                emotional_count += 1
        
        total = len(claims)
        emotional_ratio = emotional_count / total if total > 0 else 0.0
        score = emotional_ratio
        
        interpretation = (f"{'High emotional content' if score > 0.5 else 'Factual tone'}: "
                         f"{emotional_count} claims ({emotional_ratio:.1%}) use fear/anger language")
        
        return {
            'score': round(score, 3),
            'emotional_claim_count': emotional_count,
            'emotional_ratio': round(emotional_ratio, 3),
            'interpretation': interpretation,
            'note': 'Using simplified keyword detection'
        }
    
    def _calculate_authority_mismatch(self, claims: List[Dict]) -> Dict:
        """
        Calculate authority domain mismatch signal (SIMPLIFIED VERSION)
        
        Uses keyword detection for credential concerns
        Future: Integrate with authority_data from extractors
        
        Returns:
            {
                "score": 0.0-1.0,
                "mismatch_count": int,
                "mismatch_ratio": float,
                "interpretation": str,
                "note": str
            }
        """
        
        # Mismatch indicator keywords
        mismatch_keywords = [
            'expert', 'authority', 'consensus', 'established',
            'mainstream', 'official', 'certified', 'credentialed'
        ]
        
        # Count claims mentioning authority/expertise
        mismatch_count = 0
        for claim in claims:
            text = claim.get('claim_text', '').lower()
            if any(keyword in text for keyword in mismatch_keywords):
                mismatch_count += 1
        
        total = len(claims)
        mismatch_ratio = mismatch_count / total if total > 0 else 0.0
        score = mismatch_ratio
        
        interpretation = (f"{'Some authority concerns' if score > 0.2 else 'No authority issues'}: "
                         f"{mismatch_count} potential mismatches")
        
        return {
            'score': round(score, 3),
            'mismatch_count': mismatch_count,
            'mismatch_ratio': round(mismatch_ratio, 3),
            'interpretation': interpretation,
            'note': 'Using simplified keyword detection'
        }
    
    def _aggregate_signals(self, signals: Dict) -> Tuple[float, float]:
        """
        Aggregate individual signals into overall suppression score
        
        Args:
            signals: Dict of signal results
            
        Returns:
            (suppression_score, confidence)
        """
        
        # Calculate weighted score
        suppression_score = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.WEIGHTS.items():
            if signal_name in signals:
                score = signals[signal_name].get('score', 0.0)
                suppression_score += score * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            suppression_score = suppression_score / total_weight
        
        # Calculate confidence based on data availability
        # All 5 signals present = 1.0 confidence
        # Fewer signals = lower confidence
        confidence = len(signals) / 5.0
        
        return suppression_score, confidence
    
    def _interpret_score(self, score: float, signals: Dict) -> str:
        """
        Generate human-readable interpretation of suppression score
        
        Args:
            score: Aggregate suppression score
            signals: Individual signal results
            
        Returns:
            Interpretation string
        """
        
        # Classify overall score
        if score >= 0.8:
            classification = "STRONG suppression pattern detected"
        elif score >= 0.6:
            classification = "MODERATE suppression pattern detected"
        elif score >= 0.4:
            classification = "WEAK suppression signals detected"
        elif score >= 0.2:
            classification = "MINIMAL suppression indicators"
        else:
            classification = "NO significant suppression detected"
        
        # Find strongest signals
        signal_scores = [(name, s.get('score', 0.0)) for name, s in signals.items()]
        signal_scores.sort(key=lambda x: x[1], reverse=True)
        top_signals = [name.replace('_', ' ').title() for name, _ in signal_scores[:2]]
        
        if signal_scores[0][1] > 0.5:
            interpretation = f"{classification}. Strongest signals: {', '.join(top_signals)}."
        else:
            interpretation = f"{classification}."
        
        return interpretation
    
    def _count_claim_types(self, claims: List[Dict]) -> Dict:
        """
        Count claims by type
        
        Returns:
            Dict with counts by type
        """
        
        counts = {
            'total': len(claims),
            'primary': 0,
            'secondary': 0,
            'meta': 0,
            'contextual': 0
        }
        
        for claim in claims:
            claim_type = claim.get('claim_type', '').upper()
            if claim_type == 'PRIMARY':
                counts['primary'] += 1
            elif claim_type == 'SECONDARY':
                counts['secondary'] += 1
            elif claim_type == 'META':
                counts['meta'] += 1
            elif claim_type == 'CONTEXTUAL':
                counts['contextual'] += 1
        
        return counts
    
    def _empty_result(self, topic: str) -> Dict:
        """
        Return empty result when no claims found
        """
        
        return {
            'topic': topic,
            'suppression_score': 0.0,
            'confidence': 0.0,
            'claims_analyzed': 0,
            'interpretation': 'No claims found matching query',
            'signals': {},
            'claim_counts': {
                'total': 0,
                'primary': 0,
                'secondary': 0,
                'meta': 0,
                'contextual': 0
            }
        }
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            self.logger.info("✓ Neo4j connection closed")


# ============================================================================
# CLI Testing Interface
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test detector
    detector = SuppressionDetector()
    
    print("\n" + "="*80)
    print("AEGIS SUPPRESSION DETECTOR - TEST RUN")
    print("="*80)
    
    # Test topics
    test_topics = [
        "historical topic",
        "archaeology",
        "alternative history"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {topic}")
        print('='*80)
        
        try:
            result = detector.detect_suppression(topic)
            
            print(f"\nSuppression Score: {result['suppression_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Claims Analyzed: {result['claims_analyzed']}")
            print(f"\nInterpretation: {result['interpretation']}")
            
            print(f"\nSignal Breakdown:")
            for signal_name, signal_data in result['signals'].items():
                print(f"  {signal_name.replace('_', ' ').title()}: {signal_data.get('score', 0.0):.3f}")
                print(f"    → {signal_data.get('interpretation', 'N/A')}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    detector.close()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
