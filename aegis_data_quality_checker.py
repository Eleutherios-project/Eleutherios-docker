#!/usr/bin/env python3
"""
Aegis Insight - Data Quality Checker

Assesses whether a dataset has sufficient diversity for meaningful detection.
Provides warnings and recommendations when data is insufficient.

Usage:
    from aegis_data_quality_checker import DataQualityChecker
    
    checker = DataQualityChecker()
    result = checker.assess_for_suppression("historical topic")
    print(result['warnings'])
"""

import logging
from typing import Dict, List, Optional
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Check if dataset is sufficient for detection analysis.
    
    Provides pre-flight checks before running expensive detection algorithms,
    warning users when data lacks the diversity needed for meaningful results.
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "aegistrusted"):
        """Initialize with Neo4j connection"""
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        logger.info("DataQualityChecker initialized")
    
    def assess_for_suppression(self, topic: str) -> Dict:
        """
        Check if dataset has enough diversity for suppression detection.
        
        Suppression detection requires:
        - PRIMARY claims (substantive research/evidence)
        - META claims (institutional responses, dismissals)
        - Multiple sources (for citation asymmetry analysis)
        
        Args:
            topic: Topic to analyze
            
        Returns:
            {
                'sufficient': bool,
                'warnings': List[str],
                'metrics': Dict,
                'recommendations': List[str]
            }
        """
        with self.driver.session() as session:
            # Count claims by type
            result = session.run("""
                MATCH (c:Claim)
                WHERE toLower(c.claim_text) CONTAINS toLower($topic)
                RETURN 
                    c.claim_type as type,
                    count(*) as count
            """, topic=topic)
            
            type_counts = {}
            for record in result:
                claim_type = record['type'] or 'UNKNOWN'
                type_counts[claim_type] = record['count']
            
            # Count unique sources
            result = session.run("""
                MATCH (c:Claim)
                WHERE toLower(c.claim_text) CONTAINS toLower($topic)
                RETURN count(DISTINCT c.source_file) as source_count
            """, topic=topic)
            
            source_count = result.single()['source_count']
            
            # Build assessment
            warnings = []
            recommendations = []
            
            total_claims = sum(type_counts.values())
            primary_count = type_counts.get('PRIMARY', 0)
            meta_count = type_counts.get('META', 0)
            secondary_count = type_counts.get('SECONDARY', 0)
            contextual_count = type_counts.get('CONTEXTUAL', 0)
            
            # Check total claims
            if total_claims == 0:
                warnings.append(f"⚠️ No claims found for topic '{topic}'")
                return {
                    'sufficient': False,
                    'warnings': warnings,
                    'metrics': {'total_claims': 0},
                    'recommendations': ["Import documents related to this topic"]
                }
            
            if total_claims < 10:
                warnings.append(f"⚠️ Very few claims found ({total_claims}) - results may be unreliable")
            
            # Check PRIMARY claims
            if primary_count < 10:
                warnings.append(f"⚠️ Few PRIMARY claims ({primary_count}) - need substantive research claims")
            
            # Check META claims - critical for suppression detection
            if meta_count == 0:
                warnings.append("⚠️ No META claims detected - suppression analysis will be inconclusive")
                warnings.append("   META claims are institutional responses, dismissals, or claims about claims")
                recommendations.append("Add institutional response documents:")
                recommendations.append("  • FDA/CDC statements and press releases")
                recommendations.append("  • Medical journal editorials")
                recommendations.append("  • Fact-checker articles (Reuters, AP, etc.)")
                recommendations.append("  • Official rebuttals or policy statements")
            elif meta_count < 5:
                warnings.append(f"⚠️ Few META claims ({meta_count}) - limited suppression signals available")
            
            # Check source diversity
            if source_count == 1:
                warnings.append("⚠️ Single-source dataset - cannot detect citation asymmetry")
                recommendations.append("Add documents from multiple sources for comparison")
            elif source_count < 3:
                warnings.append(f"⚠️ Limited source diversity ({source_count} sources)")
            
            # Determine sufficiency
            # Suppression detection CAN run but results may be limited
            sufficient = (
                total_claims >= 10 and
                (primary_count >= 5 or total_claims >= 20)
            )
            
            # Add context about what WILL work
            if warnings and sufficient:
                warnings.insert(0, "ℹ️ Detection will run but results may be limited:")
            
            return {
                'sufficient': sufficient,
                'has_meta_claims': meta_count > 0,
                'warnings': warnings,
                'metrics': {
                    'total_claims': total_claims,
                    'primary_claims': primary_count,
                    'meta_claims': meta_count,
                    'secondary_claims': secondary_count,
                    'contextual_claims': contextual_count,
                    'unique_sources': source_count
                },
                'recommendations': recommendations,
                'analysis_notes': self._get_analysis_notes(meta_count, source_count)
            }
    
    def assess_for_coordination(self, topic: str) -> Dict:
        """
        Check if dataset has enough diversity for coordination detection.
        
        Coordination detection requires:
        - Multiple independent sources
        - Temporal data for clustering analysis
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Assessment dict
        """
        with self.driver.session() as session:
            # Count sources and temporal coverage
            result = session.run("""
                MATCH (c:Claim)
                WHERE toLower(c.claim_text) CONTAINS toLower($topic)
                RETURN 
                    count(*) as total,
                    count(DISTINCT c.source_file) as source_count,
                    count(CASE WHEN c.temporal_data IS NOT NULL THEN 1 END) as with_temporal
            """, topic=topic)
            
            record = result.single()
            total = record['total']
            source_count = record['source_count']
            with_temporal = record['with_temporal']
            
            warnings = []
            recommendations = []
            
            if total == 0:
                return {
                    'sufficient': False,
                    'warnings': [f"⚠️ No claims found for topic '{topic}'"],
                    'metrics': {'total_claims': 0},
                    'recommendations': ["Import documents related to this topic"]
                }
            
            if source_count < 3:
                warnings.append(f"⚠️ Coordination detection requires 3+ independent sources")
                warnings.append(f"   Current dataset has {source_count} source(s)")
                recommendations.append("Add documents from additional independent sources")
            
            if with_temporal < total * 0.3:
                warnings.append(f"⚠️ Limited temporal data ({with_temporal}/{total} claims)")
                warnings.append("   Temporal clustering analysis may be limited")
            
            sufficient = source_count >= 3 and total >= 10
            
            return {
                'sufficient': sufficient,
                'warnings': warnings,
                'metrics': {
                    'total_claims': total,
                    'unique_sources': source_count,
                    'claims_with_temporal': with_temporal
                },
                'recommendations': recommendations
            }
    
    def assess_for_anomaly(self, topic: str) -> Dict:
        """
        Check if dataset has geographic diversity for anomaly detection.
        
        Anomaly detection requires:
        - Geographic diversity (multiple regions/cultures)
        - Cross-cultural content
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Assessment dict
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Claim)
                WHERE toLower(c.claim_text) CONTAINS toLower($topic)
                RETURN 
                    count(*) as total,
                    count(CASE WHEN c.geographic_data IS NOT NULL THEN 1 END) as with_geo
            """, topic=topic)
            
            record = result.single()
            total = record['total']
            with_geo = record['with_geo']
            
            warnings = []
            recommendations = []
            
            if total == 0:
                return {
                    'sufficient': False,
                    'warnings': [f"⚠️ No claims found for topic '{topic}'"],
                    'metrics': {'total_claims': 0},
                    'recommendations': ["Import documents related to this topic"]
                }
            
            if with_geo < 2:
                warnings.append(f"⚠️ Anomaly detection requires geographic diversity")
                warnings.append(f"   Found {with_geo} claims with geographic context")
                recommendations.append("Add documents from multiple geographic regions")
                recommendations.append("Include cross-cultural perspectives on the topic")
            
            # Note: Anomaly detector also has embedding issues (separate problem)
            warnings.append("ℹ️ Note: Anomaly detector is currently limited (embedding architecture)")
            
            sufficient = with_geo >= 2 and total >= 10
            
            return {
                'sufficient': sufficient,
                'warnings': warnings,
                'metrics': {
                    'total_claims': total,
                    'claims_with_geographic': with_geo
                },
                'recommendations': recommendations
            }
    
    def assess_all(self, topic: str) -> Dict:
        """
        Run all assessments for a topic.
        
        Returns:
            Combined assessment for all detection types
        """
        return {
            'topic': topic,
            'suppression': self.assess_for_suppression(topic),
            'coordination': self.assess_for_coordination(topic),
            'anomaly': self.assess_for_anomaly(topic)
        }
    
    def _get_analysis_notes(self, meta_count: int, source_count: int) -> List[str]:
        """Generate notes about what analysis IS possible"""
        notes = []
        
        if meta_count == 0:
            notes.append("✓ Can analyze: claim quality, authority scores, emotional tone")
            notes.append("✗ Cannot analyze: suppression patterns (no institutional responses)")
        else:
            notes.append("✓ Full suppression analysis available")
        
        if source_count >= 2:
            notes.append("✓ Can analyze: citation patterns between sources")
        else:
            notes.append("✗ Cannot analyze: citation asymmetry (single source)")
        
        return notes
    
    def close(self):
        """Close database connection"""
        self.driver.close()


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    topic = sys.argv[1] if len(sys.argv) > 1 else "historical topic"
    
    print(f"\nAssessing data quality for topic: '{topic}'")
    print("=" * 60)
    
    checker = DataQualityChecker()
    
    try:
        # Suppression assessment
        print("\n📊 SUPPRESSION DETECTION")
        print("-" * 40)
        result = checker.assess_for_suppression(topic)
        print(f"Sufficient: {'✓' if result['sufficient'] else '✗'}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        if result['warnings']:
            print("Warnings:")
            for w in result['warnings']:
                print(f"  {w}")
        if result.get('recommendations'):
            print("Recommendations:")
            for r in result['recommendations']:
                print(f"  {r}")
        
        # Coordination assessment
        print("\n📊 COORDINATION DETECTION")
        print("-" * 40)
        result = checker.assess_for_coordination(topic)
        print(f"Sufficient: {'✓' if result['sufficient'] else '✗'}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        if result['warnings']:
            print("Warnings:")
            for w in result['warnings']:
                print(f"  {w}")
        
        # Anomaly assessment
        print("\n📊 ANOMALY DETECTION")
        print("-" * 40)
        result = checker.assess_for_anomaly(topic)
        print(f"Sufficient: {'✓' if result['sufficient'] else '✗'}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        if result['warnings']:
            print("Warnings:")
            for w in result['warnings']:
                print(f"  {w}")
        
    finally:
        checker.close()
    
    print("\n" + "=" * 60)
