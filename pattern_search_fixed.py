#!/usr/bin/env python3
"""
Pattern Search - Fixed Version
Actually queries Neo4j and uses results in LLM prompt
"""

from neo4j import GraphDatabase
import ollama
import re
from typing import List, Dict


class PatternSearchFixed:
    """Pattern search that actually uses the database"""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", 
                 neo4j_password="aegistrusted"):
        self.driver = GraphDatabase.driver(neo4j_uri, 
                                          auth=(neo4j_user, neo4j_password))
        self.ollama_model = "qwen2.5:7b"  # Or whatever model you're using
    
    def search(self, query: str) -> Dict:
        """
        Main search function
        
        Returns:
            dict with 'answer', 'confidence', 'sources', 'graph_stats'
        """
        
        print(f"\n🔍 Searching for: {query}")
        print("=" * 60)
        
        # Step 1: Extract key terms from query
        print("Step 1: Extracting key terms...")
        key_terms = self._extract_key_terms(query)
        print(f"  Key terms: {key_terms}")
        
        # Step 2: Query Neo4j
        print("Step 2: Querying knowledge graph...")
        graph_results = self._query_neo4j(key_terms)
        print(f"  Found {len(graph_results)} claims")
        
        if len(graph_results) == 0:
            return {
                "answer": "No relevant information found in the knowledge graph.",
                "confidence": 0.0,
                "sources": [],
                "graph_stats": {
                    "claims_found": 0,
                    "entities_found": 0
                }
            }
        
        # Step 3: Format as context
        print("Step 3: Formatting context...")
        context = self._format_context(graph_results)
        
        # Step 4: Generate LLM response WITH context
        print("Step 4: Generating response...")
        answer = self._generate_response(query, context)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(graph_results)
        
        print("✅ Search complete!")
        print("=" * 60)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": self._format_sources(graph_results),
            "graph_stats": {
                "claims_found": len(graph_results),
                "entities_found": len(set(r['entity'] for r in graph_results)),
                "avg_trust": context['avg_trust'],
                "suppression_score": context['suppression_score']
            }
        }
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important words from query"""
        # Convert to lowercase
        query = query.lower()
        
        # Split into words
        words = re.findall(r'\b\w+\b', query)
        
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'what', 'when', 'where', 'who', 'how', 'why'
        }
        
        terms = [w for w in words if w not in stopwords and len(w) > 2]
        
        return terms
    
    def _query_neo4j(self, key_terms: List[str]) -> List[Dict]:
        """Query Neo4j for relevant claims"""
        
        query = """
        // Find entities matching key terms (case-insensitive)
        MATCH (e:Entity)
        WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS term)
        
        // Get claims mentioning these entities
        MATCH (e)<-[:MENTIONS]-(chunk)-[:CONTAINS_CLAIM]->(claim:Claim)
        
        // Get other related entities in same chunks
        MATCH (chunk)-[:MENTIONS]->(related:Entity)
        WHERE related <> e
        
        RETURN 
            e.name as entity,
            claim.text as claim_text,
            claim.trust_score as trust_score,
            coalesce(claim.perspective, 'neutral') as perspective,
            coalesce(claim.claim_type, 'CONTEXTUAL') as claim_type,
            collect(DISTINCT related.name)[0..5] as related_entities
        ORDER BY claim.trust_score DESC
        LIMIT 50
        """
        
        with self.driver.session() as session:
            result = session.run(query, terms=key_terms)
            records = [dict(record) for record in result]
        
        return records
    
    def _format_context(self, results: List[Dict]) -> Dict:
        """Format graph results into context for LLM"""
        
        if not results:
            return {
                'claims_text': '',
                'entities_text': '',
                'avg_trust': 0.0,
                'mainstream_count': 0,
                'alternative_count': 0,
                'suppression_score': 0.0,
                'suppression_summary': ''
            }
        
        # Format claims as text
        claims_text = "\n\n".join([
            f"• [{r['perspective'].upper()}] {r['claim_text']}\n"
            f"  Trust: {r['trust_score']:.2f} | Type: {r['claim_type']}"
            for r in results
        ])
        
        # Get unique entities
        entities = set(r['entity'] for r in results)
        entities_text = ", ".join(sorted(entities))
        
        # Count by perspective
        mainstream = [r for r in results if r['perspective'] == 'mainstream']
        alternative = [r for r in results if r['perspective'] == 'alternative']
        
        # Calculate average trust
        avg_trust = sum(r['trust_score'] for r in results) / len(results)
        
        # Detect suppression
        meta_dismissal = [r for r in results 
                         if r['claim_type'] == 'META' 
                         and any(word in r['claim_text'].lower() 
                                for word in ['dismiss', 'debunk', 'censor', 'suppress'])]
        
        suppression_score = len(meta_dismissal) / len(results) if results else 0.0
        
        if suppression_score > 0.3:
            suppression_summary = f"⚠️ HIGH SUPPRESSION DETECTED ({suppression_score:.0%}): Multiple META claims show dismissal/censorship patterns."
        elif suppression_score > 0.1:
            suppression_summary = f"Moderate suppression indicators present ({suppression_score:.0%})."
        else:
            suppression_summary = "No significant suppression detected."
        
        return {
            'claims_text': claims_text,
            'entities_text': entities_text,
            'avg_trust': avg_trust,
            'mainstream_count': len(mainstream),
            'alternative_count': len(alternative),
            'suppression_score': suppression_score,
            'suppression_summary': suppression_summary
        }
    
    def _generate_response(self, query: str, context: Dict) -> str:
        """Generate LLM response using graph context"""
        
        prompt = f"""You are analyzing information from a knowledge graph database.

QUERY: {query}

RELEVANT CLAIMS FROM DATABASE:
{context['claims_text']}

ENTITIES MENTIONED:
{context['entities_text']}

STATISTICS:
- Total claims analyzed: Based on knowledge graph
- Average trust score: {context['avg_trust']:.2f}
- Mainstream perspective: {context['mainstream_count']} claims
- Alternative perspective: {context['alternative_count']} claims

SUPPRESSION ANALYSIS:
{context['suppression_summary']}

INSTRUCTIONS:
1. Answer the query using ONLY the claims provided above
2. Cite trust scores and perspectives when relevant
3. Note any patterns (agreement, disagreement, suppression)
4. If claims conflict, present both sides
5. Keep answer concise but informative

Answer:"""
        
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on results"""
        
        if not results:
            return 0.0
        
        # Factors:
        # 1. Number of claims (more is better)
        # 2. Average trust score
        # 3. Diversity of sources
        
        claim_count_score = min(len(results) / 20.0, 1.0)  # Max at 20 claims
        avg_trust = sum(r['trust_score'] for r in results) / len(results)
        
        # Diversity: different entities
        entities = set(r['entity'] for r in results)
        diversity_score = min(len(entities) / 5.0, 1.0)  # Max at 5 entities
        
        # Weighted average
        confidence = (claim_count_score * 0.3 + 
                     avg_trust * 0.5 + 
                     diversity_score * 0.2)
        
        return confidence
    
    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        """Format top sources for display"""
        
        # Group by entity, get highest trust
        entity_claims = {}
        for r in results:
            entity = r['entity']
            if entity not in entity_claims or r['trust_score'] > entity_claims[entity]['trust_score']:
                entity_claims[entity] = r
        
        # Sort by trust score
        top_sources = sorted(entity_claims.values(), 
                           key=lambda x: x['trust_score'], 
                           reverse=True)[:5]
        
        return [
            {
                'entity': s['entity'],
                'claim': s['claim_text'][:100] + '...' if len(s['claim_text']) > 100 else s['claim_text'],
                'trust': s['trust_score']
            }
            for s in top_sources
        ]
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()


# Example usage
if __name__ == "__main__":
    searcher = PatternSearchFixed()
    
    # Test query
    result = searcher.search("meditation and qi gong")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nConfidence: {result['confidence']:.0%}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nGraph Stats:")
    print(f"  Claims found: {result['graph_stats']['claims_found']}")
    print(f"  Entities found: {result['graph_stats']['entities_found']}")
    print(f"  Avg trust: {result['graph_stats']['avg_trust']:.2f}")
    
    print(f"\nTop Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['entity']} (trust: {source['trust']:.2f})")
        print(f"     {source['claim']}")
    
    searcher.close()
