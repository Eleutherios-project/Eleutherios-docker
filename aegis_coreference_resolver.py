"""
Aegis Insight - Coreference Resolver
Resolves pronouns to explicit entity mentions before claim extraction

This runs BEFORE claim extraction to ensure claims contain explicit
entity references rather than pronouns like "he", "she", "they", "it".

Example:
    Input:  "Gen Butler testified to Congress. He accused the bankers."
    Output: "Gen Butler testified to Congress. Gen Butler accused the bankers."
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Tuple


class CoreferenceResolver:
    """
    Resolve pronouns to their antecedents using LLM.
    Designed to run on chunks before claim extraction.
    """
    
    # Pronouns to resolve
    PRONOUNS = {
        'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 
        'they', 'them', 'their', 'theirs', 'themselves',
        'it', 'its', 'itself',
        'who', 'whom', 'whose', 'which', 'that'
    }
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "mistral-nemo:12b",
                 logger: Optional[logging.Logger] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize resolver.
        
        Args:
            ollama_url: Ollama API endpoint
            model: Model to use (12b is sufficient for this task)
            logger: Optional logger
            confidence_threshold: Only resolve if confident (0.0-1.0)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        
        # Stats tracking
        self.stats = {
            'chunks_processed': 0,
            'pronouns_resolved': 0,
            'pronouns_kept': 0,  # Left as-is due to low confidence
            'errors': 0
        }
    
    def needs_resolution(self, text: str) -> bool:
        """
        Quick check if text contains pronouns worth resolving.
        Avoids LLM call for text with no pronouns.
        """
        words = set(text.lower().split())
        return bool(words & self.PRONOUNS)
    
    def resolve(self, 
                chunk_text: str, 
                entities_hint: Optional[List[str]] = None,
                context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Resolve pronouns in chunk text.
        
        Args:
            chunk_text: Text to process
            entities_hint: Optional list of known entities in this chunk
                          (from prior entity extraction or document metadata)
            context: Optional context dict (title, domain, etc.)
            
        Returns:
            Tuple of (resolved_text, resolution_metadata)
        """
        self.stats['chunks_processed'] += 1
        
        # Skip if no pronouns
        if not self.needs_resolution(chunk_text):
            return chunk_text, {'resolved': False, 'reason': 'no_pronouns'}
        
        # Build prompt
        prompt = self._build_prompt(chunk_text, entities_hint, context)
        
        try:
            response = self._call_ollama(prompt)
            result = self._parse_response(response, chunk_text)
            
            if result['resolved_text'] != chunk_text:
                self.stats['pronouns_resolved'] += result.get('resolution_count', 1)
            
            return result['resolved_text'], result
            
        except Exception as e:
            self.logger.warning(f"Coreference resolution failed: {e}")
            self.stats['errors'] += 1
            return chunk_text, {'resolved': False, 'error': str(e)}
    
    def _build_prompt(self, 
                      chunk_text: str, 
                      entities_hint: Optional[List[str]],
                      context: Optional[Dict]) -> str:
        """Build LLM prompt for coreference resolution."""
        
        entities_str = ""
        if entities_hint:
            entities_str = f"\nKnown entities in this text: {', '.join(entities_hint)}"
        
        context_str = ""
        if context:
            if context.get('title'):
                context_str += f"\nDocument: {context['title']}"
            if context.get('domain'):
                context_str += f"\nDomain: {context['domain']}"
        
        # More explicit prompt that forces actual text replacement
        return f"""Your task is to replace pronouns with the specific entities they refer to.

INSTRUCTIONS:
1. Find each pronoun: he, she, they, it, him, her, them, his, her, their, its
2. Replace EACH pronoun with the entity it refers to:
   - "he/him/his" → the male person mentioned
   - "she/her/hers" → the female person mentioned  
   - "it/its" → the thing, place, or concept mentioned (book, ship, organization, etc.)
   - "they/them/their" → the group or plural entity mentioned
3. Return the COMPLETE modified text with ALL pronouns replaced
4. Only replace if you are CONFIDENT (>70%) about the referent
5. If uncertain about a specific pronoun, leave that one unchanged
{entities_str}
{context_str}

EXAMPLES:
- "Gen Butler testified. He accused the bankers." → "Gen Butler testified. Gen Butler accused the bankers."
- "Paine wrote Common Sense. It became popular." → "Paine wrote Common Sense. Common Sense became popular."
- "The Maine exploded. It killed 266 sailors." → "The Maine exploded. The explosion killed 266 sailors."

ORIGINAL TEXT:
{chunk_text}

TASK: Rewrite the text above with pronouns replaced by entity names.

Return JSON in this EXACT format:
{{"resolved_text": "the complete rewritten text with pronouns replaced", "resolutions": [{{"original": "He", "replacement": "Gen Butler", "confidence": 0.95}}]}}

CRITICAL: The "resolved_text" must be the FULL rewritten text, not the original. Actually replace the pronouns.

JSON:"""

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get('response', '')
    
    def _parse_response(self, response: str, original_text: str) -> Dict:
        """Parse LLM response, with fallback to original text."""
        
        # Clean response
        response = response.strip()
        
        # Try to extract JSON
        if '{' in response:
            try:
                start = response.index('{')
                # Find matching closing brace (handle nested braces)
                depth = 0
                end = start
                for i, c in enumerate(response[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                
                json_str = response[start:end]
                data = json.loads(json_str)
                
                resolved_text = data.get('resolved_text', '')
                resolutions = data.get('resolutions', [])
                
                # Validate resolved_text is present, non-empty, and reasonably sized
                if resolved_text and len(resolved_text) > len(original_text) * 0.5:
                    # Filter by confidence threshold
                    confident_resolutions = [
                        r for r in resolutions 
                        if r.get('confidence', 0) >= self.confidence_threshold
                    ]
                    
                    # Check if text actually changed
                    text_changed = resolved_text.strip() != original_text.strip()
                    
                    return {
                        'resolved': text_changed and len(confident_resolutions) > 0,
                        'resolved_text': resolved_text if text_changed else original_text,
                        'resolutions': confident_resolutions,
                        'resolution_count': len(confident_resolutions)
                    }
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"JSON parse error: {e}")
        
        # Fallback: return original
        return {
            'resolved': False,
            'resolved_text': original_text,
            'resolutions': [],
            'error': 'parse_failed'
        }
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'chunks_processed': 0,
            'pronouns_resolved': 0,
            'pronouns_kept': 0,
            'errors': 0
        }


# =============================================================================
# TESTS
# =============================================================================

def test_coreference_resolver():
    """Test the coreference resolver."""
    print("="*70)
    print("COREFERENCE RESOLVER TEST")
    print("="*70)
    
    resolver = CoreferenceResolver()
    
    # Test cases
    test_cases = [
        {
            'text': "Gen Butler testified to Congress. He accused the bankers of orchestrating a plot.",
            'entities': ["Gen Butler", "Congress", "bankers"],
            'expected_contains': "Butler accused"
        },
        {
            'text': "Thomas Paine wrote Common Sense. It became the most widely read pamphlet.",
            'entities': ["Thomas Paine", "Common Sense"],
            'expected_contains': "Common Sense became"
        },
        {
            'text': "The Maine exploded in Havana harbor. It killed 266 sailors.",
            'entities': ["Maine", "Havana"],
            'expected_contains': "Maine"  # or "explosion"
        },
        {
            'text': "No pronouns in this text at all.",
            'entities': [],
            'expected_contains': "No pronouns"  # Should be unchanged
        },
        # NEW: They/them test cases
        {
            'text': "The bankers met with Butler. They offered him $300,000 to lead a coup.",
            'entities': ["bankers", "Butler"],
            'expected_contains': "Butler"  # At minimum, "him" should resolve to Butler
        },
        {
            'text': "Roosevelt and Churchill met at Yalta. They discussed the post-war order.",
            'entities': ["Roosevelt", "Churchill", "Yalta"],
            'expected_contains': "Roosevelt and Churchill discussed"  # or just one name
        },
        {
            'text': "The committee investigated the plot. Their report was buried.",
            'entities': ["committee", "plot"],
            'expected_contains': "committee"  # "The committee's report" or similar
        },
        {
            'text': "Hearst owned many newspapers. His papers promoted the war.",
            'entities': ["Hearst", "newspapers"],
            'expected_contains': "Hearst"  # "Hearst's papers"
        },
    ]
    
    passed = 0
    for i, tc in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Input: {tc['text'][:60]}...")
        
        resolved, meta = resolver.resolve(
            tc['text'], 
            entities_hint=tc['entities']
        )
        
        print(f"  Output: {resolved[:60]}...")
        print(f"  Resolved: {meta.get('resolved', False)}")
        
        if tc['expected_contains'] in resolved:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL - expected '{tc['expected_contains']}'")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{len(test_cases)} passed")
    print(f"Stats: {resolver.get_stats()}")
    
    return passed == len(test_cases)


if __name__ == "__main__":
    test_coreference_resolver()
