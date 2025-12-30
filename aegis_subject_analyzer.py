#!/usr/bin/env python3
"""
Aegis Insight - Document Subject Analyzer

Analyzes documents to detect primary and secondary topics at ingestion time.
Topics are used to scope searches and eliminate cross-topic bleed-through.

Analysis Methods (in order of confidence):
1. Filename parsing - highest confidence (0.90-0.99)
2. Entity frequency - count dominant entities (0.70-0.90)
3. LLM analysis - for ambiguous cases (0.60-0.85)

Usage:
    analyzer = SubjectAnalyzer()
    result = analyzer.analyze("mark_twain_censorship.pdf", document_text)
    print(result.primary_topics)  # [TopicMatch(topic='Mark Twain', confidence=0.95)]

Author: Aegis Insight Team
Date: December 2025
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class TopicMatch:
    """A detected topic with confidence scoring"""
    topic: str
    topic_type: str  # person, event, theme, organization
    confidence: float  # 0.0-1.0
    source: str  # filename, entity_frequency, llm_analysis, manual
    is_primary: bool = False
    aliases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'topic': self.topic,
            'topic_type': self.topic_type,
            'confidence': self.confidence,
            'source': self.source,
            'is_primary': self.is_primary,
            'aliases': self.aliases
        }


@dataclass
class SubjectAnalysisResult:
    """Result of document subject analysis"""
    primary_topics: List[TopicMatch]
    secondary_topics: List[TopicMatch]
    all_matches: List[TopicMatch]
    analysis_methods_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'primary_topics': [t.to_dict() for t in self.primary_topics],
            'secondary_topics': [t.to_dict() for t in self.secondary_topics],
            'all_matches': [t.to_dict() for t in self.all_matches],
            'analysis_methods_used': self.analysis_methods_used
        }
    
    def get_topic_names(self) -> List[str]:
        """Get list of all topic names for caching on Document"""
        return [m.topic for m in self.all_matches]
    
    def get_primary_topic(self) -> Optional[str]:
        """Get the primary topic name, if any"""
        if self.primary_topics:
            return self.primary_topics[0].topic
        return None


class SubjectAnalyzer:
    """
    Analyzes documents to detect primary and secondary topics.
    
    Features:
    - Filename parsing with configurable patterns
    - Entity frequency analysis
    - Optional LLM analysis for ambiguous cases
    - Confidence scoring and merging
    """
    
    # =========================================================================
    # FILENAME PATTERNS
    # Pattern → (topic_name, topic_type, aliases)
    # =========================================================================
    
    FILENAME_PATTERNS = {
        # People
        r'twain|clemens|mark_twain|marktwain': 
            ('Mark Twain', 'person', ['Samuel Clemens', 'Clemens']),
        r'butler.*smedley|smedley.*butler|smedley_butler': 
            ('Smedley Butler', 'person', ['General Butler', 'Maj. Gen. Butler']),
        r'thomas.?paine|paine': 
            ('Thomas Paine', 'person', ['Paine']),
        r'hancock|graham.?hancock': 
            ('Graham Hancock', 'person', ['Hancock']),
        r'mercola|joe.?mercola': 
            ('Dr. Smith', 'person', ['John Smith', 'Smith']),
        r'naomi.?wolf': 
            ('Historical Author', 'person', ['Author']),
        
        # Events
        r'business.?plot|plot.?1933|fascist.?plot': 
            ('Business Plot', 'event', ['Wall Street Putsch', '1933 Plot']),
        r'spanish.?american|maine.?explosion|uss.?maine': 
            ('Spanish-American War', 'event', ['USS Maine', 'Remember the Maine']),
        r'prohibition|volstead|18th.?amendment': 
            ('Prohibition', 'event', ['Volstead Act']),
        r'boxer.?rebellion': 
            ('Boxer Rebellion', 'event', []),
        
        # Organizations
        r'fbi.?vault|fbi_vault|fbi.?files': 
            ('FBI', 'organization', ['Federal Bureau of Investigation']),
        r'american.?legion': 
            ('American Legion', 'organization', []),
        
        # Themes
        r'censorship|banned.?book|suppressed': 
            ('Censorship', 'theme', ['Book Banning', 'Suppression']),
        r'imperialism|anti.?imperial': 
            ('Imperialism', 'theme', ['Anti-Imperialism']),
    }
    
    # =========================================================================
    # KNOWN ENTITY → TOPIC MAPPINGS
    # For entity frequency analysis
    # =========================================================================
    
    ENTITY_TO_TOPIC = {
        'mark twain': ('Mark Twain', 'person'),
        'samuel clemens': ('Mark Twain', 'person'),
        'clemens': ('Mark Twain', 'person'),
        'twain': ('Mark Twain', 'person'),
        'smedley butler': ('Smedley Butler', 'person'),
        'general butler': ('Smedley Butler', 'person'),
        'smedley d. butler': ('Smedley Butler', 'person'),
        'thomas paine': ('Thomas Paine', 'person'),
        'paine': ('Thomas Paine', 'person'),
        'huckleberry finn': ('Mark Twain', 'person'),  # Work → Author
        'huck finn': ('Mark Twain', 'person'),
        'tom sawyer': ('Mark Twain', 'person'),
    }
    
    def __init__(self,
                 llm_enabled: bool = False,
                 llm_client = None,
                 llm_model: str = "mistral-nemo:12b",
                 min_entity_mentions: int = 3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Subject Analyzer.
        
        Args:
            llm_enabled: Whether to use LLM for ambiguous cases
            llm_client: Ollama or other LLM client (optional)
            llm_model: Model to use for LLM analysis
            min_entity_mentions: Minimum mentions for entity to be considered
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.llm_enabled = llm_enabled
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.min_entity_mentions = min_entity_mentions
        
        # Compile filename patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.compiled_filename_patterns = []
        for pattern, topic_info in self.FILENAME_PATTERNS.items():
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                self.compiled_filename_patterns.append((regex, topic_info))
            except re.error as e:
                self.logger.warning(f"Invalid pattern '{pattern}': {e}")
    
    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================
    
    def analyze(self,
                document_path: str,
                document_text: str = "",
                entities: List[str] = None,
                force_llm: bool = False) -> SubjectAnalysisResult:
        """
        Analyze document for topic affinity.
        
        Args:
            document_path: Filename/path for parsing
            document_text: Full or sampled text content
            entities: Pre-extracted entities (optional)
            force_llm: Force LLM analysis even if other methods succeed
            
        Returns:
            SubjectAnalysisResult with topics and confidence scores
        """
        all_matches: List[TopicMatch] = []
        methods_used: List[str] = []
        
        # Method 1: Filename parsing (always run - fast and high confidence)
        filename_matches = self._analyze_filename(document_path)
        if filename_matches:
            all_matches.extend(filename_matches)
            methods_used.append('filename')
            self.logger.debug(f"Filename analysis found: {[m.topic for m in filename_matches]}")
        
        # Method 2: Entity frequency (if text and/or entities provided)
        if document_text or entities:
            freq_matches = self._analyze_entity_frequency(document_text, entities)
            if freq_matches:
                all_matches.extend(freq_matches)
                methods_used.append('entity_frequency')
                self.logger.debug(f"Entity frequency found: {[m.topic for m in freq_matches]}")
        
        # Method 3: LLM analysis (if enabled and no high-confidence matches)
        high_conf = [m for m in all_matches if m.confidence >= 0.85]
        if self.llm_enabled and (force_llm or not high_conf):
            if document_text:
                llm_matches = self._analyze_with_llm(document_text)
                if llm_matches:
                    all_matches.extend(llm_matches)
                    methods_used.append('llm_analysis')
                    self.logger.debug(f"LLM analysis found: {[m.topic for m in llm_matches]}")
        
        # Merge duplicates and resolve conflicts
        merged = self._merge_matches(all_matches)
        
        # Separate primary and secondary
        primary = [m for m in merged if m.is_primary]
        secondary = [m for m in merged if not m.is_primary]
        
        # If no primary but have secondary, promote highest confidence
        if not primary and secondary:
            secondary.sort(key=lambda x: -x.confidence)
            secondary[0].is_primary = True
            primary = [secondary[0]]
            secondary = secondary[1:]
        
        return SubjectAnalysisResult(
            primary_topics=primary,
            secondary_topics=secondary,
            all_matches=merged,
            analysis_methods_used=methods_used
        )
    
    # =========================================================================
    # FILENAME ANALYSIS
    # =========================================================================
    
    def _analyze_filename(self, path: str) -> List[TopicMatch]:
        """
        Analyze filename/path for topic signals.
        
        High confidence because explicit naming is intentional.
        """
        if not path:
            return []
        
        # Normalize path for matching
        search_text = Path(path).stem.lower()
        # Also check parent directories (e.g., "package_5_mark_twain/...")
        full_path_lower = path.lower()
        
        matches = []
        seen_topics = set()
        
        for regex, (topic_name, topic_type, aliases) in self.compiled_filename_patterns:
            # Check both filename and full path
            if regex.search(search_text) or regex.search(full_path_lower):
                if topic_name not in seen_topics:
                    matches.append(TopicMatch(
                        topic=topic_name,
                        topic_type=topic_type,
                        confidence=0.95,
                        source='filename',
                        is_primary=len(matches) == 0,  # First match is primary
                        aliases=aliases
                    ))
                    seen_topics.add(topic_name)
        
        return matches
    
    # =========================================================================
    # ENTITY FREQUENCY ANALYSIS
    # =========================================================================
    
    def _analyze_entity_frequency(self,
                                   text: str,
                                   entities: List[str] = None) -> List[TopicMatch]:
        """
        Count entity mentions to find dominant subjects.
        
        Rules:
        - Entity must appear min_entity_mentions+ times
        - Top entity by count is primary topic
        - Entities with >50% of top count are secondary
        """
        if not text and not entities:
            return []
        
        # Build entity list
        if entities is None:
            entities = []
        
        # Add known entity patterns to search for
        entities_to_check = list(set(entities + list(self.ENTITY_TO_TOPIC.keys())))
        
        # Count mentions
        text_lower = text.lower() if text else ""
        counts: Dict[str, int] = {}
        
        for entity in entities_to_check:
            entity_lower = entity.lower()
            # Use word boundary matching for short names
            if len(entity_lower) <= 6:
                pattern = r'\b' + re.escape(entity_lower) + r'\b'
                count = len(re.findall(pattern, text_lower))
            else:
                count = text_lower.count(entity_lower)
            
            if count >= self.min_entity_mentions:
                # Map to canonical topic if known
                if entity_lower in self.ENTITY_TO_TOPIC:
                    topic_name, topic_type = self.ENTITY_TO_TOPIC[entity_lower]
                    # Accumulate counts for same topic
                    counts[topic_name] = counts.get(topic_name, 0) + count
                else:
                    counts[entity] = count
        
        if not counts:
            return []
        
        # Score by frequency
        max_count = max(counts.values())
        matches = []
        
        for topic, count in sorted(counts.items(), key=lambda x: -x[1]):
            ratio = count / max_count
            if ratio >= 0.3:  # Include if at least 30% of top
                # Calculate confidence based on ratio
                confidence = 0.70 + (ratio * 0.20)  # 0.70-0.90
                
                # Determine topic type
                topic_lower = topic.lower()
                if topic_lower in self.ENTITY_TO_TOPIC:
                    _, topic_type = self.ENTITY_TO_TOPIC[topic_lower]
                else:
                    topic_type = 'unknown'
                
                matches.append(TopicMatch(
                    topic=topic,
                    topic_type=topic_type,
                    confidence=round(confidence, 3),
                    source='entity_frequency',
                    is_primary=(count == max_count),
                    aliases=[]
                ))
        
        return matches[:5]  # Cap at 5 topics
    
    # =========================================================================
    # LLM ANALYSIS
    # =========================================================================
    
    LLM_PROMPT = """Analyze this document excerpt and identify its primary subjects.

Document excerpt:
{text}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "primary_subject": "Name of main person/event/topic this document is about",
  "primary_type": "person|event|theme|organization",
  "secondary_subjects": ["Other significant subjects discussed"],
  "themes": ["Thematic categories like: censorship, suppression, war, politics"]
}}"""

    def _analyze_with_llm(self, text: str, sample_size: int = 4000) -> List[TopicMatch]:
        """
        Use LLM for ambiguous documents.
        
        Sampling strategy:
        - Documents < sample_size: analyze entire text
        - Larger documents: first 2000 chars + last 1000 chars
        """
        if not self.llm_enabled or not text:
            return []
        
        # Sample text
        if len(text) > sample_size:
            sample = text[:2500] + "\n\n[...]\n\n" + text[-1500:]
        else:
            sample = text
        
        try:
            # Try Ollama if available
            if self.llm_client:
                response = self.llm_client.generate(
                    model=self.llm_model,
                    prompt=self.LLM_PROMPT.format(text=sample)
                )
                result_text = response.get('response', '')
            else:
                # Try direct ollama call
                import subprocess
                cmd = ['ollama', 'run', self.llm_model, self.LLM_PROMPT.format(text=sample[:2000])]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                result_text = result.stdout
            
            # Parse JSON from response
            # Find JSON in response (may have extra text)
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if not json_match:
                self.logger.warning("LLM response did not contain valid JSON")
                return []
            
            result = json.loads(json_match.group())
            
            matches = []
            
            # Primary subject
            if result.get('primary_subject'):
                matches.append(TopicMatch(
                    topic=result['primary_subject'],
                    topic_type=result.get('primary_type', 'unknown'),
                    confidence=0.75,
                    source='llm_analysis',
                    is_primary=True,
                    aliases=[]
                ))
            
            # Secondary subjects
            for subject in result.get('secondary_subjects', [])[:3]:
                if subject and subject != result.get('primary_subject'):
                    matches.append(TopicMatch(
                        topic=subject,
                        topic_type='unknown',
                        confidence=0.60,
                        source='llm_analysis',
                        is_primary=False,
                        aliases=[]
                    ))
            
            # Themes (lower confidence)
            for theme in result.get('themes', [])[:2]:
                if theme:
                    matches.append(TopicMatch(
                        topic=theme.title(),
                        topic_type='theme',
                        confidence=0.55,
                        source='llm_analysis',
                        is_primary=False,
                        aliases=[]
                    ))
            
            return matches
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}")
            return []
    
    # =========================================================================
    # MERGE AND RESOLVE
    # =========================================================================
    
    def _merge_matches(self, matches: List[TopicMatch]) -> List[TopicMatch]:
        """
        Merge duplicate topics from different sources.
        
        Rules:
        - Same topic from multiple sources: keep highest confidence
        - Combine aliases
        - Preserve is_primary if any source marked it primary
        """
        if not matches:
            return []
        
        # Group by normalized topic name
        groups: Dict[str, List[TopicMatch]] = {}
        for match in matches:
            key = match.topic.lower().strip()
            if key not in groups:
                groups[key] = []
            groups[key].append(match)
        
        # Merge each group
        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Take highest confidence
                group.sort(key=lambda x: -x.confidence)
                best = group[0]
                
                # Combine properties from all matches
                all_aliases = set(best.aliases)
                is_primary = any(m.is_primary for m in group)
                sources = list(set(m.source for m in group))
                
                for other in group[1:]:
                    all_aliases.update(other.aliases)
                
                merged.append(TopicMatch(
                    topic=best.topic,
                    topic_type=best.topic_type,
                    confidence=best.confidence,
                    source=sources[0] if len(sources) == 1 else 'multiple',
                    is_primary=is_primary,
                    aliases=list(all_aliases)
                ))
        
        # Sort by confidence
        merged.sort(key=lambda x: (-x.is_primary, -x.confidence))
        
        return merged
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def add_filename_pattern(self, pattern: str, topic: str, 
                             topic_type: str, aliases: List[str] = None):
        """Add a custom filename pattern at runtime"""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            self.compiled_filename_patterns.append(
                (regex, (topic, topic_type, aliases or []))
            )
            self.logger.info(f"Added pattern '{pattern}' → {topic}")
        except re.error as e:
            self.logger.error(f"Invalid pattern '{pattern}': {e}")
    
    def add_entity_mapping(self, entity: str, topic: str, topic_type: str):
        """Add a custom entity → topic mapping at runtime"""
        self.ENTITY_TO_TOPIC[entity.lower()] = (topic, topic_type)
        self.logger.info(f"Added entity mapping '{entity}' → {topic}")


# =============================================================================
# STANDALONE TESTING
# =============================================================================

def test_analyzer():
    """Test the subject analyzer"""
    
    logging.basicConfig(level=logging.DEBUG)
    analyzer = SubjectAnalyzer(llm_enabled=False)
    
    print("=" * 70)
    print("SUBJECT ANALYZER TEST")
    print("=" * 70)
    
    # Test cases
    test_cases = [
        ("package_5_mark_twain/side_a_suppressed/king_leopolds_soliloquy_twain.txt", ""),
        ("Smedley Butler Part 01 of 01_FBI_vault.jsonl", ""),
        ("marktwainbiograp01pain_marktwainbiograp01pain_djvu.txt", "Mark Twain was born..."),
        ("unknown_document.pdf", "This document discusses Thomas Paine and his views on religion."),
        ("banned_books_history.pdf", "Mark Twain's works were frequently censored. Twain faced significant opposition."),
    ]
    
    for path, text in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Document: {path[:60]}...")
        
        result = analyzer.analyze(path, text)
        
        print(f"Methods used: {result.analysis_methods_used}")
        print(f"Primary topics: {[(t.topic, t.confidence) for t in result.primary_topics]}")
        print(f"Secondary topics: {[(t.topic, t.confidence) for t in result.secondary_topics]}")
        print(f"All topic names: {result.get_topic_names()}")


if __name__ == "__main__":
    test_analyzer()
