#!/usr/bin/env python3
"""
Aegis Insight - Document Temporal Context Analyzer

Detects document-level temporal context from:
1. Publication dates (copyright, published year)
2. Era references (1880s, Civil War, etc.)
3. Historical period markers

This provides "soft dates" for claims that lack explicit temporal data.

Usage:
    analyzer = TemporalContextAnalyzer()
    result = analyzer.analyze("Twain - Letters from the Earth.pdf", first_page_text)
    
    # Result:
    {
        "publication_date": "1962",
        "publication_confidence": 0.85,
        "composition_date": "1909",  # When written vs published
        "primary_era": "early 20th century",
        "era_range": [1900, 1920],
        "historical_period": "Progressive Era",
        "sources": ["title_year", "copyright_notice"]
    }

Author: Aegis Insight Team
Date: December 2025
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TemporalContext:
    """Document-level temporal context"""
    publication_date: Optional[str] = None
    publication_confidence: float = 0.0
    composition_date: Optional[str] = None  # When written (may differ from publication)
    composition_confidence: float = 0.0
    primary_era: Optional[str] = None
    era_range: Optional[Tuple[int, int]] = None
    historical_period: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'publication_date': self.publication_date,
            'publication_confidence': self.publication_confidence,
            'composition_date': self.composition_date,
            'composition_confidence': self.composition_confidence,
            'primary_era': self.primary_era,
            'era_range': list(self.era_range) if self.era_range else None,
            'historical_period': self.historical_period,
            'sources': self.sources
        }
    
    def get_soft_date(self) -> Optional[str]:
        """Get best available date for soft assignment"""
        if self.publication_date:
            return self.publication_date
        if self.composition_date:
            return self.composition_date
        if self.era_range:
            # Return midpoint of era
            return str((self.era_range[0] + self.era_range[1]) // 2)
        return None
    
    def get_era_description(self) -> Optional[str]:
        """Get human-readable era description"""
        if self.historical_period:
            return self.historical_period
        if self.primary_era:
            return self.primary_era
        if self.era_range:
            start, end = self.era_range
            if end - start <= 10:
                return f"{start}s"
            elif end - start <= 30:
                return f"circa {start}-{end}"
            else:
                return f"{start//100 + 1}th century"
        return None


class TemporalContextAnalyzer:
    """
    Analyzes documents for temporal context.
    
    Extraction strategies:
    1. Filename parsing (year in filename)
    2. Copyright/publication notices
    3. Era references in text
    4. Historical period markers
    """
    
    # Publication date patterns (high confidence)
    PUBLICATION_PATTERNS = [
        (r'copyright\s*[©]?\s*(\d{4})', 0.95, 'copyright'),
        (r'©\s*(\d{4})', 0.95, 'copyright_symbol'),
        (r'published\s+(?:in\s+)?(\d{4})', 0.90, 'published'),
        (r'first\s+published\s*[:\-]?\s*(\d{4})', 0.95, 'first_published'),
        (r'first\s+edition\s*[:\-]?\s*(\d{4})', 0.90, 'first_edition'),
        (r'printed\s+(?:in\s+)?(\d{4})', 0.85, 'printed'),
        (r'\((\d{4})\)\s*$', 0.80, 'trailing_year'),  # Year at end of title
    ]
    
    # Filename year patterns
    FILENAME_YEAR_PATTERNS = [
        (r'(\d{4})', 0.70, 'filename_year'),  # Any 4-digit year
        (r'_(\d{4})_', 0.80, 'filename_year_bounded'),
        (r'-(\d{4})-', 0.80, 'filename_year_bounded'),
    ]
    
    # Era decade patterns
    ERA_PATTERNS = [
        (r'\b(1[789]\d{2})s\b', 'decade'),  # 1880s, 1920s
        (r'\bearly\s+(1[789]\d{2})s\b', 'early_decade'),
        (r'\blate\s+(1[789]\d{2})s\b', 'late_decade'),
        (r'\bmid[- ]?(1[789]\d{2})s\b', 'mid_decade'),
        (r'\b(1[789]\d{2})\s*-\s*(1[789]\d{2})\b', 'year_range'),
    ]
    
    # Historical period markers
    HISTORICAL_PERIODS = {
        r'civil\s+war': ('Civil War', (1861, 1865)),
        r'reconstruction': ('Reconstruction', (1865, 1877)),
        r'gilded\s+age': ('Gilded Age', (1870, 1900)),
        r'progressive\s+era': ('Progressive Era', (1896, 1920)),
        r'world\s+war\s+(?:i|1|one)': ('World War I', (1914, 1918)),
        r'world\s+war\s+(?:ii|2|two)': ('World War II', (1939, 1945)),
        r'prohibition': ('Prohibition Era', (1920, 1933)),
        r'great\s+depression': ('Great Depression', (1929, 1939)),
        r'colonial\s+(?:era|period|america)': ('Colonial Era', (1607, 1776)),
        r'revolutionary\s+(?:war|period)': ('Revolutionary War', (1775, 1783)),
        r'antebellum': ('Antebellum Period', (1812, 1861)),
        r'spanish[- ]american\s+war': ('Spanish-American War', (1898, 1898)),
        r'cold\s+war': ('Cold War', (1947, 1991)),
        r'vietnam\s+(?:war|era)': ('Vietnam War', (1955, 1975)),
    }
    
    # Known document/author date mappings for common corpus items
    KNOWN_DOCUMENTS = {
        r'letters\s+from\s+(?:the\s+)?earth': {'composition': '1909', 'publication': '1962'},
        r'huckleberry\s+finn': {'composition': '1884', 'publication': '1885'},
        r'tom\s+sawyer': {'publication': '1876'},
        r'innocents\s+abroad': {'publication': '1869'},
        r'king\s+leopold.?s\s+soliloquy': {'publication': '1905'},
        r'mysterious\s+stranger': {'composition': '1898', 'publication': '1916'},
        r'war\s+is\s+a\s+racket': {'publication': '1935'},
        r'common\s+sense': {'publication': '1776'},
        r'rights\s+of\s+man': {'publication': '1791'},
        r'age\s+of\s+reason': {'publication': '1794'},
    }
    
    def __init__(self, 
                 min_year: int = 1600,
                 max_year: int = 2030,
                 sample_size: int = 5000):
        """
        Initialize temporal context analyzer.
        
        Args:
            min_year: Earliest valid year to consider
            max_year: Latest valid year to consider  
            sample_size: Characters to sample from document start
        """
        self.min_year = min_year
        self.max_year = max_year
        self.sample_size = sample_size
    
    def analyze(self, 
                document_path: str,
                document_text: str = "",
                title: str = "") -> TemporalContext:
        """
        Analyze document for temporal context.
        
        Args:
            document_path: Filename or path
            document_text: Text content (first pages recommended)
            title: Document title if available
            
        Returns:
            TemporalContext with detected dates and eras
        """
        result = TemporalContext()
        
        # Combine search text
        filename = Path(document_path).stem.lower() if document_path else ""
        search_text = f"{filename} {title} {document_text[:self.sample_size]}".lower()
        
        # 1. Check known documents first (highest confidence)
        known = self._check_known_documents(filename + " " + title)
        if known:
            if 'publication' in known:
                result.publication_date = known['publication']
                result.publication_confidence = 0.98
                result.sources.append('known_document')
            if 'composition' in known:
                result.composition_date = known['composition']
                result.composition_confidence = 0.98
                result.sources.append('known_document')
        
        # 2. Check for publication dates in text
        if not result.publication_date:
            pub_date, pub_conf, pub_source = self._extract_publication_date(search_text)
            if pub_date:
                result.publication_date = pub_date
                result.publication_confidence = pub_conf
                result.sources.append(pub_source)
        
        # 3. Check filename for year
        if not result.publication_date:
            year = self._extract_filename_year(filename)
            if year:
                result.publication_date = year
                result.publication_confidence = 0.70
                result.sources.append('filename_year')
        
        # 4. Detect era from text
        era_info = self._detect_era(search_text)
        if era_info:
            result.primary_era = era_info['era']
            result.era_range = era_info.get('range')
            if 'source' in era_info:
                result.sources.append(era_info['source'])
        
        # 5. Detect historical period
        period_info = self._detect_historical_period(search_text)
        if period_info:
            result.historical_period = period_info['period']
            if not result.era_range:
                result.era_range = period_info['range']
            result.sources.append('historical_period')
        
        # 6. Infer era from dates if not detected
        if not result.era_range and result.publication_date:
            try:
                year = int(result.publication_date[:4])
                decade_start = (year // 10) * 10
                result.era_range = (decade_start, decade_start + 9)
                result.primary_era = f"{decade_start}s"
            except (ValueError, TypeError):
                pass
        
        return result
    
    def _check_known_documents(self, text: str) -> Optional[Dict]:
        """Check if document matches known works"""
        text_lower = text.lower()
        for pattern, dates in self.KNOWN_DOCUMENTS.items():
            if re.search(pattern, text_lower):
                return dates
        return None
    
    def _extract_publication_date(self, text: str) -> Tuple[Optional[str], float, str]:
        """Extract publication date from text"""
        best_date = None
        best_conf = 0.0
        best_source = ""
        
        for pattern, confidence, source in self.PUBLICATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                year = match if isinstance(match, str) else match[0]
                try:
                    year_int = int(year)
                    if self.min_year <= year_int <= self.max_year:
                        if confidence > best_conf:
                            best_date = year
                            best_conf = confidence
                            best_source = source
                except ValueError:
                    continue
        
        return best_date, best_conf, best_source
    
    def _extract_filename_year(self, filename: str) -> Optional[str]:
        """Extract year from filename"""
        for pattern, confidence, source in self.FILENAME_YEAR_PATTERNS:
            matches = re.findall(pattern, filename)
            for match in matches:
                try:
                    year = int(match)
                    if self.min_year <= year <= self.max_year:
                        return str(year)
                except ValueError:
                    continue
        return None
    
    def _detect_era(self, text: str) -> Optional[Dict]:
        """Detect era/decade references"""
        era_mentions = {}
        
        for pattern, era_type in self.ERA_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if era_type == 'year_range':
                    start_year, end_year = int(match[0]), int(match[1])
                    if self.min_year <= start_year and end_year <= self.max_year:
                        key = f"{start_year}-{end_year}"
                        era_mentions[key] = era_mentions.get(key, 0) + 1
                else:
                    decade = int(match) if isinstance(match, str) else int(match[0])
                    if self.min_year <= decade <= self.max_year:
                        key = f"{decade}s"
                        era_mentions[key] = era_mentions.get(key, 0) + 1
        
        if not era_mentions:
            return None
        
        # Return most mentioned era
        most_common = max(era_mentions.items(), key=lambda x: x[1])
        era_str = most_common[0]
        
        if '-' in era_str:
            start, end = era_str.split('-')
            return {
                'era': era_str,
                'range': (int(start), int(end)),
                'source': 'year_range_mention'
            }
        else:
            decade = int(era_str.rstrip('s'))
            return {
                'era': era_str,
                'range': (decade, decade + 9),
                'source': 'decade_mention'
            }
    
    def _detect_historical_period(self, text: str) -> Optional[Dict]:
        """Detect historical period references"""
        period_mentions = {}
        
        for pattern, (period_name, year_range) in self.HISTORICAL_PERIODS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                period_mentions[period_name] = len(matches)
        
        if not period_mentions:
            return None
        
        # Return most mentioned period
        most_common = max(period_mentions.items(), key=lambda x: x[1])
        period_name = most_common[0]
        
        # Look up the range
        for pattern, (name, year_range) in self.HISTORICAL_PERIODS.items():
            if name == period_name:
                return {
                    'period': period_name,
                    'range': year_range
                }
        
        return None


def integrate_with_subject_analyzer():
    """
    Code to add to aegis_subject_analyzer.py to include temporal context.
    
    Add this to the SubjectAnalyzer class:
    """
    integration_code = '''
    # In SubjectAnalyzer.__init__, add:
    from aegis_temporal_context import TemporalContextAnalyzer
    self.temporal_analyzer = TemporalContextAnalyzer()
    
    # Add method to SubjectAnalyzer:
    def analyze_with_temporal(self, document_path: str, document_text: str = "",
                              entities: List[str] = None) -> Tuple[SubjectAnalysisResult, TemporalContext]:
        """
        Analyze document for both topic affinity and temporal context.
        """
        topic_result = self.analyze(document_path, document_text, entities)
        temporal_result = self.temporal_analyzer.analyze(document_path, document_text)
        return topic_result, temporal_result
    '''
    return integration_code


# =============================================================================
# STANDALONE TESTING
# =============================================================================

def test_analyzer():
    """Test the temporal context analyzer"""
    
    analyzer = TemporalContextAnalyzer()
    
    print("=" * 70)
    print("TEMPORAL CONTEXT ANALYZER TEST")
    print("=" * 70)
    
    test_cases = [
        ("Twain - Letters from the Earth.pdf", 
         "Copyright 1962 by Harper & Row. Written by Mark Twain in 1909.",
         "Letters from the Earth"),
        
        ("huckleberry_finn_twain.txt",
         "First published in 1885. Set in the 1840s along the Mississippi.",
         "Adventures of Huckleberry Finn"),
        
        ("war_is_a_racket_butler.pdf",
         "Published 1935. A speech and book by Major General Smedley Butler.",
         "War is a Racket"),
        
        ("unknown_document_1880.pdf",
         "This document discusses events of the Civil War and Reconstruction era.",
         ""),
        
        ("spanish_american_war_analysis.pdf",
         "The Spanish-American War of 1898 marked a turning point...",
         "Spanish-American War Analysis"),
    ]
    
    for path, text, title in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Document: {path}")
        print(f"Title: {title}")
        
        result = analyzer.analyze(path, text, title)
        
        print(f"\nResults:")
        print(f"  Publication date: {result.publication_date} (conf: {result.publication_confidence:.2f})")
        print(f"  Composition date: {result.composition_date} (conf: {result.composition_confidence:.2f})")
        print(f"  Primary era: {result.primary_era}")
        print(f"  Era range: {result.era_range}")
        print(f"  Historical period: {result.historical_period}")
        print(f"  Sources: {result.sources}")
        print(f"  Soft date: {result.get_soft_date()}")
        print(f"  Era description: {result.get_era_description()}")


if __name__ == "__main__":
    test_analyzer()
