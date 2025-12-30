"""
Aegis Insight - Entity Normalizer and Cluster Matcher
Normalizes entity names and matches to existing entities to prevent fragmentation

Pipeline position: After entity extraction, before Neo4j load

Functions:
1. Normalize names (Gen → General, case standardization, whitespace cleanup)
2. Match against existing entities using embedding similarity
3. Create SAME_AS relationships or reuse existing entity

Example:
    Extracted: "Gen Butler", "SMEDLEY D. BUTLER", "General Smedley Butler"
    All normalize and cluster to single canonical entity: "Smedley Butler"
"""

import json
import logging
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# For embedding similarity
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


@dataclass
class NormalizedEntity:
    """Represents a normalized entity."""
    raw_name: str
    normalized_name: str
    canonical_name: Optional[str] = None  # Set if matched to existing
    entity_type: Optional[str] = None
    confidence: float = 1.0
    aliases: Set[str] = field(default_factory=set)
    matched_existing: bool = False
    existing_entity_id: Optional[str] = None


class EntityNormalizer:
    """
    Normalizes entity names for consistency.
    """
    
    # Title/honorific expansions
    TITLE_EXPANSIONS = {
        # Military
        'gen.': 'General',
        'gen ': 'General ',
        'maj.': 'Major',
        'maj ': 'Major ',
        'col.': 'Colonel',
        'col ': 'Colonel ',
        'lt.': 'Lieutenant',
        'lt ': 'Lieutenant ',
        'capt.': 'Captain',
        'capt ': 'Captain ',
        'sgt.': 'Sergeant',
        'sgt ': 'Sergeant ',
        'adm.': 'Admiral',
        'adm ': 'Admiral ',
        
        # Academic/Professional
        'dr.': 'Doctor',
        'dr ': 'Doctor ',
        'prof.': 'Professor',
        'prof ': 'Professor ',
        'mr.': 'Mister',
        'mr ': 'Mister ',
        'mrs.': 'Mistress',
        'mrs ': 'Mistress ',
        'ms.': 'Ms',
        'ms ': 'Ms ',
        
        # Political
        'sen.': 'Senator',
        'sen ': 'Senator ',
        'rep.': 'Representative',
        'rep ': 'Representative ',
        'gov.': 'Governor',
        'gov ': 'Governor ',
        'pres.': 'President',
        'pres ': 'President ',
        
        # Religious
        'rev.': 'Reverend',
        'rev ': 'Reverend ',
        'fr.': 'Father',
        'fr ': 'Father ',
        'sr.': 'Sister',
        'sr ': 'Sister ',
    }
    
    # Words to remove for core name extraction
    TITLE_WORDS = {
        'general', 'major', 'colonel', 'lieutenant', 'captain', 'sergeant', 'admiral',
        'doctor', 'professor', 'mister', 'mistress', 'ms',
        'senator', 'representative', 'governor', 'president',
        'reverend', 'father', 'sister', 'sir', 'lord', 'lady', 'dame'
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def normalize(self, raw_name: str, entity_type: Optional[str] = None) -> NormalizedEntity:
        """
        Normalize an entity name.
        
        Args:
            raw_name: The raw extracted entity name
            entity_type: Optional entity type (Person, Organization, etc.)
            
        Returns:
            NormalizedEntity with normalized name and metadata
        """
        if not raw_name:
            return NormalizedEntity(raw_name="", normalized_name="")
        
        name = raw_name.strip()
        
        # Step 1: Normalize whitespace
        name = ' '.join(name.split())
        
        # Step 2: Expand abbreviations (case-insensitive)
        name_lower = name.lower()
        for abbrev, expansion in self.TITLE_EXPANSIONS.items():
            if abbrev in name_lower:
                # Find the position and replace preserving following case
                idx = name_lower.find(abbrev)
                name = name[:idx] + expansion + name[idx + len(abbrev):]
                name_lower = name.lower()
        
        # Step 3: Smart title case (preserve acronyms)
        name = self._smart_title_case(name)
        
        # Step 4: Extract core name (without titles) for matching
        core_name = self._extract_core_name(name)
        
        return NormalizedEntity(
            raw_name=raw_name,
            normalized_name=name,
            canonical_name=core_name,  # Will be updated if matched
            entity_type=entity_type,
            aliases={raw_name} if raw_name != name else set()
        )
    
    def _smart_title_case(self, name: str) -> str:
        """
        Title case that preserves acronyms and handles special cases.
        
        Rules:
        - True acronyms (USS, NATO, FBI): preserve all caps
        - Words following ship/vehicle prefixes (USS, HMS, RMS): preserve original case
        - Shouted names (SMEDLEY D. BUTLER): convert to title case
        - Mixed case: preserve
        - Lowercase: capitalize
        
        Heuristic: If the word is all caps AND:
        - Length <= 4 AND no vowels except maybe one → likely acronym
        - OR is in known acronym list → preserve
        - Otherwise → title case (likely shouted name)
        """
        KNOWN_ACRONYMS = {
            'USS', 'NATO', 'FBI', 'CIA', 'NSA', 'USA', 'UK', 'UN', 'EU',
            'CEO', 'CFO', 'CTO', 'PhD', 'MD', 'JD', 'MBA', 'BA', 'BS', 'MA',
            'WWII', 'WWI', 'NYPD', 'LAPD', 'IRS', 'FDA', 'CDC', 'NIH',
            'MIT', 'UCLA', 'USC', 'NYU', 'USMC', 'USAF', 'USN',
            'HMS', 'RMS', 'SS', 'MV',  # Ship prefixes
        }
        
        # Prefixes where following word should preserve case (ship names, etc.)
        PRESERVE_FOLLOWING = {'USS', 'HMS', 'RMS', 'SS', 'MV'}
        
        words = name.split()
        result = []
        preserve_next = False
        
        for word in words:
            # Check if previous word was a prefix that preserves following case
            if preserve_next:
                result.append(word)  # Keep original case
                preserve_next = False
                continue
            
            # Check if this is a prefix that preserves the following word
            if word.upper() in PRESERVE_FOLLOWING:
                result.append(word.upper())
                preserve_next = True
                continue
                
            # Check if it's a known acronym
            if word.upper() in KNOWN_ACRONYMS:
                result.append(word.upper())
            # All caps - decide if acronym or shouted
            elif word.isupper() and len(word) > 1:
                # Short words (<=4 chars) with few vowels are likely acronyms
                vowel_count = sum(1 for c in word if c in 'AEIOU')
                if len(word) <= 4 and vowel_count <= 1:
                    result.append(word)  # Preserve as acronym
                else:
                    # Longer all-caps words are likely shouted names
                    result.append(word.capitalize())
            # Single uppercase letter (middle initial) - preserve
            elif word.isupper() and len(word) == 1:
                result.append(word)
            # Mixed case already = preserve
            elif not word.islower() and not word.isupper():
                result.append(word)
            # Otherwise title case
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def _extract_core_name(self, name: str) -> str:
        """
        Extract core name without titles for matching.
        "General Smedley Butler" → "Smedley Butler"
        """
        words = name.split()
        core_words = [w for w in words if w.lower() not in self.TITLE_WORDS]
        return ' '.join(core_words) if core_words else name


class EntityClusterMatcher:
    """
    Matches new entities against existing entities using embedding similarity.
    Creates clusters of equivalent entities.
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.85,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize matcher.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider a match
            embedding_model: Sentence transformer model for embeddings
            logger: Optional logger
        """
        self.similarity_threshold = similarity_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embedder = None
        if HAS_EMBEDDINGS:
            try:
                self.embedder = SentenceTransformer(embedding_model, local_files_only=True)
                self.logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                self.logger.warning(f"Could not load embedding model: {e}")
        
        # Cache of existing entities: name → embedding
        self.entity_cache: Dict[str, np.ndarray] = {}
        
        # Known clusters: canonical_name → set of variant names
        self.clusters: Dict[str, Set[str]] = defaultdict(set)
        
        # Reverse lookup: variant_name → canonical_name
        self.variant_to_canonical: Dict[str, str] = {}
    
    def load_existing_entities(self, entities: List[Dict]):
        """
        Load existing entities from database into cache.
        
        Args:
            entities: List of dicts with 'name', 'entity_type', optional 'aliases'
        """
        names = [e['name'] for e in entities]
        
        if self.embedder and names:
            embeddings = self.embedder.encode(names, show_progress_bar=False)
            for name, emb in zip(names, embeddings):
                self.entity_cache[name] = emb
                
                # Also cache aliases
                for entity in entities:
                    if entity['name'] == name:
                        for alias in entity.get('aliases', []):
                            self.variant_to_canonical[alias.lower()] = name
                        break
        
        self.logger.info(f"Loaded {len(names)} existing entities into cache")
    
    def find_match(self, 
                   normalized_entity: NormalizedEntity,
                   context_text: Optional[str] = None) -> Optional[str]:
        """
        Find matching existing entity for a normalized entity.
        
        Args:
            normalized_entity: The normalized entity to match
            context_text: Optional context for better matching
            
        Returns:
            Canonical entity name if match found, None otherwise
        """
        name = normalized_entity.normalized_name
        core_name = normalized_entity.canonical_name or name
        
        # Step 1: Exact match on name or core name
        if name in self.entity_cache:
            return name
        if core_name in self.entity_cache:
            return core_name
        
        # Step 2: Check known variants
        if name.lower() in self.variant_to_canonical:
            return self.variant_to_canonical[name.lower()]
        if core_name.lower() in self.variant_to_canonical:
            return self.variant_to_canonical[core_name.lower()]
        
        # Step 3: Embedding similarity search
        if self.embedder and self.entity_cache:
            return self._find_similar_entity(name, core_name, context_text)
        
        return None
    
    def _find_similar_entity(self,
                             name: str,
                             core_name: str,
                             context_text: Optional[str] = None) -> Optional[str]:
        """Find similar entity using embeddings."""
        
        if not HAS_NUMPY:
            return None
        
        # Encode the query (use core name for better matching)
        query = core_name
        if context_text:
            query = f"{core_name} ({context_text[:100]})"
        
        query_emb = self.embedder.encode([query], show_progress_bar=False)[0]
        
        # Compare against all cached entities
        best_match = None
        best_score = 0.0
        
        for entity_name, entity_emb in self.entity_cache.items():
            score = self._cosine_similarity(query_emb, entity_emb)
            if score > best_score:
                best_score = score
                best_match = entity_name
        
        if best_score >= self.similarity_threshold:
            self.logger.debug(f"Matched '{name}' to '{best_match}' (score: {best_score:.3f})")
            return best_match
        
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def register_entity(self, 
                        normalized_entity: NormalizedEntity,
                        canonical_name: Optional[str] = None):
        """
        Register an entity in the cache and clusters.
        
        Args:
            normalized_entity: The entity to register
            canonical_name: If matched to existing, the canonical name
        """
        name = normalized_entity.normalized_name
        
        if canonical_name:
            # This is a variant of an existing entity
            self.clusters[canonical_name].add(name)
            self.variant_to_canonical[name.lower()] = canonical_name
            
            # Add raw name as variant too
            if normalized_entity.raw_name != name:
                self.clusters[canonical_name].add(normalized_entity.raw_name)
                self.variant_to_canonical[normalized_entity.raw_name.lower()] = canonical_name
        else:
            # This is a new canonical entity
            canonical = normalized_entity.canonical_name or name
            self.clusters[canonical].add(name)
            
            # Cache embedding
            if self.embedder:
                emb = self.embedder.encode([canonical], show_progress_bar=False)[0]
                self.entity_cache[canonical] = emb
    
    def get_clusters(self) -> Dict[str, Set[str]]:
        """Return all entity clusters."""
        return dict(self.clusters)


class EntityProcessor:
    """
    High-level entity processor combining normalization and clustering.
    This is the main interface for the ingestion pipeline.
    """
    
    def __init__(self,
                 neo4j_driver = None,
                 similarity_threshold: float = 0.85,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize processor.
        
        Args:
            neo4j_driver: Neo4j driver for loading existing entities
            similarity_threshold: Threshold for entity matching
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.normalizer = EntityNormalizer(logger=self.logger)
        self.matcher = EntityClusterMatcher(
            similarity_threshold=similarity_threshold,
            logger=self.logger
        )
        self.neo4j_driver = neo4j_driver
        
        # Stats
        self.stats = {
            'entities_processed': 0,
            'matched_existing': 0,
            'new_entities': 0,
            'aliases_created': 0
        }
        
        # Load existing entities if driver provided
        if neo4j_driver:
            self._load_existing_entities()
    
    def _load_existing_entities(self):
        """Load existing entities from Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name as name, 
                           e.entity_type as entity_type,
                           e.aliases as aliases
                    LIMIT 50000
                """)
                
                entities = []
                for record in result:
                    entities.append({
                        'name': record['name'],
                        'entity_type': record['entity_type'],
                        'aliases': record['aliases'] or []
                    })
                
                self.matcher.load_existing_entities(entities)
                self.logger.info(f"Loaded {len(entities)} existing entities")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing entities: {e}")
    
    def process(self, 
                raw_name: str, 
                entity_type: Optional[str] = None,
                context_text: Optional[str] = None) -> NormalizedEntity:
        """
        Process an extracted entity: normalize and match to existing.
        
        Args:
            raw_name: Raw entity name from extraction
            entity_type: Entity type (Person, Organization, etc.)
            context_text: Optional context for better matching
            
        Returns:
            NormalizedEntity with canonical name and match status
        """
        self.stats['entities_processed'] += 1
        
        # Step 1: Normalize
        normalized = self.normalizer.normalize(raw_name, entity_type)
        
        # Step 2: Find match
        matched_canonical = self.matcher.find_match(normalized, context_text)
        
        if matched_canonical:
            # Found existing entity
            normalized.canonical_name = matched_canonical
            normalized.matched_existing = True
            self.stats['matched_existing'] += 1
            
            if raw_name != matched_canonical:
                self.stats['aliases_created'] += 1
        else:
            # New entity
            self.stats['new_entities'] += 1
        
        # Step 3: Register in cache
        self.matcher.register_entity(normalized, matched_canonical)
        
        return normalized
    
    def process_batch(self,
                      entities: List[Dict],
                      context_text: Optional[str] = None) -> List[NormalizedEntity]:
        """
        Process a batch of entities.
        
        Args:
            entities: List of dicts with 'name' and optional 'type'
            context_text: Optional context for matching
            
        Returns:
            List of NormalizedEntity objects
        """
        results = []
        for entity in entities:
            result = self.process(
                raw_name=entity.get('name', ''),
                entity_type=entity.get('type'),
                context_text=context_text
            )
            results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return self.stats.copy()
    
    def get_clusters(self) -> Dict[str, Set[str]]:
        """Return all entity clusters."""
        return self.matcher.get_clusters()


# =============================================================================
# TESTS
# =============================================================================

def test_entity_normalizer():
    """Test entity name normalization."""
    print("="*70)
    print("ENTITY NORMALIZER TEST")
    print("="*70)
    
    normalizer = EntityNormalizer()
    
    test_cases = [
        ("Gen Butler", "General Butler"),
        ("gen. smedley butler", "General Smedley Butler"),
        ("SMEDLEY D. BUTLER", "Smedley D. Butler"),
        ("Dr. Anthony Fauci", "Doctor Anthony Fauci"),
        ("Sen. McCarthy", "Senator McCarthy"),
        ("thomas paine", "Thomas Paine"),
        ("USS MAINE", "USS MAINE"),  # Preserve acronym
        ("  extra   spaces  ", "Extra Spaces"),
    ]
    
    passed = 0
    for raw, expected in test_cases:
        result = normalizer.normalize(raw)
        status = "✅" if result.normalized_name == expected else "❌"
        print(f"{status} '{raw}' → '{result.normalized_name}' (expected: '{expected}')")
        if result.normalized_name == expected:
            passed += 1
    
    print(f"\nResults: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_entity_clustering():
    """Test entity clustering without database."""
    print("\n" + "="*70)
    print("ENTITY CLUSTERING TEST")
    print("="*70)
    
    processor = EntityProcessor(neo4j_driver=None, similarity_threshold=0.80)
    
    # Simulate processing multiple variants of same person
    variants = [
        "Smedley Butler",
        "Gen Butler", 
        "General Smedley Butler",
        "SMEDLEY D. BUTLER",
        "Maj. Gen. Smedley D. Butler",
        "Butler",  # This one might not match due to ambiguity
    ]
    
    print("\nProcessing Butler variants:")
    for name in variants:
        result = processor.process(name, entity_type="Person")
        match_status = f"→ {result.canonical_name}" if result.matched_existing else "(new)"
        print(f"  '{name}' {match_status}")
    
    print(f"\nClusters formed:")
    for canonical, variants in processor.get_clusters().items():
        print(f"  {canonical}: {variants}")
    
    print(f"\nStats: {processor.get_stats()}")
    
    # Check that most variants clustered together
    clusters = processor.get_clusters()
    max_cluster_size = max(len(v) for v in clusters.values()) if clusters else 0
    
    success = max_cluster_size >= 4  # At least 4 variants should cluster
    print(f"\n{'✅' if success else '❌'} Largest cluster: {max_cluster_size} variants")
    
    return success


def test_all():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING ALL ENTITY PROCESSOR TESTS")
    print("="*70 + "\n")
    
    results = []
    results.append(("Normalizer", test_entity_normalizer()))
    results.append(("Clustering", test_entity_clustering()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")
    
    return all(r[1] for r in results)


if __name__ == "__main__":
    test_all()
