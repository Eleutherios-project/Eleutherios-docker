import os
"""
Aegis Insight - Graph Builder v2
Builds Neo4j knowledge graph from extracted data
Now with EntityProcessor integration for entity clustering

Key improvements over v1:
- Uses EntityProcessor for name normalization and clustering
- Creates SAME_AS relationships between entity variants
- Tracks canonical entities to prevent fragmentation
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from neo4j import GraphDatabase
from datetime import datetime
from aegis_config import Config

# Topic affinity support
try:
    from aegis_subject_analyzer import SubjectAnalyzer
    HAS_SUBJECT_ANALYZER = True
except ImportError:
    HAS_SUBJECT_ANALYZER = False
    print("Note: SubjectAnalyzer not available - topic detection disabled")


# Import entity processor for clustering
try:
    from aegis_entity_processor import EntityProcessor, NormalizedEntity
    HAS_ENTITY_PROCESSOR = True
except ImportError:
    HAS_ENTITY_PROCESSOR = False
    print("Warning: EntityProcessor not available. Entity clustering disabled.")


class GraphBuilderV2:
    """
    Builds Neo4j graph from extracted data with entity clustering.
    
    Features:
    - Creates Document, Chunk, Entity, Claim nodes
    - Creates CONTAINS, MENTIONS, CONTAINS_CLAIM relationships
    - NEW: Creates SAME_AS relationships between entity variants
    - NEW: Uses EntityProcessor for clustering
    - NEW: Topic affinity detection on import
    - Proper deduplication using MERGE
    - Batch operations for performance
    """
    
    def __init__(self,
                 neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = None,
                 enable_clustering: bool = True,
                 similarity_threshold: float = 0.85,
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or logging.getLogger(__name__)
        self.enable_clustering = enable_clustering and HAS_ENTITY_PROCESSOR
        
        # Connect to Neo4j
        # Use config fallback if password not provided
        neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            self.driver.verify_connectivity()
            self.logger.info("[OK] Connected to Neo4j")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        # Initialize subject analyzer for topic detection
        self.subject_analyzer = None
        if HAS_SUBJECT_ANALYZER:
            try:
                self.subject_analyzer = SubjectAnalyzer(llm_enabled=True, logger=self.logger)
                self.logger.info("[OK] Topic detection enabled")
            except Exception as e:
                self.logger.debug(f"Topic detection not available: {e}")
        
        # Initialize entity processor for clustering
        self.entity_processor = None
        if self.enable_clustering:
            try:
                self.entity_processor = EntityProcessor(
                    neo4j_driver=self.driver,
                    similarity_threshold=similarity_threshold,
                    logger=self.logger
                )
                self.logger.info(f"[OK] Entity clustering enabled (threshold: {similarity_threshold})")
            except Exception as e:
                self.logger.warning(f"Entity clustering disabled: {e}")
                self.enable_clustering = False
        
        # Track created SAME_AS relationships to avoid duplicates
        self.same_as_created: Set[tuple] = set()
    
    def build_graph(self, extracted_data: List[Dict]) -> Dict:
        """
        Build Neo4j graph from extracted data
        
        Args:
            extracted_data: List of extraction results from orchestrator
            
        Returns:
            Dict with stats on nodes/relationships created
        """
        
        if not extracted_data:
            self.logger.warning("No data to build graph from")
            return {'nodes_created': 0, 'relationships_created': 0}
        
        self.logger.info(f"Building graph from {len(extracted_data)} chunks...")
        
        stats = {
            'documents': 0,
            'chunks': 0,
            'entities': 0,
            'entities_matched_existing': 0,
            'claims': 0,
            'contains_rels': 0,
            'mentions_rels': 0,
            'contains_claim_rels': 0,
            'same_as_rels': 0,
            'maybe_same_as_rels': 0,
            'topics_created': 0
        }
        
        # Track documents we've already analyzed for topics (avoid re-analyzing per chunk)
        docs_analyzed_for_topics = set()
        
        with self.driver.session() as session:
            for item in extracted_data:
                # Create document node
                doc_id = item['context']['source_file']
                session.execute_write(self._create_document, item['context'])
                stats['documents'] += 1
                
                # Analyze and create topic relationships (once per document)
                if doc_id not in docs_analyzed_for_topics:
                    if self.subject_analyzer:
                        sample = item.get('text', item.get('chunk_text', ''))[:1000]
                        topics = self._analyze_and_create_topics(session, doc_id, sample)
                        stats['topics_created'] += len(topics)
                    docs_analyzed_for_topics.add(doc_id)
                
                # Create chunk node
                session.execute_write(self._create_chunk, item, doc_id)
                stats['chunks'] += 1
                
                # Create entities with clustering
                for entity in item.get('entities', []):
                    result = self._process_entity(session, entity, item['chunk_id'])
                    stats['entities'] += 1
                    stats['mentions_rels'] += 1
                    if result.get('matched_existing'):
                        stats['entities_matched_existing'] += 1
                    if result.get('same_as_created'):
                        stats['same_as_rels'] += 1
                    if result.get('maybe_same_as_created'):
                        stats['maybe_same_as_rels'] += 1
                
                # Create claims and relationships
                for claim in item.get('claims', []):
                    session.execute_write(self._create_claim, claim, item['context'])
                    session.execute_write(self._create_contains_claim_rel, item['chunk_id'], claim)
                    stats['claims'] += 1
                    stats['contains_claim_rels'] += 1
                
                # Create chunk->document relationship
                session.execute_write(self._create_contains_rel, doc_id, item['chunk_id'])
                stats['contains_rels'] += 1
        
        self.logger.info("Graph building complete!")
        self._print_stats(stats)
        
        return stats
    
    def _process_entity(self, session, entity: Dict, chunk_id: str) -> Dict:
        """
        Process entity with optional clustering using tiered confidence.
        
        Confidence tiers:
        - 95%+: Auto-link with SAME_AS
        - 70-94%: Queue for user verification with MAYBE_SAME_AS
        - <70%: No automatic linking
        
        Returns dict with:
        - matched_existing: bool
        - same_as_created: bool
        - maybe_same_as_created: bool
        - canonical_name: str
        - confidence: float
        """
        result = {
            'matched_existing': False,
            'same_as_created': False,
            'maybe_same_as_created': False,
            'canonical_name': entity.get('name'),
            'confidence': 0.0
        }
        
        raw_name = entity.get('name')
        entity_type = entity.get('type')
        
        # Use entity processor if available
        if self.enable_clustering and self.entity_processor:
            normalized = self.entity_processor.process(raw_name, entity_type)
            
            if normalized.matched_existing:
                result['matched_existing'] = True
                result['canonical_name'] = normalized.canonical_name
                result['confidence'] = normalized.confidence
                
                # Create entity with normalized name
                entity_copy = entity.copy()
                entity_copy['canonical_name'] = normalized.canonical_name
                entity_copy['raw_name'] = raw_name
                entity_copy['match_confidence'] = normalized.confidence
                
                session.execute_write(self._create_entity_with_canonical, entity_copy)
                session.execute_write(self._create_mentions_rel, chunk_id, entity_copy)
                
                # Create SAME_AS or MAYBE_SAME_AS based on confidence
                if normalized.confidence >= 0.95:
                    # High confidence - auto-link
                    if raw_name.lower() != normalized.canonical_name.lower():
                        rel_key = (raw_name.lower(), normalized.canonical_name.lower())
                        if rel_key not in self.same_as_created:
                            session.execute_write(
                                self._create_same_as_rel,
                                raw_name, normalized.canonical_name, normalized.confidence
                            )
                            self.same_as_created.add(rel_key)
                            result['same_as_created'] = True
                            self.logger.debug(f"SAME_AS: '{raw_name}' -> '{normalized.canonical_name}' ({normalized.confidence:.2f})")
                
                elif normalized.confidence >= 0.70:
                    # Medium confidence - queue for verification
                    if raw_name.lower() != normalized.canonical_name.lower():
                        rel_key = (raw_name.lower(), normalized.canonical_name.lower())
                        if rel_key not in self.same_as_created:
                            session.execute_write(
                                self._create_maybe_same_as_rel,
                                raw_name, normalized.canonical_name, normalized.confidence
                            )
                            self.same_as_created.add(rel_key)
                            result['maybe_same_as_created'] = True
                            self.logger.debug(f"MAYBE_SAME_AS: '{raw_name}' -> '{normalized.canonical_name}' ({normalized.confidence:.2f})")
                
                return result
        
        # Standard entity creation (no clustering match)
        session.execute_write(self._create_entity, entity)
        session.execute_write(self._create_mentions_rel, chunk_id, entity)
        return result
    
    @staticmethod
    def _create_document(tx, context: Dict):
        """Create or update document node"""
        query = """
        MERGE (d:Document {source_file: $source_file})
        ON CREATE SET
            d.title = $title,
            d.domain = $domain,
            d.created_at = datetime()
        ON MATCH SET
            d.title = COALESCE($title, d.title),
            d.domain = COALESCE($domain, d.domain)
        """
        tx.run(query,
               source_file=context.get('source_file'),
               title=context.get('title'),
               domain=context.get('domain'))
    
    @staticmethod
    def _create_chunk(tx, item: Dict, doc_id: str):
        """Create chunk node"""
        query = """
        MERGE (ch:Chunk {chunk_id: $chunk_id})
        ON CREATE SET
            ch.text = $text,
            ch.chunk_index = $chunk_index,
            ch.source_file = $source_file,
            ch.created_at = datetime()
        """
        tx.run(query,
               chunk_id=item['chunk_id'],
               text=item.get('text', item.get('chunk_text', '')),
               chunk_index=item.get('chunk_index', 0),
               source_file=doc_id)
    
    @staticmethod
    def _create_entity(tx, entity: Dict):
        """Create entity node (basic)"""
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.created_at = datetime()
        ON MATCH SET
            e.type = COALESCE($type, e.type)
        """
        tx.run(query,
               name=entity.get('name'),
               type=entity.get('type'))
    
    @staticmethod
    def _create_entity_with_canonical(tx, entity: Dict):
        """Create entity node with canonical reference"""
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.canonical_name = $canonical_name,
            e.raw_name = $raw_name,
            e.match_confidence = $match_confidence,
            e.created_at = datetime()
        ON MATCH SET
            e.type = COALESCE($type, e.type),
            e.canonical_name = COALESCE($canonical_name, e.canonical_name)
        """
        tx.run(query,
               name=entity.get('name'),
               type=entity.get('type'),
               canonical_name=entity.get('canonical_name'),
               raw_name=entity.get('raw_name'),
               match_confidence=entity.get('match_confidence'))
    
    @staticmethod
    def _create_same_as_rel(tx, name1: str, name2: str, confidence: float):
        """Create SAME_AS relationship between entities"""
        query = """
        MATCH (e1:Entity {name: $name1})
        MATCH (e2:Entity {name: $name2})
        WHERE e1 <> e2
        MERGE (e1)-[r:SAME_AS]->(e2)
        ON CREATE SET
            r.confidence = $confidence,
            r.auto_linked = true,
            r.created_at = datetime()
        """
        tx.run(query, name1=name1, name2=name2, confidence=confidence)
    
    @staticmethod
    def _create_maybe_same_as_rel(tx, name1: str, name2: str, confidence: float):
        """Create MAYBE_SAME_AS relationship for user verification"""
        query = """
        MATCH (e1:Entity {name: $name1})
        MATCH (e2:Entity {name: $name2})
        WHERE e1 <> e2
        MERGE (e1)-[r:MAYBE_SAME_AS]->(e2)
        ON CREATE SET
            r.confidence = $confidence,
            r.verified = false,
            r.created_at = datetime()
        """
        tx.run(query, name1=name1, name2=name2, confidence=confidence)
    
    @staticmethod
    def _create_claim(tx, claim: Dict, context: Dict):
        """Create claim node"""
        query = """
        MERGE (c:Claim {claim_id: $claim_id})
        ON CREATE SET
            c.claim_text = $claim_text,
            c.claim_type = $claim_type,
            c.confidence = $confidence,
            c.source_file = $source_file,
            c.temporal_data = $temporal_data,
            c.geographic_data = $geographic_data,
            c.citation_data = $citation_data,
            c.emotional_data = $emotional_data,
            c.authority_data = $authority_data,
            c.created_at = datetime()
        """
        tx.run(query,
               claim_id=claim.get('claim_id'),
               claim_text=claim.get('claim_text'),
               claim_type=claim.get('claim_type'),
               confidence=claim.get('confidence', 0.8),
               source_file=context.get('source_file'),
               temporal_data=json.dumps(claim.get('temporal_data', {})) if claim.get('temporal_data') else None,
               geographic_data=json.dumps(claim.get('geographic_data', {})) if claim.get('geographic_data') else None,
               citation_data=json.dumps(claim.get('citation_data', {})) if claim.get('citation_data') else None,
               emotional_data=json.dumps(claim.get('emotional_data', {})) if claim.get('emotional_data') else None,
               authority_data=json.dumps(claim.get('authority_data', {})) if claim.get('authority_data') else None)
    
    @staticmethod
    def _create_mentions_rel(tx, chunk_id: str, entity: Dict):
        """Create MENTIONS relationship between chunk and entity"""
        query = """
        MATCH (ch:Chunk {chunk_id: $chunk_id})
        MATCH (e:Entity {name: $entity_name})
        MERGE (ch)-[r:MENTIONS]->(e)
        ON CREATE SET
            r.created_at = datetime()
        """
        tx.run(query,
               chunk_id=chunk_id,
               entity_name=entity.get('name'))
    
    @staticmethod
    def _create_contains_rel(tx, doc_id: str, chunk_id: str):
        """Create CONTAINS relationship between document and chunk"""
        query = """
        MATCH (d:Document {source_file: $doc_id})
        MATCH (ch:Chunk {chunk_id: $chunk_id})
        MERGE (d)-[r:CONTAINS]->(ch)
        ON CREATE SET
            r.created_at = datetime()
        """
        tx.run(query, doc_id=doc_id, chunk_id=chunk_id)
    
    @staticmethod
    def _create_contains_claim_rel(tx, chunk_id: str, claim: Dict):
        """Create CONTAINS_CLAIM relationship between chunk and claim"""
        query = """
        MATCH (ch:Chunk {chunk_id: $chunk_id})
        MATCH (c:Claim {claim_id: $claim_id})
        MERGE (ch)-[r:CONTAINS_CLAIM]->(c)
        ON CREATE SET
            r.created_at = datetime()
        """
        tx.run(query,
               chunk_id=chunk_id,
               claim_id=claim.get('claim_id'))
    
    def _print_stats(self, stats: Dict):
        """Print summary statistics"""
        self.logger.info("="*60)
        self.logger.info("GRAPH BUILDING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Documents created: {stats['documents']}")
        self.logger.info(f"Chunks created: {stats['chunks']}")
        self.logger.info(f"Entities created: {stats['entities']}")
        self.logger.info(f"  - Matched existing: {stats['entities_matched_existing']}")
        self.logger.info(f"Claims created: {stats['claims']}")
        self.logger.info(f"Topics created: {stats.get('topics_created', 0)}")
        self.logger.info(f"CONTAINS relationships: {stats['contains_rels']}")
        self.logger.info(f"MENTIONS relationships: {stats['mentions_rels']}")
        self.logger.info(f"CONTAINS_CLAIM relationships: {stats['contains_claim_rels']}")
        self.logger.info(f"SAME_AS relationships (auto-linked): {stats['same_as_rels']}")
        self.logger.info(f"MAYBE_SAME_AS relationships (pending verification): {stats['maybe_same_as_rels']}")
        self.logger.info("="*60 + "\n")
        
        if self.entity_processor:
            ep_stats = self.entity_processor.get_stats()
            self.logger.info("ENTITY PROCESSOR STATS:")
            self.logger.info(f"  Processed: {ep_stats['entities_processed']}")
            self.logger.info(f"  Matched: {ep_stats['matched_existing']}")
            self.logger.info(f"  New: {ep_stats['new_entities']}")
            self.logger.info(f"  Aliases: {ep_stats['aliases_created']}")
    
    def _analyze_and_create_topics(self, session, doc_id: str, sample_text: str = ""):
        """Analyze document for topics and create ABOUT relationships"""
        if not self.subject_analyzer:
            return []
        
        result = self.subject_analyzer.analyze(document_path=doc_id, document_text=sample_text)
        if not result.all_matches:
            return []
        
        topics_created = []
        for match in result.all_matches:
            self._create_topic_relationship(
                session, doc_id, match.topic, match.topic_type,
                match.confidence, match.source, match.is_primary, match.aliases
            )
            topics_created.append(match.topic)
        
        if topics_created:
            self.logger.debug(f"Topics for {doc_id[:40]}: {topics_created}")
        return topics_created
    
    def _create_topic_relationship(self, session, doc_id: str, topic_name: str,
                                   topic_type: str, confidence: float, source: str,
                                   is_primary: bool, aliases: list):
        """Create Topic node (separate label) and ABOUT relationship"""
        session.run("""
            MERGE (t:Topic {name: $topic_name})
            ON CREATE SET t.topic_type = $topic_type, 
                          t.aliases = $aliases,
                          t.created_at = datetime(), 
                          t.document_count = 0
            ON MATCH SET t.topic_type = COALESCE(t.topic_type, $topic_type),
                         t.aliases = CASE 
                             WHEN t.aliases IS NULL THEN $aliases
                             ELSE t.aliases 
                         END
            WITH t
            MATCH (d:Document {source_file: $doc_id})
            MERGE (d)-[r:ABOUT]->(t)
            ON CREATE SET r.confidence = $confidence, 
                          r.source = $source,
                          r.is_primary = $is_primary, 
                          r.detected_at = datetime()
            SET t.document_count = t.document_count + 1,
                d.topics = CASE WHEN d.topics IS NULL THEN [$topic_name]
                           WHEN NOT $topic_name IN d.topics THEN d.topics + $topic_name
                           ELSE d.topics END
            WITH d WHERE $is_primary
            SET d.primary_topic = $topic_name
        """, doc_id=doc_id, topic_name=topic_name, topic_type=topic_type,
             confidence=confidence, source=source, is_primary=is_primary,
             aliases=aliases or [])

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file (one JSON object per line)"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping line {line_num}: {e}")
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build Neo4j graph from extracted JSONL files (v2 with clustering)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python aegis_graph_builder.py file.jsonl
    python aegis_graph_builder.py *.jsonl
    python aegis_graph_builder.py --dir /path/to/checkpoints/
    python aegis_graph_builder.py --dir /path/to/checkpoints/ --enable-clustering
        """
    )
    parser.add_argument('files', nargs='*', help='JSONL files to load')
    parser.add_argument('--dir', '-d', type=str, help='Directory containing *_extracted.jsonl files')
    parser.add_argument('--neo4j-uri', default='os.environ.get("NEO4J_URI", "bolt://localhost:7687")', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j user')
    parser.add_argument('--neo4j-password', default='aegistrusted', help='Neo4j password')
    parser.add_argument('--enable-clustering', action='store_true', default=True,
                        help='Enable entity clustering (default: True)')
    parser.add_argument('--no-clustering', action='store_false', dest='enable_clustering',
                        help='Disable entity clustering')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                        help='Entity similarity threshold (default: 0.85)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Collect files to process
    files_to_process = []
    
    if args.dir:
        dir_path = Path(args.dir)
        files_to_process.extend(sorted(dir_path.glob('*_extracted.jsonl')))
    
    if args.files:
        for f in args.files:
            p = Path(f)
            if p.exists():
                files_to_process.append(p)
            else:
                logging.warning(f"File not found: {f}")
    
    if not files_to_process:
        print("No files to process. Provide JSONL files or use --dir")
        print("Usage: python aegis_graph_builder.py file.jsonl")
        print("       python aegis_graph_builder.py --dir /path/to/checkpoints/")
        sys.exit(1)
    
    print(f"Found {len(files_to_process)} files to process")
    print(f"Entity clustering: {'enabled' if args.enable_clustering else 'disabled'}")
    
    # Build graph
    builder = GraphBuilderV2(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        enable_clustering=args.enable_clustering,
        similarity_threshold=args.similarity_threshold
    )
    
    total_stats = {
        'documents': 0,
        'chunks': 0,
        'entities': 0,
        'entities_matched_existing': 0,
        'claims': 0,
        'contains_rels': 0,
        'mentions_rels': 0,
        'contains_claim_rels': 0,
        'same_as_rels': 0,
        'maybe_same_as_rels': 0,
        'topics_created': 0
    }
    
    for filepath in files_to_process:
        print(f"\nLoading: {filepath.name}")
        
        data = load_jsonl(filepath)
        if not data:
            print(f"  ⚠ No data in file")
            continue
        
        print(f"  {len(data)} chunks loaded")
        stats = builder.build_graph(data)
        
        # Accumulate stats
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    builder.close()
    
    # Print totals
    print("\n" + "="*60)
    print("TOTAL GRAPH BUILDING SUMMARY")
    print("="*60)
    print(f"Files processed: {len(files_to_process)}")
    print(f"Documents created: {total_stats['documents']}")
    print(f"Chunks created: {total_stats['chunks']}")
    print(f"Entities created: {total_stats['entities']}")
    print(f"  - Matched existing: {total_stats['entities_matched_existing']}")
    print(f"Claims created: {total_stats['claims']}")
    print(f"Topics created: {total_stats['topics_created']}")
    print(f"SAME_AS (auto-linked): {total_stats['same_as_rels']}")
    print(f"MAYBE_SAME_AS (pending): {total_stats['maybe_same_as_rels']}")
    print("="*60)


if __name__ == "__main__":
    main()
