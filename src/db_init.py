#!/usr/bin/env python3
"""
AegisTrustNet - Neo4j Database Initialization and Migration Tool
This script initializes the Neo4j database structure for the Truth Network Overlay
and migrates existing data from the Pattern RAG SQLite database.

Requirements:
- neo4j (pip install neo4j)
- sqlite3
"""

import os
import sys
import json
import sqlite3
import time
import logging
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pickle
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('aegis-trustnet-neo4j')

# Load environment variables
load_dotenv()

# Constants and configuration
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATTERN_RAG_DIR = os.environ.get("PATTERN_RAG_DIR", "")

# Source SQLite databases from Pattern RAG
SOURCE_METADATA_DB = os.path.join(PATTERN_RAG_DIR, "metadata/metadata.db")
SOURCE_GRAPH_FILE = os.path.join(PATTERN_RAG_DIR, "graph/knowledge_graph.pickle")

# Neo4j configuration - using Community Edition on a single machine
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

# Create directories if they don't exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

class Neo4jTrustNetwork:
    def __init__(self):
        """Initialize the Neo4j connection and basic structure"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600
            )
            logger.info(f"Connected to Neo4j at {NEO4J_URI}")
            self._test_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def _test_connection(self):
        """Test the Neo4j connection"""
        with self.driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message")
            for record in result:
                logger.info(record["message"])
                
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def initialize_schema(self):
        """Create initial schema with constraints and indices"""
        logger.info("Initializing Neo4j schema...")
        
        # Schema definition queries
        schema_queries = [
            # Node constraints
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            
            # Indices for performance
            "CREATE INDEX source_name_idx IF NOT EXISTS FOR (s:Source) ON (s.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX claim_text_idx IF NOT EXISTS FOR (c:Claim) ON (c.claim_text)",
            "CREATE INDEX document_path_idx IF NOT EXISTS FOR (d:Document) ON (d.path)",
            
            # Relationship indices
            "CREATE INDEX cites_idx IF NOT EXISTS FOR ()-[r:CITES]-() ON (r.sentiment)",
            "CREATE INDEX supports_idx IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.strength)",
            "CREATE INDEX contradicts_idx IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.strength)",
            "CREATE INDEX mentions_idx IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.frequency)"
        ]
        
        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.info(f"Schema operation successful: {query}")
                except Exception as e:
                    logger.error(f"Error executing schema query {query}: {str(e)}")
        
        logger.info("Schema initialization completed")

    def initialize_trust_tables(self):
        """Initialize the core trust network structure"""
        logger.info("Initializing trust network structure...")
        
        # Create basic trust network structure with Cypher
        trust_queries = [
            # Create initial trust network metadata
            """
            MERGE (tn:TrustNetwork {id: 'aegis-main'})
            SET tn.name = 'Aegis Trust Network',
                tn.created = datetime(),
                tn.description = 'Primary trust network for pattern analysis and alternative viewpoint evaluation'
            RETURN tn
            """
        ]
        
        with self.driver.session() as session:
            for query in trust_queries:
                try:
                    result = session.run(query)
                    logger.info(f"Trust network initialization successful")
                except Exception as e:
                    logger.error(f"Error initializing trust network: {str(e)}")
                    
        logger.info("Trust network structure initialization completed")

    def migrate_entities_from_sqlite(self):
        """Migrate entity data from SQLite to Neo4j"""
        if not os.path.exists(SOURCE_METADATA_DB):
            logger.error(f"Source metadata database not found: {SOURCE_METADATA_DB}")
            return False
            
        logger.info(f"Migrating entities from SQLite: {SOURCE_METADATA_DB}")
        
        # Connect to SQLite
        try:
            conn = sqlite3.connect(SOURCE_METADATA_DB)
            cursor = conn.cursor()
            
            # Get entity count
            cursor.execute("SELECT COUNT(*) FROM entities")
            entity_count = cursor.fetchone()[0]
            logger.info(f"Found {entity_count} entities to migrate")
            
            # Process in batches
            batch_size = 1000
            total_migrated = 0
            
            for offset in range(0, entity_count, batch_size):
                # Fetch batch of entities
                cursor.execute(
                    "SELECT id, name, type, frequency, last_seen FROM entities LIMIT ? OFFSET ?",
                    (batch_size, offset)
                )
                entities = cursor.fetchall()
                
                if not entities:
                    break
                    
                # Use Neo4j batch processing for efficiency
                with self.driver.session() as session:
                    # Unpack entity data and create batch query
                    for entity_id, name, entity_type, frequency, last_seen in entities:
                        # Skip empty names
                        if not name or name.strip() == "":
                            continue
                            
                        # Clean data
                        name = name.replace('"', '\\"').replace("'", "\\'")
                        entity_type = entity_type or "UNKNOWN"
                        
                        # Convert timestamp if present
                        neo4j_timestamp = "null"
                        if last_seen:
                            try:
                                dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                                neo4j_timestamp = f"datetime('{dt.isoformat()}')"
                            except:
                                neo4j_timestamp = "datetime()"
                        
                        # Create or update entity
                        query = f"""
                        MERGE (e:Entity {{name: "{name}"}})
                        ON CREATE SET 
                            e.id = {entity_id},
                            e.type = "{entity_type}",
                            e.frequency = {frequency},
                            e.last_seen = {neo4j_timestamp},
                            e.source = "pattern-rag"
                        ON MATCH SET 
                            e.frequency = {frequency},
                            e.type = "{entity_type}",
                            e.last_seen = {neo4j_timestamp}
                        RETURN e.name
                        """
                        
                        try:
                            session.run(query)
                        except Exception as e:
                            logger.error(f"Error creating entity '{name}': {str(e)}")
                
                total_migrated += len(entities)
                logger.info(f"Migrated {total_migrated}/{entity_count} entities")
            
            conn.close()
            logger.info(f"Entity migration completed: {total_migrated} entities migrated")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating entities: {str(e)}")
            return False

    def migrate_documents_from_sqlite(self):
        """Migrate document data from SQLite to Neo4j"""
        if not os.path.exists(SOURCE_METADATA_DB):
            logger.error(f"Source metadata database not found: {SOURCE_METADATA_DB}")
            return False
            
        logger.info(f"Migrating documents from SQLite: {SOURCE_METADATA_DB}")
        
        # Connect to SQLite
        try:
            conn = sqlite3.connect(SOURCE_METADATA_DB)
            cursor = conn.cursor()
            
            # Get document count
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            logger.info(f"Found {doc_count} documents to migrate")
            
            # Process in batches
            batch_size = 100
            total_migrated = 0
            
            for offset in range(0, doc_count, batch_size):
                # Fetch batch of documents
                cursor.execute(
                    """SELECT id, path, title, author, file_type, size, chunk_count, 
                       indexed_date, themes, categories FROM documents LIMIT ? OFFSET ?""",
                    (batch_size, offset)
                )
                documents = cursor.fetchall()
                
                if not documents:
                    break
                    
                # Process each document
                with self.driver.session() as session:
                    for doc in documents:
                        doc_id, path, title, author, file_type, size, chunk_count, indexed_date, themes, categories = doc
                        
                        # Clean data
                        title = (title or "Untitled").replace('"', '\\"').replace("'", "\\'")
                        author = (author or "Unknown").replace('"', '\\"').replace("'", "\\'")
                        path = path.replace('\\', '\\\\').replace('"', '\\"')
                        
                        # Convert themes to list if present
                        themes_list = "[]"
                        if themes:
                            theme_items = themes.split(',')
                            themes_list = json.dumps(theme_items)
                        
                        # Categories list
                        categories_list = "[]"
                        if categories:
                            category_items = categories.split(',')
                            categories_list = json.dumps(category_items)
                        
                        # Create document node
                        query = f"""
                        MERGE (d:Document {{id: {doc_id}}})
                        ON CREATE SET 
                            d.path = "{path}",
                            d.title = "{title}",
                            d.author = "{author}",
                            d.file_type = "{file_type or ''}",
                            d.size = {size or 0},
                            d.chunk_count = {chunk_count or 0},
                            d.indexed_date = datetime('{indexed_date or datetime.now().isoformat()}'),
                            d.themes = {themes_list},
                            d.categories = {categories_list},
                            d.source = "pattern-rag"
                        RETURN d.id
                        """
                        
                        try:
                            session.run(query)
                            
                            # Now link document to its entities mentioned in chunks
                            if themes:
                                # Get document chunks to find entities
                                cursor.execute(
                                    "SELECT entities FROM chunks WHERE doc_id = ? AND entities IS NOT NULL AND entities != ''",
                                    (doc_id,)
                                )
                                chunk_entities = cursor.fetchall()
                                
                                # Process entity mentions
                                entity_mentions = {}
                                for chunk_row in chunk_entities:
                                    if chunk_row[0]:
                                        entities = chunk_row[0].split(',')
                                        for entity in entities:
                                            entity = entity.strip()
                                            if entity:
                                                entity_mentions[entity] = entity_mentions.get(entity, 0) + 1
                                
                                # Create document-entity relationships
                                for entity, frequency in entity_mentions.items():
                                    entity_clean = entity.replace('"', '\\"').replace("'", "\\'")
                                    relation_query = f"""
                                    MATCH (d:Document {{id: {doc_id}}}), (e:Entity {{name: "{entity_clean}"}})
                                    MERGE (d)-[r:MENTIONS]->(e)
                                    ON CREATE SET r.frequency = {frequency}
                                    ON MATCH SET r.frequency = r.frequency + {frequency}
                                    """
                                    try:
                                        session.run(relation_query)
                                    except Exception as e:
                                        pass  # Skip if entity doesn't exist
                        except Exception as e:
                            logger.error(f"Error creating document '{title}': {str(e)}")
                
                total_migrated += len(documents)
                logger.info(f"Migrated {total_migrated}/{doc_count} documents")
            
            conn.close()
            logger.info(f"Document migration completed: {total_migrated} documents migrated")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating documents: {str(e)}")
            return False

    def migrate_relationships_from_sqlite(self):
        """Migrate relationship data from SQLite to Neo4j"""
        if not os.path.exists(SOURCE_METADATA_DB):
            logger.error(f"Source metadata database not found: {SOURCE_METADATA_DB}")
            return False
            
        logger.info(f"Migrating relationships from SQLite: {SOURCE_METADATA_DB}")
        
        # Connect to SQLite
        try:
            conn = sqlite3.connect(SOURCE_METADATA_DB)
            cursor = conn.cursor()
            
            # Get relationship count
            cursor.execute("SELECT COUNT(*) FROM relationships")
            rel_count = cursor.fetchone()[0]
            logger.info(f"Found {rel_count} relationships to migrate")
            
            # Process in batches
            batch_size = 1000
            total_migrated = 0
            
            for offset in range(0, rel_count, batch_size):
                # Fetch batch of relationships
                cursor.execute(
                    """SELECT r.id, r.source_id, r.target_id, r.type, r.frequency, r.chunks,
                       s.name AS source_name, t.name AS target_name
                       FROM relationships r
                       JOIN entities s ON r.source_id = s.id
                       JOIN entities t ON r.target_id = t.id
                       LIMIT ? OFFSET ?""",
                    (batch_size, offset)
                )
                relationships = cursor.fetchall()
                
                if not relationships:
                    break
                    
                # Process each relationship
                with self.driver.session() as session:
                    for rel in relationships:
                        rel_id, source_id, target_id, rel_type, frequency, chunks, source_name, target_name = rel
                        
                        # Skip if names are missing
                        if not source_name or not target_name:
                            continue
                        
                        # Clean data
                        source_name = source_name.replace('"', '\\"').replace("'", "\\'")
                        target_name = target_name.replace('"', '\\"').replace("'", "\\'")
                        rel_type = rel_type or "RELATED_TO"
                        
                        # Determine relationship type to use in Neo4j
                        neo4j_rel_type = "RELATED_TO"
                        if rel_type.lower() in ["say", "tell", "claim", "state", "mention", "write", "publish"]:
                            neo4j_rel_type = "CLAIMS"
                        elif rel_type.lower() in ["support", "agree", "confirm", "verify", "endorse"]:
                            neo4j_rel_type = "SUPPORTS"
                        elif rel_type.lower() in ["contradict", "oppose", "deny", "refute", "criticize"]:
                            neo4j_rel_type = "CONTRADICTS"
                        
                        # Create relationship - using direct string replacement instead of f-string for better escaping
                        query = """
                        MATCH (source:Entity {name: "SOURCE_NAME"}), (target:Entity {name: "TARGET_NAME"})
                        MERGE (source)-[r:REL_TYPE]->(target)
                        ON CREATE SET 
                            r.original_type = "ORIG_TYPE",
                            r.frequency = FREQ,
                            r.source = "pattern-rag"
                        ON MATCH SET 
                            r.frequency = r.frequency + FREQ
                        RETURN r
                        """
                        # Perform safe replacements
                        query = query.replace("SOURCE_NAME", source_name)
                        query = query.replace("TARGET_NAME", target_name)
                        query = query.replace("REL_TYPE", neo4j_rel_type)
                        query = query.replace("ORIG_TYPE", rel_type)
                        query = query.replace("FREQ", str(frequency or 1))
                        
                        try:
                            session.run(query)
                        except Exception as e:
                            logger.error(f"Error creating relationship {source_name}->{target_name}: {str(e)}")
                
                total_migrated += len(relationships)
                logger.info(f"Migrated {total_migrated}/{rel_count} relationships")
            
            conn.close()
            logger.info(f"Relationship migration completed: {total_migrated} relationships migrated")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating relationships: {str(e)}")
            return False

    def migrate_networkx_graph(self):
        """Migrate NetworkX graph data to Neo4j"""
        if not os.path.exists(SOURCE_GRAPH_FILE):
            logger.error(f"Source graph file not found: {SOURCE_GRAPH_FILE}")
            return False
            
        logger.info(f"Migrating NetworkX graph from: {SOURCE_GRAPH_FILE}")
        
        try:
            # Load NetworkX graph
            with open(SOURCE_GRAPH_FILE, 'rb') as f:
                graph = pickle.load(f)
                
            logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            # Process nodes in batches
            node_batch_size = 1000
            nodes = list(graph.nodes(data=True))
            total_nodes = len(nodes)
            nodes_migrated = 0
            
            with self.driver.session() as session:
                for i in range(0, total_nodes, node_batch_size):
                    batch = nodes[i:i+node_batch_size]
                    
                    for node_name, attrs in batch:
                        # Skip if node name is empty
                        if not node_name:
                            continue
                            
                        # Clean node name
                        clean_name = str(node_name).replace('"', '\\"').replace("'", "\\'")
                        
                        # Get node type
                        node_type = attrs.get('type', 'ENTITY')
                        weight = attrs.get('weight', 1)
                        
                        # Create node if it doesn't exist
                        query = f"""
                        MERGE (e:Entity {{name: "{clean_name}"}})
                        ON CREATE SET 
                            e.type = "{node_type}",
                            e.weight = {weight},
                            e.source = "networkx"
                        ON MATCH SET
                            e.weight = {weight}
                        RETURN e.name
                        """
                        
                        try:
                            session.run(query)
                        except Exception as e:
                            logger.error(f"Error creating node '{clean_name}': {str(e)}")
                    
                    nodes_migrated += len(batch)
                    logger.info(f"Migrated {nodes_migrated}/{total_nodes} nodes")
            
            # Process edges in batches
            edge_batch_size = 2000
            edges = list(graph.edges(data=True))
            total_edges = len(edges)
            edges_migrated = 0
            
            with self.driver.session() as session:
                for i in range(0, total_edges, edge_batch_size):
                    batch = edges[i:i+edge_batch_size]
                    
                    for source, target, attrs in batch:
                        # Skip if node names are empty
                        if not source or not target:
                            continue
                            
                        # Clean node names
                        source_clean = str(source).replace('"', '\\"').replace("'", "\\'")
                        target_clean = str(target).replace('"', '\\"').replace("'", "\\'")
                        
                        # Get edge properties
                        edge_type = attrs.get('type', 'RELATED_TO')
                        weight = attrs.get('weight', 1)
                        
                        # Determine relationship type to use in Neo4j
                        neo4j_rel_type = "RELATED_TO"
                        if edge_type and isinstance(edge_type, str):
                            if edge_type.lower() in ["say", "tell", "claim", "state", "mention", "write", "publish"]:
                                neo4j_rel_type = "CLAIMS"
                            elif edge_type.lower() in ["support", "agree", "confirm", "verify", "endorse"]:
                                neo4j_rel_type = "SUPPORTS"
                            elif edge_type.lower() in ["contradict", "oppose", "deny", "refute", "criticize"]:
                                neo4j_rel_type = "CONTRADICTS"
                        
                        # Create relationship - using direct string replacement instead of f-string
                        query = """
                        MATCH (source:Entity {name: "SOURCE_NAME"}), (target:Entity {name: "TARGET_NAME"})
                        MERGE (source)-[r:REL_TYPE]->(target)
                        ON CREATE SET 
                            r.original_type = "ORIG_TYPE",
                            r.weight = WEIGHT_VAL,
                            r.source = "networkx"
                        ON MATCH SET 
                            r.weight = WEIGHT_VAL
                        """
                        # Perform safe replacements
                        query = query.replace("SOURCE_NAME", source_clean)
                        query = query.replace("TARGET_NAME", target_clean)
                        query = query.replace("REL_TYPE", neo4j_rel_type)
                        query = query.replace("ORIG_TYPE", str(edge_type))
                        query = query.replace("WEIGHT_VAL", str(weight))
                        
                        try:
                            session.run(query)
                        except Exception as e:
                            pass  # Skip if entities don't exist
                    
                    edges_migrated += len(batch)
                    logger.info(f"Migrated {edges_migrated}/{total_edges} edges")
            
            logger.info(f"NetworkX graph migration completed")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating NetworkX graph: {str(e)}")
            return False
            
    def create_initial_claims(self):
        """Create initial claim nodes based on extracted patterns"""
        logger.info("Creating initial claims from existing entities and relationships")
        
        with self.driver.session() as session:
            # Find entities with high connectivity as potential claim subjects
            query = """
            MATCH (e:Entity)-[r]->()
            WITH e, count(r) as rel_count
            WHERE rel_count > 3
            RETURN e.name as name, e.type as type, rel_count
            ORDER BY rel_count DESC
            LIMIT 100
            """
            
            result = session.run(query)
            high_value_entities = [(record["name"], record["type"], record["rel_count"]) for record in result]
            
            logger.info(f"Found {len(high_value_entities)} high-value entities for claim extraction")
            
            # For each high-value entity, find relationships that might form claims
            claims_created = 0
            for entity_name, entity_type, _ in high_value_entities:
                # Clean entity name
                clean_name = entity_name.replace('"', '\\"').replace("'", "\\'")
                
                # Find significant relationships for this entity
                rel_query = f"""
                MATCH (e:Entity {{name: "{clean_name}"}})-[r]->(target:Entity)
                WHERE r.weight > 1 OR r.frequency > 1
                RETURN type(r) as rel_type, target.name as target_name, 
                       coalesce(r.weight, 1) as weight, r.original_type as original_type
                ORDER BY weight DESC
                LIMIT 5
                """
                
                rel_result = session.run(rel_query)
                
                for record in rel_result:
                    rel_type = record["rel_type"]
                    target_name = record["target_name"]
                    weight = record["weight"]
                    original_verb = record.get("original_type", rel_type.lower())
                    
                    # Construct a claim text
                    claim_text = f"{entity_name} {original_verb} {target_name}"
                    
                    # Create claim node - using direct string replacement instead of f-string
                    claim_text_clean = claim_text.replace('"', '\\"')
                    target_name_clean = target_name.replace('"', '\\"')
                    
                    claim_query = """
                    MERGE (c:Claim {claim_text: "CLAIM_TEXT"})
                    ON CREATE SET 
                        c.id = apoc.create.uuid(),
                        c.first_seen = datetime(),
                        c.frequency = 1,
                        c.source = "auto-generated"
                    ON MATCH SET 
                        c.frequency = c.frequency + 1
                    
                    WITH c
                    
                    MATCH (source:Entity {name: "SOURCE_NAME"}), (target:Entity {name: "TARGET_NAME"})
                    MERGE (source)-[r1:MENTIONED_IN]->(c)
                    MERGE (target)-[r2:MENTIONED_IN]->(c)
                    MERGE (source)-[r3:REL_TYPE]->(c)
                    
                    RETURN c.claim_text
                    """
                    
                    # Perform safe replacements
                    claim_query = claim_query.replace("CLAIM_TEXT", claim_text_clean)
                    claim_query = claim_query.replace("SOURCE_NAME", clean_name)
                    claim_query = claim_query.replace("TARGET_NAME", target_name_clean)
                    claim_query = claim_query.replace("REL_TYPE", rel_type)
                    
                    try:
                        claim_result = session.run(claim_query)
                        claims_created += 1
                    except Exception as e:
                        logger.error(f"Error creating claim '{claim_text}': {str(e)}")
            
            logger.info(f"Created {claims_created} initial claims")

    def create_initial_trust_network(self):
        """Create initial trust seeds based on document sources and entities"""
        logger.info("Creating initial trust network seeds")
        
        with self.driver.session() as session:
            # Create trusted sources node group
            sources_query = """
            MERGE (ts:TrustSeeds {id: 'trusted-sources'})
            SET ts.name = 'Trusted Source Seeds',
                ts.created = datetime(),
                ts.description = 'Initial trusted sources for bootstrapping trust network'
            RETURN ts
            """
            
            session.run(sources_query)
            
            # Find top authors by document count
            authors_query = """
            MATCH (d:Document)
            WHERE d.author IS NOT NULL AND d.author <> 'Unknown' AND d.author <> ''
            WITH d.author as author, count(d) as doc_count
            WHERE doc_count > 1
            RETURN author, doc_count
            ORDER BY doc_count DESC
            LIMIT 20
            """
            
            result = session.run(authors_query)
            top_authors = [(record["author"], record["doc_count"]) for record in result]
            
            # Create trusted Source nodes for top authors
            for author, doc_count in top_authors:
                author_clean = author.replace('"', '\\"').replace("'", "\\'")
                
                source_query = f"""
                MERGE (s:Source {{name: "{author_clean}"}})
                ON CREATE SET 
                    s.id = apoc.create.uuid(),
                    s.description = "Author of {doc_count} documents",
                    s.source_type = "AUTHOR",
                    s.first_indexed = datetime(),
                    s.last_indexed = datetime(),
                    s.initial_trust = 0.5
                
                WITH s
                
                MATCH (ts:TrustSeeds {{id: 'trusted-sources'}})
                MERGE (ts)-[r:CONTAINS]->(s)
                
                RETURN s.name
                """
                
                try:
                    session.run(source_query)
                except Exception as e:
                    logger.error(f"Error creating source for '{author}': {str(e)}")
            
            logger.info(f"Created {len(top_authors)} initial trusted sources")
            
            # Create network measures indexes to improve algorithm performance
            indices_query = """
            CALL db.index.fulltext.createNodeIndex(
                "entity_names",
                ["Entity", "Source", "Claim"],
                ["name", "claim_text"]
            )
            """
            
            try:
                session.run(indices_query)
                logger.info("Created fulltext search index for entities, sources and claims")
            except Exception as e:
                logger.error(f"Error creating fulltext index: {str(e)}")
                # May fail if fulltext index already exists or APOC not available

def main():
    """Main execution function"""
    start_time = time.time()
    logger.info("Starting AegisTrustNet Neo4j initialization and migration")
    
    # Initialize Neo4j connection
    try:
        trust_network = Neo4jTrustNetwork()
        
        # Initialize schema
        trust_network.initialize_schema()
        
        # Initialize trust network structure
        trust_network.initialize_trust_tables()
        
        # Migrate data from Pattern RAG
        trust_network.migrate_entities_from_sqlite()
        trust_network.migrate_documents_from_sqlite()
        trust_network.migrate_relationships_from_sqlite()
        trust_network.migrate_networkx_graph()
        
        # Create initial trust data structures
        trust_network.create_initial_claims()
        trust_network.create_initial_trust_network()
        
        # Close connection
        trust_network.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Migration completed in {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
