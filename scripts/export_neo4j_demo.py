#!/usr/bin/env python3
"""
Neo4j Demo Data Exporter
========================

Exports Neo4j graph data to portable Cypher statements that can be
imported on any Neo4j version.

This script does NOT require APOC - it uses standard Cypher queries.

Usage:
    python3 export_neo4j_demo.py --output demo-data/neo4j_export.cypher
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Generator, Dict, Any, List

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j driver not installed. Run: pip install neo4j")
    sys.exit(1)

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "aegistrusted"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def escape_cypher_string(value: str) -> str:
    """Escape a string for use in Cypher"""
    if value is None:
        return "null"
    # Escape backslashes, single quotes, and newlines
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return f"'{escaped}'"


def format_property_value(value: Any) -> str:
    """Format a property value for Cypher"""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return escape_cypher_string(value)
    elif isinstance(value, list):
        items = [format_property_value(v) for v in value]
        return f"[{', '.join(items)}]"
    elif isinstance(value, dict):
        # Convert dict to JSON string
        return escape_cypher_string(json.dumps(value))
    else:
        return escape_cypher_string(str(value))


def format_properties(props: Dict[str, Any]) -> str:
    """Format a properties dict as Cypher map"""
    if not props:
        return "{}"
    parts = []
    for key, value in props.items():
        # Skip internal Neo4j properties
        if key.startswith("_"):
            continue
        parts.append(f"{key}: {format_property_value(value)}")
    return "{" + ", ".join(parts) + "}"


class Neo4jExporter:
    """Export Neo4j graph to Cypher statements"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_count = 0
        self.rel_count = 0
    
    def close(self):
        self.driver.close()
    
    def get_all_labels(self) -> List[str]:
        """Get all node labels in the database"""
        with self.driver.session() as session:
            result = session.run("CALL db.labels() YIELD label RETURN label")
            return [record["label"] for record in result]
    
    def get_all_relationship_types(self) -> List[str]:
        """Get all relationship types in the database"""
        with self.driver.session() as session:
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            return [record["relationshipType"] for record in result]
    
    def get_node_count(self, label: str) -> int:
        """Get count of nodes with given label"""
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            return result.single()["count"]
    
    def export_nodes(self, label: str, batch_size: int = 1000) -> Generator[str, None, None]:
        """Export all nodes with a given label as CREATE statements"""
        with self.driver.session() as session:
            # Get total count
            total = self.get_node_count(label)
            logger.info(f"Exporting {total:,} {label} nodes...")
            
            # Export in batches
            skip = 0
            while skip < total:
                result = session.run(
                    f"MATCH (n:{label}) RETURN elementId(n) as id, properties(n) as props "
                    f"SKIP {skip} LIMIT {batch_size}"
                )
                
                for record in result:
                    node_id = record["id"]
                    props = record["props"] or {}
                    
                    # Create node with temporary ID property for relationship matching
                    props["_import_id"] = node_id
                    
                    cypher = f"CREATE (:{label} {format_properties(props)});"
                    yield cypher
                    self.node_count += 1
                
                skip += batch_size
                if skip % 10000 == 0:
                    logger.info(f"  Exported {skip:,}/{total:,} {label} nodes")
    
    def export_relationships(self, rel_type: str, batch_size: int = 1000) -> Generator[str, None, None]:
        """Export all relationships of a given type"""
        with self.driver.session() as session:
            # Get count
            result = session.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            )
            total = result.single()["count"]
            logger.info(f"Exporting {total:,} {rel_type} relationships...")
            
            # Export in batches
            skip = 0
            while skip < total:
                result = session.run(
                    f"MATCH (a)-[r:{rel_type}]->(b) "
                    f"RETURN elementId(a) as start_id, elementId(b) as end_id, "
                    f"labels(a)[0] as start_label, labels(b)[0] as end_label, "
                    f"properties(r) as props "
                    f"SKIP {skip} LIMIT {batch_size}"
                )
                
                for record in result:
                    start_label = record["start_label"]
                    end_label = record["end_label"]
                    start_id = record["start_id"]
                    end_id = record["end_id"]
                    props = record["props"] or {}
                    
                    props_str = format_properties(props) if props else ""
                    props_clause = f" {props_str}" if props_str and props_str != "{}" else ""
                    
                    cypher = (
                        f"MATCH (a:{start_label} {{_import_id: '{start_id}'}}), "
                        f"(b:{end_label} {{_import_id: '{end_id}'}}) "
                        f"CREATE (a)-[:{rel_type}{props_clause}]->(b);"
                    )
                    yield cypher
                    self.rel_count += 1
                
                skip += batch_size
    
    def export_indexes(self) -> Generator[str, None, None]:
        """Export index creation statements"""
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties, type")
            for record in result:
                name = record["name"]
                labels = record["labelsOrTypes"]
                properties = record["properties"]
                idx_type = record["type"]
                
                if labels and properties:
                    label = labels[0]
                    props = ", ".join([f"n.{p}" for p in properties])
                    
                    if idx_type == "FULLTEXT":
                        yield f"// Fulltext index: {name} (create manually)"
                    elif idx_type == "RANGE":
                        yield f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON ({props});"
    
    def export_constraints(self) -> Generator[str, None, None]:
        """Export constraint creation statements"""
        with self.driver.session() as session:
            result = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties, type")
            for record in result:
                name = record["name"]
                labels = record["labelsOrTypes"]
                properties = record["properties"]
                con_type = record["type"]
                
                if labels and properties:
                    label = labels[0]
                    props = ", ".join([f"n.{p}" for p in properties])
                    
                    if con_type == "UNIQUENESS":
                        yield f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE ({props}) IS UNIQUE;"
    
    def export_all(self, output_file: str):
        """Export entire database to Cypher file"""
        labels = self.get_all_labels()
        rel_types = self.get_all_relationship_types()
        
        logger.info(f"Found {len(labels)} labels: {labels}")
        logger.info(f"Found {len(rel_types)} relationship types: {rel_types}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            # Header
            f.write(f"// Aegis Insight Demo Data Export\n")
            f.write(f"// Generated: {datetime.now().isoformat()}\n")
            f.write(f"// Labels: {', '.join(labels)}\n")
            f.write(f"// Relationship Types: {', '.join(rel_types)}\n")
            f.write("\n")
            
            # Clear existing data (optional)
            f.write("// Clear existing data (uncomment if needed)\n")
            f.write("// MATCH (n) DETACH DELETE n;\n\n")
            
            # Indexes and constraints first
            f.write("// === INDEXES ===\n")
            for stmt in self.export_indexes():
                f.write(stmt + "\n")
            f.write("\n")
            
            f.write("// === CONSTRAINTS ===\n")
            for stmt in self.export_constraints():
                f.write(stmt + "\n")
            f.write("\n")
            
            # Create index on import ID for relationship matching
            f.write("// Temporary import index\n")
            for label in labels:
                f.write(f"CREATE INDEX import_idx_{label.lower()} IF NOT EXISTS FOR (n:{label}) ON (n._import_id);\n")
            f.write("\n")
            
            # Nodes
            f.write("// === NODES ===\n")
            for label in labels:
                f.write(f"\n// --- {label} nodes ---\n")
                for stmt in self.export_nodes(label):
                    f.write(stmt + "\n")
            f.write("\n")
            
            # Relationships
            f.write("// === RELATIONSHIPS ===\n")
            for rel_type in rel_types:
                f.write(f"\n// --- {rel_type} relationships ---\n")
                for stmt in self.export_relationships(rel_type):
                    f.write(stmt + "\n")
            f.write("\n")
            
            # Cleanup import IDs
            f.write("// === CLEANUP ===\n")
            f.write("// Remove temporary import IDs\n")
            for label in labels:
                f.write(f"MATCH (n:{label}) REMOVE n._import_id;\n")
            f.write("\n")
            
            # Drop temporary indexes
            for label in labels:
                f.write(f"DROP INDEX import_idx_{label.lower()} IF EXISTS;\n")
        
        logger.info(f"Export complete!")
        logger.info(f"  Nodes: {self.node_count:,}")
        logger.info(f"  Relationships: {self.rel_count:,}")
        logger.info(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Export Neo4j graph to Cypher")
    parser.add_argument("--uri", default=NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--user", default=NEO4J_USER, help="Neo4j user")
    parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j password")
    parser.add_argument("--output", default="demo-data/neo4j_export.cypher", help="Output file")
    
    args = parser.parse_args()
    
    exporter = Neo4jExporter(args.uri, args.user, args.password)
    try:
        exporter.export_all(args.output)
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
