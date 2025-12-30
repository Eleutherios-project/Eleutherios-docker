#!/usr/bin/env python3
"""
Aegis Insight - First Run Data Seeder
======================================

This script handles importing demo data on first run.
It's called automatically by docker-entrypoint.sh if no data exists.

Features:
- Checks if data already exists (skip if so)
- Imports Neo4j Cypher file in batches
- Imports PostgreSQL SQL file
- Creates required indexes
- Reports progress

Usage:
    python3 seed_demo_data.py

Environment variables:
    NEO4J_URI          - Neo4j connection (default: bolt://neo4j:7687)
    NEO4J_USER         - Neo4j user (default: neo4j)
    NEO4J_PASSWORD     - Neo4j password (default: aegistrusted)
    POSTGRES_HOST      - PostgreSQL host (default: postgres)
    POSTGRES_DB        - PostgreSQL database (default: aegis_insight)
    POSTGRES_USER      - PostgreSQL user (default: aegis)
    POSTGRES_PASSWORD  - PostgreSQL password (default: aegis_trusted_2025)
    DEMO_DATA_DIR      - Demo data directory (default: /app/demo-data)
"""

import gzip
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "aegistrusted")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "aegis_insight")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "aegis")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "aegis_trusted_2025")

DEMO_DATA_DIR = os.environ.get("DEMO_DATA_DIR", "/app/demo-data")


def wait_for_neo4j(max_attempts: int = 30, delay: int = 2) -> bool:
    """Wait for Neo4j to be ready"""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j driver not installed, skipping Neo4j seeding")
        return False
    
    logger.info(f"Waiting for Neo4j at {NEO4J_URI}...")
    
    for attempt in range(max_attempts):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            logger.info("Neo4j is ready!")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.debug(f"Attempt {attempt + 1}/{max_attempts}: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Neo4j not ready after {max_attempts} attempts")
                return False
    return False


def wait_for_postgres(max_attempts: int = 30, delay: int = 2) -> bool:
    """Wait for PostgreSQL to be ready"""
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed, skipping PostgreSQL seeding")
        return False
    
    logger.info(f"Waiting for PostgreSQL at {POSTGRES_HOST}...")
    
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                connect_timeout=5
            )
            conn.close()
            logger.info("PostgreSQL is ready!")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.debug(f"Attempt {attempt + 1}/{max_attempts}: {e}")
                time.sleep(delay)
            else:
                logger.error(f"PostgreSQL not ready after {max_attempts} attempts")
                return False
    return False


def check_neo4j_has_data() -> bool:
    """Check if Neo4j already has data"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
        driver.close()
        return count > 0
    except Exception as e:
        logger.error(f"Error checking Neo4j data: {e}")
        return False


def check_postgres_has_data() -> bool:
    """Check if PostgreSQL already has data"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM claim_embeddings")
            count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        # Table might not exist yet
        logger.debug(f"Error checking PostgreSQL data: {e}")
        return False


def seed_neo4j(cypher_file: Path) -> bool:
    """Import Cypher statements into Neo4j"""
    from neo4j import GraphDatabase
    
    logger.info(f"Importing Neo4j data from {cypher_file}...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Read file (handle gzip)
    if cypher_file.suffix == '.gz':
        with gzip.open(cypher_file, 'rt', encoding='utf-8') as f:
            content = f.read()
    else:
        with open(cypher_file, 'r', encoding='utf-8') as f:
            content = f.read()
    
    # Split into statements
    statements = []
    for line in content.split('\n'):
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith('//'):
            continue
        statements.append(line)
    
    logger.info(f"Found {len(statements)} Cypher statements")
    
    # Execute in batches
    batch_size = 100
    success_count = 0
    error_count = 0
    
    with driver.session() as session:
        for i, stmt in enumerate(statements):
            try:
                session.run(stmt)
                success_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Only log first 10 errors
                    logger.warning(f"Statement error: {str(e)[:100]}")
            
            # Progress
            if (i + 1) % 1000 == 0:
                logger.info(f"Progress: {i + 1}/{len(statements)} statements")
    
    driver.close()
    
    logger.info(f"Neo4j import complete: {success_count} success, {error_count} errors")
    return error_count < len(statements) * 0.1  # Allow up to 10% errors


def seed_postgres(sql_file: Path) -> bool:
    """Import SQL file into PostgreSQL"""
    logger.info(f"Importing PostgreSQL data from {sql_file}...")
    
    # Use psql for reliable import
    cmd = ["psql", f"-h{POSTGRES_HOST}", f"-U{POSTGRES_USER}", f"-d{POSTGRES_DB}"]
    
    env = os.environ.copy()
    env["PGPASSWORD"] = POSTGRES_PASSWORD
    
    # Handle gzip
    if sql_file.suffix == '.gz':
        # Decompress and pipe
        with gzip.open(sql_file, 'rt', encoding='utf-8') as f:
            content = f.read()
        
        result = subprocess.run(
            cmd,
            input=content,
            capture_output=True,
            text=True,
            env=env
        )
    else:
        with open(sql_file, 'r') as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
                env=env
            )
    
    if result.returncode != 0:
        logger.error(f"PostgreSQL import failed: {result.stderr[:500]}")
        return False
    
    logger.info("PostgreSQL import complete")
    return True


def main():
    logger.info("=" * 50)
    logger.info("Aegis Insight - First Run Data Seeder")
    logger.info("=" * 50)
    
    demo_dir = Path(DEMO_DATA_DIR)
    
    # Check demo data exists
    if not demo_dir.exists():
        logger.warning(f"Demo data directory not found: {demo_dir}")
        logger.info("Starting with empty database")
        return 0
    
    # Find data files
    neo4j_file = None
    postgres_file = None
    
    for pattern in ["neo4j_export.cypher", "neo4j_export.cypher.gz", "*.cypher", "*.cypher.gz"]:
        files = list(demo_dir.glob(pattern))
        if files:
            neo4j_file = files[0]
            break
    
    for pattern in ["aegis_insight.sql.gz", "aegis_insight.sql", "*.sql.gz", "*.sql"]:
        files = list(demo_dir.glob(pattern))
        if files:
            postgres_file = files[0]
            break
    
    logger.info(f"Neo4j file: {neo4j_file or 'not found'}")
    logger.info(f"PostgreSQL file: {postgres_file or 'not found'}")
    
    # Wait for services
    neo4j_ready = wait_for_neo4j()
    postgres_ready = wait_for_postgres()
    
    # Check existing data
    if neo4j_ready and check_neo4j_has_data():
        logger.info("Neo4j already has data, skipping import")
        neo4j_file = None
    
    if postgres_ready and check_postgres_has_data():
        logger.info("PostgreSQL already has data, skipping import")
        postgres_file = None
    
    # Import data
    success = True
    
    if neo4j_file and neo4j_ready:
        if not seed_neo4j(neo4j_file):
            logger.error("Neo4j seeding failed")
            success = False
    
    if postgres_file and postgres_ready:
        if not seed_postgres(postgres_file):
            logger.error("PostgreSQL seeding failed")
            success = False
    
    # Summary
    logger.info("=" * 50)
    if success:
        logger.info("Data seeding complete!")
    else:
        logger.warning("Data seeding completed with errors")
    logger.info("=" * 50)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
