"""
Aegis Insight - Centralized Configuration
==========================================
All configuration values with environment variable support and secure defaults.

Usage:
    from aegis_config import Config
    
    # Access configuration
    driver = GraphDatabase.driver(
        Config.NEO4J_URI, 
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    )

Environment Variables:
    NEO4J_URI           - Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER          - Neo4j username (default: neo4j)
    NEO4J_PASSWORD      - Neo4j password (default: aegistrusted)
    POSTGRES_HOST       - PostgreSQL host (default: localhost)
    POSTGRES_PORT       - PostgreSQL port (default: 5432)
    POSTGRES_USER       - PostgreSQL username (default: aegis)
    POSTGRES_PASSWORD   - PostgreSQL password (default: aegis_trusted_2025)
    POSTGRES_DB         - PostgreSQL database (default: aegis_insight)
    OLLAMA_HOST         - Ollama API URL (default: http://localhost:11434)
    OLLAMA_MODEL        - Default extraction model (default: mistral-nemo:12b)
    AEGIS_LOG_LEVEL     - Logging level (default: INFO)
"""

import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=os.getenv('AEGIS_LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class Config:
    """
    Centralized configuration with environment variable support.
    All values can be overridden via environment variables.
    """
    
    # =========================================================================
    # Neo4j Configuration
    # =========================================================================
    NEO4J_URI: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_HOST: str = os.getenv('NEO4J_HOST', 'localhost')
    NEO4J_USER: str = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD: str = os.getenv('NEO4J_PASSWORD', 'aegistrusted')
    
    # =========================================================================
    # PostgreSQL Configuration
    # =========================================================================
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT: int = int(os.getenv('POSTGRES_PORT', '5432'))
    POSTGRES_USER: str = os.getenv('POSTGRES_USER', 'aegis')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD', 'aegis_trusted_2025')
    POSTGRES_DB: str = os.getenv('POSTGRES_DB', 'aegis_insight')
    
    @classmethod
    def get_postgres_uri(cls) -> str:
        """Get PostgreSQL connection URI."""
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    # =========================================================================
    # Ollama Configuration
    # =========================================================================
    OLLAMA_HOST: str = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'mistral-nemo:12b')
    OLLAMA_EMBEDDING_MODEL: str = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
    
    # =========================================================================
    # Application Configuration
    # =========================================================================
    LOG_LEVEL: str = os.getenv('AEGIS_LOG_LEVEL', 'INFO')
    WORKERS: int = int(os.getenv('AEGIS_WORKERS', '4'))
    
    # Data directories
    DATA_DIR: str = os.getenv('AEGIS_DATA_DIR', './data')
    INBOX_DIR: str = os.getenv('AEGIS_INBOX_DIR', './data/inbox')
    CALIBRATION_DIR: str = os.getenv('AEGIS_CALIBRATION_DIR', './data/calibration_profiles')
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @classmethod
    def get_neo4j_auth(cls) -> tuple:
        """Get Neo4j authentication tuple."""
        return (cls.NEO4J_USER, cls.NEO4J_PASSWORD)
    
    @classmethod
    def get_postgres_config(cls) -> dict:
        """Get PostgreSQL connection configuration dict."""
        return {
            'host': cls.POSTGRES_HOST,
            'port': cls.POSTGRES_PORT,
            'user': cls.POSTGRES_USER,
            'password': cls.POSTGRES_PASSWORD,
            'database': cls.POSTGRES_DB
        }
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration and log warnings for default passwords.
        Returns True if configuration is valid.
        """
        warnings = []
        
        # Check for default passwords in production
        if os.getenv('AEGIS_ENVIRONMENT') == 'production':
            if cls.NEO4J_PASSWORD == 'aegistrusted':
                warnings.append("NEO4J_PASSWORD is using default value in production!")
            if cls.POSTGRES_PASSWORD == 'aegis_trusted_2025':
                warnings.append("POSTGRES_PASSWORD is using default value in production!")
        
        for warning in warnings:
            logger.warning(f"SECURITY WARNING: {warning}")
        
        return len(warnings) == 0
    
    @classmethod
    def log_config(cls, include_passwords: bool = False) -> None:
        """Log current configuration (optionally masking passwords)."""
        logger.info("=== Aegis Insight Configuration ===")
        logger.info(f"NEO4J_URI: {cls.NEO4J_URI}")
        logger.info(f"NEO4J_USER: {cls.NEO4J_USER}")
        logger.info(f"NEO4J_PASSWORD: {'***' if not include_passwords else cls.NEO4J_PASSWORD}")
        logger.info(f"POSTGRES_HOST: {cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}")
        logger.info(f"POSTGRES_USER: {cls.POSTGRES_USER}")
        logger.info(f"POSTGRES_PASSWORD: {'***' if not include_passwords else cls.POSTGRES_PASSWORD}")
        logger.info(f"POSTGRES_DB: {cls.POSTGRES_DB}")
        logger.info(f"OLLAMA_HOST: {cls.OLLAMA_HOST}")
        logger.info(f"OLLAMA_MODEL: {cls.OLLAMA_MODEL}")
        logger.info("===================================")


# Legacy compatibility - direct imports
# These allow: from aegis_config import NEO4J_PASSWORD
NEO4J_URI = Config.NEO4J_URI
NEO4J_HOST = Config.NEO4J_HOST
NEO4J_USER = Config.NEO4J_USER
NEO4J_PASSWORD = Config.NEO4J_PASSWORD

POSTGRES_HOST = Config.POSTGRES_HOST
POSTGRES_PORT = Config.POSTGRES_PORT
POSTGRES_USER = Config.POSTGRES_USER
POSTGRES_PASSWORD = Config.POSTGRES_PASSWORD
POSTGRES_DB = Config.POSTGRES_DB

OLLAMA_HOST = Config.OLLAMA_HOST
OLLAMA_MODEL = Config.OLLAMA_MODEL


if __name__ == '__main__':
    # Test configuration
    Config.log_config()
    Config.validate()
