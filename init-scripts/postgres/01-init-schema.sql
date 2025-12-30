-- ============================================================================
-- Aegis Insight - PostgreSQL Initialization Script
-- ============================================================================
-- Creates the schema for vector embeddings storage
-- This script runs automatically on first container start
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE IF NOT EXISTS claim_embeddings (
    id SERIAL PRIMARY KEY,
    claim_id VARCHAR(255) UNIQUE NOT NULL,
    claim_text TEXT,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create IVFFlat index for approximate nearest neighbor search
-- Uses 100 lists for good balance of speed and accuracy
CREATE INDEX IF NOT EXISTS claim_embeddings_embedding_idx 
ON claim_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index on claim_id for fast lookups
CREATE INDEX IF NOT EXISTS claim_embeddings_claim_id_idx 
ON claim_embeddings (claim_id);

-- Create index on created_at for temporal queries
CREATE INDEX IF NOT EXISTS claim_embeddings_created_at_idx 
ON claim_embeddings (created_at);

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE claim_embeddings TO aegis;
GRANT USAGE, SELECT ON SEQUENCE claim_embeddings_id_seq TO aegis;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Aegis Insight PostgreSQL schema initialized successfully';
END $$;
