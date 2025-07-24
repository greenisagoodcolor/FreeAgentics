-- Enable pgvector and h3-pg extensions for FreeAgentics
-- This file is executed during PostgreSQL initialization

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable h3-pg extension for hierarchical hexagonal indexing
-- Note: h3-pg may need to be installed separately in the Docker image
-- CREATE EXTENSION IF NOT EXISTS h3 CASCADE;

-- Create schema for vector operations if needed
CREATE SCHEMA IF NOT EXISTS vectors;

-- Grant permissions to the freeagentics user for vector operations
GRANT ALL PRIVILEGES ON SCHEMA vectors TO freeagentics;

-- Log the extension setup
DO $$
BEGIN
    RAISE NOTICE 'FreeAgentics extensions enabled: pgvector ready for AI embeddings';
    RAISE NOTICE 'Note: h3-pg extension may need additional setup depending on Docker image';
END$$;
