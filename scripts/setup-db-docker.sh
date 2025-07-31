#!/bin/bash
# Setup PostgreSQL with pgvector using Docker

set -e

echo "🐘 Setting up PostgreSQL with pgvector using Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create postgres init directory if it doesn't exist
mkdir -p postgres/init

# Create initialization SQL if it doesn't exist
if [ ! -f postgres/init/01-init.sql ]; then
    cat > postgres/init/01-init.sql << 'EOF'
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create initial schema
CREATE SCHEMA IF NOT EXISTS freeagentics;

-- Grant permissions
GRANT ALL ON SCHEMA freeagentics TO freeagentics;
EOF
    echo "✅ Created postgres/init/01-init.sql"
fi

# Start PostgreSQL with pgvector
echo "🚀 Starting PostgreSQL with pgvector..."
docker-compose -f docker-compose.db.yml up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker-compose -f docker-compose.db.yml exec -T postgres pg_isready -U freeagentics > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Verify pgvector is installed
echo "🔍 Verifying pgvector installation..."
if docker-compose -f docker-compose.db.yml exec -T postgres psql -U freeagentics -d freeagentics -c "SELECT vector_version();" > /dev/null 2>&1; then
    echo "✅ pgvector is installed and working!"
else
    echo "⚠️  pgvector verification failed. Attempting to create extension..."
    docker-compose -f docker-compose.db.yml exec -T postgres psql -U freeagentics -d freeagentics -c "CREATE EXTENSION IF NOT EXISTS vector;"
fi

# Display connection info
echo ""
echo "📋 Database Connection Info:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: freeagentics"
echo "   User: freeagentics"
echo "   Password: freeagentics_dev"
echo ""
echo "   DATABASE_URL=postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics"
echo ""
echo "🎉 PostgreSQL with pgvector is ready!"
echo ""
echo "Next steps:"
echo "1. Copy the DATABASE_URL to your .env file"
echo "2. Run migrations: make db-migrate"
echo "3. Start developing!"