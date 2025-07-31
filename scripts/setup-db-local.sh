#!/bin/bash
# Setup PostgreSQL with pgvector locally

set -e

echo "🐘 Setting up PostgreSQL with pgvector locally..."

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/debian_version ]; then
        OS="debian"
    elif [ -f /etc/redhat-release ]; then
        OS="redhat"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "🔍 Detected OS: $OS"

# Install PostgreSQL and pgvector based on OS
case $OS in
    macos)
        echo "📦 Installing PostgreSQL and pgvector using Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "❌ Homebrew is not installed. Please install it first."
            echo "   Visit: https://brew.sh"
            exit 1
        fi
        
        # Install PostgreSQL if not installed
        if ! brew list postgresql@16 &> /dev/null; then
            brew install postgresql@16
        fi
        
        # Install pgvector
        if ! brew list pgvector &> /dev/null; then
            brew install pgvector
        fi
        
        # Start PostgreSQL
        brew services start postgresql@16
        
        # Set PostgreSQL path
        export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
        ;;
        
    debian)
        echo "📦 Installing PostgreSQL and pgvector using apt..."
        
        # Add PostgreSQL APT repository
        sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
        wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
        sudo apt-get update
        
        # Install PostgreSQL and pgvector
        sudo apt-get install -y postgresql-16 postgresql-client-16 postgresql-16-pgvector
        
        # Start PostgreSQL
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
        ;;
        
    *)
        echo "❌ Unsupported OS: $OS"
        echo "   Please install PostgreSQL 16 and pgvector manually."
        echo "   Visit: https://www.postgresql.org/download/"
        exit 1
        ;;
esac

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if pg_isready > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Create database and user
echo "🔨 Creating database and user..."
if [ "$OS" == "macos" ]; then
    # macOS: current user has superuser access
    createdb freeagentics 2>/dev/null || echo "Database 'freeagentics' already exists"
    psql -d freeagentics -c "CREATE EXTENSION IF NOT EXISTS vector;"
else
    # Linux: use postgres user
    sudo -u postgres createdb freeagentics 2>/dev/null || echo "Database 'freeagentics' already exists"
    sudo -u postgres psql -d freeagentics -c "CREATE EXTENSION IF NOT EXISTS vector;"
    
    # Create user with same name as current user
    sudo -u postgres createuser -s $(whoami) 2>/dev/null || echo "User '$(whoami)' already exists"
fi

# Verify pgvector is installed
echo "🔍 Verifying pgvector installation..."
if psql -d freeagentics -c "SELECT vector_version();" > /dev/null 2>&1; then
    echo "✅ pgvector is installed and working!"
else
    echo "❌ pgvector verification failed. Please check the installation."
    exit 1
fi

# Display connection info
echo ""
echo "📋 Database Connection Info:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: freeagentics"
echo "   User: $(whoami)"
echo ""
echo "   DATABASE_URL=postgresql://$(whoami)@localhost:5432/freeagentics"
echo ""
echo "🎉 PostgreSQL with pgvector is ready!"
echo ""
echo "Next steps:"
echo "1. Copy the DATABASE_URL to your .env file"
echo "2. Run migrations: make db-migrate"
echo "3. Start developing!"