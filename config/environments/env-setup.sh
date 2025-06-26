#!/bin/bash

# CogniticNet Environment Setup Script
# This script helps set up different environments for CogniticNet

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "CogniticNet Environment Setup"
    echo ""
    echo "Usage: $0 <environment> [options]"
    echo ""
    echo "Environments:"
    echo "  local       - Local development environment (single developer)"
    echo "  development - Shared development environment"
    echo "  testing     - Testing environment for CI/CD"
    echo "  staging     - Staging/pre-production environment"
    echo "  production  - Production environment"
    echo ""
    echo "Options:"
    echo "  --copy-from <env>  Copy configuration from another environment"
    echo "  --force           Overwrite existing .env file"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local"
    echo "  $0 staging --copy-from development"
    echo "  $0 production --force"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Parse arguments
ENVIRONMENT=$1
COPY_FROM=""
FORCE=false

shift # Remove first argument

while [[ $# -gt 0 ]]; do
    case $1 in
        --copy-from)
            COPY_FROM="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
VALID_ENVIRONMENTS=("local" "development" "testing" "staging" "production")
if [[ ! " ${VALID_ENVIRONMENTS[@]} " =~ " ${ENVIRONMENT} " ]]; then
    print_error "Invalid environment: ${ENVIRONMENT}"
    print_info "Valid environments: ${VALID_ENVIRONMENTS[*]}"
    exit 1
fi

# Check if .env already exists
if [ -f ".env" ] && [ "$FORCE" = false ]; then
    print_warning ".env file already exists!"
    echo -n "Do you want to overwrite it? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Aborted."
        exit 0
    fi
fi

# Create environments directory if it doesn't exist
if [ ! -d "environments" ]; then
    print_error "environments directory not found!"
    print_info "Please run this script from the project root directory."
    exit 1
fi

# Determine source file
if [ -n "$COPY_FROM" ]; then
    SOURCE_FILE="environments/env.${COPY_FROM}"
    if [ ! -f "$SOURCE_FILE" ]; then
        print_error "Source environment file not found: $SOURCE_FILE"
        exit 1
    fi
else
    SOURCE_FILE="environments/env.${ENVIRONMENT}"
    if [ ! -f "$SOURCE_FILE" ]; then
        # If specific environment file doesn't exist, use example
        print_warning "Environment file not found: $SOURCE_FILE"
        print_info "Using example template instead"
        SOURCE_FILE="environments/env.example"
    fi
fi

# Copy the environment file
print_info "Setting up ${ENVIRONMENT} environment..."
cp "$SOURCE_FILE" .env

# Update ENVIRONMENT variable in .env
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^ENVIRONMENT=.*/ENVIRONMENT=${ENVIRONMENT}/" .env
else
    # Linux
    sed -i "s/^ENVIRONMENT=.*/ENVIRONMENT=${ENVIRONMENT}/" .env
fi

print_success "Environment file created: .env"

# Check for required API keys
print_info "Checking for required API keys..."
MISSING_KEYS=()

if ! grep -q "^ANTHROPIC_API_KEY=.*[^[:space:]]" .env; then
    MISSING_KEYS+=("ANTHROPIC_API_KEY")
fi

if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
    print_warning "The following required API keys are missing:"
    for key in "${MISSING_KEYS[@]}"; do
        echo "  - $key"
    done
    print_info "Please update your .env file with the required API keys."
fi

# Environment-specific instructions
case $ENVIRONMENT in
    "local")
        print_info "Local environment setup complete!"
        print_info "To start the development environment:"
        print_info "  docker compose -f docker-compose.yml -f environments/docker/docker-compose.local.yml up"
        ;;
    "development")
        print_info "Development environment setup complete!"
        print_info "To start the development environment:"
        print_info "  docker compose -f docker-compose.yml -f environments/docker/docker-compose.development.yml up"
        ;;
    "testing")
        print_info "Testing environment setup complete!"
        print_info "To run tests:"
        print_info "  docker compose -f docker-compose.yml -f environments/docker/docker-compose.testing.yml up"
        ;;
    "staging")
        print_info "Staging environment setup complete!"
        print_warning "Remember to set production-like secrets!"
        print_info "To deploy to staging:"
        print_info "  docker compose -f docker-compose.yml -f environments/docker/docker-compose.staging.yml up -d"
        ;;
    "production")
        print_info "Production environment setup complete!"
        print_warning "⚠️  IMPORTANT: Production environment requires additional setup:"
        print_warning "  - Set all production secrets (don't use default values)"
        print_warning "  - Configure SSL certificates"
        print_warning "  - Set up monitoring and logging"
        print_warning "  - Configure backup strategies"
        print_info "To deploy to production:"
        print_info "  docker compose -f docker-compose.yml -f environments/docker/docker-compose.production.yml up -d"
        ;;
esac

print_success "Environment setup complete!"
