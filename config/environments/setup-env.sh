#!/bin/bash

# CogniticNet Environment Setup Script
# This script helps create environment files from templates

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}CogniticNet Environment Setup${NC}"
echo "================================"

# Function to create env file from template
create_env_file() {
    local template=$1
    local target=$2

    if [ -f "$SCRIPT_DIR/$target" ]; then
        echo -e "${YELLOW}Warning: $target already exists${NC}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping $target"
            return
        fi
    fi

    cp "$SCRIPT_DIR/$template" "$SCRIPT_DIR/$target"
    echo -e "${GREEN}Created $target from $template${NC}"

    # Set proper permissions
    chmod 600 "$SCRIPT_DIR/$target"
    echo "Set permissions to 600 (read/write for owner only)"
}

# Function to generate secure random key
generate_key() {
    openssl rand -base64 32 | tr -d '\n'
}

# Check if templates exist
if [ ! -f "$SCRIPT_DIR/env.example" ]; then
    echo -e "${RED}Error: env.example not found${NC}"
    echo "Please ensure you're running this script from the environments directory"
    exit 1
fi

# Menu
echo
echo "Select environment to set up:"
echo "1) Development (.env.development)"
echo "2) Test (.env.test)"
echo "3) Demo (.env.demo)"
echo "4) Production (.env.production)"
echo "5) All environments"
echo "6) Generate secure keys only"
echo "0) Exit"
echo

read -p "Enter your choice [0-6]: " choice

case $choice in
    1)
        create_env_file "env.development" ".env.development"
        echo
        echo "Next steps for development:"
        echo "1. Add your API keys (ANTHROPIC_API_KEY, etc.)"
        echo "2. Update database password if needed"
        echo "3. Run: docker-compose -f docker/development/docker-compose.yml up"
        ;;
    2)
        create_env_file "env.test" ".env.test"
        echo
        echo "Test environment created with mock providers"
        ;;
    3)
        create_env_file "env.demo" ".env.demo"
        echo
        echo "Demo environment created"
        echo "Run: docker-compose -f docker/demo/docker-compose.yml up"
        ;;
    4)
        create_env_file "env.production" ".env.production"
        echo
        echo -e "${YELLOW}IMPORTANT for production:${NC}"
        echo "1. Generate all secure keys (option 6)"
        echo "2. Update all placeholder values"
        echo "3. Set proper domain names"
        echo "4. Configure external services"
        echo "5. Use secrets management service"
        ;;
    5)
        create_env_file "env.development" ".env.development"
        create_env_file "env.test" ".env.test"
        create_env_file "env.demo" ".env.demo"
        create_env_file "env.production" ".env.production"
        echo
        echo "All environment files created"
        ;;
    6)
        echo
        echo "Generating secure keys..."
        echo
        echo "ENCRYPTION_KEY=$(generate_key)"
        echo "JWT_SECRET=$(generate_key)"
        echo "API_KEY_SALT=$(generate_key)"
        echo "SESSION_SECRET=$(generate_key)"
        echo
        echo -e "${YELLOW}Copy these values to your .env files${NC}"
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}Setup complete!${NC}"
echo
echo "Remember:"
echo "- Never commit .env files to version control"
echo "- Keep production secrets secure"
echo "- Rotate keys regularly"
echo "- Use different keys for each environment"
