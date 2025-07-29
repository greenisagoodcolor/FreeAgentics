#!/bin/bash
# Fix dependencies script

echo "Fixing dependencies..."

# Install PyJWT
./venv/bin/pip install PyJWT

# Install web dependencies
cd web && npm install

echo "Dependencies fixed!"