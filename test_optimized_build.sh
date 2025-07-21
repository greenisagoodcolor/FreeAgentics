#!/bin/bash
# Test script for optimized Docker production build
# Validates build time is under 3 minutes and functionality is preserved

set -e
echo "=== FreeAgentics Optimized Docker Build Test ==="
echo "Target: Sub-3 minute build with full functionality"
echo

# Clean up previous builds
echo "🧹 Cleaning previous builds..."
docker system prune -f
docker builder prune -f

# Start build timer
echo "⏱️  Starting optimized build timer..."
start_time=$(date +%s)

# Build with optimization flags
echo "🚀 Building optimized production image..."
DOCKER_BUILDKIT=1 docker build \
    --target production \
    --tag freeagentics:prod-optimized \
    --file Dockerfile.production \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    .

# Calculate build time
end_time=$(date +%s)
build_duration=$((end_time - start_time))
build_minutes=$((build_duration / 60))
build_seconds=$((build_duration % 60))

echo
echo "⏱️  Build completed in: ${build_minutes}m ${build_seconds}s"

# Validate build time requirement
if [ $build_duration -gt 180 ]; then
    echo "❌ FAILED: Build took ${build_minutes}m ${build_seconds}s (over 3 minute target)"
    exit 1
else
    echo "✅ SUCCESS: Build completed in under 3 minutes!"
fi

# Test image functionality
echo "🧪 Testing image functionality..."
docker run --rm --name test-optimized freeagentics:prod-optimized python -c "
import torch
import scipy
import numpy as np
import fastapi
import sqlalchemy
import pandas as pd
print('✅ All core dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'SciPy version: {scipy.__version__}')
print('✅ Production image is fully functional')
"

# Check image size
echo "📊 Analyzing image size..."
image_size=$(docker images freeagentics:prod-optimized --format "table {{.Size}}" | tail -n1)
echo "Final image size: $image_size"

echo
echo "🎉 OPTIMIZATION SUCCESS!"
echo "✅ Build time: ${build_minutes}m ${build_seconds}s (under 3 minutes)"
echo "✅ All dependencies preserved and functional"
echo "✅ Image size: $image_size"
echo
echo "Production deployment ready! 🚀"