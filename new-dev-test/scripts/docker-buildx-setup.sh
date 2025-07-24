#!/usr/bin/env bash
# Docker Buildx setup for multi-architecture builds
# Supports linux/amd64 and linux/arm64 platforms
# Following Evan You & Rich Harris principles: Fast, reliable, bulletproof builds

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILDER_NAME="freeagentics-builder"
PLATFORMS="linux/amd64,linux/arm64"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
REPOSITORY="${DOCKER_REPOSITORY:-yourusername/freeagentics}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_info "Docker version: $(docker --version)"
}

# Check if Docker Buildx is available
check_buildx() {
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is not available. Please update Docker."
        exit 1
    fi
    log_info "Docker Buildx version: $(docker buildx version)"
}

# Setup QEMU for cross-platform builds
setup_qemu() {
    log_info "Setting up QEMU for cross-platform builds..."

    # Check if running in CI environment
    if [ -n "${CI:-}" ]; then
        # GitHub Actions specific setup
        docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    else
        # Local development setup
        if ! docker run --rm --privileged multiarch/qemu-user-static --reset -p yes; then
            log_warning "Failed to setup QEMU. Trying alternative method..."
            docker run --rm --privileged tonistiigi/binfmt --install all
        fi
    fi

    log_success "QEMU setup completed"
}

# Create or update buildx builder
setup_builder() {
    log_info "Setting up Docker Buildx builder: ${BUILDER_NAME}"

    # Check if builder already exists
    if docker buildx ls | grep -q "${BUILDER_NAME}"; then
        log_info "Builder ${BUILDER_NAME} already exists. Removing old builder..."
        docker buildx rm "${BUILDER_NAME}" || true
    fi

    # Create new builder with proper configuration
    docker buildx create \
        --name "${BUILDER_NAME}" \
        --driver docker-container \
        --driver-opt network=host \
        --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=50000000 \
        --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=50000000 \
        --platform "${PLATFORMS}" \
        --use

    # Bootstrap the builder
    docker buildx inspect --bootstrap

    log_success "Builder ${BUILDER_NAME} created and ready"
}

# Verify multi-arch support
verify_platforms() {
    log_info "Verifying platform support..."

    local supported_platforms=$(docker buildx inspect --bootstrap | grep Platforms | cut -d: -f2)

    for platform in ${PLATFORMS//,/ }; do
        if echo "$supported_platforms" | grep -q "$platform"; then
            log_success "Platform $platform is supported"
        else
            log_error "Platform $platform is NOT supported"
            exit 1
        fi
    done
}

# Build multi-architecture images
build_images() {
    local tag="${1:-latest}"
    local push="${2:-false}"

    log_info "Building multi-architecture images..."
    log_info "Tag: ${tag}"
    log_info "Push to registry: ${push}"

    # Build backend image
    log_info "Building backend image..."
    docker buildx build \
        --platform "${PLATFORMS}" \
        --file Dockerfile.multiarch \
        --target production \
        --tag "${REGISTRY}/${REPOSITORY}-backend:${tag}" \
        --tag "${REGISTRY}/${REPOSITORY}-backend:latest" \
        --cache-from "type=registry,ref=${REGISTRY}/${REPOSITORY}-backend:buildcache" \
        --cache-to "type=registry,ref=${REGISTRY}/${REPOSITORY}-backend:buildcache,mode=max" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="${tag}" \
        $([ "$push" = "true" ] && echo "--push" || echo "--load") \
        .

    # Build frontend image
    log_info "Building frontend image..."
    docker buildx build \
        --platform "${PLATFORMS}" \
        --file web/Dockerfile.multiarch \
        --target production \
        --tag "${REGISTRY}/${REPOSITORY}-frontend:${tag}" \
        --tag "${REGISTRY}/${REPOSITORY}-frontend:latest" \
        --cache-from "type=registry,ref=${REGISTRY}/${REPOSITORY}-frontend:buildcache" \
        --cache-to "type=registry,ref=${REGISTRY}/${REPOSITORY}-frontend:buildcache,mode=max" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="${tag}" \
        --build-arg NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-https://api.yourdomain.com}" \
        $([ "$push" = "true" ] && echo "--push" || echo "--load") \
        web/

    log_success "Multi-architecture images built successfully"
}

# Main execution
main() {
    log_info "Starting Docker Buildx setup for FreeAgentics..."

    # Parse command line arguments
    local command="${1:-setup}"
    local tag="${2:-latest}"
    local push="${3:-false}"

    case "$command" in
        setup)
            check_docker
            check_buildx
            setup_qemu
            setup_builder
            verify_platforms
            log_success "Docker Buildx setup completed successfully!"
            ;;
        build)
            check_docker
            check_buildx
            build_images "$tag" "$push"
            ;;
        test)
            check_docker
            check_buildx
            log_info "Testing multi-arch build (dry run)..."
            docker buildx build \
                --platform "${PLATFORMS}" \
                --file Dockerfile.multiarch \
                --target production \
                --no-cache \
                --progress plain \
                .
            log_success "Test build completed successfully!"
            ;;
        clean)
            log_info "Cleaning up Docker Buildx builder..."
            docker buildx rm "${BUILDER_NAME}" || true
            log_success "Cleanup completed"
            ;;
        *)
            echo "Usage: $0 {setup|build|test|clean} [tag] [push]"
            echo "  setup - Setup Docker Buildx for multi-arch builds"
            echo "  build - Build multi-arch images"
            echo "  test  - Test multi-arch build (dry run)"
            echo "  clean - Remove Docker Buildx builder"
            echo ""
            echo "Examples:"
            echo "  $0 setup                    # Setup buildx"
            echo "  $0 build v1.0.0 true       # Build and push v1.0.0"
            echo "  $0 build latest false      # Build latest locally"
            echo "  $0 test                    # Test build process"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
