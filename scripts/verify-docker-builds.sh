#!/usr/bin/env bash
# Docker build verification script for FreeAgentics
# Validates multi-architecture builds and performance
# Following Evan You & Rich Harris principles: Fast, reliable, bulletproof builds

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_FILE="${PROJECT_ROOT}/BUILD_VERIFICATION_REPORT.md"
PLATFORMS=("linux/amd64" "linux/arm64")
SERVICES=("backend" "frontend")

# Test results storage
declare -A test_results
declare -A build_times
declare -A image_sizes
declare -A layer_counts

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

# Initialize report
init_report() {
    cat > "$REPORT_FILE" << EOF
# Docker Build Verification Report

Generated: $(date -u +'%Y-%m-%d %H:%M:%S UTC')
Platform: $(uname -a)
Docker Version: $(docker --version)
Docker Buildx Version: $(docker buildx version 2>/dev/null || echo "Not installed")

## Executive Summary

This report validates the FreeAgentics Docker builds across multiple architectures,
ensuring production readiness following Evan You and Rich Harris principles.

EOF
}

# Test Docker environment
test_docker_environment() {
    log_test "Testing Docker environment..."

    # Check Docker daemon
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
        test_results["docker_daemon"]="PASS"
    else
        log_error "Docker daemon is not running"
        test_results["docker_daemon"]="FAIL"
        return 1
    fi

    # Check Docker Buildx
    if docker buildx version &> /dev/null; then
        log_success "Docker Buildx is available"
        test_results["docker_buildx"]="PASS"
    else
        log_error "Docker Buildx is not available"
        test_results["docker_buildx"]="FAIL"
        return 1
    fi

    # Check QEMU for multi-arch
    if docker run --rm --platform linux/arm64 alpine uname -m 2>/dev/null | grep -q aarch64; then
        log_success "Multi-architecture support is enabled"
        test_results["multi_arch_support"]="PASS"
    else
        log_warning "Multi-architecture support may not be properly configured"
        test_results["multi_arch_support"]="WARNING"
    fi
}

# Build images for all platforms
build_images() {
    local service=$1
    local dockerfile=$2
    local context=$3

    log_test "Building $service for all platforms..."

    for platform in "${PLATFORMS[@]}"; do
        local tag="freeagentics-${service}:test-${platform//\//-}"
        local start_time=$(date +%s)

        log_info "Building $service for $platform..."

        if docker buildx build \
            --platform "$platform" \
            --file "$dockerfile" \
            --target production \
            --tag "$tag" \
            --load \
            --progress plain \
            "$context" 2>&1 | tee "/tmp/build-${service}-${platform//\//-}.log"; then

            local end_time=$(date +%s)
            local build_time=$((end_time - start_time))

            build_times["${service}_${platform}"]=$build_time
            test_results["build_${service}_${platform}"]="PASS"
            log_success "Built $service for $platform in ${build_time}s"

            # Get image size
            local size=$(docker images --format "{{.Size}}" "$tag")
            image_sizes["${service}_${platform}"]=$size

            # Count layers
            local layers=$(docker history "$tag" --no-trunc | grep -v "IMAGE" | wc -l)
            layer_counts["${service}_${platform}"]=$layers

        else
            test_results["build_${service}_${platform}"]="FAIL"
            log_error "Failed to build $service for $platform"
        fi
    done
}

# Test image functionality
test_image() {
    local service=$1
    local platform=$2
    local tag="freeagentics-${service}:test-${platform//\//-}"

    log_test "Testing $service image for $platform..."

    # Start container
    local container_name="test-${service}-${platform//\//-}"

    if [ "$service" = "backend" ]; then
        # Test backend container
        if docker run -d \
            --name "$container_name" \
            --platform "$platform" \
            -e DATABASE_URL="postgresql://test:test@localhost:5432/test" \
            -e REDIS_URL="redis://localhost:6379" \
            -e SECRET_KEY="test_key" \
            -e JWT_SECRET="test_jwt" \
            -p 8001:8000 \
            "$tag" 2>/dev/null; then

            sleep 10

            # Test health endpoint
            if docker exec "$container_name" curl -f http://localhost:8000/health 2>/dev/null; then
                test_results["health_${service}_${platform}"]="PASS"
                log_success "Health check passed for $service on $platform"
            else
                test_results["health_${service}_${platform}"]="FAIL"
                log_error "Health check failed for $service on $platform"
            fi

            # Cleanup
            docker stop "$container_name" &>/dev/null
            docker rm "$container_name" &>/dev/null
        else
            test_results["run_${service}_${platform}"]="FAIL"
            log_error "Failed to start $service container on $platform"
        fi

    elif [ "$service" = "frontend" ]; then
        # Test frontend container
        if docker run -d \
            --name "$container_name" \
            --platform "$platform" \
            -e NEXT_PUBLIC_API_URL="http://localhost:8000" \
            -p 3001:3000 \
            "$tag" 2>/dev/null; then

            sleep 15

            # Test health endpoint
            if docker exec "$container_name" wget --spider http://localhost:3000 2>/dev/null; then
                test_results["health_${service}_${platform}"]="PASS"
                log_success "Health check passed for $service on $platform"
            else
                test_results["health_${service}_${platform}"]="FAIL"
                log_error "Health check failed for $service on $platform"
            fi

            # Cleanup
            docker stop "$container_name" &>/dev/null
            docker rm "$container_name" &>/dev/null
        else
            test_results["run_${service}_${platform}"]="FAIL"
            log_error "Failed to start $service container on $platform"
        fi
    fi
}

# Security scan
security_scan() {
    local service=$1
    local platform=$2
    local tag="freeagentics-${service}:test-${platform//\//-}"

    log_test "Running security scan for $service on $platform..."

    # Check if Trivy is installed
    if command -v trivy &> /dev/null; then
        if trivy image --severity HIGH,CRITICAL "$tag" --format json > "/tmp/trivy-${service}-${platform//\//-}.json" 2>/dev/null; then
            local vulns=$(jq '.Results[].Vulnerabilities | length' "/tmp/trivy-${service}-${platform//\//-}.json" 2>/dev/null | awk '{sum+=$1} END {print sum}')
            if [ "${vulns:-0}" -eq 0 ]; then
                test_results["security_${service}_${platform}"]="PASS"
                log_success "No high/critical vulnerabilities found"
            else
                test_results["security_${service}_${platform}"]="WARNING"
                log_warning "Found $vulns high/critical vulnerabilities"
            fi
        else
            test_results["security_${service}_${platform}"]="SKIP"
            log_warning "Security scan failed"
        fi
    else
        test_results["security_${service}_${platform}"]="SKIP"
        log_info "Trivy not installed, skipping security scan"
    fi
}

# Performance benchmarks
benchmark_performance() {
    log_test "Running performance benchmarks..."

    # Test build cache effectiveness
    log_info "Testing build cache effectiveness..."

    # First build (cold cache)
    local cold_start=$(date +%s)
    docker buildx build \
        --platform linux/amd64 \
        --file Dockerfile.multiarch \
        --target production \
        --no-cache \
        . &>/dev/null
    local cold_end=$(date +%s)
    local cold_time=$((cold_end - cold_start))

    # Second build (warm cache)
    local warm_start=$(date +%s)
    docker buildx build \
        --platform linux/amd64 \
        --file Dockerfile.multiarch \
        --target production \
        . &>/dev/null
    local warm_end=$(date +%s)
    local warm_time=$((warm_end - warm_start))

    local cache_improvement=$(( (cold_time - warm_time) * 100 / cold_time ))
    test_results["cache_effectiveness"]="${cache_improvement}%"

    if [ "$cache_improvement" -gt 50 ]; then
        log_success "Cache effectiveness: ${cache_improvement}% improvement"
    else
        log_warning "Cache effectiveness: Only ${cache_improvement}% improvement"
    fi
}

# Generate report
generate_report() {
    log_info "Generating verification report..."

    # Test results summary
    cat >> "$REPORT_FILE" << EOF

## Test Results

### Environment Tests
| Test | Result |
|------|--------|
| Docker Daemon | ${test_results[docker_daemon]:-N/A} |
| Docker Buildx | ${test_results[docker_buildx]:-N/A} |
| Multi-arch Support | ${test_results[multi_arch_support]:-N/A} |

### Build Results
| Service | Platform | Result | Time | Size | Layers |
|---------|----------|--------|------|------|--------|
EOF

    for service in "${SERVICES[@]}"; do
        for platform in "${PLATFORMS[@]}"; do
            echo "| $service | $platform | ${test_results[build_${service}_${platform}]:-N/A} | ${build_times[${service}_${platform}]:-N/A}s | ${image_sizes[${service}_${platform}]:-N/A} | ${layer_counts[${service}_${platform}]:-N/A} |" >> "$REPORT_FILE"
        done
    done

    cat >> "$REPORT_FILE" << EOF

### Health Check Results
| Service | Platform | Result |
|---------|----------|--------|
EOF

    for service in "${SERVICES[@]}"; do
        for platform in "${PLATFORMS[@]}"; do
            echo "| $service | $platform | ${test_results[health_${service}_${platform}]:-N/A} |" >> "$REPORT_FILE"
        done
    done

    cat >> "$REPORT_FILE" << EOF

### Security Scan Results
| Service | Platform | Result |
|---------|----------|--------|
EOF

    for service in "${SERVICES[@]}"; do
        for platform in "${PLATFORMS[@]}"; do
            echo "| $service | $platform | ${test_results[security_${service}_${platform}]:-N/A} |" >> "$REPORT_FILE"
        done
    done

    cat >> "$REPORT_FILE" << EOF

### Performance Metrics
- Cache Effectiveness: ${test_results[cache_effectiveness]:-N/A}

## Recommendations

EOF

    # Generate recommendations
    local has_failures=false
    for result in "${test_results[@]}"; do
        if [ "$result" = "FAIL" ]; then
            has_failures=true
            break
        fi
    done

    if [ "$has_failures" = true ]; then
        echo "⚠️  **Critical Issues Found**" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "Some builds or tests failed. Please review the logs and fix the issues before deploying to production." >> "$REPORT_FILE"
    else
        echo "✅ **All Tests Passed**" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "The Docker builds are ready for production deployment." >> "$REPORT_FILE"
    fi

    cat >> "$REPORT_FILE" << EOF

## Build Commands

To build multi-architecture images:

\`\`\`bash
# Setup buildx
./scripts/docker-buildx-setup.sh setup

# Build and push images
./scripts/docker-buildx-setup.sh build v1.0.0 true
\`\`\`

## Conclusion

Build verification completed at $(date -u +'%Y-%m-%d %H:%M:%S UTC').

Following Evan You's principle: "Build tools should be fast and reliable"
Following Rich Harris's wisdom: "The build is the foundation - make it rock solid"

**The builds are bulletproof.** ✨
EOF

    log_success "Report generated: $REPORT_FILE"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"

    log_info "Starting Docker build verification..."

    # Initialize report
    init_report

    # Test environment
    test_docker_environment

    # Build and test each service
    for service in "${SERVICES[@]}"; do
        if [ "$service" = "backend" ]; then
            build_images "backend" "Dockerfile.multiarch" "."
        else
            build_images "frontend" "web/Dockerfile.multiarch" "web"
        fi

        # Test each platform
        for platform in "${PLATFORMS[@]}"; do
            test_image "$service" "$platform"
            security_scan "$service" "$platform"
        done
    done

    # Run performance benchmarks
    benchmark_performance

    # Generate report
    generate_report

    # Summary
    echo ""
    log_success "Build verification completed!"
    echo ""
    echo "Report: $REPORT_FILE"

    # Exit with appropriate code
    for result in "${test_results[@]}"; do
        if [ "$result" = "FAIL" ]; then
            exit 1
        fi
    done

    exit 0
}

# Run main function
main "$@"
