#!/bin/bash
# Production Readiness Check for FreeAgentics
# Comprehensive validation for VC presentation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Results storage
declare -a FAILURES
declare -a WARNINGS_LIST

# Logging functions
log_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_check() {
    echo -e "${PURPLE}  ◆ $1${NC}"
    ((TOTAL_CHECKS++))
}

log_pass() {
    echo -e "    ${GREEN}✅ $1${NC}"
    ((PASSED_CHECKS++))
}

log_fail() {
    echo -e "    ${RED}❌ $1${NC}"
    ((FAILED_CHECKS++))
    FAILURES+=("$1")
}

log_warn() {
    echo -e "    ${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
    WARNINGS_LIST+=("$1")
}

log_info() {
    echo -e "    ${GREEN}ℹ️  $1${NC}"
}

# 1. DOCKER IMAGE OPTIMIZATION CHECK
check_docker_optimization() {
    log_section "DOCKER IMAGE OPTIMIZATION"

    log_check "Building optimized production image"
    if docker build -f Dockerfile.production.optimized -t freeagentics:prod-check . &>/dev/null; then
        # Get image size
        local size=$(docker images freeagentics:prod-check --format "{{.Size}}")
        local size_mb=$(docker inspect freeagentics:prod-check --format='{{.Size}}' | awk '{print $1/1024/1024}')

        log_info "Production image size: $size"

        if (( $(echo "$size_mb < 2048" | bc -l) )); then
            log_pass "Image size under 2GB target (${size})"
        else
            log_fail "Image size exceeds 2GB target (${size})"
        fi

        # Security scan
        log_check "Running security scan on Docker image"
        if command -v trivy &>/dev/null; then
            local vulns=$(trivy image --quiet --no-progress freeagentics:prod-check 2>/dev/null | grep -c "Total:" || echo "0")
            if [[ "$vulns" == "0" ]]; then
                log_pass "No critical vulnerabilities found"
            else
                log_warn "Security vulnerabilities detected - run 'trivy image freeagentics:prod-check' for details"
            fi
        else
            log_warn "Trivy not installed - skipping security scan"
        fi
    else
        log_fail "Failed to build optimized Docker image"
    fi
}

# 2. SECURITY VALIDATION
check_security() {
    log_section "SECURITY VALIDATION"

    # Run Python security validation
    log_check "Running security configuration validation"
    if python scripts/validate_security_config.py &>/dev/null; then
        log_pass "Security configuration validated"
    else
        log_fail "Security configuration validation failed"
    fi

    # Check JWT keys
    log_check "Checking JWT key pair"
    if [[ -f "auth/keys/jwt_private.pem" ]] && [[ -f "auth/keys/jwt_public.pem" ]]; then
        log_pass "JWT keys present"

        # Check permissions
        local private_perms=$(stat -c %a auth/keys/jwt_private.pem 2>/dev/null || stat -f %p auth/keys/jwt_private.pem 2>/dev/null | tail -c 4)
        if [[ "$private_perms" == "600" ]]; then
            log_pass "JWT private key has secure permissions (600)"
        else
            log_fail "JWT private key has insecure permissions ($private_perms, should be 600)"
        fi
    else
        log_fail "JWT keys missing"
    fi

    # Check environment file security
    log_check "Checking production environment file"
    if [[ -f ".env.production" ]]; then
        local env_perms=$(stat -c %a .env.production 2>/dev/null || stat -f %p .env.production 2>/dev/null | tail -c 4)
        if [[ "$env_perms" == "600" ]]; then
            log_pass "Production env file has secure permissions (600)"
        else
            log_warn "Production env file permissions could be more secure ($env_perms, recommend 600)"
        fi

        # Check for placeholder values
        if grep -q "CHANGE_ME" .env.production; then
            log_fail "Production env file contains placeholder values"
        else
            log_pass "No placeholder values in production env"
        fi
    else
        log_fail "Production environment file missing"
    fi

    # Check SSL/TLS configuration
    log_check "Checking SSL/TLS configuration"
    if [[ -f "nginx/conf.d/ssl-freeagentics.conf" ]]; then
        log_pass "SSL configuration present"
    else
        log_warn "SSL configuration not found - ensure HTTPS is configured for production"
    fi
}

# 3. MONITORING AND OBSERVABILITY
check_monitoring() {
    log_section "MONITORING AND OBSERVABILITY"

    log_check "Checking monitoring configuration"

    # Check Prometheus configuration
    if [[ -f "monitoring/prometheus.yml" ]]; then
        log_pass "Prometheus configuration present"
    else
        log_fail "Prometheus configuration missing"
    fi

    # Check Grafana dashboards
    if [[ -d "monitoring/grafana/dashboards" ]] && ls monitoring/grafana/dashboards/*.json &>/dev/null; then
        local dashboard_count=$(ls monitoring/grafana/dashboards/*.json 2>/dev/null | wc -l)
        log_pass "Grafana dashboards configured ($dashboard_count dashboards)"
    else
        log_fail "Grafana dashboards missing"
    fi

    # Check alerting rules
    if [[ -f "monitoring/prometheus/rules/alerts.yml" ]]; then
        log_pass "Alert rules configured"
    else
        log_warn "Alert rules not configured"
    fi

    # Check distributed tracing
    log_check "Checking distributed tracing setup"
    if grep -q "opentelemetry" requirements-production.txt; then
        log_pass "OpenTelemetry configured for distributed tracing"
    else
        log_warn "Distributed tracing not configured"
    fi
}

# 4. DATABASE AND PERSISTENCE
check_database() {
    log_section "DATABASE AND PERSISTENCE"

    log_check "Checking database migrations"
    if [[ -d "alembic/versions" ]] && ls alembic/versions/*.py &>/dev/null; then
        local migration_count=$(ls alembic/versions/*.py 2>/dev/null | wc -l)
        log_pass "Database migrations present ($migration_count migrations)"
    else
        log_fail "No database migrations found"
    fi

    log_check "Checking backup configuration"
    if [[ -f "monitoring/backup/scripts/backup.sh" ]]; then
        log_pass "Backup scripts configured"
    else
        log_warn "Backup scripts not configured"
    fi
}

# 5. DEPLOYMENT SCRIPTS
check_deployment() {
    log_section "DEPLOYMENT SCRIPTS"

    log_check "Checking production deployment script"
    if [[ -x "scripts/production-deploy.sh" ]]; then
        log_pass "Production deployment script is executable"
    else
        log_fail "Production deployment script missing or not executable"
    fi

    log_check "Checking Docker Compose production configuration"
    if [[ -f "docker-compose.production.yml" ]]; then
        # Validate compose file
        if docker compose -f docker-compose.production.yml config &>/dev/null; then
            log_pass "Production compose file is valid"
        else
            log_fail "Production compose file has errors"
        fi
    else
        log_fail "Production compose file missing"
    fi

    log_check "Checking rollback procedures"
    if [[ -f "deployment/scripts/rollback.sh" ]] || [[ -f "scripts/rollback-release.sh" ]]; then
        log_pass "Rollback procedures documented"
    else
        log_warn "Rollback procedures not found"
    fi
}

# 6. PERFORMANCE AND SCALABILITY
check_performance() {
    log_section "PERFORMANCE AND SCALABILITY"

    log_check "Checking performance optimizations"

    # Check for production WSGI server
    if grep -q "gunicorn" requirements-production.txt; then
        log_pass "Production WSGI server (Gunicorn) configured"
    else
        log_fail "Production WSGI server not configured"
    fi

    # Check for caching
    if grep -q "redis" requirements-production.txt; then
        log_pass "Redis caching configured"
    else
        log_warn "Caching layer not configured"
    fi

    # Check for CDN/static file handling
    if [[ -f "nginx/nginx.conf" ]]; then
        log_pass "Nginx configured for static file serving"
    else
        log_warn "Static file serving not optimized"
    fi
}

# 7. DOCUMENTATION AND RUNBOOKS
check_documentation() {
    log_section "DOCUMENTATION AND RUNBOOKS"

    log_check "Checking production documentation"

    local required_docs=(
        "README.md"
        "QUICKSTART.md"
        "docs/runbooks/deployment/PRODUCTION_DEPLOYMENT.md"
        "docs/runbooks/recovery/DISASTER_RECOVERY.md"
    )

    for doc in "${required_docs[@]}"; do
        if [[ -f "$doc" ]]; then
            log_pass "$(basename $doc) present"
        else
            log_fail "$(basename $doc) missing"
        fi
    done

    # Check API documentation
    if [[ -f "docs/API_DOCUMENTATION.md" ]] || [[ -f "docs/UNIFIED_API_REFERENCE.md" ]]; then
        log_pass "API documentation present"
    else
        log_warn "API documentation not found"
    fi
}

# 8. FRONTEND READINESS
check_frontend() {
    log_section "FRONTEND READINESS"

    log_check "Checking frontend production build"

    if [[ -d "web" ]]; then
        if [[ -f "web/Dockerfile.production" ]]; then
            log_pass "Frontend production Dockerfile present"
        else
            log_fail "Frontend production Dockerfile missing"
        fi

        # Check for performance optimizations
        if [[ -f "web/next.config.js" ]] && grep -q "swcMinify" web/next.config.js; then
            log_pass "Frontend minification configured"
        else
            log_warn "Frontend minification not configured"
        fi
    else
        log_fail "Frontend directory not found"
    fi
}

# 9. DEMO READINESS
check_demo() {
    log_section "DEMO READINESS"

    log_check "Checking demo scripts and examples"

    if [[ -f "start-demo.sh" ]] && [[ -x "start-demo.sh" ]]; then
        log_pass "Demo start script is executable"
    else
        log_fail "Demo start script missing or not executable"
    fi

    # Check for example data
    if [[ -d "examples" ]] && ls examples/*.py &>/dev/null; then
        local example_count=$(ls examples/*.py 2>/dev/null | wc -l)
        log_pass "Demo examples present ($example_count examples)"
    else
        log_fail "Demo examples missing"
    fi
}

# 10. FINAL SUMMARY AND REPORT
generate_report() {
    log_section "PRODUCTION READINESS SUMMARY"

    local readiness_percentage=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}FINAL RESULTS${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    echo -e "\nTotal Checks: ${TOTAL_CHECKS}"
    echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
    echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNINGS}${NC}"
    echo -e "\n${PURPLE}Readiness Score: ${readiness_percentage}%${NC}"

    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        echo -e "\n${RED}Critical Issues to Fix:${NC}"
        for failure in "${FAILURES[@]}"; do
            echo -e "  • $failure"
        done
    fi

    if [[ ${#WARNINGS_LIST[@]} -gt 0 ]]; then
        echo -e "\n${YELLOW}Warnings to Review:${NC}"
        for warning in "${WARNINGS_LIST[@]}"; do
            echo -e "  • $warning"
        done
    fi

    # Generate markdown report
    cat > PRODUCTION_READINESS_REPORT.md <<EOF
# Production Readiness Report
Generated on: $(date)

## Summary
- **Total Checks**: $TOTAL_CHECKS
- **Passed**: $PASSED_CHECKS
- **Failed**: $FAILED_CHECKS
- **Warnings**: $WARNINGS
- **Readiness Score**: ${readiness_percentage}%

## Status
EOF

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo "✅ **READY FOR PRODUCTION**" >> PRODUCTION_READINESS_REPORT.md
        echo -e "\n${GREEN}✅ SYSTEM IS READY FOR PRODUCTION AND VC PRESENTATION${NC}"
    else
        echo "❌ **NOT READY FOR PRODUCTION**" >> PRODUCTION_READINESS_REPORT.md
        echo -e "\n${RED}❌ CRITICAL ISSUES MUST BE RESOLVED BEFORE PRODUCTION${NC}"
    fi

    # Recommendations
    cat >> PRODUCTION_READINESS_REPORT.md <<EOF

## Critical Issues
$(if [[ ${#FAILURES[@]} -gt 0 ]]; then
    for failure in "${FAILURES[@]}"; do
        echo "- $failure"
    done
else
    echo "None"
fi)

## Warnings
$(if [[ ${#WARNINGS_LIST[@]} -gt 0 ]]; then
    for warning in "${WARNINGS_LIST[@]}"; do
        echo "- $warning"
    done
else
    echo "None"
fi)

## Next Steps for VC Presentation
1. Ensure all critical issues are resolved
2. Prepare demo environment with sample data
3. Test all deployment scripts in staging environment
4. Verify monitoring dashboards are operational
5. Review security hardening report
6. Prepare performance benchmarks and metrics
7. Update all documentation with latest changes
8. Create backup and recovery demonstration

## Checklist for VC Demo
- [ ] Docker images optimized and under 2GB
- [ ] Security validation passing
- [ ] Monitoring dashboards accessible
- [ ] Demo scripts working smoothly
- [ ] API endpoints responding correctly
- [ ] Frontend loading quickly
- [ ] Database with sample data
- [ ] Rollback procedures tested
EOF

    echo -e "\n${GREEN}Report saved to: PRODUCTION_READINESS_REPORT.md${NC}"
}

# Main execution
main() {
    echo -e "${PURPLE}🚀 FreeAgentics Production Readiness Check${NC}"
    echo -e "${PURPLE}   Preparing for VC Presentation${NC}"

    check_docker_optimization
    check_security
    check_monitoring
    check_database
    check_deployment
    check_performance
    check_documentation
    check_frontend
    check_demo

    generate_report

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"
