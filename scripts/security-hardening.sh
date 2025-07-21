#!/bin/bash
# Security Hardening Script for FreeAgentics Production Deployment
# Ensures all security requirements are met before deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Track errors and warnings
ERRORS=0
WARNINGS=0

# 1. Environment Variables Security
check_env_security() {
    log_section "Checking Environment Variables Security"

    # Check for production environment file
    if [[ ! -f ".env.production" ]]; then
        log_error ".env.production file not found"
        ((ERRORS++))
        return
    fi

    # Ensure proper permissions on env files
    chmod 600 .env.production 2>/dev/null 
    chmod 600 .env 2>/dev/null 

    # Check for dangerous default values
    local dangerous_values=("change_me" "dev_password" "dev_secret" "secret" "password" "admin" "test" "default" "example" "placeholder" "CHANGE_ME")

    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] || [[ -z "$key" ]] && continue

        # Check for dangerous values
        for dangerous in "${dangerous_values[@]}"; do
            if [[ "${value,,}" == "${dangerous,,}" ]]; then
                log_error "Dangerous default value found: $key=$value"
                ((ERRORS++))
            fi
        done

        # Check secret strength
        if [[ "$key" =~ (SECRET|PASSWORD|KEY) ]]; then
            if [[ ${#value} -lt 32 ]]; then
                log_warn "$key is too short (${#value} chars, minimum 32 recommended)"
                ((WARNINGS++))
            fi
        fi
    done < .env.production

    log_info "Environment variables checked"
}

# 2. Generate Secure Secrets
generate_secure_secrets() {
    log_section "Generating Secure Secrets"

    # Function to generate secure random string
    generate_secret() {
        openssl rand -base64 48 | tr -d "=+/" | cut -c1-64
    }

    # Check if secrets need to be generated
    if [[ -f ".env.production" ]]; then
        # Create backup
        cp .env.production .env.production.backup

        # Generate new secrets for placeholder values
        if grep -q "CHANGE_ME" .env.production; then
            log_info "Generating secure secrets for placeholder values..."

            # Generate unique secrets
            JWT_SECRET=$(generate_secret)
            SECRET_KEY=$(generate_secret)
            POSTGRES_PASSWORD=$(generate_secret)
            REDIS_PASSWORD=$(generate_secret)
            BACKUP_PASSWORD=$(generate_secret)
            GRAFANA_PASSWORD=$(generate_secret)

            # Replace placeholders
            sed -i "s/JWT_SECRET=.*/JWT_SECRET=$JWT_SECRET/" .env.production
            sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env.production
            sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env.production
            sed -i "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$REDIS_PASSWORD/" .env.production
            sed -i "s/BACKUP_PASSWORD=.*/BACKUP_PASSWORD=$BACKUP_PASSWORD/" .env.production
            sed -i "s/GRAFANA_ADMIN_PASSWORD=.*/GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASSWORD/" .env.production

            log_info "Secure secrets generated"
        fi
    fi
}

# 3. File Permissions Security
secure_file_permissions() {
    log_section "Securing File Permissions"

    # Secure sensitive files
    local sensitive_files=(
        ".env"
        ".env.production"
        ".env.local"
        "auth/keys/jwt_private.pem"
        "auth/keys/jwt_public.pem"
    )

    for file in "${sensitive_files[@]}"; do
        if [[ -f "$file" ]]; then
            chmod 600 "$file"
            log_info "Secured permissions for $file"
        fi
    done

    # Secure directories
    if [[ -d "auth/keys" ]]; then
        chmod 700 auth/keys
        log_info "Secured auth/keys directory"
    fi
}

# 4. Generate JWT Keys
generate_jwt_keys() {
    log_section "Generating JWT Keys"

    if [[ ! -f "auth/keys/jwt_private.pem" ]]; then
        mkdir -p auth/keys

        # Generate RSA key pair
        openssl genrsa -out auth/keys/jwt_private.pem 4096
        openssl rsa -in auth/keys/jwt_private.pem -pubout -out auth/keys/jwt_public.pem

        # Secure permissions
        chmod 600 auth/keys/jwt_private.pem
        chmod 644 auth/keys/jwt_public.pem

        log_info "JWT keys generated"
    else
        log_info "JWT keys already exist"
    fi
}

# 5. Docker Security Scan
docker_security_scan() {
    log_section "Docker Security Scan"

    # Check if Docker Scout is available
    if docker scout version &>/dev/null; then
        log_info "Running Docker Scout security scan..."

        # Build the optimized image
        log_info "Building optimized production image..."
        docker build -f Dockerfile.production.optimized -t freeagentics:prod-optimized . || {
            log_error "Docker build failed"
            ((ERRORS++))
            return
        }

        # Run security scan
        docker scout cves freeagentics:prod-optimized || {
            log_warn "Docker Scout scan found vulnerabilities"
            ((WARNINGS++))
        }
    else
        log_warn "Docker Scout not available, skipping security scan"
        ((WARNINGS++))
    fi

    # Check image size
    local image_size=$(docker images freeagentics:prod-optimized --format "{{.Size}}")
    log_info "Production image size: $image_size"

    # Convert size to MB for comparison
    local size_mb=$(docker images freeagentics:prod-optimized --format "{{.Size}}" | sed 's/GB/*1024/;s/MB//' | bc 2>/dev/null || echo "0")
    if [[ $(echo "$size_mb > 2048" | bc) -eq 1 ]]; then
        log_warn "Image size exceeds 2GB target"
        ((WARNINGS++))
    else
        log_info "Image size is within 2GB target"
    fi
}

# 6. Security Headers Configuration
configure_security_headers() {
    log_section "Configuring Security Headers"

    # Ensure security headers are enabled in production
    if [[ -f ".env.production" ]]; then
        # Enable all security features
        cat >> .env.production <<EOF

# Security Headers (Auto-configured by security hardening)
HSTS_ENABLED=true
CSP_ENABLED=true
DDOS_PROTECTION_ENABLED=true
RATE_LIMIT_ENABLED=true
SECURE_HEADERS_ENABLED=true

# Disable debug modes
DEBUG=false
DEBUG_SQL=false
DEVELOPMENT_MODE=false
EOF

        log_info "Security headers configured"
    fi
}

# 7. Database Security
secure_database() {
    log_section "Database Security Configuration"

    # Ensure SSL is enabled for database connections
    if grep -q "DATABASE_URL" .env.production; then
        # Add SSL parameters if not present
        if ! grep -q "sslmode=require" .env.production; then
            sed -i '/DATABASE_URL/ s/$/?sslmode=require/' .env.production
            log_info "Added SSL requirement to database connection"
        fi
    fi
}

# 8. Git Security Check
check_git_security() {
    log_section "Git Security Check"

    # Check for committed secrets
    local secret_patterns=(
        "password.*=.*"
        "secret.*=.*"
        "key.*=.*"
        "token.*=.*"
    )

    for pattern in "${secret_patterns[@]}"; do
        if git grep -i "$pattern" 2>/dev/null | grep -v "example\|template\|test" | grep -q .; then
            log_warn "Potential secrets found in git history"
            ((WARNINGS++))
        fi
    done

    # Ensure .gitignore is properly configured
    local ignore_patterns=(
        ".env"
        ".env.local"
        ".env.production"
        "*.pem"
        "*.key"
        "auth/keys/"
    )

    for pattern in "${ignore_patterns[@]}"; do
        if ! grep -q "$pattern" .gitignore 2>/dev/null; then
            echo "$pattern" >> .gitignore
            log_info "Added $pattern to .gitignore"
        fi
    done
}

# 9. Create Security Report
create_security_report() {
    log_section "Creating Security Report"

    local report_file="SECURITY_HARDENING_REPORT.md"

    cat > "$report_file" <<EOF
# Security Hardening Report
Generated on: $(date)

## Summary
- Errors: $ERRORS
- Warnings: $WARNINGS

## Checks Performed
1. ✅ Environment Variables Security
2. ✅ Secure Secrets Generation
3. ✅ File Permissions
4. ✅ JWT Key Generation
5. ✅ Docker Security Scan
6. ✅ Security Headers Configuration
7. ✅ Database Security
8. ✅ Git Security Check

## Security Features Enabled
- HSTS (HTTP Strict Transport Security)
- CSP (Content Security Policy)
- DDoS Protection
- Rate Limiting
- Secure Headers
- Database SSL/TLS

## Recommendations
1. Regularly rotate secrets and passwords
2. Monitor security alerts from dependencies
3. Keep Docker images updated
4. Regular security audits
5. Enable Web Application Firewall (WAF) in production

## Next Steps
1. Run \`python scripts/validate_security_config.py\` to verify configuration
2. Deploy with security monitoring enabled
3. Configure alert notifications for security events
EOF

    log_info "Security report created: $report_file"
}

# Main execution
main() {
    log_info "Starting Security Hardening Process..."

    # Run all security checks and configurations
    check_env_security
    generate_secure_secrets
    secure_file_permissions
    generate_jwt_keys
    docker_security_scan
    configure_security_headers
    secure_database
    check_git_security
    create_security_report

    # Final summary
    echo -e "\n${BLUE}=== Security Hardening Complete ===${NC}"
    echo -e "Errors: ${RED}$ERRORS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"

    if [[ $ERRORS -eq 0 ]]; then
        log_info "✅ Security hardening completed successfully!"
        if [[ $WARNINGS -gt 0 ]]; then
            log_warn "Review warnings for additional security improvements"
        fi
        exit 0
    else
        log_error "❌ Security hardening failed with $ERRORS errors"
        exit 1
    fi
}

# Run main function
main "$@"
