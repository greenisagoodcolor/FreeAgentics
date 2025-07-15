#!/bin/bash
# Generate production environment variables with secure defaults

set -euo pipefail

ENV_FILE=".env.production"
BACKUP_FILE=".env.production.backup.$(date +%Y%m%d_%H%M%S)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Generate secure random string
generate_secret() {
    openssl rand -base64 48 | tr -d "=+/" | cut -c1-64
}

generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

log_info "Generating secure production environment variables..."

# Create backup if file exists
if [[ -f "$ENV_FILE" ]]; then
    cp "$ENV_FILE" "$BACKUP_FILE"
    log_info "Backup created: $BACKUP_FILE"
fi

# Generate secure values
JWT_SECRET=$(generate_secret)
SECRET_KEY=$(generate_secret)
POSTGRES_PASSWORD=$(generate_password)
REDIS_PASSWORD=$(generate_password)
BACKUP_PASSWORD=$(generate_password)
GRAFANA_PASSWORD=$(generate_password)

# Create/update production environment file
cat > "$ENV_FILE" <<EOF
# FreeAgentics Production Environment Configuration
# Generated on: $(date)
# NEVER commit this file to version control

# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================

# Environment mode
ENVIRONMENT=production
DEVELOPMENT_MODE=false
PRODUCTION=true
DEBUG=false
DEBUG_SQL=false
LOG_LEVEL=INFO

# =============================================================================
# SECURITY SECRETS (Generated)
# =============================================================================

# Application secrets
SECRET_KEY=$SECRET_KEY
JWT_SECRET=$JWT_SECRET

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL Database
DATABASE_URL=postgresql://freeagentics:$POSTGRES_PASSWORD@postgres:5432/freeagentics
POSTGRES_USER=freeagentics
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DB=freeagentics

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================

# Redis for caching and sessions
REDIS_URL=redis://:$REDIS_PASSWORD@redis:6379/0
REDIS_PASSWORD=$REDIS_PASSWORD

# =============================================================================
# SECURITY HEADERS & PROTECTION
# =============================================================================

# Security features (enable all for production)
HSTS_ENABLED=true
CSP_ENABLED=true
DDOS_PROTECTION_ENABLED=true
RATE_LIMIT_ENABLED=true
SECURE_HEADERS_ENABLED=true

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

# Grafana
GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASSWORD
GRAFANA_ADMIN_USER=admin

# Backup configuration
BACKUP_PASSWORD=$BACKUP_PASSWORD
BACKUP_REPOSITORY=s3:freeagentics-backup

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS settings (adjust for production domains)
CORS_ORIGINS=["https://your-domain.com","https://www.your-domain.com"]
CORS_ALLOW_CREDENTIALS=true

# =============================================================================
# EMAIL CONFIGURATION (Optional)
# =============================================================================

# SMTP settings for notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@company.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@company.com

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================

# LLM Providers (add your API keys)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Other services
SENTRY_DSN=your-sentry-dsn

EOF

# Set secure permissions
chmod 600 "$ENV_FILE"

log_info "Production environment file generated successfully!"
log_info "File: $ENV_FILE"
log_info "Permissions: 600 (owner read/write only)"

log_warn "IMPORTANT: Update the following values for your production environment:"
echo "  • CORS_ORIGINS: Add your actual production domains"
echo "  • SMTP_* settings: Configure email notifications"
echo "  • OPENAI_API_KEY: Add your OpenAI API key"
echo "  • ANTHROPIC_API_KEY: Add your Anthropic API key"
echo "  • BACKUP_REPOSITORY: Configure your backup storage"
echo "  • SENTRY_DSN: Add your Sentry DSN for error tracking"

log_info "Generated passwords saved to environment file."
log_info "Run 'python scripts/validate_security_config.py' to verify configuration."