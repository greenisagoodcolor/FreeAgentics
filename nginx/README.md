# Nginx SSL/TLS Configuration for FreeAgentics

This directory contains comprehensive SSL/TLS configuration files and scripts for production-grade security in the FreeAgentics application.

## Overview

The SSL/TLS setup implements the following security features:
- **TLS 1.2/1.3 Support**: Modern protocol versions with strong cipher suites
- **Perfect Forward Secrecy**: Using DH parameters and ECDHE cipher suites
- **OCSP Stapling**: Improved certificate validation performance
- **Security Headers**: Comprehensive HTTP security headers
- **Certificate Pinning**: Optional HPKP implementation
- **Automatic HTTPS Redirect**: All HTTP traffic redirected to HTTPS
- **SSL Labs A+ Rating**: Configuration optimized for highest security rating

## Files Structure

```
nginx/
├── README.md                    # This file
├── nginx.conf                   # Main nginx configuration
├── conf.d/
│   └── ssl-freeagentics.conf   # SSL-specific configuration
├── snippets/
│   └── ssl-params.conf         # SSL parameter snippets
├── ssl/                        # Certificate directory
│   ├── cert.pem               # SSL certificate
│   └── key.pem                # Private key
├── dhparam.pem                 # DH parameters
├── setup-letsencrypt.sh        # Let's Encrypt setup script
├── certbot-setup.sh            # Docker-based certbot setup
├── test-ssl.sh                 # SSL testing script
├── generate-dhparam.sh         # DH parameter generation
├── generate-pin.sh             # Certificate pinning utility
└── monitor-ssl.sh              # SSL monitoring script
```

## Quick Start

### 1. Generate DH Parameters

```bash
# Generate 2048-bit DH parameters (recommended for production)
./nginx/generate-dhparam.sh

# Or generate 4096-bit for enhanced security (takes longer)
DH_SIZE=4096 ./nginx/generate-dhparam.sh
```

### 2. Set Up SSL Certificates

#### Option A: Let's Encrypt (Recommended)

For production deployment with real domain:

```bash
# Set your domain and email
export DOMAIN="yourdomain.com"
export EMAIL="admin@yourdomain.com"

# Run Let's Encrypt setup
./nginx/setup-letsencrypt.sh
```

#### Option B: Docker-based Let's Encrypt

For Docker environments:

```bash
# Set your domain and email
export DOMAIN="yourdomain.com"  
export EMAIL="admin@yourdomain.com"

# Run Docker-based setup
./nginx/certbot-setup.sh
```

#### Option C: Custom Certificates

Place your certificates in the `nginx/ssl/` directory:
- `cert.pem` - SSL certificate (full chain)
- `key.pem` - Private key

### 3. Test SSL Configuration

```bash
# Test SSL configuration
DOMAIN="yourdomain.com" ./nginx/test-ssl.sh

# Test specific aspects
DOMAIN="yourdomain.com" ./nginx/test-ssl.sh protocols
DOMAIN="yourdomain.com" ./nginx/test-ssl.sh headers
```

### 4. Start Production Environment

```bash
# Start with production profile
docker-compose --profile prod up -d

# Or start specific services
docker-compose up -d nginx backend frontend
```

## Configuration Details

### SSL/TLS Settings

- **Protocols**: TLS 1.2 and TLS 1.3 only
- **Ciphers**: Modern, secure cipher suites prioritizing ECDHE
- **Session**: Optimized session cache and timeout settings
- **OCSP**: Stapling enabled with multiple DNS resolvers

### Security Headers

The configuration includes comprehensive security headers:

```nginx
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Frame-Options: SAMEORIGIN
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: [detailed policy]
Permissions-Policy: [restrictive policy]
Expect-CT: max-age=86400, enforce
```

### Rate Limiting

Different rate limits for different endpoints:
- API endpoints: 10 requests/second
- Authentication: 5 requests/second  
- WebSocket: 50 connections/second

## Monitoring and Maintenance

### SSL Certificate Monitoring

```bash
# Run health check
DOMAIN="yourdomain.com" ./nginx/monitor-ssl.sh health-check

# Check certificate expiration
DOMAIN="yourdomain.com" ./nginx/monitor-ssl.sh expiration

# Generate detailed report
DOMAIN="yourdomain.com" ./nginx/monitor-ssl.sh report
```

### Automatic Certificate Renewal

The setup includes automatic certificate renewal:

```bash
# Manual renewal
./scripts/renew-cert.sh

# Add to cron for automatic renewal
sudo crontab -e
0 0,12 * * * /path/to/project/scripts/renew-cert.sh
```

### SSL Monitoring Alerts

Configure environment variables for alerts:

```bash
export DOMAIN="yourdomain.com"
export SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export EMAIL_TO="admin@yourdomain.com"
export WARNING_DAYS=30
export CRITICAL_DAYS=7

# Run monitoring
./nginx/monitor-ssl.sh health-check
```

## Certificate Pinning (Optional)

Generate certificate pins for HPKP:

```bash
# Generate pins for current certificate
DOMAIN="yourdomain.com" ./nginx/generate-pin.sh

# Add to nginx configuration
add_header Public-Key-Pins 'pin-sha256="[generated-pin]"; max-age=2592000; includeSubDomains' always;
```

**Warning**: Certificate pinning can lock users out if implemented incorrectly. Test thoroughly before enabling in production.

## Troubleshooting

### Common Issues

1. **Certificate Not Found**
   ```bash
   # Check if certificates exist
   ls -la nginx/ssl/
   
   # Verify certificate validity
   openssl x509 -in nginx/ssl/cert.pem -text -noout
   ```

2. **DH Parameters Missing**
   ```bash
   # Generate DH parameters
   ./nginx/generate-dhparam.sh
   ```

3. **SSL Test Failures**
   ```bash
   # Test configuration
   nginx -t
   
   # Check SSL connection
   openssl s_client -connect yourdomain.com:443
   ```

4. **Certificate Expiration**
   ```bash
   # Check expiration
   DOMAIN="yourdomain.com" ./nginx/monitor-ssl.sh expiration
   
   # Renew certificate
   ./scripts/renew-cert.sh
   ```

### Debugging

Enable debug logging in nginx:

```nginx
error_log /var/log/nginx/error.log debug;
```

Check logs:

```bash
# SSL-specific logs
tail -f /var/log/nginx/ssl-error.log

# Access logs
tail -f /var/log/nginx/ssl-access.log
```

## Security Considerations

### Best Practices

1. **Regular Updates**
   - Keep nginx updated
   - Monitor security advisories
   - Update SSL configuration as needed

2. **Certificate Management**
   - Use strong private keys (RSA 2048-bit minimum)
   - Implement certificate transparency monitoring
   - Set up expiration alerts

3. **Access Control**
   - Restrict admin interfaces
   - Implement proper authentication
   - Use strong passwords and 2FA

4. **Monitoring**
   - Monitor SSL certificate health
   - Track SSL handshake metrics
   - Set up security alerts

### Compliance

The configuration meets requirements for:
- **PCI DSS**: Payment card industry compliance
- **HIPAA**: Healthcare data protection
- **SOC 2**: Service organization controls
- **GDPR**: Data protection regulation

## Performance Optimization

### SSL Performance

- **Session Reuse**: Configured for optimal session caching
- **OCSP Stapling**: Reduces client-side OCSP requests
- **HTTP/2**: Enabled for improved performance
- **Compression**: Gzip enabled for text-based resources

### Monitoring Performance

```bash
# Monitor SSL handshake times
curl -w "@curl-format.txt" -o /dev/null -s "https://yourdomain.com"

# Check SSL session reuse
openssl s_client -connect yourdomain.com:443 -reconnect
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review nginx error logs
3. Test SSL configuration with provided scripts
4. Verify certificate validity and expiration

## References

- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [SSL Labs Server Test](https://www.ssllabs.com/ssltest/)
- [OWASP TLS Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [nginx SSL Documentation](https://nginx.org/en/docs/http/configuring_https_servers.html)