# JWT Key Management

This directory should contain the JWT signing keys, but they must NOT be committed to version control.

## Key Generation

To generate new RSA keys for JWT signing:

```bash
# Generate private key
openssl genrsa -out jwt_private.pem 4096

# Extract public key
openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem

# Set proper permissions
chmod 600 jwt_private.pem
chmod 644 jwt_public.pem
```

## Production Deployment

In production environments:

1. Keys should be managed through a secure key management service (AWS KMS, HashiCorp Vault, etc.)
2. Use environment variables or secure configuration management
3. Implement key rotation policies (90 days recommended)
4. Never commit keys to version control

## Development

For local development, the application will auto-generate keys if they don't exist.