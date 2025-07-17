# Zero-Trust Network Architecture

A comprehensive implementation of zero-trust security principles for microservices architecture.

## Quick Start

```python
from security.zero_trust import MTLSManager, IdentityAwareProxy, ProxyConfig

# Initialize mTLS manager
mtls_manager = MTLSManager(enable_key_cache=True)

# Generate service certificate
cert_info = mtls_manager.generate_service_certificate("my-service")

# Configure identity-aware proxy
config = ProxyConfig(enable_mtls=True, enable_risk_scoring=True)
proxy = IdentityAwareProxy(config)
```

## Features

### ✅ Mutual TLS (mTLS) Management
- Automatic certificate generation and rotation
- Certificate revocation and validation
- Performance optimized with key caching
- Secure certificate storage

### ✅ Identity-Aware Proxy
- Request validation at every hop
- Dynamic permission evaluation
- Session risk scoring
- Continuous verification

### ✅ Service Mesh Integration
- Istio configuration generation
- Linkerd policy management
- Traffic encryption policies
- Canary deployment support

### ✅ Performance Optimized
- <10ms certificate generation (with caching)
- <10ms request validation overhead
- Efficient session management
- Background key generation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Service A     │    │   Service B     │    │   Service C     │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │   mTLS    │  │    │  │   mTLS    │  │    │  │   mTLS    │  │
│  │Certificate│  │    │  │Certificate│  │    │  │Certificate│  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Identity-Aware  │
                    │     Proxy       │
                    │                 │
                    │ • mTLS Validate │
                    │ • Risk Scoring  │
                    │ • Policy Check  │
                    │ • Continuous    │
                    │   Verification  │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Service Mesh    │
                    │ Configuration   │
                    │                 │
                    │ • Traffic Rules │
                    │ • Encryption    │
                    │ • Load Balance  │
                    │ • Canary Deploy │
                    └─────────────────┘
```

## Security Principles

### 🔐 Never Trust, Always Verify
- Every request is authenticated with mTLS
- Certificate validation on every hop
- Continuous session verification

### 🛡️ Least Privilege Access
- Fine-grained service permissions
- Operation-level access control
- Time-based restrictions

### 🎯 Assume Breach
- Risk-based access decisions
- Automatic session termination
- Comprehensive audit logging

## Performance Metrics

- **Certificate Generation**: <10ms average (with caching)
- **Request Validation**: <10ms latency overhead
- **Risk Scoring**: <5ms calculation time
- **Config Generation**: <10ms for complete mesh

## Example Usage

### Certificate Management
```python
# Generate certificate with custom DNS names
cert_info = mtls_manager.generate_service_certificate(
    service_name="my-service",
    dns_names=["my-service.local", "my-service.cluster.local"],
    validity_days=30
)

# Set rotation policy
policy = CertificateRotationPolicy(
    strategy=RotationStrategy.TIME_BASED,
    rotation_interval_days=30,
    auto_rotate=True
)
mtls_manager.set_rotation_policy("my-service", policy)
```

### Service Policies
```python
# Add service access policy
proxy.add_service_policy(ServicePolicy(
    source_service="frontend",
    target_service="backend",
    allowed_operations=["read", "write"],
    conditions={
        "time_window": {"start": "08:00", "end": "18:00"},
        "max_requests_per_minute": 100
    },
    mtls_required=True
))
```

### Service Mesh Configuration
```python
# Configure service mesh
mesh_config = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO)
mesh_config.add_service("frontend", namespace="prod", port=80)
mesh_config.add_service("backend", namespace="prod", port=8080)

# Add traffic policy
mesh_config.add_traffic_policy(TrafficPolicy(
    name="frontend-to-backend",
    source_service="frontend",
    destination_service="backend",
    tls_mode="ISTIO_MUTUAL",
    retry_attempts=3
))

# Generate Istio configuration
istio_config = generate_istio_config(mesh_config)
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/security/test_zero_trust_architecture.py -v
```

Or run the example demo:

```bash
python security/zero_trust/example_usage.py
```

## Configuration Files

- `service_mesh_config.yaml`: Example service mesh configuration
- `mtls_manager.py`: Certificate management implementation
- `identity_proxy.py`: Identity-aware proxy implementation
- `service_mesh_config.py`: Service mesh configuration generator

## Production Deployment

### Prerequisites
- Redis for session management
- Certificate storage with backup
- Service mesh platform (Istio/Linkerd)
- Monitoring and alerting

### Configuration
1. Set up certificate storage paths
2. Configure service mesh integration
3. Set up monitoring and alerting
4. Configure backup and recovery

### Maintenance
- Regular certificate rotation
- Policy reviews and updates
- Performance monitoring
- Security assessments

## Compliance

- **NIST SP 800-207**: Zero Trust Architecture
- **RFC 8446**: TLS 1.3
- **RFC 5280**: X.509 certificates
- **OWASP**: Security best practices

## Support

For issues and questions:
- Check the comprehensive test suite
- Review the implementation summary
- Run the example usage script
- Consult the API documentation