"""
Zero-Trust Network Architecture Components.

This package provides comprehensive zero-trust security implementation including:
- Mutual TLS (mTLS) certificate management
- Identity-aware proxy for request validation
- Service mesh configuration and deployment
- Continuous verification and risk scoring
"""

from .identity_proxy import (
    IdentityAwareProxy,
    ProxyConfig,
    RequestContext,
    ServicePolicy,
    SessionRiskScore,
)
from .mtls_manager import (
    CertificateInfo,
    CertificateRotationPolicy,
    MTLSManager,
    RotationStrategy,
)
from .service_mesh_config import ServiceMeshConfig, ServiceMeshType
from .service_mesh_config import ServicePolicy as MeshServicePolicy
from .service_mesh_config import (
    TrafficPolicy,
    generate_istio_config,
    generate_linkerd_config,
)

__all__ = [
    # mTLS Manager
    "MTLSManager",
    "CertificateInfo",
    "CertificateRotationPolicy",
    "RotationStrategy",
    # Identity Proxy
    "IdentityAwareProxy",
    "ProxyConfig",
    "RequestContext",
    "ServicePolicy",
    "SessionRiskScore",
    # Service Mesh
    "ServiceMeshConfig",
    "ServiceMeshType",
    "MeshServicePolicy",
    "TrafficPolicy",
    "generate_istio_config",
    "generate_linkerd_config",
]

# Version information
__version__ = "1.0.0"
__author__ = "Zero Trust Security Team"
