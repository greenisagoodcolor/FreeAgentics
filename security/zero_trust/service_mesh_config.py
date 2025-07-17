"""
Service Mesh Configuration for Zero-Trust Architecture.

This module provides configuration generation for service mesh platforms:
- Istio configuration with mTLS and traffic policies
- Linkerd configuration with service policies
- Generic service mesh abstractions
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ServiceMeshType(str, Enum):
    """Supported service mesh types."""

    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL = "consul"


class TLSMode(str, Enum):
    """TLS modes for service communication."""

    DISABLE = "DISABLE"
    SIMPLE = "SIMPLE"
    MUTUAL = "MUTUAL"
    ISTIO_MUTUAL = "ISTIO_MUTUAL"


@dataclass
class ServiceDefinition:
    """Service definition for mesh configuration."""

    name: str
    namespace: str = "default"
    port: int = 8080
    labels: Dict[str, str] = field(default_factory=dict)
    version: str = "v1"


@dataclass
class TrafficPolicy:
    """Traffic management policy."""

    name: str
    source_service: str
    destination_service: str
    tls_mode: str = TLSMode.ISTIO_MUTUAL
    load_balancer: str = "ROUND_ROBIN"
    connection_pool: Dict[str, Any] = field(default_factory=dict)
    outlier_detection: Dict[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3
    timeout_seconds: int = 30


@dataclass
class ServicePolicy:
    """Service-level access policy."""

    source_service: str
    target_service: str
    allowed_operations: List[str]
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    mtls_required: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptionPolicy:
    """Encryption policy for service traffic."""

    service: str
    min_tls_version: str = "1.3"
    cipher_suites: List[str] = field(default_factory=list)
    client_cert_required: bool = True
    verify_subject_alt_name: List[str] = field(default_factory=list)


class ServiceMeshConfig:
    """Service mesh configuration manager."""

    def __init__(self, mesh_type: ServiceMeshType = ServiceMeshType.ISTIO):
        self.mesh_type = mesh_type
        self.services: Dict[str, ServiceDefinition] = {}
        self.traffic_policies: List[TrafficPolicy] = []
        self.service_policies: List[ServicePolicy] = []
        self.encryption_policies: Dict[str, EncryptionPolicy] = {}
        self.routing_rules: Dict[str, Dict[str, Any]] = {}

    def add_service(
        self,
        name: str,
        namespace: str = "default",
        port: int = 8080,
        version: str = "v1",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a service to the mesh configuration."""
        service = ServiceDefinition(
            name=name,
            namespace=namespace,
            port=port,
            version=version,
            labels=labels or {},
        )
        self.services[name] = service
        logger.info(f"Added service {name} to mesh configuration")

    def add_traffic_policy(self, policy: TrafficPolicy) -> None:
        """Add a traffic management policy."""
        self.traffic_policies.append(policy)
        logger.info(
            f"Added traffic policy: {policy.source_service} -> {policy.destination_service}"
        )

    def add_service_policy(self, policy: ServicePolicy) -> None:
        """Add a service access policy."""
        self.service_policies.append(policy)
        logger.info(f"Added service policy: {policy.source_service} -> {policy.target_service}")

    def set_encryption_policy(
        self,
        service: str,
        min_tls_version: str = "1.3",
        cipher_suites: Optional[List[str]] = None,
        client_cert_required: bool = True,
    ) -> None:
        """Set encryption policy for a service."""
        policy = EncryptionPolicy(
            service=service,
            min_tls_version=min_tls_version,
            cipher_suites=cipher_suites
            or [
                "TLS_AES_256_GCM_SHA384",
                "TLS_AES_128_GCM_SHA256",
                "TLS_CHACHA20_POLY1305_SHA256",
            ],
            client_cert_required=client_cert_required,
        )
        self.encryption_policies[service] = policy
        logger.info(f"Set encryption policy for {service}")

    def add_routing_rule(
        self,
        service: str,
        version_weights: Dict[str, int],
        sticky_sessions: bool = False,
        load_balancer_type: str = "ROUND_ROBIN",
    ) -> None:
        """Add routing rule for service versions."""
        self.routing_rules[service] = {
            "version_weights": version_weights,
            "sticky_sessions": sticky_sessions,
            "load_balancer_type": load_balancer_type,
        }
        logger.info(f"Added routing rule for {service}: {version_weights}")

    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            "mesh_type": self.mesh_type.value,
            "services": {
                name: {
                    "namespace": svc.namespace,
                    "port": svc.port,
                    "version": svc.version,
                    "labels": svc.labels,
                }
                for name, svc in self.services.items()
            },
            "traffic_policies": [
                {
                    "name": tp.name,
                    "source": tp.source_service,
                    "destination": tp.destination_service,
                    "tls_mode": tp.tls_mode,
                    "load_balancer": tp.load_balancer,
                    "retry_attempts": tp.retry_attempts,
                    "timeout_seconds": tp.timeout_seconds,
                }
                for tp in self.traffic_policies
            ],
            "service_policies": [
                {
                    "source": sp.source_service,
                    "target": sp.target_service,
                    "allowed_operations": sp.allowed_operations,
                    "allowed_methods": sp.allowed_methods,
                    "mtls_required": sp.mtls_required,
                }
                for sp in self.service_policies
            ],
            "encryption_policies": {
                service: {
                    "min_tls_version": ep.min_tls_version,
                    "cipher_suites": ep.cipher_suites,
                    "client_cert_required": ep.client_cert_required,
                }
                for service, ep in self.encryption_policies.items()
            },
            "routing_rules": self.routing_rules,
        }

    def export_yaml(self, file_path: str) -> None:
        """Export configuration to YAML file."""
        config = self.get_config()

        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported configuration to {file_path}")


def generate_istio_config(config: ServiceMeshConfig) -> Dict[str, Any]:
    """Generate Istio-specific configuration."""
    istio_config = {
        "apiVersion": "networking.istio.io/v1beta1",
        "kind": "Configuration",
        "items": [],
    }

    # Generate VirtualService for each service
    for service_name, service in config.services.items():
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{service_name}-vs",
                "namespace": service.namespace,
            },
            "spec": {
                "hosts": [service_name],
                "http": [
                    {
                        "match": [{"uri": {"prefix": "/"}}],
                        "route": [
                            {
                                "destination": {
                                    "host": service_name,
                                    "port": {"number": service.port},
                                },
                            }
                        ],
                    }
                ],
            },
        }

        # Add routing rules if defined
        if service_name in config.routing_rules:
            routing = config.routing_rules[service_name]
            routes = []

            for version, weight in routing["version_weights"].items():
                routes.append(
                    {
                        "destination": {
                            "host": service_name,
                            "subset": version,
                        },
                        "weight": weight,
                    }
                )

            virtual_service["spec"]["http"][0]["route"] = routes

        istio_config["items"].append(virtual_service)

    # Generate DestinationRule for each service
    for service_name, service in config.services.items():
        destination_rule = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": f"{service_name}-dr",
                "namespace": service.namespace,
            },
            "spec": {
                "host": service_name,
                "trafficPolicy": {
                    "tls": {"mode": "ISTIO_MUTUAL"},
                },
            },
        }

        # Add traffic policies
        for policy in config.traffic_policies:
            if policy.destination_service == service_name:
                destination_rule["spec"]["trafficPolicy"].update(
                    {
                        "connectionPool": policy.connection_pool
                        or {
                            "tcp": {"maxConnections": 100},
                            "http": {
                                "http1MaxPendingRequests": 10,
                                "http2MaxRequests": 100,
                            },
                        },
                        "loadBalancer": {"simple": policy.load_balancer},
                        "outlierDetection": policy.outlier_detection
                        or {
                            "consecutiveErrors": 5,
                            "interval": "30s",
                            "baseEjectionTime": "30s",
                        },
                    }
                )

        # Add subsets for versions
        if service_name in config.routing_rules:
            subsets = []
            for version in config.routing_rules[service_name]["version_weights"]:
                subsets.append(
                    {
                        "name": version,
                        "labels": {"version": version},
                    }
                )
            destination_rule["spec"]["subsets"] = subsets

        istio_config["items"].append(destination_rule)

    # Generate PeerAuthentication for mTLS
    for service_name, service in config.services.items():
        peer_auth = {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "PeerAuthentication",
            "metadata": {
                "name": f"{service_name}-mtls",
                "namespace": service.namespace,
            },
            "spec": {
                "selector": {
                    "matchLabels": {"app": service_name},
                },
                "mtls": {"mode": "STRICT"},
            },
        }
        istio_config["items"].append(peer_auth)

    # Generate AuthorizationPolicy
    for policy in config.service_policies:
        auth_policy = {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"{policy.source_service}-to-{policy.target_service}",
                "namespace": "default",
            },
            "spec": {
                "selector": {
                    "matchLabels": {"app": policy.target_service},
                },
                "rules": [
                    {
                        "from": [
                            {
                                "source": {
                                    "principals": [
                                        f"cluster.local/ns/default/sa/{policy.source_service}"
                                    ],
                                },
                            }
                        ],
                        "to": [
                            {
                                "operation": {
                                    "methods": policy.allowed_methods,
                                },
                            }
                        ],
                    }
                ],
            },
        }
        istio_config["items"].append(auth_policy)

    return istio_config


def generate_linkerd_config(config: ServiceMeshConfig) -> Dict[str, Any]:
    """Generate Linkerd-specific configuration."""
    linkerd_config = {
        "apiVersion": "policy.linkerd.io/v1beta1",
        "kind": "Configuration",
        "items": [],
    }

    # Generate Server for each service
    for service_name, service in config.services.items():
        server = {
            "apiVersion": "policy.linkerd.io/v1beta1",
            "kind": "Server",
            "metadata": {
                "name": service_name,
                "namespace": service.namespace,
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"app": service_name},
                },
                "port": service.port,
                "proxyProtocol": "HTTP/2",
            },
        }
        linkerd_config["items"].append(server)

    # Generate ServerAuthorization
    for policy in config.service_policies:
        server_auth = {
            "apiVersion": "policy.linkerd.io/v1beta1",
            "kind": "ServerAuthorization",
            "metadata": {
                "name": f"{policy.source_service}-to-{policy.target_service}",
                "namespace": "default",
            },
            "spec": {
                "server": {
                    "name": policy.target_service,
                },
                "client": {
                    "meshTLS": {
                        "identities": [
                            f"{policy.source_service}.default.serviceaccount.identity.linkerd.cluster.local"
                        ],
                    },
                },
            },
        }
        linkerd_config["items"].append(server_auth)

    # Generate HTTPRoute for traffic management
    for service_name, service in config.services.items():
        if service_name in config.routing_rules:
            routing = config.routing_rules[service_name]

            http_route = {
                "apiVersion": "policy.linkerd.io/v1beta1",
                "kind": "HTTPRoute",
                "metadata": {
                    "name": f"{service_name}-route",
                    "namespace": service.namespace,
                },
                "spec": {
                    "parentRefs": [
                        {
                            "name": service_name,
                            "kind": "Server",
                        }
                    ],
                    "rules": [],
                },
            }

            # Add weighted routing
            for version, weight in routing["version_weights"].items():
                rule = {
                    "matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                    "backendRefs": [
                        {
                            "name": f"{service_name}-{version}",
                            "port": service.port,
                            "weight": weight,
                        }
                    ],
                }
                http_route["spec"]["rules"].append(rule)

            linkerd_config["items"].append(http_route)

    return linkerd_config


def create_example_configuration() -> ServiceMeshConfig:
    """Create example service mesh configuration."""
    config = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO)

    # Add services
    config.add_service("frontend", namespace="production", port=80)
    config.add_service("api-gateway", namespace="production", port=8080)
    config.add_service("auth-service", namespace="production", port=8443)
    config.add_service("data-service", namespace="production", port=9090)

    # Add traffic policies
    config.add_traffic_policy(
        TrafficPolicy(
            name="frontend-to-api",
            source_service="frontend",
            destination_service="api-gateway",
            tls_mode=TLSMode.ISTIO_MUTUAL,
            retry_attempts=3,
            timeout_seconds=30,
        )
    )

    config.add_traffic_policy(
        TrafficPolicy(
            name="api-to-auth",
            source_service="api-gateway",
            destination_service="auth-service",
            tls_mode=TLSMode.ISTIO_MUTUAL,
            retry_attempts=2,
            timeout_seconds=10,
        )
    )

    # Add service policies
    config.add_service_policy(
        ServicePolicy(
            source_service="frontend",
            target_service="api-gateway",
            allowed_operations=["read", "write"],
            allowed_methods=["GET", "POST"],
        )
    )

    config.add_service_policy(
        ServicePolicy(
            source_service="api-gateway",
            target_service="auth-service",
            allowed_operations=["authenticate", "validate"],
            allowed_methods=["POST"],
        )
    )

    # Set encryption policies
    config.set_encryption_policy(
        service="auth-service",
        min_tls_version="1.3",
        cipher_suites=[
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
        ],
    )

    # Add canary deployment routing
    config.add_routing_rule(
        service="api-gateway",
        version_weights={"v1": 90, "v2": 10},
        sticky_sessions=True,
    )

    return config


if __name__ == "__main__":
    # Create example configuration
    config = create_example_configuration()

    # Export to YAML
    config.export_yaml("service-mesh-config.yaml")

    # Generate Istio configuration
    istio_config = generate_istio_config(config)
    with open("istio-config.yaml", "w") as f:
        yaml.dump(istio_config, f, default_flow_style=False)

    # Generate Linkerd configuration
    linkerd_config = generate_linkerd_config(config)
    with open("linkerd-config.yaml", "w") as f:
        yaml.dump(linkerd_config, f, default_flow_style=False)
