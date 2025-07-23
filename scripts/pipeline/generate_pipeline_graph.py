#!/usr/bin/env python3
"""
Pipeline Graph Generator for PIPELINE-ARCHITECT
Creates visual representations of the unified pipeline
"""

import argparse
import json
from pathlib import Path


# Color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class PipelineGraphGenerator:
    """Generates visual pipeline graphs in various formats."""

    def __init__(self):
        self.pipeline_stages = [
            {
                "id": "commit",
                "name": "💻 Commit",
                "type": "trigger",
                "description": "Developer commits code",
            },
            {
                "id": "pre_flight",
                "name": "🔍 Pre-flight Checks",
                "type": "validation",
                "description": "Fast feedback - code quality and security validation",
                "timeout": "5 minutes",
                "substages": [
                    "🎯 Code Quality Gate",
                    "🔐 Secret Scanning",
                    "🛡️ Dependency Security",
                ],
            },
            {
                "id": "build",
                "name": "🏗️ Build & Package",
                "type": "build",
                "description": "Artifact creation with multi-arch support",
                "timeout": "15 minutes",
                "substages": [
                    "🏗️ Backend Build",
                    "🎨 Frontend Build",
                    "📦 Multi-arch Images",
                ],
            },
            {
                "id": "test",
                "name": "🧪 Comprehensive Test Suite",
                "type": "testing",
                "description": "Multi-layered testing with parallel execution",
                "timeout": "20 minutes",
                "substages": [
                    "🧪 Unit Tests",
                    "🔗 Integration Tests",
                    "🎨 Frontend Tests",
                ],
            },
            {
                "id": "security",
                "name": "🔒 Security Validation",
                "type": "security",
                "description": "Comprehensive security testing with zero-tolerance",
                "timeout": "15 minutes",
                "substages": [
                    "🔒 SAST Analysis",
                    "🐳 Container Security",
                    "📋 Compliance Check",
                ],
            },
            {
                "id": "performance",
                "name": "⚡ Performance Verification",
                "type": "performance",
                "description": "Performance testing with regression detection",
                "timeout": "25 minutes",
                "substages": [
                    "⚡ Benchmarks",
                    "📊 Regression Analysis",
                    "🎯 Baseline Comparison",
                ],
            },
            {
                "id": "e2e",
                "name": "🌐 End-to-End Tests",
                "type": "integration",
                "description": "Full system integration testing",
                "timeout": "30 minutes",
                "substages": ["🌐 E2E Scenarios", "🧪 Smoke Tests", "🏥 Health Checks"],
            },
            {
                "id": "deployment_readiness",
                "name": "🚀 Deployment Readiness",
                "type": "gate",
                "description": "Final validation before deployment",
                "timeout": "8 minutes",
            },
            {
                "id": "deploy_staging",
                "name": "🎭 Staging Deploy",
                "type": "deployment",
                "description": "Deploy to staging environment",
                "timeout": "15 minutes",
            },
            {
                "id": "deploy_production",
                "name": "🚀 Production Deploy",
                "type": "deployment",
                "description": "Blue-green deployment to production",
                "timeout": "20 minutes",
            },
        ]

    def generate_mermaid_graph(self, detailed: bool = False) -> str:
        """Generate Mermaid flowchart of the pipeline."""

        if detailed:
            return self._generate_detailed_mermaid()
        else:
            return self._generate_simple_mermaid()

    def _generate_simple_mermaid(self) -> str:
        """Generate simple Mermaid flowchart."""

        mermaid = """```mermaid
graph TD
    A[💻 Commit] --> B[🔍 Pre-flight Checks]
    B --> C[🏗️ Build & Package]
    C --> D[🧪 Test Suite]
    D --> E[🔒 Security Validation]
    E --> F[⚡ Performance Tests]
    F --> G[🌐 E2E Tests]
    G --> H[🚀 Deployment Readiness]
    H --> I{Ready?}
    I -->|✅ Yes| J[🎭 Staging Deploy]
    I -->|❌ No| K[🛑 Pipeline Failed]
    J --> L[🧪 Staging Tests]
    L --> M{Production Branch?}
    M -->|✅ Main| N[🚀 Production Deploy]
    M -->|❌ Other| O[✅ Complete]
    N --> P[🏥 Health Checks]
    P --> Q[✅ Complete]
    K --> R[📧 Notify Team]

    classDef triggerClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef validationClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef buildClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef testingClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef securityClass fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef deploymentClass fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef gateClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef failureClass fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px
    classDef successClass fill:#c8e6c9,stroke:#388e3c,stroke-width:2px

    class A triggerClass
    class B validationClass
    class C buildClass
    class D,L testingClass
    class E securityClass
    class F testingClass
    class G testingClass
    class H,I,M gateClass
    class J,N deploymentClass
    class K,R failureClass
    class O,P,Q successClass
```"""
        return mermaid

    def _generate_detailed_mermaid(self) -> str:
        """Generate detailed Mermaid flowchart with substages."""

        mermaid = """```mermaid
graph TD
    A[💻 Commit] --> B[🔍 Pre-flight Checks]

    %% Pre-flight substages
    B --> B1[🎯 Code Quality Gate]
    B --> B2[🔐 Secret Scanning]
    B --> B3[🛡️ Dependency Security]
    B1 --> C[🏗️ Build & Package]
    B2 --> C
    B3 --> C

    %% Build substages
    C --> C1[🏗️ Backend Build]
    C --> C2[🎨 Frontend Build]
    C1 --> D[🧪 Test Suite]
    C2 --> D

    %% Test substages
    D --> D1[🧪 Unit Tests]
    D --> D2[🔗 Integration Tests]
    D --> D3[🎨 Frontend Tests]
    D1 --> E[🔒 Security Validation]
    D2 --> E
    D3 --> E

    %% Security substages
    E --> E1[🔒 SAST Analysis]
    E --> E2[🐳 Container Security]
    E --> E3[📋 Compliance Check]
    E1 --> F[⚡ Performance Tests]
    E2 --> F
    E3 --> F

    %% Performance substages
    F --> F1[⚡ Benchmarks]
    F --> F2[📊 Regression Analysis]
    F1 --> G[🌐 E2E Tests]
    F2 --> G

    %% E2E substages
    G --> G1[🌐 E2E Scenarios]
    G --> G2[🧪 Smoke Tests]
    G1 --> H[🚀 Deployment Readiness]
    G2 --> H

    %% Deployment flow
    H --> I{All Gates Passed?}
    I -->|✅ Yes| J[🎭 Staging Deploy]
    I -->|❌ No| K[🛑 Pipeline Failed]

    J --> J1[📦 Deploy Artifacts]
    J1 --> J2[🧪 Staging Smoke Tests]
    J2 --> M{Production Branch?}

    M -->|✅ Main Branch| N[🚀 Production Deploy]
    M -->|❌ Feature Branch| O[✅ Pipeline Complete]

    N --> N1[🔵 Blue Environment]
    N1 --> N2[🏥 Health Checks]
    N2 --> N3[🔄 Traffic Switch]
    N3 --> N4[🟢 Green Cleanup]
    N4 --> Q[✅ Production Complete]

    %% Failure handling
    K --> K1[📧 Notify Team]
    K1 --> K2[📊 Generate Report]
    K2 --> K3[🔄 Manual Review]

    %% Quality gates
    H --> H1[📊 Metrics Check]
    H --> H2[🔒 Security Score]
    H --> H3[📈 Performance Gate]
    H1 --> I
    H2 --> I
    H3 --> I

    classDef triggerClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef preflightClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef buildClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef testClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef securityClass fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef performanceClass fill:#e0f7fa,stroke:#006064,stroke-width:2px
    classDef e2eClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef deployClass fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef gateClass fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    classDef failureClass fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px
    classDef successClass fill:#c8e6c9,stroke:#388e3c,stroke-width:2px

    class A triggerClass
    class B,B1,B2,B3 preflightClass
    class C,C1,C2 buildClass
    class D,D1,D2,D3,J2 testClass
    class E,E1,E2,E3,H2 securityClass
    class F,F1,F2,H3 performanceClass
    class G,G1,G2 e2eClass
    class J,J1,N,N1,N2,N3,N4 deployClass
    class H,H1,I,M gateClass
    class K,K1,K2,K3 failureClass
    class O,Q successClass
```"""
        return mermaid

    def generate_ascii_graph(self) -> str:
        """Generate ASCII art representation of the pipeline."""

        ascii_graph = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        UNIFIED CI/CD PIPELINE ARCHITECTURE                   ║
║                     Martin Fowler + Jessica Kerr Principles                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

                                    💻 COMMIT
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        🔍 PRE-FLIGHT CHECKS                             │
    │                           (< 5 minutes)                                │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
    │  │  🎯 Code Quality │ │ 🔐 Secret Scan  │ │ 🛡️ Dependency Security │    │
    │  │     Gate        │ │                 │ │                        │    │
    │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        🏗️ BUILD & PACKAGE                               │
    │                          (< 15 minutes)                                │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
    │  │ 🏗️ Backend Build │ │ 🎨 Frontend     │ │ 📦 Multi-arch Images    │    │
    │  │                 │ │    Build        │ │                        │    │
    │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     🧪 COMPREHENSIVE TEST SUITE                         │
    │                          (< 20 minutes)                                │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
    │  │  🧪 Unit Tests  │ │ 🔗 Integration  │ │ 🎨 Frontend Tests       │    │
    │  │                 │ │    Tests        │ │                        │    │
    │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      🔒 SECURITY VALIDATION                             │
    │                          (< 15 minutes)                                │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
    │  │ 🔒 SAST Analysis│ │ 🐳 Container    │ │ 📋 Compliance Check     │    │
    │  │                 │ │    Security     │ │                        │    │
    │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ⚡ PERFORMANCE VERIFICATION                           │
    │                          (< 25 minutes)                                │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
    │  │ ⚡ Benchmarks   │ │ 📊 Regression   │ │ 🎯 Baseline Comparison  │    │
    │  │                 │ │    Analysis     │ │                        │    │
    │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       🌐 END-TO-END TESTS                              │
    │                          (< 30 minutes)                                │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
    │  │ 🌐 E2E Scenarios│ │ 🧪 Smoke Tests  │ │ 🏥 Health Checks        │    │
    │  │                 │ │                 │ │                        │    │
    │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                                ┌───────────────┐
                                │ 🚀 DEPLOYMENT │
                                │   READINESS   │
                                └───────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   ALL GATES     │ ──── ❌ ──► 🛑 PIPELINE FAILED
                              │    PASSED?      │
                              └─────────────────┘
                                        │ ✅
                                        ▼
                                ┌───────────────┐
                                │ 🎭 STAGING    │
                                │   DEPLOY      │
                                └───────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  MAIN BRANCH?   │ ──── ❌ ──► ✅ COMPLETE
                              │                 │
                              └─────────────────┘
                                        │ ✅
                                        ▼
                                ┌───────────────┐
                                │ 🚀 PRODUCTION │
                                │ BLUE-GREEN    │
                                │   DEPLOY      │
                                └───────────────┘
                                        │
                                        ▼
                                ┌───────────────┐
                                │ 🏥 HEALTH     │
                                │   CHECKS      │
                                └───────────────┘
                                        │
                                        ▼
                                  ✅ COMPLETE

═══════════════════════════════════════════════════════════════════════════════
                           ZERO-TOLERANCE QUALITY GATES
                             NO BYPASS MECHANISMS
═══════════════════════════════════════════════════════════════════════════════
        """
        return ascii_graph

    def generate_json_representation(self) -> str:
        """Generate JSON representation of the pipeline."""

        pipeline_json = {
            "pipeline": {
                "name": "Unified CI/CD Pipeline",
                "version": "1.0.0",
                "methodology": "Martin Fowler + Jessica Kerr",
                "principles": [
                    "Zero-Tolerance Quality Gates",
                    "No Bypass Mechanisms",
                    "Fast Feedback Loops",
                    "Progressive Quality Gates",
                    "Comprehensive Observability",
                ],
                "stages": [],
            }
        }

        for stage in self.pipeline_stages:
            stage_data = {
                "id": stage["id"],
                "name": stage["name"],
                "type": stage["type"],
                "description": stage["description"],
            }

            if "timeout" in stage:
                stage_data["timeout"] = stage["timeout"]

            if "substages" in stage:
                stage_data["substages"] = stage["substages"]

            pipeline_json["pipeline"]["stages"].append(stage_data)

        return json.dumps(pipeline_json, indent=2)

    def generate_dot_graph(self) -> str:
        """Generate DOT format graph for Graphviz."""

        dot = """digraph PipelineArchitecture {
    rankdir=TB;
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=8];

    // Define node styles
    node [shape=box, style=filled];

    // Trigger nodes
    commit [label="💻 Commit", fillcolor="#e1f5fe", shape=ellipse];

    // Validation nodes
    preflight [label="🔍 Pre-flight\\nChecks", fillcolor="#f3e5f5"];
    quality_gate [label="🎯 Code Quality\\nGate", fillcolor="#f3e5f5"];
    secret_scan [label="🔐 Secret\\nScanning", fillcolor="#f3e5f5"];
    dependency_sec [label="🛡️ Dependency\\nSecurity", fillcolor="#f3e5f5"];

    // Build nodes
    build [label="🏗️ Build &\\nPackage", fillcolor="#e8f5e8"];
    backend_build [label="🏗️ Backend\\nBuild", fillcolor="#e8f5e8"];
    frontend_build [label="🎨 Frontend\\nBuild", fillcolor="#e8f5e8"];

    // Test nodes
    test_suite [label="🧪 Test\\nSuite", fillcolor="#fff3e0"];
    unit_tests [label="🧪 Unit\\nTests", fillcolor="#fff3e0"];
    integration_tests [label="🔗 Integration\\nTests", fillcolor="#fff3e0"];
    frontend_tests [label="🎨 Frontend\\nTests", fillcolor="#fff3e0"];

    // Security nodes
    security [label="🔒 Security\\nValidation", fillcolor="#ffebee"];
    sast [label="🔒 SAST\\nAnalysis", fillcolor="#ffebee"];
    container_sec [label="🐳 Container\\nSecurity", fillcolor="#ffebee"];
    compliance [label="📋 Compliance\\nCheck", fillcolor="#ffebee"];

    // Performance nodes
    performance [label="⚡ Performance\\nTests", fillcolor="#e0f7fa"];

    // E2E nodes
    e2e [label="🌐 E2E\\nTests", fillcolor="#fce4ec"];

    // Deployment nodes
    readiness [label="🚀 Deployment\\nReadiness", fillcolor="#fff8e1", shape=diamond];
    staging [label="🎭 Staging\\nDeploy", fillcolor="#e0f2f1"];
    production [label="🚀 Production\\nDeploy", fillcolor="#e0f2f1"];

    // Decision nodes
    gates_passed [label="All Gates\\nPassed?", fillcolor="#fff8e1", shape=diamond];
    main_branch [label="Main\\nBranch?", fillcolor="#fff8e1", shape=diamond];

    // End nodes
    complete [label="✅ Complete", fillcolor="#c8e6c9", shape=ellipse];
    failed [label="🛑 Failed", fillcolor="#ffcdd2", shape=ellipse];

    // Define connections
    commit -> preflight;
    preflight -> quality_gate;
    preflight -> secret_scan;
    preflight -> dependency_sec;

    quality_gate -> build;
    secret_scan -> build;
    dependency_sec -> build;

    build -> backend_build;
    build -> frontend_build;

    backend_build -> test_suite;
    frontend_build -> test_suite;

    test_suite -> unit_tests;
    test_suite -> integration_tests;
    test_suite -> frontend_tests;

    unit_tests -> security;
    integration_tests -> security;
    frontend_tests -> security;

    security -> sast;
    security -> container_sec;
    security -> compliance;

    sast -> performance;
    container_sec -> performance;
    compliance -> performance;

    performance -> e2e;
    e2e -> readiness;
    readiness -> gates_passed;

    gates_passed -> staging [label="✅ Yes"];
    gates_passed -> failed [label="❌ No"];

    staging -> main_branch;
    main_branch -> production [label="✅ Main"];
    main_branch -> complete [label="❌ Other"];

    production -> complete;

    // Clustering
    subgraph cluster_preflight {
        label="Pre-flight Stage";
        color=purple;
        quality_gate; secret_scan; dependency_sec;
    }

    subgraph cluster_build {
        label="Build Stage";
        color=green;
        backend_build; frontend_build;
    }

    subgraph cluster_test {
        label="Test Stage";
        color=orange;
        unit_tests; integration_tests; frontend_tests;
    }

    subgraph cluster_security {
        label="Security Stage";
        color=red;
        sast; container_sec; compliance;
    }

    subgraph cluster_deploy {
        label="Deployment Stage";
        color=teal;
        staging; production;
    }
}"""
        return dot


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pipeline Graph Generator")
    parser.add_argument(
        "--format",
        choices=["mermaid", "mermaid-detailed", "ascii", "json", "dot"],
        default="mermaid",
        help="Output format",
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed graph")

    args = parser.parse_args()

    generator = PipelineGraphGenerator()

    # Generate graph based on format
    if args.format == "mermaid" or args.format == "mermaid-detailed":
        graph = generator.generate_mermaid_graph(
            detailed=args.detailed or args.format == "mermaid-detailed"
        )
    elif args.format == "ascii":
        graph = generator.generate_ascii_graph()
    elif args.format == "json":
        graph = generator.generate_json_representation()
    elif args.format == "dot":
        graph = generator.generate_dot_graph()
    else:
        print(f"{Colors.RED}❌ Unknown format: {args.format}{Colors.RESET}")
        return

    # Output graph
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(graph)
        print(f"{Colors.GREEN}✅ Graph saved to: {args.output}{Colors.RESET}")
    else:
        print(graph)


if __name__ == "__main__":
    main()
