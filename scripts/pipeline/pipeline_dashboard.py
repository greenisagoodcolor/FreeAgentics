#!/usr/bin/env python3
"""
Pipeline Dashboard Generator for PIPELINE-ARCHITECT
Creates visual pipeline status dashboards and metrics
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(message: str, color: str = Colors.BLUE):
    """Print a formatted header."""
    print(f"\n{color}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{message}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def print_status(status: str) -> str:
    """Return formatted status with emoji and color."""
    status_map = {
        "success": f"{Colors.GREEN}✅ SUCCESS{Colors.RESET}",
        "failure": f"{Colors.RED}❌ FAILURE{Colors.RESET}",
        "skipped": f"{Colors.YELLOW}⏭️ SKIPPED{Colors.RESET}",
        "in_progress": f"{Colors.BLUE}🔄 IN PROGRESS{Colors.RESET}",
        "pending": f"{Colors.PURPLE}⏳ PENDING{Colors.RESET}",
        "cancelled": f"{Colors.RED}🚫 CANCELLED{Colors.RESET}",
        "completed": f"{Colors.GREEN}✅ COMPLETED{Colors.RESET}",
    }
    return status_map.get(status.lower(), f"{Colors.WHITE}{status.upper()}{Colors.RESET}")


class PipelineDashboard:
    """Pipeline dashboard generator and metrics collector."""

    def __init__(self):
        self.pipeline_data = {}
        self.metrics = {
            "total_pipelines": 0,
            "success_rate": 0.0,
            "average_duration": 0.0,
            "failure_trends": [],
            "performance_trends": [],
        }

    def load_pipeline_data(self, data_path: str) -> bool:
        """Load pipeline data from JSON file or GitHub Actions."""
        try:
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    self.pipeline_data = json.load(f)
                return True
            else:
                print(f"{Colors.YELLOW}⚠️ Pipeline data file not found: {data_path}{Colors.RESET}")
                return False
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading pipeline data: {e}{Colors.RESET}")
            return False

    def generate_terminal_dashboard(self) -> str:
        """Generate a terminal-based dashboard."""
        dashboard = []

        # Header
        dashboard.append(f"{Colors.CYAN}{Colors.BOLD}")
        dashboard.append("🚀 PIPELINE-ARCHITECT DASHBOARD")
        dashboard.append("=" * 50)
        dashboard.append(f"{Colors.RESET}")

        if not self.pipeline_data:
            dashboard.append(f"{Colors.YELLOW}⚠️ No pipeline data available{Colors.RESET}")
            return "\n".join(dashboard)

        # Pipeline Overview
        pipeline_id = self.pipeline_data.get("pipeline_id", "Unknown")
        status = self.pipeline_data.get("status", "unknown")
        commit_sha = self.pipeline_data.get("commit_sha", "Unknown")[:8]
        branch = self.pipeline_data.get("branch", "Unknown")

        dashboard.append(f"📊 {Colors.BOLD}Pipeline Overview{Colors.RESET}")
        dashboard.append(f"   ID: {Colors.CYAN}{pipeline_id}{Colors.RESET}")
        dashboard.append(f"   Status: {print_status(status)}")
        dashboard.append(f"   Commit: {Colors.CYAN}{commit_sha}{Colors.RESET}")
        dashboard.append(f"   Branch: {Colors.CYAN}{branch}{Colors.RESET}")
        dashboard.append("")

        # Stage Status
        stages = self.pipeline_data.get("stages", {})
        dashboard.append(f"🏗️ {Colors.BOLD}Stage Status{Colors.RESET}")

        stage_info = [
            ("🔍 Pre-flight", "pre_flight"),
            ("🏗️ Build", "build"),
            ("🧪 Tests", "tests"),
            ("🔒 Security", "security"),
            ("⚡ Performance", "performance"),
            ("🌐 E2E", "e2e"),
            ("🎭 Staging", "staging_deploy"),
            ("🚀 Production", "production_deploy"),
        ]

        for stage_name, stage_key in stage_info:
            stage_status = stages.get(stage_key, "unknown")
            dashboard.append(f"   {stage_name}: {print_status(stage_status)}")

        dashboard.append("")

        # Quality Metrics
        dashboard.append(f"📋 {Colors.BOLD}Quality Metrics{Colors.RESET}")
        change_scope = self.pipeline_data.get("change_scope", "unknown")
        security_sensitive = self.pipeline_data.get("security_sensitive", False)
        deployment_ready = self.pipeline_data.get("deployment_ready", False)

        dashboard.append(f"   Change Scope: {Colors.CYAN}{change_scope}{Colors.RESET}")
        dashboard.append(
            f"   Security Sensitive: {Colors.RED if security_sensitive else Colors.GREEN}{'Yes' if security_sensitive else 'No'}{Colors.RESET}"
        )
        dashboard.append(
            f"   Deployment Ready: {Colors.GREEN if deployment_ready else Colors.RED}{'Yes' if deployment_ready else 'No'}{Colors.RESET}"
        )
        dashboard.append("")

        # Timing Information
        start_time = self.pipeline_data.get("start_time")
        end_time = self.pipeline_data.get("end_time")

        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            duration = end_dt - start_dt

            dashboard.append(f"⏱️ {Colors.BOLD}Timing{Colors.RESET}")
            dashboard.append(
                f"   Started: {Colors.CYAN}{start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}{Colors.RESET}"
            )
            dashboard.append(f"   Duration: {Colors.CYAN}{duration}{Colors.RESET}")
            dashboard.append("")

        return "\n".join(dashboard)

    def generate_html_dashboard(self) -> str:
        """Generate an HTML dashboard."""
        if not self.pipeline_data:
            return "<html><body><h1>No pipeline data available</h1></body></html>"

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pipeline Dashboard - {self.pipeline_data.get("pipeline_id", "Unknown")}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header .pipeline-id {{
                    font-size: 1.2em;
                    opacity: 0.9;
                    margin-top: 10px;
                }}
                .content {{
                    padding: 30px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .status-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .status-card {{
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    border-left: 4px solid #3498db;
                }}
                .status-success {{ border-left-color: #27ae60; }}
                .status-failure {{ border-left-color: #e74c3c; }}
                .status-warning {{ border-left-color: #f39c12; }}
                .status-info {{ border-left-color: #3498db; }}
                .stage-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .stage-table th,
                .stage-table td {{
                    padding: 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .stage-table th {{
                    background: #34495e;
                    color: white;
                    font-weight: 600;
                }}
                .stage-table tr:hover {{
                    background: #f5f5f5;
                }}
                .status-badge {{
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: 600;
                    text-transform: uppercase;
                }}
                .status-success {{ background: #d4edda; color: #155724; }}
                .status-failure {{ background: #f8d7da; color: #721c24; }}
                .status-skipped {{ background: #fff3cd; color: #856404; }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .footer {{
                    background: #ecf0f1;
                    padding: 20px;
                    text-align: center;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Pipeline Dashboard</h1>
                    <div class="pipeline-id">Pipeline ID: {self.pipeline_data.get("pipeline_id", "Unknown")}</div>
                </div>

                <div class="content">
                    <div class="section">
                        <h2>📊 Overview</h2>
                        <div class="status-grid">
                            <div class="status-card status-{"success" if self.pipeline_data.get("deployment_ready") else "failure"}">
                                <h3>Pipeline Status</h3>
                                <p><strong>{"✅ Ready for Deployment" if self.pipeline_data.get("deployment_ready") else "❌ Not Ready"}</strong></p>
                                <p>Commit: {self.pipeline_data.get("commit_sha", "Unknown")[:8]}</p>
                                <p>Branch: {self.pipeline_data.get("branch", "Unknown")}</p>
                            </div>
                            <div class="status-card status-info">
                                <h3>Change Analysis</h3>
                                <p><strong>Scope:</strong> {self.pipeline_data.get("change_scope", "Unknown")}</p>
                                <p><strong>Security Sensitive:</strong> {"Yes" if self.pipeline_data.get("security_sensitive") else "No"}</p>
                                <p><strong>Actor:</strong> {self.pipeline_data.get("actor", "Unknown")}</p>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>🏗️ Stage Results</h2>
                        <table class="stage-table">
                            <thead>
                                <tr>
                                    <th>Stage</th>
                                    <th>Status</th>
                                    <th>Description</th>
                                </tr>
                            </thead>
                            <tbody>
        """

        # Add stage rows
        stages = self.pipeline_data.get("stages", {})
        stage_info = [
            (
                "🔍 Pre-flight Checks",
                "pre_flight",
                "Code quality and security validation",
            ),
            ("🏗️ Build & Package", "build", "Artifact creation and containerization"),
            ("🧪 Test Suite", "tests", "Unit and integration testing"),
            ("🔒 Security Validation", "security", "Security scanning and compliance"),
            ("⚡ Performance Tests", "performance", "Performance benchmarks"),
            ("🌐 E2E Tests", "e2e", "End-to-end system validation"),
            ("🎭 Staging Deploy", "staging_deploy", "Staging environment deployment"),
            (
                "🚀 Production Deploy",
                "production_deploy",
                "Production environment deployment",
            ),
        ]

        for stage_name, stage_key, stage_desc in stage_info:
            status = stages.get(stage_key, "unknown")
            status_class = (
                "status-success"
                if status == "success"
                else "status-failure"
                if status == "failure"
                else "status-skipped"
            )
            status_text = (
                "✅ Success"
                if status == "success"
                else "❌ Failed"
                if status == "failure"
                else "⏭️ Skipped"
            )

            html += f"""
                                <tr>
                                    <td>{stage_name}</td>
                                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                                    <td>{stage_desc}</td>
                                </tr>
            """

        html += f"""
                            </tbody>
                        </table>
                    </div>

                    <div class="section">
                        <h2>📈 Metrics</h2>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-value">{len([s for s in stages.values() if s == "success"])}</div>
                                <div class="metric-label">Successful Stages</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{len([s for s in stages.values() if s == "failure"])}</div>
                                <div class="metric-label">Failed Stages</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{len([s for s in stages.values() if s == "skipped"])}</div>
                                <div class="metric-label">Skipped Stages</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{self.pipeline_data.get("trigger", "Unknown")}</div>
                                <div class="metric-label">Trigger Event</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="footer">
                    <p>Generated by PIPELINE-ARCHITECT • {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
                    <p>Martin Fowler + Jessica Kerr Principles • Zero-Tolerance Quality Gates</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def generate_mermaid_pipeline_graph(self) -> str:
        """Generate a Mermaid diagram showing pipeline flow."""
        mermaid = """
        ```mermaid
        graph TD
            A[💻 Commit] --> B[🔍 Pre-flight Checks]
            B --> B1[🎯 Code Quality]
            B --> B2[🔐 Secret Scan]
            B --> B3[🛡️ Dependency Check]
            B1 --> C[🏗️ Build & Package]
            B2 --> C
            B3 --> C
            C --> C1[🏗️ Backend Build]
            C --> C2[🎨 Frontend Build]
            C1 --> D[🧪 Test Suite]
            C2 --> D
            D --> D1[🧪 Unit Tests]
            D --> D2[🔗 Integration Tests]
            D --> D3[🎨 Frontend Tests]
            D1 --> E[🔒 Security Validation]
            D2 --> E
            D3 --> E
            E --> E1[🔒 SAST]
            E --> E2[🐳 Container Scan]
            E --> E3[📋 Compliance]
            E1 --> F[⚡ Performance Tests]
            E2 --> F
            E3 --> F
            F --> G[🌐 E2E Tests]
            G --> H[🚀 Deployment Readiness]
            H --> I{Ready?}
            I -->|✅ Yes| J[🎭 Staging Deploy]
            I -->|❌ No| K[🛑 Pipeline Failed]
            J --> L[🧪 Staging Tests]
            L --> M{Production?}
            M -->|✅ Main Branch| N[🚀 Production Deploy]
            M -->|❌ Other| O[✅ Pipeline Complete]
            N --> P[🏥 Health Checks]
            P --> Q[✅ Pipeline Complete]

            classDef successClass fill:#d4edda,stroke:#28a745,stroke-width:2px
            classDef failureClass fill:#f8d7da,stroke:#dc3545,stroke-width:2px
            classDef warningClass fill:#fff3cd,stroke:#ffc107,stroke-width:2px
            classDef infoClass fill:#d1ecf1,stroke:#17a2b8,stroke-width:2px

            class A,B,C,D,E,F,G,J,N successClass
            class K failureClass
            class H,I,M warningClass
            class O,Q infoClass
        ```
        """
        return mermaid

    def generate_pipeline_report(self, output_format: str = "terminal") -> str:
        """Generate comprehensive pipeline report."""
        if output_format == "html":
            return self.generate_html_dashboard()
        elif output_format == "mermaid":
            return self.generate_mermaid_pipeline_graph()
        else:
            return self.generate_terminal_dashboard()

    def save_report(self, report: str, output_path: str):
        """Save report to file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"{Colors.GREEN}✅ Report saved to: {output_path}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error saving report: {e}{Colors.RESET}")


def create_sample_pipeline_data() -> Dict[str, Any]:
    """Create sample pipeline data for testing."""
    return {
        "pipeline_id": f"pipeline-{int(datetime.now().timestamp())}-sample",
        "status": "success",
        "start_time": datetime.now().isoformat(),
        "end_time": (datetime.now() + timedelta(minutes=25)).isoformat(),
        "commit_sha": "a1b2c3d4e5f6789012345678901234567890abcd",
        "branch": "main",
        "trigger": "push",
        "actor": "developer",
        "change_scope": "full-stack",
        "security_sensitive": True,
        "deployment_ready": True,
        "stages": {
            "pre_flight": "success",
            "build": "success",
            "tests": "success",
            "security": "success",
            "performance": "success",
            "e2e": "success",
            "staging_deploy": "success",
            "production_deploy": "success",
        },
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pipeline Dashboard Generator")
    parser.add_argument("--data", type=str, help="Path to pipeline data JSON file")
    parser.add_argument(
        "--format",
        choices=["terminal", "html", "mermaid"],
        default="terminal",
        help="Output format",
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--sample", action="store_true", help="Use sample data")

    args = parser.parse_args()

    print_header("🚀 Pipeline Dashboard Generator", Colors.CYAN)

    dashboard = PipelineDashboard()

    if args.sample:
        print(f"{Colors.BLUE}📊 Using sample pipeline data{Colors.RESET}")
        dashboard.pipeline_data = create_sample_pipeline_data()
    elif args.data:
        if not dashboard.load_pipeline_data(args.data):
            print(f"{Colors.RED}❌ Failed to load pipeline data{Colors.RESET}")
            sys.exit(1)
    else:
        print(f"{Colors.YELLOW}⚠️ No data source specified. Use --data or --sample{Colors.RESET}")
        sys.exit(1)

    # Generate report
    report = dashboard.generate_pipeline_report(args.format)

    if args.output:
        dashboard.save_report(report, args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()
