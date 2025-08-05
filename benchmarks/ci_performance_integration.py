#!/usr/bin/env python3
"""
CI Integration for Performance Gates.

Simple script for GitHub Actions integration.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_benchmarks():
    """Run benchmarks and return results."""
    try:
        print("Running benchmarks...")
        result = subprocess.run(
            ["python", "benchmarks/simple_benchmark_runner.py"],
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark execution failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_performance_gates():
    """Check performance gates."""
    if not Path("latest_benchmark_results.json").exists():
        print("❌ No benchmark results found")
        return False

    try:
        result = subprocess.run(
            ["python", "benchmarks/performance_gate.py", "latest_benchmark_results.json"],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"Performance gate check failed: {e}")
        return False


def create_pr_comment():
    """Create PR comment with performance results."""
    if not Path("latest_benchmark_results.json").exists():
        return "⚠️ No performance data available"

    try:
        with open("latest_benchmark_results.json") as f:
            results = json.load(f)

        metrics = results.get("metrics", {})
        summary = results.get("summary", {})

        comment = [
            "## 📊 Performance Benchmark Results",
            "",
            "### Core Metrics",
            "",
            f"| Metric | Value | Target | Status |",
            f"|--------|-------|--------|--------|",
        ]

        # Agent spawn metrics
        if "agent_spawning" in metrics:
            spawn_p95 = metrics["agent_spawning"]["p95_ms"]
            spawn_status = "✅ PASS" if spawn_p95 < 50.0 else "❌ FAIL"
            comment.append(f"| Agent Spawn P95 | {spawn_p95:.1f}ms | <50ms | {spawn_status} |")

        # Memory metrics
        if "memory_usage" in metrics:
            memory_per_agent = metrics["memory_usage"]["per_agent_mb"]
            memory_status = "✅ PASS" if memory_per_agent < 34.5 else "❌ FAIL"
            comment.append(
                f"| Memory per Agent | {memory_per_agent:.1f}MB | <34.5MB | {memory_status} |"
            )

        # PyMDP metrics
        if "pymdp_inference" in metrics:
            pymdp_p95 = metrics["pymdp_inference"]["p95_ms"]
            comment.append(f"| PyMDP P95 | {pymdp_p95:.1f}ms | - | ℹ️ INFO |")

        # API metrics
        if "api_performance" in metrics:
            api_p95 = metrics["api_performance"]["p95_ms"]
            comment.append(f"| API P95 | {api_p95:.1f}ms | - | ℹ️ INFO |")

        comment.extend(
            [
                "",
                "### Summary",
                "",
                f"- **Total Duration**: {summary.get('total_duration_ms', 0):.0f}ms",
                f"- **Benchmarks Run**: {summary.get('benchmarks_run', 0)}",
            ]
        )

        # Overall status
        if summary.get("agent_spawn_passed", True) and summary.get("memory_passed", True):
            comment.append("- **Overall Status**: ✅ All performance gates passed")
        else:
            comment.append("- **Overall Status**: ❌ Some performance gates failed")

        return "\n".join(comment)

    except Exception as e:
        return f"⚠️ Error generating performance report: {e}"


def main():
    """Main CI integration function."""
    print("=" * 60)
    print("PERFORMANCE CI INTEGRATION")
    print("=" * 60)

    # Run benchmarks
    if not run_benchmarks():
        print("❌ Benchmark execution failed")
        sys.exit(1)

    # Check performance gates
    if not check_performance_gates():
        print("❌ Performance gates failed")

        # Generate PR comment for failed gates
        comment = create_pr_comment()
        print("\n" + "=" * 40)
        print("PR COMMENT CONTENT:")
        print("=" * 40)
        print(comment)

        sys.exit(1)

    # Generate success PR comment
    comment = create_pr_comment()
    print("\n" + "=" * 40)
    print("PR COMMENT CONTENT:")
    print("=" * 40)
    print(comment)

    print("\n✅ Performance CI check passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
