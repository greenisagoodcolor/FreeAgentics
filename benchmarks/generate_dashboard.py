#!/usr/bin/env python3
"""
Performance Dashboard Generator
PERF-ENGINEER: Bryan Cantrill + Brendan Gregg Methodology
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class PerformanceDashboard:
    """Generate performance dashboards in various formats."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.template = self._load_template()

    def _load_template(self) -> str:
        """Load HTML template for dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreeAgentics Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .metric-change {
            font-size: 14px;
            margin-top: 5px;
        }
        .improvement { color: #10b981; }
        .regression { color: #ef4444; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        canvas {
            max-height: 400px;
        }
        .benchmark-table {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        th {
            font-weight: 600;
            color: #374151;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-pass { background: #d1fae5; color: #065f46; }
        .status-warning { background: #fed7aa; color: #92400e; }
        .status-fail { background: #fee2e2; color: #991b1b; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 FreeAgentics Performance Dashboard</h1>
        <p>Following Bryan Cantrill + Brendan Gregg Methodology</p>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="metrics-grid">
        {metrics_cards}
    </div>

    <div class="chart-container">
        <div class="chart-title">Performance Trends (Last 30 Days)</div>
        <canvas id="trendsChart"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Benchmark Comparison</div>
        <canvas id="comparisonChart"></canvas>
    </div>

    <div class="benchmark-table">
        <h2>Detailed Benchmark Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Duration (ms)</th>
                    <th>Throughput (ops/s)</th>
                    <th>Memory (MB)</th>
                    <th>Change</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <script>
        // Trends Chart
        const trendsCtx = document.getElementById('trendsChart').getContext('2d');
        new Chart(trendsCtx, {{
            type: 'line',
            data: {trends_data},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        // Comparison Chart
        const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
        new Chart(comparisonCtx, {{
            type: 'bar',
            data: {comparison_data},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

    def load_latest_results(self) -> Optional[Dict[str, Any]]:
        """Load the latest benchmark results."""
        daily_dir = self.results_dir / "daily"
        if not daily_dir.exists():
            return None

        # Find most recent file
        files = sorted(daily_dir.glob("*_results.json"), reverse=True)
        if not files:
            return None

        with open(files[0], "r") as f:
            return json.load(f)

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline results."""
        baseline_file = self.results_dir / "baseline" / "baseline_results.json"
        if not baseline_file.exists():
            return None

        with open(baseline_file, "r") as f:
            return json.load(f)

    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from results."""
        metrics = {
            "total_benchmarks": 0,
            "avg_duration_ms": 0,
            "total_throughput": 0,
            "avg_memory_mb": 0,
            "improvements": 0,
            "regressions": 0,
        }

        if "benchmarks" in results:
            benchmarks = results["benchmarks"]
            metrics["total_benchmarks"] = len(benchmarks)

            durations = []
            throughputs = []
            memories = []

            for bench in benchmarks:
                stats = bench.get("stats", {})
                durations.append(stats.get("mean", 0) * 1000)  # Convert to ms

                if stats.get("mean", 0) > 0:
                    throughputs.append(1.0 / stats["mean"])

                extra = bench.get("extra_info", {})
                if "memory_mb" in extra:
                    memories.append(extra["memory_mb"])

            if durations:
                metrics["avg_duration_ms"] = statistics.mean(durations)
            if throughputs:
                metrics["total_throughput"] = sum(throughputs)
            if memories:
                metrics["avg_memory_mb"] = statistics.mean(memories)

        return metrics

    def generate_trends_data(self, days: int = 30) -> Dict[str, Any]:
        """Generate trend data for charts."""
        daily_dir = self.results_dir / "daily"
        if not daily_dir.exists():
            return {"labels": [], "datasets": []}

        # Load last N days of data
        files = sorted(daily_dir.glob("*_results.json"))[-days:]

        labels = []
        duration_data = []
        throughput_data = []
        memory_data = []

        for file in files:
            # Extract date from filename
            date_str = file.stem.replace("_results", "")
            labels.append(date_str)

            with open(file, "r") as f:
                data = json.load(f)
                metrics = self.calculate_metrics(data)
                duration_data.append(metrics["avg_duration_ms"])
                throughput_data.append(metrics["total_throughput"])
                memory_data.append(metrics["avg_memory_mb"])

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Avg Duration (ms)",
                    "data": duration_data,
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.1)",
                },
                {
                    "label": "Total Throughput (ops/s)",
                    "data": throughput_data,
                    "borderColor": "rgb(54, 162, 235)",
                    "backgroundColor": "rgba(54, 162, 235, 0.1)",
                },
                {
                    "label": "Avg Memory (MB)",
                    "data": memory_data,
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.1)",
                },
            ],
        }

    def generate_comparison_data(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison data for bar chart."""
        if "benchmarks" not in current:
            return {"labels": [], "datasets": []}

        labels = []
        durations = []

        for bench in current["benchmarks"][:10]:  # Top 10 benchmarks
            name = bench["name"].split(".")[-1]
            labels.append(name)
            durations.append(bench.get("stats", {}).get("mean", 0) * 1000)

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Duration (ms)",
                    "data": durations,
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.6)",
                        "rgba(54, 162, 235, 0.6)",
                        "rgba(255, 206, 86, 0.6)",
                        "rgba(75, 192, 192, 0.6)",
                        "rgba(153, 102, 255, 0.6)",
                        "rgba(255, 159, 64, 0.6)",
                        "rgba(199, 199, 199, 0.6)",
                        "rgba(83, 102, 255, 0.6)",
                        "rgba(255, 99, 132, 0.6)",
                        "rgba(54, 162, 235, 0.6)",
                    ],
                }
            ],
        }

    def generate_metric_card(self, title: str, value: Any, change: Optional[float] = None) -> str:
        """Generate HTML for a metric card."""
        change_html = ""
        if change is not None:
            change_class = "improvement" if change < 0 else "regression"
            change_symbol = "↓" if change < 0 else "↑"
            change_html = f'<div class="metric-change {change_class}">{change_symbol} {abs(change):.1f}%</div>'

        return f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {change_html}
        </div>
        """

    def generate_table_row(
        self, benchmark: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate HTML for a benchmark table row."""
        name = benchmark["name"].split(".")[-1]
        stats = benchmark.get("stats", {})
        duration = stats.get("mean", 0) * 1000
        throughput = 1.0 / stats.get("mean", 1) if stats.get("mean", 0) > 0 else 0
        memory = benchmark.get("extra_info", {}).get("memory_mb", 0)

        # Calculate change
        change = 0
        status = "pass"
        if baseline:
            # Find matching baseline
            for base_bench in baseline.get("benchmarks", []):
                if base_bench["name"] == benchmark["name"]:
                    base_duration = base_bench.get("stats", {}).get("mean", 0) * 1000
                    if base_duration > 0:
                        change = ((duration - base_duration) / base_duration) * 100
                        if change > 10:
                            status = "fail"
                        elif change > 5:
                            status = "warning"
                    break

        change_str = f"{change:+.1f}%" if change != 0 else "—"
        status_badge = f'<span class="status-badge status-{status}">{status.upper()}</span>'

        return f"""
        <tr>
            <td>{name}</td>
            <td>{duration:.2f}</td>
            <td>{throughput:.1f}</td>
            <td>{memory:.1f}</td>
            <td>{change_str}</td>
            <td>{status_badge}</td>
        </tr>
        """

    def generate_html(self, output_file: Path):
        """Generate HTML dashboard."""
        # Load data
        current = self.load_latest_results()
        baseline = self.load_baseline()

        if not current:
            print("No benchmark results found")
            return

        # Calculate metrics
        metrics = self.calculate_metrics(current)

        # Generate metric cards
        metric_cards = [
            self.generate_metric_card("Total Benchmarks", metrics["total_benchmarks"]),
            self.generate_metric_card("Avg Duration", f"{metrics['avg_duration_ms']:.1f}ms"),
            self.generate_metric_card(
                "Total Throughput", f"{metrics['total_throughput']:.0f} ops/s"
            ),
            self.generate_metric_card("Avg Memory", f"{metrics['avg_memory_mb']:.1f}MB"),
        ]

        # Generate table rows
        table_rows = []
        if "benchmarks" in current:
            for bench in current["benchmarks"]:
                table_rows.append(self.generate_table_row(bench, baseline))

        # Generate chart data
        trends_data = json.dumps(self.generate_trends_data())
        comparison_data = json.dumps(self.generate_comparison_data(current))

        # Fill template
        html = self.template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics_cards="\n".join(metric_cards),
            table_rows="\n".join(table_rows),
            trends_data=trends_data,
            comparison_data=comparison_data,
        )

        # Write output
        with open(output_file, "w") as f:
            f.write(html)

        print(f"✅ Dashboard generated: {output_file}")

    def generate_markdown(self, output_file: Path):
        """Generate Markdown dashboard."""
        # Load data
        current = self.load_latest_results()
        baseline = self.load_baseline()

        if not current:
            print("No benchmark results found")
            return

        # Calculate metrics
        metrics = self.calculate_metrics(current)

        md = f"""# 🚀 FreeAgentics Performance Dashboard

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Summary Metrics

- **Total Benchmarks:** {metrics['total_benchmarks']}
- **Average Duration:** {metrics['avg_duration_ms']:.1f}ms
- **Total Throughput:** {metrics['total_throughput']:.0f} ops/s
- **Average Memory:** {metrics['avg_memory_mb']:.1f}MB

## 📈 Benchmark Results

| Benchmark | Duration (ms) | Throughput (ops/s) | Memory (MB) | Change | Status |
|-----------|--------------|-------------------|-------------|---------|---------|
"""

        if "benchmarks" in current:
            for bench in current["benchmarks"]:
                name = bench["name"].split(".")[-1]
                stats = bench.get("stats", {})
                duration = stats.get("mean", 0) * 1000
                throughput = 1.0 / stats.get("mean", 1) if stats.get("mean", 0) > 0 else 0
                memory = bench.get("extra_info", {}).get("memory_mb", 0)

                # Calculate change
                change = 0
                status = "✅"
                if baseline:
                    for base_bench in baseline.get("benchmarks", []):
                        if base_bench["name"] == bench["name"]:
                            base_duration = base_bench.get("stats", {}).get("mean", 0) * 1000
                            if base_duration > 0:
                                change = ((duration - base_duration) / base_duration) * 100
                                if change > 10:
                                    status = "❌"
                                elif change > 5:
                                    status = "⚠️"
                            break

                change_str = f"{change:+.1f}%" if change != 0 else "—"
                md += f"| {name} | {duration:.2f} | {throughput:.1f} | {memory:.1f} | {change_str} | {status} |\n"

        md += "\n---\n*Powered by PERF-ENGINEER • Following Bryan Cantrill + Brendan Gregg Methodology*\n"

        with open(output_file, "w") as f:
            f.write(md)

        print(f"✅ Markdown dashboard generated: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate performance dashboard")

    parser.add_argument("--output", default="performance_dashboard.html", help="Output file path")

    parser.add_argument(
        "--format", choices=["html", "markdown"], default="html", help="Output format"
    )

    parser.add_argument("--results-dir", default="benchmarks/results", help="Results directory")

    args = parser.parse_args()

    # Generate dashboard
    dashboard = PerformanceDashboard(args.results_dir)

    output_path = Path(args.output)

    if args.format == "html":
        dashboard.generate_html(output_path)
    else:
        dashboard.generate_markdown(output_path)


if __name__ == "__main__":
    main()
