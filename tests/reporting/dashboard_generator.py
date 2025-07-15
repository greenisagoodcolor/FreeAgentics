"""Test metrics and coverage dashboard generator."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generates comprehensive test metrics and coverage dashboard."""

    def __init__(
        self,
        metrics_db: str = "tests/reporting/test_metrics.db",
        coverage_db: str = "tests/reporting/coverage.db",
    ):
        self.metrics_db = Path(metrics_db)
        self.coverage_db = Path(coverage_db)

    def generate_dashboard(self, output_path: str = "tests/reporting/dashboard.html") -> str:
        """Generate complete dashboard HTML."""
        # Collect all data
        dashboard_data = self._collect_dashboard_data()

        # Generate HTML
        html_content = self._generate_html(dashboard_data)

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Dashboard generated: {output_path}")
        return output_path

    def _collect_dashboard_data(self) -> Dict[str, Any]:
        """Collect all dashboard data from databases."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "metrics_summary": self._get_metrics_summary(),
            "coverage_summary": self._get_coverage_summary(),
            "test_trends": self._get_test_trends(),
            "coverage_trends": self._get_coverage_trends(),
            "flaky_tests": self._get_flaky_tests(),
            "slow_tests": self._get_slow_tests(),
            "zero_coverage_files": self._get_zero_coverage_files(),
            "coverage_gaps": self._get_coverage_gaps(),
            "test_execution_history": self._get_test_execution_history(),
            "quality_metrics": self._get_quality_metrics(),
        }

        return data

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get test metrics summary."""
        if not self.metrics_db.exists():
            return {}

        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()

        # Get latest test run
        cursor.execute(
            """
            SELECT test_run_id, timestamp, total_tests, passed_tests, failed_tests, 
                   skipped_tests, error_tests, total_duration, environment
            FROM test_runs 
            ORDER BY timestamp DESC LIMIT 1
        """
        )

        latest_run = cursor.fetchone()
        if not latest_run:
            conn.close()
            return {}

        # Get test count trends (last 7 days)
        cursor.execute(
            """
            SELECT DATE(timestamp) as date, COUNT(*) as runs, 
                   AVG(CAST(passed_tests AS REAL) / total_tests * 100) as avg_pass_rate
            FROM test_runs 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        )

        trends = cursor.fetchall()
        conn.close()

        pass_rate = (latest_run[3] / latest_run[2]) * 100 if latest_run[2] > 0 else 0

        return {
            "latest_run": {
                "test_run_id": latest_run[0],
                "timestamp": latest_run[1],
                "total_tests": latest_run[2],
                "passed_tests": latest_run[3],
                "failed_tests": latest_run[4],
                "skipped_tests": latest_run[5],
                "error_tests": latest_run[6],
                "total_duration": latest_run[7],
                "environment": latest_run[8],
                "pass_rate": pass_rate,
            },
            "trends": trends,
        }

    def _get_coverage_summary(self) -> Dict[str, Any]:
        """Get coverage summary."""
        if not self.coverage_db.exists():
            return {}

        conn = sqlite3.connect(self.coverage_db)
        cursor = conn.cursor()

        # Get latest coverage run
        cursor.execute(
            """
            SELECT test_run_id, timestamp, total_statements, total_missing, total_coverage
            FROM coverage_runs 
            ORDER BY timestamp DESC LIMIT 1
        """
        )

        latest_run = cursor.fetchone()
        if not latest_run:
            conn.close()
            return {}

        # Get coverage trends (last 7 days)
        cursor.execute(
            """
            SELECT DATE(timestamp) as date, AVG(total_coverage) as avg_coverage
            FROM coverage_runs 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        )

        trends = cursor.fetchall()
        conn.close()

        return {
            "latest_run": {
                "test_run_id": latest_run[0],
                "timestamp": latest_run[1],
                "total_statements": latest_run[2],
                "total_missing": latest_run[3],
                "total_coverage": latest_run[4],
            },
            "trends": trends,
        }

    def _get_test_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get test execution trends."""
        if not self.metrics_db.exists():
            return []

        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT DATE(timestamp) as date, 
                   COUNT(*) as total_runs,
                   AVG(total_tests) as avg_tests,
                   AVG(CAST(passed_tests AS REAL) / total_tests * 100) as avg_pass_rate,
                   AVG(total_duration) as avg_duration
            FROM test_runs 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """,
            (since_date,),
        )

        trends = []
        for row in cursor.fetchall():
            trends.append(
                {
                    "date": row[0],
                    "total_runs": row[1],
                    "avg_tests": row[2],
                    "avg_pass_rate": row[3],
                    "avg_duration": row[4],
                }
            )

        conn.close()
        return trends

    def _get_coverage_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get coverage trends."""
        if not self.coverage_db.exists():
            return []

        conn = sqlite3.connect(self.coverage_db)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT DATE(timestamp) as date, 
                   AVG(total_coverage) as avg_coverage,
                   COUNT(*) as runs
            FROM coverage_runs 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """,
            (since_date,),
        )

        trends = []
        for row in cursor.fetchall():
            trends.append({"date": row[0], "avg_coverage": row[1], "runs": row[2]})

        conn.close()
        return trends

    def _get_flaky_tests(self) -> List[Dict[str, Any]]:
        """Get flaky tests."""
        if not self.metrics_db.exists():
            return []

        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT test_id, test_name, test_file, flaky_count, total_runs, 
                   flaky_percentage, first_flaky_run, last_flaky_run
            FROM flaky_tests 
            WHERE resolved_at IS NULL
            ORDER BY flaky_percentage DESC
            LIMIT 20
        """
        )

        flaky_tests = []
        for row in cursor.fetchall():
            flaky_tests.append(
                {
                    "test_id": row[0],
                    "test_name": row[1],
                    "test_file": row[2],
                    "flaky_count": row[3],
                    "total_runs": row[4],
                    "flaky_percentage": row[5],
                    "first_flaky_run": row[6],
                    "last_flaky_run": row[7],
                }
            )

        conn.close()
        return flaky_tests

    def _get_slow_tests(self) -> List[Dict[str, Any]]:
        """Get slow tests."""
        if not self.metrics_db.exists():
            return []

        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT test_id, test_name, test_file, avg_duration, max_duration, 
                   slow_run_count, first_detected, last_detected
            FROM slow_tests 
            ORDER BY avg_duration DESC
            LIMIT 20
        """
        )

        slow_tests = []
        for row in cursor.fetchall():
            slow_tests.append(
                {
                    "test_id": row[0],
                    "test_name": row[1],
                    "test_file": row[2],
                    "avg_duration": row[3],
                    "max_duration": row[4],
                    "slow_run_count": row[5],
                    "first_detected": row[6],
                    "last_detected": row[7],
                }
            )

        conn.close()
        return slow_tests

    def _get_zero_coverage_files(self) -> List[Dict[str, Any]]:
        """Get zero coverage files."""
        if not self.coverage_db.exists():
            return []

        conn = sqlite3.connect(self.coverage_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT file_path, first_detected, last_detected, times_detected
            FROM zero_coverage_tracking 
            WHERE resolved_at IS NULL
            ORDER BY times_detected DESC
            LIMIT 20
        """
        )

        zero_coverage = []
        for row in cursor.fetchall():
            zero_coverage.append(
                {
                    "file_path": row[0],
                    "first_detected": row[1],
                    "last_detected": row[2],
                    "times_detected": row[3],
                }
            )

        conn.close()
        return zero_coverage

    def _get_coverage_gaps(self) -> List[Dict[str, Any]]:
        """Get coverage gaps."""
        if not self.coverage_db.exists():
            return []

        conn = sqlite3.connect(self.coverage_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT fc.file_path, fc.coverage_percent, fc.missing, fc.statements
            FROM file_coverage fc
            JOIN coverage_runs cr ON fc.run_id = cr.id
            WHERE cr.timestamp = (
                SELECT MAX(timestamp) FROM coverage_runs
            ) AND fc.coverage_percent < 80
            ORDER BY fc.coverage_percent ASC
            LIMIT 20
        """
        )

        gaps = []
        for row in cursor.fetchall():
            gaps.append(
                {
                    "file_path": row[0],
                    "coverage_percent": row[1],
                    "missing_lines": row[2],
                    "total_statements": row[3],
                }
            )

        conn.close()
        return gaps

    def _get_test_execution_history(self) -> List[Dict[str, Any]]:
        """Get test execution history."""
        if not self.metrics_db.exists():
            return []

        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT test_run_id, timestamp, total_tests, passed_tests, failed_tests, 
                   total_duration, environment
            FROM test_runs 
            ORDER BY timestamp DESC
            LIMIT 50
        """
        )

        history = []
        for row in cursor.fetchall():
            pass_rate = (row[3] / row[2]) * 100 if row[2] > 0 else 0
            history.append(
                {
                    "test_run_id": row[0],
                    "timestamp": row[1],
                    "total_tests": row[2],
                    "passed_tests": row[3],
                    "failed_tests": row[4],
                    "total_duration": row[5],
                    "environment": row[6],
                    "pass_rate": pass_rate,
                }
            )

        conn.close()
        return history

    def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get overall quality metrics."""
        metrics = {
            "test_reliability": 0.0,
            "coverage_health": 0.0,
            "performance_score": 0.0,
            "overall_score": 0.0,
        }

        # Calculate test reliability (based on flaky tests)
        flaky_tests = self._get_flaky_tests()
        metrics_summary = self._get_metrics_summary()

        if metrics_summary and "latest_run" in metrics_summary:
            total_tests = metrics_summary["latest_run"]["total_tests"]
            if total_tests > 0:
                flaky_rate = len(flaky_tests) / total_tests
                metrics["test_reliability"] = max(0, 100 - (flaky_rate * 100))

        # Calculate coverage health
        coverage_summary = self._get_coverage_summary()
        if coverage_summary and "latest_run" in coverage_summary:
            metrics["coverage_health"] = coverage_summary["latest_run"]["total_coverage"]

        # Calculate performance score (based on slow tests)
        slow_tests = self._get_slow_tests()
        if metrics_summary and "latest_run" in metrics_summary:
            total_tests = metrics_summary["latest_run"]["total_tests"]
            if total_tests > 0:
                slow_rate = len(slow_tests) / total_tests
                metrics["performance_score"] = max(0, 100 - (slow_rate * 10))

        # Calculate overall score
        scores = [
            metrics["test_reliability"],
            metrics["coverage_health"],
            metrics["performance_score"],
        ]

        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            metrics["overall_score"] = sum(valid_scores) / len(valid_scores)

        return metrics

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML dashboard."""
        # Generate JavaScript data
        js_data = json.dumps(data, indent=2)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FreeAgentics Test Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 2rem;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    font-size: 2.5rem;
                    margin-bottom: 0.5rem;
                }}
                
                .header p {{
                    font-size: 1.1rem;
                    opacity: 0.9;
                }}
                
                .dashboard-container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 2rem;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                
                .metric-card {{
                    background: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.2s;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-2px);
                }}
                
                .metric-value {{
                    font-size: 2.5rem;
                    font-weight: bold;
                    margin-bottom: 0.5rem;
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: #666;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                
                .good {{ color: #4caf50; }}
                .warning {{ color: #ff9800; }}
                .critical {{ color: #f44336; }}
                
                .chart-container {{
                    background: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                }}
                
                .chart-title {{
                    font-size: 1.3rem;
                    margin-bottom: 1rem;
                    color: #333;
                }}
                
                .chart-canvas {{
                    width: 100%;
                    height: 400px;
                }}
                
                .table-container {{
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    overflow: hidden;
                    margin-bottom: 2rem;
                }}
                
                .table-header {{
                    background: #f8f9fa;
                    padding: 1rem 1.5rem;
                    border-bottom: 1px solid #e9ecef;
                    font-weight: bold;
                    color: #333;
                }}
                
                .table-content {{
                    max-height: 400px;
                    overflow-y: auto;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                
                th, td {{
                    padding: 0.75rem 1rem;
                    text-align: left;
                    border-bottom: 1px solid #e9ecef;
                }}
                
                th {{
                    background: #f8f9fa;
                    font-weight: 600;
                    color: #333;
                }}
                
                .flaky-row {{
                    background-color: #ffebee;
                }}
                
                .slow-row {{
                    background-color: #fff3e0;
                }}
                
                .zero-coverage-row {{
                    background-color: #ffcdd2;
                }}
                
                .progress-bar {{
                    background: #e9ecef;
                    border-radius: 10px;
                    height: 20px;
                    overflow: hidden;
                    margin-top: 0.5rem;
                }}
                
                .progress-fill {{
                    height: 100%;
                    transition: width 0.3s ease;
                }}
                
                .grid-2 {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 2rem;
                }}
                
                @media (max-width: 768px) {{
                    .grid-2 {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .dashboard-container {{
                        padding: 1rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 FreeAgentics Test Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="dashboard-container">
                <!-- Quality Metrics -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value good" id="overall-score">--</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="test-reliability">--</div>
                        <div class="metric-label">Test Reliability</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="coverage-health">--</div>
                        <div class="metric-label">Coverage Health</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="performance-score">--</div>
                        <div class="metric-label">Performance Score</div>
                    </div>
                </div>
                
                <!-- Test and Coverage Summary -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="total-tests">--</div>
                        <div class="metric-label">Total Tests</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="pass-rate">--</div>
                        <div class="metric-label">Pass Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="total-coverage">--</div>
                        <div class="metric-label">Code Coverage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value warning" id="flaky-count">--</div>
                        <div class="metric-label">Flaky Tests</div>
                    </div>
                </div>
                
                <!-- Charts -->
                <div class="grid-2">
                    <div class="chart-container">
                        <div class="chart-title">Test Pass Rate Trend</div>
                        <canvas id="passRateChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">Coverage Trend</div>
                        <canvas id="coverageChart" class="chart-canvas"></canvas>
                    </div>
                </div>
                
                <!-- Test Execution Duration -->
                <div class="chart-container">
                    <div class="chart-title">Test Execution Duration</div>
                    <canvas id="durationChart" class="chart-canvas"></canvas>
                </div>
                
                <!-- Data Tables -->
                <div class="grid-2">
                    <div class="table-container">
                        <div class="table-header">🔀 Flaky Tests</div>
                        <div class="table-content">
                            <table id="flaky-tests-table">
                                <thead>
                                    <tr>
                                        <th>Test Name</th>
                                        <th>Flaky %</th>
                                        <th>Failures</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="table-container">
                        <div class="table-header">🐌 Slow Tests</div>
                        <div class="table-content">
                            <table id="slow-tests-table">
                                <thead>
                                    <tr>
                                        <th>Test Name</th>
                                        <th>Avg Duration</th>
                                        <th>Max Duration</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="grid-2">
                    <div class="table-container">
                        <div class="table-header">📊 Zero Coverage Files</div>
                        <div class="table-content">
                            <table id="zero-coverage-table">
                                <thead>
                                    <tr>
                                        <th>File Path</th>
                                        <th>Times Detected</th>
                                        <th>First Detected</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="table-container">
                        <div class="table-header">⚠️ Coverage Gaps</div>
                        <div class="table-content">
                            <table id="coverage-gaps-table">
                                <thead>
                                    <tr>
                                        <th>File Path</th>
                                        <th>Coverage %</th>
                                        <th>Missing Lines</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Test Runs -->
                <div class="table-container">
                    <div class="table-header">📈 Recent Test Runs</div>
                    <div class="table-content">
                        <table id="test-history-table">
                            <thead>
                                <tr>
                                    <th>Run ID</th>
                                    <th>Timestamp</th>
                                    <th>Tests</th>
                                    <th>Pass Rate</th>
                                    <th>Duration</th>
                                    <th>Environment</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <script>
                // Dashboard data
                const dashboardData = {js_data};
                
                // Initialize dashboard
                function initDashboard() {{
                    updateMetrics();
                    createCharts();
                    populateTables();
                }}
                
                function updateMetrics() {{
                    const metrics = dashboardData.metrics_summary;
                    const coverage = dashboardData.coverage_summary;
                    const quality = dashboardData.quality_metrics;
                    
                    // Quality metrics
                    document.getElementById('overall-score').textContent = quality.overall_score ? quality.overall_score.toFixed(1) + '%' : '--';
                    document.getElementById('test-reliability').textContent = quality.test_reliability ? quality.test_reliability.toFixed(1) + '%' : '--';
                    document.getElementById('coverage-health').textContent = quality.coverage_health ? quality.coverage_health.toFixed(1) + '%' : '--';
                    document.getElementById('performance-score').textContent = quality.performance_score ? quality.performance_score.toFixed(1) + '%' : '--';
                    
                    // Test metrics
                    if (metrics.latest_run) {{
                        document.getElementById('total-tests').textContent = metrics.latest_run.total_tests;
                        document.getElementById('pass-rate').textContent = metrics.latest_run.pass_rate.toFixed(1) + '%';
                        
                        // Color code pass rate
                        const passRateElement = document.getElementById('pass-rate');
                        if (metrics.latest_run.pass_rate >= 90) {{
                            passRateElement.className = 'metric-value good';
                        }} else if (metrics.latest_run.pass_rate >= 70) {{
                            passRateElement.className = 'metric-value warning';
                        }} else {{
                            passRateElement.className = 'metric-value critical';
                        }}
                    }}
                    
                    // Coverage metrics
                    if (coverage.latest_run) {{
                        document.getElementById('total-coverage').textContent = coverage.latest_run.total_coverage.toFixed(1) + '%';
                        
                        // Color code coverage
                        const coverageElement = document.getElementById('total-coverage');
                        if (coverage.latest_run.total_coverage >= 80) {{
                            coverageElement.className = 'metric-value good';
                        }} else if (coverage.latest_run.total_coverage >= 60) {{
                            coverageElement.className = 'metric-value warning';
                        }} else {{
                            coverageElement.className = 'metric-value critical';
                        }}
                    }}
                    
                    // Flaky tests count
                    document.getElementById('flaky-count').textContent = dashboardData.flaky_tests.length;
                }}
                
                function createCharts() {{
                    // Pass rate trend chart
                    const passRateCtx = document.getElementById('passRateChart').getContext('2d');
                    new Chart(passRateCtx, {{
                        type: 'line',
                        data: {{
                            labels: dashboardData.test_trends.map(t => t.date),
                            datasets: [{{
                                label: 'Pass Rate %',
                                data: dashboardData.test_trends.map(t => t.avg_pass_rate),
                                borderColor: '#4caf50',
                                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                tension: 0.4
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    max: 100
                                }}
                            }}
                        }}
                    }});
                    
                    // Coverage trend chart
                    const coverageCtx = document.getElementById('coverageChart').getContext('2d');
                    new Chart(coverageCtx, {{
                        type: 'line',
                        data: {{
                            labels: dashboardData.coverage_trends.map(t => t.date),
                            datasets: [{{
                                label: 'Coverage %',
                                data: dashboardData.coverage_trends.map(t => t.avg_coverage),
                                borderColor: '#2196f3',
                                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                                tension: 0.4
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    max: 100
                                }}
                            }}
                        }}
                    }});
                    
                    // Duration chart
                    const durationCtx = document.getElementById('durationChart').getContext('2d');
                    new Chart(durationCtx, {{
                        type: 'bar',
                        data: {{
                            labels: dashboardData.test_trends.map(t => t.date),
                            datasets: [{{
                                label: 'Duration (seconds)',
                                data: dashboardData.test_trends.map(t => t.avg_duration),
                                backgroundColor: '#ff9800',
                                borderColor: '#f57c00',
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }});
                }}
                
                function populateTables() {{
                    // Flaky tests table
                    const flakyTable = document.getElementById('flaky-tests-table').querySelector('tbody');
                    dashboardData.flaky_tests.forEach(test => {{
                        const row = flakyTable.insertRow();
                        row.className = 'flaky-row';
                        row.insertCell(0).textContent = test.test_name;
                        row.insertCell(1).textContent = test.flaky_percentage.toFixed(1) + '%';
                        row.insertCell(2).textContent = test.flaky_count + '/' + test.total_runs;
                    }});
                    
                    // Slow tests table
                    const slowTable = document.getElementById('slow-tests-table').querySelector('tbody');
                    dashboardData.slow_tests.forEach(test => {{
                        const row = slowTable.insertRow();
                        row.className = 'slow-row';
                        row.insertCell(0).textContent = test.test_name;
                        row.insertCell(1).textContent = test.avg_duration.toFixed(2) + 's';
                        row.insertCell(2).textContent = test.max_duration.toFixed(2) + 's';
                    }});
                    
                    // Zero coverage table
                    const zeroTable = document.getElementById('zero-coverage-table').querySelector('tbody');
                    dashboardData.zero_coverage_files.forEach(file => {{
                        const row = zeroTable.insertRow();
                        row.className = 'zero-coverage-row';
                        row.insertCell(0).textContent = file.file_path;
                        row.insertCell(1).textContent = file.times_detected;
                        row.insertCell(2).textContent = file.first_detected;
                    }});
                    
                    // Coverage gaps table
                    const gapsTable = document.getElementById('coverage-gaps-table').querySelector('tbody');
                    dashboardData.coverage_gaps.forEach(gap => {{
                        const row = gapsTable.insertRow();
                        row.insertCell(0).textContent = gap.file_path;
                        row.insertCell(1).textContent = gap.coverage_percent.toFixed(1) + '%';
                        row.insertCell(2).textContent = gap.missing_lines;
                    }});
                    
                    // Test history table
                    const historyTable = document.getElementById('test-history-table').querySelector('tbody');
                    dashboardData.test_execution_history.forEach(run => {{
                        const row = historyTable.insertRow();
                        row.insertCell(0).textContent = run.test_run_id;
                        row.insertCell(1).textContent = new Date(run.timestamp).toLocaleString();
                        row.insertCell(2).textContent = run.total_tests;
                        row.insertCell(3).textContent = run.pass_rate.toFixed(1) + '%';
                        row.insertCell(4).textContent = run.total_duration.toFixed(1) + 's';
                        row.insertCell(5).textContent = run.environment;
                    }});
                }}
                
                // Initialize dashboard when page loads
                document.addEventListener('DOMContentLoaded', initDashboard);
            </script>
        </body>
        </html>
        """

        return html_content

    def generate_json_export(self, output_path: str = "tests/reporting/dashboard_data.json") -> str:
        """Export dashboard data as JSON."""
        data = self._collect_dashboard_data()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Dashboard data exported: {output_path}")
        return output_path


def main():
    """Generate dashboard."""
    generator = DashboardGenerator()

    # Generate HTML dashboard
    dashboard_path = generator.generate_dashboard()
    print(f"Dashboard generated: {dashboard_path}")

    # Export JSON data
    json_path = generator.generate_json_export()
    print(f"JSON data exported: {json_path}")


if __name__ == "__main__":
    main()
