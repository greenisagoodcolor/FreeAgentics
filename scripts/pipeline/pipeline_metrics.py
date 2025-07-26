#!/usr/bin/env python3
"""
Pipeline Metrics Collector for PIPELINE-ARCHITECT
Collects, analyzes, and reports pipeline performance metrics
"""

import argparse
import json
import sqlite3
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple


class PipelineMetrics:
    """Pipeline metrics collector and analyzer."""

    def __init__(self, db_path: str = ".pipeline-data/metrics.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for metrics storage."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT UNIQUE NOT NULL,
                commit_sha TEXT NOT NULL,
                branch TEXT NOT NULL,
                trigger_event TEXT NOT NULL,
                actor TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                duration_seconds INTEGER,
                status TEXT NOT NULL,
                change_scope TEXT,
                security_sensitive BOOLEAN,
                deployment_ready BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pipeline_id) REFERENCES pipeline_runs (pipeline_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                threshold REAL,
                passed BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pipeline_id) REFERENCES pipeline_runs (pipeline_id)
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_branch ON pipeline_runs(branch)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date ON pipeline_runs(started_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_stage_metrics_pipeline ON stage_metrics(pipeline_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality_metrics_pipeline ON quality_metrics(pipeline_id)"
        )

        conn.commit()
        conn.close()

    def record_pipeline_run(self, pipeline_data: Dict[str, Any]) -> bool:
        """Record a pipeline run in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate duration if both times are available
            duration = None
            if pipeline_data.get("start_time") and pipeline_data.get("end_time"):
                start = datetime.fromisoformat(pipeline_data["start_time"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(pipeline_data["end_time"].replace("Z", "+00:00"))
                duration = int((end - start).total_seconds())

            # Insert pipeline run
            cursor.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs
                (pipeline_id, commit_sha, branch, trigger_event, actor, started_at,
                 completed_at, duration_seconds, status, change_scope,
                 security_sensitive, deployment_ready)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pipeline_data.get("pipeline_id"),
                    pipeline_data.get("commit_sha"),
                    pipeline_data.get("branch"),
                    pipeline_data.get("trigger"),
                    pipeline_data.get("actor"),
                    pipeline_data.get("start_time"),
                    pipeline_data.get("end_time"),
                    duration,
                    pipeline_data.get("status", "unknown"),
                    pipeline_data.get("change_scope"),
                    pipeline_data.get("security_sensitive", False),
                    pipeline_data.get("deployment_ready", False),
                ),
            )

            # Insert stage metrics
            stages = pipeline_data.get("stages", {})
            for stage_name, stage_status in stages.items():
                cursor.execute(
                    """
                    INSERT INTO stage_metrics
                    (pipeline_id, stage_name, status)
                    VALUES (?, ?, ?)
                """,
                    (pipeline_data.get("pipeline_id"), stage_name, stage_status),
                )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"❌ Error recording pipeline run: {e}")
            return False

    def get_success_rate(self, days: int = 30, branch: str = None) -> float:
        """Calculate pipeline success rate over the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT status, COUNT(*) as count
            FROM pipeline_runs
            WHERE started_at >= ?
        """
        params = [since_date.isoformat()]

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        query += " GROUP BY status"

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        total = sum(count for _, count in results)
        if total == 0:
            return 0.0

        success_count = sum(count for status, count in results if status == "success")
        return (success_count / total) * 100

    def get_average_duration(self, days: int = 30, branch: str = None) -> Tuple[float, List[int]]:
        """Get average pipeline duration and trend data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT duration_seconds
            FROM pipeline_runs
            WHERE started_at >= ? AND duration_seconds IS NOT NULL
        """
        params = [since_date.isoformat()]

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        cursor.execute(query, params)
        durations = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not durations:
            return 0.0, []

        avg_duration = statistics.mean(durations)
        return avg_duration, durations

    def get_failure_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze failure trends by stage and time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = datetime.now() - timedelta(days=days)

        # Get stage failure rates
        cursor.execute(
            """
            SELECT stage_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) as failures
            FROM stage_metrics sm
            JOIN pipeline_runs pr ON sm.pipeline_id = pr.pipeline_id
            WHERE pr.started_at >= ?
            GROUP BY stage_name
            ORDER BY failures DESC
        """,
            [since_date.isoformat()],
        )

        stage_failures = []
        for stage_name, total, failures in cursor.fetchall():
            failure_rate = (failures / total) * 100 if total > 0 else 0
            stage_failures.append(
                {
                    "stage": stage_name,
                    "total_runs": total,
                    "failures": failures,
                    "failure_rate": failure_rate,
                }
            )

        # Get daily failure counts
        cursor.execute(
            """
            SELECT DATE(started_at) as date,
                   COUNT(*) as total,
                   SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as failures
            FROM pipeline_runs
            WHERE started_at >= ?
            GROUP BY DATE(started_at)
            ORDER BY date
        """,
            [since_date.isoformat()],
        )

        daily_trends = []
        for date, total, failures in cursor.fetchall():
            failure_rate = (failures / total) * 100 if total > 0 else 0
            daily_trends.append(
                {
                    "date": date,
                    "total_runs": total,
                    "failures": failures,
                    "failure_rate": failure_rate,
                }
            )

        conn.close()

        return {"stage_failures": stage_failures, "daily_trends": daily_trends}

    def get_pipeline_health_score(self) -> Dict[str, Any]:
        """Calculate overall pipeline health score."""
        success_rate_30d = self.get_success_rate(30)
        success_rate_7d = self.get_success_rate(7)
        avg_duration, _ = self.get_average_duration(30)
        failure_trends = self.get_failure_trends(7)

        # Calculate health score (0-100)
        health_score = 0

        # Success rate component (40% weight)
        health_score += min(success_rate_30d, 100) * 0.4

        # Duration component (20% weight) - target 15 minutes (900 seconds)
        target_duration = 900  # 15 minutes
        if avg_duration > 0:
            duration_score = max(
                0, 100 - ((avg_duration - target_duration) / target_duration * 100)
            )
            health_score += max(0, min(100, duration_score)) * 0.2

        # Trend component (20% weight)
        trend_score = 100
        if success_rate_7d < success_rate_30d:
            trend_score = max(0, trend_score - ((success_rate_30d - success_rate_7d) * 2))
        health_score += trend_score * 0.2

        # Stability component (20% weight) - based on stage failure rates
        stability_score = 100
        for stage in failure_trends["stage_failures"]:
            if stage["failure_rate"] > 10:  # More than 10% failure rate
                stability_score -= stage["failure_rate"]
        health_score += max(0, stability_score) * 0.2

        return {
            "health_score": round(health_score, 1),
            "success_rate_30d": round(success_rate_30d, 1),
            "success_rate_7d": round(success_rate_7d, 1),
            "avg_duration_minutes": round(avg_duration / 60, 1) if avg_duration else 0,
            "trend_direction": "improving" if success_rate_7d >= success_rate_30d else "declining",
            "most_failing_stage": (
                failure_trends["stage_failures"][0]["stage"]
                if failure_trends["stage_failures"]
                else None
            ),
        }

    def generate_metrics_report(self, format: str = "json") -> str:
        """Generate comprehensive metrics report."""
        health_score = self.get_pipeline_health_score()
        failure_trends = self.get_failure_trends(30)

        # Get recent pipelines
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT pipeline_id, branch, status, started_at, duration_seconds
            FROM pipeline_runs
            ORDER BY started_at DESC
            LIMIT 10
        """
        )
        recent_pipelines = [
            {
                "pipeline_id": row[0],
                "branch": row[1],
                "status": row[2],
                "started_at": row[3],
                "duration_minutes": round(row[4] / 60, 1) if row[4] else None,
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "health_score": health_score,
            "failure_trends": failure_trends,
            "recent_pipelines": recent_pipelines,
            "summary": {
                "total_pipelines": len(recent_pipelines),
                "health_grade": self._get_health_grade(health_score["health_score"]),
                "recommendation": self._get_recommendation(health_score),
            },
        }

        if format == "json":
            return json.dumps(report_data, indent=2)
        elif format == "markdown":
            return self._format_markdown_report(report_data)
        else:
            return str(report_data)

    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _get_recommendation(self, health_data: Dict[str, Any]) -> str:
        """Generate recommendations based on health data."""
        score = health_data["health_score"]

        if score >= 90:
            return "Pipeline health is excellent. Continue current practices."
        elif score >= 80:
            return "Pipeline health is good. Monitor for any declining trends."
        elif score >= 70:
            return f"Pipeline health needs improvement. Focus on {health_data.get('most_failing_stage', 'failing stages')}."
        elif score >= 60:
            return "Pipeline health is poor. Immediate attention required for stability issues."
        else:
            return "Pipeline health is critical. Urgent intervention needed across all stages."

    def _format_markdown_report(self, data: Dict[str, Any]) -> str:
        """Format metrics report as Markdown."""
        health = data["health_score"]

        markdown = f"""# Pipeline Metrics Report

**Generated:** {data["generated_at"]}
**Health Score:** {health["health_score"]}/100 (Grade: {data["summary"]["health_grade"]})
**Trend:** {health["trend_direction"].title()}

## 📊 Key Metrics

- **Success Rate (30 days):** {health["success_rate_30d"]}%
- **Success Rate (7 days):** {health["success_rate_7d"]}%
- **Average Duration:** {health["avg_duration_minutes"]} minutes
- **Most Failing Stage:** {health["most_failing_stage"] or "None"}

## 🎯 Recommendation

{data["summary"]["recommendation"]}

## 📈 Stage Failure Analysis

| Stage | Failure Rate | Total Runs | Failures |
|-------|--------------|------------|----------|
"""

        for stage in data["failure_trends"]["stage_failures"][:5]:
            markdown += f"| {stage['stage']} | {stage['failure_rate']:.1f}% | {stage['total_runs']} | {stage['failures']} |\n"

        markdown += """
## 🚀 Recent Pipelines

| Pipeline ID | Branch | Status | Duration |
|-------------|--------|--------|----------|
"""

        for pipeline in data["recent_pipelines"][:5]:
            status_emoji = (
                "✅"
                if pipeline["status"] == "success"
                else "❌" if pipeline["status"] == "failure" else "⏭️"
            )
            duration = f"{pipeline['duration_minutes']}m" if pipeline["duration_minutes"] else "N/A"
            markdown += f"| {pipeline['pipeline_id'][:12]}... | {pipeline['branch']} | {status_emoji} {pipeline['status']} | {duration} |\n"

        markdown += "\n---\n*Generated by PIPELINE-ARCHITECT Metrics*"

        return markdown


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pipeline Metrics Collector")
    parser.add_argument("--record", type=str, help="Record pipeline data from JSON file")
    parser.add_argument(
        "--report",
        choices=["json", "markdown"],
        default="json",
        help="Generate metrics report",
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--health", action="store_true", help="Show health score only")

    args = parser.parse_args()

    metrics = PipelineMetrics()

    if args.record:
        # Record pipeline data
        try:
            with open(args.record, "r") as f:
                pipeline_data = json.load(f)

            if metrics.record_pipeline_run(pipeline_data):
                print(f"✅ Pipeline data recorded: {pipeline_data.get('pipeline_id')}")
            else:
                print("❌ Failed to record pipeline data")
        except Exception as e:
            print(f"❌ Error reading pipeline data: {e}")

    elif args.health:
        # Show health score
        health = metrics.get_pipeline_health_score()
        print(f"Pipeline Health Score: {health['health_score']}/100")
        print(f"Grade: {metrics._get_health_grade(health['health_score'])}")
        print(f"Trend: {health['trend_direction'].title()}")

    else:
        # Generate report
        report = metrics.generate_metrics_report(args.report)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(report)
            print(f"✅ Report saved to: {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()
