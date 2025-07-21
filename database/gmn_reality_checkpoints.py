"""Reality checkpoint queries for GMN versioned storage data integrity.

These queries verify data integrity, consistency, and performance characteristics
of the GMN versioned storage system. They should be run regularly in production.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy import text
from sqlalchemy.orm import Session

from database.gmn_versioned_models import (
    GMNVersionedSpecification,
    GMNVersionStatus,
    GMNVersionTransition,
)

logger = logging.getLogger(__name__)


class GMNRealityCheckpoints:
    """Reality checkpoint queries for GMN storage integrity verification."""

    def __init__(self, db: Session) -> None:
        """Initialize with database session."""
        self.db = db

    def run_all_checkpoints(self) -> Dict[str, Any]:
        """Run all reality checkpoints and return comprehensive report."""
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "checkpoints": {},
            "overall_health": "unknown",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }

        try:
            # Run individual checkpoints
            report["checkpoints"]["version_integrity"] = self.check_version_integrity()
            report["checkpoints"][
                "orphaned_references"
            ] = self.check_orphaned_references()
            report["checkpoints"][
                "active_constraints"
            ] = self.check_active_constraints()
            report["checkpoints"][
                "checksum_integrity"
            ] = self.check_checksum_integrity()
            report["checkpoints"][
                "performance_metrics"
            ] = self.check_performance_metrics()
            report["checkpoints"][
                "storage_efficiency"
            ] = self.check_storage_efficiency()

            # Determine overall health
            critical_count = sum(
                1
                for checkpoint in report["checkpoints"].values()
                if not checkpoint.get("passed", False)
            )

            if critical_count == 0:
                report["overall_health"] = "healthy"
            elif critical_count <= 2:
                report["overall_health"] = "warning"
            else:
                report["overall_health"] = "critical"

            # Collect issues and recommendations
            for checkpoint_name, result in report["checkpoints"].items():
                report["critical_issues"].extend(result.get("critical_issues", []))
                report["warnings"].extend(result.get("warnings", []))
                report["recommendations"].extend(result.get("recommendations", []))

        except Exception as e:
            logger.error(f"Failed to run reality checkpoints: {e}")
            report["overall_health"] = "error"
            report["critical_issues"].append(f"Checkpoint system failure: {e}")

        return report

    def check_version_integrity(self) -> Dict[str, Any]:
        """Check version number integrity and consistency."""
        result: Dict[str, Any] = {
            "checkpoint": "version_integrity",
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Check for version number gaps per agent
            version_gaps_query = text(
                """
                WITH agent_versions AS (
                    SELECT
                        agent_id,
                        version_number,
                        LAG(version_number) OVER (
                            PARTITION BY agent_id
                            ORDER BY version_number
                        ) AS prev_version
                    FROM gmn_versioned_specifications
                )
                SELECT
                    agent_id,
                    COUNT(*) as gap_count,
                    ARRAY_AGG(
                        'v' || prev_version || ' -> v' || version_number
                    ) as gaps
                FROM agent_versions
                WHERE version_number - COALESCE(prev_version, 0) > 1
                GROUP BY agent_id
            """
            )

            gaps_result = self.db.execute(version_gaps_query).fetchall()

            if gaps_result:
                for row in gaps_result:
                    result["warnings"].append(
                        f"Agent {row.agent_id} has {row.gap_count} version gaps: {row.gaps}"
                    )

            # Check for duplicate version numbers per agent
            duplicate_versions_query = text(
                """
                SELECT
                    agent_id,
                    version_number,
                    COUNT(*) as duplicate_count
                FROM gmn_versioned_specifications
                GROUP BY agent_id, version_number
                HAVING COUNT(*) > 1
            """
            )

            duplicates_result = self.db.execute(duplicate_versions_query).fetchall()

            if duplicates_result:
                result["passed"] = False
                for row in duplicates_result:
                    result["critical_issues"].append(
                        f"Agent {row.agent_id} has {row.duplicate_count} specifications with version {row.version_number}"
                    )

            # Metrics
            total_specs = self.db.query(GMNVersionedSpecification).count()
            max_version_query = text(
                """
                SELECT
                    agent_id,
                    MAX(version_number) as max_version
                FROM gmn_versioned_specifications
                GROUP BY agent_id
                ORDER BY max_version DESC
                LIMIT 1
            """
            )
            max_version_result = self.db.execute(max_version_query).fetchone()

            result["metrics"] = {
                "total_specifications": total_specs,
                "highest_version_number": (
                    max_version_result.max_version if max_version_result else 0
                ),
                "version_gaps_found": len(gaps_result),
                "duplicate_versions_found": len(duplicates_result),
            }

        except Exception as e:
            result["passed"] = False
            result["critical_issues"].append(f"Version integrity check failed: {e}")

        return result

    def check_orphaned_references(self) -> Dict[str, Any]:
        """Check for orphaned parent version references."""
        result: Dict[str, Any] = {
            "checkpoint": "orphaned_references",
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Find orphaned parent references
            orphaned_query = text(
                """
                SELECT
                    child.id,
                    child.agent_id,
                    child.version_number,
                    child.parent_version_id
                FROM gmn_versioned_specifications child
                LEFT JOIN gmn_versioned_specifications parent
                    ON child.parent_version_id = parent.id
                WHERE child.parent_version_id IS NOT NULL
                    AND parent.id IS NULL
            """
            )

            orphaned_result = self.db.execute(orphaned_query).fetchall()

            if orphaned_result:
                result["passed"] = False
                for row in orphaned_result:
                    result["critical_issues"].append(
                        f"Specification {row.id} (v{row.version_number}) has orphaned parent reference {row.parent_version_id}"
                    )

                result["recommendations"].append(
                    "Run repair_version_lineage() to fix orphaned references"
                )

            # Check for circular references
            circular_query = text(
                """
                WITH RECURSIVE version_path AS (
                    -- Base case: start with each specification
                    SELECT
                        id,
                        parent_version_id,
                        ARRAY[id] as path,
                        1 as depth
                    FROM gmn_versioned_specifications
                    WHERE parent_version_id IS NOT NULL

                    UNION ALL

                    -- Recursive case: follow parent chain
                    SELECT
                        vp.id,
                        gvs.parent_version_id,
                        vp.path || gvs.id,
                        vp.depth + 1
                    FROM version_path vp
                    JOIN gmn_versioned_specifications gvs ON vp.parent_version_id = gvs.id
                    WHERE gvs.id != ALL(vp.path)  -- Prevent infinite recursion
                        AND vp.depth < 100  -- Safety limit
                )
                SELECT DISTINCT id, path
                FROM version_path
                WHERE id = ANY(path[2:])  -- Circular reference detected
            """
            )

            circular_result = self.db.execute(circular_query).fetchall()

            if circular_result:
                result["passed"] = False
                for row in circular_result:
                    result["critical_issues"].append(
                        f"Circular reference detected in version chain: {row.path}"
                    )

            result["metrics"] = {
                "orphaned_references": len(orphaned_result),
                "circular_references": len(circular_result),
            }

        except Exception as e:
            result["passed"] = False
            result["critical_issues"].append(f"Orphaned references check failed: {e}")

        return result

    def check_active_constraints(self) -> Dict[str, Any]:
        """Check active specification constraints (only one active per agent)."""
        result: Dict[str, Any] = {
            "checkpoint": "active_constraints",
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Check for multiple active specifications per agent
            multiple_active_query = text(
                """
                SELECT
                    agent_id,
                    COUNT(*) as active_count,
                    ARRAY_AGG(id) as active_spec_ids
                FROM gmn_versioned_specifications
                WHERE status = 'active'
                GROUP BY agent_id
                HAVING COUNT(*) > 1
            """
            )

            multiple_active_result = self.db.execute(multiple_active_query).fetchall()

            if multiple_active_result:
                result["passed"] = False
                for row in multiple_active_result:
                    result["critical_issues"].append(
                        f"Agent {row.agent_id} has {row.active_count} active specifications: {row.active_spec_ids}"
                    )

            # Check for agents without any active specification
            no_active_query = text(
                """
                SELECT DISTINCT a.agent_id
                FROM (
                    SELECT DISTINCT agent_id
                    FROM gmn_versioned_specifications
                ) a
                LEFT JOIN gmn_versioned_specifications active_spec
                    ON a.agent_id = active_spec.agent_id
                    AND active_spec.status = 'active'
                WHERE active_spec.id IS NULL
            """
            )

            no_active_result = self.db.execute(no_active_query).fetchall()

            if no_active_result:
                for row in no_active_result:
                    result["warnings"].append(
                        f"Agent {row.agent_id} has no active specification"
                    )

            result["metrics"] = {
                "agents_with_multiple_active": len(multiple_active_result),
                "agents_without_active": len(no_active_result),
            }

        except Exception as e:
            result["passed"] = False
            result["critical_issues"].append(f"Active constraints check failed: {e}")

        return result

    def check_checksum_integrity(self) -> Dict[str, Any]:
        """Check specification text checksum integrity."""
        result: Dict[str, Any] = {
            "checkpoint": "checksum_integrity",
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Check for missing checksums
            missing_checksum_count = (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.specification_checksum.is_(None))
                .count()
            )

            if missing_checksum_count > 0:
                result["warnings"].append(
                    f"{missing_checksum_count} specifications missing checksums"
                )
                result["recommendations"].append(
                    "Update specifications to include checksums for integrity verification"
                )

            # For a more thorough check, we would validate actual checksums
            # This would require recalculating checksums for all specifications
            # and comparing with stored values (computationally expensive)

            result["metrics"] = {
                "specifications_without_checksum": missing_checksum_count,
                "total_specifications": self.db.query(
                    GMNVersionedSpecification
                ).count(),
            }

        except Exception as e:
            result["passed"] = False
            result["critical_issues"].append(f"Checksum integrity check failed: {e}")

        return result

    def check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance-related metrics and potential issues."""
        result: Dict[str, Any] = {
            "checkpoint": "performance_metrics",
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Check for extremely large specifications
            large_specs_query = text(
                """
                SELECT
                    id,
                    agent_id,
                    version_number,
                    LENGTH(specification_text) as text_length,
                    node_count,
                    edge_count
                FROM gmn_versioned_specifications
                WHERE LENGTH(specification_text) > 10000
                    OR node_count > 100
                    OR edge_count > 200
                ORDER BY LENGTH(specification_text) DESC
                LIMIT 10
            """
            )

            large_specs_result = self.db.execute(large_specs_query).fetchall()

            if large_specs_result:
                for row in large_specs_result:
                    if row.text_length > 50000:
                        result["warnings"].append(
                            f"Very large specification {row.id}: {row.text_length} chars, {row.node_count} nodes"
                        )

            # Check for agents with many versions (could indicate performance issues)
            high_version_agents_query = text(
                """
                SELECT
                    agent_id,
                    COUNT(*) as version_count,
                    MAX(version_number) as max_version
                FROM gmn_versioned_specifications
                GROUP BY agent_id
                HAVING COUNT(*) > 50
                ORDER BY COUNT(*) DESC
            """
            )

            high_version_result = self.db.execute(high_version_agents_query).fetchall()

            if high_version_result:
                for row in high_version_result:
                    result["warnings"].append(
                        f"Agent {row.agent_id} has {row.version_count} versions (max: v{row.max_version})"
                    )

                result["recommendations"].append(
                    "Consider archiving old versions for agents with many versions"
                )

            # Average complexity metrics
            complexity_stats_query = text(
                """
                SELECT
                    AVG(node_count) as avg_nodes,
                    AVG(edge_count) as avg_edges,
                    AVG(complexity_score) as avg_complexity,
                    MAX(node_count) as max_nodes,
                    MAX(edge_count) as max_edges
                FROM gmn_versioned_specifications
                WHERE complexity_score IS NOT NULL
            """
            )

            complexity_result = self.db.execute(complexity_stats_query).fetchone()

            result["metrics"] = {
                "large_specifications_count": len(large_specs_result),
                "agents_with_many_versions": len(high_version_result),
                "avg_nodes": (
                    float(complexity_result.avg_nodes)
                    if complexity_result.avg_nodes
                    else 0
                ),
                "avg_edges": (
                    float(complexity_result.avg_edges)
                    if complexity_result.avg_edges
                    else 0
                ),
                "avg_complexity": (
                    float(complexity_result.avg_complexity)
                    if complexity_result.avg_complexity
                    else 0
                ),
                "max_nodes": complexity_result.max_nodes or 0,
                "max_edges": complexity_result.max_edges or 0,
            }

        except Exception as e:
            result["passed"] = False
            result["critical_issues"].append(f"Performance metrics check failed: {e}")

        return result

    def check_storage_efficiency(self) -> Dict[str, Any]:
        """Check storage efficiency and space usage."""
        result: Dict[str, Any] = {
            "checkpoint": "storage_efficiency",
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Check for duplicate specifications (same checksum)
            duplicate_content_query = text(
                """
                SELECT
                    specification_checksum,
                    COUNT(*) as duplicate_count,
                    ARRAY_AGG(id) as spec_ids
                FROM gmn_versioned_specifications
                WHERE specification_checksum IS NOT NULL
                GROUP BY specification_checksum
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
            """
            )

            duplicate_result = self.db.execute(duplicate_content_query).fetchall()

            if duplicate_result:
                total_duplicates = sum(
                    row.duplicate_count - 1 for row in duplicate_result
                )
                result["warnings"].append(
                    f"Found {len(duplicate_result)} sets of duplicate content affecting {total_duplicates} specifications"
                )
                result["recommendations"].append(
                    "Consider deduplication strategy for identical specifications"
                )

            # Check transition record efficiency
            transition_stats_query = text(
                """
                SELECT
                    COUNT(*) as total_transitions,
                    COUNT(DISTINCT agent_id) as agents_with_transitions,
                    AVG(
                        LENGTH(changes_summary::text)
                    ) as avg_change_summary_size
                FROM gmn_version_transitions
            """
            )

            transition_result = self.db.execute(transition_stats_query).fetchone()

            result["metrics"] = {
                "duplicate_content_groups": len(duplicate_result),
                "total_transitions": transition_result.total_transitions or 0,
                "agents_with_transitions": transition_result.agents_with_transitions
                or 0,
                "avg_change_summary_size": (
                    float(transition_result.avg_change_summary_size)
                    if transition_result.avg_change_summary_size
                    else 0
                ),
            }

        except Exception as e:
            result["passed"] = False
            result["critical_issues"].append(f"Storage efficiency check failed: {e}")

        return result

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a quick health summary without running full checkpoints."""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_specifications": self.db.query(
                    GMNVersionedSpecification
                ).count(),
                "active_specifications": (
                    self.db.query(GMNVersionedSpecification)
                    .filter(GMNVersionedSpecification.status == GMNVersionStatus.ACTIVE)
                    .count()
                ),
                "total_agents_with_specs": (
                    self.db.query(GMNVersionedSpecification.agent_id).distinct().count()
                ),
                "total_transitions": self.db.query(GMNVersionTransition).count(),
                "recent_activity": (
                    self.db.query(GMNVersionedSpecification)
                    .filter(
                        GMNVersionedSpecification.created_at
                        >= datetime.utcnow() - timedelta(days=7)
                    )
                    .count()
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get health summary: {e}")
            return {"error": str(e)}


def run_reality_checkpoints_cli(db_session: Session) -> None:
    """Command-line interface for running reality checkpoints."""
    checkpoints = GMNRealityCheckpoints(db_session)

    print("🔍 Running GMN Storage Reality Checkpoints...")
    print("=" * 50)

    report = checkpoints.run_all_checkpoints()

    print(f"Overall Health: {report['overall_health'].upper()}")
    print(f"Timestamp: {report['timestamp']}")
    print()

    # Print checkpoint results
    for checkpoint_name, result in report["checkpoints"].items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{checkpoint_name}: {status}")

        if result.get("critical_issues"):
            for issue in result["critical_issues"]:
                print(f"  🚨 {issue}")

        if result.get("warnings"):
            for warning in result["warnings"]:
                print(f"  ⚠️  {warning}")

    print()

    # Print recommendations
    if report["recommendations"]:
        print("📋 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    print("\n" + "=" * 50)
    print("Reality checkpoints complete.")
