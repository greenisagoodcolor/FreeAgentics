"""Enhanced GMN repository with version tracking and efficient querying.

Following TDD principles, this repository implements the functionality
required by the failing tests in test_gmn_storage_schema.py.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.gmn_versioned_models import (
    GMNVersionedSpecification,
    GMNVersionStatus,
    GMNVersionTransition,
)

logger = logging.getLogger(__name__)


class GMNVersionedRepository:
    """Enhanced repository for GMN version tracking and efficient querying."""

    def __init__(self, db: Session) -> None:
        """Initialize repository with database session."""
        if db is None:
            raise ValueError("Database session cannot be None")
        self.db = db

    def create_gmn_specification_versioned(
        self,
        agent_id: uuid.UUID,
        specification: str,
        name: str,
        version_number: Optional[int] = None,
        parent_version_id: Optional[uuid.UUID] = None,
        version_metadata: Optional[Dict[str, Any]] = None,
        parsed_data: Optional[Dict[str, Any]] = None,
    ) -> GMNVersionedSpecification:
        """Create a new versioned GMN specification.

        This method implements the versioned creation functionality
        required by the failing tests.
        """
        try:
            # Auto-increment version number if not provided
            if version_number is None:
                max_version = (
                    self.db.query(
                        func.max(GMNVersionedSpecification.version_number)
                    )
                    .filter(GMNVersionedSpecification.agent_id == agent_id)
                    .scalar()
                )
                version_number = (max_version or 0) + 1

            # Calculate checksum for integrity
            checksum = hashlib.sha256(
                specification.encode("utf-8")
            ).hexdigest()

            # Calculate basic metrics from parsed data
            node_count = 0
            edge_count = 0
            complexity_score = None

            if parsed_data:
                node_count = len(parsed_data.get("nodes", []))
                edge_count = len(parsed_data.get("edges", []))
                # Simple complexity: ratio of edges to nodes
                if node_count > 0:
                    complexity_score = min(edge_count / (node_count * 2), 1.0)

            # Create the specification
            gmn_spec = GMNVersionedSpecification(
                agent_id=agent_id,
                version_number=version_number,
                parent_version_id=parent_version_id,
                specification_text=specification,
                parsed_specification=parsed_data or {},
                name=name,
                version_metadata=version_metadata or {},
                specification_checksum=checksum,
                node_count=node_count,
                edge_count=edge_count,
                complexity_score=complexity_score,
                status=GMNVersionStatus.DRAFT,
            )

            self.db.add(gmn_spec)
            self.db.commit()
            self.db.refresh(gmn_spec)

            # Record transition
            self._record_transition(
                agent_id=agent_id,
                from_version_id=parent_version_id,
                to_version_id=gmn_spec.id,
                transition_type="create",
                changes_summary={
                    "action": "created_new_version",
                    "version_number": version_number,
                    "node_count": node_count,
                    "edge_count": edge_count,
                },
            )

            logger.debug(
                f"Created versioned GMN specification {gmn_spec.id} v{version_number}"
            )
            return gmn_spec

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create versioned GMN specification: {e}")
            raise

    def create_new_version(
        self,
        parent_specification_id: uuid.UUID,
        specification: str,
        version_metadata: Optional[Dict[str, Any]] = None,
        parsed_data: Optional[Dict[str, Any]] = None,
    ) -> GMNVersionedSpecification:
        """Create a new version from an existing specification."""
        try:
            # Get parent specification
            parent_spec = self.get_specification_by_id(parent_specification_id)
            if not parent_spec:
                raise ValueError(
                    f"Parent specification {parent_specification_id} not found"
                )

            # Generate name for new version
            new_name = f"{parent_spec.name} v{parent_spec.version_number + 1}"

            return self.create_gmn_specification_versioned(
                agent_id=parent_spec.agent_id,
                specification=specification,
                name=new_name,
                parent_version_id=parent_specification_id,
                version_metadata=version_metadata,
                parsed_data=parsed_data,
            )

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create new version: {e}")
            raise

    def get_version_lineage(self, agent_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get version history with parent-child lineage."""
        try:
            # Get all versions for the agent ordered by creation
            versions = (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.agent_id == agent_id)
                .order_by(GMNVersionedSpecification.version_number)
                .all()
            )

            # Build lineage tree
            lineage = []
            version_map = {str(v.id): v for v in versions}

            for version in versions:
                lineage_entry = {
                    "id": str(version.id),
                    "version_number": version.version_number,
                    "name": version.name,
                    "status": version.status.value,
                    "parent_version_id": (
                        str(version.parent_version_id)
                        if version.parent_version_id
                        else None
                    ),
                    "children": [],
                    "created_at": version.created_at.isoformat(),
                    "node_count": version.node_count,
                    "edge_count": version.edge_count,
                }

                # Find children
                children = [
                    v for v in versions if v.parent_version_id == version.id
                ]
                lineage_entry["children"] = [str(c.id) for c in children]

                lineage.append(lineage_entry)

            return lineage

        except SQLAlchemyError as e:
            logger.error(f"Failed to get version lineage: {e}")
            return []

    def rollback_to_version(
        self,
        agent_id: uuid.UUID,
        target_version_id: uuid.UUID,
        rollback_reason: Optional[str] = None,
    ) -> bool:
        """Rollback to a previous version."""
        try:
            # Get target version
            target_version = self.get_specification_by_id(target_version_id)
            if not target_version or target_version.agent_id != agent_id:
                logger.error(
                    f"Target version {target_version_id} not found or invalid agent"
                )
                return False

            # Get current active version
            current_active = self.get_active_specification(agent_id)

            # Deactivate current version
            if current_active:
                current_active.status = GMNVersionStatus.DEPRECATED
                current_active.deprecated_at = datetime.utcnow()

            # Activate target version
            target_version.status = GMNVersionStatus.ACTIVE
            target_version.activated_at = datetime.utcnow()

            # Record rollback transition
            self._record_transition(
                agent_id=agent_id,
                from_version_id=current_active.id if current_active else None,
                to_version_id=target_version_id,
                transition_type="rollback",
                transition_reason=rollback_reason,
                changes_summary={
                    "action": "rollback",
                    "from_version": current_active.version_number
                    if current_active
                    else None,
                    "to_version": target_version.version_number,
                    "reason": rollback_reason,
                },
            )

            self.db.commit()
            logger.debug(
                f"Rolled back agent {agent_id} to version {target_version.version_number}"
            )
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to rollback to version: {e}")
            return False

    def compare_versions(
        self,
        version_a_id: uuid.UUID,
        version_b_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Compare two versions to show differences."""
        try:
            version_a = self.get_specification_by_id(version_a_id)
            version_b = self.get_specification_by_id(version_b_id)

            if not version_a or not version_b:
                raise ValueError("One or both versions not found")

            # Basic comparison
            diff = {
                "version_a": {
                    "id": str(version_a.id),
                    "version_number": version_a.version_number,
                    "name": version_a.name,
                    "node_count": version_a.node_count,
                    "edge_count": version_a.edge_count,
                    "complexity_score": version_a.complexity_score,
                    "checksum": version_a.specification_checksum,
                },
                "version_b": {
                    "id": str(version_b.id),
                    "version_number": version_b.version_number,
                    "name": version_b.name,
                    "node_count": version_b.node_count,
                    "edge_count": version_b.edge_count,
                    "complexity_score": version_b.complexity_score,
                    "checksum": version_b.specification_checksum,
                },
                "differences": {
                    "node_count_diff": version_b.node_count
                    - version_a.node_count,
                    "edge_count_diff": version_b.edge_count
                    - version_a.edge_count,
                    "complexity_diff": (version_b.complexity_score or 0)
                    - (version_a.complexity_score or 0),
                    "specification_changed": version_a.specification_checksum
                    != version_b.specification_checksum,
                },
                "compatibility": version_a.is_compatible_with(version_b),
            }

            return diff

        except SQLAlchemyError as e:
            logger.error(f"Failed to compare versions: {e}")
            return {}

    def search_by_parsed_content(
        self,
        agent_id: uuid.UUID,
        node_type: Optional[str] = None,
        property_filter: Optional[Dict[str, Any]] = None,
    ) -> List[GMNVersionedSpecification]:
        """Search GMN specifications by parsed content."""
        try:
            query = self.db.query(GMNVersionedSpecification).filter(
                GMNVersionedSpecification.agent_id == agent_id
            )

            # Filter by node type if specified
            if node_type:
                # Use JSON path query to find specifications with nodes of specified type
                query = query.filter(
                    GMNVersionedSpecification.parsed_specification["nodes"].op(
                        "@>"
                    )([{"type": node_type}])
                )

            # Apply property filters if specified
            if property_filter:
                for key, value in property_filter.items():
                    # Simple property filtering - can be enhanced for complex queries
                    query = query.filter(
                        GMNVersionedSpecification.parsed_specification.op(
                            "->"
                        )(key).astext
                        == str(value)
                    )

            return query.order_by(
                desc(GMNVersionedSpecification.created_at)
            ).all()

        except SQLAlchemyError as e:
            logger.error(f"Failed to search by parsed content: {e}")
            return []

    def get_by_complexity_range(
        self,
        agent_id: uuid.UUID,
        min_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        complexity_score_range: Optional[Tuple[float, float]] = None,
    ) -> List[GMNVersionedSpecification]:
        """Query specifications by complexity metrics."""
        try:
            query = self.db.query(GMNVersionedSpecification).filter(
                GMNVersionedSpecification.agent_id == agent_id
            )

            if min_nodes is not None:
                query = query.filter(
                    GMNVersionedSpecification.node_count >= min_nodes
                )

            if max_edges is not None:
                query = query.filter(
                    GMNVersionedSpecification.edge_count <= max_edges
                )

            if complexity_score_range:
                min_score, max_score = complexity_score_range
                query = query.filter(
                    and_(
                        GMNVersionedSpecification.complexity_score
                        >= min_score,
                        GMNVersionedSpecification.complexity_score
                        <= max_score,
                    )
                )

            return query.order_by(
                desc(GMNVersionedSpecification.complexity_score)
            ).all()

        except SQLAlchemyError as e:
            logger.error(f"Failed to query by complexity: {e}")
            return []

    def get_by_time_range(
        self,
        agent_id: uuid.UUID,
        start_time: datetime,
        end_time: datetime,
        include_inactive: bool = False,
    ) -> List[GMNVersionedSpecification]:
        """Query specifications within time ranges."""
        try:
            query = (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.agent_id == agent_id)
                .filter(GMNVersionedSpecification.created_at >= start_time)
                .filter(GMNVersionedSpecification.created_at <= end_time)
            )

            if not include_inactive:
                query = query.filter(
                    GMNVersionedSpecification.status == GMNVersionStatus.ACTIVE
                )

            return query.order_by(
                desc(GMNVersionedSpecification.created_at)
            ).all()

        except SQLAlchemyError as e:
            logger.error(f"Failed to query by time range: {e}")
            return []

    def get_detailed_statistics(
        self,
        agent_id: uuid.UUID,
        time_window_days: int = 30,
        include_trends: bool = False,
    ) -> Dict[str, Any]:
        """Get detailed statistics about specifications."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=time_window_days)

            # Base query for specifications in time window
            specs_query = (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.agent_id == agent_id)
                .filter(GMNVersionedSpecification.created_at >= cutoff)
            )

            specifications = specs_query.all()

            # Calculate basic statistics
            total_specs = len(specifications)
            active_specs = len(
                [
                    s
                    for s in specifications
                    if s.status == GMNVersionStatus.ACTIVE
                ]
            )

            # Complexity statistics
            complexities = [
                s.complexity_score
                for s in specifications
                if s.complexity_score is not None
            ]
            avg_complexity = (
                sum(complexities) / len(complexities) if complexities else 0
            )

            # Node/edge statistics
            node_counts = [s.node_count for s in specifications]
            edge_counts = [s.edge_count for s in specifications]

            stats = {
                "total_specifications": total_specs,
                "active_specifications": active_specs,
                "avg_complexity": avg_complexity,
                "avg_nodes": sum(node_counts) / len(node_counts)
                if node_counts
                else 0,
                "avg_edges": sum(edge_counts) / len(edge_counts)
                if edge_counts
                else 0,
                "max_version_number": max(
                    (s.version_number for s in specifications), default=0
                ),
                "status_distribution": {
                    status.value: len(
                        [s for s in specifications if s.status == status]
                    )
                    for status in GMNVersionStatus
                },
                "time_window_days": time_window_days,
            }

            # Add trends if requested
            if include_trends and total_specs > 1:
                # Simple trend: specifications per week
                weekly_counts = {}
                for spec in specifications:
                    week_start = spec.created_at.strftime("%Y-W%U")
                    weekly_counts[week_start] = (
                        weekly_counts.get(week_start, 0) + 1
                    )

                stats["trends"] = {
                    "weekly_creation_counts": weekly_counts,
                    "creation_trend": "increasing"
                    if len(weekly_counts) > 1
                    else "stable",
                }

            return stats

        except SQLAlchemyError as e:
            logger.error(f"Failed to get detailed statistics: {e}")
            return {}

    def validate_data_integrity(
        self,
        agent_id: uuid.UUID,
        check_version_consistency: bool = True,
        check_parsed_data_sync: bool = True,
    ) -> Dict[str, Any]:
        """Validate GMN specification data integrity."""
        try:
            integrity_report = {
                "agent_id": str(agent_id),
                "checks_performed": [],
                "issues_found": [],
                "warnings": [],
                "is_valid": True,
            }

            specifications = (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.agent_id == agent_id)
                .all()
            )

            if check_version_consistency:
                integrity_report["checks_performed"].append(
                    "version_consistency"
                )

                # Check for version number gaps
                version_numbers = sorted(
                    [s.version_number for s in specifications]
                )
                for i in range(1, len(version_numbers)):
                    if version_numbers[i] - version_numbers[i - 1] > 1:
                        integrity_report["warnings"].append(
                            f"Version gap detected: {version_numbers[i - 1]} -> {version_numbers[i]}"
                        )

                # Check for orphaned parent references
                spec_ids = {s.id for s in specifications}
                for spec in specifications:
                    if (
                        spec.parent_version_id
                        and spec.parent_version_id not in spec_ids
                    ):
                        integrity_report["issues_found"].append(
                            f"Orphaned parent reference in version {spec.version_number}: {spec.parent_version_id}"
                        )
                        integrity_report["is_valid"] = False

            if check_parsed_data_sync:
                integrity_report["checks_performed"].append("parsed_data_sync")

                # Check if checksum matches specification text
                for spec in specifications:
                    if spec.specification_checksum:
                        actual_checksum = hashlib.sha256(
                            spec.specification_text.encode("utf-8")
                        ).hexdigest()
                        if actual_checksum != spec.specification_checksum:
                            integrity_report["issues_found"].append(
                                f"Checksum mismatch in version {spec.version_number}"
                            )
                            integrity_report["is_valid"] = False

            return integrity_report

        except SQLAlchemyError as e:
            logger.error(f"Failed to validate data integrity: {e}")
            return {"error": str(e), "is_valid": False}

    def detect_orphaned_versions(self, agent_id: uuid.UUID) -> List[uuid.UUID]:
        """Detect orphaned versions without proper lineage."""
        try:
            specifications = (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.agent_id == agent_id)
                .all()
            )

            spec_ids = {s.id for s in specifications}
            orphaned = []

            for spec in specifications:
                if (
                    spec.parent_version_id
                    and spec.parent_version_id not in spec_ids
                ):
                    orphaned.append(spec.id)

            return orphaned

        except SQLAlchemyError as e:
            logger.error(f"Failed to detect orphaned versions: {e}")
            return []

    def repair_version_lineage(
        self,
        agent_id: uuid.UUID,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Repair broken version lineage."""
        try:
            repair_result = {
                "agent_id": str(agent_id),
                "dry_run": dry_run,
                "repairs_needed": [],
                "repairs_applied": [],
            }

            orphaned_ids = self.detect_orphaned_versions(agent_id)

            for orphaned_id in orphaned_ids:
                orphaned_spec = self.get_specification_by_id(orphaned_id)
                if not orphaned_spec:
                    continue

                # Find potential parent (previous version by creation time)
                potential_parent = (
                    self.db.query(GMNVersionedSpecification)
                    .filter(GMNVersionedSpecification.agent_id == agent_id)
                    .filter(
                        GMNVersionedSpecification.created_at
                        < orphaned_spec.created_at
                    )
                    .order_by(desc(GMNVersionedSpecification.created_at))
                    .first()
                )

                if potential_parent:
                    repair_action = {
                        "orphaned_version_id": str(orphaned_id),
                        "suggested_parent_id": str(potential_parent.id),
                        "action": "link_to_parent",
                    }
                    repair_result["repairs_needed"].append(repair_action)

                    if not dry_run:
                        orphaned_spec.parent_version_id = potential_parent.id
                        repair_result["repairs_applied"].append(repair_action)

            if not dry_run and repair_result["repairs_applied"]:
                self.db.commit()

            return repair_result

        except SQLAlchemyError as e:
            if not dry_run:
                self.db.rollback()
            logger.error(f"Failed to repair version lineage: {e}")
            return {"error": str(e)}

    # Helper methods
    def get_specification_by_id(
        self, spec_id: uuid.UUID
    ) -> Optional[GMNVersionedSpecification]:
        """Get specification by ID."""
        try:
            return (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.id == spec_id)
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"Failed to get specification by ID: {e}")
            return None

    def get_active_specification(
        self, agent_id: uuid.UUID
    ) -> Optional[GMNVersionedSpecification]:
        """Get active specification for agent."""
        try:
            return (
                self.db.query(GMNVersionedSpecification)
                .filter(GMNVersionedSpecification.agent_id == agent_id)
                .filter(
                    GMNVersionedSpecification.status == GMNVersionStatus.ACTIVE
                )
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"Failed to get active specification: {e}")
            return None

    def _record_transition(
        self,
        agent_id: uuid.UUID,
        from_version_id: Optional[uuid.UUID],
        to_version_id: uuid.UUID,
        transition_type: str,
        transition_reason: Optional[str] = None,
        changes_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a version transition."""
        try:
            transition = GMNVersionTransition(
                agent_id=agent_id,
                from_version_id=from_version_id,
                to_version_id=to_version_id,
                transition_type=transition_type,
                transition_reason=transition_reason,
                changes_summary=changes_summary or {},
            )

            self.db.add(transition)
            # Note: Commit is handled by calling method

        except SQLAlchemyError as e:
            logger.error(f"Failed to record transition: {e}")
            # Don't raise here as this is a supporting operation
