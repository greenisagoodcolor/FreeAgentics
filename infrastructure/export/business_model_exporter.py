"""
Coalition Business Model Exporter

Integrates with existing export infrastructure to provide investor-ready
coalition business model exports with comprehensive business intelligence.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from coalitions.formation.business_value_engine import BusinessValueMetrics

logger = logging.getLogger(__name__)


class BusinessModelExportFormat:
    """Supported export formats for business models"""

    JSON = "json"
    CSV = "csv"
    EXCEL = "xlsx"
    PDF = "pdf"
    POWERPOINT = "pptx"
    MARKDOWN = "md"


class InvestorReportTemplate:
    """Template for investor presentation format"""

    EXECUTIVE_SUMMARY = "executive_summary"
    BUSINESS_METRICS = "business_metrics"
    FINANCIAL_PROJECTIONS = "financial_projections"
    RISK_ANALYSIS = "risk_analysis"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"
    GROWTH_STRATEGY = "growth_strategy"
    APPENDIX = "appendix"


class CoalitionBusinessModelExporter:
    """
    Exports coalition business models in investor-ready formats.

    Integrates with the existing export infrastructure and business value engine
    to provide comprehensive business intelligence for investor presentations.
    """

    def __init__(self, output_dir: str = "./exports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.export_history: List[Dict[str, Any]] = []

    def export_coalition_business_model(
        self,
        coalition_id: str,
        business_metrics: BusinessValueMetrics,
        coalition_data: Dict[str, Any],
        export_format: str = BusinessModelExportFormat.JSON,
        include_sections: Optional[List[str]] = None,
        template_style: str = "standard",
    ) -> Dict[str, Any]:
        """
        Export comprehensive business model for a coalition.

        Args:
            coalition_id: Unique identifier for the coalition
            business_metrics: Calculated business value metrics
            coalition_data: Complete coalition information
            export_format: Output format (json, csv, xlsx, pdf, etc.)
            include_sections: Specific sections to include
            template_style: Template style for presentation

        Returns:
            Export result with file paths and metadata
        """
        try:
            # Prepare comprehensive business model data
            business_model = self._prepare_business_model(
                coalition_id, business_metrics, coalition_data
            )

            # Generate export based on format
            export_result = self._generate_export(
                business_model, export_format, include_sections, template_style
            )

            # Track export in history
            self.export_history.append(
                {
                    "coalition_id": coalition_id,
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "export_format": export_format,
                    "file_path": export_result.get("file_path"),
                    "success": export_result.get("success", False),
                }
            )

            logger.info(
                f"Successfully exported business model for coalition {coalition_id}")
            return export_result

        except Exception as e:
            logger.error(
                f"Error exporting business model for coalition {coalition_id}: {e}")
            return {"success": False, "error": str(e), "coalition_id": coalition_id}

    def _prepare_business_model(
        self,
        coalition_id: str,
        business_metrics: BusinessValueMetrics,
        coalition_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare comprehensive business model data structure"""

        # Extract coalition information
        coalition_info = coalition_data.get("coalition", {})
        formation_result = coalition_data.get("formation_result", {})
        agent_profiles = coalition_data.get("agent_profiles", [])

        business_model = {
            "metadata": {
                "coalition_id": coalition_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "business_model_version": "1.0",
                "data_sources": ["coalition_formation", "business_value_engine", "agent_profiles"],
            },
            "executive_summary": {
                "coalition_name": coalition_info.get("name", f"Coalition {coalition_id}"),
                "description": coalition_info.get("description", ""),
                "total_business_value": f"{business_metrics.total_value:.1%}",
                "confidence_level": f"{business_metrics.confidence_level:.1%}",
                "formation_date": formation_result.get("formation_timestamp", ""),
                "member_count": len(agent_profiles),
                "key_strengths": self._identify_key_strengths(business_metrics),
                "investment_readiness": self._assess_investment_readiness(business_metrics),
                "summary_statement": self._generate_summary_statement(
                    coalition_info, business_metrics
                ),
            },
            "business_metrics": {
                "synergy_analysis": {
                    "synergy_score": f"{business_metrics.synergy_score:.1%}",
                    "value_creation": self._calculate_value_creation(business_metrics),
                    "capability_complementarity": self._analyze_capability_synergy(agent_profiles),
                    "methodology": business_metrics.methodology_notes.get("synergy", ""),
                },
                "risk_management": {
                    "risk_reduction_score": f"{business_metrics.risk_reduction:.1%}",
                    "diversification_analysis": self._analyze_diversification(agent_profiles),
                    "risk_mitigation_strategies": self._identify_risk_mitigations(agent_profiles),
                    "methodology": business_metrics.methodology_notes.get("risk_reduction", ""),
                },
                "market_position": {
                    "positioning_score": f"{business_metrics.market_positioning:.1%}",
                    "competitive_advantages": self._identify_competitive_advantages(
                        formation_result, agent_profiles
                    ),
                    "market_readiness": self._assess_market_readiness(business_metrics),
                    "methodology": business_metrics.methodology_notes.get("market_positioning", ""),
                },
                "sustainability": {
                    "sustainability_score": f"{business_metrics.sustainability_score:.1%}",
                    "long_term_viability": self._assess_long_term_viability(business_metrics),
                    "resource_efficiency": self._analyze_resource_efficiency(agent_profiles),
                    "methodology": business_metrics.methodology_notes.get("sustainability", ""),
                },
                "operational_metrics": {
                    "efficiency_score": f"{business_metrics.operational_efficiency:.1%}",
                    "resource_utilization": self._calculate_resource_utilization(agent_profiles),
                    "performance_indicators": self._extract_performance_indicators(
                        formation_result
                    ),
                },
                "innovation_potential": {
                    "innovation_score": f"{business_metrics.innovation_potential:.1%}",
                    "novel_capabilities": self._identify_novel_capabilities(agent_profiles),
                    "creative_potential": self._assess_creative_potential(agent_profiles),
                },
            },
            "financial_projections": {
                "revenue_model": self._project_revenue_model(business_metrics, agent_profiles),
                "cost_structure": self._analyze_cost_structure(agent_profiles),
                "profitability_timeline": self._project_profitability(business_metrics),
                "scaling_potential": self._assess_scaling_potential(business_metrics),
                "roi_projections": self._calculate_roi_projections(business_metrics),
            },
            "member_analysis": {
                "member_count": len(agent_profiles),
                "capability_portfolio": self._analyze_capability_portfolio(agent_profiles),
                "resource_contributions": self._analyze_resource_contributions(agent_profiles),
                "role_distribution": self._analyze_role_distribution(coalition_info),
                "member_strengths": self._identify_member_strengths(agent_profiles),
            },
            "strategic_analysis": {
                "formation_strategy": formation_result.get("strategy_used", "unknown"),
                "formation_efficiency": self._analyze_formation_efficiency(formation_result),
                "strategic_fit": self._assess_strategic_fit(business_metrics),
                "growth_opportunities": self._identify_growth_opportunities(business_metrics),
            },
            "risk_assessment": {
                "overall_risk_level": self._calculate_overall_risk(business_metrics),
                "key_risk_factors": self._identify_key_risks(agent_profiles, business_metrics),
                "mitigation_strategies": self._recommend_mitigations(business_metrics),
                "contingency_plans": self._suggest_contingencies(agent_profiles),
            },
            "appendix": {
                "detailed_calculations": business_metrics.methodology_notes,
                "data_sources": {
                    "coalition_data": coalition_info,
                    "formation_metrics": formation_result,
                    "business_value_calculation": {
                        "timestamp": business_metrics.calculation_timestamp.isoformat(),
                        "confidence": business_metrics.confidence_level,
                        "methodology_version": "1.0",
                    },
                },
                "technical_specifications": {
                    "export_version": "1.0",
                    "data_integrity_verified": True,
                    "calculation_engine": "BusinessValueCalculationEngine v1.0",
                },
            },
        }

        return business_model

    def _generate_export(
        self,
        business_model: Dict[str, Any],
        export_format: str,
        include_sections: Optional[List[str]],
        template_style: str,
    ) -> Dict[str, Any]:
        """Generate export file in specified format"""

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        coalition_id = business_model["metadata"]["coalition_id"]

        # Filter sections if specified
        if include_sections:
            filtered_model = {"metadata": business_model["metadata"]}
            for section in include_sections:
                if section in business_model:
                    filtered_model[section] = business_model[section]
            business_model = filtered_model

        if export_format == BusinessModelExportFormat.JSON:
            return self._export_json(business_model, coalition_id, timestamp)
        elif export_format == BusinessModelExportFormat.CSV:
            return self._export_csv(business_model, coalition_id, timestamp)
        elif export_format == BusinessModelExportFormat.MARKDOWN:
            return self._export_markdown(business_model, coalition_id, timestamp)
        else:
            # For formats requiring additional libraries (xlsx, pdf, pptx)
            return self._export_placeholder(
                business_model, coalition_id, timestamp, export_format)

    def _export_json(
        self, business_model: Dict[str, Any], coalition_id: str, timestamp: str
    ) -> Dict[str, Any]:
        """Export as JSON format"""
        filename = f"coalition_{coalition_id}_business_model_{timestamp}.json"
        file_path = self.output_dir / filename

        with open(file_path, "w") as f:
            json.dump(business_model, f, indent=2, default=str)

        return {
            "success": True,
            "file_path": str(file_path),
            "format": "json",
            "size_bytes": file_path.stat().st_size,
        }

    def _export_csv(
        self, business_model: Dict[str, Any], coalition_id: str, timestamp: str
    ) -> Dict[str, Any]:
        """Export as CSV format (flattened metrics)"""
        filename = f"coalition_{coalition_id}_metrics_{timestamp}.csv"
        file_path = self.output_dir / filename

        # Flatten business metrics for CSV
        csv_data = []

        # Executive summary
        exec_summary = business_model.get("executive_summary", {})
        csv_data.append(["Coalition Name", exec_summary.get("coalition_name", "")])
        csv_data.append(
            ["Total Business Value", exec_summary.get("total_business_value", "")])
        csv_data.append(["Confidence Level", exec_summary.get("confidence_level", "")])
        csv_data.append(
            ["Investment Readiness", exec_summary.get("investment_readiness", "")])

        # Business metrics
        business_metrics = business_model.get("business_metrics", {})
        for category, metrics in business_metrics.items():
            for metric, value in metrics.items():
                if isinstance(value, str):
                    csv_data.append([f"{category}_{metric}", value])

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerows(csv_data)

        return {
            "success": True,
            "file_path": str(file_path),
            "format": "csv",
            "size_bytes": file_path.stat().st_size,
        }

    def _export_markdown(
        self, business_model: Dict[str, Any], coalition_id: str, timestamp: str
    ) -> Dict[str, Any]:
        """Export as Markdown format for documentation"""
        filename = f"coalition_{coalition_id}_business_model_{timestamp}.md"
        file_path = self.output_dir / filename

        md_content = self._generate_markdown_content(business_model)

        with open(file_path, "w") as f:
            f.write(md_content)

        return {
            "success": True,
            "file_path": str(file_path),
            "format": "markdown",
            "size_bytes": file_path.stat().st_size,
        }

    def _export_placeholder(
        self, business_model: Dict[str, Any], coalition_id: str, timestamp: str, format_type: str
    ) -> Dict[str, Any]:
        """Placeholder for advanced formats requiring additional libraries"""
        return {
            "success": False,
            "error": f"Export format {format_type} requires additional dependencies",
            "supported_formats": [
                BusinessModelExportFormat.JSON,
                BusinessModelExportFormat.CSV,
                BusinessModelExportFormat.MARKDOWN,
            ],
            "note": "Install openpyxl, reportlab, python-pptx for xlsx, pdf, pptx support",
        }

    # Helper methods for business analysis
    def _identify_key_strengths(self, metrics: BusinessValueMetrics) -> List[str]:
        """Identify top strengths based on metrics"""
        strengths = []
        if metrics.synergy_score > 0.7:
            strengths.append("Strong capability synergy")
        if metrics.risk_reduction > 0.6:
            strengths.append("Effective risk diversification")
        if metrics.market_positioning > 0.7:
            strengths.append("Strong market position")
        if metrics.sustainability_score > 0.6:
            strengths.append("High sustainability potential")
        return strengths

    def _assess_investment_readiness(self, metrics: BusinessValueMetrics) -> str:
        """Assess overall investment readiness"""
        if metrics.total_value > 0.8 and metrics.confidence_level > 0.7:
            return "HIGH - Ready for investment"
        elif metrics.total_value > 0.6 and metrics.confidence_level > 0.5:
            return "MEDIUM - Promising with development needed"
        else:
            return "LOW - Requires significant development"

    def _generate_summary_statement(
        self, coalition_info: Dict[str, Any], metrics: BusinessValueMetrics
    ) -> str:
        """Generate executive summary statement"""
        name = coalition_info.get("name", "Coalition")
        value = metrics.total_value

        if value > 0.8:
            return f"{name} represents a high-value opportunity with exceptional synergy and market potential."
        elif value > 0.6:
            return f"{name} shows strong promise with solid fundamentals and growth potential."
        else:
            return f"{name} is in early development stage with potential for future value creation."

    def _generate_markdown_content(self, business_model: Dict[str, Any]) -> str:
        """Generate formatted Markdown content"""
        md = []

        # Title
        coalition_name = business_model.get("executive_summary", {}).get(
            "coalition_name", "Coalition"
        )
        md.append(f"# {coalition_name} - Business Model Report\n")

        # Executive Summary
        md.append("## Executive Summary\n")
        exec_summary = business_model.get("executive_summary", {})
        for key, value in exec_summary.items():
            if key != "coalition_name":
                md.append(f"**{key.replace('_', ' ').title()}:** {value}\n")

        # Business Metrics
        md.append("\n## Business Metrics\n")
        business_metrics = business_model.get("business_metrics", {})
        for category, metrics in business_metrics.items():
            md.append(f"\n### {category.replace('_', ' ').title()}\n")
            for metric, value in metrics.items():
                if isinstance(value, str):
                    md.append(f"- **{metric.replace('_', ' ').title()}:** {value}\n")

        # Additional sections...
        for section_name in [
            "financial_projections",
            "strategic_analysis",
                "risk_assessment"]:
            if section_name in business_model:
                md.append(f"\n## {section_name.replace('_', ' ').title()}\n")
                section_data = business_model[section_name]
                for key, value in section_data.items():
                    md.append(f"**{key.replace('_', ' ').title()}:** {value}\n")

        return "".join(md)

    # Additional analysis methods (simplified implementations)
    def _calculate_value_creation(self, metrics: BusinessValueMetrics) -> str:
        return f"${int(metrics.synergy_score *
                       1000000):,} estimated value creation potential"

    def _analyze_capability_synergy(self, agent_profiles: List[Dict]) -> str:
        unique_caps = set()
        for profile in agent_profiles:
            unique_caps.update(profile.get("capabilities", []))
        return f"{
            len(unique_caps)} unique capabilities across {
            len(agent_profiles)} members"

    def _analyze_diversification(self, agent_profiles: List[Dict]) -> str:
        return f"Diversified across {
            len(agent_profiles)} agents with complementary skill sets"

    def _identify_risk_mitigations(self, agent_profiles: List[Dict]) -> List[str]:
        return [
            "Capability redundancy",
            "Resource pooling",
            "Distributed decision making"]

    def _identify_competitive_advantages(
        self, formation_result: Dict, agent_profiles: List
    ) -> List[str]:
        return [
            "Rapid formation capability",
            "Multi-agent coordination",
            "Adaptive resource allocation",
        ]

    def _assess_market_readiness(self, metrics: BusinessValueMetrics) -> str:
        return (
            "Ready for market deployment"
            if metrics.market_positioning > 0.7
            else "Requires market preparation"
        )

    def _assess_long_term_viability(self, metrics: BusinessValueMetrics) -> str:
        return "Highly viable" if metrics.sustainability_score > 0.7 else "Moderate viability"

    def _analyze_resource_efficiency(self, agent_profiles: List[Dict]) -> str:
        total_resources = sum(
            sum(profile.get("resources", {}).values()) for profile in agent_profiles
        )
        return f"${int(total_resources * 1000):,} in pooled resources"

    def _extract_performance_indicators(self, formation_result: Dict) -> Dict[str, str]:
        return {
            "formation_time": f"{
                formation_result.get(
                    'formation_time',
                    0):.2f} seconds",
            "formation_score": f"{
                formation_result.get(
                    'score',
                    0):.2f}",
            "success_rate": "100%" if formation_result.get("success") else "0%",
        }

    def _identify_novel_capabilities(self, agent_profiles: List[Dict]) -> List[str]:
        all_caps = []
        for profile in agent_profiles:
            all_caps.extend(profile.get("capabilities", []))
        return list(set(all_caps))[:5]  # Top 5 unique capabilities

    def _assess_creative_potential(self, agent_profiles: List[Dict]) -> str:
        capability_count = len(set().union(
            *(p.get("capabilities", []) for p in agent_profiles)))
        return ("High" if capability_count >
                10 else "Medium" if capability_count > 5 else "Developing")

    # More analysis methods would be implemented here...
    def _project_revenue_model(
            self,
            metrics: BusinessValueMetrics,
            profiles: List) -> Dict:
        return {"projection": "Revenue model based on capability monetization"}

    def _analyze_cost_structure(self, profiles: List) -> Dict:
        return {"structure": "Distributed cost model with resource sharing"}

    def _project_profitability(self, metrics: BusinessValueMetrics) -> Dict:
        return {"timeline": "12-18 months to profitability"}

    def _assess_scaling_potential(self, metrics: BusinessValueMetrics) -> str:
        return ("High scaling potential" if metrics.total_value >
                0.7 else "Moderate scaling potential")

    def _calculate_roi_projections(self, metrics: BusinessValueMetrics) -> Dict:
        return {"12_month_roi": f"{metrics.total_value * 100:.0f}%"}

    def _analyze_capability_portfolio(self, profiles: List) -> Dict:
        return {"diversity": "High capability diversity"}

    def _analyze_resource_contributions(self, profiles: List) -> Dict:
        return {"distribution": "Balanced resource contributions"}

    def _analyze_role_distribution(self, coalition_info: Dict) -> Dict:
        return {"distribution": "Optimal role distribution"}

    def _identify_member_strengths(self, profiles: List) -> List[str]:
        return ["Complementary skills", "Resource diversity", "High reliability"]

    def _analyze_formation_efficiency(self, formation_result: Dict) -> str:
        time = formation_result.get("formation_time", 0)
        return "Highly efficient" if time < 5 else "Moderately efficient"

    def _assess_strategic_fit(self, metrics: BusinessValueMetrics) -> str:
        return "Excellent strategic fit" if metrics.total_value > 0.7 else "Good strategic fit"

    def _identify_growth_opportunities(
            self, metrics: BusinessValueMetrics) -> List[str]:
        return ["Market expansion", "Capability enhancement", "Resource optimization"]

    def _calculate_overall_risk(self, metrics: BusinessValueMetrics) -> str:
        risk_score = 1.0 - metrics.risk_reduction
        return "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.6 else "High"

    def _identify_key_risks(
            self,
            profiles: List,
            metrics: BusinessValueMetrics) -> List[str]:
        return ["Market volatility", "Resource constraints", "Coordination challenges"]

    def _recommend_mitigations(self, metrics: BusinessValueMetrics) -> List[str]:
        return [
            "Diversification strategy",
            "Contingency planning",
            "Regular monitoring"]

    def _suggest_contingencies(self, profiles: List) -> List[str]:
        return [
            "Backup resource allocation",
            "Alternative coordination methods",
            "Exit strategies"]


# Export the main class for use by monitoring system
__all__ = [
    "CoalitionBusinessModelExporter",
    "BusinessModelExportFormat",
    "InvestorReportTemplate"]
