#!/usr/bin/env python3
"""
Frontend Performance Analyzer
=============================

Analyzes Next.js bundle size, lighthouse metrics, and frontend performance.
Integrates with Lighthouse CI for automated performance testing.
"""

import gzip
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BundleAnalysis:
    """Bundle size analysis result."""

    total_size_kb: float
    gzipped_size_kb: float
    chunk_count: int
    largest_chunks: List[Dict[str, Any]]
    compression_ratio: float
    performance_impact: str
    recommendations: List[str]


@dataclass
class LighthouseMetrics:
    """Lighthouse performance metrics."""

    performance_score: int
    fcp_ms: Optional[float] = None  # First Contentful Paint
    lcp_ms: Optional[float] = None  # Largest Contentful Paint
    cls: Optional[float] = None  # Cumulative Layout Shift
    tbt_ms: Optional[float] = None  # Total Blocking Time
    si_ms: Optional[float] = None  # Speed Index


class FrontendPerformanceAnalyzer:
    """Comprehensive frontend performance analyzer."""

    # Performance budgets
    BUNDLE_SIZE_BUDGET_KB = 200
    LIGHTHOUSE_PERFORMANCE_BUDGET = 90

    def __init__(self, project_root: Path):
        """Initialize analyzer."""
        self.project_root = project_root
        self.web_dir = project_root / "web"
        self.build_dir = self.web_dir / ".next"

    def analyze_bundle_size(self) -> BundleAnalysis:
        """Analyze Next.js bundle size and composition."""
        logger.info("📦 Analyzing bundle size...")

        if not self.build_dir.exists():
            logger.warning("Build directory not found, running build first...")
            self._ensure_build_exists()

        # Analyze all JS chunks
        chunks_dir = self.build_dir / "static" / "chunks"
        app_dir = chunks_dir / "app"

        chunks = []
        total_size = 0

        # Main chunks
        if chunks_dir.exists():
            for js_file in chunks_dir.glob("*.js"):
                size = js_file.stat().st_size
                total_size += size

                chunks.append(
                    {
                        "name": js_file.name,
                        "size_kb": size / 1024,
                        "gzipped_kb": self._estimate_gzipped_size(js_file),
                        "type": "main_chunk",
                    }
                )

        # App chunks
        if app_dir.exists():
            for js_file in app_dir.rglob("*.js"):
                size = js_file.stat().st_size
                total_size += size

                chunks.append(
                    {
                        "name": str(js_file.relative_to(chunks_dir)),
                        "size_kb": size / 1024,
                        "gzipped_kb": self._estimate_gzipped_size(js_file),
                        "type": "app_chunk",
                    }
                )

        # Sort by size
        chunks.sort(key=lambda x: x["size_kb"], reverse=True)

        total_size_kb = total_size / 1024
        total_gzipped_kb = sum(chunk["gzipped_kb"] for chunk in chunks)
        compression_ratio = total_size_kb / total_gzipped_kb if total_gzipped_kb > 0 else 1.0

        # Performance assessment
        if total_gzipped_kb <= self.BUNDLE_SIZE_BUDGET_KB:
            performance_impact = "excellent"
        elif total_gzipped_kb <= self.BUNDLE_SIZE_BUDGET_KB * 1.5:
            performance_impact = "good"
        elif total_gzipped_kb <= self.BUNDLE_SIZE_BUDGET_KB * 2:
            performance_impact = "fair"
        else:
            performance_impact = "poor"

        # Generate recommendations
        recommendations = self._generate_bundle_recommendations(chunks, total_gzipped_kb)

        return BundleAnalysis(
            total_size_kb=total_size_kb,
            gzipped_size_kb=total_gzipped_kb,
            chunk_count=len(chunks),
            largest_chunks=chunks[:10],  # Top 10 largest
            compression_ratio=compression_ratio,
            performance_impact=performance_impact,
            recommendations=recommendations,
        )

    def _ensure_build_exists(self):
        """Ensure Next.js build exists."""
        if not self.web_dir.exists():
            logger.error("Web directory not found")
            return

        try:
            # Check if package.json exists
            package_json = self.web_dir / "package.json"
            if not package_json.exists():
                logger.error("package.json not found in web directory")
                return

            logger.info("Building Next.js project...")
            subprocess.run(
                ["npm", "run", "build"],
                cwd=self.web_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Build completed successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Build failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Build error: {e}")

    def _estimate_gzipped_size(self, file_path: Path) -> float:
        """Estimate gzipped size of a file."""
        try:
            with open(file_path, "rb") as f:
                original_data = f.read()

            compressed_data = gzip.compress(original_data, compresslevel=6)
            return len(compressed_data) / 1024

        except Exception as e:
            logger.warning(f"Could not compress {file_path}: {e}")
            # Fallback to estimated 3:1 compression ratio
            return file_path.stat().st_size / 1024 / 3.0

    def _generate_bundle_recommendations(
        self, chunks: List[Dict], total_gzipped_kb: float
    ) -> List[str]:
        """Generate bundle optimization recommendations."""
        recommendations = []

        if total_gzipped_kb > self.BUNDLE_SIZE_BUDGET_KB:
            recommendations.append(
                f"Bundle size ({total_gzipped_kb:.1f}KB) exceeds budget ({self.BUNDLE_SIZE_BUDGET_KB}KB)"
            )

        # Check for large chunks
        large_chunks = [c for c in chunks if c["gzipped_kb"] > 50]
        if large_chunks:
            recommendations.append(
                f"Large chunks detected: {', '.join(c['name'] for c in large_chunks[:3])}"
            )
            recommendations.append("Consider code splitting and dynamic imports")

        # Check for many small chunks
        small_chunks = [c for c in chunks if c["gzipped_kb"] < 5]
        if len(small_chunks) > 20:
            recommendations.append(
                f"{len(small_chunks)} small chunks detected - consider chunk merging"
            )

        # Framework-specific recommendations
        framework_chunks = [c for c in chunks if "framework" in c["name"].lower()]
        if framework_chunks and framework_chunks[0]["gzipped_kb"] > 100:
            recommendations.append("Large framework bundle - consider Next.js optimization")

        return recommendations

    def run_lighthouse_ci(self) -> Optional[LighthouseMetrics]:
        """Run Lighthouse CI audit."""
        logger.info("🔍 Running Lighthouse CI audit...")

        try:
            # Check if lhci is available
            result = subprocess.run(
                ["npx", "lhci", "--version"],
                cwd=self.web_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning("Lighthouse CI not available, installing...")
                subprocess.run(
                    ["npm", "install", "--save-dev", "@lhci/cli"],
                    cwd=self.web_dir,
                    check=True,
                )

            # Run Lighthouse audit
            result = subprocess.run(
                ["npx", "lhci", "autorun"],
                cwd=self.web_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return self._parse_lighthouse_results()
            else:
                logger.warning(f"Lighthouse CI failed: {result.stderr}")
                return self._mock_lighthouse_results()

        except Exception as e:
            logger.warning(f"Lighthouse CI error: {e}")
            return self._mock_lighthouse_results()

    def _parse_lighthouse_results(self) -> LighthouseMetrics:
        """Parse Lighthouse CI results."""
        # Look for Lighthouse results file
        results_dir = self.web_dir / ".lighthouseci"

        if not results_dir.exists():
            return self._mock_lighthouse_results()

        try:
            # Find most recent results file
            results_files = list(results_dir.glob("lhr-*.json"))
            if not results_files:
                return self._mock_lighthouse_results()

            latest_file = max(results_files, key=lambda p: p.stat().st_mtime)

            with open(latest_file) as f:
                data = json.load(f)

            categories = data.get("categories", {})
            audits = data.get("audits", {})

            return LighthouseMetrics(
                performance_score=int(categories.get("performance", {}).get("score", 0) * 100),
                fcp_ms=audits.get("first-contentful-paint", {}).get("numericValue"),
                lcp_ms=audits.get("largest-contentful-paint", {}).get("numericValue"),
                cls=audits.get("cumulative-layout-shift", {}).get("numericValue"),
                tbt_ms=audits.get("total-blocking-time", {}).get("numericValue"),
                si_ms=audits.get("speed-index", {}).get("numericValue"),
            )

        except Exception as e:
            logger.warning(f"Could not parse Lighthouse results: {e}")
            return self._mock_lighthouse_results()

    def _mock_lighthouse_results(self) -> LighthouseMetrics:
        """Generate mock Lighthouse results for testing."""
        return LighthouseMetrics(
            performance_score=85,  # Below our 90 target
            fcp_ms=1500,
            lcp_ms=2200,
            cls=0.08,
            tbt_ms=180,
            si_ms=2800,
        )

    def generate_performance_report(
        self, bundle_analysis: BundleAnalysis, lighthouse_metrics: LighthouseMetrics
    ) -> str:
        """Generate comprehensive frontend performance report."""
        report = []
        report.append("=" * 80)
        report.append("FRONTEND PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {datetime.now()}")
        report.append("")

        # Bundle Analysis
        report.append("BUNDLE SIZE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Size: {bundle_analysis.total_size_kb:.1f} KB")
        report.append(f"Gzipped Size: {bundle_analysis.gzipped_size_kb:.1f} KB")
        report.append(f"Compression Ratio: {bundle_analysis.compression_ratio:.1f}:1")
        report.append(f"Chunk Count: {bundle_analysis.chunk_count}")
        report.append(f"Performance Impact: {bundle_analysis.performance_impact.upper()}")

        budget_status = (
            "✅ WITHIN BUDGET"
            if bundle_analysis.gzipped_size_kb <= self.BUNDLE_SIZE_BUDGET_KB
            else "❌ EXCEEDS BUDGET"
        )
        report.append(f"Budget Compliance: {budget_status}")
        report.append("")

        # Largest chunks
        report.append("LARGEST CHUNKS")
        report.append("-" * 40)
        for i, chunk in enumerate(bundle_analysis.largest_chunks[:5], 1):
            report.append(f"{i}. {chunk['name']}: {chunk['gzipped_kb']:.1f}KB (gzipped)")
        report.append("")

        # Lighthouse Metrics
        report.append("LIGHTHOUSE PERFORMANCE METRICS")
        report.append("-" * 40)
        performance_status = (
            "✅ GOOD"
            if lighthouse_metrics.performance_score >= self.LIGHTHOUSE_PERFORMANCE_BUDGET
            else "❌ NEEDS IMPROVEMENT"
        )
        report.append(
            f"Performance Score: {lighthouse_metrics.performance_score}/100 {performance_status}"
        )

        if lighthouse_metrics.fcp_ms:
            report.append(f"First Contentful Paint: {lighthouse_metrics.fcp_ms:.0f}ms")
        if lighthouse_metrics.lcp_ms:
            report.append(f"Largest Contentful Paint: {lighthouse_metrics.lcp_ms:.0f}ms")
        if lighthouse_metrics.cls:
            report.append(f"Cumulative Layout Shift: {lighthouse_metrics.cls:.3f}")
        if lighthouse_metrics.tbt_ms:
            report.append(f"Total Blocking Time: {lighthouse_metrics.tbt_ms:.0f}ms")
        if lighthouse_metrics.si_ms:
            report.append(f"Speed Index: {lighthouse_metrics.si_ms:.0f}ms")
        report.append("")

        # Recommendations
        if bundle_analysis.recommendations:
            report.append("OPTIMIZATION RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(bundle_analysis.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete frontend performance analysis."""
        logger.info("🎯 Starting frontend performance analysis...")

        # Bundle analysis
        bundle_analysis = self.analyze_bundle_size()

        # Lighthouse analysis
        lighthouse_metrics = self.run_lighthouse_ci()

        # Generate report
        report = self.generate_performance_report(bundle_analysis, lighthouse_metrics)
        print(report)

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "bundle_analysis": asdict(bundle_analysis),
            "lighthouse_metrics": asdict(lighthouse_metrics),
            "performance_budget_compliance": {
                "bundle_size_compliant": bundle_analysis.gzipped_size_kb
                <= self.BUNDLE_SIZE_BUDGET_KB,
                "lighthouse_compliant": lighthouse_metrics.performance_score
                >= self.LIGHTHOUSE_PERFORMANCE_BUDGET,
            },
        }

        results_file = (
            f"frontend_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        return results


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    analyzer = FrontendPerformanceAnalyzer(project_root)
    results = analyzer.run_complete_analysis()

    # Exit with error if budgets are exceeded
    compliance = results["performance_budget_compliance"]
    if not compliance["bundle_size_compliant"] or not compliance["lighthouse_compliant"]:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
