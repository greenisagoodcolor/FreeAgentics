#!/bin/bash
#
# Run Performance Benchmarks
# ==========================
#
# This script runs the comprehensive performance benchmark suite
# and generates reports for CI/CD integration.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "benchmarks/performance_suite.py" ]; then
    echo -e "${RED}Error: Must run from FreeAgentics root directory${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${GREEN}Creating benchmark directories...${NC}"
mkdir -p benchmarks/results
mkdir -p benchmarks/baselines
mkdir -p benchmarks/ci_results
mkdir -p benchmarks/artifacts

# Install dependencies if needed
if ! python -c "import pytest_benchmark" 2>/dev/null; then
    echo -e "${YELLOW}Installing pytest-benchmark...${NC}"
    pip install pytest-benchmark psutil memory-profiler pandas
fi

# Export environment variables
export GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Parse command line arguments
UPDATE_BASELINE=false
FILTER=""
PROFILE=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --update-baseline)
            UPDATE_BASELINE=true
            shift
            ;;
        --filter)
            FILTER="-k $2"
            shift 2
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --update-baseline    Update performance baseline"
            echo "  --filter PATTERN     Filter benchmarks by pattern"
            echo "  --profile           Enable profiling"
            echo "  --quick             Run quick benchmarks only"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Run benchmarks
echo -e "${GREEN}Running performance benchmarks...${NC}"
echo "Git commit: $GIT_COMMIT"
echo "Git branch: $GIT_BRANCH"
echo ""

BENCHMARK_ARGS=""
if [ "$QUICK" = true ]; then
    BENCHMARK_ARGS="--benchmark-max-time=10 --benchmark-min-rounds=3"
else
    BENCHMARK_ARGS="--benchmark-warmup=on --benchmark-disable-gc --benchmark-min-rounds=5"
fi

# Run pytest benchmarks
python -m pytest benchmarks/performance_suite.py \
    -v \
    --benchmark-only \
    --benchmark-json=benchmarks/results/benchmark_results.json \
    --benchmark-save=performance_$(date +%Y%m%d_%H%M%S) \
    --benchmark-save-data \
    --benchmark-group-by=func \
    --benchmark-sort=name \
    --benchmark-columns=min,max,mean,stddev,median,iqr,outliers,rounds,iterations \
    $BENCHMARK_ARGS \
    $FILTER

# Check exit code
BENCHMARK_EXIT=$?
if [ $BENCHMARK_EXIT -ne 0 ]; then
    echo -e "${RED}Benchmarks failed with exit code $BENCHMARK_EXIT${NC}"
    exit $BENCHMARK_EXIT
fi

echo -e "${GREEN}Benchmarks completed successfully${NC}"

# Run regression check
echo -e "${GREEN}Checking for performance regressions...${NC}"

python benchmarks/ci_integration.py \
    --results-file benchmarks/results/benchmark_results.json \
    --baseline-dir benchmarks/baselines \
    --output-dir benchmarks/ci_results \
    --github-comment

REGRESSION_EXIT=$?

# Show results summary
if [ -f "benchmarks/ci_results/latest_regression_report.json" ]; then
    echo ""
    echo -e "${GREEN}Performance Summary:${NC}"
    
    OVERALL_STATUS=$(jq -r '.overall_status' benchmarks/ci_results/latest_regression_report.json)
    TOTAL_BENCHMARKS=$(jq -r '.summary.total_benchmarks' benchmarks/ci_results/latest_regression_report.json)
    REGRESSIONS=$(jq -r '.summary.regressions' benchmarks/ci_results/latest_regression_report.json)
    IMPROVEMENTS=$(jq -r '.summary.improvements' benchmarks/ci_results/latest_regression_report.json)
    
    echo "Overall Status: $OVERALL_STATUS"
    echo "Total Benchmarks: $TOTAL_BENCHMARKS"
    echo "Regressions: $REGRESSIONS"
    echo "Improvements: $IMPROVEMENTS"
    
    if [ "$REGRESSIONS" -gt 0 ]; then
        echo ""
        echo -e "${RED}Performance Regressions Detected:${NC}"
        jq -r '.regressions[] | "- \(.benchmark_name) (\(.metric)): \(.regression_percent | floor)% regression"' \
            benchmarks/ci_results/latest_regression_report.json
    fi
    
    if [ "$IMPROVEMENTS" -gt 0 ]; then
        echo ""
        echo -e "${GREEN}Performance Improvements:${NC}"
        jq -r '.improvements[] | "- \(.benchmark_name) (\(.metric)): \(.regression_percent | floor)% improvement"' \
            benchmarks/ci_results/latest_regression_report.json
    fi
fi

# Update baseline if requested and no regressions
if [ "$UPDATE_BASELINE" = true ]; then
    if [ "$REGRESSION_EXIT" -eq 0 ] || [ "$REGRESSIONS" -eq 0 ]; then
        echo ""
        echo -e "${GREEN}Updating performance baseline...${NC}"
        python benchmarks/ci_integration.py \
            --results-file benchmarks/results/benchmark_results.json \
            --baseline-dir benchmarks/baselines \
            --update-baseline
        echo -e "${GREEN}Baseline updated successfully${NC}"
    else
        echo -e "${YELLOW}Baseline not updated due to regressions${NC}"
    fi
fi

# Run profiling if requested
if [ "$PROFILE" = true ]; then
    echo ""
    echo -e "${GREEN}Running performance profiling...${NC}"
    
    # CPU profiling
    if command -v py-spy &> /dev/null; then
        echo "Running CPU profiling..."
        py-spy record -d 30 -o benchmarks/artifacts/cpu_profile.svg -- \
            python -m pytest benchmarks/performance_suite.py::AgentSpawnBenchmarks -v
    fi
    
    # Memory profiling
    echo "Running memory profiling..."
    python -m memory_profiler benchmarks/performance_suite.py > benchmarks/artifacts/memory_profile.txt
fi

# Generate trend report
echo ""
echo -e "${GREEN}Generating performance trend report...${NC}"
python benchmarks/ci_integration.py \
    --results-file benchmarks/results/benchmark_results.json \
    --baseline-dir benchmarks/baselines \
    --output-dir benchmarks/ci_results

echo ""
echo -e "${GREEN}Performance benchmarks complete!${NC}"
echo "Results saved to: benchmarks/results/"
echo "Reports saved to: benchmarks/ci_results/"

# Exit with regression status
exit $REGRESSION_EXIT