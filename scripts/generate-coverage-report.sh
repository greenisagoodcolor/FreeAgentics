#!/bin/bash

# FreeAgentics Coverage Report Generator
# Generates comprehensive coverage reports for both backend and frontend

set -e

echo "🚀 FreeAgentics Coverage Report Generator"
echo "=========================================="

# Create reports directory
REPORTS_DIR="test-reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORTS_DIR/backend"
mkdir -p "$REPORTS_DIR/frontend"
mkdir -p "$REPORTS_DIR/combined"

echo "📁 Reports will be saved to: $REPORTS_DIR"

# Backend Coverage
echo ""
echo "🐍 Running Backend Coverage..."
echo "--------------------------------"

# Run backend tests with coverage
PYTORCH_DISABLE_DOCSTRING_WARNINGS=1 python3 -m pytest \
  --cov=api \
  --cov=agents \
  --cov=coalitions \
  --cov=inference \
  --cov=knowledge \
  --cov=infrastructure \
  --cov=world \
  --cov-report=term \
  --cov-report=html:"$REPORTS_DIR/backend/html" \
  --cov-report=xml:"$REPORTS_DIR/backend/coverage.xml" \
  --cov-report=json:"$REPORTS_DIR/backend/coverage.json" \
  --maxfail=5 \
  -v \
  --tb=short | tee "$REPORTS_DIR/backend/pytest_output.txt"

# Extract backend coverage percentage
BACKEND_COVERAGE=$(tail -20 "$REPORTS_DIR/backend/pytest_output.txt" | grep "TOTAL" | awk '{print $4}' | sed 's/%//')

echo "✅ Backend coverage: ${BACKEND_COVERAGE}%"

# Frontend Coverage
echo ""
echo "🌐 Running Frontend Coverage..."
echo "--------------------------------"

cd web

# Run frontend tests with coverage
npm test -- \
  --coverage \
  --watchAll=false \
  --coverageReporters=text \
  --coverageReporters=html \
  --coverageReporters=json \
  --coverageDirectory="../$REPORTS_DIR/frontend" \
  --passWithNoTests | tee "../$REPORTS_DIR/frontend/jest_output.txt"

cd ..

# Extract frontend coverage percentage
FRONTEND_COVERAGE=$(grep "All files" "$REPORTS_DIR/frontend/jest_output.txt" | awk '{print $2}' | sed 's/%//')

echo "✅ Frontend coverage: ${FRONTEND_COVERAGE}%"

# Generate Combined Report
echo ""
echo "📊 Generating Combined Report..."
echo "--------------------------------"

# Calculate weighted average (68% backend, 32% frontend based on LOC)
BACKEND_WEIGHT=0.68
FRONTEND_WEIGHT=0.32

if [[ -n "$BACKEND_COVERAGE" && -n "$FRONTEND_COVERAGE" ]]; then
  COMBINED_COVERAGE=$(echo "scale=2; ($BACKEND_COVERAGE * $BACKEND_WEIGHT) + ($FRONTEND_COVERAGE * $FRONTEND_WEIGHT)" | bc)
else
  COMBINED_COVERAGE="N/A"
fi

# Create combined summary
cat > "$REPORTS_DIR/combined/COVERAGE_SUMMARY.md" << EOF
# FreeAgentics Coverage Report Summary

**Generated:** $(date)
**Report Directory:** $REPORTS_DIR

## 📊 Coverage Overview

| Component | Coverage | Weight | Contribution |
|-----------|----------|--------|-------------|
| **Backend (Python)** | ${BACKEND_COVERAGE}% | 68% | $(echo "scale=2; $BACKEND_COVERAGE * $BACKEND_WEIGHT" | bc)% |
| **Frontend (TS/JS)** | ${FRONTEND_COVERAGE}% | 32% | $(echo "scale=2; $FRONTEND_COVERAGE * $FRONTEND_WEIGHT" | bc)% |
| **Combined Total** | **${COMBINED_COVERAGE}%** | 100% | ${COMBINED_COVERAGE}% |

## 📁 Report Locations

- **Backend HTML Report:** [backend/html/index.html](backend/html/index.html)
- **Frontend HTML Report:** [frontend/index.html](frontend/index.html)
- **Backend Raw Output:** [backend/pytest_output.txt](backend/pytest_output.txt)
- **Frontend Raw Output:** [frontend/jest_output.txt](frontend/jest_output.txt)

## 🎯 Coverage Targets

| Timeframe | Target | Current | Status |
|-----------|--------|---------|---------|
| Q1 2025 | 35% | ${COMBINED_COVERAGE}% | $(if (( $(echo "$COMBINED_COVERAGE >= 35" | bc -l) )); then echo "✅ Met"; else echo "❌ Not Met"; fi) |
| Q2 2025 | 55% | ${COMBINED_COVERAGE}% | $(if (( $(echo "$COMBINED_COVERAGE >= 55" | bc -l) )); then echo "✅ Met"; else echo "⏳ Pending"; fi) |
| Q3 2025 | 75% | ${COMBINED_COVERAGE}% | $(if (( $(echo "$COMBINED_COVERAGE >= 75" | bc -l) )); then echo "✅ Met"; else echo "⏳ Pending"; fi) |

## 🚀 Next Steps

1. Review detailed HTML reports for uncovered areas
2. Prioritize testing for critical infrastructure modules
3. Fix remaining frontend test suite issues
4. Update coverage tracking documentation

---

**Command to regenerate:** \`./scripts/generate-coverage-report.sh\`
EOF

# Update latest symlink
rm -f test-reports/latest
ln -s "$(basename "$REPORTS_DIR")" test-reports/latest

echo ""
echo "✅ Coverage report generation complete!"
echo ""
echo "📊 Summary:"
echo "   Backend:  ${BACKEND_COVERAGE}%"
echo "   Frontend: ${FRONTEND_COVERAGE}%"
echo "   Combined: ${COMBINED_COVERAGE}%"
echo ""
echo "📁 Reports saved to: $REPORTS_DIR"
echo "🔗 Latest reports: test-reports/latest"
echo ""
echo "🌐 View HTML reports:"
echo "   Backend:  open $REPORTS_DIR/backend/html/index.html"
echo "   Frontend: open $REPORTS_DIR/frontend/index.html"
echo "" 