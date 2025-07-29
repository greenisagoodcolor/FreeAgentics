#!/bin/bash
# Script to commit the free energy metrics changes

cd /home/green/freeagentics || exit 1

echo "Current branch:"
git branch --show-current

echo -e "\nModified files:"
git status --short

echo -e "\nAdding all changes..."
git add .

echo -e "\nCreating commit..."
git commit -m "feat: surface PyMDP free energy metrics in UI

- Add avg_free_energy field to SystemMetrics API model
- Aggregate free energy values from active agents in /api/v1/metrics
- Display Free Energy in MetricsFooter with Activity icon
- Update frontend SystemMetrics interface to include avgFreeEnergy
- Suppress Redis warnings in demo mode for cleaner logs
- Document implementation in first-e2e-notes.md and fe-metrics-scan.md

This surfaces existing PyMDP computations without adding new logic,
following the Just-Make-It-Work approach for the first E2E demo."

echo -e "\nCommit created successfully!"
echo "Already on main branch - no merge needed."