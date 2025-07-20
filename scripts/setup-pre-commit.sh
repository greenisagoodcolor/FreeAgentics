#!/bin/bash
# Setup script for pre-commit hooks

set -e

echo "🔧 Setting up pre-commit hooks for FreeAgentics..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install the hooks
echo "⚙️ Installing pre-commit hooks..."
pre-commit install

# Install commit-msg hook for commit message validation
echo "💬 Installing commit-msg hook..."
pre-commit install --hook-type commit-msg

# Update hooks to latest versions
echo "🔄 Updating hook versions..."
pre-commit autoupdate

# Run hooks on all files to verify setup
echo "🧪 Running hooks on all files (this may take a while)..."
echo "Note: This may show many issues on first run - that's normal!"

# Run with --all-files but don't fail on errors for initial setup
pre-commit run --all-files || echo "⚠️ Some hooks failed - this is expected on first run"

echo ""
echo "✅ Pre-commit setup complete!"
echo ""
echo "📝 Usage:"
echo "  • Hooks will run automatically on git commit"
echo "  • Run manually: pre-commit run --all-files"
echo "  • Update hooks: pre-commit autoupdate"
echo "  • Bypass hooks: git commit --no-verify (use sparingly!)"
echo ""
echo "🎯 Hooks configured:"
echo "  • Python: black, isort, flake8, mypy, bandit"
echo "  • TypeScript: eslint, prettier, type-check"
echo "  • General: trailing whitespace, YAML/JSON validation"
echo "  • Security: secrets detection"
echo "  • Docker: hadolint"
echo ""
echo "🚀 Happy coding with better code quality!"
