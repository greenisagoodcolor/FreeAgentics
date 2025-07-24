#!/bin/bash

# Pre-commit hook script for FreeAgentics Web
# This enforces Uncle Bob's disciplined coding standards

echo "🚨 Running pre-commit checks..."

# Run ESLint
echo "📋 Running ESLint..."
npx eslint . --ext .js,.jsx,.ts,.tsx --max-warnings 0
if [ $? -ne 0 ]; then
    echo "❌ ESLint found errors. Please fix them before committing."
    exit 1
fi

# Run Prettier check
echo "✨ Checking Prettier formatting..."
npx prettier --check .
if [ $? -ne 0 ]; then
    echo "❌ Prettier found formatting issues. Run 'npm run format' to fix them."
    exit 1
fi

# Run TypeScript type checking
echo "🔍 Running TypeScript type check..."
npm run type-check
if [ $? -ne 0 ]; then
    echo "❌ TypeScript type errors found. Please fix them before committing."
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
npm test -- --passWithNoTests
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix them before committing."
    exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
