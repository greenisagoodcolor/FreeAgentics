module.exports = {
  // TypeScript and JavaScript files
  "*.{js,jsx,ts,tsx}": [
    "eslint --fix --max-warnings=0",
    "prettier --write",
    "jest --bail --findRelatedTests --passWithNoTests",
  ],

  // JSON files
  "*.json": ["prettier --write"],

  // CSS files
  "*.{css,scss,sass}": ["prettier --write"],

  // Markdown files
  "*.md": ["prettier --write", "markdownlint --fix"],

  // Python files (backend)
  "*.py": [
    "black --check",
    "isort --check-only",
    "flake8 --max-line-length=88 --extend-ignore=E203,W503",
  ],

  // YAML files
  "*.{yml,yaml}": ["prettier --write"],

  // Package files - sort and format
  "package.json": ["prettier --write", "npx sort-package-json"],
};
