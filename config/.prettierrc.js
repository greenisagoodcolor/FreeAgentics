// Prettier configuration for FreeAgentics
// Enforces consistent TypeScript/JavaScript formatting per ADR-004

module.exports = {
  // Basic formatting
  semi: true,
  trailingComma: 'es5',
  singleQuote: true,
  printWidth: 100,
  tabWidth: 2,
  useTabs: false,

  // JavaScript/TypeScript specific
  quoteProps: 'as-needed',
  bracketSpacing: true,
  bracketSameLine: false,
  arrowParens: 'avoid',

  // React/JSX specific
  jsxSingleQuote: true,
  jsxBracketSameLine: false,

  // File extensions to format
  overrides: [
    {
      files: '*.{js,jsx,ts,tsx}',
      options: {
        parser: 'typescript',
      },
    },
    {
      files: '*.json',
      options: {
        parser: 'json',
        printWidth: 120,
      },
    },
    {
      files: '*.md',
      options: {
        parser: 'markdown',
        printWidth: 80,
        proseWrap: 'always',
      },
    },
    {
      files: '*.yml',
      options: {
        parser: 'yaml',
        printWidth: 120,
      },
    },
    {
      files: '*.yaml',
      options: {
        parser: 'yaml',
        printWidth: 120,
      },
    },
  ],

  // Files to ignore
  ignorePath: '.prettierignore',
};
