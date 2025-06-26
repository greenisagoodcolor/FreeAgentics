module.exports = {
  root: true,
  env: {
    browser: true,
    es2021: true,
    node: true,
    jest: true,
  },
  extends: [
    "next/core-web-vitals",
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/recommended-requiring-type-checking",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "plugin:jsx-a11y/recommended",
    "plugin:import/recommended",
    "plugin:import/typescript",
    "prettier", // This must be last to override other configs
  ],
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: "latest",
    sourceType: "module",
    project: "./tsconfig.json",
    tsconfigRootDir: __dirname,
  },
  plugins: ["react", "@typescript-eslint", "import", "jsx-a11y"],
  settings: {
    react: {
      version: "detect",
    },
    "import/resolver": {
      typescript: {
        project: "./tsconfig.json",
      },
      node: {
        extensions: [".js", ".jsx", ".ts", ".tsx"],
      },
    },
  },
  rules: {
    // TypeScript specific rules
    "@typescript-eslint/explicit-function-return-type": "off",
    "@typescript-eslint/explicit-module-boundary-types": "off",
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-unused-vars": [
      "error",
      {
        argsIgnorePattern: "^_",
        varsIgnorePattern: "^_",
      },
    ],
    "@typescript-eslint/no-non-null-assertion": "warn",
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/await-thenable": "error",
    "@typescript-eslint/no-misused-promises": "error",
    "@typescript-eslint/require-await": "error",
    "@typescript-eslint/consistent-type-imports": [
      "error",
      {
        prefer: "type-imports",
        disallowTypeAnnotations: true,
      },
    ],

    // React specific rules
    "react/react-in-jsx-scope": "off", // Not needed in Next.js
    "react/prop-types": "off", // We use TypeScript
    "react/jsx-uses-react": "off",
    "react/jsx-filename-extension": ["error", { extensions: [".jsx", ".tsx"] }],
    "react/jsx-props-no-spreading": "off",
    "react/jsx-boolean-value": ["error", "never"],
    "react/self-closing-comp": "error",
    "react/jsx-sort-props": [
      "error",
      {
        callbacksLast: true,
        shorthandFirst: true,
        reservedFirst: true,
      },
    ],

    // Import rules
    "import/order": [
      "error",
      {
        groups: [
          "builtin",
          "external",
          "internal",
          "parent",
          "sibling",
          "index",
          "object",
          "type",
        ],
        "newlines-between": "always",
        alphabetize: {
          order: "asc",
          caseInsensitive: true,
        },
      },
    ],
    "import/no-duplicates": "error",
    "import/no-cycle": "error",
    "import/no-unused-modules": "error",
    "import/no-deprecated": "warn",
    "import/newline-after-import": "error",
    "import/no-mutable-exports": "error",
    "import/no-default-export": "off", // Next.js requires default exports for pages

    // General JavaScript/TypeScript rules
    "no-console": [
      "warn",
      {
        allow: ["warn", "error", "info"],
      },
    ],
    "no-debugger": "error",
    "no-alert": "error",
    "no-await-in-loop": "error",
    "no-return-await": "error",
    "no-restricted-syntax": [
      "error",
      {
        selector: "ForInStatement",
        message:
          "for..in loops iterate over the entire prototype chain, which is virtually never what you want. Use Object.{keys,values,entries}, and iterate over the resulting array.",
      },
    ],
    "prefer-const": "error",
    "no-var": "error",
    "prefer-template": "error",
    "object-shorthand": "error",
    "no-nested-ternary": "error",

    // Code quality rules
    "max-depth": ["error", 4],
    "max-lines": [
      "error",
      { max: 500, skipBlankLines: true, skipComments: true },
    ],
    "max-lines-per-function": [
      "error",
      { max: 50, skipBlankLines: true, skipComments: true },
    ],
    "max-params": ["error", 4],
    complexity: ["error", 10],

    // Accessibility
    "jsx-a11y/anchor-is-valid": [
      "error",
      {
        components: ["Link"],
        specialLink: ["hrefLeft", "hrefRight"],
        aspects: ["invalidHref", "preferButton"],
      },
    ],
  },
  overrides: [
    // Next.js specific overrides
    {
      files: ["app/**/*.tsx", "app/**/*.ts"],
      rules: {
        "import/no-default-export": "off",
        "@typescript-eslint/naming-convention": [
          "error",
          {
            selector: "function",
            format: ["camelCase", "PascalCase"],
          },
        ],
      },
    },
    // Test file overrides
    {
      files: ["**/*.test.ts", "**/*.test.tsx", "**/*.spec.ts", "**/*.spec.tsx"],
      env: {
        jest: true,
      },
      rules: {
        "@typescript-eslint/no-explicit-any": "off",
        "max-lines-per-function": "off",
      },
    },
    // Configuration file overrides
    {
      files: ["*.config.js", "*.config.ts"],
      rules: {
        "import/no-default-export": "off",
      },
    },
  ],
  ignorePatterns: [
    "node_modules",
    ".next",
    "out",
    "coverage",
    "dist",
    "*.min.js",
    "public",
    ".eslintrc.js",
  ],
};
