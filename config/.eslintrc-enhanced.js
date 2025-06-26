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
    // Enhanced Naming Convention Rules per ADR-004
    "@typescript-eslint/naming-convention": [
      "error",
      // Interfaces should start with 'I' prefix (domain interfaces)
      {
        selector: "interface",
        format: ["PascalCase"],
        prefix: ["I"],
        filter: {
          // Exception: Props interfaces don't need I prefix
          regex: "Props$",
          match: false,
        },
      },
      // Type aliases should be PascalCase
      {
        selector: "typeAlias",
        format: ["PascalCase"],
      },
      // Enums should be PascalCase
      {
        selector: "enum",
        format: ["PascalCase"],
      },
      // Enum members should be PascalCase
      {
        selector: "enumMember",
        format: ["PascalCase"],
      },
      // Classes should be PascalCase
      {
        selector: "class",
        format: ["PascalCase"],
      },
      // Functions should be camelCase (except React components)
      {
        selector: "function",
        format: ["camelCase", "PascalCase"],
      },
      {
        selector: "method",
        format: ["camelCase"],
      },
      // Variables should be camelCase or PascalCase (for React components)
      {
        selector: "variable",
        format: ["camelCase", "PascalCase", "UPPER_SNAKE_CASE"],
      },
      // Parameters should be camelCase
      {
        selector: "parameter",
        format: ["camelCase"],
        leadingUnderscore: "allow",
      },
      // Properties should be camelCase
      {
        selector: "property",
        format: ["camelCase", "PascalCase"],
      },
      // Type parameters should be PascalCase
      {
        selector: "typeParameter",
        format: ["PascalCase"],
      },
    ],

    // Prevent prohibited naming patterns per ADR-004
    "id-denylist": [
      "error",
      // Prohibited gaming terminology
      "PlayerAgent",
      "NPCAgent",
      "EnemyAgent",
      "GameWorld",
      "spawn",
      "respawn",
      // Prohibited old project names
      "CogniticNet",
      "cogniticnet",
      "COGNITICNET",
    ],

    // Require descriptive variable names
    "id-length": [
      "error",
      {
        min: 2,
        max: 50,
        exceptions: ["i", "j", "k", "x", "y", "z", "_"],
      },
    ],

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

    // Event handler naming conventions
    "react/jsx-handler-names": [
      "error",
      {
        eventHandlerPrefix: "handle",
        eventHandlerPropPrefix: "on",
        checkLocalVariables: true,
        checkInlineFunction: true,
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
        "id-denylist": "off", // Allow test-specific naming
      },
    },
    // Configuration file overrides
    {
      files: ["*.config.js", "*.config.ts"],
      rules: {
        "import/no-default-export": "off",
        "@typescript-eslint/naming-convention": "off",
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
