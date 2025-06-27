// FreeAgentics Commit Lint Configuration
// Expert Committee: Robert C. Martin, Kent Beck, Rich Hickey, Conor Heins
//
// Enforces conventional commit message standards for clear project history

module.exports = {
  extends: ["@commitlint/config-conventional"],

  rules: {
    // Expert Committee approved commit types
    "type-enum": [
      2,
      "always",
      [
        "feat", // New features (Kent Beck: incremental improvement)
        "fix", // Bug fixes (Robert C. Martin: clean code)
        "docs", // Documentation updates
        "style", // Code style changes (no logic changes)
        "refactor", // Code refactoring (Rich Hickey: simplification)
        "test", // Adding or updating tests (Kent Beck: testing)
        "chore", // Maintenance tasks
        "perf", // Performance improvements
        "ci", // CI/CD related changes
        "build", // Build system changes
        "revert", // Reverting previous commits

        // FreeAgentics-specific types
        "agents", // Agent-related changes
        "inference", // Active Inference mathematical updates (Conor Heins)
        "coalition", // Coalition formation features
        "safety", // Safety and security improvements
        "gnn", // Graph Neural Network changes
        "world", // World simulation updates
      ],
    ],

    // Scope rules for FreeAgentics project structure
    "scope-enum": [
      1,
      "always",
      [
        // Core domains (ADR-002 compliance)
        "agents",
        "inference",
        "coalitions",
        "world",

        // Interface layers
        "api",
        "web",
        "ui",

        // Infrastructure
        "db",
        "config",
        "deploy",
        "ci",
        "docs",

        // Specific components
        "gnn",
        "llm",
        "active-inference",
        "markov-blanket",
        "belief-state",
        "precision",
        "safety",
        "export",
        "taskmaster",
      ],
    ],

    // Message format requirements
    "subject-case": [2, "always", "lower-case"],
    "subject-empty": [2, "never"],
    "subject-full-stop": [2, "never", "."],
    "subject-max-length": [2, "always", 72],
    "header-max-length": [2, "always", 100],

    // Body and footer requirements for significant changes
    "body-leading-blank": [1, "always"],
    "footer-leading-blank": [1, "always"],

    // Expert Committee quality gates
    "body-max-line-length": [1, "always", 100],
  },

  // Robert C. Martin: Clear, descriptive commit messages
  // Kent Beck: Incremental, testable changes
  // Rich Hickey: Simple, understandable modifications
  // Conor Heins: Mathematically precise descriptions
};
