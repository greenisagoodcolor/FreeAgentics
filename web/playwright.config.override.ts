import { defineConfig } from "@playwright/test";
import baseConfig from "./playwright.config";

export default defineConfig({
  ...baseConfig,
  outputDir:
    "/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250630_222247/frontend/playwright/test-results",
  reporter: [
    [
      "html",
      {
        outputFolder:
          "/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250630_222247/frontend/playwright/report",
      },
    ],
    [
      "json",
      {
        outputFile:
          "/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250630_222247/frontend/playwright/results.json",
      },
    ],
    [
      "junit",
      {
        outputFile:
          "/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250630_222247/frontend/playwright/results.xml",
      },
    ],
  ],
});
