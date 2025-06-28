import { defineConfig } from '@playwright/test';
import baseConfig from './playwright.config';

export default defineConfig({
  ...baseConfig,
  outputDir: '/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250628_140340/frontend/playwright/test-results',
  reporter: [
    ['html', { outputFolder: '/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250628_140340/frontend/playwright/report' }],
    ['json', { outputFile: '/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250628_140340/frontend/playwright/results.json' }],
    ['junit', { outputFile: '/Users/matthewmoroney/builds/FreeAgentics/tests/reports/20250628_140340/frontend/playwright/results.xml' }],
  ],
});
