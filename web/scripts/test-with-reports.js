#!/usr/bin/env node
/**
 * Test runner wrapper that respects timestamped report directories
 * This ensures all Jest and Playwright outputs go to the right place
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Get environment variables
const timestamp = process.env.TEST_TIMESTAMP || new Date().toISOString().replace(/[:.]/g, '-');
const reportDir = process.env.TEST_REPORT_DIR || path.join('..', 'tests', 'reports', timestamp);

// Determine test type from command line
const testType = process.argv[2] || 'jest';
const additionalArgs = process.argv.slice(3).join(' ');

// Ensure report directory exists
const frontendReportDir = path.join(reportDir, 'frontend');
if (!fs.existsSync(frontendReportDir)) {
  fs.mkdirSync(frontendReportDir, { recursive: true });
}

// Configure based on test type
let command;
switch (testType) {
  case 'jest':
  case 'unit':
    const jestCoverageDir = process.env.JEST_COVERAGE_DIR || path.join(frontendReportDir, 'jest', 'coverage');
    command = `jest --coverage --coverageDirectory=${jestCoverageDir} ${additionalArgs}`;
    break;
    
  case 'playwright':
  case 'e2e':
    const playwrightHtmlReport = process.env.PLAYWRIGHT_HTML_REPORT || path.join(frontendReportDir, 'playwright', 'report');
    const playwrightOutputDir = process.env.PLAYWRIGHT_TEST_OUTPUT_DIR || path.join(frontendReportDir, 'playwright', 'test-results');
    
    // Set environment variables for Playwright config
    process.env.PLAYWRIGHT_HTML_REPORT = playwrightHtmlReport;
    process.env.PLAYWRIGHT_JSON_OUTPUT = path.join(frontendReportDir, 'playwright', 'results.json');
    process.env.PLAYWRIGHT_JUNIT_OUTPUT = path.join(frontendReportDir, 'playwright', 'results.xml');
    
    command = `playwright test --reporter=html:${playwrightHtmlReport} --output-folder=${playwrightOutputDir} ${additionalArgs}`;
    break;
    
  default:
    console.error(`Unknown test type: ${testType}`);
    process.exit(1);
}

console.log(`Running tests with timestamp: ${timestamp}`);
console.log(`Report directory: ${reportDir}`);
console.log(`Command: ${command}`);

try {
  execSync(command, { stdio: 'inherit' });
} catch (error) {
  process.exit(error.status || 1);
}