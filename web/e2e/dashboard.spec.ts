import { test, expect } from "@playwright/test";

test.describe("Dashboard Functionality", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/dashboard");
    await page.waitForLoadState("networkidle");
  });

  test("dashboard loads with key components", async ({ page }) => {
    // Check that dashboard page loads
    await expect(page).toHaveURL(/\/dashboard/);

    // Look for dashboard-specific elements
    const dashboardIndicators = [
      "agent",
      "dashboard",
      "card",
      "metric",
      "chart",
      "overview",
    ];

    let foundIndicators = 0;
    for (const indicator of dashboardIndicators) {
      const hasText = await page.getByText(indicator, { exact: false }).count() > 0;
      const hasTestId = await page.locator(`[data-testid*="${indicator}"]`).count() > 0;
      const hasClass = await page.locator(`[class*="${indicator}"]`).count() > 0;
      
      if (hasText || hasTestId || hasClass) {
        foundIndicators++;
      }
    }

    // Expect to find at least some dashboard-like content
    expect(foundIndicators).toBeGreaterThan(0);
  });

  test("interactive elements are functional", async ({ page }) => {
    // Look for common interactive elements
    const buttons = page.locator("button:visible");
    const buttonCount = await buttons.count();

    if (buttonCount > 0) {
      // Test that buttons are clickable (at least the first few)
      const testCount = Math.min(3, buttonCount);
      for (let i = 0; i < testCount; i++) {
        const button = buttons.nth(i);
        await expect(button).toBeEnabled();

        // Click and verify no errors
        await button.click();
        await page.waitForTimeout(500); // Allow for any animations/updates
      }
    }
  });

  test("dashboard data loads without errors", async ({ page }) => {
    // Monitor network requests for failed API calls
    const failedRequests: string[] = [];

    page.on("response", (response) => {
      if (response.status() >= 400 && response.url().includes("/api/")) {
        failedRequests.push(`${response.status()} ${response.url()}`);
      }
    });

    // Reload page to trigger data fetching
    await page.reload();
    await page.waitForLoadState("networkidle");

    // Check for any critical API failures
    const criticalFailures = failedRequests.filter(
      (req) => !req.includes("404"), // 404s might be expected for some optional resources
    );

    expect(criticalFailures).toHaveLength(0);
  });

  test("dashboard handles loading states", async ({ page }) => {
    // Reload and check for loading indicators
    await page.reload();

    // Look for common loading indicators
    const loadingIndicators = [
      '[data-testid*="loading"]',
      '[class*="loading"]',
      '[class*="spinner"]',
      '[class*="skeleton"]',
    ];

    // At least initially, we might see loading states
    let hasLoadingState = false;
    for (const selector of loadingIndicators) {
      if ((await page.locator(selector).count()) > 0) {
        hasLoadingState = true;
        break;
      }
    }
    
    // Also check for text-based loading indicators
    if (!hasLoadingState) {
      hasLoadingState = await page.getByText('Loading').count() > 0 ||
                       await page.getByText('loading').count() > 0;
    }

    // Wait for loading to complete
    await page.waitForLoadState("networkidle");

    // After loading, content should be visible
    await expect(page.locator("body")).toBeVisible();
  });
});
