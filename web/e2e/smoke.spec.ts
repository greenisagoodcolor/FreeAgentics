import { test, expect } from "@playwright/test";

test.describe("Application Smoke Tests", () => {
  test("homepage loads successfully", async ({ page }) => {
    await page.goto("/");

    // Check that the page loads without errors
    await expect(page).toHaveTitle(/FreeAgentics/);

    // Check for main heading or content
    const heading = page.locator("h1").first();
    await expect(heading).toBeVisible();
  });

  test("navigation works", async ({ page }) => {
    await page.goto("/");

    // Test navigation to different sections
    const navItems = [
      { text: "Dashboard", path: "/dashboard" },
      { text: "Agents", path: "/agents" },
      { text: "Conversations", path: "/conversations" },
      { text: "Knowledge", path: "/knowledge" },
    ];

    for (const item of navItems) {
      // Look for navigation link (might be in nav, header, or sidebar)
      const navLink = page.locator(
        `a[href="${item.path}"], a:has-text("${item.text}")`,
      );

      if ((await navLink.count()) > 0) {
        await navLink.first().click();
        await page.waitForLoadState("networkidle");

        // Check that we navigated to the correct page
        await expect(page).toHaveURL(new RegExp(item.path));

        // Go back to home for next test
        await page.goto("/");
        await page.waitForLoadState("networkidle");
      }
    }
  });

  test("responsive design works", async ({ page }) => {
    await page.goto("/");

    // Test desktop view
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator("body")).toBeVisible();

    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator("body")).toBeVisible();

    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator("body")).toBeVisible();
  });

  test("no console errors on page load", async ({ page }) => {
    const consoleErrors: string[] = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Filter out common non-critical errors
    const criticalErrors = consoleErrors.filter(
      (error) =>
        !error.includes("favicon") &&
        !error.includes("websocket") && // WebSocket errors are expected in dev
        !error.includes("hot-update") && // HMR errors are dev-only
        !error.includes("chunk-"), // Chunk loading errors can be intermittent
    );

    expect(criticalErrors).toHaveLength(0);
  });

  test("basic accessibility checks", async ({ page }) => {
    await page.goto("/");

    // Check for basic accessibility elements
    await expect(page.locator("h1")).toBeVisible(); // Main heading exists

    // Check that interactive elements are keyboard accessible
    const buttons = page.locator("button").first();
    if ((await buttons.count()) > 0) {
      await buttons.focus();
      expect(
        await buttons.evaluate((el) => document.activeElement === el),
      ).toBe(true);
    }
  });
});
