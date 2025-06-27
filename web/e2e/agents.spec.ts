import { test, expect } from "@playwright/test";

test.describe("Agents Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agents");
    await page.waitForLoadState("networkidle");
  });

  test("agents page loads successfully", async ({ page }) => {
    await expect(page).toHaveURL(/\/agents/);

    // Look for agent-related content
    const agentIndicators = ["agent", "create", "list", "add", "new"];

    let foundContent = false;
    for (const indicator of agentIndicators) {
      const elements = page.locator(`text="${indicator}" >> visible=true`);
      if ((await elements.count()) > 0) {
        foundContent = true;
        break;
      }
    }

    // Should have some agent-related content
    expect(foundContent).toBe(true);
  });

  test("can interact with agent creation flow", async ({ page }) => {
    // Look for create/add agent buttons
    const createButtons = page.locator(
      'button:has-text("Create"), button:has-text("Add"), button:has-text("New")',
    );

    if ((await createButtons.count()) > 0) {
      await createButtons.first().click();
      await page.waitForTimeout(1000);

      // Should open some kind of modal or form
      const modalOrForm = page.locator(
        '[role="dialog"], form, [data-testid*="modal"], [data-testid*="form"]',
      );

      if ((await modalOrForm.count()) > 0) {
        await expect(modalOrForm.first()).toBeVisible();

        // If there's a close button, test it
        const closeButton = page.locator(
          'button[aria-label*="close"], button:has-text("Cancel"), [data-testid*="close"]',
        );
        if ((await closeButton.count()) > 0) {
          await closeButton.first().click();
          await page.waitForTimeout(500);
        }
      }
    }
  });

  test("agent list renders properly", async ({ page }) => {
    // Look for list structures
    const listElements = page.locator(
      'ul, [role="list"], [data-testid*="list"], [class*="list"]',
    );
    const cardElements = page.locator('[data-testid*="card"], [class*="card"]');
    const tableElements = page.locator('table, [role="table"]');

    // Should have some kind of list/grid structure
    const hasListStructure =
      (await listElements.count()) > 0 ||
      (await cardElements.count()) > 0 ||
      (await tableElements.count()) > 0;

    expect(hasListStructure).toBe(true);
  });

  test("handles empty state gracefully", async ({ page }) => {
    // The page should handle the case where there are no agents
    // Look for empty state messages or default content
    const emptyStateIndicators = [
      'text="No agents"',
      'text="Empty"',
      'text="Get started"',
      'text="Create your first"',
      '[data-testid*="empty"]',
      '[class*="empty"]',
    ];

    // Either has content or shows appropriate empty state
    let hasContentOrEmptyState = false;

    for (const indicator of emptyStateIndicators) {
      if ((await page.locator(indicator).count()) > 0) {
        hasContentOrEmptyState = true;
        break;
      }
    }

    // Or has actual content
    const contentElements = page.locator(
      '[data-testid*="agent"], [class*="agent"]',
    );
    if ((await contentElements.count()) > 0) {
      hasContentOrEmptyState = true;
    }

    expect(hasContentOrEmptyState).toBe(true);
  });
});
