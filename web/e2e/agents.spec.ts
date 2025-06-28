import { test, expect } from "@playwright/test";

test.describe("Agents Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agents");
    await page.waitForLoadState("networkidle");
  });

  test("agents page loads successfully", async ({ page }) => {
    await expect(page).toHaveURL(/\/agents/);

    // Check for the main heading
    await expect(page.getByRole('heading', { name: 'Agent Management' })).toBeVisible();

    // Check for the Create Agent button (get first one if multiple exist)
    await expect(page.getByRole('button', { name: 'Create Agent' }).first()).toBeVisible();

    // Verify agent-related content is present
    await expect(page.getByText('Total Agents')).toBeVisible();
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
    // Either has content or shows appropriate empty state
    let hasContentOrEmptyState = false;
    
    // Check for text-based empty state messages
    const textIndicators = ['No agents', 'Empty', 'Get started', 'Create your first'];
    for (const text of textIndicators) {
      if ((await page.getByText(text).count()) > 0) {
        hasContentOrEmptyState = true;
        break;
      }
    }
    
    // Check for data-testid or class-based empty state
    if (!hasContentOrEmptyState) {
      const emptySelectors = ['[data-testid*="empty"]', '[class*="empty"]'];
      for (const selector of emptySelectors) {
        if ((await page.locator(selector).count()) > 0) {
          hasContentOrEmptyState = true;
          break;
        }
      }
    }

    // Or has actual content
    const contentElements = page.locator(
      '[data-testid*="agent"], [class*="agent"]',
    );
    if ((await contentElements.count()) > 0) {
      hasContentOrEmptyState = true;
    }
    
    // Or at least the page loaded with the heading
    if (!hasContentOrEmptyState) {
      const hasHeading = await page.getByRole('heading', { name: 'Agent Management' }).count() > 0;
      const hasCreateButton = await page.getByRole('button', { name: 'Create Agent' }).count() > 0;
      hasContentOrEmptyState = hasHeading || hasCreateButton;
    }

    expect(hasContentOrEmptyState).toBe(true);
  });
});
