import { Page } from "@playwright/test";

export async function waitForPageReady(page: Page) {
  await page.waitForLoadState("networkidle");
  // Wait for any React hydration
  await page.waitForTimeout(1000);
}

export async function checkForContent(
  page: Page,
  selectors: string[],
): Promise<boolean> {
  for (const selector of selectors) {
    try {
      const count = await page.locator(selector).count();
      if (count > 0) {
        return true;
      }
    } catch (e) {
      // Continue to next selector
    }
  }
  return false;
}

export async function waitForWebSocket(
  page: Page,
  timeout = 5000,
): Promise<boolean> {
  try {
    await page.waitForEvent("websocket", { timeout });
    return true;
  } catch {
    return false;
  }
}
