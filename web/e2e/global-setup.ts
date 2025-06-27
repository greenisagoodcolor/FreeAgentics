import { chromium, FullConfig } from "@playwright/test";

async function globalSetup(config: FullConfig) {
  console.log("🚀 Setting up E2E tests...");

  // Launch a browser to verify the dev server is running
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    // Wait for the dev server to be ready
    console.log("⏳ Waiting for dev server...");
    await page.goto("http://localhost:3000", { waitUntil: "networkidle" });
    console.log("✅ Dev server is ready");
  } catch (error) {
    console.error("❌ Dev server not ready:", error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;
