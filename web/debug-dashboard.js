const puppeteer = require("puppeteer");

async function debugDashboard() {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();

  try {
    console.log("🔍 Loading dashboard...");
    await page.goto("http://localhost:3000/dashboard", {
      waitUntil: "networkidle0",
    });

    // Check for knowledge graph elements
    console.log("📊 Checking Knowledge Graph elements:");

    const svgElements = await page.$$("svg");
    console.log(`Found ${svgElements.length} SVG elements`);

    for (let i = 0; i < svgElements.length; i++) {
      const svg = svgElements[i];
      const bbox = await svg.boundingBox();
      const classes = await svg.evaluate(
        (el) => el.className.baseVal || el.className,
      );
      console.log(
        `SVG ${i + 1}: ${bbox ? `${bbox.width}x${bbox.height}` : "no size"}, classes: ${classes}`,
      );
    }

    // Check for specific D3.js indicators
    const d3Elements = await page.$$(
      '[class*="node"], [class*="link"], circle, line',
    );
    console.log(`🎯 Found ${d3Elements.length} potential D3.js elements`);

    // Check for knowledge graph text
    const knowledgeText = await page.$eval("body", (el) => {
      const text = el.innerText;
      const matches = text.match(/knowledge|graph|nodes|edges/gi) || [];
      return matches;
    });
    console.log(`📝 Knowledge-related text found: ${knowledgeText.join(", ")}`);

    // Check Redux state
    const reduxState = await page.evaluate(() => {
      if (window.__REDUX_DEVTOOLS_EXTENSION__) {
        return "Redux DevTools detected";
      }
      return "No Redux DevTools";
    });
    console.log(`🔧 Redux: ${reduxState}`);

    // Take a screenshot
    await page.screenshot({ path: "dashboard-debug.png", fullPage: true });
    console.log("📸 Screenshot saved as dashboard-debug.png");
  } catch (error) {
    console.error("❌ Error:", error.message);
  } finally {
    await browser.close();
  }
}

debugDashboard();
