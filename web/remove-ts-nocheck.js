#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const glob = require("glob");

// Function to remove @ts-nocheck from test files
function removeTsNoCheck(content) {
  // Remove @ts-nocheck line
  const lines = content.split("\n");
  const filtered = lines.filter(
    (line) => !line.trim().startsWith("// @ts-nocheck"),
  );
  return filtered.join("\n");
}

// Find all test files
const testFiles = glob.sync("**/*.test.{ts,tsx}", {
  cwd: __dirname,
  ignore: ["node_modules/**", "dist/**", "build/**"],
  absolute: true,
});

console.log(`Found ${testFiles.length} test files`);

testFiles.forEach((file) => {
  try {
    const content = fs.readFileSync(file, "utf8");
    const updated = removeTsNoCheck(content);

    if (content !== updated) {
      fs.writeFileSync(file, updated);
      console.log(
        `✓ Removed @ts-nocheck from ${path.relative(__dirname, file)}`,
      );
    }
  } catch (err) {
    console.error(`✗ Error processing ${file}:`, err.message);
  }
});

console.log("\nType suppressions removed from all test files.");
