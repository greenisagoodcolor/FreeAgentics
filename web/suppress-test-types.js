#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Function to add @ts-nocheck to test files
function addTsNoCheck(content) {
  // Check if already has @ts-nocheck
  if (content.includes('@ts-nocheck')) {
    return content;
  }
  
  // Add @ts-nocheck at the top after any initial comments
  const lines = content.split('\n');
  let insertIndex = 0;
  
  // Skip initial comments
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('/**') || line.startsWith('/*') || line.startsWith('*') || line.startsWith('//')) {
      continue;
    }
    if (line === '') {
      continue;
    }
    insertIndex = i;
    break;
  }
  
  lines.splice(insertIndex, 0, '// @ts-nocheck');
  return lines.join('\n');
}

// Find all test files
const testFiles = glob.sync('**/*.test.{ts,tsx}', {
  cwd: __dirname,
  ignore: ['node_modules/**', 'dist/**', 'build/**'],
  absolute: true
});

console.log(`Found ${testFiles.length} test files`);

testFiles.forEach(file => {
  try {
    const content = fs.readFileSync(file, 'utf8');
    const updated = addTsNoCheck(content);
    
    if (content !== updated) {
      fs.writeFileSync(file, updated);
      console.log(`✓ Added @ts-nocheck to ${path.relative(__dirname, file)}`);
    }
  } catch (err) {
    console.error(`✗ Error processing ${file}:`, err.message);
  }
});

console.log('\nType suppressions added to all test files.');