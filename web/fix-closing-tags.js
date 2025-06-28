#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Function to fix closing JSX tags
function fixClosingTags(content) {
  let fixed = content;
  
  // Fix closing tags with "as any)}>);"
  fixed = fixed.replace(/\{\.\.\.{} as any\}\}>(?=\);)/g, '{...({} as any)} />');
  
  // Fix multi-line JSX components that end with >);
  fixed = fixed.replace(/\{\.\.\.{} as any\}\}\s*\/>(?=\);)/g, '{...({} as any)} />');
  
  // Fix any remaining broken closings
  fixed = fixed.replace(/>\);/g, ' />);');
  
  // Also fix ones that already have / but wrong syntax
  fixed = fixed.replace(/\{\.\.\.{} as any\}\}\s*\/>>/g, '{...({} as any)} />');
  
  return fixed;
}

// Find all test files
const testFiles = glob.sync('**/*.test.{ts,tsx}', {
  cwd: __dirname,
  ignore: ['node_modules/**', 'dist/**', 'build/**'],
  absolute: true
});

console.log(`Found ${testFiles.length} test files to fix`);

testFiles.forEach(file => {
  try {
    const content = fs.readFileSync(file, 'utf8');
    const fixed = fixClosingTags(content);
    
    if (content !== fixed) {
      fs.writeFileSync(file, fixed);
      console.log(`✓ Fixed closing tags in ${path.relative(__dirname, file)}`);
    }
  } catch (err) {
    console.error(`✗ Error processing ${file}:`, err.message);
  }
});

console.log('\nClosing tag fixes applied.');