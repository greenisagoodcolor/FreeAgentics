#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Function to fix broken JSX syntax
function fixJSXSyntax(content) {
  let fixed = content;
  
  // Fix broken JSX tags with "as any / as any"
  fixed = fixed.replace(/<([^>]+)\s+as any\s*\/\s*as any>/g, '<$1 {...({} as any)}>');
  
  // Fix broken closing tags
  fixed = fixed.replace(/<([^>]+)\s+as any\s*\/>/g, '<$1 {...({} as any)} />');
  
  // Fix render calls with broken syntax
  fixed = fixed.replace(/render\(<([^>]+)\s*\/\s*as any>\);/g, 'render(<$1 {...({} as any)}>);');
  
  // Fix multi-line JSX components
  fixed = fixed.replace(/<(\w+)\s+([^>]*?)\s*as any\s*\/>/gm, '<$1 $2 {...({} as any)} />');
  
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
    const fixed = fixJSXSyntax(content);
    
    if (content !== fixed) {
      fs.writeFileSync(file, fixed);
      console.log(`✓ Fixed JSX syntax in ${path.relative(__dirname, file)}`);
    }
  } catch (err) {
    console.error(`✗ Error processing ${file}:`, err.message);
  }
});

console.log('\nJSX syntax fixes applied.');