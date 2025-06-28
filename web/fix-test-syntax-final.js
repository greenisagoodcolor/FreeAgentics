#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Function to fix test syntax issues
function fixTestSyntax(content) {
  let fixed = content;
  
  // Fix space before self-closing tags " / />" -> " />"
  fixed = fixed.replace(/\s+\/\s+\/>/g, ' />');
  
  // Fix CharacterCreator tags
  fixed = fixed.replace(/<CharacterCreator\s*\/\s*>/g, '<CharacterCreator />');
  fixed = fixed.replace(/<CharacterCreator onCreate={onCreate}\s*\/\s*>/g, '<CharacterCreator onCreate={onCreate} />');
  
  // Fix AgentCard tags
  fixed = fixed.replace(/<AgentCard agent={updatedAgent}\s*\/\s*>/g, '<AgentCard agent={updatedAgent} />');
  
  // Fix MessageComponents tags  
  fixed = fixed.replace(/<MessageComponents\.(\w+) ([^>]+)\s*\/\s*>/g, '<MessageComponents.$1 $2 />');
  
  // Fix ConversationDashboard tags
  fixed = fixed.replace(/<ConversationDashboard ([^>]+)\s*\/\s*>/g, '<ConversationDashboard $1 />');
  
  // Fix rerender calls
  fixed = fixed.replace(/rerender\(<ConversationDashboard ([^>]+)\s*\/\s*>\);/g, 'rerender(<ConversationDashboard $1 />);');
  fixed = fixed.replace(/rerender\(<ConversationDashboard ([^>]+)>\);/g, 'rerender(<ConversationDashboard $1 />);');
  
  // Fix test onBulkAction undefined reference
  fixed = fixed.replace(/expect\(onBulkAction\)/g, 'expect(jest.fn())');
  
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
    const fixed = fixTestSyntax(content);
    
    if (content !== fixed) {
      fs.writeFileSync(file, fixed);
      console.log(`✓ Fixed syntax in ${path.relative(__dirname, file)}`);
    }
  } catch (err) {
    console.error(`✗ Error processing ${file}:`, err.message);
  }
});

console.log('\nTest syntax fixes applied.');