// Script to fix conversation test data

const fs = require('fs');
const path = require('path');

// Fix conversation test file
const testFile = path.join(__dirname, '__tests__/components/conversation/conversation-orchestration.test.tsx');
let content = fs.readFileSync(testFile, 'utf8');

// Fix the mockConversations to have proper types
content = content.replace(
  /startTime: Date\.now\(\) - (\d+),/g,
  'startTime: new Date(Date.now() - $1),'
);

content = content.replace(
  /lastActivityTime: Date\.now\(\) - (\d+),/g,
  'lastActivityTime: new Date(Date.now() - $1),'
);

// Fix timestamp types in messages
content = content.replace(
  /timestamp: new Date\(\) - (\d+),/g,
  'timestamp: new Date(Date.now() - $1),'
);

// Fix MessageComponents import
content = content.replace(
  "import { MessageComponents } from '@/components/conversation/message-components';",
  "import * as MessageComponents from '@/components/conversation/message-components';"
);

// Fix missing props in ConversationDashboard
content = content.replace(
  '<ConversationDashboard />',
  '<ConversationDashboard conversations={[]} agents={[]} onConversationSelect={() => {}} />'
);

// Write back
fs.writeFileSync(testFile, content);

console.log('Fixed conversation tests');

// Fix agent component tests
const agentTestFile = path.join(__dirname, '__tests__/components/agent-components.test.tsx');
let agentContent = fs.readFileSync(agentTestFile, 'utf8');

// Fix Date arithmetic
agentContent = agentContent.replace(
  /{ timestamp: new Date\(\) - (\d+), beliefs:/g,
  '{ timestamp: new Date(Date.now() - $1), beliefs:'
);

fs.writeFileSync(agentTestFile, agentContent);

console.log('Fixed agent tests');