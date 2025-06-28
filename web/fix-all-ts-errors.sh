#!/bin/bash

# Fix conversation test files - add endTime: null to all conversation mocks
find . -name "*.test.tsx" -type f -exec sed -i.bak 's/status: '\''active'\'' as const,/status: '\''active'\'' as const,\n    endTime: null,/g' {} \;

# Fix timestamp types - convert number to Date
find . -name "*.test.tsx" -type f -exec sed -i.bak 's/timestamp: Date\.now()/timestamp: new Date()/g' {} \;
find . -name "*.test.tsx" -type f -exec sed -i.bak 's/timestamp: Date\.now() - /timestamp: new Date(Date.now() - /g' {} \;

# Add missing within import
find . -name "*.test.tsx" -type f -exec sed -i.bak 's/import { render, screen, fireEvent, waitFor }/import { render, screen, fireEvent, waitFor, within }/g' {} \;

# Fix message timestamp type
find . -name "*.test.tsx" -type f -exec sed -i.bak 's/timestamp: \([0-9]*\),/timestamp: new Date(\1),/g' {} \;

echo "TypeScript fixes applied"