#!/bin/bash

# Documentation Validation Script
# This script validates documentation against ADR-002, ADR-003, and ADR-004

echo "=== FreeAgentics Documentation Validation ==="
echo "Validating documentation against ADR-002, ADR-003, and ADR-004"
echo

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Initialize counters
ERRORS=0
WARNINGS=0

# ==========================================
# ADR-002: Canonical Directory Structure Validation
# ==========================================
echo -e "${YELLOW}=== ADR-002: Canonical Directory Structure Validation ===${NC}"

# Check that all documentation is in the docs/ directory
echo "Checking for documentation files outside of docs/..."
DOC_FILES_OUTSIDE=$(find . -type f -not -path "./docs/*" -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./.taskmaster/*" -name "*.md" | wc -l)
if [ $DOC_FILES_OUTSIDE -gt 0 ]; then
    echo -e "${RED}ERROR: Found $DOC_FILES_OUTSIDE documentation files outside of docs/ directory${NC}"
    find . -type f -not -path "./docs/*" -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./.taskmaster/*" -name "*.md" | sort
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: All documentation files are in the docs/ directory${NC}"
fi

# Check that ADRs are in the correct directory
echo "Checking ADR locations..."
ADR_FILES_OUTSIDE=$(find ./docs -type f -name "*adr*.md" -not -path "./docs/architecture/decisions/*" | wc -l)
if [ $ADR_FILES_OUTSIDE -gt 0 ]; then
    echo -e "${RED}ERROR: Found $ADR_FILES_OUTSIDE ADR files outside of docs/architecture/decisions/ directory${NC}"
    find ./docs -type f -name "*adr*.md" -not -path "./docs/architecture/decisions/*" | sort
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: All ADR files are in the docs/architecture/decisions/ directory${NC}"
fi

# Check that API documentation is in the correct directory
echo "Checking API documentation locations..."
API_FILES_OUTSIDE=$(find ./docs -type f -name "*api*.md" -not -path "./docs/api/*" | wc -l)
if [ $API_FILES_OUTSIDE -gt 0 ]; then
    echo -e "${RED}ERROR: Found $API_FILES_OUTSIDE API documentation files outside of docs/api/ directory${NC}"
    find ./docs -type f -name "*api*.md" -not -path "./docs/api/*" | sort
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: All API documentation files are in the docs/api/ directory${NC}"
fi

# Check that tutorial documentation is in the correct directory
echo "Checking tutorial documentation locations..."
TUTORIAL_FILES_OUTSIDE=$(find ./docs -type f -name "*tutorial*.md" -not -path "./docs/tutorials/*" | wc -l)
if [ $TUTORIAL_FILES_OUTSIDE -gt 0 ]; then
    echo -e "${RED}ERROR: Found $TUTORIAL_FILES_OUTSIDE tutorial files outside of docs/tutorials/ directory${NC}"
    find ./docs -type f -name "*tutorial*.md" -not -path "./docs/tutorials/*" | sort
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: All tutorial files are in the docs/tutorials/ directory${NC}"
fi

echo

# ==========================================
# ADR-003: Dependency Rules Validation
# ==========================================
echo -e "${YELLOW}=== ADR-003: Dependency Rules Validation ===${NC}"

# Check for dependency rule violations in code examples
echo "Checking for dependency rule violations in documentation code examples..."

# Core Domain importing from Interface or Infrastructure
echo "Checking for Core Domain importing from Interface or Infrastructure..."
CORE_IMPORTING_INTERFACE=$(grep -r "from api\." --include="*.md" ./docs | wc -l)
CORE_IMPORTING_INFRA=$(grep -r "from infrastructure\." --include="*.md" ./docs | wc -l)

if [ $CORE_IMPORTING_INTERFACE -gt 0 ] || [ $CORE_IMPORTING_INFRA -gt 0 ]; then
    echo -e "${RED}ERROR: Found examples of Core Domain importing from Interface or Infrastructure layers${NC}"
    if [ $CORE_IMPORTING_INTERFACE -gt 0 ]; then
        echo -e "${RED}Found $CORE_IMPORTING_INTERFACE instances of importing from api.*${NC}"
        grep -r "from api\." --include="*.md" ./docs
    fi
    if [ $CORE_IMPORTING_INFRA -gt 0 ]; then
        echo -e "${RED}Found $CORE_IMPORTING_INFRA instances of importing from infrastructure.*${NC}"
        grep -r "from infrastructure\." --include="*.md" ./docs
    fi
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: No examples of Core Domain importing from Interface or Infrastructure layers${NC}"
fi

# Check architecture diagrams for correct dependency flow
echo "Checking architecture diagrams for correct dependency flow..."
WRONG_DEPENDENCY_FLOW=$(grep -r "Core Domain.*->.*Interface" --include="*.md" ./docs | wc -l)
if [ $WRONG_DEPENDENCY_FLOW -gt 0 ]; then
    echo -e "${RED}ERROR: Found $WRONG_DEPENDENCY_FLOW instances of incorrect dependency flow in architecture diagrams${NC}"
    grep -r "Core Domain.*->.*Interface" --include="*.md" ./docs
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: No instances of incorrect dependency flow in architecture diagrams${NC}"
fi

echo

# ==========================================
# ADR-004: Naming Conventions Validation
# ==========================================
echo -e "${YELLOW}=== ADR-004: Naming Conventions Validation ===${NC}"

# Check documentation file naming conventions
echo "Checking documentation file naming conventions..."
NON_KEBAB_CASE_FILES=$(find ./docs -type f -name "*.md" | grep -v "^[a-z0-9-]\+\.md$" | wc -l)
if [ $NON_KEBAB_CASE_FILES -gt 0 ]; then
    echo -e "${RED}ERROR: Found $NON_KEBAB_CASE_FILES documentation files not following kebab-case convention${NC}"
    find ./docs -type f -name "*.md" | grep -v "^[a-z0-9-]\+\.md$" | sort
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: All documentation files follow kebab-case convention${NC}"
fi

# Check for prohibited terms in documentation
echo "Checking for prohibited terms in documentation..."
PROHIBITED_TERMS=("PlayerAgent" "NPCAgent" "spawn(" "GameWorld")
PROHIBITED_FOUND=0

for term in "${PROHIBITED_TERMS[@]}"; do
    COUNT=$(grep -r "$term" --include="*.md" ./docs | wc -l)
    if [ $COUNT -gt 0 ]; then
        echo -e "${RED}ERROR: Found prohibited term '$term' $COUNT times in documentation${NC}"
        grep -r "$term" --include="*.md" ./docs
        PROHIBITED_FOUND=$((PROHIBITED_FOUND+1))
    else
        echo -e "${GREEN}OK: Prohibited term '$term' not found in documentation${NC}"
    fi
done

if [ $PROHIBITED_FOUND -gt 0 ]; then
    ERRORS=$((ERRORS+1))
fi

# Check code examples for naming convention violations
echo "Checking code examples for naming convention violations..."

# Python naming conventions
echo "Checking Python naming conventions..."
PYTHON_CLASS_VIOLATIONS=$(grep -r "class [a-z]" --include="*.md" ./docs | wc -l)
PYTHON_METHOD_VIOLATIONS=$(grep -r "def [A-Z]" --include="*.md" ./docs | wc -l)

if [ $PYTHON_CLASS_VIOLATIONS -gt 0 ] || [ $PYTHON_METHOD_VIOLATIONS -gt 0 ]; then
    echo -e "${RED}ERROR: Found Python naming convention violations in code examples${NC}"
    if [ $PYTHON_CLASS_VIOLATIONS -gt 0 ]; then
        echo -e "${RED}Found $PYTHON_CLASS_VIOLATIONS instances of Python classes not using PascalCase${NC}"
        grep -r "class [a-z]" --include="*.md" ./docs
    fi
    if [ $PYTHON_METHOD_VIOLATIONS -gt 0 ]; then
        echo -e "${RED}Found $PYTHON_METHOD_VIOLATIONS instances of Python methods not using snake_case${NC}"
        grep -r "def [A-Z]" --include="*.md" ./docs
    fi
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: No Python naming convention violations found in code examples${NC}"
fi

# TypeScript naming conventions
echo "Checking TypeScript naming conventions..."
TS_INTERFACE_VIOLATIONS=$(grep -r "interface [^I][A-Z]" --include="*.md" ./docs | wc -l)
TS_COMPONENT_VIOLATIONS=$(grep -r "const [a-z].* = () =>" --include="*.md" ./docs | wc -l)

if [ $TS_INTERFACE_VIOLATIONS -gt 0 ] || [ $TS_COMPONENT_VIOLATIONS -gt 0 ]; then
    echo -e "${RED}ERROR: Found TypeScript naming convention violations in code examples${NC}"
    if [ $TS_INTERFACE_VIOLATIONS -gt 0 ]; then
        echo -e "${RED}Found $TS_INTERFACE_VIOLATIONS instances of TypeScript interfaces not using 'I' prefix${NC}"
        grep -r "interface [^I][A-Z]" --include="*.md" ./docs
    fi
    if [ $TS_COMPONENT_VIOLATIONS -gt 0 ]; then
        echo -e "${RED}Found $TS_COMPONENT_VIOLATIONS instances of TypeScript components not using PascalCase${NC}"
        grep -r "const [a-z].* = () =>" --include="*.md" ./docs
    fi
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}OK: No TypeScript naming convention violations found in code examples${NC}"
fi

echo

# ==========================================
# Summary
# ==========================================
echo -e "${YELLOW}=== Validation Summary ===${NC}"
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}Found $ERRORS error(s) in documentation${NC}"
    echo -e "${RED}Documentation does not fully adhere to ADR-002, ADR-003, and ADR-004${NC}"
    exit 1
else
    echo -e "${GREEN}No errors found in documentation${NC}"
    echo -e "${GREEN}Documentation adheres to ADR-002, ADR-003, and ADR-004${NC}"
    exit 0
fi
