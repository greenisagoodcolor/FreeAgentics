# FreeAgentics Directory Structure Analysis

## Current State (CONFUSING)

```
/home/green/FreeAgentics/                    # Git repo: github.com/greenisagoodcolor/FreeAgentics
│
├── freeagentics2/                          # 275 Python files - appears to be an old copy
│   ├── freeagentics/                       # More nesting
│   └── freeagentics2/                      # Even more nesting
│
├── freeagentics2-fresh/                    # 136 Python files - staging/test environment?
│   └── (full application structure)
│
├── freeagentics2-nemesis/                  # 136 Python files - ACTIVE DEVELOPMENT
│   ├── .git/                               # Points to: github.com/greenisagoodcolor/freeagentics2
│   ├── agents/                             # Active Inference agent implementations
│   ├── api/                                # FastAPI backend
│   ├── database/                           # SQLAlchemy models
│   ├── web/                                # Next.js frontend
│   ├── freeagentics/                       # Nearly empty (legacy)
│   └── freeagentics2/                      # Empty (legacy)
│
├── new-dev-test/                           # Another test environment
│
└── (root files like Makefile, README)      # Which ones are real?
```

## Analysis Results

1. **freeagentics2-nemesis/** is the active development directory:
   - Has recent git commits
   - Contains the most organized code structure
   - Has active CI/CD configurations
   - Contains the Nemesis committee work

2. **freeagentics2/** appears to be an older copy with many duplicate files

3. **freeagentics2-fresh/** seems to be a staging or test environment

4. **Root directory** has some Makefiles and READMEs but it's unclear if they're current

## The Core Problem

- We have TWO git repositories mixed together:
  1. `/home/green/FreeAgentics/` (FreeAgentics.git)
  2. `/home/green/FreeAgentics/freeagentics2-nemesis/` (freeagentics2.git)

This is causing massive confusion for developers!

## Recommended Solution

### Option 1: Consolidate to Root (RECOMMENDED)
Move the contents of `freeagentics2-nemesis/` to the root, making it the primary structure.

### Option 2: Submodule Approach
Make `freeagentics2-nemesis` a proper git submodule with clear documentation.

### Option 3: Complete Separation
Move `freeagentics2-nemesis` outside and maintain as separate project.

## Immediate Actions Needed

1. **DECIDE**: Which git repository should be the canonical one?
2. **BACKUP**: Create full backup before any changes
3. **MIGRATE**: Move chosen structure to be the primary
4. **ARCHIVE**: Move old directories to `.archived/` with README explaining why
5. **DOCUMENT**: Update all READMEs to explain the structure clearly