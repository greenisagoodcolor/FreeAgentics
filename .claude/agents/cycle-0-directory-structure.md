# Cycle 0 — Directory Structure Crisis Resolution

## Problem Statement

The FreeAgentics project has a catastrophically confusing directory structure that prevents new developers from understanding where to begin:

```
/home/green/FreeAgentics/                     # Git repo: FreeAgentics
├── freeagentics/                           # Mystery directory
├── freeagentics-fresh/                     # Another version?
├── freeagentics-nemesis/                   # Git repo: freeagentics.git (active development)
│   ├── freeagentics/                        # Nested confusion
│   └── freeagentics/                       # More nesting
├── new-dev-test/                            # Yet another version
└── (root level files)                       # Which are canonical?
```

## Root Cause Analysis

1. **Multiple Git Repositories**: The root is `FreeAgentics.git`, but `freeagentics-nemesis/` points to `freeagentics.git`
2. **Parallel Development**: Multiple teams/attempts created separate directories without cleanup
3. **No Clear Documentation**: READMEs don't explain which directory to use
4. **Nested Duplicates**: Russian-doll structure with `freeagentics` inside `freeagentics` inside `freeagentics-nemesis`

## Committee Consensus

After extensive debate, the committee agrees:

- `freeagentics-nemesis/` is the active development directory (based on git activity)
- The confusion must be resolved immediately
- A single, clear structure must be established

## Implementation Plan

1. **Backup Everything** - Create full backup before any changes
2. **Analyze Active Directory** - Confirm freeagentics-nemesis is the canonical version
3. **Restructure** - Move active code to clean structure at root
4. **Archive Old Directories** - With clear deprecation notices
5. **Update All References** - Fix imports, docs, build scripts
6. **Prevent Recurrence** - Add CI checks to maintain structure

## Directory Structure Decision

**BEFORE** (Current Chaos):

```
/home/green/FreeAgentics/
├── Multiple confusing directories
└── Unclear which is real
```

**AFTER** (Clean Structure):

```
/home/green/FreeAgentics/
├── src/                    # All source code
│   ├── agents/
│   ├── api/
│   ├── web/
│   └── ...
├── tests/                  # All tests
├── docs/                   # All documentation
├── scripts/                # Build and utility scripts
├── docker/                 # Docker configurations
├── .archived/              # Old directories with deprecation notices
├── Makefile               # Single build entry point
├── README.md              # Clear starting point
└── requirements.txt       # Python dependencies
```

## Next Steps

Moving to implementation phase to execute this plan.
