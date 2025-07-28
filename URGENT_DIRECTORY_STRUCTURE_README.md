# 🚨 URGENT: Directory Structure Clarification 🚨

## FOR NEW DEVELOPERS - READ THIS FIRST!

The FreeAgentics project currently has a VERY CONFUSING directory structure. This document explains what's happening and how to navigate it.

## The Current Mess

```
/home/green/FreeAgentics/                    # ← You are here (main git repo)
├── freeagentics2-nemesis/                   # ← THE REAL CODE IS HERE! 
├── freeagentics2/                           # ← Old/duplicate - IGNORE
├── freeagentics2-fresh/                     # ← Test environment - IGNORE  
└── new-dev-test/                            # ← Test environment - IGNORE
```

## 🎯 THE TRUTH

**ALL ACTIVE DEVELOPMENT IS IN: `/home/green/FreeAgentics/freeagentics2-nemesis/`**

## Quick Start for New Developers

```bash
# 1. Go to the REAL directory
cd /home/green/FreeAgentics/freeagentics2-nemesis/

# 2. This is where you run everything
make install
make dev
make test

# 3. DO NOT use the root Makefile or other directories!
```

## Why This Mess Exists

1. The project started as `FreeAgentics`
2. Someone created `freeagentics2` as a rewrite
3. Multiple parallel attempts created `freeagentics2-fresh` and `freeagentics2-nemesis`
4. The Nemesis Committee chose `freeagentics2-nemesis` as the canonical version
5. Nobody cleaned up the old directories (yet)

## File Locations in freeagentics2-nemesis/

- **Backend API**: `freeagentics2-nemesis/api/`
- **Agents**: `freeagentics2-nemesis/agents/`
- **Frontend**: `freeagentics2-nemesis/web/`
- **Tests**: `freeagentics2-nemesis/tests/`
- **Database**: `freeagentics2-nemesis/database/`

## Git Repository Confusion

- Root directory (`.`) is repo: `github.com/greenisagoodcolor/FreeAgentics`
- But `freeagentics2-nemesis/` has its own `.git` pointing to: `github.com/greenisagoodcolor/freeagentics2`

**For now, make changes in freeagentics2-nemesis/ and push from there!**

## Future Plan

We will consolidate everything into a clean structure at the root level. But for now, to avoid breaking the working system:

**USE `/home/green/FreeAgentics/freeagentics2-nemesis/` FOR EVERYTHING**

---

Last updated: 2025-07-28
Status: Temporary workaround until proper restructuring