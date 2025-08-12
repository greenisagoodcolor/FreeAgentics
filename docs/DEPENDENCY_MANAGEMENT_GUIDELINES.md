# Dependency Management Guidelines

## Overview

This document establishes guidelines for managing dependencies across the FreeAgentics monorepo to prevent conflicts and ensure consistent developer experience.

## Architecture Principles

### Separation of Concerns
- **Root package.json**: Contains only project orchestration tools and shared development dependencies
- **web/package.json**: Contains all frontend-specific dependencies (React, Next.js, UI libraries)
- **Python requirements.txt**: Contains all backend-specific dependencies

### Allowed Root Dependencies
Only these types of dependencies should exist in root package.json:

```json
{
  "devDependencies": {
    "concurrently": "orchestration tool",
    "jest-websocket-mock": "shared testing utilities", 
    "@babel/preset-typescript": "shared build tools",
    "ts-jest": "shared testing configuration",
    "@types/jest": "shared type definitions"
  }
}
```

### Prohibited Root Dependencies
Never add these to root package.json:

- **Framework-specific packages**: `next`, `react`, `vue`, `svelte`
- **UI libraries**: `lucide-react`, `@heroicons/react`, `tailwindcss`
- **Web-specific tools**: `web-vitals`, `@next/bundle-analyzer`
- **Domain-specific utilities**: Any package used only by frontend or backend

## Version Consistency Requirements

### Node.js Versions
All package.json files must specify identical Node.js requirements:

```json
{
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=8.0.0"
  }
}
```

### Package Version Alignment
When the same package appears in multiple locations (allowed only for build tools):

- Use identical version ranges
- Document the reason for duplication
- Regularly audit for consolidation opportunities

## Testing Requirements

### Automated Validation
Every PR must pass:

```bash
# Onboarding validation
make test-onboarding

# Dependency conflict detection  
npm ls --all 2>/dev/null | grep -i conflict && exit 1
```

### Manual Testing Scenarios
Before major releases:

1. **Fresh Clone Test**: Clone repo in clean environment, run `make install`
2. **Cross-Platform Test**: Validate on Windows, macOS, Linux  
3. **Version Matrix Test**: Test with minimum and maximum supported Node.js versions

## Package-lock.json Management

### Commit Strategy
- **ALWAYS commit** `web/package-lock.json` for reproducible frontend builds
- **NEVER commit** root-level `package-lock.json` (orchestration only)
- Update lock files atomically with dependency changes

### CI/CD Integration
```yaml
# Example GitHub Actions validation
- name: Validate dependency integrity
  run: |
    npm ci --prefix web
    npm audit --prefix web --audit-level=moderate
```

## Conflict Resolution Procedures

### When Conflicts Arise

1. **Identify Scope**: Is this a frontend, backend, or orchestration concern?
2. **Apply Separation**: Move dependencies to appropriate package.json
3. **Test Thoroughly**: Run full onboarding validation
4. **Document Decision**: Update this guide if new patterns emerge

### Common Conflict Patterns

| Conflict Type | Solution | Prevention |
|---------------|----------|------------|
| Version Mismatch | Align versions across files | Automated version checking |
| Duplicate Dependencies | Remove from inappropriate location | Clear ownership guidelines |
| Peer Dependency Issues | Ensure peer deps match primary deps | Regular dependency audits |

## Maintenance Procedures

### Weekly Tasks
- Run `npm outdated` in all package directories
- Check for security advisories: `npm audit`
- Validate onboarding process: `make test-onboarding`

### Monthly Tasks  
- Review and consolidate dependencies
- Update dependency management guidelines
- Audit for unused dependencies: `depcheck`

### Before Major Releases
- Full cross-platform onboarding validation
- Dependency vulnerability scanning
- Performance impact analysis of version changes

## Tools and Automation

### Recommended Tools
- **depcheck**: Find unused dependencies
- **npm-check-updates**: Update dependencies systematically  
- **audit-ci**: Fail builds on security vulnerabilities
- **syncpack**: Maintain version consistency across workspace

### Integration Examples

```bash
# Add to package.json scripts
{
  "scripts": {
    "deps:check": "depcheck --ignores='@types/*'",
    "deps:update": "ncu -u && npm install",
    "deps:audit": "npm audit --audit-level=moderate"
  }
}
```

## Troubleshooting Guide

### Installation Hangs
- **Symptom**: `npm install` never completes
- **Cause**: Version conflicts in dependency tree
- **Solution**: Remove conflicting versions, clear cache, retry

### "Cannot resolve dependency" Errors  
- **Symptom**: Build fails with peer dependency warnings
- **Cause**: Mismatched versions between primary and peer dependencies
- **Solution**: Align peer dependency versions with primary packages

### Environment Inconsistencies
- **Symptom**: "Works on my machine" installation issues
- **Cause**: Different Node.js versions or missing lock files
- **Solution**: Use .nvmrc, commit lock files, validate environments

## Example Package.json Structure

### Root package.json (Orchestration Only)
```json
{
  "name": "freeagentics",
  "private": true,
  "scripts": {
    "dev": "concurrently --kill-others-on-fail \"npm run dev:backend\" \"npm run dev:frontend\"",
    "install:all": "npm install && cd web && npm install"
  },
  "devDependencies": {
    "concurrently": "^8.2.2",
    "@types/jest": "^30.0.0"
  },
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=8.0.0"
  }
}
```

### web/package.json (Frontend Dependencies)
```json
{
  "name": "freeagentics-web", 
  "dependencies": {
    "next": "14.2.30",
    "react": "^18.2.0",
    "lucide-react": "^0.525.0"
  },
  "engines": {
    "node": ">=20.0.0"
  }
}
```

---

*This document should be updated whenever new dependency patterns or conflict resolution procedures are discovered.*