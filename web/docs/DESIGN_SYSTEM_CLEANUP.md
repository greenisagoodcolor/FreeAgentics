# Design System Cleanup Strategy

## Current Problem

Multiple UI libraries causing test conflicts:

- Radix UI (primary)
- Vaul (drawers)
- Sonner (toasts)
- CMDK (command palette)
- Embla Carousel
- React Day Picker
- Recharts
- React Sparklines

## **Solution 1: Radix UI + shadcn/ui Consolidation (Recommended)**

### Step 1: Standardize on shadcn/ui Components

```bash
# Remove conflicting libraries
npm uninstall vaul sonner cmdk embla-carousel-react react-day-picker

# Keep only essential non-conflicting ones
# - recharts (charts)
# - react-sparklines (if needed)
# - d3 (visualizations)
```

### Step 2: Replace Components

- **Vaul drawers** → Radix Dialog
- **Sonner toasts** → Radix Toast (already installed)
- **CMDK** → Custom Radix Command
- **Embla Carousel** → Custom Radix component
- **React Day Picker** → Custom Radix Calendar

### Step 3: Centralized Component Library

```typescript
// components/ui/index.ts - Single export point
export * from "./button";
export * from "./select";
export * from "./dialog";
export * from "./toast";
// etc.
```

## **Solution 2: Component Facade Pattern**

Create wrapper components that hide implementation details:

```typescript
// components/design-system/Select.tsx
export const DSSelect = ({ children, ...props }) => {
  // Internal implementation can change without breaking tests
  return <RadixSelect {...props}>{children}</RadixSelect>;
};
```

## **Solution 3: Test-Only Mocking Strategy**

```typescript
// __tests__/utils/ui-mocks.ts
export const createSelectMock = () => ({
  Select: ({ children, onValueChange }) => (
    <div data-testid="mock-select">{children}</div>
  ),
  // Consistent mocks for all tests
});
```

## **Implementation Priority**

1. **Immediate**: Remove Vaul, Sonner, CMDK
2. **Short-term**: Replace with Radix equivalents
3. **Long-term**: Create design system documentation

## **Benefits**

- Single source of truth for UI
- Consistent testing patterns
- Reduced bundle size
- Better maintenance
- No more mock conflicts
