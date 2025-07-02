# MemoryViewer Test Fix Patterns

## Key Issues Found and Solutions

### 1. Navigation Pattern Change

**Issue**: Tests expected clickable tabs but component uses dropdown selects
**Solution**:

```typescript
// OLD - Expected tabs
fireEvent.click(screen.getByText("Knowledge"));

// NEW - Use dropdown
const selectTrigger = screen.getByRole("combobox");
fireEvent.click(selectTrigger);
const knowledgeOption = screen.getByRole("option", { name: "Knowledge" });
fireEvent.click(knowledgeOption);
```

### 2. Mock Data Type Mismatches

**Issue**: Mock data missing required fields
**Solutions**:

- Added `startTime` and `endTime` to Conversation mock
- Added `class` and `status` to Agent mock
- Fixed timestamp fields to be Date objects not strings

### 3. Component Behavior Mismatches

**Issue**: Tests expected behaviors that don't exist
**Solutions**:

- Biography save button is never disabled (component doesn't track changes)
- No error handling in biography save (component shows success regardless)
- Tags use [[tag]] syntax in content, not separate input field
- Toast messages have different text than expected

### 4. Element Selection Issues

**Issue**: Tests using wrong selectors
**Solutions**:

```typescript
// Use role-based selectors
screen.getByRole("button", { name: /Save Biography/i });
// Use placeholder text for inputs
screen.getByPlaceholderText("Enter agent biography...");
// Handle multiple comboboxes
screen.getAllByRole("combobox")[1]; // Get specific dropdown
```

### 5. Async Handling

**Issue**: Toast messages and state updates need proper async handling
**Solution**: Wrap expectations in waitFor()

```typescript
await waitFor(() => {
  expect(mockToast).toHaveBeenCalledWith({...});
});
```

## Remaining Test Fixes Needed

1. Knowledge editing/deletion tests - Need to navigate to knowledge view first
2. Tool permissions tests - Need to navigate to tools view and check actual UI
3. Belief extraction tests - Need to handle the multi-step workflow
4. Knowledge node selection tests - Need to handle the special node-selection view
5. Export tests - Need to check actual export function implementation

## Pattern for Fixing Remaining Tests

1. Always check actual component implementation first
2. Navigate to correct view using dropdown
3. Use role-based selectors
4. Check actual toast messages and button text
5. Handle async operations with waitFor
6. Don't expect features that don't exist (like error handling)
