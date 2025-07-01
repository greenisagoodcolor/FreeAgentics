// This file contains fixes for TypeScript errors

// Fix 1: Update conversation mock data structure
export const createMockConversation = (overrides?: any) => ({
  id: "conv-1",
  participants: ["agent-1", "agent-2"],
  messages: [],
  startTime: new Date(Date.now() - 7200000),
  endTime: null, // This was missing
  ...overrides,
});

// Fix 2: Update message mock data structure
export const createMockMessage = (overrides?: any) => ({
  id: "msg-1",
  senderId: "agent-1",
  content: "Test message",
  timestamp: new Date(), // Must be Date, not number
  ...overrides,
});

// Fix 3: Missing import
export { within } from "@testing-library/react";
