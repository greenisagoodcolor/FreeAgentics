// Unmock UI components for actual testing
jest.unmock("@/components/ui/button");
jest.unmock("@/components/ui/badge");
jest.unmock("@/components/ui/scroll-area");
jest.unmock("@/components/ui/input");
jest.unmock("@/components/ui/select");
jest.unmock("@/components/ui/checkbox");
jest.unmock("@/components/ui/dialog");
jest.unmock("@/components/ui/toast");
jest.unmock("@/components/ui/alert");
jest.unmock("@/components/ui/dropdown-menu");
jest.unmock("@/components/ui/command");

// Import React for the mock
import React from "react";

// Mock radix-ui components that are used internally
jest.mock("@radix-ui/react-slot", () => ({
  Slot: React.forwardRef(({ children, ...props }: any, ref: any) => {
    // Clone the child element and pass down all props including ref
    if (React.isValidElement(children)) {
      return React.cloneElement(children as any, { ...props, ref });
    }
    return children;
  }),
}));
