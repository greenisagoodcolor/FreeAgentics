import React from 'react';

// Simple mock factory
export const createUIComponentMock = (tagName: string, role?: string) => {
  return React.forwardRef<HTMLElement, any>(({ children, ...props }, ref) => 
    React.createElement(tagName, { ref, role, ...props }, children)
  );
};

export default {
  createUIComponentMock,
};