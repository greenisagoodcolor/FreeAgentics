import React, { ReactElement } from "react";
import { render, RenderOptions } from "@testing-library/react";

// Mock Next.js router
const mockRouter = {
  push: jest.fn(),
  replace: jest.fn(),
  back: jest.fn(),
  forward: jest.fn(),
  refresh: jest.fn(),
  prefetch: jest.fn(),
};

jest.mock("next/navigation", () => ({
  useRouter: () => mockRouter,
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

// Mock auth context
const mockAuthContext = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  login: jest.fn(),
  logout: jest.fn(),
};

jest.mock("@/hooks/use-auth", () => ({
  useAuth: () => mockAuthContext,
}));

// Mock settings context
const mockSettingsContext = {
  settings: {
    llmProvider: "openai" as const,
    llmModel: "gpt-4",
    gnnEnabled: false,
    debugLogs: false,
    autoSuggest: true,
  },
  updateSettings: jest.fn(),
  resetSettings: jest.fn(),
};

jest.mock("@/hooks/use-settings", () => ({
  useSettings: () => mockSettingsContext,
  LLM_MODELS: {
    openai: ["gpt-4", "gpt-3.5-turbo"],
    anthropic: ["claude-3-opus", "claude-3-sonnet"],
    ollama: ["llama2", "codellama"],
  },
}));

// Test provider wrapper
function TestProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

// Custom render function
function customRender(ui: ReactElement, options?: Omit<RenderOptions, "wrapper">) {
  return render(ui, { wrapper: TestProvider, ...options });
}

// Re-export everything
export * from "@testing-library/react";
export { customRender as render };
export { mockRouter, mockAuthContext, mockSettingsContext };
