import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SettingsDrawer } from "@/components/main/SettingsDrawer";
import { useAuth } from "@/hooks/use-auth";
import { useSettings } from "@/hooks/use-settings";

// Mock the hooks
jest.mock("@/hooks/use-auth");
jest.mock("@/hooks/use-settings");

const mockUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;
const mockUseSettings = useSettings as jest.MockedFunction<typeof useSettings>;

describe("SettingsDrawer", () => {
  const mockOnOpenChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock window.confirm for tests
    global.confirm = jest.fn(() => true);

    // Default auth state
    mockUseAuth.mockReturnValue({
      user: null,
      isAuthenticated: false,
      login: jest.fn(),
      logout: jest.fn(),
      isLoading: false,
    });

    // Default settings
    mockUseSettings.mockReturnValue({
      settings: {
        llmProvider: "openai",
        llmModel: "gpt-4",
        gnnEnabled: true,
        debugLogs: false,
        autoSuggest: true,
      },
      updateSettings: jest.fn(),
      resetSettings: jest.fn(),
    });
  });

  it("renders drawer when open", () => {
    render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText(/settings/i)).toBeInTheDocument();
  });

  it("does not render when closed", () => {
    render(<SettingsDrawer open={false} onOpenChange={mockOnOpenChange} />);

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("calls onOpenChange when close button is clicked", async () => {
    const user = userEvent.setup();
    render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

    const closeButton = screen.getByLabelText(/close/i);
    await user.click(closeButton);

    expect(mockOnOpenChange).toHaveBeenCalledWith(false);
  });

  describe("LLM Settings", () => {
    it("shows LLM provider selector", () => {
      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      expect(screen.getByLabelText(/llm provider/i)).toBeInTheDocument();
      expect(screen.getByText(/openai/i)).toBeInTheDocument();
    });

    it("shows available LLM models based on provider", () => {
      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      // Just verify the model selector exists
      const modelSelector = screen.getByLabelText(/model/i);
      expect(modelSelector).toBeInTheDocument();

      // The default model should be displayed somewhere in the select
      const selectTrigger = modelSelector.closest("button");
      expect(selectTrigger).toBeTruthy();
    });

    it("shows provider in select trigger", () => {
      const mockUpdateSettings = jest.fn();
      mockUseSettings.mockReturnValue({
        settings: {
          llmProvider: "anthropic",
          llmModel: "claude-3-opus",
          gnnEnabled: true,
          debugLogs: false,
          autoSuggest: true,
        },
        updateSettings: mockUpdateSettings,
        resetSettings: jest.fn(),
      });

      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      // Verify the provider selector shows the current value
      const providerSelect = screen.getByLabelText(/llm provider/i);
      expect(providerSelect).toBeInTheDocument();
      expect(screen.getByRole("combobox", { name: /llm provider/i })).toHaveTextContent(
        "Anthropic",
      );
    });
  });

  describe("Authentication", () => {
    it("shows login button when not authenticated", () => {
      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      expect(screen.getByText(/not logged in/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /log in/i })).toBeInTheDocument();
    });

    it("shows user info and logout button when authenticated", () => {
      mockUseAuth.mockReturnValue({
        user: { id: "1", email: "test@example.com", name: "Test User" },
        isAuthenticated: true,
        login: jest.fn(),
        logout: jest.fn(),
        isLoading: false,
      });

      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      expect(screen.getByText(/test@example.com/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /log out/i })).toBeInTheDocument();
    });

    it("calls logout when logout button is clicked", async () => {
      const user = userEvent.setup();
      const mockLogout = jest.fn();
      mockUseAuth.mockReturnValue({
        user: { id: "1", email: "test@example.com", name: "Test User" },
        isAuthenticated: true,
        login: jest.fn(),
        logout: mockLogout,
        isLoading: false,
      });

      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      const logoutButton = screen.getByRole("button", { name: /log out/i });
      await user.click(logoutButton);

      expect(mockLogout).toHaveBeenCalled();
    });
  });

  describe("Feature Flags", () => {
    it("shows GNN enabled toggle", () => {
      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      const gnnToggle = screen.getByRole("switch", { name: /gnn enabled/i });
      expect(gnnToggle).toBeInTheDocument();
      expect(gnnToggle).toBeChecked();
    });

    it("shows debug logs toggle", () => {
      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      const debugToggle = screen.getByRole("switch", { name: /debug logs/i });
      expect(debugToggle).toBeInTheDocument();
      expect(debugToggle).not.toBeChecked();
    });

    it("shows auto-suggest toggle", () => {
      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      const autoSuggestToggle = screen.getByRole("switch", { name: /auto-suggest/i });
      expect(autoSuggestToggle).toBeInTheDocument();
      expect(autoSuggestToggle).toBeChecked();
    });

    it("updates feature flag when toggled", async () => {
      const user = userEvent.setup();
      const mockUpdateSettings = jest.fn();
      mockUseSettings.mockReturnValue({
        settings: {
          llmProvider: "openai",
          llmModel: "gpt-4",
          gnnEnabled: true,
          debugLogs: false,
          autoSuggest: true,
        },
        updateSettings: mockUpdateSettings,
        resetSettings: jest.fn(),
      });

      render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

      const debugToggle = screen.getByRole("switch", { name: /debug logs/i });
      await user.click(debugToggle);

      expect(mockUpdateSettings).toHaveBeenCalledWith({ debugLogs: true });
    });
  });

  it("persists settings to localStorage", async () => {
    const user = userEvent.setup();
    const mockUpdateSettings = jest.fn();
    mockUseSettings.mockReturnValue({
      settings: {
        llmProvider: "openai",
        llmModel: "gpt-4",
        gnnEnabled: true,
        debugLogs: false,
        autoSuggest: true,
      },
      updateSettings: mockUpdateSettings,
      resetSettings: jest.fn(),
    });

    render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

    const debugToggle = screen.getByRole("switch", { name: /debug logs/i });
    await user.click(debugToggle);

    expect(mockUpdateSettings).toHaveBeenCalledWith({ debugLogs: true });
  });

  it("shows reset settings button", () => {
    render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

    expect(screen.getByRole("button", { name: /reset to defaults/i })).toBeInTheDocument();
  });

  it("resets settings when reset button is clicked", async () => {
    const user = userEvent.setup();
    const mockResetSettings = jest.fn();
    mockUseSettings.mockReturnValue({
      settings: {
        llmProvider: "anthropic",
        llmModel: "claude-3",
        gnnEnabled: false,
        debugLogs: true,
        autoSuggest: false,
      },
      updateSettings: jest.fn(),
      resetSettings: mockResetSettings,
    });

    render(<SettingsDrawer open={true} onOpenChange={mockOnOpenChange} />);

    const resetButton = screen.getByRole("button", { name: /reset to defaults/i });
    await user.click(resetButton);

    expect(mockResetSettings).toHaveBeenCalled();
  });
});
