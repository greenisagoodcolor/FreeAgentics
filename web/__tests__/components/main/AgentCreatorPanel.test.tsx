import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AgentCreatorPanel } from "@/components/main/AgentCreatorPanel";
import { useWebSocket } from "@/hooks/use-websocket";
import { useAgents } from "@/hooks/use-agents";

// Mock the hooks
jest.mock("@/hooks/use-websocket");
jest.mock("@/hooks/use-agents");

const mockUseWebSocket = useWebSocket as jest.MockedFunction<typeof useWebSocket>;
const mockUseAgents = useAgents as jest.MockedFunction<typeof useAgents>;

describe("AgentCreatorPanel", () => {
  const mockSendMessage = jest.fn();
  const mockCreateAgent = jest.fn();
  const mockUpdateAgent = jest.fn();
  const mockDeleteAgent = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    // Default WebSocket state
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      sendMessage: mockSendMessage,
      lastMessage: null,
      connectionState: "connected",
      error: null,
    });

    // Default agents state
    mockUseAgents.mockReturnValue({
      agents: [],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });
  });

  it("renders agent creator panel", () => {
    render(<AgentCreatorPanel />);

    expect(screen.getByText(/agent creator/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/describe the agent/i)).toBeInTheDocument();
  });

  it("shows connection status indicator", () => {
    render(<AgentCreatorPanel />);

    // Should show connected status
    expect(screen.getByTestId("connection-status")).toHaveClass("bg-green-500");
  });

  it("shows disconnected status when not connected", () => {
    mockUseWebSocket.mockReturnValue({
      isConnected: false,
      sendMessage: mockSendMessage,
      lastMessage: null,
      connectionState: "disconnected",
      error: null,
    });

    render(<AgentCreatorPanel />);

    expect(screen.getByTestId("connection-status")).toHaveClass("bg-red-500");
  });

  it("creates agent when form is submitted", async () => {
    const user = userEvent.setup();
    render(<AgentCreatorPanel />);

    const input = screen.getByPlaceholderText(/describe the agent/i);
    const submitButton = screen.getByRole("button", { name: /create agent/i });

    await user.type(input, "Create an explorer agent that searches for resources");
    await user.click(submitButton);

    expect(mockCreateAgent).toHaveBeenCalledWith({
      description: "Create an explorer agent that searches for resources",
    });
  });

  it("clears input after successful agent creation", async () => {
    const user = userEvent.setup();
    mockCreateAgent.mockResolvedValueOnce({ id: "1", name: "Explorer Agent" });

    render(<AgentCreatorPanel />);

    const input = screen.getByPlaceholderText(/describe the agent/i) as HTMLInputElement;
    await user.type(input, "Create an explorer agent");
    await user.click(screen.getByRole("button", { name: /create agent/i }));

    await waitFor(() => {
      expect(input.value).toBe("");
    });
  });

  it("shows loading state while creating agent", async () => {
    const user = userEvent.setup();
    mockCreateAgent.mockImplementation(() => new Promise(() => {})); // Never resolves

    render(<AgentCreatorPanel />);

    await user.type(screen.getByPlaceholderText(/describe the agent/i), "Test agent");
    await user.click(screen.getByRole("button", { name: /create agent/i }));

    expect(screen.getByRole("button", { name: /creating/i })).toBeDisabled();
  });

  it("displays list of existing agents", () => {
    mockUseAgents.mockReturnValue({
      agents: [
        { id: "1", name: "Explorer Agent", type: "explorer", status: "active" },
        { id: "2", name: "Resource Collector", type: "collector", status: "idle" },
      ],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });

    render(<AgentCreatorPanel />);

    expect(screen.getByText("Explorer Agent")).toBeInTheDocument();
    expect(screen.getByText("Resource Collector")).toBeInTheDocument();
  });

  it("shows agent status indicators", () => {
    mockUseAgents.mockReturnValue({
      agents: [
        { id: "1", name: "Active Agent", type: "explorer", status: "active" },
        { id: "2", name: "Idle Agent", type: "collector", status: "idle" },
        { id: "3", name: "Error Agent", type: "explorer", status: "error" },
      ],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });

    render(<AgentCreatorPanel />);

    // Check for status indicators
    expect(screen.getByTestId("agent-status-1")).toHaveClass("bg-green-500");
    expect(screen.getByTestId("agent-status-2")).toHaveClass("bg-yellow-500");
    expect(screen.getByTestId("agent-status-3")).toHaveClass("bg-red-500");
  });

  it("allows editing agent properties", async () => {
    const user = userEvent.setup();
    mockUseAgents.mockReturnValue({
      agents: [{ id: "1", name: "Explorer Agent", type: "explorer", status: "active" }],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });

    render(<AgentCreatorPanel />);

    // Click edit button
    await user.click(screen.getByLabelText(/edit Explorer Agent/i));

    // Should show edit form
    expect(screen.getByDisplayValue("Explorer Agent")).toBeInTheDocument();
  });

  it("updates agent when edit form is submitted", async () => {
    const user = userEvent.setup();
    mockUseAgents.mockReturnValue({
      agents: [{ id: "1", name: "Explorer Agent", type: "explorer", status: "active" }],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });

    render(<AgentCreatorPanel />);

    // Enter edit mode
    await user.click(screen.getByLabelText(/edit Explorer Agent/i));

    // Change name
    const nameInput = screen.getByDisplayValue("Explorer Agent");
    await user.clear(nameInput);
    await user.type(nameInput, "Super Explorer");

    // Submit
    await user.click(screen.getByRole("button", { name: /save/i }));

    expect(mockUpdateAgent).toHaveBeenCalledWith("1", {
      name: "Super Explorer",
    });
  });

  it("allows deleting agents", async () => {
    const user = userEvent.setup();
    mockUseAgents.mockReturnValue({
      agents: [{ id: "1", name: "Explorer Agent", type: "explorer", status: "idle" }],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });

    render(<AgentCreatorPanel />);

    // Click delete button
    await user.click(screen.getByLabelText(/delete Explorer Agent/i));

    expect(mockDeleteAgent).toHaveBeenCalledWith("1");
  });

  it("disables delete for active agents", () => {
    mockUseAgents.mockReturnValue({
      agents: [{ id: "1", name: "Active Agent", type: "explorer", status: "active" }],
      createAgent: mockCreateAgent,
      updateAgent: mockUpdateAgent,
      deleteAgent: mockDeleteAgent,
      isLoading: false,
      error: null,
    });

    render(<AgentCreatorPanel />);

    const deleteButton = screen.getByLabelText(/delete Active Agent/i);
    expect(deleteButton).toBeDisabled();
  });

  it("shows error message when agent creation fails", async () => {
    const user = userEvent.setup();
    mockCreateAgent.mockRejectedValueOnce(new Error("Failed to create agent"));

    render(<AgentCreatorPanel />);

    await user.type(screen.getByPlaceholderText(/describe the agent/i), "Test agent");
    await user.click(screen.getByRole("button", { name: /create agent/i }));

    await waitFor(() => {
      expect(screen.getByText(/failed to create agent/i)).toBeInTheDocument();
    });
  });

  it("handles WebSocket messages for agent updates", () => {
    const { rerender } = render(<AgentCreatorPanel />);

    // Simulate receiving a WebSocket message
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      sendMessage: mockSendMessage,
      lastMessage: {
        type: "agent_update",
        data: {
          agentId: "1",
          status: "active",
        },
      },
      connectionState: "connected",
      error: null,
    });

    rerender(<AgentCreatorPanel />);

    // Component should handle the update (implementation will determine exact behavior)
    expect(screen.getByText(/agent creator/i)).toBeInTheDocument();
  });

  it("shows empty state when no agents exist", () => {
    render(<AgentCreatorPanel />);

    expect(screen.getByText(/no agents created yet/i)).toBeInTheDocument();
    expect(screen.getByText(/create your first agent/i)).toBeInTheDocument();
  });

  it("allows keyboard submission with Enter key", async () => {
    const user = userEvent.setup();
    render(<AgentCreatorPanel />);

    const input = screen.getByPlaceholderText(/describe the agent/i);
    await user.type(input, "Create a test agent");
    await user.keyboard("{Enter}");

    expect(mockCreateAgent).toHaveBeenCalledWith({
      description: "Create a test agent",
    });
  });
});
