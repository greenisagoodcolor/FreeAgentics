/**
 * Working test implementation for ConversationSearch - following TDD
 */

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConversationSearch } from "../ConversationSearchSimple";
import type { ConversationMessage } from "../ConversationPanel";
import type { Agent } from "@/lib/types";

describe("ConversationSearch - Working Implementation", () => {
  const mockMessages: ConversationMessage[] = [
    {
      id: "msg-1",
      agentId: "agent-1",
      agent_id: "agent-1",
      conversation_id: "conv-1",
      user_id: "user-1",
      message: "Hello world, this is a test message",
      content: "Hello world, this is a test message",
      message_type: "user",
      type: "user",
      timestamp: "2023-01-01T10:00:00Z",
    },
    {
      id: "msg-2",
      agentId: "agent-1",
      agent_id: "agent-1",
      conversation_id: "conv-1",
      message: "This is an agent response about testing",
      content: "This is an agent response about testing",
      message_type: "agent",
      type: "agent",
      timestamp: "2023-01-01T10:01:00Z",
    },
    {
      id: "msg-3",
      agentId: "agent-1",
      agent_id: "agent-1",
      conversation_id: "conv-2",
      user_id: "user-2",
      message: "Another message in a different conversation",
      content: "Another message in a different conversation",
      message_type: "user",
      type: "user",
      timestamp: "2023-01-02T15:30:00Z",
    },
  ];

  const mockAgents: Agent[] = [
    {
      id: "agent-1",
      name: "Test Agent",
      template: "assistant",
      status: "active",
      pymdp_config: {},
      beliefs: {},
      preferences: {},
      metrics: {},
      parameters: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      inference_count: 0,
      total_steps: 0,
    },
  ];

  const mockUsers = [
    { id: "user-1", name: "John Doe" },
    { id: "user-2", name: "Jane Smith" },
  ];

  it("renders basic search interface", () => {
    render(<ConversationSearch messages={mockMessages} agents={mockAgents} users={mockUsers} />);

    expect(screen.getByText("Search & Export")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search messages...")).toBeInTheDocument();
    expect(screen.getByText("Filters")).toBeInTheDocument();
    expect(screen.getByText("Export")).toBeInTheDocument();
  });

  it("displays message count correctly", () => {
    render(<ConversationSearch messages={mockMessages} agents={mockAgents} users={mockUsers} />);

    expect(screen.getByText("Found 3 messages of 3 total")).toBeInTheDocument();
  });

  it("shows and hides filters", async () => {
    const user = userEvent.setup();
    render(<ConversationSearch messages={mockMessages} agents={mockAgents} users={mockUsers} />);

    const filtersButton = screen.getByText("Filters");

    // Filters should be hidden initially
    expect(screen.queryByText("Filter by Users")).not.toBeInTheDocument();

    // Show filters
    await user.click(filtersButton);
    expect(screen.getByText("Filter by Users")).toBeInTheDocument();
    expect(screen.getByText("Filter by Agents")).toBeInTheDocument();

    // Hide filters again
    await user.click(filtersButton);
    expect(screen.queryByText("Filter by Users")).not.toBeInTheDocument();
  });

  it("filters messages by search query", async () => {
    const user = userEvent.setup();
    const onFilterChange = jest.fn();

    render(
      <ConversationSearch
        messages={mockMessages}
        agents={mockAgents}
        users={mockUsers}
        onFilterChange={onFilterChange}
      />,
    );

    const searchInput = screen.getByPlaceholderText("Search messages...");
    await user.type(searchInput, "test");

    await waitFor(() => {
      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({
          searchQuery: "test",
        }),
      );
    });

    // Should show filtered results
    expect(screen.getByText("Found 2 messages of 3 total")).toBeInTheDocument();
  });

  it("filters by user selection", async () => {
    const user = userEvent.setup();
    const onFilterChange = jest.fn();

    render(
      <ConversationSearch
        messages={mockMessages}
        agents={mockAgents}
        users={mockUsers}
        onFilterChange={onFilterChange}
      />,
    );

    // Open filters
    await user.click(screen.getByText("Filters"));

    // Click on a user badge
    const userBadge = screen.getByText("John Doe");
    await user.click(userBadge);

    expect(onFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({
        userIds: ["user-1"],
      }),
    );

    // Should show active filter indicator
    expect(screen.getByText("1 filter active")).toBeInTheDocument();
  });

  it("filters by agent selection", async () => {
    const user = userEvent.setup();
    const onFilterChange = jest.fn();

    render(
      <ConversationSearch
        messages={mockMessages}
        agents={mockAgents}
        users={mockUsers}
        onFilterChange={onFilterChange}
      />,
    );

    // Open filters
    await user.click(screen.getByText("Filters"));

    // Click on an agent badge
    const agentBadge = screen.getByText("Test Agent");
    await user.click(agentBadge);

    expect(onFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({
        agentIds: ["agent-1"],
      }),
    );

    expect(screen.getByText("1 filter active")).toBeInTheDocument();
  });

  it("clears all filters", async () => {
    const user = userEvent.setup();
    const onFilterChange = jest.fn();

    render(
      <ConversationSearch
        messages={mockMessages}
        agents={mockAgents}
        users={mockUsers}
        onFilterChange={onFilterChange}
      />,
    );

    // Open filters and set some filters
    await user.click(screen.getByText("Filters"));
    await user.click(screen.getByText("John Doe"));

    // Clear filters
    const clearButton = screen.getByText("Clear All Filters");
    await user.click(clearButton);

    expect(onFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({
        searchQuery: "",
        userIds: [],
        agentIds: [],
        messageTypes: [],
      }),
    );
  });

  it("handles export functionality", async () => {
    const user = userEvent.setup();
    const onExport = jest.fn();

    render(
      <ConversationSearch
        messages={mockMessages}
        agents={mockAgents}
        users={mockUsers}
        onExport={onExport}
      />,
    );

    const exportButton = screen.getByText("Export");
    await user.click(exportButton);

    expect(onExport).toHaveBeenCalledWith("json", mockMessages);
  });

  it("changes export format", async () => {
    const user = userEvent.setup();
    const onExport = jest.fn();

    render(
      <ConversationSearch
        messages={mockMessages}
        agents={mockAgents}
        users={mockUsers}
        onExport={onExport}
      />,
    );

    // Change format to CSV
    const formatSelect = screen.getByDisplayValue("JSON");
    await user.selectOptions(formatSelect, "csv");

    const exportButton = screen.getByText("Export");
    await user.click(exportButton);

    expect(onExport).toHaveBeenCalledWith("csv", mockMessages);
  });

  it("disables export when no messages", () => {
    render(<ConversationSearch messages={[]} agents={mockAgents} users={mockUsers} />);

    const exportButton = screen.getByText("Export");
    expect(exportButton).toBeDisabled();
  });

  it("shows message preview", () => {
    render(<ConversationSearch messages={mockMessages} agents={mockAgents} users={mockUsers} />);

    expect(screen.getByText("Preview (First 5 messages)")).toBeInTheDocument();
    expect(screen.getByText("Hello world, this is a test message")).toBeInTheDocument();
    expect(screen.getByText("This is an agent response about testing")).toBeInTheDocument();
  });

  it("highlights search terms in preview", async () => {
    const user = userEvent.setup();

    render(<ConversationSearch messages={mockMessages} agents={mockAgents} users={mockUsers} />);

    const searchInput = screen.getByPlaceholderText("Search messages...");
    await user.type(searchInput, "test");

    await waitFor(() => {
      // Check that search term is highlighted in preview
      const highlightedElements = screen.getAllByText("test");
      expect(highlightedElements.some((el) => el.tagName.toLowerCase() === "mark")).toBe(true);
    });
  });
});
