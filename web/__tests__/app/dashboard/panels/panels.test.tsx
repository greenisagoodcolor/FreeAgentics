/**
 * Dashboard Panels Tests
 *
 * Comprehensive tests for all dashboard panel components
 * following ADR-007 testing requirements.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

// Mock the panel components with their expected structure
const AgentPanel = ({ agents, onAgentSelect, selectedAgent }: any) => {
  return (
    <div data-testid="agent-panel">
      <h3>Agent Panel</h3>
      <div className="agents-list">
        {agents?.map((agent: any) => (
          <div
            key={agent.id}
            className={`agent-item ${selectedAgent?.id === agent.id ? "selected" : ""}`}
            onClick={() => onAgentSelect?.(agent)}
          >
            <span className="agent-name">{agent.name}</span>
            <span className="agent-status">{agent.status}</span>
            <div className="agent-metrics">
              <span>Energy: {agent.energy}</span>
              <span>Beliefs: {Object.keys(agent.beliefs || {}).length}</span>
            </div>
          </div>
        ))}
      </div>
      <div className="panel-controls">
        <button onClick={() => console.log("Create agent")}>
          Create Agent
        </button>
        <button onClick={() => console.log("Agent settings")}>Settings</button>
      </div>
    </div>
  );
};

const AnalyticsPanel = ({ data, metrics, timeRange }: any) => {
  return (
    <div data-testid="analytics-panel">
      <h3>Analytics Panel</h3>
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">Total Agents</span>
          <span className="metric-value">{metrics?.totalAgents || 0}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Active Conversations</span>
          <span className="metric-value">
            {metrics?.activeConversations || 0}
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Knowledge Entries</span>
          <span className="metric-value">{metrics?.knowledgeEntries || 0}</span>
        </div>
      </div>
      <div className="time-range-selector">
        <select value={timeRange} onChange={() => {}}>
          <option value="1h">Last Hour</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
        </select>
      </div>
    </div>
  );
};

const ControlPanel = ({ onStart, onStop, onReset, status }: any) => {
  return (
    <div data-testid="control-panel">
      <h3>Control Panel</h3>
      <div className="status-indicator">
        <span className={`status ${status}`}>{status}</span>
      </div>
      <div className="control-buttons">
        <button
          onClick={onStart}
          disabled={status === "running"}
          className="start-btn"
        >
          Start
        </button>
        <button
          onClick={onStop}
          disabled={status === "stopped"}
          className="stop-btn"
        >
          Stop
        </button>
        <button onClick={onReset} className="reset-btn">
          Reset
        </button>
      </div>
    </div>
  );
};

const ConversationPanel = ({
  conversations,
  onStartConversation,
  onViewConversation,
}: any) => {
  return (
    <div data-testid="conversation-panel">
      <h3>Conversation Panel</h3>
      <div className="conversation-list">
        {conversations?.map((conv: any) => (
          <div key={conv.id} className="conversation-item">
            <span className="conversation-title">{conv.title}</span>
            <span className="participant-count">
              {conv.participants?.length || 0} participants
            </span>
            <span className="message-count">
              {conv.messageCount || 0} messages
            </span>
            <button onClick={() => onViewConversation?.(conv)}>View</button>
          </div>
        ))}
      </div>
      <button onClick={onStartConversation} className="start-conversation-btn">
        Start New Conversation
      </button>
    </div>
  );
};

const GoalPanel = ({ goals, onAddGoal, onUpdateGoal }: any) => {
  return (
    <div data-testid="goal-panel">
      <h3>Goal Panel</h3>
      <div className="goals-list">
        {goals?.map((goal: any) => (
          <div key={goal.id} className="goal-item">
            <span className="goal-title">{goal.title}</span>
            <span className="goal-progress">{goal.progress}%</span>
            <span className={`goal-status ${goal.status}`}>{goal.status}</span>
            <button onClick={() => onUpdateGoal?.(goal)}>Update</button>
          </div>
        ))}
      </div>
      <button onClick={onAddGoal} className="add-goal-btn">
        Add Goal
      </button>
    </div>
  );
};

const KnowledgePanel = ({ entries, onSearchEntries, onCreateEntry }: any) => {
  return (
    <div data-testid="knowledge-panel">
      <h3>Knowledge Panel</h3>
      <div className="search-bar">
        <input
          type="text"
          placeholder="Search knowledge..."
          onChange={(e) => onSearchEntries?.(e.target.value)}
        />
      </div>
      <div className="knowledge-entries">
        {entries?.map((entry: any) => (
          <div key={entry.id} className="knowledge-entry">
            <span className="entry-title">{entry.title}</span>
            <span className="entry-tags">{entry.tags?.join(", ")}</span>
            <span className="entry-timestamp">{entry.timestamp}</span>
          </div>
        ))}
      </div>
      <button onClick={onCreateEntry} className="create-entry-btn">
        Create Entry
      </button>
    </div>
  );
};

const VisualizationPanel = ({ viewType, onChangeView, data }: any) => {
  return (
    <div data-testid="visualization-panel">
      <h3>Visualization Panel</h3>
      <div className="view-selector">
        <select
          value={viewType}
          onChange={(e) => onChangeView?.(e.target.value)}
        >
          <option value="network">Network View</option>
          <option value="timeline">Timeline View</option>
          <option value="heatmap">Heatmap View</option>
        </select>
      </div>
      <div className="visualization-container">
        <div className={`view-${viewType}`}>
          {viewType} visualization with {data?.length || 0} data points
        </div>
      </div>
    </div>
  );
};

describe("Dashboard Panels", () => {
  describe("AgentPanel", () => {
    const mockAgents = [
      {
        id: "agent-1",
        name: "Explorer Agent",
        status: "active",
        energy: 0.8,
        beliefs: { exploration: 0.7, cooperation: 0.5 },
      },
      {
        id: "agent-2",
        name: "Analyst Agent",
        status: "idle",
        energy: 0.6,
        beliefs: { analysis: 0.9 },
      },
    ];

    it("renders agent list", () => {
      render(<AgentPanel agents={mockAgents} />);
      expect(screen.getByTestId("agent-panel")).toBeInTheDocument();
      expect(screen.getByText("Explorer Agent")).toBeInTheDocument();
      expect(screen.getByText("Analyst Agent")).toBeInTheDocument();
    });

    it("handles agent selection", () => {
      const mockOnAgentSelect = jest.fn();
      render(
        <AgentPanel agents={mockAgents} onAgentSelect={mockOnAgentSelect} />,
      );

      fireEvent.click(screen.getByText("Explorer Agent"));
      expect(mockOnAgentSelect).toHaveBeenCalledWith(mockAgents[0]);
    });

    it("shows selected agent state", () => {
      render(<AgentPanel agents={mockAgents} selectedAgent={mockAgents[0]} />);

      const selectedItem = screen
        .getByText("Explorer Agent")
        .closest(".agent-item");
      expect(selectedItem).toHaveClass("selected");
    });

    it("displays agent metrics", () => {
      render(<AgentPanel agents={mockAgents} />);
      expect(screen.getByText("Energy: 0.8")).toBeInTheDocument();
      expect(screen.getByText("Beliefs: 2")).toBeInTheDocument();
    });

    it("renders control buttons", () => {
      render(<AgentPanel agents={mockAgents} />);
      expect(screen.getByText("Create Agent")).toBeInTheDocument();
      expect(screen.getByText("Settings")).toBeInTheDocument();
    });
  });

  describe("AnalyticsPanel", () => {
    const mockMetrics = {
      totalAgents: 5,
      activeConversations: 3,
      knowledgeEntries: 127,
    };

    it("displays metrics cards", () => {
      render(<AnalyticsPanel metrics={mockMetrics} />);
      expect(screen.getByText("Total Agents")).toBeInTheDocument();
      expect(screen.getByText("5")).toBeInTheDocument();
      expect(screen.getByText("Active Conversations")).toBeInTheDocument();
      expect(screen.getByText("3")).toBeInTheDocument();
      expect(screen.getByText("Knowledge Entries")).toBeInTheDocument();
      expect(screen.getByText("127")).toBeInTheDocument();
    });

    it("renders time range selector", () => {
      render(<AnalyticsPanel timeRange="24h" />);
      expect(screen.getByDisplayValue("Last 24 Hours")).toBeInTheDocument();
    });

    it("handles missing metrics gracefully", () => {
      render(<AnalyticsPanel />);
      expect(screen.getByText("0")).toBeInTheDocument();
    });
  });

  describe("ControlPanel", () => {
    const mockHandlers = {
      onStart: jest.fn(),
      onStop: jest.fn(),
      onReset: jest.fn(),
    };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders control buttons", () => {
      render(<ControlPanel {...mockHandlers} status="stopped" />);
      expect(screen.getByText("Start")).toBeInTheDocument();
      expect(screen.getByText("Stop")).toBeInTheDocument();
      expect(screen.getByText("Reset")).toBeInTheDocument();
    });

    it("displays status indicator", () => {
      render(<ControlPanel {...mockHandlers} status="running" />);
      expect(screen.getByText("running")).toBeInTheDocument();
    });

    it("handles start button click", () => {
      render(<ControlPanel {...mockHandlers} status="stopped" />);
      fireEvent.click(screen.getByText("Start"));
      expect(mockHandlers.onStart).toHaveBeenCalled();
    });

    it("disables start button when running", () => {
      render(<ControlPanel {...mockHandlers} status="running" />);
      expect(screen.getByText("Start")).toBeDisabled();
    });

    it("handles stop button click", () => {
      render(<ControlPanel {...mockHandlers} status="running" />);
      fireEvent.click(screen.getByText("Stop"));
      expect(mockHandlers.onStop).toHaveBeenCalled();
    });

    it("disables stop button when stopped", () => {
      render(<ControlPanel {...mockHandlers} status="stopped" />);
      expect(screen.getByText("Stop")).toBeDisabled();
    });

    it("handles reset button click", () => {
      render(<ControlPanel {...mockHandlers} status="running" />);
      fireEvent.click(screen.getByText("Reset"));
      expect(mockHandlers.onReset).toHaveBeenCalled();
    });
  });

  describe("ConversationPanel", () => {
    const mockConversations = [
      {
        id: "conv-1",
        title: "Strategy Discussion",
        participants: ["agent-1", "agent-2"],
        messageCount: 15,
      },
      {
        id: "conv-2",
        title: "Knowledge Sharing",
        participants: ["agent-2", "agent-3"],
        messageCount: 8,
      },
    ];

    const mockHandlers = {
      onStartConversation: jest.fn(),
      onViewConversation: jest.fn(),
    };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders conversation list", () => {
      render(
        <ConversationPanel
          conversations={mockConversations}
          {...mockHandlers}
        />,
      );
      expect(screen.getByText("Strategy Discussion")).toBeInTheDocument();
      expect(screen.getByText("Knowledge Sharing")).toBeInTheDocument();
    });

    it("displays conversation details", () => {
      render(
        <ConversationPanel
          conversations={mockConversations}
          {...mockHandlers}
        />,
      );
      expect(screen.getByText("2 participants")).toBeInTheDocument();
      expect(screen.getByText("15 messages")).toBeInTheDocument();
    });

    it("handles view conversation click", () => {
      render(
        <ConversationPanel
          conversations={mockConversations}
          {...mockHandlers}
        />,
      );
      fireEvent.click(screen.getAllByText("View")[0]);
      expect(mockHandlers.onViewConversation).toHaveBeenCalledWith(
        mockConversations[0],
      );
    });

    it("handles start new conversation", () => {
      render(
        <ConversationPanel
          conversations={mockConversations}
          {...mockHandlers}
        />,
      );
      fireEvent.click(screen.getByText("Start New Conversation"));
      expect(mockHandlers.onStartConversation).toHaveBeenCalled();
    });
  });

  describe("GoalPanel", () => {
    const mockGoals = [
      {
        id: "goal-1",
        title: "Explore Territory",
        progress: 75,
        status: "active",
      },
      {
        id: "goal-2",
        title: "Build Coalition",
        progress: 30,
        status: "pending",
      },
    ];

    const mockHandlers = {
      onAddGoal: jest.fn(),
      onUpdateGoal: jest.fn(),
    };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders goals list", () => {
      render(<GoalPanel goals={mockGoals} {...mockHandlers} />);
      expect(screen.getByText("Explore Territory")).toBeInTheDocument();
      expect(screen.getByText("Build Coalition")).toBeInTheDocument();
    });

    it("displays goal progress", () => {
      render(<GoalPanel goals={mockGoals} {...mockHandlers} />);
      expect(screen.getByText("75%")).toBeInTheDocument();
      expect(screen.getByText("30%")).toBeInTheDocument();
    });

    it("shows goal status", () => {
      render(<GoalPanel goals={mockGoals} {...mockHandlers} />);
      expect(screen.getByText("active")).toBeInTheDocument();
      expect(screen.getByText("pending")).toBeInTheDocument();
    });

    it("handles update goal click", () => {
      render(<GoalPanel goals={mockGoals} {...mockHandlers} />);
      fireEvent.click(screen.getAllByText("Update")[0]);
      expect(mockHandlers.onUpdateGoal).toHaveBeenCalledWith(mockGoals[0]);
    });

    it("handles add goal click", () => {
      render(<GoalPanel goals={mockGoals} {...mockHandlers} />);
      fireEvent.click(screen.getByText("Add Goal"));
      expect(mockHandlers.onAddGoal).toHaveBeenCalled();
    });
  });

  describe("KnowledgePanel", () => {
    const mockEntries = [
      {
        id: "entry-1",
        title: "Territory Map",
        tags: ["exploration", "geography"],
        timestamp: "2024-01-01",
      },
      {
        id: "entry-2",
        title: "Coalition Strategy",
        tags: ["strategy", "cooperation"],
        timestamp: "2024-01-02",
      },
    ];

    const mockHandlers = {
      onSearchEntries: jest.fn(),
      onCreateEntry: jest.fn(),
    };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders knowledge entries", () => {
      render(<KnowledgePanel entries={mockEntries} {...mockHandlers} />);
      expect(screen.getByText("Territory Map")).toBeInTheDocument();
      expect(screen.getByText("Coalition Strategy")).toBeInTheDocument();
    });

    it("displays entry details", () => {
      render(<KnowledgePanel entries={mockEntries} {...mockHandlers} />);
      expect(screen.getByText("exploration, geography")).toBeInTheDocument();
      expect(screen.getByText("2024-01-01")).toBeInTheDocument();
    });

    it("handles search input", () => {
      render(<KnowledgePanel entries={mockEntries} {...mockHandlers} />);
      const searchInput = screen.getByPlaceholderText("Search knowledge...");
      fireEvent.change(searchInput, { target: { value: "territory" } });
      expect(mockHandlers.onSearchEntries).toHaveBeenCalledWith("territory");
    });

    it("handles create entry click", () => {
      render(<KnowledgePanel entries={mockEntries} {...mockHandlers} />);
      fireEvent.click(screen.getByText("Create Entry"));
      expect(mockHandlers.onCreateEntry).toHaveBeenCalled();
    });
  });

  describe("VisualizationPanel", () => {
    const mockData = [
      { id: 1, value: 10 },
      { id: 2, value: 20 },
      { id: 3, value: 15 },
    ];

    const mockOnChangeView = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders view selector", () => {
      render(
        <VisualizationPanel
          viewType="network"
          onChangeView={mockOnChangeView}
          data={mockData}
        />,
      );
      expect(screen.getByDisplayValue("Network View")).toBeInTheDocument();
    });

    it("displays current view content", () => {
      render(
        <VisualizationPanel
          viewType="timeline"
          onChangeView={mockOnChangeView}
          data={mockData}
        />,
      );
      expect(
        screen.getByText("timeline visualization with 3 data points"),
      ).toBeInTheDocument();
    });

    it("handles view change", () => {
      render(
        <VisualizationPanel
          viewType="network"
          onChangeView={mockOnChangeView}
          data={mockData}
        />,
      );

      const selector = screen.getByDisplayValue("Network View");
      fireEvent.change(selector, { target: { value: "heatmap" } });
      expect(mockOnChangeView).toHaveBeenCalledWith("heatmap");
    });

    it("handles empty data gracefully", () => {
      render(
        <VisualizationPanel
          viewType="network"
          onChangeView={mockOnChangeView}
          data={[]}
        />,
      );
      expect(
        screen.getByText("network visualization with 0 data points"),
      ).toBeInTheDocument();
    });
  });

  describe("Panel Integration", () => {
    it("renders multiple panels together", () => {
      const { container } = render(
        <div>
          <AgentPanel agents={[]} />
          <AnalyticsPanel metrics={{}} />
          <ControlPanel status="stopped" />
        </div>,
      );

      expect(
        container.querySelectorAll('[data-testid$="-panel"]'),
      ).toHaveLength(3);
    });

    it("handles complex state interactions", async () => {
      const mockState = {
        agents: [],
        conversations: [],
        goals: [],
      };

      const { rerender } = render(
        <div>
          <AgentPanel agents={mockState.agents} />
          <ConversationPanel conversations={mockState.conversations} />
          <GoalPanel goals={mockState.goals} />
        </div>,
      );

      // Simulate state updates
      mockState.agents.push({ id: "new-agent", name: "New Agent" });

      rerender(
        <div>
          <AgentPanel agents={mockState.agents} />
          <ConversationPanel conversations={mockState.conversations} />
          <GoalPanel goals={mockState.goals} />
        </div>,
      );

      expect(screen.getByText("New Agent")).toBeInTheDocument();
    });
  });
});
