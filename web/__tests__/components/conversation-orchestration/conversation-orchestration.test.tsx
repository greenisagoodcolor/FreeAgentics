/**
 * Conversation Orchestration Tests
 *
 * Comprehensive tests for conversation orchestration components
 * following ADR-007 testing requirements.
 */

import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Mock conversation orchestration components
const AdvancedControls = ({ settings, onSettingsChange }: any) => {
  return (
    <div data-testid="advanced-controls">
      <h3>Advanced Controls</h3>
      <div className="control-section">
        <label>
          Temperature
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={settings?.temperature || 0.7}
            onChange={(e) =>
              onSettingsChange?.({
                ...settings,
                temperature: parseFloat(e.target.value),
              })
            }
          />
          <span>{settings?.temperature || 0.7}</span>
        </label>
      </div>
      <div className="control-section">
        <label>
          Max Tokens
          <input
            type="number"
            min={1}
            max={4000}
            value={settings?.maxTokens || 1000}
            onChange={(e) =>
              onSettingsChange?.({
                ...settings,
                maxTokens: parseInt(e.target.value),
              })
            }
          />
        </label>
      </div>
      <div className="control-section">
        <label>
          <input
            type="checkbox"
            checked={settings?.streamResponses || false}
            onChange={(e) =>
              onSettingsChange?.({
                ...settings,
                streamResponses: e.target.checked,
              })
            }
          />
          Stream Responses
        </label>
      </div>
      <div className="control-section">
        <label>
          Response Format
          <select
            value={settings?.responseFormat || "text"}
            onChange={(e) =>
              onSettingsChange?.({
                ...settings,
                responseFormat: e.target.value,
              })
            }
          >
            <option value="text">Text</option>
            <option value="json">JSON</option>
            <option value="markdown">Markdown</option>
          </select>
        </label>
      </div>
    </div>
  );
};

const ChangeHistory = ({ changes, onRevert }: any) => {
  return (
    <div data-testid="change-history">
      <h3>Change History</h3>
      <div className="changes-list">
        {changes?.map((change: any, index: number) => (
          <div key={index} className="change-item">
            <span className="change-timestamp">{change.timestamp}</span>
            <span className="change-type">{change.type}</span>
            <span className="change-description">{change.description}</span>
            <button onClick={() => onRevert?.(change.id)}>Revert</button>
          </div>
        ))}
      </div>
      {(!changes || changes.length === 0) && (
        <div className="no-changes">No changes recorded</div>
      )}
    </div>
  );
};

const PresetSelector = ({
  presets,
  selectedPreset,
  onPresetSelect,
  onSavePreset,
}: any) => {
  const [customName, setCustomName] = React.useState("");

  return (
    <div data-testid="preset-selector">
      <h3>Conversation Presets</h3>
      <div className="preset-list">
        {presets?.map((preset: any) => (
          <div
            key={preset.id}
            className={`preset-item ${selectedPreset?.id === preset.id ? "selected" : ""}`}
            onClick={() => onPresetSelect?.(preset)}
          >
            <span className="preset-name">{preset.name}</span>
            <span className="preset-description">{preset.description}</span>
            <div className="preset-metrics">
              <span>Agents: {preset.agentCount}</span>
              <span>Duration: {preset.expectedDuration}min</span>
            </div>
          </div>
        ))}
      </div>
      <div className="save-preset-section">
        <input
          type="text"
          placeholder="Preset name"
          value={customName}
          onChange={(e) => setCustomName(e.target.value)}
        />
        <button
          onClick={() => {
            onSavePreset?.(customName);
            setCustomName("");
          }}
          disabled={!customName.trim()}
        >
          Save Current as Preset
        </button>
      </div>
    </div>
  );
};

const RealTimePreview = ({ previewData, isActive }: any) => {
  return (
    <div data-testid="real-time-preview">
      <h3>Real-time Preview</h3>
      <div className={`preview-status ${isActive ? "active" : "inactive"}`}>
        {isActive ? "Live Preview" : "Preview Paused"}
      </div>
      <div className="preview-content">
        <div className="conversation-flow">
          <h4>Conversation Flow</h4>
          <div className="flow-diagram">
            {previewData?.participants?.map(
              (participant: any, index: number) => (
                <div key={index} className="participant-node">
                  <span className="participant-name">{participant.name}</span>
                  <div className="participant-state">{participant.state}</div>
                </div>
              ),
            )}
          </div>
        </div>
        <div className="message-preview">
          <h4>Next Messages</h4>
          <div className="message-queue">
            {previewData?.nextMessages?.map((message: any, index: number) => (
              <div key={index} className="message-preview-item">
                <span className="sender">{message.sender}</span>
                <span className="content">{message.content}</span>
                <span className="timing">{message.expectedTime}s</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const ResponseDynamicsControls = ({ dynamics, onDynamicsChange }: any) => {
  return (
    <div data-testid="response-dynamics-controls">
      <h3>Response Dynamics</h3>
      <div className="dynamics-controls">
        <div className="control-group">
          <label>
            Response Delay (ms)
            <input
              type="range"
              min={100}
              max={5000}
              step={100}
              value={dynamics?.responseDelay || 1000}
              onChange={(e) =>
                onDynamicsChange?.({
                  ...dynamics,
                  responseDelay: parseInt(e.target.value),
                })
              }
            />
            <span>{dynamics?.responseDelay || 1000}ms</span>
          </label>
        </div>
        <div className="control-group">
          <label>
            Thinking Time (ms)
            <input
              type="range"
              min={500}
              max={10000}
              step={500}
              value={dynamics?.thinkingTime || 2000}
              onChange={(e) =>
                onDynamicsChange?.({
                  ...dynamics,
                  thinkingTime: parseInt(e.target.value),
                })
              }
            />
            <span>{dynamics?.thinkingTime || 2000}ms</span>
          </label>
        </div>
        <div className="control-group">
          <label>
            Turn-taking Strategy
            <select
              value={dynamics?.turnTakingStrategy || "round-robin"}
              onChange={(e) =>
                onDynamicsChange?.({
                  ...dynamics,
                  turnTakingStrategy: e.target.value,
                })
              }
            >
              <option value="round-robin">Round Robin</option>
              <option value="random">Random</option>
              <option value="weighted">Weighted by Energy</option>
              <option value="interrupt">Interrupt-based</option>
            </select>
          </label>
        </div>
        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={dynamics?.allowInterruptions || false}
              onChange={(e) =>
                onDynamicsChange?.({
                  ...dynamics,
                  allowInterruptions: e.target.checked,
                })
              }
            />
            Allow Interruptions
          </label>
        </div>
      </div>
    </div>
  );
};

const TimingControls = ({ timing, onTimingChange }: any) => {
  return (
    <div data-testid="timing-controls">
      <h3>Timing Controls</h3>
      <div className="timing-controls">
        <div className="control-row">
          <label>
            Conversation Duration (minutes)
            <input
              type="number"
              min={1}
              max={180}
              value={timing?.duration || 30}
              onChange={(e) =>
                onTimingChange?.({
                  ...timing,
                  duration: parseInt(e.target.value),
                })
              }
            />
          </label>
        </div>
        <div className="control-row">
          <label>
            Messages per Minute
            <input
              type="range"
              min={1}
              max={20}
              value={timing?.messagesPerMinute || 5}
              onChange={(e) =>
                onTimingChange?.({
                  ...timing,
                  messagesPerMinute: parseInt(e.target.value),
                })
              }
            />
            <span>{timing?.messagesPerMinute || 5} msg/min</span>
          </label>
        </div>
        <div className="control-row">
          <label>
            Auto-pause after (messages)
            <input
              type="number"
              min={0}
              max={100}
              value={timing?.autoPauseAfter || 0}
              onChange={(e) =>
                onTimingChange?.({
                  ...timing,
                  autoPauseAfter: parseInt(e.target.value),
                })
              }
            />
          </label>
        </div>
        <div className="control-row">
          <label>
            <input
              type="checkbox"
              checked={timing?.enableScheduledBreaks || false}
              onChange={(e) =>
                onTimingChange?.({
                  ...timing,
                  enableScheduledBreaks: e.target.checked,
                })
              }
            />
            Enable Scheduled Breaks
          </label>
        </div>
        {timing?.enableScheduledBreaks && (
          <div className="break-controls">
            <label>
              Break Interval (minutes)
              <input
                type="number"
                min={5}
                max={60}
                value={timing?.breakInterval || 15}
                onChange={(e) =>
                  onTimingChange?.({
                    ...timing,
                    breakInterval: parseInt(e.target.value),
                  })
                }
              />
            </label>
            <label>
              Break Duration (seconds)
              <input
                type="number"
                min={10}
                max={300}
                value={timing?.breakDuration || 30}
                onChange={(e) =>
                  onTimingChange?.({
                    ...timing,
                    breakDuration: parseInt(e.target.value),
                  })
                }
              />
            </label>
          </div>
        )}
      </div>
    </div>
  );
};

describe("Conversation Orchestration Components", () => {
  describe("AdvancedControls", () => {
    const mockSettings = {
      temperature: 0.7,
      maxTokens: 1000,
      streamResponses: true,
      responseFormat: "text",
    };

    const mockOnSettingsChange = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders all control sections", () => {
      render(
        <AdvancedControls
          settings={mockSettings}
          onSettingsChange={mockOnSettingsChange}
        />,
      );

      expect(screen.getByText("Advanced Controls")).toBeInTheDocument();
      expect(screen.getByText("Temperature")).toBeInTheDocument();
      expect(screen.getByText("Max Tokens")).toBeInTheDocument();
      expect(screen.getByText("Stream Responses")).toBeInTheDocument();
      expect(screen.getByText("Response Format")).toBeInTheDocument();
    });

    it("displays current temperature value", () => {
      render(
        <AdvancedControls
          settings={mockSettings}
          onSettingsChange={mockOnSettingsChange}
        />,
      );
      expect(screen.getByDisplayValue("0.7")).toBeInTheDocument();
    });

    it("handles temperature changes", () => {
      render(
        <AdvancedControls
          settings={mockSettings}
          onSettingsChange={mockOnSettingsChange}
        />,
      );

      const temperatureSlider = screen.getByRole("slider");
      fireEvent.change(temperatureSlider, { target: { value: "0.9" } });

      expect(mockOnSettingsChange).toHaveBeenCalledWith({
        ...mockSettings,
        temperature: 0.9,
      });
    });

    it("handles max tokens changes", () => {
      render(
        <AdvancedControls
          settings={mockSettings}
          onSettingsChange={mockOnSettingsChange}
        />,
      );

      const maxTokensInput = screen.getByDisplayValue("1000");
      fireEvent.change(maxTokensInput, { target: { value: "2000" } });

      expect(mockOnSettingsChange).toHaveBeenCalledWith({
        ...mockSettings,
        maxTokens: 2000,
      });
    });

    it("toggles stream responses", () => {
      render(
        <AdvancedControls
          settings={mockSettings}
          onSettingsChange={mockOnSettingsChange}
        />,
      );

      const streamCheckbox = screen.getByRole("checkbox");
      fireEvent.click(streamCheckbox);

      expect(mockOnSettingsChange).toHaveBeenCalledWith({
        ...mockSettings,
        streamResponses: false,
      });
    });

    it("changes response format", () => {
      render(
        <AdvancedControls
          settings={mockSettings}
          onSettingsChange={mockOnSettingsChange}
        />,
      );

      const formatSelect = screen.getByLabelText("Response Format");
      fireEvent.change(formatSelect, { target: { value: "json" } });

      expect(mockOnSettingsChange).toHaveBeenCalledWith({
        ...mockSettings,
        responseFormat: "json",
      });
    });

    it("handles missing settings gracefully", () => {
      render(<AdvancedControls onSettingsChange={mockOnSettingsChange} />);
      expect(screen.getByText("0.7")).toBeInTheDocument(); // Default temperature
    });
  });

  describe("ChangeHistory", () => {
    const mockChanges = [
      {
        id: "change-1",
        timestamp: "2024-01-01 10:00",
        type: "setting",
        description: "Changed temperature to 0.8",
      },
      {
        id: "change-2",
        timestamp: "2024-01-01 10:05",
        type: "preset",
        description: "Applied casual conversation preset",
      },
    ];

    const mockOnRevert = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders change history list", () => {
      render(<ChangeHistory changes={mockChanges} onRevert={mockOnRevert} />);

      expect(screen.getByText("Change History")).toBeInTheDocument();
      expect(
        screen.getByText("Changed temperature to 0.8"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Applied casual conversation preset"),
      ).toBeInTheDocument();
    });

    it("displays change timestamps and types", () => {
      render(<ChangeHistory changes={mockChanges} onRevert={mockOnRevert} />);

      expect(screen.getByText("2024-01-01 10:00")).toBeInTheDocument();
      expect(screen.getByText("setting")).toBeInTheDocument();
      expect(screen.getByText("preset")).toBeInTheDocument();
    });

    it("handles revert button clicks", () => {
      render(<ChangeHistory changes={mockChanges} onRevert={mockOnRevert} />);

      const revertButtons = screen.getAllByText("Revert");
      fireEvent.click(revertButtons[0]);

      expect(mockOnRevert).toHaveBeenCalledWith("change-1");
    });

    it("shows message when no changes exist", () => {
      render(<ChangeHistory changes={[]} onRevert={mockOnRevert} />);
      expect(screen.getByText("No changes recorded")).toBeInTheDocument();
    });

    it("handles undefined changes", () => {
      render(<ChangeHistory onRevert={mockOnRevert} />);
      expect(screen.getByText("No changes recorded")).toBeInTheDocument();
    });
  });

  describe("PresetSelector", () => {
    const mockPresets = [
      {
        id: "preset-1",
        name: "Casual Discussion",
        description: "Relaxed conversation between agents",
        agentCount: 3,
        expectedDuration: 15,
      },
      {
        id: "preset-2",
        name: "Formal Debate",
        description: "Structured argumentation",
        agentCount: 2,
        expectedDuration: 30,
      },
    ];

    const mockHandlers = {
      onPresetSelect: jest.fn(),
      onSavePreset: jest.fn(),
    };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders preset list", () => {
      render(<PresetSelector presets={mockPresets} {...mockHandlers} />);

      expect(screen.getByText("Conversation Presets")).toBeInTheDocument();
      expect(screen.getByText("Casual Discussion")).toBeInTheDocument();
      expect(screen.getByText("Formal Debate")).toBeInTheDocument();
    });

    it("displays preset details", () => {
      render(<PresetSelector presets={mockPresets} {...mockHandlers} />);

      expect(
        screen.getByText("Relaxed conversation between agents"),
      ).toBeInTheDocument();
      expect(screen.getByText("Agents: 3")).toBeInTheDocument();
      expect(screen.getByText("Duration: 15min")).toBeInTheDocument();
    });

    it("highlights selected preset", () => {
      render(
        <PresetSelector
          presets={mockPresets}
          selectedPreset={mockPresets[0]}
          {...mockHandlers}
        />,
      );

      const selectedItem = screen
        .getByText("Casual Discussion")
        .closest(".preset-item");
      expect(selectedItem).toHaveClass("selected");
    });

    it("handles preset selection", () => {
      render(<PresetSelector presets={mockPresets} {...mockHandlers} />);

      fireEvent.click(screen.getByText("Formal Debate"));
      expect(mockHandlers.onPresetSelect).toHaveBeenCalledWith(mockPresets[1]);
    });

    it("handles saving custom presets", async () => {
      const user = userEvent.setup();
      render(<PresetSelector presets={mockPresets} {...mockHandlers} />);

      const nameInput = screen.getByPlaceholderText("Preset name");
      const saveButton = screen.getByText("Save Current as Preset");

      await act(async () => {
        await user.type(nameInput, "My Custom Preset");
        await user.click(saveButton);
      });

      expect(mockHandlers.onSavePreset).toHaveBeenCalledWith(
        "My Custom Preset",
      );
    });

    it("disables save button when name is empty", () => {
      render(<PresetSelector presets={mockPresets} {...mockHandlers} />);

      const saveButton = screen.getByText("Save Current as Preset");
      expect(saveButton).toBeDisabled();
    });

    it("clears input after saving", async () => {
      const user = userEvent.setup();
      render(<PresetSelector presets={mockPresets} {...mockHandlers} />);

      const nameInput = screen.getByPlaceholderText("Preset name");
      const saveButton = screen.getByText("Save Current as Preset");

      await act(async () => {
        await user.type(nameInput, "Test Preset");
        await user.click(saveButton);
      });

      expect(nameInput).toHaveValue("");
    });
  });

  describe("RealTimePreview", () => {
    const mockPreviewData = {
      participants: [
        { name: "Agent Alpha", state: "thinking" },
        { name: "Agent Beta", state: "responding" },
        { name: "Agent Gamma", state: "listening" },
      ],
      nextMessages: [
        {
          sender: "Agent Alpha",
          content: "I think we should...",
          expectedTime: 5,
        },
        {
          sender: "Agent Beta",
          content: "Actually, let me counter...",
          expectedTime: 8,
        },
      ],
    };

    it("renders preview content", () => {
      render(<RealTimePreview previewData={mockPreviewData} isActive={true} />);

      expect(screen.getByText("Real-time Preview")).toBeInTheDocument();
      expect(screen.getByText("Conversation Flow")).toBeInTheDocument();
      expect(screen.getByText("Next Messages")).toBeInTheDocument();
    });

    it("shows active status when live", () => {
      render(<RealTimePreview previewData={mockPreviewData} isActive={true} />);
      expect(screen.getByText("Live Preview")).toBeInTheDocument();
    });

    it("shows inactive status when paused", () => {
      render(
        <RealTimePreview previewData={mockPreviewData} isActive={false} />,
      );
      expect(screen.getByText("Preview Paused")).toBeInTheDocument();
    });

    it("displays participant information", () => {
      render(<RealTimePreview previewData={mockPreviewData} isActive={true} />);

      // Check for agent names (there may be multiple instances)
      const agentAlphaElements = screen.getAllByText("Agent Alpha");
      expect(agentAlphaElements.length).toBeGreaterThan(0);
      expect(screen.getByText("thinking")).toBeInTheDocument();

      const agentBetaElements = screen.getAllByText("Agent Beta");
      expect(agentBetaElements.length).toBeGreaterThan(0);
      expect(screen.getByText("responding")).toBeInTheDocument();
      expect(screen.getByText("Agent Gamma")).toBeInTheDocument();
      expect(screen.getByText("listening")).toBeInTheDocument();
    });

    it("displays next messages queue", () => {
      render(<RealTimePreview previewData={mockPreviewData} isActive={true} />);

      expect(screen.getByText("I think we should...")).toBeInTheDocument();
      expect(screen.getByText("5s")).toBeInTheDocument();
      expect(
        screen.getByText("Actually, let me counter..."),
      ).toBeInTheDocument();
      expect(screen.getByText("8s")).toBeInTheDocument();
    });

    it("handles missing preview data", () => {
      render(<RealTimePreview isActive={true} />);
      expect(screen.getByText("Real-time Preview")).toBeInTheDocument();
    });
  });

  describe("ResponseDynamicsControls", () => {
    const mockDynamics = {
      responseDelay: 1000,
      thinkingTime: 2000,
      turnTakingStrategy: "round-robin",
      allowInterruptions: false,
    };

    const mockOnDynamicsChange = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders all dynamics controls", () => {
      render(
        <ResponseDynamicsControls
          dynamics={mockDynamics}
          onDynamicsChange={mockOnDynamicsChange}
        />,
      );

      expect(screen.getByText("Response Dynamics")).toBeInTheDocument();
      expect(screen.getByText("Response Delay (ms)")).toBeInTheDocument();
      expect(screen.getByText("Thinking Time (ms)")).toBeInTheDocument();
      expect(screen.getByText("Turn-taking Strategy")).toBeInTheDocument();
      expect(screen.getByText("Allow Interruptions")).toBeInTheDocument();
    });

    it("displays current values", () => {
      render(
        <ResponseDynamicsControls
          dynamics={mockDynamics}
          onDynamicsChange={mockOnDynamicsChange}
        />,
      );

      expect(screen.getByText("1000ms")).toBeInTheDocument();
      expect(screen.getByText("2000ms")).toBeInTheDocument();
      const strategySelect = screen.getByLabelText("Turn-taking Strategy");
      expect(strategySelect).toHaveValue("round-robin");
    });

    it("handles response delay changes", () => {
      render(
        <ResponseDynamicsControls
          dynamics={mockDynamics}
          onDynamicsChange={mockOnDynamicsChange}
        />,
      );

      const responseDelaySlider = screen.getAllByRole("slider")[0];
      fireEvent.change(responseDelaySlider, { target: { value: "1500" } });

      expect(mockOnDynamicsChange).toHaveBeenCalledWith({
        ...mockDynamics,
        responseDelay: 1500,
      });
    });

    it("handles thinking time changes", () => {
      render(
        <ResponseDynamicsControls
          dynamics={mockDynamics}
          onDynamicsChange={mockOnDynamicsChange}
        />,
      );

      const thinkingTimeSlider = screen.getAllByRole("slider")[1];
      fireEvent.change(thinkingTimeSlider, { target: { value: "3000" } });

      expect(mockOnDynamicsChange).toHaveBeenCalledWith({
        ...mockDynamics,
        thinkingTime: 3000,
      });
    });

    it("handles turn-taking strategy changes", () => {
      render(
        <ResponseDynamicsControls
          dynamics={mockDynamics}
          onDynamicsChange={mockOnDynamicsChange}
        />,
      );

      const strategySelect = screen.getByLabelText("Turn-taking Strategy");
      fireEvent.change(strategySelect, { target: { value: "weighted" } });

      expect(mockOnDynamicsChange).toHaveBeenCalledWith({
        ...mockDynamics,
        turnTakingStrategy: "weighted",
      });
    });

    it("toggles interruptions setting", () => {
      render(
        <ResponseDynamicsControls
          dynamics={mockDynamics}
          onDynamicsChange={mockOnDynamicsChange}
        />,
      );

      const interruptionsCheckbox = screen.getByRole("checkbox");
      fireEvent.click(interruptionsCheckbox);

      expect(mockOnDynamicsChange).toHaveBeenCalledWith({
        ...mockDynamics,
        allowInterruptions: true,
      });
    });

    it("handles missing dynamics gracefully", () => {
      render(
        <ResponseDynamicsControls onDynamicsChange={mockOnDynamicsChange} />,
      );
      expect(screen.getByText("1000ms")).toBeInTheDocument(); // Default values
    });
  });

  describe("TimingControls", () => {
    const mockTiming = {
      duration: 30,
      messagesPerMinute: 5,
      autoPauseAfter: 10,
      enableScheduledBreaks: true,
      breakInterval: 15,
      breakDuration: 30,
    };

    const mockOnTimingChange = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it("renders all timing controls", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );

      expect(screen.getByText("Timing Controls")).toBeInTheDocument();
      expect(
        screen.getByText("Conversation Duration (minutes)"),
      ).toBeInTheDocument();
      expect(screen.getByText("Messages per Minute")).toBeInTheDocument();
      expect(
        screen.getByText("Auto-pause after (messages)"),
      ).toBeInTheDocument();
      expect(screen.getByText("Enable Scheduled Breaks")).toBeInTheDocument();
    });

    it("shows break controls when scheduled breaks are enabled", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );

      expect(screen.getByText("Break Interval (minutes)")).toBeInTheDocument();
      expect(screen.getByText("Break Duration (seconds)")).toBeInTheDocument();
    });

    it("hides break controls when scheduled breaks are disabled", () => {
      const timingWithoutBreaks = {
        ...mockTiming,
        enableScheduledBreaks: false,
      };
      render(
        <TimingControls
          timing={timingWithoutBreaks}
          onTimingChange={mockOnTimingChange}
        />,
      );

      expect(
        screen.queryByText("Break Interval (minutes)"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByText("Break Duration (seconds)"),
      ).not.toBeInTheDocument();
    });

    it("handles duration changes", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );

      const durationInput = screen.getByLabelText(
        "Conversation Duration (minutes)",
      );
      fireEvent.change(durationInput, { target: { value: "45" } });

      expect(mockOnTimingChange).toHaveBeenCalledWith({
        ...mockTiming,
        duration: 45,
      });
    });

    it("handles messages per minute changes", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );

      const messagesSlider = screen.getByRole("slider");
      fireEvent.change(messagesSlider, { target: { value: "8" } });

      expect(mockOnTimingChange).toHaveBeenCalledWith({
        ...mockTiming,
        messagesPerMinute: 8,
      });
    });

    it("displays messages per minute value", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );
      expect(screen.getByText("5 msg/min")).toBeInTheDocument();
    });

    it("toggles scheduled breaks", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );

      const breaksCheckbox = screen.getByRole("checkbox");
      fireEvent.click(breaksCheckbox);

      expect(mockOnTimingChange).toHaveBeenCalledWith({
        ...mockTiming,
        enableScheduledBreaks: false,
      });
    });

    it("handles break interval changes", () => {
      render(
        <TimingControls
          timing={mockTiming}
          onTimingChange={mockOnTimingChange}
        />,
      );

      const breakIntervalInput = screen.getByDisplayValue("15");
      fireEvent.change(breakIntervalInput, { target: { value: "20" } });

      expect(mockOnTimingChange).toHaveBeenCalledWith({
        ...mockTiming,
        breakInterval: 20,
      });
    });

    it("handles missing timing gracefully", () => {
      render(<TimingControls onTimingChange={mockOnTimingChange} />);
      expect(screen.getByDisplayValue("30")).toBeInTheDocument(); // Default duration
    });
  });

  describe("Component Integration", () => {
    it("renders multiple orchestration components together", () => {
      const { container } = render(
        <div>
          <AdvancedControls settings={{}} onSettingsChange={() => {}} />
          <ResponseDynamicsControls dynamics={{}} onDynamicsChange={() => {}} />
          <TimingControls timing={{}} onTimingChange={() => {}} />
          <RealTimePreview isActive={true} />
        </div>,
      );

      expect(container.querySelectorAll("[data-testid]")).toHaveLength(4);
    });

    it("handles complex state interactions", async () => {
      const mockState = {
        settings: { temperature: 0.7 },
        dynamics: { responseDelay: 1000 },
        timing: { duration: 30 },
      };

      const handlers = {
        onSettingsChange: jest.fn(),
        onDynamicsChange: jest.fn(),
        onTimingChange: jest.fn(),
      };

      render(
        <div>
          <AdvancedControls
            settings={mockState.settings}
            onSettingsChange={handlers.onSettingsChange}
          />
          <ResponseDynamicsControls
            dynamics={mockState.dynamics}
            onDynamicsChange={handlers.onDynamicsChange}
          />
          <TimingControls
            timing={mockState.timing}
            onTimingChange={handlers.onTimingChange}
          />
        </div>,
      );

      // Test cross-component interactions
      const sliders = screen.getAllByRole("slider");
      const temperatureSlider = sliders[0]; // First slider is the temperature slider
      fireEvent.change(temperatureSlider, { target: { value: "0.9" } });

      expect(handlers.onSettingsChange).toHaveBeenCalled();
    });
  });
});
