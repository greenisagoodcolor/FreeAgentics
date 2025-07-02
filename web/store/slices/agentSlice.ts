import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { nanoid } from "nanoid";
import type { Agent } from "@/lib/types";

// Types from PRD
export interface AgentTemplate {
  id: string;
  name: string;
  category: "researcher" | "student" | "expert" | "generalist" | "contrarian";
  defaultBiography: string;
  defaultKnowledgeDomains: string[];
  defaultParameters: {
    responseThreshold: number; // 0-1
    turnTakingProbability: number; // 0-1
    conversationEngagement: number; // 0-1
  };
  avatarUrl: string;
  icon: string; // Icon component name
  color: string;
}

// Use the unified Agent interface from @/lib/types
// The Agent interface is now imported from the main types file

interface AgentState {
  agents: Record<string, Agent>;
  templates: Record<string, AgentTemplate>;
  selectedAgentId: string | null;
  typingAgents: string[];
  agentOrder: string[]; // For drag-and-drop reordering
}

// Default templates from PRD
const defaultTemplates: Record<string, AgentTemplate> = {
  explorer: {
    id: "explorer",
    name: "Explorer",
    category: "researcher",
    defaultBiography:
      "An adventurous agent that discovers new territories and maps unknown environments. Specializes in exploration and discovery.",
    defaultKnowledgeDomains: [
      "exploration",
      "mapping",
      "discovery",
      "navigation",
    ],
    defaultParameters: {
      responseThreshold: 0.6,
      turnTakingProbability: 0.7,
      conversationEngagement: 0.8,
    },
    avatarUrl: "/avatars/explorer.svg",
    icon: "Search",
    color: "#10B981",
  },
  merchant: {
    id: "merchant",
    name: "Merchant",
    category: "expert",
    defaultBiography:
      "A savvy trader that optimizes resource allocation and market dynamics. Expert in negotiations and value assessment.",
    defaultKnowledgeDomains: [
      "trading",
      "economics",
      "negotiation",
      "resource-management",
    ],
    defaultParameters: {
      responseThreshold: 0.7,
      turnTakingProbability: 0.6,
      conversationEngagement: 0.7,
    },
    avatarUrl: "/avatars/merchant.svg",
    icon: "ShoppingCart",
    color: "#3B82F6",
  },
  scholar: {
    id: "scholar",
    name: "Scholar",
    category: "student",
    defaultBiography:
      "A learned agent that analyzes patterns and synthesizes knowledge. Dedicated to understanding and teaching.",
    defaultKnowledgeDomains: ["analysis", "synthesis", "education", "research"],
    defaultParameters: {
      responseThreshold: 0.8,
      turnTakingProbability: 0.5,
      conversationEngagement: 0.6,
    },
    avatarUrl: "/avatars/scholar.svg",
    icon: "BookOpen",
    color: "#8B5CF6",
  },
  guardian: {
    id: "guardian",
    name: "Guardian",
    category: "expert",
    defaultBiography:
      "A protective agent that safeguards systems and responds to threats. Specializes in security and defense.",
    defaultKnowledgeDomains: [
      "security",
      "defense",
      "protection",
      "threat-analysis",
    ],
    defaultParameters: {
      responseThreshold: 0.5,
      turnTakingProbability: 0.8,
      conversationEngagement: 0.6,
    },
    avatarUrl: "/avatars/guardian.svg",
    icon: "Shield",
    color: "#EF4444",
  },
  generalist: {
    id: "generalist",
    name: "Generalist",
    category: "generalist",
    defaultBiography:
      "An adaptable problem solver with broad capabilities. Can handle diverse tasks and situations.",
    defaultKnowledgeDomains: [
      "problem-solving",
      "adaptation",
      "general-knowledge",
      "collaboration",
    ],
    defaultParameters: {
      responseThreshold: 0.6,
      turnTakingProbability: 0.6,
      conversationEngagement: 0.7,
    },
    avatarUrl: "/avatars/generalist.svg",
    icon: "Brain",
    color: "#F59E0B",
  },
};

// Demo agents for CEO presentation
const demoAgents: Record<string, Agent> = {
  "demo-agent-1": {
    id: "demo-agent-1",
    name: "Explorer Alpha",
    templateId: "explorer",
    biography:
      "An adventurous agent that discovers new territories and maps unknown environments. Specializes in exploration and discovery.",
    knowledgeDomains: ["exploration", "mapping", "discovery", "navigation"],
    parameters: {
      responseThreshold: 0.6,
      turnTakingProbability: 0.7,
      conversationEngagement: 0.8,
    },
    status: "active",
    avatarUrl: "/avatars/explorer.svg",
    color: "#10B981",
    createdAt: Date.now() - 7200000,
    lastActive: Date.now() - 300000,
    inConversation: true,
    autonomyEnabled: true,
    activityMetrics: {
      messagesCount: 47,
      beliefCount: 12,
      responseTime: [340, 280, 410, 290],
    },
  },
  "demo-agent-2": {
    id: "demo-agent-2",
    name: "Scholar Beta",
    templateId: "scholar",
    biography:
      "A learned agent that analyzes patterns and synthesizes knowledge. Dedicated to understanding and teaching.",
    knowledgeDomains: ["analysis", "synthesis", "education", "research"],
    parameters: {
      responseThreshold: 0.8,
      turnTakingProbability: 0.5,
      conversationEngagement: 0.6,
    },
    status: "active",
    avatarUrl: "/avatars/scholar.svg",
    color: "#8B5CF6",
    createdAt: Date.now() - 5400000,
    lastActive: Date.now() - 120000,
    inConversation: true,
    autonomyEnabled: true,
    activityMetrics: {
      messagesCount: 38,
      beliefCount: 19,
      responseTime: [520, 480, 390, 450],
    },
  },
  "demo-agent-3": {
    id: "demo-agent-3",
    name: "Merchant Gamma",
    templateId: "merchant",
    biography:
      "A savvy trader that optimizes resource allocation and market dynamics. Expert in negotiations and value assessment.",
    knowledgeDomains: [
      "trading",
      "economics",
      "negotiation",
      "resource-management",
    ],
    parameters: {
      responseThreshold: 0.7,
      turnTakingProbability: 0.6,
      conversationEngagement: 0.7,
    },
    status: "idle",
    avatarUrl: "/avatars/merchant.svg",
    color: "#3B82F6",
    createdAt: Date.now() - 3600000,
    lastActive: Date.now() - 600000,
    inConversation: false,
    autonomyEnabled: false,
    activityMetrics: {
      messagesCount: 23,
      beliefCount: 8,
      responseTime: [380, 320, 290, 350],
    },
  },
  "demo-agent-4": {
    id: "demo-agent-4",
    name: "Guardian Delta",
    templateId: "guardian",
    biography:
      "A protective agent that safeguards systems and responds to threats. Specializes in security and defense.",
    knowledgeDomains: ["security", "defense", "protection", "threat-analysis"],
    parameters: {
      responseThreshold: 0.5,
      turnTakingProbability: 0.8,
      conversationEngagement: 0.6,
    },
    status: "processing",
    avatarUrl: "/avatars/guardian.svg",
    color: "#EF4444",
    createdAt: Date.now() - 1800000,
    lastActive: Date.now() - 30000,
    inConversation: true,
    autonomyEnabled: true,
    activityMetrics: {
      messagesCount: 31,
      beliefCount: 6,
      responseTime: [210, 190, 240, 220],
    },
  },
};

const initialState: AgentState = {
  agents: demoAgents, // ← NOW HAS DEMO AGENTS
  templates: defaultTemplates,
  selectedAgentId: null,
  typingAgents: ["demo-agent-4"], // Guardian is typing
  agentOrder: ["demo-agent-1", "demo-agent-2", "demo-agent-3", "demo-agent-4"],
};

const agentSlice = createSlice({
  name: "agents",
  initialState,
  reducers: {
    // Create agent from template
    createAgent: (
      state,
      action: PayloadAction<{
        templateId: string;
        name?: string;
        parameterOverrides?: Partial<Agent["parameters"]>;
      }>,
    ) => {
      const { templateId, name, parameterOverrides } = action.payload;
      const template = state.templates[templateId];
      if (!template) return;

      const agentId = nanoid();
      const agentNumber =
        Object.keys(state.agents).filter(
          (id) => state.agents[id].templateId === templateId,
        ).length + 1;

      const agent: Agent = {
        id: agentId,
        name: name || `${template.name} ${agentNumber}`,
        templateId,
        biography: template.defaultBiography,
        knowledgeDomains: [...template.defaultKnowledgeDomains],
        parameters: {
          ...template.defaultParameters,
          ...parameterOverrides,
        },
        status: "idle",
        avatarUrl: template.avatarUrl,
        color: template.color,
        createdAt: Date.now(),
        lastActive: Date.now(),
        inConversation: false,
        autonomyEnabled: false,
        activityMetrics: {
          messagesCount: 0,
          beliefCount: 0,
          responseTime: [],
        },
      };

      state.agents[agentId] = agent;
      state.agentOrder.push(agentId);
    },

    // Update agent status
    updateAgentStatus: (
      state,
      action: PayloadAction<{
        agentId: string;
        status: Agent["status"];
      }>,
    ) => {
      const { agentId, status } = action.payload;
      if (state.agents[agentId]) {
        state.agents[agentId].status = status;
        state.agents[agentId].lastActive = Date.now();
      }
    },

    // Set typing agents
    setTypingAgents: (state, action: PayloadAction<string[]>) => {
      state.typingAgents = action.payload;
      // Update agent statuses
      action.payload.forEach((agentId) => {
        if (state.agents[agentId]) {
          state.agents[agentId].status = "typing";
        }
      });
    },

    // Select agent
    selectAgent: (state, action: PayloadAction<string | null>) => {
      state.selectedAgentId = action.payload;
    },

    // Update agent position (for spatial grid)
    updateAgentPosition: (
      state,
      action: PayloadAction<{
        agentId: string;
        position: { x: number; y: number };
      }>,
    ) => {
      const { agentId, position } = action.payload;
      if (state.agents[agentId]) {
        state.agents[agentId].position = position;
      }
    },

    // Update agent parameters
    updateAgentParameters: (
      state,
      action: PayloadAction<{
        agentId: string;
        parameters: Partial<Agent["parameters"]>;
      }>,
    ) => {
      const { agentId, parameters } = action.payload;
      if (state.agents[agentId]) {
        state.agents[agentId].parameters = {
          ...state.agents[agentId].parameters,
          ...parameters,
        };
      }
    },

    // Toggle agent autonomy
    toggleAgentAutonomy: (state, action: PayloadAction<string>) => {
      const agentId = action.payload;
      if (state.agents[agentId]) {
        state.agents[agentId].autonomyEnabled =
          !state.agents[agentId].autonomyEnabled;
      }
    },

    // Update activity metrics
    updateActivityMetrics: (
      state,
      action: PayloadAction<{
        agentId: string;
        metrics: Partial<Agent["activityMetrics"]>;
      }>,
    ) => {
      const { agentId, metrics } = action.payload;
      if (state.agents[agentId]) {
        state.agents[agentId].activityMetrics = {
          ...state.agents[agentId].activityMetrics,
          ...metrics,
        };
      }
    },

    // Reorder agents (for drag-and-drop)
    reorderAgents: (state, action: PayloadAction<string[]>) => {
      state.agentOrder = action.payload;
    },

    // Delete agent
    deleteAgent: (state, action: PayloadAction<string>) => {
      const agentId = action.payload;
      delete state.agents[agentId];
      state.agentOrder = state.agentOrder.filter((id) => id !== agentId);
      if (state.selectedAgentId === agentId) {
        state.selectedAgentId = null;
      }
    },

    // Batch create agents (Quick Start)
    quickStartAgents: (state) => {
      const templates = ["explorer", "scholar", "merchant"];
      templates.forEach((templateId, index) => {
        const template = state.templates[templateId];
        if (!template) return;

        const agentId = nanoid();
        const agent: Agent = {
          id: agentId,
          name: `${template.name} 1`,
          templateId,
          biography: template.defaultBiography,
          knowledgeDomains: [...template.defaultKnowledgeDomains],
          parameters: { ...template.defaultParameters },
          status: "idle",
          avatarUrl: template.avatarUrl,
          color: template.color,
          createdAt: Date.now() + index,
          lastActive: Date.now() + index,
          inConversation: false,
          autonomyEnabled: true,
          activityMetrics: {
            messagesCount: 0,
            beliefCount: 0,
            responseTime: [],
          },
        };

        state.agents[agentId] = agent;
        state.agentOrder.push(agentId);
      });
    },

    // Generic update agent
    updateAgent: (
      state,
      action: PayloadAction<{
        id: string;
        updates: Partial<Agent>;
      }>,
    ) => {
      const { id, updates } = action.payload;
      if (state.agents[id]) {
        state.agents[id] = {
          ...state.agents[id],
          ...updates,
          lastActive: Date.now(),
        };
      }
    },

    // Add agent
    addAgent: (state, action: PayloadAction<Agent>) => {
      const agent = action.payload;
      state.agents[agent.id] = agent;
      if (!state.agentOrder.includes(agent.id)) {
        state.agentOrder.push(agent.id);
      }
    },

    // Remove agent (alias for deleteAgent)
    removeAgent: (state, action: PayloadAction<string>) => {
      const agentId = action.payload;
      delete state.agents[agentId];
      state.agentOrder = state.agentOrder.filter((id) => id !== agentId);
      if (state.selectedAgentId === agentId) {
        state.selectedAgentId = null;
      }
    },

    // Set all agents (for bulk updates)
    setAgents: (state, action: PayloadAction<Agent[]>) => {
      state.agents = {};
      state.agentOrder = [];
      action.payload.forEach((agent) => {
        state.agents[agent.id] = agent;
        state.agentOrder.push(agent.id);
      });
    },

    // Set selected agent
    setSelectedAgent: (state, action: PayloadAction<Agent | null>) => {
      state.selectedAgentId = action.payload?.id || null;
    },
    
    // Demo data actions for compatibility
    setDemoAgent: (state, action: PayloadAction<Agent>) => {
      const agent = action.payload;
      state.agents[agent.id] = agent;
      if (!state.agentOrder.includes(agent.id)) {
        state.agentOrder.push(agent.id);
      }
    },
    
    clearAgents: (state) => {
      state.agents = {};
      state.agentOrder = [];
      state.selectedAgentId = null;
    },
  },
});

export const {
  createAgent,
  updateAgentStatus,
  setTypingAgents,
  selectAgent,
  updateAgentPosition,
  updateAgentParameters,
  toggleAgentAutonomy,
  updateActivityMetrics,
  reorderAgents,
  deleteAgent,
  quickStartAgents,
  updateAgent,
  addAgent,
  removeAgent,
  setAgents,
  setSelectedAgent,
  setDemoAgent,
  clearAgents,
} = agentSlice.actions;

export default agentSlice.reducer;
