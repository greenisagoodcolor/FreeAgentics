import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { nanoid } from "nanoid";

// Types from PRD
export interface Message {
  id: string;
  conversationId: string;
  agentId: string;
  content: string;
  timestamp: number;
  type: "agent" | "system" | "user";
  metadata?: {
    respondingTo?: string;
    processingTime?: number;
    confidence?: number;
  };
  status: "sending" | "delivered" | "failed";
}

export interface Conversation {
  id: string;
  name: string;
  type: "all" | "user-initiated" | "autonomous" | "proximity";
  participants: string[]; // Agent IDs
  messages: Message[];
  startTime: number;
  endTime: number | null; // Add missing endTime field
  lastActivity: number;
  messageRate: number; // messages per minute
  isActive: boolean;
}

export interface ConversationFilters {
  type: "all" | "user-initiated" | "autonomous" | "proximity";
  agentIds: string[];
  searchQuery: string;
  dateRange?: {
    start: number;
    end: number;
  };
}

interface ConversationState {
  conversations: Record<string, Conversation>;
  activeConversationId: string | null;
  messageQueue: Message[]; // Pending messages
  filters: ConversationFilters;
  autoScroll: boolean;
  unreadCounts: Record<string, number>;
  typingIndicators: Record<string, string[]>; // conversationId -> agentIds
}

// Demo messages for CEO presentation
const demoMessages: Message[] = [
  {
    id: "msg-1",
    conversationId: "main",
    agentId: "demo-agent-1",
    content:
      "I've discovered an interesting pattern in the knowledge graph topology. The coalition formation nodes seem to cluster around active inference frameworks.",
    timestamp: Date.now() - 1800000,
    type: "agent",
    metadata: {
      processingTime: 340,
      confidence: 0.87,
    },
    status: "delivered",
  },
  {
    id: "msg-2",
    conversationId: "main",
    agentId: "demo-agent-2",
    content:
      "That aligns with my analysis. The belief propagation networks show increased connectivity when agents form coalitions. The emergent communication protocols appear to be self-organizing.",
    timestamp: Date.now() - 1680000,
    type: "agent",
    metadata: {
      respondingTo: "msg-1",
      processingTime: 520,
      confidence: 0.92,
    },
    status: "delivered",
  },
  {
    id: "msg-3",
    conversationId: "main",
    agentId: "demo-agent-3",
    content:
      "From a resource allocation perspective, these coalition patterns optimize for both information sharing and computational efficiency. The market dynamics suggest stable equilibrium states.",
    timestamp: Date.now() - 1560000,
    type: "agent",
    metadata: {
      respondingTo: "msg-2",
      processingTime: 380,
      confidence: 0.79,
    },
    status: "delivered",
  },
  {
    id: "msg-4",
    conversationId: "main",
    agentId: "demo-agent-4",
    content:
      "Security analysis confirms the robustness of these patterns. The guardian protocols maintain system integrity while allowing for adaptive coalition restructuring.",
    timestamp: Date.now() - 1440000,
    type: "agent",
    metadata: {
      respondingTo: "msg-3",
      processingTime: 210,
      confidence: 0.95,
    },
    status: "delivered",
  },
  {
    id: "msg-5",
    conversationId: "main",
    agentId: "demo-agent-1",
    content:
      "The spatial proximity analysis reveals that agents within 2-3 grid units show significantly higher collaboration rates. This could inform our deployment strategies.",
    timestamp: Date.now() - 1320000,
    type: "agent",
    metadata: {
      respondingTo: "msg-4",
      processingTime: 290,
      confidence: 0.83,
    },
    status: "delivered",
  },
];

const initialState: ConversationState = {
  conversations: {
    main: {
      id: "main",
      name: "Main Conversation",
      type: "all",
      participants: [
        "demo-agent-1",
        "demo-agent-2",
        "demo-agent-3",
        "demo-agent-4",
      ],
      messages: demoMessages, // ← NOW HAS DEMO MESSAGES
      startTime: Date.now() - 1800000,
      endTime: null, // Add missing endTime field
      lastActivity: Date.now() - 300000,
      messageRate: 4.2, // messages per minute
      isActive: true,
    },
  },
  activeConversationId: "main",
  messageQueue: [
    {
      id: "queue-1",
      conversationId: "main",
      agentId: "demo-agent-4",
      content:
        "Analyzing new threat vectors in the coalition formation process...",
      timestamp: Date.now(),
      type: "agent",
      metadata: {
        processingTime: 180,
        confidence: 0.88,
      },
      status: "sending",
    },
  ],
  filters: {
    type: "all",
    agentIds: [],
    searchQuery: "",
  },
  autoScroll: true,
  unreadCounts: {},
  typingIndicators: {
    main: ["demo-agent-4"], // Guardian is typing
  },
};

const conversationSlice = createSlice({
  name: "conversations",
  initialState,
  reducers: {
    // Add message to conversation
    addMessage: (
      state,
      action: PayloadAction<Omit<Message, "id" | "timestamp">>,
    ) => {
      const message: Message = {
        ...action.payload,
        id: nanoid(),
        timestamp: Date.now(),
      };

      const conversation = state.conversations[message.conversationId];
      if (conversation) {
        conversation.messages.push(message);
        conversation.lastActivity = message.timestamp;

        // Update participants
        if (!conversation.participants.includes(message.agentId)) {
          conversation.participants.push(message.agentId);
        }

        // Update message rate
        const timeWindow = 60000; // 1 minute
        const recentMessages = conversation.messages.filter(
          (m) => m.timestamp > Date.now() - timeWindow,
        );
        conversation.messageRate = recentMessages.length;

        // Update unread count if not active conversation
        if (state.activeConversationId !== message.conversationId) {
          state.unreadCounts[message.conversationId] =
            (state.unreadCounts[message.conversationId] || 0) + 1;
        }
      }
    },

    // Add message to queue
    queueMessage: (
      state,
      action: PayloadAction<Omit<Message, "id" | "timestamp" | "status">>,
    ) => {
      const message: Message = {
        ...action.payload,
        id: nanoid(),
        timestamp: Date.now(),
        status: "sending",
      };
      state.messageQueue.push(message);
    },

    // Process message from queue
    processQueuedMessage: (state, action: PayloadAction<string>) => {
      const messageId = action.payload;
      const queueIndex = state.messageQueue.findIndex(
        (m) => m.id === messageId,
      );

      if (queueIndex !== -1) {
        const message = state.messageQueue[queueIndex];
        message.status = "delivered";

        // Move to conversation
        const conversation = state.conversations[message.conversationId];
        if (conversation) {
          conversation.messages.push(message);
          conversation.lastActivity = message.timestamp;
        }

        // Remove from queue
        state.messageQueue.splice(queueIndex, 1);
      }
    },

    // Create new conversation
    createConversation: (
      state,
      action: PayloadAction<{
        name: string;
        type: Conversation["type"];
        participants?: string[];
      }>,
    ) => {
      const { name, type, participants = [] } = action.payload;
      const conversationId = nanoid();

      state.conversations[conversationId] = {
        id: conversationId,
        name,
        type,
        participants,
        messages: [],
        startTime: Date.now(),
        endTime: null,
        lastActivity: Date.now(),
        messageRate: 0,
        isActive: true,
      };
    },

    // Set active conversation
    setActiveConversation: (state, action: PayloadAction<string>) => {
      const conversationId = action.payload;
      if (state.conversations[conversationId]) {
        state.activeConversationId = conversationId;
        // Clear unread count
        delete state.unreadCounts[conversationId];
      }
    },

    // Update filters
    updateFilters: (
      state,
      action: PayloadAction<Partial<ConversationFilters>>,
    ) => {
      state.filters = {
        ...state.filters,
        ...action.payload,
      };
    },

    // Toggle auto-scroll
    toggleAutoScroll: (state) => {
      state.autoScroll = !state.autoScroll;
    },

    // Set typing indicators
    setTypingIndicators: (
      state,
      action: PayloadAction<{
        conversationId: string;
        agentIds: string[];
      }>,
    ) => {
      const { conversationId, agentIds } = action.payload;
      if (agentIds.length > 0) {
        state.typingIndicators[conversationId] = agentIds;
      } else {
        delete state.typingIndicators[conversationId];
      }
    },

    // Mark conversation as read
    markAsRead: (state, action: PayloadAction<string>) => {
      const conversationId = action.payload;
      delete state.unreadCounts[conversationId];
    },

    // Update message status
    updateMessageStatus: (
      state,
      action: PayloadAction<{
        messageId: string;
        status: Message["status"];
      }>,
    ) => {
      const { messageId, status } = action.payload;

      // Check in queue first
      const queuedMessage = state.messageQueue.find((m) => m.id === messageId);
      if (queuedMessage) {
        queuedMessage.status = status;
        return;
      }

      // Check in conversations
      Object.values(state.conversations).forEach((conversation) => {
        const message = conversation.messages.find((m) => m.id === messageId);
        if (message) {
          message.status = status;
        }
      });
    },

    // Batch add messages (for initial load or import)
    batchAddMessages: (
      state,
      action: PayloadAction<{
        conversationId: string;
        messages: Omit<Message, "id" | "timestamp">[];
      }>,
    ) => {
      const { conversationId, messages } = action.payload;
      const conversation = state.conversations[conversationId];

      if (conversation) {
        const processedMessages = messages.map((msg) => ({
          ...msg,
          id: nanoid(),
          timestamp: Date.now(),
        }));

        conversation.messages.push(...processedMessages);
        conversation.lastActivity = Date.now();

        // Update participants
        const newParticipants = [...new Set(messages.map((m) => m.agentId))];
        newParticipants.forEach((agentId) => {
          if (!conversation.participants.includes(agentId)) {
            conversation.participants.push(agentId);
          }
        });
      }
    },

    // Clear conversation
    clearConversation: (state, action: PayloadAction<string>) => {
      const conversationId = action.payload;
      if (state.conversations[conversationId]) {
        state.conversations[conversationId].messages = [];
        state.conversations[conversationId].messageRate = 0;
      }
    },

    // Delete conversation
    deleteConversation: (state, action: PayloadAction<string>) => {
      const conversationId = action.payload;
      if (conversationId !== "main") {
        // Prevent deleting main conversation
        delete state.conversations[conversationId];
        delete state.unreadCounts[conversationId];
        delete state.typingIndicators[conversationId];

        if (state.activeConversationId === conversationId) {
          state.activeConversationId = "main";
        }
      }
    },
    
    // Demo data actions for compatibility
    setDemoConversation: (state, action: PayloadAction<Conversation>) => {
      const conversation = action.payload;
      state.conversations[conversation.id] = conversation;
    },
    
    clearConversations: (state) => {
      state.conversations = { main: state.conversations.main }; // Keep main conversation
      state.unreadCounts = {};
      state.typingIndicators = {};
      state.activeConversationId = "main";
    },
  },
});

export const {
  addMessage,
  queueMessage,
  processQueuedMessage,
  createConversation,
  setActiveConversation,
  updateFilters,
  toggleAutoScroll,
  setTypingIndicators,
  markAsRead,
  updateMessageStatus,
  batchAddMessages,
  clearConversation,
  deleteConversation,
  setDemoConversation,
  clearConversations,
} = conversationSlice.actions;

export default conversationSlice.reducer;
