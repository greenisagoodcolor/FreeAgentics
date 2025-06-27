import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { nanoid } from 'nanoid';

// Types from PRD
export interface Message {
  id: string;
  conversationId: string;
  agentId: string;
  content: string;
  timestamp: number;
  type: 'agent' | 'system' | 'user';
  metadata?: {
    respondingTo?: string;
    processingTime?: number;
    confidence?: number;
  };
  status: 'sending' | 'delivered' | 'failed';
}

export interface Conversation {
  id: string;
  name: string;
  type: 'all' | 'user-initiated' | 'autonomous' | 'proximity';
  participants: string[]; // Agent IDs
  messages: Message[];
  startTime: number;
  lastActivity: number;
  messageRate: number; // messages per minute
  isActive: boolean;
}

export interface ConversationFilters {
  type: 'all' | 'user-initiated' | 'autonomous' | 'proximity';
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

const initialState: ConversationState = {
  conversations: {
    main: {
      id: 'main',
      name: 'Main Conversation',
      type: 'all',
      participants: [],
      messages: [],
      startTime: Date.now(),
      lastActivity: Date.now(),
      messageRate: 0,
      isActive: true,
    },
  },
  activeConversationId: 'main',
  messageQueue: [],
  filters: {
    type: 'all',
    agentIds: [],
    searchQuery: '',
  },
  autoScroll: true,
  unreadCounts: {},
  typingIndicators: {},
};

const conversationSlice = createSlice({
  name: 'conversations',
  initialState,
  reducers: {
    // Add message to conversation
    addMessage: (state, action: PayloadAction<Omit<Message, 'id' | 'timestamp'>>) => {
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
          m => m.timestamp > Date.now() - timeWindow
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
    queueMessage: (state, action: PayloadAction<Omit<Message, 'id' | 'timestamp' | 'status'>>) => {
      const message: Message = {
        ...action.payload,
        id: nanoid(),
        timestamp: Date.now(),
        status: 'sending',
      };
      state.messageQueue.push(message);
    },

    // Process message from queue
    processQueuedMessage: (state, action: PayloadAction<string>) => {
      const messageId = action.payload;
      const queueIndex = state.messageQueue.findIndex(m => m.id === messageId);
      
      if (queueIndex !== -1) {
        const message = state.messageQueue[queueIndex];
        message.status = 'delivered';
        
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
    createConversation: (state, action: PayloadAction<{
      name: string;
      type: Conversation['type'];
      participants?: string[];
    }>) => {
      const { name, type, participants = [] } = action.payload;
      const conversationId = nanoid();
      
      state.conversations[conversationId] = {
        id: conversationId,
        name,
        type,
        participants,
        messages: [],
        startTime: Date.now(),
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
    updateFilters: (state, action: PayloadAction<Partial<ConversationFilters>>) => {
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
    setTypingIndicators: (state, action: PayloadAction<{
      conversationId: string;
      agentIds: string[];
    }>) => {
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
    updateMessageStatus: (state, action: PayloadAction<{
      messageId: string;
      status: Message['status'];
    }>) => {
      const { messageId, status } = action.payload;
      
      // Check in queue first
      const queuedMessage = state.messageQueue.find(m => m.id === messageId);
      if (queuedMessage) {
        queuedMessage.status = status;
        return;
      }
      
      // Check in conversations
      Object.values(state.conversations).forEach(conversation => {
        const message = conversation.messages.find(m => m.id === messageId);
        if (message) {
          message.status = status;
        }
      });
    },

    // Batch add messages (for initial load or import)
    batchAddMessages: (state, action: PayloadAction<{
      conversationId: string;
      messages: Omit<Message, 'id' | 'timestamp'>[];
    }>) => {
      const { conversationId, messages } = action.payload;
      const conversation = state.conversations[conversationId];
      
      if (conversation) {
        const processedMessages = messages.map(msg => ({
          ...msg,
          id: nanoid(),
          timestamp: Date.now(),
        }));
        
        conversation.messages.push(...processedMessages);
        conversation.lastActivity = Date.now();
        
        // Update participants
        const newParticipants = [...new Set(messages.map(m => m.agentId))];
        newParticipants.forEach(agentId => {
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
      if (conversationId !== 'main') { // Prevent deleting main conversation
        delete state.conversations[conversationId];
        delete state.unreadCounts[conversationId];
        delete state.typingIndicators[conversationId];
        
        if (state.activeConversationId === conversationId) {
          state.activeConversationId = 'main';
        }
      }
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
} = conversationSlice.actions;

export default conversationSlice.reducer; 