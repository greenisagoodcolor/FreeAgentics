import { PromptAgent } from "@/types/agent";
import { apiGet, apiPost, apiPut, apiDelete, ApiError } from "./api";

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface KnowledgeGraph {
  nodes: Array<{
    id: string;
    label: string;
    type: string;
    properties?: Record<string, unknown>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relationship: string;
  }>;
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl = "/api") {
    this.baseUrl = baseUrl;
  }

  private async request<T>(apiCall: () => Promise<T>): Promise<ApiResponse<T>> {
    try {
      const data = await apiCall();
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error instanceof ApiError ? error.message : error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  // Agent endpoints
  async getAgents(): Promise<ApiResponse<PromptAgent[]>> {
    return this.request<PromptAgent[]>(() => apiGet(`${this.baseUrl}/agents`));
  }

  async createAgent(data: { description: string }): Promise<ApiResponse<PromptAgent>> {
    return this.request<PromptAgent>(() => apiPost(`${this.baseUrl}/agents`, data));
  }

  async updateAgent(id: string, data: Partial<PromptAgent>): Promise<ApiResponse<PromptAgent>> {
    return this.request<PromptAgent>(() => apiPut(`${this.baseUrl}/agents/${id}`, data));
  }

  async deleteAgent(id: string): Promise<ApiResponse<void>> {
    return this.request<void>(() => apiDelete(`${this.baseUrl}/agents/${id}`));
  }

  // Prompt processing
  async processPrompt(data: { prompt: string; conversationId?: string }): Promise<
    ApiResponse<{
      agents: PromptAgent[];
      knowledgeGraph: KnowledgeGraph;
      suggestions: string[];
      conversationId: string;
    }>
  > {
    return this.request(() => apiPost(`${this.baseUrl}/process-prompt`, data));
  }

  // Knowledge graph
  async getKnowledgeGraph(): Promise<ApiResponse<KnowledgeGraph>> {
    return this.request<KnowledgeGraph>(() => apiGet(`${this.baseUrl}/knowledge-graph`));
  }

  // Suggestions
  async getSuggestions(prompt: string): Promise<ApiResponse<string[]>> {
    return this.request<string[]>(() => apiPost(`${this.baseUrl}/suggestions`, { prompt }));
  }

  // Conversation
  async getConversation(id: string): Promise<ApiResponse<ConversationMessage[]>> {
    return this.request<ConversationMessage[]>(() => apiGet(`${this.baseUrl}/conversations/${id}`));
  }

  async clearConversation(id: string): Promise<ApiResponse<void>> {
    return this.request<void>(() => apiDelete(`${this.baseUrl}/conversations/${id}`));
  }
}

// Default instance
export const apiClient = new ApiClient();
