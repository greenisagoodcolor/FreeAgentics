import { PromptAgent } from "@/types/agent";

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
  private headers: HeadersInit;

  constructor(baseUrl = "/api") {
    this.baseUrl = baseUrl;
    this.headers = {
      "Content-Type": "application/json",
    };
  }

  private async request<T>(path: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        headers: {
          ...this.headers,
          ...options.headers,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }

      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  // Agent endpoints
  async getAgents(): Promise<ApiResponse<PromptAgent[]>> {
    return this.request<PromptAgent[]>("/agents");
  }

  async createAgent(data: { description: string }): Promise<ApiResponse<PromptAgent>> {
    return this.request<PromptAgent>("/agents", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async updateAgent(id: string, data: Partial<PromptAgent>): Promise<ApiResponse<PromptAgent>> {
    return this.request<PromptAgent>(`/agents/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async deleteAgent(id: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/agents/${id}`, {
      method: "DELETE",
    });
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
    return this.request("/process-prompt", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Knowledge graph
  async getKnowledgeGraph(): Promise<ApiResponse<KnowledgeGraph>> {
    return this.request<KnowledgeGraph>("/knowledge-graph");
  }

  // Suggestions
  async getSuggestions(prompt: string): Promise<ApiResponse<string[]>> {
    return this.request<string[]>("/suggestions", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    });
  }

  // Conversation
  async getConversation(id: string): Promise<ApiResponse<ConversationMessage[]>> {
    return this.request<ConversationMessage[]>(`/conversations/${id}`);
  }

  async clearConversation(id: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/conversations/${id}`, {
      method: "DELETE",
    });
  }
}

// Default instance
export const apiClient = new ApiClient();
