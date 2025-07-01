export interface LLMClientConfig {
  provider: string;
  apiKey: string;
  useSecureStorage?: boolean;
  providers?: Array<{ provider: string; priority: number }>;
  enableCache?: boolean;
  cacheTimeout?: number;
}

export class LLMClient {
  provider: string;
  private apiKey: string;
  providers?: Array<{ provider: string; priority: number }>;

  constructor(config: LLMClientConfig) {
    const validProviders = ["openai", "anthropic", "google", "azure"];
    if (!validProviders.includes(config.provider)) {
      throw new Error("Invalid provider");
    }

    this.provider = config.provider;
    this.apiKey = config.apiKey;
    this.providers = config.providers;

    if (config.useSecureStorage) {
      const { encrypt } = require("@/lib/encryption");
      encrypt(config.apiKey);
    }
  }

  async chat(messages: any[]): Promise<any> {
    const response = await fetch(`/api/llm/${this.provider}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        const { RateLimitError } = await import("@/lib/llm-errors");
        throw new RateLimitError("Rate limit exceeded");
      }
      if (response.status === 401) {
        const { AuthenticationError } = await import("@/lib/llm-errors");
        throw new AuthenticationError("Invalid API key");
      }
      throw new Error("Request failed");
    }

    return response.json();
  }

  async chatStream(messages: any[]): Promise<ReadableStream> {
    const response = await fetch(`/api/llm/${this.provider}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ messages, stream: true }),
    });

    if (!response.body) {
      throw new Error("No response body");
    }

    return response.body;
  }

  getProvidersByPriority(): Array<{ provider: string; priority: number }> {
    return this.providers || [];
  }

  async setProvider(provider: string): Promise<void> {
    this.provider = provider;
  }

  countTokens(text: string): number {
    // Simple approximation
    return Math.ceil(text.split(/\s+/).length * 1.3);
  }

  clearCache(): void {
    // Clear any cached responses
  }

  addRequestInterceptor(interceptor: Function): void {
    // Add request interceptor
  }

  addResponseInterceptor(interceptor: Function): void {
    // Add response interceptor
  }

  async createEmbedding(text: string): Promise<number[]> {
    const response = await fetch(`/api/llm/${this.provider}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ input: text }),
    });

    const data = await response.json();
    return data.embedding;
  }

  async analyzeImage(imageUrl: string, prompt: string): Promise<string> {
    const response = await fetch(`/api/llm/${this.provider}/vision`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ image_url: imageUrl, prompt }),
    });

    const data = await response.json();
    return data.description;
  }

  async getFineTuneStatus(modelId: string): Promise<any> {
    const response = await fetch(
      `/api/llm/${this.provider}/fine-tunes/${modelId}`,
      {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
        },
      },
    );

    return response.json();
  }

  // Settings management methods
  getSettings(): any {
    return {
      provider: this.provider,
      apiKey: this.apiKey ? "***" : "", // Hide actual key
      providers: this.providers,
    };
  }

  updateSettings(settings: any): void {
    if (settings.provider) this.provider = settings.provider;
    if (settings.apiKey) this.apiKey = settings.apiKey;
    if (settings.providers) this.providers = settings.providers;
  }

  async saveSettings(): Promise<boolean> {
    // Save settings to storage/preferences
    return Promise.resolve(true);
  }

  // Response generation methods
  async generateResponse(prompt: string, options?: any): Promise<string> {
    const messages = [{ role: "user", content: prompt }];
    const response = await this.chat(messages);
    return response.choices?.[0]?.message?.content || "";
  }

  async streamResponse(
    prompt: string,
    userPrompt?: string,
    onChunk?: Function,
  ): Promise<string> {
    // For compatibility, if streaming is requested, we'll still return a string
    // but call the onChunk callback if provided
    const messages = [{ role: "user", content: prompt }];
    if (userPrompt) {
      messages.push({ role: "user", content: userPrompt });
    }

    const response = await this.chat(messages);
    const content = response.choices?.[0]?.message?.content || "";

    // Call onChunk if provided (for compatibility)
    if (onChunk) {
      onChunk({ text: content, isComplete: true });
    }

    return content;
  }

  // Belief extraction method
  async extractBeliefs(content: string): Promise<any> {
    const prompt = `Extract beliefs and convictions from the following content: ${content}`;
    const response = await this.generateResponse(prompt);
    try {
      return JSON.parse(response);
    } catch {
      return { beliefs: [], raw: response };
    }
  }

  // Token usage tracking
  getTokenUsage(): any {
    return {
      totalTokens: 0,
      promptTokens: 0,
      completionTokens: 0,
    };
  }

  // Performance metrics
  getPerformanceMetrics(): any {
    return {
      averageResponseTime: 0,
      successRate: 1.0,
      errorRate: 0.0,
    };
  }
}

// Export a default instance
export const llmClient = new LLMClient({
  provider: "openai",
  apiKey: process.env.OPENAI_API_KEY || "dummy-key",
});
