/**
 * Backend Services Tests
 *
 * Comprehensive tests for backend service integrations
 * following ADR-007 testing requirements.
 */

import { jest } from "@jest/globals";

// Mock backend service implementations
class AgentService {
  private agents: Map<string, any> = new Map();
  private wsConnections: Set<WebSocket> = new Set();

  async createAgent(config: any): Promise<any> {
    const agent = {
      id: `agent-${Date.now()}`,
      name: config.name || "Unnamed Agent",
      type: config.type || "generic",
      status: "created",
      energy: 1.0,
      beliefs: config.initialBeliefs || {},
      goals: config.goals || [],
      createdAt: new Date(),
      lastActive: new Date(),
    };

    this.agents.set(agent.id, agent);
    this.broadcastUpdate("agent_created", agent);
    return agent;
  }

  async getAgent(id: string): Promise<any | null> {
    return this.agents.get(id) || null;
  }

  async listAgents(filters?: any): Promise<any[]> {
    let agentList = Array.from(this.agents.values());

    if (filters?.status) {
      agentList = agentList.filter((agent) => agent.status === filters.status);
    }

    if (filters?.type) {
      agentList = agentList.filter((agent) => agent.type === filters.type);
    }

    return agentList;
  }

  async updateAgent(id: string, updates: any): Promise<any | null> {
    const agent = this.agents.get(id);
    if (!agent) return null;

    const updatedAgent = { ...agent, ...updates, lastActive: new Date() };
    this.agents.set(id, updatedAgent);
    this.broadcastUpdate("agent_updated", updatedAgent);
    return updatedAgent;
  }

  async deleteAgent(id: string): Promise<boolean> {
    const existed = this.agents.has(id);
    if (existed) {
      this.agents.delete(id);
      this.broadcastUpdate("agent_deleted", { id });
    }
    return existed;
  }

  async activateAgent(id: string): Promise<boolean> {
    const agent = await this.updateAgent(id, { status: "active" });
    return agent !== null;
  }

  async deactivateAgent(id: string): Promise<boolean> {
    const agent = await this.updateAgent(id, { status: "inactive" });
    return agent !== null;
  }

  subscribeToUpdates(ws: WebSocket): void {
    this.wsConnections.add(ws);
  }

  unsubscribeFromUpdates(ws: WebSocket): void {
    this.wsConnections.delete(ws);
  }

  private broadcastUpdate(type: string, data: any): void {
    const message = JSON.stringify({ type, data, timestamp: new Date() });
    this.wsConnections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }

  getStats(): any {
    const agents = Array.from(this.agents.values());
    return {
      totalAgents: agents.length,
      activeAgents: agents.filter((a) => a.status === "active").length,
      inactiveAgents: agents.filter((a) => a.status === "inactive").length,
      averageEnergy:
        agents.reduce((sum, a) => sum + (a.energy || 0), 0) / agents.length ||
        0,
    };
  }
}

class ConversationService {
  private conversations: Map<string, any> = new Map();
  private messageQueue: any[] = [];
  private isProcessing = false;

  async createConversation(config: any): Promise<any> {
    const conversation = {
      id: `conv-${Date.now()}`,
      title: config.title || "Untitled Conversation",
      participants: config.participants || [],
      messages: [],
      status: "created",
      settings: config.settings || {},
      createdAt: new Date(),
      lastActivity: new Date(),
    };

    this.conversations.set(conversation.id, conversation);
    return conversation;
  }

  async getConversation(id: string): Promise<any | null> {
    return this.conversations.get(id) || null;
  }

  async listConversations(filters?: any): Promise<any[]> {
    let convList = Array.from(this.conversations.values());

    if (filters?.status) {
      convList = convList.filter((conv) => conv.status === filters.status);
    }

    if (filters?.participant) {
      convList = convList.filter((conv) =>
        conv.participants.includes(filters.participant),
      );
    }

    return convList.sort(
      (a, b) =>
        new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime(),
    );
  }

  async addMessage(conversationId: string, message: any): Promise<boolean> {
    const conversation = this.conversations.get(conversationId);
    if (!conversation) return false;

    const fullMessage = {
      id: `msg-${Date.now()}`,
      ...message,
      timestamp: new Date(),
    };

    conversation.messages.push(fullMessage);
    conversation.lastActivity = new Date();

    this.queueMessage(conversationId, fullMessage);
    return true;
  }

  async startConversation(id: string): Promise<boolean> {
    const conversation = this.conversations.get(id);
    if (!conversation) return false;

    conversation.status = "active";
    conversation.startedAt = new Date();
    this.startMessageProcessing();
    return true;
  }

  async pauseConversation(id: string): Promise<boolean> {
    const conversation = this.conversations.get(id);
    if (!conversation) return false;

    conversation.status = "paused";
    return true;
  }

  async stopConversation(id: string): Promise<boolean> {
    const conversation = this.conversations.get(id);
    if (!conversation) return false;

    conversation.status = "stopped";
    conversation.endedAt = new Date();
    return true;
  }

  private queueMessage(conversationId: string, message: any): void {
    this.messageQueue.push({ conversationId, message });
  }

  private async startMessageProcessing(): Promise<void> {
    if (this.isProcessing) return;

    this.isProcessing = true;
    while (this.messageQueue.length > 0) {
      const { conversationId, message } = this.messageQueue.shift()!;
      await this.processMessage(conversationId, message);
      await new Promise((resolve) => setTimeout(resolve, 100)); // Rate limiting
    }
    this.isProcessing = false;
  }

  private async processMessage(
    conversationId: string,
    message: any,
  ): Promise<void> {
    // Simulate message processing
    const conversation = this.conversations.get(conversationId);
    if (conversation) {
      conversation.lastProcessedMessage = message.id;
    }
  }

  getQueueStats(): any {
    return {
      queueLength: this.messageQueue.length,
      isProcessing: this.isProcessing,
    };
  }
}

class KnowledgeService {
  private knowledge: Map<string, any> = new Map();
  private searchIndex: Map<string, Set<string>> = new Map();

  async createEntry(entry: any): Promise<any> {
    const knowledgeEntry = {
      id: `knowledge-${Date.now()}`,
      title: entry.title,
      content: entry.content,
      tags: entry.tags || [],
      metadata: entry.metadata || {},
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.knowledge.set(knowledgeEntry.id, knowledgeEntry);
    this.updateSearchIndex(knowledgeEntry);
    return knowledgeEntry;
  }

  async getEntry(id: string): Promise<any | null> {
    return this.knowledge.get(id) || null;
  }

  async searchEntries(query: string, options?: any): Promise<any[]> {
    if (!query.trim()) {
      return Array.from(this.knowledge.values())
        .sort(
          (a, b) =>
            new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
        )
        .slice(0, options?.limit || 50);
    }

    const queryTerms = query.toLowerCase().split(/\s+/);
    const matchingIds = new Set<string>();

    queryTerms.forEach((term) => {
      const ids = this.searchIndex.get(term) || new Set();
      ids.forEach((id) => matchingIds.add(id));
    });

    return Array.from(matchingIds)
      .map((id) => this.knowledge.get(id)!)
      .filter((entry) => entry)
      .sort(
        (a, b) =>
          new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
      );
  }

  async updateEntry(id: string, updates: any): Promise<any | null> {
    const entry = this.knowledge.get(id);
    if (!entry) return null;

    const updatedEntry = { ...entry, ...updates, updatedAt: new Date() };
    this.knowledge.set(id, updatedEntry);
    this.updateSearchIndex(updatedEntry);
    return updatedEntry;
  }

  async deleteEntry(id: string): Promise<boolean> {
    const existed = this.knowledge.has(id);
    if (existed) {
      this.knowledge.delete(id);
      this.removeFromSearchIndex(id);
    }
    return existed;
  }

  async getEntriesByTags(tags: string[]): Promise<any[]> {
    return Array.from(this.knowledge.values())
      .filter((entry) => tags.some((tag) => entry.tags.includes(tag)))
      .sort(
        (a, b) =>
          new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
      );
  }

  async exportKnowledge(format: "json" | "csv" = "json"): Promise<string> {
    const entries = Array.from(this.knowledge.values());

    if (format === "json") {
      return JSON.stringify(entries, null, 2);
    }

    if (format === "csv") {
      const headers = [
        "id",
        "title",
        "content",
        "tags",
        "createdAt",
        "updatedAt",
      ];
      const rows = entries.map((entry) => [
        entry.id,
        entry.title,
        entry.content.replace(/[\n\r]/g, " "),
        entry.tags.join("; "),
        entry.createdAt.toISOString(),
        entry.updatedAt.toISOString(),
      ]);

      return [headers, ...rows].map((row) => row.join(",")).join("\n");
    }

    throw new Error(`Unsupported format: ${format}`);
  }

  async importKnowledge(
    data: string,
    format: "json" | "csv" = "json",
  ): Promise<number> {
    let entries: any[] = [];

    if (format === "json") {
      entries = JSON.parse(data);
    } else if (format === "csv") {
      const lines = data.split("\n");
      const headers = lines[0].split(",");
      entries = lines.slice(1).map((line) => {
        const values = line.split(",");
        return headers.reduce((obj, header, index) => {
          obj[header] = values[index];
          return obj;
        }, {} as any);
      });
    }

    let importedCount = 0;
    for (const entry of entries) {
      if (entry.title && entry.content) {
        await this.createEntry(entry);
        importedCount++;
      }
    }

    return importedCount;
  }

  private updateSearchIndex(entry: any): void {
    const searchableText = `${entry.title} ${entry.content} ${entry.tags.join(" ")}`;
    const terms = searchableText.toLowerCase().split(/\s+/);

    terms.forEach((term) => {
      if (term.length > 2) {
        if (!this.searchIndex.has(term)) {
          this.searchIndex.set(term, new Set());
        }
        this.searchIndex.get(term)!.add(entry.id);
      }
    });
  }

  private removeFromSearchIndex(entryId: string): void {
    this.searchIndex.forEach((idSet) => {
      idSet.delete(entryId);
    });
  }

  getStats(): any {
    const entries = Array.from(this.knowledge.values());
    const allTags = new Set<string>();
    entries.forEach((entry) =>
      entry.tags.forEach((tag: string) => allTags.add(tag)),
    );

    return {
      totalEntries: entries.length,
      totalTags: allTags.size,
      averageTagsPerEntry:
        entries.reduce((sum, entry) => sum + entry.tags.length, 0) /
          entries.length || 0,
      searchIndexSize: this.searchIndex.size,
    };
  }
}

describe("Backend Services", () => {
  describe("AgentService", () => {
    let agentService: AgentService;

    beforeEach(() => {
      agentService = new AgentService();
    });

    describe("Agent Creation", () => {
      it("creates agent with valid configuration", async () => {
        const config = {
          name: "Test Agent",
          type: "explorer",
          initialBeliefs: { exploration: 0.8 },
          goals: ["explore_territory"],
        };

        const agent = await agentService.createAgent(config);

        expect(agent).toMatchObject({
          name: "Test Agent",
          type: "explorer",
          status: "created",
          energy: 1.0,
          beliefs: { exploration: 0.8 },
          goals: ["explore_territory"],
        });
        expect(agent.id).toMatch(/^agent-\d+$/);
        expect(agent.createdAt).toBeInstanceOf(Date);
      });

      it("creates agent with minimal configuration", async () => {
        const agent = await agentService.createAgent({});

        expect(agent.name).toBe("Unnamed Agent");
        expect(agent.type).toBe("generic");
        expect(agent.beliefs).toEqual({});
        expect(agent.goals).toEqual([]);
      });

      it("assigns unique IDs to agents", async () => {
        const agent1 = await agentService.createAgent({ name: "Agent 1" });
        const agent2 = await agentService.createAgent({ name: "Agent 2" });

        expect(agent1.id).not.toBe(agent2.id);
      });
    });

    describe("Agent Retrieval", () => {
      it("retrieves existing agent by ID", async () => {
        const created = await agentService.createAgent({ name: "Test Agent" });
        const retrieved = await agentService.getAgent(created.id);

        expect(retrieved).toEqual(created);
      });

      it("returns null for non-existent agent", async () => {
        const result = await agentService.getAgent("non-existent-id");
        expect(result).toBeNull();
      });

      it("lists all agents", async () => {
        await agentService.createAgent({ name: "Agent 1" });
        await agentService.createAgent({ name: "Agent 2" });

        const agents = await agentService.listAgents();
        expect(agents).toHaveLength(2);
        expect(agents.map((a) => a.name)).toEqual(["Agent 1", "Agent 2"]);
      });

      it("filters agents by status", async () => {
        const agent1 = await agentService.createAgent({ name: "Agent 1" });
        const agent2 = await agentService.createAgent({ name: "Agent 2" });

        await agentService.activateAgent(agent1.id);

        const activeAgents = await agentService.listAgents({
          status: "active",
        });
        expect(activeAgents).toHaveLength(1);
        expect(activeAgents[0].name).toBe("Agent 1");
      });

      it("filters agents by type", async () => {
        await agentService.createAgent({ name: "Explorer", type: "explorer" });
        await agentService.createAgent({ name: "Analyst", type: "analyst" });

        const explorers = await agentService.listAgents({ type: "explorer" });
        expect(explorers).toHaveLength(1);
        expect(explorers[0].name).toBe("Explorer");
      });
    });

    describe("Agent Updates", () => {
      it("updates existing agent", async () => {
        const agent = await agentService.createAgent({ name: "Original Name" });

        const updated = await agentService.updateAgent(agent.id, {
          name: "Updated Name",
          energy: 0.5,
        });

        expect(updated?.name).toBe("Updated Name");
        expect(updated?.energy).toBe(0.5);
        expect(updated?.lastActive).toBeInstanceOf(Date);
      });

      it("returns null when updating non-existent agent", async () => {
        const result = await agentService.updateAgent("non-existent", {
          name: "Test",
        });
        expect(result).toBeNull();
      });

      it("activates agent", async () => {
        const agent = await agentService.createAgent({ name: "Test Agent" });
        const success = await agentService.activateAgent(agent.id);

        expect(success).toBe(true);

        const updated = await agentService.getAgent(agent.id);
        expect(updated?.status).toBe("active");
      });

      it("deactivates agent", async () => {
        const agent = await agentService.createAgent({ name: "Test Agent" });
        await agentService.activateAgent(agent.id);

        const success = await agentService.deactivateAgent(agent.id);
        expect(success).toBe(true);

        const updated = await agentService.getAgent(agent.id);
        expect(updated?.status).toBe("inactive");
      });
    });

    describe("Agent Deletion", () => {
      it("deletes existing agent", async () => {
        const agent = await agentService.createAgent({ name: "Test Agent" });

        const success = await agentService.deleteAgent(agent.id);
        expect(success).toBe(true);

        const retrieved = await agentService.getAgent(agent.id);
        expect(retrieved).toBeNull();
      });

      it("returns false when deleting non-existent agent", async () => {
        const success = await agentService.deleteAgent("non-existent");
        expect(success).toBe(false);
      });
    });

    describe("Statistics", () => {
      it("calculates agent statistics", async () => {
        const agent1 = await agentService.createAgent({ name: "Agent 1" });
        const agent2 = await agentService.createAgent({ name: "Agent 2" });

        await agentService.activateAgent(agent1.id);
        await agentService.updateAgent(agent2.id, { energy: 0.5 });

        const stats = agentService.getStats();

        expect(stats).toEqual({
          totalAgents: 2,
          activeAgents: 1,
          inactiveAgents: 0,
          averageEnergy: 0.75, // (1.0 + 0.5) / 2
        });
      });

      it("handles empty agent list", () => {
        const stats = agentService.getStats();

        expect(stats).toEqual({
          totalAgents: 0,
          activeAgents: 0,
          inactiveAgents: 0,
          averageEnergy: 0,
        });
      });
    });
  });

  describe("ConversationService", () => {
    let conversationService: ConversationService;

    beforeEach(() => {
      conversationService = new ConversationService();
    });

    describe("Conversation Creation", () => {
      it("creates conversation with configuration", async () => {
        const config = {
          title: "Test Conversation",
          participants: ["agent-1", "agent-2"],
          settings: { maxDuration: 30 },
        };

        const conversation =
          await conversationService.createConversation(config);

        expect(conversation).toMatchObject({
          title: "Test Conversation",
          participants: ["agent-1", "agent-2"],
          messages: [],
          status: "created",
          settings: { maxDuration: 30 },
        });
        expect(conversation.id).toMatch(/^conv-\d+$/);
      });

      it("creates conversation with defaults", async () => {
        const conversation = await conversationService.createConversation({});

        expect(conversation.title).toBe("Untitled Conversation");
        expect(conversation.participants).toEqual([]);
        expect(conversation.settings).toEqual({});
      });
    });

    describe("Message Management", () => {
      it("adds message to conversation", async () => {
        const conversation = await conversationService.createConversation({
          title: "Test Conversation",
        });

        const success = await conversationService.addMessage(conversation.id, {
          sender: "agent-1",
          content: "Hello world",
          type: "text",
        });

        expect(success).toBe(true);

        const updated = await conversationService.getConversation(
          conversation.id,
        );
        expect(updated?.messages).toHaveLength(1);
        expect(updated?.messages[0]).toMatchObject({
          sender: "agent-1",
          content: "Hello world",
          type: "text",
        });
        expect(updated?.messages[0].id).toMatch(/^msg-\d+$/);
      });

      it("rejects message for non-existent conversation", async () => {
        const success = await conversationService.addMessage("non-existent", {
          sender: "agent-1",
          content: "Hello",
        });

        expect(success).toBe(false);
      });

      it("updates lastActivity when message is added", async () => {
        const conversation = await conversationService.createConversation({});
        const originalActivity = conversation.lastActivity;

        // Wait a bit to ensure timestamp difference
        await new Promise((resolve) => setTimeout(resolve, 10));

        await conversationService.addMessage(conversation.id, {
          sender: "agent-1",
          content: "Test message",
        });

        const updated = await conversationService.getConversation(
          conversation.id,
        );
        expect(updated?.lastActivity.getTime()).toBeGreaterThan(
          originalActivity.getTime(),
        );
      });
    });

    describe("Conversation Control", () => {
      it("starts conversation", async () => {
        const conversation = await conversationService.createConversation({});

        const success = await conversationService.startConversation(
          conversation.id,
        );
        expect(success).toBe(true);

        const updated = await conversationService.getConversation(
          conversation.id,
        );
        expect(updated?.status).toBe("active");
        expect(updated?.startedAt).toBeInstanceOf(Date);
      });

      it("pauses conversation", async () => {
        const conversation = await conversationService.createConversation({});
        await conversationService.startConversation(conversation.id);

        const success = await conversationService.pauseConversation(
          conversation.id,
        );
        expect(success).toBe(true);

        const updated = await conversationService.getConversation(
          conversation.id,
        );
        expect(updated?.status).toBe("paused");
      });

      it("stops conversation", async () => {
        const conversation = await conversationService.createConversation({});
        await conversationService.startConversation(conversation.id);

        const success = await conversationService.stopConversation(
          conversation.id,
        );
        expect(success).toBe(true);

        const updated = await conversationService.getConversation(
          conversation.id,
        );
        expect(updated?.status).toBe("stopped");
        expect(updated?.endedAt).toBeInstanceOf(Date);
      });

      it("returns false for non-existent conversation operations", async () => {
        expect(
          await conversationService.startConversation("non-existent"),
        ).toBe(false);
        expect(
          await conversationService.pauseConversation("non-existent"),
        ).toBe(false);
        expect(await conversationService.stopConversation("non-existent")).toBe(
          false,
        );
      });
    });

    describe("Conversation Listing", () => {
      it("lists conversations by recent activity", async () => {
        const conv1 = await conversationService.createConversation({
          title: "First",
        });
        await new Promise((resolve) => setTimeout(resolve, 10));
        const conv2 = await conversationService.createConversation({
          title: "Second",
        });

        const conversations = await conversationService.listConversations();

        expect(conversations).toHaveLength(2);
        expect(conversations[0].title).toBe("Second"); // More recent
        expect(conversations[1].title).toBe("First");
      });

      it("filters by status", async () => {
        const conv1 = await conversationService.createConversation({
          title: "Active",
        });
        const conv2 = await conversationService.createConversation({
          title: "Inactive",
        });

        await conversationService.startConversation(conv1.id);

        const activeConversations = await conversationService.listConversations(
          { status: "active" },
        );
        expect(activeConversations).toHaveLength(1);
        expect(activeConversations[0].title).toBe("Active");
      });

      it("filters by participant", async () => {
        await conversationService.createConversation({
          title: "Conv 1",
          participants: ["agent-1", "agent-2"],
        });
        await conversationService.createConversation({
          title: "Conv 2",
          participants: ["agent-2", "agent-3"],
        });

        const agent1Conversations = await conversationService.listConversations(
          {
            participant: "agent-1",
          },
        );

        expect(agent1Conversations).toHaveLength(1);
        expect(agent1Conversations[0].title).toBe("Conv 1");
      });
    });

    describe("Message Queue", () => {
      it("provides queue statistics", () => {
        const stats = conversationService.getQueueStats();

        expect(stats).toEqual({
          queueLength: 0,
          isProcessing: false,
        });
      });

      it("processes messages in queue", async () => {
        const conversation = await conversationService.createConversation({});
        await conversationService.startConversation(conversation.id);

        await conversationService.addMessage(conversation.id, {
          sender: "agent-1",
          content: "Test message",
        });

        // Allow processing to complete
        await new Promise((resolve) => setTimeout(resolve, 200));

        const updated = await conversationService.getConversation(
          conversation.id,
        );
        expect(updated?.lastProcessedMessage).toBeTruthy();
      });
    });
  });

  describe("KnowledgeService", () => {
    let knowledgeService: KnowledgeService;

    beforeEach(() => {
      knowledgeService = new KnowledgeService();
    });

    describe("Entry Management", () => {
      it("creates knowledge entry", async () => {
        const entry = {
          title: "Test Knowledge",
          content: "This is test content",
          tags: ["test", "sample"],
          metadata: { source: "manual" },
        };

        const created = await knowledgeService.createEntry(entry);

        expect(created).toMatchObject({
          title: "Test Knowledge",
          content: "This is test content",
          tags: ["test", "sample"],
          metadata: { source: "manual" },
        });
        expect(created.id).toMatch(/^knowledge-\d+$/);
        expect(created.createdAt).toBeInstanceOf(Date);
        expect(created.updatedAt).toBeInstanceOf(Date);
      });

      it("creates entry with minimal data", async () => {
        const entry = {
          title: "Minimal Entry",
          content: "Content only",
        };

        const created = await knowledgeService.createEntry(entry);

        expect(created.tags).toEqual([]);
        expect(created.metadata).toEqual({});
      });

      it("retrieves entry by ID", async () => {
        const created = await knowledgeService.createEntry({
          title: "Test Entry",
          content: "Test content",
        });

        const retrieved = await knowledgeService.getEntry(created.id);
        expect(retrieved).toEqual(created);
      });

      it("returns null for non-existent entry", async () => {
        const result = await knowledgeService.getEntry("non-existent");
        expect(result).toBeNull();
      });

      it("updates existing entry", async () => {
        const entry = await knowledgeService.createEntry({
          title: "Original Title",
          content: "Original content",
        });

        const updated = await knowledgeService.updateEntry(entry.id, {
          title: "Updated Title",
          tags: ["updated"],
        });

        expect(updated?.title).toBe("Updated Title");
        expect(updated?.content).toBe("Original content"); // Unchanged
        expect(updated?.tags).toEqual(["updated"]);
        expect(updated?.updatedAt.getTime()).toBeGreaterThan(
          entry.createdAt.getTime(),
        );
      });

      it("deletes entry", async () => {
        const entry = await knowledgeService.createEntry({
          title: "To Delete",
          content: "Will be deleted",
        });

        const success = await knowledgeService.deleteEntry(entry.id);
        expect(success).toBe(true);

        const retrieved = await knowledgeService.getEntry(entry.id);
        expect(retrieved).toBeNull();
      });
    });

    describe("Search Functionality", () => {
      beforeEach(async () => {
        await knowledgeService.createEntry({
          title: "JavaScript Basics",
          content: "Introduction to JavaScript programming language",
          tags: ["programming", "javascript"],
        });

        await knowledgeService.createEntry({
          title: "React Components",
          content: "Building user interfaces with React components",
          tags: ["programming", "react", "frontend"],
        });

        await knowledgeService.createEntry({
          title: "Database Design",
          content: "Principles of good database design and normalization",
          tags: ["database", "design"],
        });
      });

      it("searches by title content", async () => {
        const results = await knowledgeService.searchEntries("JavaScript");

        expect(results).toHaveLength(1);
        expect(results[0].title).toBe("JavaScript Basics");
      });

      it("searches by content", async () => {
        const results = await knowledgeService.searchEntries("user interfaces");

        expect(results).toHaveLength(1);
        expect(results[0].title).toBe("React Components");
      });

      it("searches by tags", async () => {
        const results = await knowledgeService.searchEntries("programming");

        expect(results).toHaveLength(2);
        expect(results.map((r) => r.title)).toContain("JavaScript Basics");
        expect(results.map((r) => r.title)).toContain("React Components");
      });

      it("returns all entries for empty query", async () => {
        const results = await knowledgeService.searchEntries("");
        expect(results).toHaveLength(3);
      });

      it("limits search results", async () => {
        const results = await knowledgeService.searchEntries("", { limit: 2 });
        expect(results).toHaveLength(2);
      });

      it("searches entries by tags", async () => {
        const results = await knowledgeService.getEntriesByTags(["react"]);

        expect(results).toHaveLength(1);
        expect(results[0].title).toBe("React Components");
      });

      it("finds entries with any of multiple tags", async () => {
        const results = await knowledgeService.getEntriesByTags([
          "javascript",
          "database",
        ]);

        expect(results).toHaveLength(2);
        expect(results.map((r) => r.title)).toContain("JavaScript Basics");
        expect(results.map((r) => r.title)).toContain("Database Design");
      });
    });

    describe("Import/Export", () => {
      beforeEach(async () => {
        await knowledgeService.createEntry({
          title: "Entry 1",
          content: "Content 1",
          tags: ["tag1"],
        });

        await knowledgeService.createEntry({
          title: "Entry 2",
          content: "Content 2",
          tags: ["tag2"],
        });
      });

      it("exports knowledge as JSON", async () => {
        const exported = await knowledgeService.exportKnowledge("json");
        const parsed = JSON.parse(exported);

        expect(parsed).toHaveLength(2);
        expect(parsed[0]).toMatchObject({
          title: expect.any(String),
          content: expect.any(String),
          tags: expect.any(Array),
        });
      });

      it("exports knowledge as CSV", async () => {
        const exported = await knowledgeService.exportKnowledge("csv");
        const lines = exported.split("\n");

        expect(lines[0]).toBe("id,title,content,tags,createdAt,updatedAt");
        expect(lines).toHaveLength(3); // Header + 2 entries
      });

      it("throws error for unsupported export format", async () => {
        await expect(
          knowledgeService.exportKnowledge("xml" as any),
        ).rejects.toThrow("Unsupported format: xml");
      });

      it("imports knowledge from JSON", async () => {
        const data = JSON.stringify([
          {
            title: "Imported Entry 1",
            content: "Imported content 1",
            tags: ["imported"],
          },
          {
            title: "Imported Entry 2",
            content: "Imported content 2",
            tags: ["imported"],
          },
        ]);

        const count = await knowledgeService.importKnowledge(data, "json");
        expect(count).toBe(2);

        const results = await knowledgeService.searchEntries("Imported");
        expect(results).toHaveLength(2);
      });

      it("skips invalid entries during import", async () => {
        const data = JSON.stringify([
          {
            title: "Valid Entry",
            content: "Valid content",
          },
          {
            title: "Invalid Entry",
            // Missing content
          },
          {
            // Missing title and content
            tags: ["invalid"],
          },
        ]);

        const count = await knowledgeService.importKnowledge(data, "json");
        expect(count).toBe(1); // Only valid entry imported
      });
    });

    describe("Statistics", () => {
      it("calculates knowledge statistics", async () => {
        await knowledgeService.createEntry({
          title: "Entry 1",
          content: "Content 1",
          tags: ["tag1", "tag2"],
        });

        await knowledgeService.createEntry({
          title: "Entry 2",
          content: "Content 2",
          tags: ["tag2", "tag3"],
        });

        const stats = knowledgeService.getStats();

        expect(stats).toEqual({
          totalEntries: 2,
          totalTags: 3, // tag1, tag2, tag3
          averageTagsPerEntry: 2,
          searchIndexSize: expect.any(Number),
        });
      });

      it("handles empty knowledge base", () => {
        const stats = knowledgeService.getStats();

        expect(stats).toEqual({
          totalEntries: 0,
          totalTags: 0,
          averageTagsPerEntry: 0,
          searchIndexSize: 0,
        });
      });
    });
  });

  describe("Service Integration", () => {
    it("integrates agent and conversation services", async () => {
      const agentService = new AgentService();
      const conversationService = new ConversationService();

      // Create agents
      const agent1 = await agentService.createAgent({ name: "Agent 1" });
      const agent2 = await agentService.createAgent({ name: "Agent 2" });

      // Create conversation
      const conversation = await conversationService.createConversation({
        title: "Agent Discussion",
        participants: [agent1.id, agent2.id],
      });

      // Add messages
      await conversationService.addMessage(conversation.id, {
        sender: agent1.id,
        content: "Hello from Agent 1",
      });

      await conversationService.addMessage(conversation.id, {
        sender: agent2.id,
        content: "Hello from Agent 2",
      });

      // Verify integration
      const finalConversation = await conversationService.getConversation(
        conversation.id,
      );
      expect(finalConversation?.messages).toHaveLength(2);
      expect(finalConversation?.participants).toContain(agent1.id);
      expect(finalConversation?.participants).toContain(agent2.id);
    });

    it("integrates conversation and knowledge services", async () => {
      const conversationService = new ConversationService();
      const knowledgeService = new KnowledgeService();

      // Create conversation
      const conversation = await conversationService.createConversation({
        title: "Knowledge Discussion",
      });

      // Add message with knowledge reference
      await conversationService.addMessage(conversation.id, {
        sender: "agent-1",
        content: "Let me share some knowledge about React components",
      });

      // Create related knowledge entry
      const knowledge = await knowledgeService.createEntry({
        title: "React Components from Conversation",
        content: "Knowledge extracted from agent conversation",
        tags: ["react", "conversation"],
        metadata: {
          sourceConversation: conversation.id,
          extractedFrom: "agent discussion",
        },
      });

      // Verify integration
      expect(knowledge.metadata.sourceConversation).toBe(conversation.id);

      const searchResults =
        await knowledgeService.searchEntries("conversation");
      expect(searchResults).toHaveLength(1);
      expect(searchResults[0].title).toContain("React Components");
    });
  });
});
