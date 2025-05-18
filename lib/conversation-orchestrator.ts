import type { Agent, Conversation, Message } from "@/lib/types"
import type { LLMClient } from "@/lib/llm-client"
import type { KnowledgeRetriever } from "@/lib/knowledge-retriever"
import { ConversationLogger } from "@/lib/conversation-logger"
import type { KnowledgeEntry } from "@/lib/types"

export interface ResponseOptions {
  maxKnowledgeEntries?: number
  includeAgentKnowledge?: boolean
  includeTags?: string[]
  streamResponse?: boolean
  onChunk?: (text: string, isComplete: boolean) => void
  onMessageGenerated?: (message: Message) => void // Callback for when a message is generated
  messageToRespondTo?: Message // Specific message to respond to
  responseDelay?: number // Override the default response delay
  force?: boolean // Force the agent to respond regardless of dynamics
}

export interface OrchestratorOptions {
  maxConcurrentResponses?: number
  responseDelay?: number | [number, number] // Fixed delay or [min, max] range
  autoSelectRespondents?: boolean
  onMessageGenerated?: (message: Message) => void // Global callback for when a message is generated
  onError?: (error: Error) => void // Callback for errors
}

// Interface for queued response
interface QueuedResponse {
  agentId: string
  messageId: string // ID of the message to respond to
  options: ResponseOptions
}

export class ConversationOrchestrator {
  private conversation: Conversation
  private agents: Map<string, Agent>
  private llmClient: LLMClient
  private knowledgeRetriever: KnowledgeRetriever
  private options: OrchestratorOptions
  private responseQueue: QueuedResponse[] = []
  private processingAgents: Set<string> = new Set()
  private lastProcessedMessageId: string | null = null
  private messageCache: Map<string, Message> = new Map() // Cache messages to handle race conditions
  private queueProcessorInterval: NodeJS.Timeout | null = null
  private isProcessingQueue = false
  private logger: ReturnType<typeof ConversationLogger.init>

  constructor(
    conversation: Conversation,
    agents: Agent[],
    llmClient: LLMClient,
    knowledgeRetriever: KnowledgeRetriever,
    options: OrchestratorOptions = {},
  ) {
    this.logger = ConversationLogger.init(conversation.id)
    // /* USEFUL FOR PRODUCTION TESTING: Log orchestrator initialization */
    // this.logger.log("INIT", "ConversationOrchestrator constructor called", {
    //   conversationId: conversation.id,
    //   agentsCount: agents.length,
    //   llmClientAvailable: !!llmClient,
    // })

    this.conversation = conversation
    this.agents = new Map(agents.map((agent) => [agent.id, agent]))
    this.llmClient = llmClient
    this.knowledgeRetriever = knowledgeRetriever
    this.options = {
      maxConcurrentResponses: 1,
      responseDelay: [500, 2000], // Random delay between 500ms and 2000ms
      autoSelectRespondents: true,
      ...options,
    }

    // Initialize lastProcessedMessageId if there are messages
    if (conversation.messages.length > 0) {
      this.lastProcessedMessageId = conversation.messages[conversation.messages.length - 1].id
      // /* USEFUL FOR PRODUCTION TESTING: Log last processed message ID */
      // this.logger.log("INIT", "Set last processed message ID", {
      //   messageId: this.lastProcessedMessageId,
      // })
    }

    // Initialize message cache with current messages
    this.updateMessageCache(conversation.messages)
    // /* USEFUL FOR PRODUCTION TESTING: Log message cache initialization */
    // this.logger.log("INIT", "Initialized message cache with existing messages", {
    //   messageCount: conversation.messages.length,
    // })

    // Start queue processor
    this.startQueueProcessor()
    // /* USEFUL FOR PRODUCTION TESTING: Log queue processor start */
    // this.logger.log("INIT", "Started queue processor")
  }

  /**
   * Starts the queue processor interval
   */
  private startQueueProcessor(): void {
    // Clear any existing interval
    if (this.queueProcessorInterval) {
      clearInterval(this.queueProcessorInterval)
    }

    // Process the queue every 100ms
    this.queueProcessorInterval = setInterval(() => {
      this.processQueue()
    }, 100)
  }

  /**
   * Updates the message cache with new messages
   */
  private updateMessageCache(messages: Message[]): void {
    messages.forEach((message) => {
      this.messageCache.set(message.id, message)
    })
  }

  /**
   * Updates the conversation reference
   */
  updateConversation(conversation: Conversation): void {
    // /* USEFUL FOR PRODUCTION TESTING: Log conversation update */
    // this.logger.log("UPDATE", "Updating conversation reference", {
    //   oldMessageCount: this.conversation.messages.length,
    //   newMessageCount: conversation.messages.length,
    // })

    // Update message cache with any new messages
    this.updateMessageCache(conversation.messages)

    this.conversation = conversation

    // Update lastProcessedMessageId if there are new messages
    if (conversation.messages.length > 0) {
      const latestMessageId = conversation.messages[conversation.messages.length - 1].id
      if (this.lastProcessedMessageId !== latestMessageId) {
        // Only update if we're not currently processing this message
        const isBeingProcessed =
          this.responseQueue.some((item) => item.messageId === latestMessageId) ||
          Array.from(this.processingAgents).some((agentId) =>
            this.responseQueue.some((item) => item.agentId === agentId && item.messageId === latestMessageId),
          )

        if (!isBeingProcessed) {
          this.lastProcessedMessageId = latestMessageId
          // /* USEFUL FOR PRODUCTION TESTING: Log last processed message ID update */
          // this.logger.log("UPDATE", "Updated last processed message ID", {
          //   messageId: this.lastProcessedMessageId,
          // })
        }
      }
    }
  }

  /**
   * Updates the agents map
   */
  updateAgents(agents: Agent[]): void {
    // /* USEFUL FOR PRODUCTION TESTING: Log agents update */
    // this.logger.log("UPDATE", "Updating agents", {
    //   oldAgentCount: this.agents.size,
    //   newAgentCount: agents.length,
    // })
    this.agents = new Map(agents.map((agent) => [agent.id, agent]))
  }

  /**
   * Determines which agents should respond to a message
   */
  determineRespondents(message: Message): string[] {
    // If not auto-selecting, return empty array (manual selection)
    if (!this.options.autoSelectRespondents) {
      return []
    }

    // Get all agents in the conversation
    const conversationAgents = this.conversation.participants
      .map((id) => this.agents.get(id))
      .filter((agent): agent is Agent => agent !== undefined)

    // Skip the agent who sent the message
    const eligibleAgents = conversationAgents.filter((agent) => agent.id !== message.senderId)

    if (eligibleAgents.length === 0) {
      return []
    }

    // Check if this is an autonomous conversation
    const isAutonomousConversation = this.conversation.isAutonomous === true

    if (isAutonomousConversation) {
      // Count non-system messages
      const messageCount = this.conversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length

      // Get the LLM client settings
      const settings = this.llmClient.getSettings()
      const maxMessages = settings.maxAutonomousMessages || 10

      // If we've reached the maximum, end the conversation by returning no respondents
      if (messageCount >= maxMessages) {
        this.logger.log(
          "RESPONDENTS",
          `Maximum conversation depth reached (${messageCount}/${maxMessages}), no more responses`,
        )
        return []
      }

      // For autonomous conversations, always have at least one agent respond
      // Choose the agent who hasn't spoken most recently
      if (eligibleAgents.length > 1) {
        // Find the agent who hasn't spoken in the longest time
        const agentLastSpokenMap = new Map<string, number>()

        // Initialize all agents as never having spoken
        eligibleAgents.forEach((agent) => {
          agentLastSpokenMap.set(agent.id, -1)
        })

        // Update with the last time each agent spoke
        for (let i = this.conversation.messages.length - 1; i >= 0; i--) {
          const msg = this.conversation.messages[i]
          if (agentLastSpokenMap.has(msg.senderId) && agentLastSpokenMap.get(msg.senderId) === -1) {
            agentLastSpokenMap.set(msg.senderId, i)
          }
        }

        // Sort agents by who spoke least recently
        const sortedAgents = [...agentLastSpokenMap.entries()].sort((a, b) => a[1] - b[1]).map((entry) => entry[0])

        // Return the agent who hasn't spoken in the longest time
        return [sortedAgents[0]]
      }

      return eligibleAgents.map((agent) => agent.id)
    }

    // CRITICAL FIX: Special handling for conversation starters
    if (message.metadata?.type === "conversation_starter") {
      this.logger.log("RESPONDENTS", "Determining respondents for conversation starter message", {
        eligibleAgents: eligibleAgents.map((a) => a.name),
      })

      // All eligible agents should respond to conversation starters
      return eligibleAgents.map((agent) => agent.id)
    }

    // UPDATED: Check for mentions anywhere in the message, not just at the beginning
    // First, check for the traditional format at the beginning: "Agent X, [message]" or "@Agent X [message]"
    const beginningMentionMatch = message.content.match(/^(?:@?(.+?),?\s+)/i)
    const directedToNameAtBeginning = beginningMentionMatch ? beginningMentionMatch[1] : null

    // Then, check for mentions anywhere in the message
    const mentionedAgents = new Set<string>()

    // If there's a mention at the beginning, add it
    if (directedToNameAtBeginning) {
      mentionedAgents.add(directedToNameAtBeginning.toLowerCase())
    }

    // Check for other mentions in the format "Agent X" or "@Agent X" throughout the message
    const allMentionsRegex = /\b@?([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\b/g
    const allMatches = [...message.content.matchAll(allMentionsRegex)]

    for (const match of allMatches) {
      const potentialName = match[1]
      // Check if this potential name matches any agent
      for (const agent of eligibleAgents) {
        if (
          agent.name.toLowerCase() === potentialName.toLowerCase() ||
          agent.name.toLowerCase().startsWith(potentialName.toLowerCase())
        ) {
          mentionedAgents.add(potentialName.toLowerCase())
          break
        }
      }
    }

    // If we found mentions, determine which agents should respond
    if (mentionedAgents.size > 0) {
      const matchedAgents = eligibleAgents.filter((agent) => {
        const agentNameLower = agent.name.toLowerCase()
        return Array.from(mentionedAgents).some(
          (mentionedName) => agentNameLower === mentionedName || agentNameLower.startsWith(mentionedName),
        )
      })

      if (matchedAgents.length) {
        // /* USEFUL FOR PRODUCTION TESTING: Log mentioned agents */
        // this.logger.log("RESPONDENTS", "Found mentioned agents", {
        //   mentions: Array.from(mentionedAgents),
        //   matchedAgents: matchedAgents.map((a) => a.name),
        // })
        return matchedAgents.map((agent) => agent.id)
      }
    }

    // For the first message, have all agents respond
    if (this.conversation.messages.length <= 1) {
      // /* USEFUL FOR PRODUCTION TESTING: Log first message response */
      // this.logger.log("RESPONDENTS", "First message in conversation, all agents will respond")
      return eligibleAgents.map((agent) => agent.id)
    }

    // For other messages, have a random subset respond
    const respondents = eligibleAgents
      .filter(() => Math.random() > 0.3) // 70% chance each agent responds
      .map((agent) => agent.id)

    // /* USEFUL FOR PRODUCTION TESTING: Log random respondents */
    // this.logger.log("RESPONDENTS", "Selected random subset of agents to respond", {
    //   respondentCount: respondents.length,
    //   respondents: respondents.map((id) => this.agents.get(id)?.name || id),
    // })

    return respondents
  }

  /**
   * Queues an agent to generate a response to a specific message
   */
  queueAgentResponse(agentId: string, options: ResponseOptions = {}): void {
    const agentLogger = ConversationLogger.agent(agentId)

    // Verify agent exists and is in the conversation
    const agent = this.agents.get(agentId)
    if (!agent || !this.conversation.participants.includes(agentId)) {
      agentLogger.error("QUEUE", `Agent ${agentId} is not valid or not in the conversation`)
      return
    }

    // Check if there are any messages in the conversation
    if (this.conversation.messages.length === 0) {
      agentLogger.error("QUEUE", `Cannot queue agent response: No messages in conversation for agent ${agentId}`)
      return
    }

    // Determine which message to respond to
    const messageToRespondTo =
      options.messageToRespondTo || this.conversation.messages[this.conversation.messages.length - 1]

    // Store the message in the cache to ensure we can access it later
    this.messageCache.set(messageToRespondTo.id, messageToRespondTo)

    // Check if this agent is already responding to this message
    const isAlreadyQueued = this.responseQueue.some(
      (item) => item.agentId === agentId && item.messageId === messageToRespondTo.id,
    )

    const isCurrentlyProcessing = this.processingAgents.has(agentId)

    if (isAlreadyQueued) {
      agentLogger.warn(
        "QUEUE",
        `Agent ${agentId} (${agent.name}) is already queued to respond to message ${messageToRespondTo.id}`,
      )
      return
    }

    if (isCurrentlyProcessing) {
      agentLogger.warn("QUEUE", `Agent ${agentId} (${agent.name}) is currently processing another response`)
      return
    }

    // Add to queue
    this.responseQueue.push({
      agentId,
      messageId: messageToRespondTo.id,
      options,
    })

    agentLogger.log("QUEUE", `Queued agent ${agentId} (${agent.name}) to respond to message ${messageToRespondTo.id}`, {
      isStarterMessage: messageToRespondTo.metadata?.type === "conversation_starter",
      force: options.force,
    })

    // Immediately try to process the queue
    setTimeout(() => this.processQueue(), 0)
  }

  /**
   * Checks if a conversation should end based on message count
   */
  shouldEndConversation(conversation: Conversation): boolean {
    // Only apply automatic ending to autonomous conversations
    if (!conversation || !conversation.isAutonomous) return false

    // Count non-system messages
    const messageCount = conversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length

    // Check if the conversation has reached the maximum message count
    return messageCount >= this.options.maxAutonomousMessages
  }

  /**
   * Processes the response queue
   */
  private processQueue(): void {
    // Prevent concurrent processing
    if (this.isProcessingQueue) {
      return
    }

    // If we're already at max concurrent responses, wait
    if (this.processingAgents.size >= (this.options.maxConcurrentResponses || 1)) {
      // /* USEFUL FOR PRODUCTION TESTING: Log queue processing pause */
      // this.logger.debug(
      //   "QUEUE",
      //   `Queue processing paused: ${this.processingAgents.size}/${this.options.maxConcurrentResponses} agents already processing`,
      // )
      return
    }

    // If there are no items in the queue, nothing to do
    if (this.responseQueue.length === 0) {
      return
    }

    // Set processing flag
    this.isProcessingQueue = true

    try {
      // Get the next item from the queue
      const nextItem = this.responseQueue.shift()
      if (!nextItem) {
        this.isProcessingQueue = false
        return
      }

      const { agentId, messageId, options } = nextItem

      // Mark agent as processing
      this.processingAgents.add(agentId)

      // Get the message to respond to from cache
      const messageToRespondTo = this.messageCache.get(messageId)
      if (!messageToRespondTo) {
        this.logger.error("PROCESS", `Message ${messageId} not found in cache`)
        this.processingAgents.delete(agentId)
        this.isProcessingQueue = false
        return
      }

      // Get the agent
      const agent = this.agents.get(agentId)
      if (!agent) {
        this.logger.error("PROCESS", `Agent ${agentId} not found`)
        this.processingAgents.delete(agentId)
        this.isProcessingQueue = false
        return
      }

      // Calculate response delay
      let responseDelay = options.responseDelay
      if (responseDelay === undefined) {
        if (Array.isArray(this.options.responseDelay)) {
          const [min, max] = this.options.responseDelay
          responseDelay = Math.floor(Math.random() * (max - min + 1)) + min
        } else {
          responseDelay = this.options.responseDelay || 0
        }
      }

      // Process the response after the delay
      setTimeout(() => {
        this.generateAgentResponse(agentId, messageToRespondTo, options)
          .catch((error) => {
            this.logger.error("PROCESS", `Error generating response for agent ${agentId}:`, error)
            if (this.options.onError) {
              this.options.onError(error)
            }
          })
          .finally(() => {
            // Mark agent as no longer processing
            this.processingAgents.delete(agentId)
          })
      }, responseDelay)
    } finally {
      // Reset processing flag
      this.isProcessingQueue = false
    }
  }

  /**
   * Generates a response from an agent to a specific message
   */
  private async generateAgentResponse(
    agentId: string,
    messageToRespondTo: Message,
    options: ResponseOptions = {},
  ): Promise<void> {
    const agent = this.agents.get(agentId)
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`)
    }

    const agentLogger = ConversationLogger.agent(agentId)
    agentLogger.log(
      "GENERATE",
      `Generating response for agent ${agentId} (${agent.name}) to message ${messageToRespondTo.id}`,
      { messageContent: messageToRespondTo.content.substring(0, 50) + "..." },
    )

    try {
      // Get conversation history for context
      const conversationHistory = this.conversation.messages.slice(-10) // Last 10 messages for context
      agentLogger.log("GENERATE", `Using ${conversationHistory.length} messages for context`)

      // Get relevant knowledge if requested
      let relevantKnowledge: KnowledgeEntry[] = []
      if (options.includeAgentKnowledge !== false) {
        // Get agent's knowledge
        relevantKnowledge = agent.knowledge
        agentLogger.log("GENERATE", `Agent has ${agent.knowledge.length} knowledge entries`)

        // If we have a knowledge retriever, use it to find relevant knowledge
        if (this.knowledgeRetriever && messageToRespondTo.content) {
          agentLogger.log("GENERATE", "Using knowledge retriever to find relevant knowledge")
          try {
            const retrievalResult = this.knowledgeRetriever.retrieveRelevant(
              messageToRespondTo.content,
              agent.knowledge,
              {
                maxResults: options.maxKnowledgeEntries || 3,
                includeTags: options.includeTags,
              },
            )

            if (retrievalResult.entries.length > 0) {
              relevantKnowledge = retrievalResult.entries
              agentLogger.log("GENERATE", `Found ${retrievalResult.entries.length} relevant knowledge entries`)
            } else {
              agentLogger.log("GENERATE", "No relevant knowledge entries found")
            }
          } catch (retrievalError) {
            agentLogger.error("GENERATE", "Error retrieving relevant knowledge:", retrievalError)
          }
        }
      }

      // Create a system prompt for the agent
      const systemPrompt = `You are ${agent.name}, with the following biography: ${agent.biography}

You are participating in a multi-agent conversation with other AI agents.
Your responses should be consistent with your character's knowledge, personality, and background.
You should respond naturally as if you are having a conversation with multiple participants.

IMPORTANT: Always start your response with "${agent.name}:" followed by your message.

${
  messageToRespondTo.metadata?.type === "conversation_starter"
    ? `IMPORTANT: This is the start of a new conversation. You should respond enthusiastically and engage with the topic.
Ask questions and show interest in what the other agent has said.`
    : ""
}

${
  messageToRespondTo.metadata?.type === "conversation_prompt"
    ? `IMPORTANT: You've been directly asked to respond. Please provide a thoughtful and engaging response.`
    : ""
}

When a message is clearly directed at another agent (e.g., addressed by name), you should:
1. Only respond if you have something valuable to add
2. Acknowledge that the message was primarily for another agent
3. Keep your response brief and relevant

When a message is directed at you specifically, provide a complete and helpful response.
When a message is directed at everyone or no one specific, respond naturally.

${relevantKnowledge.length > 0 ? "You have access to the following knowledge:" : "You have no specific knowledge on this topic."}`

      // Create a user prompt with conversation history and the message to respond to
      const userPrompt = `${relevantKnowledge.length > 0 ? "YOUR KNOWLEDGE:\n" + relevantKnowledge.map((k) => `- ${k.title}: ${k.content}`).join("\n") + "\n\n" : ""}CONVERSATION HISTORY:
${conversationHistory
  .map((msg) => {
    const senderName = msg.senderId === "user" ? "User" : this.agents.get(msg.senderId)?.name || "Unknown Agent"
    return `${senderName}: ${msg.content}`
  })
  .join("\n")}

Based on the conversation history and your knowledge, provide a response as ${agent.name}.
Your response should be a single message in a conversational tone.
Remember to start your response with "${agent.name}:" followed by your message.
If the message was clearly directed at another agent and you don't have anything valuable to add, respond with "SKIP_RESPONSE" and I will not include your message.`

      agentLogger.log("GENERATE", "Prepared prompts for LLM", {
        systemPromptLength: systemPrompt.length,
        userPromptLength: userPrompt.length,
      })

      // Check if LLM client is available
      if (!this.llmClient) {
        throw new Error("LLM client is not available")
      }

      // Generate the response using the LLM client
      let response: string
      agentLogger.log("GENERATE", "Calling LLM client to generate response")

      if (options.streamResponse && options.onChunk) {
        // Use streaming if requested
        agentLogger.log("GENERATE", "Using streaming response generation")
        try {
          response = await this.llmClient.streamResponse(systemPrompt, userPrompt, options.onChunk)
          agentLogger.log("GENERATE", "Streaming response completed", { responseLength: response.length })
        } catch (streamError) {
          agentLogger.error("GENERATE", "Error in streaming response generation:", streamError)
          throw streamError
        }
      } else {
        // Otherwise use regular generation
        agentLogger.log("GENERATE", "Using regular response generation")
        try {
          response = await this.llmClient.generateResponse(systemPrompt, userPrompt)
          agentLogger.log("GENERATE", "Regular response completed", { responseLength: response.length })
        } catch (genError) {
          agentLogger.error("GENERATE", "Error in regular response generation:", genError)
          throw genError
        }
      }

      // Skip empty responses or SKIP_RESPONSE
      if (!response.trim() || response.includes("SKIP_RESPONSE")) {
        agentLogger.log("GENERATE", `Agent ${agentId} (${agent.name}) decided to skip responding`)
        return
      }

      // CRITICAL FIX: Ensure agent name is prepended to the response if not already present
      let processedResponse = response.trim()
      const expectedPrefix = `${agent.name}:`

      // Check if the response already starts with the agent name
      if (!processedResponse.startsWith(expectedPrefix)) {
        // If it doesn't, add the prefix
        processedResponse = `${expectedPrefix} ${processedResponse}`
        agentLogger.log("GENERATE", `Added agent name prefix to response: ${expectedPrefix}`)
      }

      // Create the message
      const message: Message = {
        id: `msg-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        content: processedResponse,
        senderId: agentId,
        timestamp: new Date(),
        metadata: {
          isGeneratedByLLM: true,
          respondingTo: messageToRespondTo.id,
        },
      }

      agentLogger.log("GENERATE", "Created message from response", { messageId: message.id })

      // Call the onMessageGenerated callback
      if (options.onMessageGenerated) {
        agentLogger.log("GENERATE", "Calling options.onMessageGenerated callback")
        options.onMessageGenerated(message)
      } else if (this.options.onMessageGenerated) {
        agentLogger.log("GENERATE", "Calling this.options.onMessageGenerated callback")
        this.options.onMessageGenerated(message)
      } else {
        agentLogger.warn("GENERATE", "No onMessageGenerated callback available")
      }

      agentLogger.log("GENERATE", `Generated response for agent ${agentId} (${agent.name}):`, {
        messageId: message.id,
        contentPreview: message.content.substring(0, 50) + (message.content.length > 50 ? "..." : ""),
      })
    } catch (error) {
      agentLogger.error("GENERATE", `Error generating response for agent ${agentId} (${agent.name}):`, error)
      throw error
    }
  }

  /**
   * Processes a new message in the conversation
   */
  processNewMessage(message: Message): void {
    this.logger.log("PROCESS", `Processing new message ${message.id} from ${message.senderId}`)

    // Add to message cache
    this.messageCache.set(message.id, message)

    // Update last processed message ID
    this.lastProcessedMessageId = message.id

    // Determine which agents should respond
    const respondentIds = this.determineRespondents(message)

    this.logger.log("PROCESS", `Determined respondents for message ${message.id}:`, {
      respondentCount: respondentIds.length,
      respondents: respondentIds.map((id) => this.agents.get(id)?.name || id),
    })

    // Queue responses for each respondent
    respondentIds.forEach((agentId) => {
      this.queueAgentResponse(agentId, {
        messageToRespondTo: message,
      })
    })
  }

  /**
   * Cancels all pending responses
   */
  cancelAllResponses(): void {
    this.logger.log("CANCEL", "Cancelling all pending responses")

    // Clear the queue
    this.responseQueue = []

    // Clear processing agents
    this.processingAgents.clear()
  }

  /**
   * Cleans up resources when the orchestrator is no longer needed
   */
  cleanup(): void {
    this.logger.log("CLEANUP", "Cleaning up conversation orchestrator")

    // Clear the queue processor interval
    if (this.queueProcessorInterval) {
      clearInterval(this.queueProcessorInterval)
      this.queueProcessorInterval = null
    }

    // Cancel all responses
    this.cancelAllResponses()
  }

  /**
   * Returns the list of agents currently processing responses
   */
  getProcessingAgents(): string[] {
    return Array.from(this.processingAgents)
  }

  /**
   * Returns the list of agents queued to respond
   */
  getQueuedAgents(): string[] {
    return this.responseQueue.map((item) => item.agentId)
  }

  /**
   * Returns the list of message IDs currently being processed
   */
  getProcessingMessageIds(): string[] {
    return Array.from(
      new Set([
        ...this.responseQueue.map((item) => item.messageId),
        ...Array.from(this.processingAgents)
          .map((agentId) => {
            const queueItem = this.responseQueue.find((item) => item.agentId === agentId)
            return queueItem ? queueItem.messageId : ""
          })
          .filter((id) => id !== ""),
      ]),
    )
  }
}
