/**
 * Defines the conversation dynamics between agents in a multi-agent system.
 * This helps determine when and how agents should respond to messages.
 */

import type { Agent, Conversation, Message } from "@/lib/types"

export interface ConversationDynamicsOptions {
  /**
   * Probability [0-1] that an agent will respond to a message not directed at them
   * Lower values make agents less likely to jump into conversations
   */
  responseThreshold: number

  /**
   * Minimum delay in milliseconds before an agent responds
   */
  minResponseDelay: number

  /**
   * Maximum delay in milliseconds before an agent responds
   */
  maxResponseDelay: number

  /**
   * Whether all agents should respond to the first message in a conversation
   */
  allRespondToFirst: boolean

  /**
   * Whether to enable dynamic turn-taking between agents
   */
  enableTurnTaking: boolean

  /**
   * Maximum number of agents that should respond to any single message
   * Set to 0 for unlimited respondents
   */
  maxRespondentsPerMessage: number

  /**
   * How strongly to prioritize agents who haven't spoken recently
   * Higher values give more priority to agents who have been quiet
   * Range: 0-1
   */
  turnTakingStrength: number
}

// Default conversation dynamics options
export const defaultDynamicsOptions: ConversationDynamicsOptions = {
  responseThreshold: 0.7,
  minResponseDelay: 500,
  maxResponseDelay: 3000,
  allRespondToFirst: true,
  enableTurnTaking: true,
  maxRespondentsPerMessage: 2,
  turnTakingStrength: 0.3,
}

export class ConversationDynamics {
  private options: ConversationDynamicsOptions
  private lastSpeakingTurn: Record<string, number> = {}

  constructor(options: Partial<ConversationDynamicsOptions> = {}) {
    this.options = {
      ...defaultDynamicsOptions,
      ...options,
    }
  }

  /**
   * Updates the dynamics options
   */
  updateOptions(options: Partial<ConversationDynamicsOptions>): void {
    this.options = {
      ...this.options,
      ...options,
    }
  }

  /**
   * Determines which agents should respond to a message
   */
  determineRespondents(
    message: Message,
    agents: Agent[],
    conversation: Conversation,
    currentMessageIndex: number,
  ): string[] {
    // Skip if there are no agents
    if (!agents.length) return []

    // Record valid participant agents (in the conversation)
    const participantAgents = agents.filter(
      (agent) => conversation.participants.includes(agent.id) && agent.id !== message.senderId,
    )

    if (!participantAgents.length) return []

    // Check if this is directed to a specific agent with a mention
    // Format: "Agent X, [message]" or "@Agent X [message]"
    const mentionMatch = message.content.match(/^(?:@?(.+?),?\s+)/i)
    const directedToName = mentionMatch ? mentionMatch[1] : null

    // If directed to a specific agent by name
    if (directedToName) {
      const matchedAgents = participantAgents.filter(
        (agent) =>
          agent.name.toLowerCase() === directedToName.toLowerCase() ||
          agent.name.toLowerCase().startsWith(directedToName.toLowerCase()),
      )

      if (matchedAgents.length) {
        // Record the speaking turn for this agent
        matchedAgents.forEach((agent) => {
          this.lastSpeakingTurn[agent.id] = currentMessageIndex
        })
        return matchedAgents.map((agent) => agent.id)
      }
    }

    // If this is the first message in the conversation
    if (currentMessageIndex === 0 && this.options.allRespondToFirst) {
      // All agents respond to the first message with varying delays
      participantAgents.forEach((agent) => {
        this.lastSpeakingTurn[agent.id] = currentMessageIndex
      })
      return participantAgents.map((agent) => agent.id)
    }

    // For other messages, determine who responds based on dynamics
    let respondents: string[] = []

    // Calculate response probabilities based on turn-taking
    const responseProbabilities = this.calculateResponseProbabilities(
      participantAgents,
      conversation,
      currentMessageIndex,
    )

    // Select respondents based on probabilities and maximum
    for (const agent of participantAgents) {
      const probability = responseProbabilities[agent.id] || this.options.responseThreshold

      if (Math.random() < probability) {
        respondents.push(agent.id)
      }
    }

    // Limit to max respondents if specified
    if (this.options.maxRespondentsPerMessage > 0 && respondents.length > this.options.maxRespondentsPerMessage) {
      respondents = respondents.slice(0, this.options.maxRespondentsPerMessage)
    }

    // Record the speaking turn for these agents
    respondents.forEach((agentId) => {
      this.lastSpeakingTurn[agentId] = currentMessageIndex
    })

    return respondents
  }

  /**
   * Calculates the probability of each agent responding to a message
   * based on turn-taking dynamics
   */
  private calculateResponseProbabilities(
    agents: Agent[],
    conversation: Conversation,
    currentMessageIndex: number,
  ): Record<string, number> {
    const probabilities: Record<string, number> = {}

    if (!this.options.enableTurnTaking) {
      // If turn-taking is disabled, use flat probabilities
      agents.forEach((agent) => {
        probabilities[agent.id] = this.options.responseThreshold
      })
      return probabilities
    }

    // Get the last time each agent spoke
    const lastSpokeTurnsAgo: Record<string, number> = {}

    agents.forEach((agent) => {
      // If the agent has never spoken, use a large number
      const lastTurn = this.lastSpeakingTurn[agent.id] ?? -100
      lastSpokeTurnsAgo[agent.id] = currentMessageIndex - lastTurn
    })

    // Find the agent who's been quiet the longest and shortest
    const maxTurnsSilent = Math.max(...Object.values(lastSpokeTurnsAgo))
    const minTurnsSilent = Math.min(...Object.values(lastSpokeTurnsAgo))
    const range = Math.max(1, maxTurnsSilent - minTurnsSilent)

    // Calculate probabilities based on how long they've been quiet
    agents.forEach((agent) => {
      const turnsSilent = lastSpokeTurnsAgo[agent.id]

      // Normalize to 0-1 range
      const normalizedSilence = (turnsSilent - minTurnsSilent) / range

      // Apply turn-taking weight
      // Higher values of turnTakingStrength make agents who haven't spoken more likely to speak
      const baseThreshold = this.options.responseThreshold
      const turnBonus = normalizedSilence * this.options.turnTakingStrength

      probabilities[agent.id] = Math.min(0.95, baseThreshold + turnBonus)
    })

    return probabilities
  }

  /**
   * Calculates the response delay for an agent
   */
  calculateResponseDelay(agentId: string, message: Message): number {
    // Calculate a delay between min and max
    const baseDelay =
      Math.random() * (this.options.maxResponseDelay - this.options.minResponseDelay) + this.options.minResponseDelay

    // You could add agent-specific factors here
    // For example, some agents might respond faster than others

    // Consider message length - longer messages take longer to read
    const messageLength = message.content.length
    const lengthFactor = Math.min(1, messageLength / 200) // Cap at 200 chars
    const lengthDelay = lengthFactor * 500 // Up to 500ms extra for long messages

    return Math.round(baseDelay + lengthDelay)
  }

  /**
   * Records that an agent has spoken
   */
  recordAgentResponse(agentId: string, messageIndex: number): void {
    this.lastSpeakingTurn[agentId] = messageIndex
  }

  /**
   * Resets the conversation dynamics
   */
  reset(): void {
    this.lastSpeakingTurn = {}
  }
}
