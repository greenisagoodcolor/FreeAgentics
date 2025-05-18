import type { Agent, Conversation, Message, Position, KnowledgeEntry } from "@/lib/types"
import { KnowledgeRetriever } from "@/lib/knowledge-retriever"

// Types for autonomous conversation triggers
export type TriggerType = "proximity" | "knowledge_overlap" | "scheduled" | "user_initiated" | "continuation"

export interface TriggerContext {
  agents: Agent[]
  location?: Position
  knowledgeEntries?: KnowledgeEntry[]
  previousConversation?: Conversation
  topic?: string
}

export interface AutonomousConversationOptions {
  // Minimum distance for proximity-based conversations
  proximityThreshold: number

  // Minimum knowledge overlap score to trigger a conversation
  knowledgeOverlapThreshold: number

  // Maximum number of autonomous conversations that can happen simultaneously
  maxSimultaneousConversations: number

  // Cooldown period (in ms) before an agent can participate in another autonomous conversation
  conversationCooldown: number

  // Maximum duration (in ms) for an autonomous conversation before it's automatically ended
  maxConversationDuration: number

  // Whether to enable scheduled conversations
  enableScheduledConversations: boolean

  // Whether to enable knowledge-based triggers
  enableKnowledgeBasedTriggers: boolean

  // Maximum messages before forced ending
  maxAutonomousMessages: number
}

// Default options
export const defaultAutonomousOptions: AutonomousConversationOptions = {
  proximityThreshold: 1, // Adjacent cells
  knowledgeOverlapThreshold: 0.3, // 30% overlap
  maxSimultaneousConversations: 1,
  conversationCooldown: 5000, // Reduced from 60000 (1 minute) to 5000 (5 seconds) for testing
  maxConversationDuration: 300000, // 5 minutes
  enableScheduledConversations: false,
  enableKnowledgeBasedTriggers: true,
  maxAutonomousMessages: 4, // Default maximum of 4 messages
}

export class AutonomousConversationSystem {
  private options: AutonomousConversationOptions
  private knowledgeRetriever: KnowledgeRetriever
  private activeConversations: Set<string> = new Set()
  private agentLastConversationTime: Map<string, number> = new Map()
  private conversationTimeouts: Map<string, NodeJS.Timeout> = new Map()

  constructor(options: Partial<AutonomousConversationOptions> = {}) {
    this.options = { ...defaultAutonomousOptions, ...options }
    this.knowledgeRetriever = new KnowledgeRetriever()
  }

  /**
   * Updates the system options
   */
  updateOptions(options: Partial<AutonomousConversationOptions>): void {
    this.options = { ...this.options, ...options }
  }

  /**
   * Resets the cooldown for specific agents
   */
  resetCooldown(agentIds: string[]): void {
    // /* USEFUL FOR PRODUCTION TESTING: Log cooldown reset */
    // console.log(`Resetting cooldown for agents: ${agentIds.join(", ")}`)
    agentIds.forEach((id) => {
      this.agentLastConversationTime.delete(id)
    })
  }

  /**
   * Checks if an autonomous conversation should be triggered based on agent proximity
   */
  checkProximityTrigger(agents: Agent[]): { shouldTrigger: boolean; participants: Agent[] } {
    // Filter agents that have autonomy enabled
    const autonomousAgents = agents.filter((agent) => agent.autonomyEnabled)

    // /* USEFUL FOR PRODUCTION TESTING: Log proximity check */
    // console.log(
    //   "Checking proximity trigger with autonomous agents:",
    //   autonomousAgents.map(
    //     (a) =>
    //       `${a.name} (autonomy: ${a.autonomyEnabled ? "enabled" : "disabled"}, inConversation: ${a.inConversation ? "yes" : "no"}, position: ${a.position.x},${a.position.y})`,
    //   ),
    // )

    // We need at least 2 autonomous agents to have a conversation
    if (autonomousAgents.length < 2) {
      // console.log("Not enough autonomous agents for proximity trigger")
      return { shouldTrigger: false, participants: [] }
    }

    // Check for agents that are close to each other
    const agentGroups: Agent[][] = []

    // For each agent, find other agents within proximity threshold
    for (const agent of autonomousAgents) {
      // Skip agents already in a conversation
      if (agent.inConversation) {
        // console.log(`Agent ${agent.name} is already in a conversation, skipping`)
        continue
      }

      // Skip agents on cooldown
      const lastConversationTime = this.agentLastConversationTime.get(agent.id) || 0
      const timeSinceLastConversation = Date.now() - lastConversationTime
      const isOnCooldown = timeSinceLastConversation < this.options.conversationCooldown

      if (isOnCooldown) {
        // /* USEFUL FOR PRODUCTION TESTING: Log cooldown status */
        // console.log(
        //   `Agent ${agent.name} is on cooldown (${Math.round(timeSinceLastConversation / 1000)}s elapsed, cooldown: ${Math.round(this.options.conversationCooldown / 1000)}s), skipping`,
        // )
        continue
      }

      const nearbyAgents = autonomousAgents.filter((otherAgent) => {
        // Skip self, agents in conversation, and agents in conversation
        if (otherAgent.id === agent.id || otherAgent.inConversation) {
          return false
        }

        const otherLastConversationTime = this.agentLastConversationTime.get(otherAgent.id) || 0
        const otherTimeSinceLastConversation = Date.now() - otherLastConversationTime
        const otherIsOnCooldown = otherTimeSinceLastConversation < this.options.conversationCooldown

        if (otherIsOnCooldown) {
          // /* USEFUL FOR PRODUCTION TESTING: Log nearby agent cooldown */
          // console.log(
          //   `Nearby agent ${otherAgent.name} is on cooldown (${Math.round(otherTimeSinceLastConversation / 1000)}s elapsed), skipping`,
          // )
          return false
        }

        // Check if within proximity threshold
        const distance = Math.sqrt(
          Math.pow(agent.position.x - otherAgent.position.x, 2) + Math.pow(agent.position.y - otherAgent.position.y, 2),
        )

        const isNearby = distance <= this.options.proximityThreshold
        if (isNearby) {
          // /* USEFUL FOR PRODUCTION TESTING: Log nearby agent detection */
          // console.log(
          //   `Agent ${agent.name} is near ${otherAgent.name} (distance: ${distance.toFixed(2)}, threshold: ${this.options.proximityThreshold})`,
          // )
        }

        return isNearby
      })

      // If we found nearby agents, create a group
      if (nearbyAgents.length > 0) {
        // /* USEFUL FOR PRODUCTION TESTING: Log nearby agents found */
        // console.log(
        //   `Found nearby agents for ${agent.name}:`,
        //   nearbyAgents.map((a) => a.name),
        // )
        agentGroups.push([agent, ...nearbyAgents])
      }
    }

    // If we found any groups, return the first one
    if (agentGroups.length > 0) {
      // Limit to a reasonable number of participants (2-3 is ideal for conversation)
      const participants = agentGroups[0].slice(0, 3)
      // /* USEFUL FOR PRODUCTION TESTING: Log proximity trigger success */
      // console.log(
      //   "Proximity trigger successful with participants:",
      //   participants.map((a) => a.name),
      // )
      return { shouldTrigger: true, participants }
    }

    return { shouldTrigger: false, participants: [] }
  }

  /**
   * Checks if an autonomous conversation should be triggered based on knowledge overlap
   */
  checkKnowledgeOverlapTrigger(agents: Agent[]): { shouldTrigger: boolean; participants: Agent[]; topic: string } {
    // Only proceed if knowledge-based triggers are enabled
    if (!this.options.enableKnowledgeBasedTriggers) {
      return { shouldTrigger: false, participants: [], topic: "" }
    }

    // Filter agents that have autonomy enabled
    const autonomousAgents = agents.filter((agent) => agent.autonomyEnabled)

    // We need at least 2 autonomous agents to have a conversation
    if (autonomousAgents.length < 2) {
      return { shouldTrigger: false, participants: [], topic: "" }
    }

    // Find agents with overlapping knowledge
    const agentPairs: { agents: [Agent, Agent]; overlapScore: number; topic: string }[] = []

    // Compare each pair of agents
    for (let i = 0; i < autonomousAgents.length; i++) {
      const agent1 = autonomousAgents[i]

      // Skip agents already in a conversation
      if (agent1.inConversation) continue

      // Skip agents on cooldown
      const lastConversationTime1 = this.agentLastConversationTime.get(agent1.id) || 0
      if (Date.now() - lastConversationTime1 < this.options.conversationCooldown) continue

      for (let j = i + 1; j < autonomousAgents.length; j++) {
        const agent2 = autonomousAgents[j]

        // Skip agents already in a conversation
        if (agent2.inConversation) continue

        // Skip agents on cooldown
        const lastConversationTime2 = this.agentLastConversationTime.get(agent2.id) || 0
        if (Date.now() - lastConversationTime2 < this.options.conversationCooldown) continue

        // Calculate knowledge overlap
        const { overlapScore, commonTags } = this.calculateKnowledgeOverlap(agent1, agent2)

        if (overlapScore >= this.options.knowledgeOverlapThreshold && commonTags.length > 0) {
          // Choose a random common tag as the conversation topic
          const topic = commonTags[Math.floor(Math.random() * commonTags.length)]

          agentPairs.push({
            agents: [agent1, agent2],
            overlapScore,
            topic,
          })
        }
      }
    }

    // Sort by overlap score (highest first)
    agentPairs.sort((a, b) => b.overlapScore - a.overlapScore)

    // If we found any pairs with sufficient overlap, return the highest scoring pair
    if (agentPairs.length > 0) {
      const { agents, topic } = agentPairs[0]
      return { shouldTrigger: true, participants: agents, topic }
    }

    return { shouldTrigger: false, participants: [], topic: "" }
  }

  /**
   * Calculates knowledge overlap between two agents
   */
  private calculateKnowledgeOverlap(agent1: Agent, agent2: Agent): { overlapScore: number; commonTags: string[] } {
    // Extract all tags from both agents' knowledge
    const tags1 = new Set<string>()
    const tags2 = new Set<string>()

    agent1.knowledge.forEach((entry) => {
      entry.tags.forEach((tag) => tags1.add(tag))
    })

    agent2.knowledge.forEach((entry) => {
      entry.tags.forEach((tag) => tags2.add(tag))
    })

    // Find common tags
    const commonTags: string[] = []
    tags1.forEach((tag) => {
      if (tags2.has(tag)) {
        commonTags.push(tag)
      }
    })

    // Calculate overlap score
    const totalUniqueTags = new Set([...Array.from(tags1), ...Array.from(tags2)]).size
    const overlapScore = totalUniqueTags > 0 ? commonTags.length / totalUniqueTags : 0

    return { overlapScore, commonTags }
  }

  /**
   * Initiates an autonomous conversation between agents
   */
  initiateConversation(participants: Agent[], trigger: TriggerType, topic?: string): Conversation | null {
    console.log(
      `AUTONOMOUS SYSTEM: Initiating ${trigger} conversation between: ${participants.map((a) => a.name).join(", ")}${topic ? ` about ${topic}` : ""}`,
    )

    // Check if we're at the maximum number of simultaneous conversations
    if (this.activeConversations.size >= this.options.maxSimultaneousConversations) {
      console.log(
        `AUTONOMOUS SYSTEM: Cannot initiate conversation: Maximum simultaneous conversations (${this.options.maxSimultaneousConversations}) reached`,
      )
      return null
    }

    // Create a new conversation
    const conversation: Conversation = {
      id: `auto-conv-${Date.now()}`,
      participants: participants.map((agent) => agent.id),
      messages: [],
      startTime: new Date(),
      endTime: null,
      isAutonomous: true,
      trigger,
    }

    // Add a system message about the conversation start
    const systemMessage: Message = {
      id: `msg-system-${Date.now()}`,
      content: this.generateConversationStartMessage(participants, trigger, topic),
      senderId: "system",
      timestamp: new Date(),
      metadata: {
        isSystemMessage: true,
        type: "conversation_start",
      },
    }

    conversation.messages.push(systemMessage)

    // Track this conversation
    this.activeConversations.add(conversation.id)

    // Update last conversation time for all participants
    participants.forEach((agent) => {
      this.agentLastConversationTime.set(agent.id, Date.now())
      // /* USEFUL FOR PRODUCTION TESTING: Log cooldown set */
      // console.log(`AUTONOMOUS SYSTEM: Set cooldown for ${agent.name} at ${new Date().toISOString()}`)
    })

    // Set a timeout to end the conversation after maxConversationDuration
    const timeout = setTimeout(() => {
      console.log(`AUTONOMOUS SYSTEM: Conversation ${conversation.id} reached maximum duration, ending automatically`)
      this.endConversation(conversation.id)
    }, this.options.maxConversationDuration)

    this.conversationTimeouts.set(conversation.id, timeout)

    console.log(
      `AUTONOMOUS SYSTEM: Conversation ${conversation.id} initiated successfully with ${participants.length} participants`,
    )
    return conversation
  }

  /**
   * Ends an autonomous conversation
   */
  endConversation(conversationId: string): void {
    console.log(`Ending conversation ${conversationId}`)

    // Remove from active conversations
    this.activeConversations.delete(conversationId)

    // Clear the timeout
    const timeout = this.conversationTimeouts.get(conversationId)
    if (timeout) {
      clearTimeout(timeout)
      this.conversationTimeouts.delete(conversationId)
      console.log(`Cleared timeout for conversation ${conversationId}`)
    }

    console.log(`Conversation ${conversationId} ended successfully`)
  }

  /**
   * Checks if a conversation should end based on message count
   */
  shouldEndConversation(conversation: Conversation): boolean {
    // This method should only be called for autonomous conversations
    if (!conversation || !conversation.isAutonomous) return false

    // Count non-system messages
    const messageCount = conversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length

    // Check if the conversation has reached the maximum message count
    return messageCount >= this.options.maxAutonomousMessages
  }

  /**
   * Generates a system message for the start of an autonomous conversation
   */
  private generateConversationStartMessage(participants: Agent[], trigger: TriggerType, topic?: string): string {
    const agentNames = participants.map((agent) => agent.name).join(", ")

    switch (trigger) {
      case "proximity":
        return `${agentNames} have encountered each other and started a conversation.`

      case "knowledge_overlap":
        if (topic) {
          return `${agentNames} have started a conversation about their shared knowledge of ${topic}.`
        }
        return `${agentNames} have started a conversation about their shared knowledge.`

      case "scheduled":
        return `${agentNames} have started a scheduled conversation.`

      case "user_initiated":
        return `${agentNames} have been prompted to start a conversation.`

      case "continuation":
        return `${agentNames} have continued their previous conversation.`

      default:
        return `${agentNames} have started a conversation.`
    }
  }

  /**
   * Suggests a topic for conversation based on agents' knowledge
   */
  suggestConversationTopic(agents: Agent[]): string | null {
    // Extract all tags from all agents' knowledge
    const tagCounts = new Map<string, number>()

    agents.forEach((agent) => {
      agent.knowledge.forEach((entry) => {
        entry.tags.forEach((tag) => {
          tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1)
        })
      })
    })

    // Find tags that appear in multiple agents' knowledge
    const commonTags: [string, number][] = []
    tagCounts.forEach((count, tag) => {
      if (count >= 2) {
        // At least 2 agents have this tag
        commonTags.push([tag, count])
      }
    })

    // Sort by frequency (highest first)
    commonTags.sort((a, b) => b[1] - a[1])

    // Return the most common tag, or null if none found
    return commonTags.length > 0 ? commonTags[0][0] : null
  }

  /**
   * Checks all possible triggers and returns the first one that should trigger a conversation
   */
  checkAllTriggers(agents: Agent[]): {
    shouldTrigger: boolean
    participants: Agent[]
    trigger: TriggerType
    topic?: string
  } {
    // /* USEFUL FOR PRODUCTION TESTING: Log trigger check */
    // console.log("Checking all autonomous conversation triggers")

    // Check proximity trigger
    const proximityResult = this.checkProximityTrigger(agents)
    if (proximityResult.shouldTrigger) {
      const topic = this.suggestConversationTopic(proximityResult.participants)
      return {
        ...proximityResult,
        trigger: "proximity",
        topic,
      }
    }

    // Check knowledge overlap trigger
    const knowledgeResult = this.checkKnowledgeOverlapTrigger(agents)
    if (knowledgeResult.shouldTrigger) {
      return {
        ...knowledgeResult,
        trigger: "knowledge_overlap",
      }
    }

    // No triggers matched
    return { shouldTrigger: false, participants: [], trigger: "proximity" }
  }

  /**
   * Checks if a conversation has reached minimum depth
   */
  hasReachedMinimumDepth(conversation: Conversation): boolean {
    // Always return true since we're not using minimum depth anymore
    return true
  }
}
