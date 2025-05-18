"use client"

import { useEffect, useRef, useState } from "react"
import type { Agent, Conversation, Message } from "@/lib/types"
import { useConversationOrchestrator } from "@/hooks/use-conversation-orchestrator"
import { useLLM } from "@/contexts/llm-context"
import { createLogger } from "@/lib/debug-logger"

// Create a logger for this component
const logger = createLogger("AutonomousConversationManager")

interface AutonomousConversationManagerProps {
  conversation: Conversation | null
  agents: Agent[]
  onSendMessage: (content: string, senderId: string) => void
}

export default function AutonomousConversationManager({
  conversation,
  agents,
  onSendMessage,
}: AutonomousConversationManagerProps) {
  // Use the conversation orchestrator hook properly within a component
  const { queueAgentResponse, processNewMessage } = useConversationOrchestrator(
    conversation,
    agents,
    {
      autoSelectRespondents: true,
      responseDelay: [800, 2000],
    },
    onSendMessage,
  )

  // Get LLM client for generating conversation starters
  const { settings, client: llmClient } = useLLM()

  // Track if we've already sent a starter message
  const hasInitializedRef = useRef(false)

  // Track if we're currently generating a starter message
  const isGeneratingStarterRef = useRef(false)

  // Track the current message count for monitoring
  const [messageCount, setMessageCount] = useState(0)

  // Update message count when conversation changes
  useEffect(() => {
    if (conversation) {
      const nonSystemMessages = conversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length
      setMessageCount(nonSystemMessages)
    }
  }, [conversation])

  // Function to generate a conversation starter message using the LLM
  async function generateConversationStarter(
    firstAgent: Agent,
    participants: Agent[],
    topic?: string,
  ): Promise<string> {
    if (!llmClient) {
      logger.error("Cannot generate conversation starter: LLM client not available")
      return fallbackStarterMessage(firstAgent, topic)
    }

    // Collect all agents' information
    const agentInfos = participants.map((agent) => ({
      name: agent.name,
      biography: agent.biography,
      isStarter: agent.id === firstAgent.id,
    }))

    // Create a system prompt that explains what we want
    const systemPrompt = `You are helping to start a conversation between AI agents.
Generate a conversation starter message from the perspective of ${firstAgent.name}.
The message should:
1. Be prefixed with "${firstAgent.name}: " (include the colon and space)
2. Acknowledge the other participants
3. Reference the agent's own background/expertise
4. Mention the other agents' backgrounds/expertise
5. ${topic ? `Relate to the provided topic: ${topic}` : "Suggest a relevant topic based on the agents' backgrounds"}
6. Encourage collaboration and brainstorming
7. Ask a question that invites response

IMPORTANT: Always start with "${firstAgent.name}: " followed by the message.`

    // Create a user prompt with agent information
    const userPrompt = `Agents in conversation:
${agentInfos.map((info) => `- ${info.name}: ${info.biography}`).join("\n")}

${topic ? `Conversation topic: ${topic}` : "No specific topic provided, but suggest something relevant to the agents' backgrounds."}

Write a conversation starter message from ${firstAgent.name}'s perspective that will engage the other agents.
Remember to start with "${firstAgent.name}: " followed by the message.`

    try {
      logger.log("Generating conversation starter message", {
        firstAgent: firstAgent.name,
        participantCount: participants.length,
        topic,
      })

      // Generate the starter message
      const response = await llmClient.generateResponse(systemPrompt, userPrompt)

      // Ensure the response starts with the agent name
      let formattedResponse = response.trim()
      const expectedPrefix = `${firstAgent.name}:`

      if (!formattedResponse.startsWith(expectedPrefix)) {
        formattedResponse = `${expectedPrefix} ${formattedResponse}`
        logger.log("Added missing agent name prefix to starter message")
      }

      logger.log("Successfully generated conversation starter", {
        messagePreview: formattedResponse.substring(0, 50) + "...",
      })

      return formattedResponse
    } catch (error) {
      logger.error("Error generating conversation starter:", error)
      return fallbackStarterMessage(firstAgent, topic)
    }
  }

  // Fallback message in case LLM generation fails
  function fallbackStarterMessage(agent: Agent, topic?: string): string {
    logger.log("Using fallback conversation starter message")
    return `${agent.name}: Hello everyone! I'm ${agent.name}, ${agent.biography.split(".")[0]}. ${
      topic
        ? `I'd love to discuss ${topic} with you all.`
        : "I'd love to discuss our backgrounds and see how we might collaborate."
    } What are your thoughts?`
  }

  // Update the useEffect to use the enhanced conversation starter
  useEffect(() => {
    if (!conversation || !conversation.isAutonomous) return

    // Check if we've already initialized this conversation
    if (hasInitializedRef.current || isGeneratingStarterRef.current) return

    logger.log("Checking conversation state", {
      conversationId: conversation.id,
      messageCount: conversation.messages.length,
      participants: conversation.participants,
    })

    // Check if there are any messages or only a system message
    const onlyHasSystemMessage =
      conversation.messages.length === 1 && conversation.messages[0].metadata?.isSystemMessage === true

    if (conversation.messages.length === 0 || onlyHasSystemMessage) {
      logger.log("Initializing autonomous conversation", {
        conversationId: conversation.id,
        participants: conversation.participants,
        trigger: conversation.trigger,
        topic: conversation.topic,
      })

      try {
        // Find the first agent to use as the starter
        const firstAgent = agents.find((agent) => conversation.participants.includes(agent.id))
        if (firstAgent) {
          // Get all participating agents
          const participatingAgents = agents.filter((agent) => conversation.participants.includes(agent.id))

          // Set the generating flag to prevent duplicate attempts
          isGeneratingStarterRef.current = true

          // Generate the conversation starter asynchronously
          generateConversationStarter(firstAgent, participatingAgents, conversation.topic)
            .then((starterContent) => {
              // Create a conversation starter message
              const starterMessage: Message = {
                id: `msg-starter-${Date.now()}`,
                content: starterContent, // Use the generated content
                senderId: firstAgent.id,
                timestamp: new Date(),
                metadata: {
                  isGeneratedByLLM: true,
                  type: "conversation_starter",
                },
              }

              logger.log("Sending starter message", {
                messageId: starterMessage.id,
                sender: firstAgent.name,
                contentPreview: starterMessage.content.substring(0, 50) + "...",
              })

              // Send the message
              onSendMessage(starterContent, firstAgent.id)

              // Mark as initialized to prevent duplicate messages
              hasInitializedRef.current = true
              isGeneratingStarterRef.current = false

              // Process the message to trigger responses after a short delay
              setTimeout(() => {
                try {
                  logger.log("Processing starter message to trigger responses")
                  processNewMessage(starterMessage)
                } catch (error) {
                  logger.error("Error processing starter message:", error)
                }
              }, 1000)
            })
            .catch((error) => {
              logger.error("Error in conversation starter generation:", error)
              isGeneratingStarterRef.current = false
            })
        } else {
          logger.error("No agents found for conversation")
        }
      } catch (error) {
        logger.error("Error initializing conversation:", error)
        isGeneratingStarterRef.current = false
      }
    } else {
      // If there are already messages, mark as initialized
      logger.log("Conversation already has messages, marking as initialized")
      hasInitializedRef.current = true
    }
  }, [conversation, agents, onSendMessage, processNewMessage, llmClient])

  // Add a new effect to monitor conversation progress and ensure it reaches minimum message count
  useEffect(() => {
    if (!conversation || !conversation.isAutonomous || !hasInitializedRef.current) return

    // Get the minimum and maximum message counts from settings
    const minMessages = settings.minAutonomousMessages || 4
    const maxMessages = settings.maxAutonomousMessages || 10

    // Count non-system messages
    const nonSystemMessages = conversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length

    // If we haven't reached the minimum message count, set up a monitoring interval
    if (nonSystemMessages < minMessages) {
      logger.log(`Setting up conversation progress monitor: ${nonSystemMessages}/${minMessages} messages`)

      // Set up an interval to check if the conversation needs to be continued
      const intervalId = setInterval(() => {
        // Skip if conversation has been deleted or changed
        if (!conversation) {
          clearInterval(intervalId)
          return
        }

        // Recount non-system messages (they might have changed)
        const currentNonSystemMessages = conversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length

        // If we've reached the minimum, clear the interval
        if (currentNonSystemMessages >= minMessages) {
          logger.log(
            `Conversation reached minimum message count (${currentNonSystemMessages}/${minMessages}), stopping monitor`,
          )
          clearInterval(intervalId)
          return
        }

        // Get the last message
        const lastMessage = conversation.messages[conversation.messages.length - 1]
        if (!lastMessage) return

        // Check if the last message was sent more than 5 seconds ago
        const timeSinceLastMessage = Date.now() - new Date(lastMessage.timestamp).getTime()

        if (timeSinceLastMessage > 5000) {
          // 5 seconds
          logger.log(
            `Conversation stalled at ${currentNonSystemMessages}/${minMessages} messages, prompting continuation`,
          )

          // Find an agent who hasn't spoken recently
          const lastSpeaker = lastMessage.senderId
          const availableAgents = agents.filter(
            (agent) => conversation.participants.includes(agent.id) && agent.id !== lastSpeaker,
          )

          if (availableAgents.length > 0) {
            // Pick a random agent to continue the conversation
            const nextAgent = availableAgents[Math.floor(Math.random() * availableAgents.length)]

            logger.log(`Prompting ${nextAgent.name} to continue the conversation`)

            // Queue a response from this agent to keep the conversation going
            queueAgentResponse(nextAgent.id, {
              messageToRespondTo: lastMessage,
              responseDelay: 500,
              force: true, // Force response regardless of dynamics
            })
          }
        }
      }, 3000) // Check every 3 seconds

      // Clean up the interval when the component unmounts or conversation changes
      return () => clearInterval(intervalId)
    }
  }, [conversation, messageCount, agents, settings, queueAgentResponse])

  // Reset initialization when conversation changes
  useEffect(() => {
    return () => {
      hasInitializedRef.current = false
      isGeneratingStarterRef.current = false
    }
  }, [conversation?.id])

  // This component doesn't render anything
  return null
}
