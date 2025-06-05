"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import type { Agent, Conversation, Message } from "@/lib/types"
import {
  ConversationOrchestrator,
  type OrchestratorOptions,
  type ResponseOptions,
} from "@/lib/conversation-orchestrator"
import { KnowledgeRetriever } from "@/lib/knowledge-retriever"
import { useLLM } from "@/contexts/llm-context"
import { useIsSending } from "@/contexts/is-sending-context"

export function useConversationOrchestrator(
  conversation: Conversation | null,
  agents: Agent[],
  options: OrchestratorOptions = {},
  onSendMessage?: (content: string, senderId: string) => void,
) {
  const { client: llmClient } = useLLM()

  // Add defensive check for isSending
  const isSendingContext = useIsSending()
  const isSending = typeof isSendingContext?.isSending === "boolean" ? isSendingContext.isSending : false

  const [processingAgents, setProcessingAgents] = useState<string[]>([])
  const [queuedAgents, setQueuedAgents] = useState<string[]>([])
  const [typingAgents, setTypingAgents] = useState<
    Record<string, { text: string; isComplete: boolean; messageId: string }>
  >({})
  const [error, setError] = useState<string | null>(null)
  const [processingMessageIds, setProcessingMessageIds] = useState<string[]>([])
  const [isProcessing, setIsProcessing] = useState(false)

  // Create refs for stable values
  const orchestratorRef = useRef<ConversationOrchestrator | null>(null)
  const conversationRef = useRef(conversation)
  const agentsRef = useRef(agents)
  const onSendMessageRef = useRef(onSendMessage)
  const optionsRef = useRef(options)
  const processedMessageRef = useRef<string | null>(null) // useRef for processed message ID

  // Create stable callback references
  const handleMessageGenerated = useCallback((message: Message) => {
    console.log("Message generated:", message)

    // Skip empty messages (when agent decided not to respond)
    if (!message.content.trim()) {
      console.log(`Skipping empty message from ${message.senderId}`)
      return
    }

    // Call the onSendMessage callback to add the message to the conversation
    if (typeof onSendMessageRef.current === "function") {
      onSendMessageRef.current(message.content, message.senderId)
    } else {
      console.warn("onSendMessage is not a function", typeof onSendMessageRef.current)
    }
  }, [])

  const handleOrchestratorError = useCallback((error: Error) => {
    console.error("Orchestrator error:", error)
    setError(`Error: ${error.message}`)

    // Clear error after 5 seconds
    setTimeout(() => {
      setError(null)
    }, 5000)
  }, [])

  // Helper function to create orchestrator on demand - NOT a hook
  const createOrchestratorOnDemand = () => {
    if (!orchestratorRef.current && conversationRef.current && llmClient) {
      console.log("[HOOK] Creating orchestrator on demand for queueing response")
      const knowledgeRetriever = new KnowledgeRetriever()
      orchestratorRef.current = new ConversationOrchestrator(
        conversationRef.current,
        agentsRef.current,
        llmClient,
        knowledgeRetriever,
        {
          ...optionsRef.current,
          onMessageGenerated: handleMessageGenerated,
          onError: handleOrchestratorError,
        },
      )
      return true
    }
    return false
  }

  // Update refs when props change
  useEffect(() => {
    conversationRef.current = conversation
    agentsRef.current = agents
    onSendMessageRef.current = onSendMessage
    optionsRef.current = options

    // Update orchestrator if it exists
    if (orchestratorRef.current && conversation) {
      orchestratorRef.current.updateConversation(conversation)
      orchestratorRef.current.updateAgents(agents)
    }
  }, [conversation, agents, onSendMessage, options])

  // Initialize orchestrator
  useEffect(() => {
    // Create or update the orchestrator when conversation changes
    if (conversation && llmClient) {
      // Create knowledge retriever if needed
      const knowledgeRetriever = new KnowledgeRetriever()

      // If orchestrator doesn't exist yet, create it
      if (!orchestratorRef.current) {
        console.log("Creating new conversation orchestrator")
        orchestratorRef.current = new ConversationOrchestrator(conversation, agents, llmClient, knowledgeRetriever, {
          ...options,
          onMessageGenerated: handleMessageGenerated,
          onError: handleOrchestratorError,
        })
      } else {
        // Otherwise update the existing one
        console.log("Updating existing conversation orchestrator")
        orchestratorRef.current.updateConversation(conversation)
        orchestratorRef.current.updateAgents(agents)
      }
    } else if (orchestratorRef.current) {
      // Clean up if conversation becomes null
      console.log("Cleaning up conversation orchestrator")
      orchestratorRef.current.cleanup()
      orchestratorRef.current = null
    }

    // Set up polling to update processing state
    const intervalId = setInterval(() => {
      if (orchestratorRef.current) {
        setProcessingAgents(orchestratorRef.current.getProcessingAgents())
        setQueuedAgents(orchestratorRef.current.getQueuedAgents())
        setProcessingMessageIds(orchestratorRef.current.getProcessingMessageIds())
        setIsProcessing(
          orchestratorRef.current.getProcessingAgents().length > 0 ||
            orchestratorRef.current.getQueuedAgents().length > 0,
        )
      }
    }, 200)

    return () => {
      clearInterval(intervalId)
    }
  }, [conversation, agents, options, llmClient, handleMessageGenerated, handleOrchestratorError])

  // Function to queue an agent response
  const queueAgentResponse = useCallback(
    (agentId: string, responseOptions: ResponseOptions = {}) => {
      console.log(`[HOOK] queueAgentResponse called for agent ${agentId}`, {
        hasMessageToRespondTo: !!responseOptions.messageToRespondTo,
        force: responseOptions.force,
        streamResponse: responseOptions.streamResponse,
        hasOnChunk: !!responseOptions.onChunk,
      })

      // Create orchestrator if needed (using the helper function)
      createOrchestratorOnDemand()

      if (!orchestratorRef.current) {
        console.error("[HOOK] Cannot queue response: Conversation orchestrator not initialized")
        setError("Cannot queue response: Conversation orchestrator not initialized")
        return
      }

      // Check if there are messages in the conversation
      if (!conversationRef.current || conversationRef.current.messages.length === 0) {
        console.error("[HOOK] Cannot generate response: No messages in conversation")
        setError("Cannot generate response: No messages in conversation")
        return
      }

      try {
        // Get the message to respond to
        const messageToRespondTo =
          responseOptions.messageToRespondTo ||
          conversationRef.current.messages[conversationRef.current.messages.length - 1]

        console.log("[HOOK] Message to respond to:", {
          id: messageToRespondTo.id,
          sender: messageToRespondTo.senderId,
          content: messageToRespondTo.content.substring(0, 30) + "...",
          type: messageToRespondTo.metadata?.type,
        })

        // Create a typing indicator immediately
        setTypingAgents((prev) => ({
          ...prev,
          [agentId]: {
            text: "...",
            isComplete: false,
            messageId: messageToRespondTo.id,
          },
        }))

        console.log(`[HOOK] Created typing indicator for agent ${agentId}`)

        // Create a safe onChunk callback - NOT using useCallback
        const safeOnChunk = (text: string, isComplete: boolean) => {
          console.log(`[HOOK] onChunk called for agent ${agentId}:`, {
            textLength: text?.length || 0,
            isComplete,
          })

          try {
            setTypingAgents((prevState) => {
              // Safety check to ensure the agent is still in the typing state
              if (!prevState[agentId]) {
                console.log(`[HOOK] Agent ${agentId} no longer in typing state, creating new entry`)
                // Create a new entry if it doesn't exist
                return {
                  ...prevState,
                  [agentId]: {
                    text: text || "",
                    isComplete: isComplete,
                    messageId: messageToRespondTo.id,
                  },
                }
              }

              const updated = { ...prevState }

              if (isComplete) {
                // Mark as complete but don't remove yet (will be removed by cleanup timer)
                updated[agentId] = { ...updated[agentId], isComplete: true }
              } else {
                // Append text safely
                const currentText = updated[agentId]?.text || ""
                const newText = text || ""
                updated[agentId] = {
                  text: currentText + newText,
                  isComplete: false,
                  messageId: messageToRespondTo.id,
                }
              }

              return updated
            })
          } catch (error) {
            console.error(`[HOOK] Error in typing indicator update for agent ${agentId}:`, error)
          }
        }

        // Set up options with the safe onChunk callback
        const options: ResponseOptions = {
          ...responseOptions,
          streamResponse: true,
          messageToRespondTo,
          onChunk: safeOnChunk,
          onMessageGenerated: handleMessageGenerated,
        }

        console.log(`[HOOK] Calling orchestratorRef.current.queueAgentResponse for agent ${agentId}`)

        // Queue the response
        orchestratorRef.current.queueAgentResponse(agentId, options)

        console.log(`[HOOK] Successfully queued response for agent ${agentId}`)
        setError(null) // Clear any previous errors
      } catch (err) {
        console.error("[HOOK] Error queueing agent response:", err)
        setError(`Failed to queue response for ${agentId}: ${err instanceof Error ? err.message : String(err)}`)
      }
    },
    [handleMessageGenerated],
  )

  // Function to process a new message
  const processNewMessage = useCallback(
    (message: Message) => {
      // Create orchestrator if needed (using the helper function)
      createOrchestratorOnDemand()

      if (!orchestratorRef.current) {
        console.error("Cannot process message: Conversation orchestrator not initialized and no conversation available")
        setError("Cannot process message: No active conversation")
        return
      }

      try {
        // Add message to conversation reference first (safety check)
        if (conversationRef.current && !conversationRef.current.messages.some((m) => m.id === message.id)) {
          console.log(`Adding message ${message.id} to conversation reference`)
          conversationRef.current = {
            ...conversationRef.current,
            messages: [...conversationRef.current.messages, message],
          }
        }

        // CRITICAL FIX: Add more detailed logging for conversation starter messages
        if (message.metadata?.type === "conversation_starter") {
          console.log("PROCESSING CONVERSATION STARTER MESSAGE:", {
            messageId: message.id,
            senderId: message.senderId,
            content: message.content,
            metadata: message.metadata,
            conversationId: conversationRef.current?.id,
            participantCount: conversationRef.current?.participants.length,
          })

          // Double check that we have participants to respond
          if (conversationRef.current) {
            const respondingAgents = agentsRef.current.filter(
              (agent) => conversationRef.current?.participants.includes(agent.id) && agent.id !== message.senderId,
            )

            console.log(
              `Found ${respondingAgents.length} agents to respond to conversation starter:`,
              respondingAgents.map((a) => a.name),
            )

            if (respondingAgents.length === 0) {
              console.error("No agents available to respond to conversation starter!")
            }
          }
        }

        // Then process the message
        console.log(`Triggering processNewMessage for message: ${message.id}`)
        orchestratorRef.current.processNewMessage(message)
        setError(null) // Clear any previous errors
      } catch (err) {
        console.error("Error processing message:", err)
        setError(`Failed to process message: ${err instanceof Error ? err.message : String(err)}`)
      }
    },
    [handleMessageGenerated, handleOrchestratorError],
  )

  // Function to cancel all responses
  const cancelAllResponses = useCallback(() => {
    if (orchestratorRef.current) {
      orchestratorRef.current.cancelAllResponses()
    }
    // Clear typing indicators
    setTypingAgents({})
    setError(null) // Clear any previous errors
  }, [])

  // Clean up completed typing indicators
  useEffect(() => {
    const cleanupTimer = setInterval(() => {
      setTypingAgents((prev) => {
        const updated = { ...prev }
        let changed = false

        // Remove completed typing indicators that are no longer processing
        Object.entries(updated).forEach(([agentId, state]) => {
          // Remove if complete or if the text contains SKIP_RESPONSE
          if (
            state.isComplete ||
            (state.text && state.text.includes("SKIP_RESPONSE")) ||
            (!processingAgents.includes(agentId) && !queuedAgents.includes(agentId))
          ) {
            delete updated[agentId]
            changed = true
          }
        })

        return changed ? updated : prev
      })
    }, 300) // Check more frequently

    return () => clearInterval(cleanupTimer)
  }, [processingAgents, queuedAgents])

  // CRITICAL FIX: Update the useEffect that checks for conversation starter messages
  useEffect(() => {
    if (!conversation || !conversation.messages || conversation.messages.length === 0) return

    const latestMessage = conversation.messages[conversation.messages.length - 1]
    if (!latestMessage) return

    // Create a stable reference to the latest message ID to prevent infinite loops
    const latestMessageId = latestMessage.id

    // Check if we've already processed this message
    if (processedMessageRef.current === latestMessageId) {
      return // Skip processing if already processed
    }

    // Log conversation starter messages with more detail
    if (latestMessage.metadata?.type === "conversation_starter") {
      console.log("CONVERSATION ORCHESTRATOR: Detected conversation starter message:", {
        messageId: latestMessage.id,
        content: latestMessage.content,
        senderId: latestMessage.senderId,
        metadata: latestMessage.metadata,
        conversationId: conversation?.id,
        participantCount: conversation?.participants.length,
        orchestratorExists: !!orchestratorRef.current,
        isProcessing: isProcessing,
        isSending: isSending,
      })

      // If we're not already processing, trigger responses
      if (!isProcessing && !isSending && orchestratorRef.current) {
        console.log("CONVERSATION ORCHESTRATOR: Triggering responses to conversation starter message")

        // Get all agents in the conversation except the sender
        const respondingAgents = agents.filter(
          (agent) => conversation.participants.includes(agent.id) && agent.id !== latestMessage.senderId,
        )

        console.log(
          `CONVERSATION ORCHESTRATOR: Found ${respondingAgents.length} agents to respond to conversation starter:`,
          respondingAgents.map((a) => a.name),
        )

        if (respondingAgents.length === 0) {
          console.error("CONVERSATION ORCHESTRATOR: No agents available to respond to conversation starter!")
        } else {
          // Queue responses from all agents with slight delays
          respondingAgents.forEach((agent, index) => {
            console.log(`CONVERSATION ORCHESTRATOR: Queueing response from ${agent.name} to conversation starter`)
            try {
              // Force response and use a longer delay to ensure proper processing
              queueAgentResponse(agent.id, {
                messageToRespondTo: latestMessage,
                responseDelay: 1000 + index * 1500, // Longer staggered delays
                force: true, // Force response regardless of dynamics
              })
            } catch (error) {
              console.error(`CONVERSATION ORCHESTRATOR: Error queueing response for ${agent.name}:`, error)
            }
          })
        }

        // Mark this message as processed
        processedMessageRef.current = latestMessageId
      } else {
        console.log(
          `CONVERSATION ORCHESTRATOR: Not triggering responses to conversation starter: orchestratorExists=${!!orchestratorRef.current}, isProcessing=${isProcessing}, isSending=${isSending}`,
        )
      }
    }

    // Skip other system messages
    if (latestMessage.metadata?.isSystemMessage) return

    // Skip if it's not a user message
    if (latestMessage.senderId !== "user") return

    // Check for direct mentions anywhere in the message
    const mentionedAgents = new Set<Agent>()

    // First check for traditional format at beginning: "Agent X, [message]" or "@Agent X [message]"
    const beginningMentionMatch = latestMessage.content.match(/^(?:@?(.+?),?\s+)/i)
    if (beginningMentionMatch) {
      const mentionedName = beginningMentionMatch[1]
      const agent = agents.find(
        (agent) =>
          agent.name.toLowerCase() === mentionedName.toLowerCase() ||
          agent.name.toLowerCase().startsWith(mentionedName.toLowerCase()),
      )
      if (agent) mentionedAgents.add(agent)
    }

    // Then check for mentions anywhere in the message
    const allMentionsRegex = /\b@?([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\b/g
    const allMatches = [...latestMessage.content.matchAll(allMentionsRegex)]

    for (const match of allMatches) {
      const potentialName = match[1]
      // Check if this potential name matches any agent
      for (const agent of agents) {
        if (
          agent.name.toLowerCase() === potentialName.toLowerCase() ||
          agent.name.toLowerCase().startsWith(potentialName.toLowerCase())
        ) {
          mentionedAgents.add(agent)
          break
        }
      }
    }

    // Queue responses for all mentioned agents that are in the conversation
    for (const mentionedAgent of mentionedAgents) {
      if (conversation.participants.includes(mentionedAgent.id)) {
        queueAgentResponse(mentionedAgent.id, {
          messageToRespondTo: latestMessage,
          responseDelay: 300, // Quick response for direct mentions
          force: true, // Force response regardless of dynamics
        })
      }
    }

    // Mark this message as processed
    processedMessageRef.current = latestMessageId
  }, [conversation, agents, isSending, isProcessing, queueAgentResponse]) // Simplified dependencies

  // Return the hook's API
  return {
    queueAgentResponse,
    processNewMessage,
    cancelAllResponses,
    processingAgents,
    queuedAgents,
    typingAgents,
    processingMessageIds,
    isProcessing: processingAgents.length > 0 || queuedAgents.length > 0,
    error,
  }
}
