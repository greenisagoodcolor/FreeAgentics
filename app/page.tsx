"use client"

import type React from "react"

import { useState, useRef, useEffect, useMemo } from "react"
import type { Agent, Conversation, Position, KnowledgeEntry, Message } from "@/lib/types"
import { AutonomousConversationSystem } from "@/lib/autonomous-conversation"
import { createLogger } from "@/lib/debug-logger"
// Add the missing imports at the top of the file
import AgentList from "@/components/agent-list"
import MemoryViewer from "@/components/memory-viewer"
import GridWorld from "@/components/grid-world"
import ChatWindow from "@/components/chat-window"
import AutonomousConversationManager from "@/components/autonomous-conversation-manager"
import { useLLM } from "@/lib/use-llm"
import { exportAgentsKnowledge } from "@/lib/knowledge-export"
import { importAgentsAndSettingsFromZip, mergeImportedAgents } from "@/lib/knowledge-import"
import { useToast } from "@/hooks/use-toast"
import AboutModal from "@/components/about-modal"

// Create a module-specific logger
const logger = createLogger("app/page")

export default function Home() {
  // First, let's add some debugging to see if the component is rendering at all
  console.log("Home component rendering")

  // Add toast for notifications
  const { toast } = useToast()

  // Update the initial agents state to have autonomy enabled by default
  const [agents, setAgents] = useState<Agent[]>([
    {
      id: "1",
      name: "Agent 1",
      biography: "AI researcher specializing in natural language processing.",
      inConversation: false,
      position: { x: 2, y: 3 },
      color: "#4f46e5",
      knowledge: [],
      autonomyEnabled: true, // Enable autonomy by default
    },
    {
      id: "2",
      name: "Agent 2",
      biography: "Expert in machine learning and neural networks.",
      inConversation: false,
      position: { x: 4, y: 3 },
      color: "#10b981",
      knowledge: [],
      autonomyEnabled: true, // Enable autonomy by default
    },
  ])

  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null)
  const [conversationHistory, setConversationHistory] = useState<Conversation[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)

  // State for tracking selected knowledge node from global graph
  const [selectedKnowledgeNode, setSelectedKnowledgeNode] = useState<{
    type: "entry" | "tag"
    id: string
    title: string
  } | null>(null)

  const [showAboutModal, setShowAboutModal] = useState<boolean>(true) // Show on startup

  // Panel resizing state
  const [panelWidths, setPanelWidths] = useState<number[]>([25, 25, 25, 25]) // Percentages
  const [resizing, setResizing] = useState<number | null>(null)
  const mainRef = useRef<HTMLElement>(null)

  // Autonomous conversation system
  const autonomousSystemRef = useRef<AutonomousConversationSystem>(new AutonomousConversationSystem())

  // Add a useRef for tracking conversation starter attempts
  const conversationStarterAttemptsRef = useRef<Record<string, number>>({})
  const maxConversationStarterAttempts = 3

  // Add a ref to track if a conversation is being initiated to prevent race conditions
  const isInitiatingConversationRef = useRef<boolean>(false)

  // Add a ref to track pending messages to ensure they're processed in order
  const pendingMessagesRef = useRef<Message[]>([])

  // Get LLM client and settings - memoized to prevent duplicate declarations
  const { client } = useLLM()
  const llmClient = useMemo(() => client, [client])

  // Handle exporting agents
  const handleExportAgents = async (
    agentIds: string[],
    options: {
      includeSettings: boolean
      includeApiKeys: boolean
      includeConversations: boolean
    },
  ) => {
    try {
      // Filter agents to export
      const agentsToExport = agents.filter((agent) => agentIds.includes(agent.id))

      if (agentsToExport.length === 0 && !options.includeSettings && !options.includeConversations) {
        toast({
          title: "Export failed",
          description: "No agents selected for export",
          variant: "destructive",
          duration: 3000,
        })
        return
      }

      // Get current LLM settings if needed
      let settings = undefined
      if (options.includeSettings) {
        settings = llmClient?.getSettings()

        // Log settings before export
        logger.debug("Exporting settings:", {
          provider: settings?.provider,
          model: settings?.model,
          hasApiKey: !!settings?.apiKey,
          hasApiKeySessionId: !!settings?.apiKeySessionId,
        })
      }

      // Filter conversations if needed
      const conversationsToExport = options.includeConversations
        ? conversationHistory.filter((conv) =>
            // Only include conversations where at least one selected agent participated
            conv.participants.some((participantId) => agentIds.includes(participantId)),
          )
        : undefined

      // Export the agents with or without settings and conversations
      await exportAgentsKnowledge(agentsToExport, {
        includeSettings: options.includeSettings,
        settings,
        includeApiKeys: options.includeApiKeys,
        includeConversations: options.includeConversations,
        conversations: conversationsToExport,
      })

      // Show success toast
      let description = ""
      const agentCount = agentsToExport.length
      const conversationCount = conversationsToExport?.length || 0

      const parts = []
      if (agentCount > 0) {
        parts.push(`${agentCount} agent${agentCount > 1 ? "s" : ""}`)
      }
      if (options.includeSettings) {
        parts.push("settings")
      }
      if (options.includeConversations && conversationCount > 0) {
        parts.push(`${conversationCount} conversation${conversationCount > 1 ? "s" : ""}`)
      }

      description = `Exported ${parts.join(", ")}`

      toast({
        title: "Export successful",
        description,
        duration: 3000,
      })
    } catch (error) {
      console.error("Error exporting agents:", error)
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
        duration: 5000,
      })
    }
  }

  // Handle importing agents
  const handleImportAgents = async (
    file: File,
    options: {
      mode: "replace" | "new" | "merge" | "settings-only"
      importSettings: boolean
      importApiKeys: boolean
      importConversations: boolean // New option
    },
  ) => {
    try {
      // Show loading toast
      toast({
        title: "Importing data",
        description: "Please wait while we process your file...",
        duration: 3000,
      })

      logger.info("Starting import process", { fileName: file.name, fileSize: file.size, options })

      // Import agents, settings, and conversations from the ZIP file
      const {
        agents: importedAgents,
        settings: importedSettings,
        conversations: importedConversations,
      } = await importAgentsAndSettingsFromZip(file, options)

      // Handle settings import if requested and available
      if (options.importSettings && importedSettings && llmClient) {
        // Log the imported settings before any modifications
        logger.info("Raw imported settings before processing:", {
          provider: importedSettings.provider,
          model: importedSettings.model,
          hasApiKey: !!importedSettings.apiKey,
          apiKeyLength: importedSettings.apiKey ? importedSettings.apiKey.length : 0,
        })

        // Handle API key based on import options
        const currentSettings = llmClient.getSettings()
        if (!options.importApiKeys) {
          // Not importing API keys, preserve the current one
          if (currentSettings.apiKey) {
            logger.info(`Preserving current API key (length: ${currentSettings.apiKey.length})`)
            importedSettings.apiKey = currentSettings.apiKey
          } else {
            logger.warn("No current API key to preserve")
          }
        } else {
          // We are importing API keys
          if (importedSettings.apiKey) {
            logger.info(`Using imported API key (length: ${importedSettings.apiKey.length})`)
          } else {
            logger.warn("Import API keys option is true, but no API key found in imported settings")
            // If we're supposed to import an API key but none was found, keep the current one
            if (currentSettings.apiKey) {
              logger.info(`No API key in import, keeping current key (length: ${currentSettings.apiKey.length})`)
              importedSettings.apiKey = currentSettings.apiKey
            }
          }
        }

        // Log the settings we're about to apply
        logger.info("Settings about to be applied:", {
          provider: importedSettings.provider,
          model: importedSettings.model,
          hasApiKey: !!importedSettings.apiKey,
          apiKeyLength: importedSettings.apiKey ? importedSettings.apiKey.length : 0,
        })

        // Update settings
        llmClient.updateSettings(importedSettings)

        // Verify the settings were applied correctly
        const updatedSettings = llmClient.getSettings()
        logger.info("Settings after update:", {
          provider: updatedSettings.provider,
          model: updatedSettings.model,
          hasApiKey: !!updatedSettings.apiKey,
          apiKeyLength: updatedSettings.apiKey ? updatedSettings.apiKey.length : 0,
        })

        // Update autonomous conversation system options
        autonomousSystemRef.current.updateOptions({
          maxAutonomousMessages: importedSettings.maxAutonomousMessages || 4,
        })

        toast({
          title: "Settings imported",
          description: `LLM settings updated to use ${updatedSettings.provider} with model ${updatedSettings.model}${updatedSettings.apiKey ? " (API key included)" : " (NO API KEY)"}`,
          duration: 5000,
        })
      }

      // Handle agents import if not in settings-only mode and agents were found
      if (options.mode !== "settings-only" && importedAgents && importedAgents.length > 0) {
        // Merge imported agents with existing agents based on the selected mode
        const mergedAgents = mergeImportedAgents(importedAgents, agents, options)

        // Update the agents state
        setAgents(mergedAgents)

        // If the selected agent was modified, update it too
        if (selectedAgent) {
          const updatedSelectedAgent = mergedAgents.find((agent) => agent.id === selectedAgent.id)
          if (updatedSelectedAgent) {
            setSelectedAgent(updatedSelectedAgent)
          }
        }

        // Show success toast for agents
        toast({
          title: "Agents imported",
          description: `Imported ${importedAgents.length} agent${importedAgents.length > 1 ? "s" : ""}`,
          duration: 3000,
        })
      } else if (options.mode !== "settings-only" && (!importedAgents || importedAgents.length === 0)) {
        // Show warning if no agents were found but we weren't in settings-only mode
        toast({
          title: "No agents found",
          description: "The import file did not contain any valid agents",
          variant: "destructive",
          duration: 3000,
        })
      }

      // Handle conversation import if requested and available
      if (options.importConversations && importedConversations && importedConversations.length > 0) {
        // Add imported conversations to conversation history
        setConversationHistory((prevHistory) => {
          // Create a set of existing conversation IDs for quick lookup
          const existingIds = new Set(prevHistory.map((conv) => conv.id))

          // Filter out any conversations that already exist
          const newConversations = importedConversations.filter((conv) => !existingIds.has(conv.id))

          // Add the new conversations to the history
          return [...prevHistory, ...newConversations]
        })

        toast({
          title: "Conversations imported",
          description: `Imported ${importedConversations.length} conversation${importedConversations.length > 1 ? "s" : ""}`,
          duration: 3000,
        })
      }
    } catch (error) {
      logger.error("Error importing:", error)
      toast({
        title: "Import failed",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
        duration: 5000,
      })
    }
  }

  // Modify the sendConversationStarterMessage function to ensure the starter message is properly created and added
  // Replace the existing sendConversationStarterMessage function with this improved version

  const sendConversationStarterMessage = (conversation: Conversation, firstAgent: Agent, topic?: string) => {
    console.log(`APP: Sending conversation starter message from ${firstAgent.name}`)

    // Set the initiating flag to prevent race conditions
    isInitiatingConversationRef.current = true

    // Create a conversation starter message with clear metadata
    const starterMessage: Message = {
      id: `msg-starter-${Date.now()}`,
      content: `Hello! I noticed we're both here. Let's have a conversation${topic ? ` about ${topic}` : ""}. What are your thoughts?`,
      senderId: firstAgent.id,
      timestamp: new Date(),
      metadata: {
        isGeneratedByLLM: true,
        type: "conversation_starter",
      },
    }

    // Add the message directly to the conversation state to avoid race conditions
    setActiveConversation((prev) => {
      if (!prev) return null

      console.log("APP: Adding starter message directly to conversation:", {
        messageId: starterMessage.id,
        conversationId: prev.id,
        currentMessageCount: prev.messages.length,
      })

      // Check if this message already exists to prevent duplicates
      if (prev.messages.some((msg) => msg.id === starterMessage.id)) {
        return prev
      }

      return {
        ...prev,
        messages: [...prev.messages, starterMessage],
      }
    })

    // Clear the initiating flag after a short delay to ensure state updates
    setTimeout(() => {
      isInitiatingConversationRef.current = false

      // Log to verify the conversation state
      console.log("APP: Conversation after adding starter:", {
        id: conversation.id,
        participants: conversation.participants,
        messageCount: activeConversation?.messages.length || 0,
        hasStarterMessage:
          activeConversation?.messages.some((msg) => msg.metadata?.type === "conversation_starter") || false,
      })
    }, 300)
  }

  // Update the useEffect that handles autonomous conversations
  useEffect(() => {
    // Skip if there's already an active conversation
    if (activeConversation) return

    const checkInterval = setInterval(() => {
      try {
        // Check for autonomous conversation triggers
        const { shouldTrigger, participants, trigger, topic } = autonomousSystemRef.current.checkAllTriggers(agents)

        if (shouldTrigger && participants.length >= 2) {
          console.log("APP: Autonomous conversation triggered:", {
            trigger,
            participants: participants.map((a) => a.name),
            topic,
          })

          // Create the conversation
          const newConversation = autonomousSystemRef.current.initiateConversation(participants, trigger, topic)

          if (newConversation) {
            // Update agents' inConversation status
            setAgents((prevAgents) =>
              prevAgents.map((agent) => ({
                ...agent,
                inConversation: newConversation.participants.includes(agent.id),
              })),
            )

            // Set as active conversation with topic if available
            setActiveConversation({
              ...newConversation,
              topic: topic, // Store the topic in the conversation
            })

            // Reset conversation starter attempts for this conversation
            conversationStarterAttemptsRef.current[newConversation.id] = 0

            // No need to call sendConversationStarterMessage here anymore
            // The AutonomousConversationManager will handle it
          }
        }
      } catch (error) {
        console.error("APP: Error in autonomous conversation check:", error)
      }
    }, 5000) // Check every 5 seconds

    return () => clearInterval(checkInterval)
  }, [agents, activeConversation])

  // Modify the useEffect that monitors conversation progress to improve stall detection
  // Find the useEffect that contains the checkStalled function and replace it with this improved version

  // Add a new useEffect to monitor conversation progress and retry if needed
  useEffect(() => {
    // Skip if there's no active conversation or if it's not autonomous
    if (!activeConversation || !activeConversation.isAutonomous) return

    logger.debug("Setting up conversation progress monitor", {
      conversationId: activeConversation.id,
      messageCount: activeConversation.messages.length,
      hasStarterMessage: activeConversation.messages.some((msg) => msg.metadata?.type === "conversation_starter"),
    })

    // Check if the conversation has stalled (only one message - the starter)
    const checkStalled = () => {
      // If conversation has been deleted or is no longer active, do nothing
      if (!activeConversation) {
        logger.debug("Conversation no longer active, skipping stall check")
        return
      }

      // Log the current state of the conversation for debugging
      logger.debug("Checking if conversation is stalled", {
        messageCount: activeConversation.messages.length,
        messages: activeConversation.messages.map((m) => ({
          id: m.id,
          sender: m.senderId,
          type: m.metadata?.type,
          isSystem: m.metadata?.isSystemMessage,
        })),
      })

      // If conversation has more than 2 messages, it's progressing normally
      if (activeConversation.messages.length > 2) {
        logger.debug("Conversation is progressing normally", {
          messageCount: activeConversation.messages.length,
        })
        return
      }

      // If we're already initiating a conversation, skip
      if (isInitiatingConversationRef.current) {
        logger.debug("Already initiating a conversation, skipping stall check")
        return
      }

      // Find the starter message
      const starterMessage = activeConversation.messages.find((msg) => msg.metadata?.type === "conversation_starter")

      // If no starter message is found, create one
      if (!starterMessage) {
        logger.warn("No starter message found in conversation, creating one")

        // Find the first agent to use as the sender
        const firstAgent = agents.find((agent) => activeConversation.participants.includes(agent.id))
        if (firstAgent) {
          sendConversationStarterMessage(activeConversation, firstAgent, activeConversation.topic)
          return
        } else {
          logger.error("No agents found for conversation, cannot create starter message")
        }
        return
      }

      // If we've already tried the maximum number of times, end the conversation
      const attempts = conversationStarterAttemptsRef.current[activeConversation.id] || 0
      if (attempts >= maxConversationStarterAttempts) {
        logger.warn(`Conversation ${activeConversation.id} failed to start after ${attempts} attempts, ending it`)

        // Reset cooldowns for all participants
        autonomousSystemRef.current.resetCooldown(activeConversation.participants)

        // End the conversation
        endConversation()
        return
      }

      // Increment the attempt counter
      conversationStarterAttemptsRef.current[activeConversation.id] = attempts + 1

      logger.info(
        `Conversation ${activeConversation.id} appears stalled (attempt ${attempts + 1}/${maxConversationStarterAttempts}), retrying...`,
      )

      // Find agents that should respond to the starter message
      const respondingAgents = agents.filter(
        (agent) => activeConversation.participants.includes(agent.id) && agent.id !== starterMessage.senderId,
      )

      if (respondingAgents.length === 0) {
        logger.warn("No responding agents available")
        return
      }

      // Send a direct prompt from the system to encourage response
      const systemPrompt: Message = {
        id: `msg-system-retry-${Date.now()}`,
        content: `${respondingAgents[0].name}, what do you think about what ${agents.find((a) => a.id === starterMessage.senderId)?.name || "the other agent"} just said?`,
        senderId: "system",
        timestamp: new Date(),
        metadata: {
          isSystemMessage: true,
          type: "conversation_prompt",
        },
      }

      logger.info("Sending system prompt to encourage response", {
        prompt: systemPrompt.content,
        targetAgent: respondingAgents[0].name,
      })

      // Add the system prompt directly to the conversation
      setActiveConversation((prev) => {
        if (!prev) return null
        return {
          ...prev,
          messages: [...prev.messages, systemPrompt],
        }
      })
    }

    // Check after 15 seconds if the conversation has stalled (increased from 10 to 15)
    const timeoutId = setTimeout(checkStalled, 15000)

    return () => {
      logger.debug("Clearing conversation progress monitor")
      clearTimeout(timeoutId)
    }
  }, [activeConversation, agents])

  // Process pending messages
  useEffect(() => {
    if (pendingMessagesRef.current.length === 0) return

    // Only process if we're not initiating a conversation
    if (!isInitiatingConversationRef.current && activeConversation) {
      const pendingMessage = pendingMessagesRef.current.shift()
      if (pendingMessage) {
        logger.debug("Processing pending message", {
          id: pendingMessage.id,
          sender: pendingMessage.senderId,
          contentPreview: pendingMessage.content.substring(0, 30) + "...",
        })

        // Update the conversation with the pending message
        setActiveConversation((prev) => {
          if (!prev) return null
          return {
            ...prev,
            messages: [...prev.messages, pendingMessage],
          }
        })
      }
    }
  }, [activeConversation]) // Remove pendingMessagesRef.current from dependencies

  // Add a new useEffect hook after the existing useEffect that monitors conversation progress
  // This should be placed before the return statement, with the other useEffect hooks

  // Add this new useEffect to check if autonomous conversations should end based on message count
  useEffect(() => {
    // Skip if there's no active conversation or if it's not autonomous
    if (!activeConversation || !activeConversation.isAutonomous) return

    // Get the non-system message count
    const messageCount = activeConversation.messages.filter((msg) => !msg.metadata?.isSystemMessage).length

    // Log the current message count for debugging
    logger.debug("Checking if autonomous conversation should end", {
      conversationId: activeConversation.id,
      messageCount,
      maxMessages: autonomousSystemRef.current.options.maxAutonomousMessages,
      isAutonomous: activeConversation.isAutonomous,
    })

    // Check if the conversation should end based on message count
    if (autonomousSystemRef.current.shouldEndConversation(activeConversation)) {
      logger.info(
        `Automatically ending autonomous conversation ${activeConversation.id} after reaching ${messageCount} messages (max: ${autonomousSystemRef.current.options.maxAutonomousMessages})`,
      )

      // End the conversation using the existing function
      endConversation()
    }
  }, [activeConversation]) // Only run when the activeConversation changes

  // Checking for any other problematic usage
  const createAgent = () => {
    const newAgent: Agent = {
      id: `${agents.length + 1}`,
      name: `Agent ${agents.length + 1}`,
      biography: "New agent biography...",
      inConversation: false,
      position: { x: Math.floor(Math.random() * 10), y: Math.floor(Math.random() * 10) },
      color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
      knowledge: [],
      autonomyEnabled: true, // Enable autonomy by default for new agents too
    }
    setAgents([...agents, newAgent])
  }

  const createAgentWithName = (name: string) => {
    const newAgent: Agent = {
      id: `${agents.length + 1}`,
      name: name,
      biography: "New agent biography...",
      inConversation: false,
      position: { x: Math.floor(Math.random() * 10), y: Math.floor(Math.random() * 10) },
      color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
      knowledge: [],
      autonomyEnabled: true, // Enable autonomy by default for new agents too
    }
    setAgents([...agents, newAgent])
  }

  const deleteAgent = (agentId: string) => {
    setAgents(agents.filter((agent) => agent.id !== agentId))
    if (selectedAgent?.id === agentId) {
      setSelectedAgent(null)
    }
  }

  // Update the addAgentToConversation function to explicitly mark user-initiated conversations
  // Find the addAgentToConversation function and replace it with this version:

  const addAgentToConversation = (agentId: string) => {
    setAgents(agents.map((agent) => (agent.id === agentId ? { ...agent, inConversation: true } : agent)))

    // If no active conversation, create one
    if (!activeConversation) {
      setActiveConversation({
        id: `conv-${Date.now()}`,
        participants: [agentId],
        messages: [],
        startTime: new Date(),
        endTime: null,
        isAutonomous: false, // Explicitly mark as user-initiated
      })
    } else {
      // Update the active conversation with the new participant
      const updatedConversation = {
        ...activeConversation,
        participants: [...activeConversation.participants, agentId],
      }
      setActiveConversation(updatedConversation)

      // If there are messages in the conversation, add a system message about the new agent
      if (updatedConversation.messages.length > 0) {
        const newAgentName = agents.find((agent) => agent.id === agentId)?.name || "Unknown Agent"
        const systemMessage = {
          id: `msg-system-${Date.now()}`,
          content: `${newAgentName} has joined the conversation.`,
          senderId: "system",
          timestamp: new Date(),
          metadata: {
            isSystemMessage: true,
            type: "agent_joined",
          },
        }

        // Add the system message to the conversation
        const updatedWithSystemMessage = {
          ...updatedConversation,
          messages: [...updatedConversation.messages, systemMessage],
        }
        setActiveConversation(updatedWithSystemMessage)

        // Check if the last user message mentioned this agent
        const lastMessage = updatedConversation.messages[updatedConversation.messages.length - 1]
        if (lastMessage && lastMessage.senderId === "user") {
          const mentionRegex = new RegExp(`\\b${newAgentName}\\b`, "i")
          if (mentionRegex.test(lastMessage.content)) {
            // The new agent was mentioned in the last message, trigger a response
            // We'll use a timeout to ensure the conversation state is updated first
            setTimeout(() => {
              // This will be handled by the conversation orchestrator in ChatWindow
              logger.debug(`New agent ${newAgentName} was mentioned, should respond to: ${lastMessage.content}`)
            }, 500)
          }
        }
      }
    }
  }

  const removeAgentFromConversation = (agentId: string) => {
    setAgents(agents.map((agent) => (agent.id === agentId ? { ...agent, inConversation: false } : agent)))

    if (activeConversation) {
      const updatedParticipants = activeConversation.participants.filter((id) => id !== agentId)

      if (updatedParticipants.length === 0) {
        // End conversation if no participants left
        const endedConversation = {
          ...activeConversation,
          endTime: new Date(),
        }
        setConversationHistory([...conversationHistory, endedConversation])
        setActiveConversation(null)
      } else {
        setActiveConversation({
          ...activeConversation,
          participants: updatedParticipants,
        })
      }
    }
  }

  const updateAgentPosition = (agentId: string, position: Position) => {
    setAgents(agents.map((agent) => (agent.id === agentId ? { ...agent, position } : agent)))

    // Check for adjacent agents to start spontaneous conversation
    const updatedAgent = agents.find((agent) => agent.id === agentId)
    if (updatedAgent && !activeConversation) {
      const adjacentAgents = agents.filter(
        (agent) =>
          agent.id !== agentId &&
          Math.abs(agent.position.x - position.x) <= 1 &&
          Math.abs(agent.position.y - position.y) <= 1,
      )

      if (adjacentAgents.length > 0) {
        // Check if both the moved agent and at least one adjacent agent have autonomy enabled
        const autonomousAgents = [updatedAgent, ...adjacentAgents.filter((agent) => agent.autonomyEnabled)].filter(
          (agent) => agent.autonomyEnabled,
        )

        // Only start a conversation if we have at least 2 autonomous agents
        if (autonomousAgents.length >= 2) {
          const participants = autonomousAgents.map((agent) => agent.id)

          // Start a new conversation with adjacent agents
          setActiveConversation({
            id: `conv-${Date.now()}`,
            participants,
            messages: [],
            startTime: new Date(),
            endTime: null,
            isAutonomous: true,
            trigger: "proximity",
          })

          // Update agents' inConversation status
          setAgents(
            agents.map((agent) => (participants.includes(agent.id) ? { ...agent, inConversation: true } : agent)),
          )
        }
      }
    }
  }

  const updateAgentColor = (agentId: string, color: string) => {
    setAgents(agents.map((agent) => (agent.id === agentId ? { ...agent, color } : agent)))

    // Update selected agent if it's the one being modified
    if (selectedAgent?.id === agentId) {
      setSelectedAgent({
        ...selectedAgent,
        color,
      })
    }
  }

  // New function to toggle agent autonomy
  const toggleAgentAutonomy = (agentId: string, enabled: boolean) => {
    logger.info(`Toggling autonomy for agent ${agentId} to ${enabled ? "enabled" : "disabled"}`)

    setAgents(agents.map((agent) => (agent.id === agentId ? { ...agent, autonomyEnabled: enabled } : agent)))

    // Update selected agent if it's the one being modified
    if (selectedAgent?.id === agentId) {
      setSelectedAgent({
        ...selectedAgent,
        autonomyEnabled: enabled,
      })
    }

    // Log the updated agents with autonomy status
    setTimeout(() => {
      const autonomousAgents = agents.filter((agent) => agent.autonomyEnabled)
      logger.debug(
        "Agents with autonomy enabled:",
        autonomousAgents.map((a) => a.name),
      )
    }, 100)
  }

  // Find the addKnowledgeToAgent function and replace it with this implementation
  // that uses the functional form of setState to ensure proper state updates

  const addKnowledgeToAgent = (agentId: string, knowledge: KnowledgeEntry) => {
    console.log(`Adding knowledge to agent ${agentId}:`, knowledge)

    // Use the functional form of setState to ensure we're working with the latest state
    setAgents((currentAgents) => {
      // Find the agent to update
      const agentToUpdate = currentAgents.find((agent) => agent.id === agentId)
      if (!agentToUpdate) {
        console.log(`Agent with ID ${agentId} not found`)
        return currentAgents
      }

      // Check if this knowledge entry already exists (by ID)
      const existingEntry = agentToUpdate.knowledge.find((k) => k.id === knowledge.id)
      if (existingEntry) {
        console.log(`Knowledge entry with ID ${knowledge.id} already exists, skipping`)
        return currentAgents
      }

      // Create a new array of agents with the updated knowledge
      return currentAgents.map((agent) => {
        if (agent.id === agentId) {
          console.log(`Adding knowledge entry to agent ${agent.name}, current count: ${agent.knowledge.length}`)
          return {
            ...agent,
            knowledge: [...agent.knowledge, knowledge],
          }
        }
        return agent
      })
    })

    // Also update selectedAgent if it's the one being modified
    if (selectedAgent?.id === agentId) {
      setSelectedAgent((current) => {
        if (!current) return null

        // Check if this knowledge entry already exists (by ID)
        const existingEntry = current.knowledge.find((k) => k.id === knowledge.id)
        if (existingEntry) {
          console.log(`Knowledge entry with ID ${knowledge.id} already exists in selectedAgent, skipping`)
          return current
        }

        console.log(`Updating selectedAgent knowledge, current count: ${current.knowledge.length}`)
        return {
          ...current,
          knowledge: [...current.knowledge, knowledge],
        }
      })
    }
  }

  // New function to delete knowledge from an agent
  const deleteKnowledgeFromAgent = (agentId: string, knowledgeId: string) => {
    // Create a new agents array with the updated knowledge
    const updatedAgents = agents.map((agent) =>
      agent.id === agentId ? { ...agent, knowledge: agent.knowledge.filter((k) => k.id !== knowledgeId) } : agent,
    )

    setAgents(updatedAgents)

    // Update selected agent if it's the one being modified
    if (selectedAgent?.id === agentId) {
      setSelectedAgent({
        ...selectedAgent,
        knowledge: selectedAgent.knowledge.filter((k) => k.id !== knowledgeId),
      })
    }
  }

  // New function to update knowledge in an agent
  const updateKnowledgeInAgent = (agentId: string, knowledgeId: string, updates: Partial<KnowledgeEntry>) => {
    // Create a new agents array with the updated knowledge
    const updatedAgents = agents.map((agent) => {
      if (agent.id === agentId) {
        const updatedKnowledge = agent.knowledge.map((k) => (k.id === knowledgeId ? { ...k, ...updates } : k))
        return { ...agent, knowledge: updatedKnowledge }
      }
      return agent
    })

    setAgents(updatedAgents)

    // Update selected agent if it's the one being modified
    if (selectedAgent?.id === agentId) {
      setSelectedAgent({
        ...selectedAgent,
        knowledge: selectedAgent.knowledge.map((k) => (k.id === knowledgeId ? { ...k, ...updates } : k)),
      })
    }
  }

  const updateAgent = (agentId: string, updates: { name?: string; biography?: string }) => {
    // Create a new agents array with the updated agent
    const updatedAgents = agents.map((agent) => (agent.id === agentId ? { ...agent, ...updates } : agent))

    setAgents(updatedAgents)

    // Update selected agent if it's the one being modified
    if (selectedAgent?.id === agentId) {
      setSelectedAgent({
        ...selectedAgent,
        ...updates,
      })
    }

    // If we're updating a conversation participant, update the active conversation too
    if (activeConversation && activeConversation.participants.includes(agentId)) {
      // We don't need to update the conversation structure, just trigger a re-render
      setActiveConversation({ ...activeConversation })
    }
  }

  // Add these missing functions after the updateAgent function:

  const sendMessage = (content: string, senderId: string) => {
    if (!activeConversation) return

    // Create a new message
    const newMessage: Message = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      content,
      senderId,
      timestamp: new Date(),
    }

    logger.debug("Sending new message", {
      senderId,
      contentPreview: content.substring(0, 30) + "...",
      messageId: newMessage.id,
    })

    // If we're currently initiating a conversation, add to pending messages
    if (isInitiatingConversationRef.current) {
      logger.debug("Currently initiating a conversation, adding message to pending queue")
      pendingMessagesRef.current.push(newMessage)
      return
    }

    // Add message to conversation
    setActiveConversation((prev) => {
      if (!prev) return null
      return {
        ...prev,
        messages: [...prev.messages, newMessage],
      }
    })
  }

  const endConversation = () => {
    if (!activeConversation) return

    // Add completed conversation to history
    const completedConversation = {
      ...activeConversation,
      endTime: new Date(),
    }
    setConversationHistory((prev) => [...prev, completedConversation])

    // Set agents as no longer in conversation
    setAgents((prevAgents) =>
      prevAgents.map((agent) =>
        activeConversation.participants.includes(agent.id) ? { ...agent, inConversation: false } : agent,
      ),
    )

    // Clear active conversation
    setActiveConversation(null)

    // Reset conversation starter attempts
    conversationStarterAttemptsRef.current = {}
  }

  // Add the panel resizing functions
  const handleResizeStart = (index: number) => {
    setResizing(index)
  }

  const handleResizeMove = (e: React.MouseEvent) => {
    if (resizing === null || !mainRef.current) return

    const containerWidth = mainRef.current.clientWidth
    const mouseX = e.clientX - mainRef.current.getBoundingClientRect().left
    const newPosition = (mouseX / containerWidth) * 100

    setPanelWidths((prev) => {
      // Ensure minimum width of panels
      const minWidth = 15 // Minimum percentage width
      const maxWidth = 70 // Maximum percentage width

      // Calculate new widths
      const newWidths = [...prev]
      const leftPanel = resizing
      const rightPanel = resizing + 1

      const leftCurrentWidth = newWidths[leftPanel]
      const rightCurrentWidth = newWidths[rightPanel]
      const totalWidth = leftCurrentWidth + rightCurrentWidth

      let leftNewWidth = newPosition - prev.slice(0, leftPanel).reduce((a, b) => a + b, 0)
      leftNewWidth = Math.max(minWidth, Math.min(maxWidth, leftNewWidth))

      const rightNewWidth = totalWidth - leftNewWidth

      if (rightNewWidth < minWidth || rightNewWidth > maxWidth) {
        return prev
      }

      newWidths[leftPanel] = leftNewWidth
      newWidths[rightPanel] = rightNewWidth

      return newWidths
    })
  }

  const handleResizeEnd = () => {
    setResizing(null)
  }

  // Add a useEffect to update autonomous conversation settings when they change
  useEffect(() => {
    if (llmClient) {
      // Get the current settings
      const settings = llmClient.getSettings()

      // Update the autonomous conversation system options
      autonomousSystemRef.current.updateOptions({
        maxAutonomousMessages: settings.maxAutonomousMessages || 4,
      })

      console.log("Updated autonomous conversation settings:", {
        maxAutonomousMessages: settings.maxAutonomousMessages || 4,
      })
    }
  }, [llmClient]) // This should run once when the component mounts

  // Handle exporting agents

  // Return the UI structure with CORRECT ORDER: Chat, Agents, Grid World, Memory Viewer
  return (
    <main
      ref={mainRef}
      className="min-h-screen flex flex-col bg-gradient-to-br from-purple-950 to-indigo-950"
      onMouseMove={resizing !== null ? handleResizeMove : undefined}
      onMouseUp={resizing !== null ? handleResizeEnd : undefined}
      onMouseLeave={resizing !== null ? handleResizeEnd : undefined}
    >
      <div className="flex-1 flex">
        {/* Chat Window Panel - FIRST */}
        <div className="border-r border-purple-800" style={{ width: `${panelWidths[0]}%` }}>
          <ChatWindow
            conversation={activeConversation}
            agents={agents}
            onSendMessage={sendMessage}
            onEndConversation={endConversation}
          />
        </div>

        {/* Resizer */}
        <div
          className="w-1 bg-purple-800 cursor-col-resize hover:bg-purple-600"
          onMouseDown={() => handleResizeStart(0)}
        />

        {/* Agent List Panel - SECOND */}
        <div className="border-r border-purple-800" style={{ width: `${panelWidths[1]}%` }}>
          <AgentList
            agents={agents}
            selectedAgent={selectedAgent}
            onSelectAgent={setSelectedAgent}
            onCreateAgent={createAgent}
            onCreateAgentWithName={createAgentWithName}
            onDeleteAgent={deleteAgent}
            onAddToConversation={addAgentToConversation}
            onRemoveFromConversation={removeAgentFromConversation}
            onUpdateAgentColor={updateAgentColor}
            onToggleAutonomy={toggleAgentAutonomy}
            onExportAgents={handleExportAgents}
            onImportAgents={handleImportAgents}
            activeConversation={!!activeConversation}
            llmSettings={llmClient?.getSettings()}
          />
        </div>

        {/* Resizer */}
        <div
          className="w-1 bg-purple-800 cursor-col-resize hover:bg-purple-600"
          onMouseDown={() => handleResizeStart(1)}
        />

        {/* Grid World Panel - THIRD */}
        <div className="border-r border-purple-800" style={{ width: `${panelWidths[2]}%` }}>
          <GridWorld
            agents={agents}
            onUpdatePosition={updateAgentPosition}
            activeConversation={activeConversation}
            onSelectKnowledgeNode={(type, id, title) => setSelectedKnowledgeNode({ type, id, title })}
            onShowAbout={() => setShowAboutModal(true)}
          />
        </div>

        {/* Resizer */}
        <div
          className="w-1 bg-purple-800 cursor-col-resize hover:bg-purple-600"
          onMouseDown={() => handleResizeStart(2)}
        />

        {/* Memory Viewer Panel - FOURTH */}
        <div style={{ width: `${panelWidths[3]}%` }}>
          <MemoryViewer
            selectedAgent={selectedAgent}
            conversationHistory={conversationHistory}
            onAddKnowledge={addKnowledgeToAgent}
            onUpdateAgent={updateAgent}
            onDeleteKnowledge={deleteKnowledgeFromAgent}
            onUpdateKnowledge={updateKnowledgeInAgent}
            agents={agents}
            selectedKnowledgeNode={selectedKnowledgeNode}
            onClearSelectedKnowledgeNode={() => setSelectedKnowledgeNode(null)}
            onSelectAgent={setSelectedAgent}
          />
        </div>
      </div>

      {/* Autonomous Conversation Manager - hidden component */}
      {activeConversation?.isAutonomous && (
        <AutonomousConversationManager conversation={activeConversation} agents={agents} onSendMessage={sendMessage} />
      )}

      {/* About Modal */}
      <AboutModal isOpen={showAboutModal} onClose={() => setShowAboutModal(false)} />
    </main>
  )
}
