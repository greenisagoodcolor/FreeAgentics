"use client"

import { useState, useEffect, useRef } from "react"
import type { Agent, Conversation } from "@/lib/types"
import { AutonomousConversationSystem, type TriggerType } from "@/lib/autonomous-conversation"

interface UseAutonomousConversationsOptions {
  checkInterval?: number
  onConversationStart?: (conversation: Conversation) => void
  onConversationEnd?: (conversation: Conversation) => void
}

export function useAutonomousConversations(
  agents: Agent[],
  activeConversation: Conversation | null,
  options: UseAutonomousConversationsOptions = {},
) {
  const { checkInterval = 5000, onConversationStart, onConversationEnd } = options

  const [isEnabled, setIsEnabled] = useState(true)
  const [lastTrigger, setLastTrigger] = useState<{
    type: TriggerType
    time: Date
    participants: string[]
    topic?: string
  } | null>(null)

  // Create refs for the autonomous system and conversation orchestrator
  const autonomousSystemRef = useRef<AutonomousConversationSystem>(new AutonomousConversationSystem())

  // Check for autonomous conversation triggers periodically
  useEffect(() => {
    if (!isEnabled || activeConversation) return

    console.log("Autonomous conversation check running, agents:", agents.length)

    // Log the autonomous agents
    const autonomousAgents = agents.filter((agent) => agent.autonomyEnabled)
    console.log(
      "Autonomous agents:",
      autonomousAgents.map((a) => a.name),
    )

    if (autonomousAgents.length < 2) {
      console.log("Not enough autonomous agents to trigger a conversation")
      return
    }

    const intervalId = setInterval(() => {
      // Check for autonomous conversation triggers
      const { shouldTrigger, participants, trigger, topic } = autonomousSystemRef.current.checkAllTriggers(agents)

      if (shouldTrigger && participants.length >= 2) {
        console.log("Autonomous conversation triggered:", {
          trigger,
          participants: participants.map((a) => a.name),
          topic,
        })

        // Create the conversation
        const newConversation = autonomousSystemRef.current.initiateConversation(participants, trigger, topic)

        if (newConversation && onConversationStart) {
          // Record the trigger
          setLastTrigger({
            type: trigger,
            time: new Date(),
            participants: participants.map((a) => a.id),
            topic,
          })

          // Notify parent component
          onConversationStart(newConversation)

          // Log the new conversation
          console.log("New autonomous conversation created:", {
            id: newConversation.id,
            participants: newConversation.participants,
            trigger,
            topic,
          })
        } else {
          console.error("Failed to create autonomous conversation")
        }
      }
    }, checkInterval)

    return () => clearInterval(intervalId)
  }, [agents, activeConversation, isEnabled, checkInterval, onConversationStart])

  // Handle ending autonomous conversations
  useEffect(() => {
    // Only apply to autonomous conversations
    if (!activeConversation?.isAutonomous) return

    // Check for maximum message count
    const checkMessageCount = () => {
      if (activeConversation && autonomousSystemRef.current.shouldEndConversation(activeConversation)) {
        console.log(`Ending autonomous conversation ${activeConversation.id} due to reaching maximum message count`)
        if (onConversationEnd) {
          onConversationEnd(activeConversation)
        }
        return true
      }
      return false
    }

    // First check if we should end immediately due to message count
    if (checkMessageCount()) return

    // Set a timeout to end the conversation after maxConversationDuration
    // This only applies to autonomous conversations
    const timeout = setTimeout(() => {
      if (!checkMessageCount() && onConversationEnd) {
        console.log(`Ending autonomous conversation ${activeConversation.id} due to reaching maximum duration`)
        onConversationEnd(activeConversation)
      }
    }, autonomousSystemRef.current.options.maxConversationDuration)

    return () => clearTimeout(timeout)
  }, [activeConversation, onConversationEnd])

  // Function to manually trigger an autonomous conversation
  const triggerConversation = (
    participantIds: string[],
    triggerType: TriggerType = "user_initiated",
    topic?: string,
  ): Conversation | null => {
    // Find the agent objects for the given IDs
    const participants = agents.filter((agent) => participantIds.includes(agent.id))

    if (participants.length < 2) {
      console.error("Cannot trigger conversation: Need at least 2 participants")
      return null
    }

    // Create the conversation
    const newConversation = autonomousSystemRef.current.initiateConversation(participants, triggerType, topic)

    if (newConversation && onConversationStart) {
      // Record the trigger
      setLastTrigger({
        type: triggerType,
        time: new Date(),
        participants: participants.map((a) => a.id),
        topic,
      })

      // Notify parent component
      onConversationStart(newConversation)
    }

    return newConversation
  }

  // Function to update autonomous system options
  const updateOptions = (newOptions: Partial<typeof autonomousSystemRef.current.options>) => {
    autonomousSystemRef.current.updateOptions(newOptions)
  }

  return {
    isEnabled,
    setIsEnabled,
    lastTrigger,
    triggerConversation,
    updateOptions,
    autonomousSystem: autonomousSystemRef.current,
  }
}
