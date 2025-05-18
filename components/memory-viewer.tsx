"use client"

import type React from "react"

import { useState, useEffect, useMemo, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Save, Trash, Edit, ArrowLeft, Search, X } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { formatTimestamp, extractTagsFromMarkdown } from "@/lib/utils"
import type { Agent, Conversation, KnowledgeEntry } from "@/lib/types"
import { useToast } from "@/hooks/use-toast"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { useLLM } from "@/contexts/llm-context"
import { type ExtractedBelief, type RefinedBelief, parseBeliefs, parseRefinedBeliefs } from "@/lib/belief-extraction"
import { exportAgentKnowledge } from "@/lib/knowledge-export"
import { debugLog } from "@/lib/debug-logger"

// Custom styles for the conversation history scrollbar
const conversationHistoryScrollbarStyles = `
  .conversation-history-scrollbar [data-radix-scroll-area-scrollbar] {
    width: 12px !important;
    padding: 0 2px !important;
  }
  
  .conversation-history-scrollbar [data-radix-scroll-area-thumb] {
    background-color: rgba(139, 92, 246, 0.7) !important;
    width: 8px !important;
    border-radius: 10px !important;
  }
  
  .conversation-history-scrollbar [data-radix-scroll-area-scrollbar]:hover [data-radix-scroll-area-thumb] {
    background-color: rgba(139, 92, 246, 0.9) !important;
    width: 8px !important;
  }
  
  .conversation-history-scrollbar [data-radix-scroll-area-scrollbar][data-orientation="vertical"] {
    display: block !important;
    opacity: 1 !important;
  }
`

// Define the AgentToolPermissions type
export type AgentToolPermissions = {
  // Information Access Tools
  internetSearch: boolean
  webScraping: boolean
  wikipediaAccess: boolean
  newsApi: boolean
  academicSearch: boolean
  documentRetrieval: boolean

  // Content Generation & Processing
  imageGeneration: boolean
  textSummarization: boolean
  translation: boolean
  codeExecution: boolean

  // Knowledge & Reasoning Tools
  calculator: boolean
  knowledgeGraphQuery: boolean
  factChecking: boolean
  timelineGenerator: boolean

  // External Integrations
  weatherData: boolean
  mapLocationData: boolean
  financialData: boolean
  publicDatasets: boolean

  // Agent-Specific Tools
  memorySearch: boolean
  crossAgentKnowledge: boolean
  conversationAnalysis: boolean
}

// Update the MemoryViewerProps interface to include the toolPermissions update
interface MemoryViewerProps {
  selectedAgent: Agent | null
  conversationHistory: Conversation[]
  onAddKnowledge: (agentId: string, knowledge: KnowledgeEntry) => void
  onUpdateAgent: (
    agentId: string,
    updates: { name?: string; biography?: string; toolPermissions?: AgentToolPermissions },
  ) => void
  onDeleteKnowledge: (agentId: string, knowledgeId: string) => void
  onUpdateKnowledge: (agentId: string, knowledgeId: string, updates: Partial<KnowledgeEntry>) => void
  agents: Agent[]
  selectedKnowledgeNode?: {
    type: "entry" | "tag"
    id: string
    title: string
  } | null
  onClearSelectedKnowledgeNode?: () => void
  onSelectAgent?: (agent: Agent) => void
}

// Add default tool permissions
const defaultToolPermissions: AgentToolPermissions = {
  // Information Access Tools
  internetSearch: false,
  webScraping: false,
  wikipediaAccess: false,
  newsApi: false,
  academicSearch: false,
  documentRetrieval: false,

  // Content Generation & Processing
  imageGeneration: false,
  textSummarization: false,
  translation: false,
  codeExecution: false,

  // Knowledge & Reasoning Tools
  calculator: false,
  knowledgeGraphQuery: false,
  factChecking: false,
  timelineGenerator: false,

  // External Integrations
  weatherData: false,
  mapLocationData: false,
  financialData: false,
  publicDatasets: false,

  // Agent-Specific Tools
  memorySearch: false,
  crossAgentKnowledge: false,
  conversationAnalysis: false,
}

export default function MemoryViewer({
  selectedAgent,
  conversationHistory,
  onAddKnowledge,
  onUpdateAgent,
  onDeleteKnowledge,
  onUpdateKnowledge,
  agents,
  selectedKnowledgeNode = null,
  onClearSelectedKnowledgeNode = () => {},
  onSelectAgent,
}: MemoryViewerProps) {
  const [biography, setBiography] = useState<string>("")
  const [selectedView, setSelectedView] = useState<string>("biography")
  const { toast } = useToast()

  // Get LLM context
  const llmContext = useLLM()
  const { isProcessing, setIsProcessing } = llmContext

  // Knowledge state
  const [knowledgeTab, setKnowledgeTab] = useState<string>("browse")
  const [newKnowledgeTitle, setNewKnowledgeTitle] = useState<string>("")
  const [newKnowledgeContent, setNewKnowledgeContent] = useState<string>("")
  const [selectedKnowledge, setSelectedKnowledge] = useState<KnowledgeEntry | null>(null)
  const [editingKnowledge, setEditingKnowledge] = useState<boolean>(false)
  const [beliefsPrompt, setBeliefsPrompt] = useState<string>(
    "Extract factual [[knowledge]], user [[preferences]], and [[research-relevant]] information. Focus on substantive content that would help with research projects and future conversations.",
  )
  const [editedKnowledgeContent, setEditedKnowledgeContent] = useState<string>("")
  const [editedKnowledgeTitle, setEditedKnowledgeTitle] = useState<string>("")

  // Knowledge search and filter state
  const [searchQuery, setSearchQuery] = useState<string>("")
  const [selectedTag, setSelectedTag] = useState<string>("all_tags")
  const [sortBy, setSortBy] = useState<"newest" | "oldest" | "title">("newest")

  // Delete confirmation dialog
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false)
  const [knowledgeToDelete, setKnowledgeToDelete] = useState<KnowledgeEntry | null>(null)

  // System prompt state
  const [systemPrompt, setSystemPrompt] = useState<string>("")
  const [systemPromptName, setSystemPromptName] = useState<string>("Default")

  // Belief extraction state
  const [extractedBeliefs, setExtractedBeliefs] = useState<RefinedBelief[]>([])
  const [rawBeliefs, setRawBeliefs] = useState<ExtractedBelief[]>([])
  const [isExtractingBeliefs, setIsExtractingBeliefs] = useState(false)
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null)
  const [extractionStep, setExtractionStep] = useState<"idle" | "extracting" | "refining" | "complete">("idle")
  const [extractionProgress, setExtractionProgress] = useState(0)
  const [inferenceTab, setInferenceTab] = useState<"prompt" | "results" | "preview">("prompt")

  // Add these state variables to the MemoryViewer component, near the other state declarations
  const [toolPermissions, setToolPermissions] = useState<AgentToolPermissions>(defaultToolPermissions)
  const [isSavingTools, setIsSavingTools] = useState<boolean>(false)
  const [hasToolChanges, setHasToolChanges] = useState<boolean>(false)

  // Add a ref to track pending knowledge selection after agent change
  const pendingKnowledgeSelectionRef = useRef<KnowledgeEntry | null>(null)
  const previousAgentIdRef = useRef<string | null>(null)

  // Update biography state when selected agent changes
  useEffect(() => {
    if (selectedAgent) {
      setBiography(selectedAgent.biography)

      // Only reset selectedKnowledge if we don't have a pending selection
      // and if the agent has actually changed
      if (!pendingKnowledgeSelectionRef.current && previousAgentIdRef.current !== selectedAgent.id) {
        setSelectedKnowledge(null)
      }

      setEditingKnowledge(false)

      // Reset search and filter when changing agents
      setSearchQuery("")
      setSelectedTag("all_tags")

      // Initialize tool permissions with agent's existing permissions or defaults
      setToolPermissions(selectedAgent.toolPermissions || defaultToolPermissions)
      setHasToolChanges(false)

      // Update the previous agent id ref
      previousAgentIdRef.current = selectedAgent.id
    }
  }, [selectedAgent])

  // Add a new useEffect to handle pending knowledge selection
  useEffect(() => {
    // If we have a pending knowledge selection and a selected agent
    if (pendingKnowledgeSelectionRef.current && selectedAgent) {
      // Find the matching knowledge entry in the selected agent's knowledge
      const matchingEntry = selectedAgent.knowledge.find(
        (entry) => entry.id === pendingKnowledgeSelectionRef.current?.id,
      )

      // If we found a matching entry, select it
      if (matchingEntry) {
        setSelectedKnowledge(matchingEntry)
      }
      // If we didn't find a matching entry but have a title, try to find by title
      else if (pendingKnowledgeSelectionRef.current.title) {
        const entryByTitle = selectedAgent.knowledge.find(
          (entry) => entry.title === pendingKnowledgeSelectionRef.current?.title,
        )

        if (entryByTitle) {
          setSelectedKnowledge(entryByTitle)
        }
      }

      // Clear the pending selection
      pendingKnowledgeSelectionRef.current = null
    }
  }, [selectedAgent])

  // Update edited knowledge content when selected knowledge changes
  useEffect(() => {
    if (selectedKnowledge) {
      setEditedKnowledgeContent(selectedKnowledge.content)
      setEditedKnowledgeTitle(selectedKnowledge.title)
    }
  }, [selectedKnowledge])

  // When a knowledge node is selected from the global graph, switch to the node selection view
  useEffect(() => {
    if (selectedKnowledgeNode) {
      setSelectedView("node-selection")
    }
  }, [selectedKnowledgeNode])

  // Get all unique tags from the selected agent's knowledge
  const uniqueTags = useMemo(() => {
    if (!selectedAgent) return []

    const tags = new Set<string>()
    selectedAgent.knowledge.forEach((entry) => {
      entry.tags.forEach((tag) => tags.add(tag))
    })

    return Array.from(tags).sort()
  }, [selectedAgent])

  // Filter and sort knowledge entries based on search, tag, and sort criteria
  const filteredKnowledge = useMemo(() => {
    if (!selectedAgent) return []

    let filtered = [...selectedAgent.knowledge]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (entry) => entry.title.toLowerCase().includes(query) || entry.content.toLowerCase().includes(query),
      )
    }

    // Apply tag filter
    if (selectedTag && selectedTag !== "all_tags") {
      filtered = filtered.filter((entry) => entry.tags.includes(selectedTag))
    }

    // Apply sorting
    switch (sortBy) {
      case "newest":
        filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        break
      case "oldest":
        filtered.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
        break
      case "title":
        filtered.sort((a, b) => a.title.localeCompare(b.title))
        break
    }

    return filtered
  }, [selectedAgent, searchQuery, selectedTag, sortBy])

  // Get relevant conversations for the selected agent
  const relevantConversations = useMemo(() => {
    if (!selectedAgent) return []

    return conversationHistory
      .filter((conv) => conv.participants.includes(selectedAgent.id))
      .sort((a, b) => b.startTime.getTime() - a.startTime.getTime())
  }, [selectedAgent, conversationHistory])

  const handleSaveBiography = () => {
    if (!selectedAgent) return
    onUpdateAgent(selectedAgent.id, { biography })
    toast({
      title: "Biography updated",
      description: `${selectedAgent.name}'s biography has been updated.`,
      duration: 3000,
    })
  }

  const handleAddKnowledge = () => {
    if (!selectedAgent || !newKnowledgeTitle.trim() || !newKnowledgeContent.trim()) return

    const newKnowledge: KnowledgeEntry = {
      id: `knowledge-${Date.now()}`,
      title: newKnowledgeTitle,
      content: newKnowledgeContent,
      timestamp: new Date(),
      tags: extractTagsFromMarkdown(newKnowledgeContent),
    }

    onAddKnowledge(selectedAgent.id, newKnowledge)
    setNewKnowledgeTitle("")
    setNewKnowledgeContent("")

    toast({
      title: "Knowledge added",
      description: `"${newKnowledgeTitle}" has been added to ${selectedAgent.name}'s knowledge.`,
      duration: 3000,
    })
  }

  // Handle belief extraction with progress tracking
  const handleUpdateBeliefs = async () => {
    if (!selectedAgent) {
      toast({
        title: "No agent selected",
        description: "Please select an agent first",
        variant: "destructive",
        duration: 3000,
      })
      return
    }

    // Get the selected conversation or the most recent one
    const targetConversation = selectedConversationId
      ? conversationHistory.find((conv) => conv.id === selectedConversationId)
      : conversationHistory
          .filter((conv) => conv.participants.includes(selectedAgent.id))
          .sort((a, b) => b.startTime.getTime() - a.startTime.getTime())[0]

    if (!targetConversation) {
      toast({
        title: "No conversation available",
        description: "There are no conversations for this agent to analyze",
        variant: "destructive",
        duration: 3000,
      })
      return
    }

    setIsExtractingBeliefs(true)
    setExtractionStep("extracting")
    setExtractionProgress(10)
    setInferenceTab("results")

    try {
      // Format the conversation for analysis
      const conversationText = targetConversation.messages
        .map((msg) => {
          const senderName =
            msg.senderId === "user" ? "User" : agents.find((a) => a.id === msg.senderId)?.name || "Unknown"
          return `${senderName}: ${msg.content}`
        })
        .join("\n\n")

      debugLog("Formatted conversation for belief extraction:", conversationText)

      // FIXED: Use the client's extractBeliefs method directly instead of calling the function
      // with settings that don't contain the API key
      setExtractionProgress(30)
      try {
        // Call the client's extractBeliefs method which handles API key retrieval
        const rawBeliefsResponse = await llmContext.client.extractBeliefs(
          conversationText,
          selectedAgent.name,
          beliefsPrompt,
        )

        // Parse the response - the client returns a string, but we need to parse it into beliefs
        const rawBeliefs = parseBeliefs(rawBeliefsResponse)
        debugLog("Raw beliefs extracted:", rawBeliefs)

        setRawBeliefs(rawBeliefs)
        setExtractionProgress(60)
        setExtractionStep("refining")

        // Now refine the beliefs using the client's method
        try {
          // Get the existing knowledge to check for duplicates
          const existingKnowledge = selectedAgent.knowledge

          // Call the refine method on the client
          const refinedResponse = await llmContext.client.generateResponse(
            // System prompt for refinement
            `You are an AI assistant that refines and enhances extracted beliefs for a knowledge base.
Your task is to analyze each belief, rate its accuracy and relevance, categorize it, and suggest a title.

IMPORTANT: Maintain the Obsidian-style markdown format with [[double brackets]] around key concepts.`,
            // User prompt with the raw beliefs
            `Below are beliefs extracted from a conversation. 
Refine these beliefs according to these priorities: ${beliefsPrompt}

EXTRACTED BELIEFS:
${rawBeliefs.map((belief, index) => `${index + 1}. ${belief.content} (${belief.confidence})`).join("\n")}

PRIORITIZE:
1. Factual knowledge about topics discussed (not about the agents themselves)
2. User preferences and research goals
3. Information that would be valuable for future reference

DEPRIORITIZE OR REMOVE:
1. Observations about the agent's behavior or willingness to help
2. Generic statements without specific information
3. Low-value or redundant information

For each belief, provide:
1. Accuracy (1-5 scale, where 5 is highest)
2. Relevance (1-5 scale, where 5 is highest) - rate higher for factual knowledge and user preferences
3. Category (Fact, Opinion, Preference, Relationship, or Other)
4. A concise title for the knowledge entry
5. Refined content (maintain or enhance the Obsidian-style [[tags]])
6. Set "selected": false for any beliefs that are about the agent itself rather than substantive knowledge

Format your response as a JSON array with one object per belief:
[
 {
   "originalIndex": 1,
   "accuracy": 4,
   "relevance": 5,
   "category": "Fact",
   "title": "Example Title",
   "refined_content": "Refined belief with [[tags]]",
   "confidence": "High",
   "selected": true
 },
 ...
]`,
          )

          // Parse the refined beliefs from the response
          const refined = parseRefinedBeliefs(refinedResponse, rawBeliefs)
          debugLog("Refined beliefs:", refined)

          if (refined && refined.length > 0) {
            setExtractedBeliefs(refined)
            setExtractionProgress(100)
            setExtractionStep("complete")

            toast({
              title: "Beliefs extracted",
              description: `Found ${refined.length} potential new beliefs for ${selectedAgent.name}`,
              duration: 3000,
            })
          } else {
            throw new Error("No beliefs could be extracted from this conversation")
          }
        } catch (refineError) {
          console.error("Error refining beliefs:", refineError)
          toast({
            title: "Error refining beliefs",
            description: "The system encountered an error while processing the extracted beliefs. Please try again.",
            variant: "destructive",
            duration: 5000,
          })
          setExtractionStep("idle")
        }
      } catch (extractError) {
        console.error("Error extracting raw beliefs:", extractError)
        toast({
          title: "Error extracting beliefs",
          description: "The system encountered an error while analyzing the conversation. Please try again.",
          variant: "destructive",
          duration: 5000,
        })
        setExtractionStep("idle")
      }
    } catch (error) {
      console.error("Error in belief extraction process:", error)
      toast({
        title: "Error extracting beliefs",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
        duration: 5000,
      })
      setExtractionStep("idle")
    } finally {
      setIsExtractingBeliefs(false)
    }
  }

  // Handle toggling belief selection
  const handleToggleBelief = (index: number) => {
    setExtractedBeliefs((prev) =>
      prev.map((belief, i) =>
        i === index ? { ...belief, selected: belief.selected === false ? true : false } : belief,
      ),
    )
  }

  // Handle adding selected beliefs to knowledge
  const handleAddSelectedBeliefs = async () => {
    if (!selectedAgent) return

    const selectedBeliefs = extractedBeliefs.filter((belief) => belief.selected !== false)
    console.log(`Selected beliefs count: ${selectedBeliefs.length}`, selectedBeliefs)

    if (selectedBeliefs.length === 0) {
      toast({
        title: "No beliefs selected",
        description: "Please select at least one belief to add to knowledge",
        variant: "destructive",
        duration: 3000,
      })
      return
    }

    try {
      // Create knowledge entries from selected beliefs
      const knowledgeEntries = selectedBeliefs.map((belief) => ({
        id: `knowledge-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        title: belief.title,
        content: belief.refined_content,
        timestamp: new Date(),
        tags: belief.tags,
      }))

      console.log(`Adding ${knowledgeEntries.length} knowledge entries to agent's knowledge`)

      // Add each entry to the agent's knowledge
      for (const entry of knowledgeEntries) {
        console.log(`Adding entry: ${entry.title}`)
        onAddKnowledge(selectedAgent.id, entry)
      }

      // Clear extracted beliefs
      setExtractedBeliefs([])
      setRawBeliefs([])
      setExtractionStep("idle")

      toast({
        title: "Knowledge updated",
        description: `Added ${knowledgeEntries.length} new knowledge entries to ${selectedAgent.name}'s knowledge`,
        duration: 3000,
      })

      // Switch to browse tab
      setKnowledgeTab("browse")
    } catch (error) {
      console.error("Error adding beliefs to knowledge:", error)
      toast({
        title: "Error adding beliefs",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
        duration: 5000,
      })
    }
  }

  const handleSaveKnowledgeChanges = () => {
    if (!selectedAgent || !selectedKnowledge) return

    // Extract tags from the updated content
    const updatedTags = extractTagsFromMarkdown(editedKnowledgeContent)

    // Create the updates object
    const updates: Partial<KnowledgeEntry> = {
      title: editedKnowledgeTitle,
      content: editedKnowledgeContent,
      tags: updatedTags,
    }

    // Call the update function
    onUpdateKnowledge(selectedAgent.id, selectedKnowledge.id, updates)

    // Update the local state
    setSelectedKnowledge({
      ...selectedKnowledge,
      ...updates,
    })

    // Exit editing mode
    setEditingKnowledge(false)

    toast({
      title: "Knowledge updated",
      description: "Knowledge entry has been updated successfully.",
      duration: 3000,
    })
  }

  const handleDeleteKnowledge = () => {
    if (!selectedAgent || !knowledgeToDelete) return

    // Call the delete function
    onDeleteKnowledge(selectedAgent.id, knowledgeToDelete.id)

    // Clear the selected knowledge if it's the one being deleted
    if (selectedKnowledge?.id === knowledgeToDelete.id) {
      setSelectedKnowledge(null)
    }

    // Close the dialog
    setIsDeleteDialogOpen(false)
    setKnowledgeToDelete(null)

    toast({
      title: "Knowledge deleted",
      description: `"${knowledgeToDelete.title}" has been deleted from ${selectedAgent.name}'s knowledge.`,
      duration: 3000,
    })
  }

  const handleSelectAgentForKnowledge = (agent: Agent) => {
    // Find the specific knowledge entry if we're looking at an entry
    if (selectedKnowledgeNode?.type === "entry") {
      // For consolidated entries, check by title
      const entry = agent.knowledge.find((k) => k.title === selectedKnowledgeNode.title)
      if (entry) {
        // Store the entry we want to select in the ref
        pendingKnowledgeSelectionRef.current = entry

        // Select the agent if it's not already selected
        if (onSelectAgent && selectedAgent?.id !== agent.id) {
          onSelectAgent(agent)
        } else if (selectedAgent?.id === agent.id) {
          // If the agent is already selected, we can set the knowledge directly
          setSelectedKnowledge(entry)
        }

        setKnowledgeTab("browse")
        setSelectedView("knowledge")
        onClearSelectedKnowledgeNode()
      }
    }
    // If we're looking at a tag, switch to the knowledge view with that tag
    else if (selectedKnowledgeNode?.type === "tag") {
      setSelectedView("knowledge")
      setKnowledgeTab("browse")
      setSelectedTag(selectedKnowledgeNode.title)
      onClearSelectedKnowledgeNode()

      // Select the agent if it's not already selected
      if (onSelectAgent && selectedAgent?.id !== agent.id) {
        onSelectAgent(agent)
      }
    }
  }

  // Add a new function to handle clicking on a specific knowledge entry
  const handleSelectKnowledgeEntry = (agent: Agent, entry: KnowledgeEntry, event: React.MouseEvent) => {
    // Prevent the click from propagating to the agent card
    event.stopPropagation()

    // Store the entry we want to select in the ref
    pendingKnowledgeSelectionRef.current = entry

    // Select the agent using the onSelectAgent prop
    if (onSelectAgent && selectedAgent?.id !== agent.id) {
      onSelectAgent(agent)
    } else if (selectedAgent?.id === agent.id) {
      // If the agent is already selected, we can set the knowledge directly
      setSelectedKnowledge(entry)
    }

    // Switch to the knowledge view
    setKnowledgeTab("browse")
    setSelectedView("knowledge")

    // Clear the selected knowledge node
    onClearSelectedKnowledgeNode()
  }

  const getAgentsWithSelectedNode = () => {
    if (!selectedKnowledgeNode) return []

    return agents.filter((agent) => {
      if (selectedKnowledgeNode.type === "entry") {
        // For consolidated entries, check by title
        return agent.knowledge.some((entry) => entry.title === selectedKnowledgeNode.title)
      } else if (selectedKnowledgeNode.type === "tag") {
        return agent.knowledge.some((entry) => entry.tags.includes(selectedKnowledgeNode.title))
      }
      return false
    })
  }

  // Get knowledge entries that match the selected tag
  const getEntriesWithTag = (agent: Agent, tag: string) => {
    return agent.knowledge.filter((entry) => entry.tags.includes(tag))
  }

  // Get knowledge entries that match the selected title
  const getEntriesWithTitle = (agent: Agent, title: string) => {
    return agent.knowledge.filter((entry) => entry.title === title)
  }

  // Clear all search and filter criteria
  const clearFilters = () => {
    setSearchQuery("")
    setSelectedTag("all_tags")
    setSortBy("newest")
  }

  // Helper function to render markdown with highlighted tags
  const renderMarkdownWithTags = (content: string) => {
    // Replace [[tags]] with highlighted spans that have a data-tag attribute
    return content.replace(
      /\[\[(.*?)\]\]/g,
      '<span class="bg-purple-500/20 text-purple-900 px-1 rounded cursor-pointer hover:bg-purple-500/30 transition-colors" data-tag="$1">[[<span class="font-medium">$1</span>]]</span>',
    )
  }

  // Handle saving system prompt
  const handleSaveSystemPrompt = () => {
    if (!selectedAgent || !systemPrompt.trim()) {
      toast({
        title: "Cannot save system prompt",
        description: "Agent or prompt not available",
        variant: "destructive",
        duration: 3000,
      })
      return
    }

    setIsProcessing(true)

    // Simulate a delay
    setTimeout(() => {
      try {
        toast({
          title: "System prompt saved",
          description: `System prompt "${systemPromptName}" has been saved for ${selectedAgent.name}.`,
          duration: 3000,
        })
      } catch (error) {
        console.error("Error in handleSaveSystemPrompt:", error)
        toast({
          title: "Error saving system prompt",
          description: error instanceof Error ? error.message : "An unknown error occurred",
          variant: "destructive",
          duration: 5000,
        })
      } finally {
        setIsProcessing(false)
      }
    }, 1000)
  }

  // Add this function inside the MemoryViewer component
  const handleTagClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Check if the clicked element or its parent has a data-tag attribute
    const target = e.target as HTMLElement
    const tagElement = target.closest("[data-tag]")

    if (tagElement) {
      const tag = tagElement.getAttribute("data-tag")
      if (tag) {
        setSelectedTag(tag)
        setSelectedKnowledge(null)
        setKnowledgeTab("browse")
      }
    }
  }

  // New function to extract knowledge from conversations
  const handleExtractKnowledge = async (conversation) => {
    try {
      setIsProcessing(true)

      // Get the latest settings to ensure we have the API key
      const currentSettings = llmContext.client.getSettings()

      // Log the settings to verify API key is present
      console.log("Knowledge extraction settings:", {
        provider: currentSettings.provider,
        model: currentSettings.model,
        hasApiKey: !!currentSettings.apiKey,
        apiKeyLength: currentSettings.apiKey ? currentSettings.apiKey.length : 0,
      })

      // Make sure we have an API key before proceeding
      if (!currentSettings.apiKey) {
        throw new Error(`API key is required for ${currentSettings.provider} provider during knowledge extraction`)
      }

      // Proceed with knowledge extraction using the complete settings
      // ...
    } catch (error) {
      console.error("Error extracting knowledge:", error)
      toast({
        title: "Knowledge Extraction Failed",
        description: error.message || "Failed to extract knowledge from conversation",
        variant: "destructive",
        duration: 5000,
      })
    } finally {
      setIsProcessing(false)
    }
  }

  // Add these handler functions for tool permissions
  const handleToolChange = (toolKey: keyof AgentToolPermissions, checked: boolean) => {
    setToolPermissions((prev) => {
      const updated = { ...prev, [toolKey]: checked }
      // Mark that we have unsaved changes
      setHasToolChanges(true)
      return updated
    })
  }

  const handleSaveToolSettings = () => {
    if (!selectedAgent) return

    setIsSavingTools(true)

    // Update the agent with new tool permissions
    onUpdateAgent(selectedAgent.id, { toolPermissions })

    // Reset the changes flag
    setHasToolChanges(false)

    // Show success message
    toast({
      title: "Tool settings saved",
      description: `Tool permissions for ${selectedAgent.name} have been updated.`,
      duration: 3000,
    })

    setTimeout(() => {
      setIsSavingTools(false)
    }, 500)
  }

  // Add the custom scrollbar styles for conversation history
  useEffect(() => {
    // Add the styles to the document head
    const styleElement = document.createElement("style")
    styleElement.innerHTML = conversationHistoryScrollbarStyles
    document.head.appendChild(styleElement)

    // Clean up on unmount
    return () => {
      document.head.removeChild(styleElement)
    }
  }, [])

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-border">
        <h2 className="text-xl font-bold text-white">Memory Viewer</h2>
      </div>

      {selectedKnowledgeNode && selectedView === "node-selection" ? (
        <div className="flex-1 flex flex-col">
          <div className="px-4 pt-4 flex items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                onClearSelectedKnowledgeNode()
                setSelectedView("biography")
              }}
              className="mr-2 bg-purple-900/30 text-white hover:bg-purple-800/50 hover:text-white"
            >
              <ArrowLeft size={16} className="mr-1" />
              Back
            </Button>
            <h3 className="text-lg font-semibold text-white">
              {selectedKnowledgeNode.type === "entry" ? "Knowledge Entry" : "Tag"}: {selectedKnowledgeNode.title}
            </h3>
          </div>

          <div className="flex-1 p-4">
            <Card className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  Agents with{" "}
                  {selectedKnowledgeNode.type === "entry" ? "this knowledge" : `"${selectedKnowledgeNode.title}" tag`}
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[calc(100%-60px)]">
                <ScrollArea className="h-full pr-4">
                  {getAgentsWithSelectedNode().length > 0 ? (
                    <div className="space-y-4">
                      {getAgentsWithSelectedNode().map((agent) => (
                        <div
                          key={agent.id}
                          className="p-4 border rounded-md cursor-pointer hover:bg-muted transition-colors"
                          onClick={() => handleSelectAgentForKnowledge(agent)}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: agent.color }} />
                            <h3 className="font-medium">{agent.name}</h3>
                          </div>

                          {selectedKnowledgeNode.type === "tag" ? (
                            <div className="mt-2">
                              <p className="text-sm text-muted-foreground mb-1">
                                {getEntriesWithTag(agent, selectedKnowledgeNode.title).length} entries with this tag:
                              </p>
                              <div className="space-y-1 ml-2">
                                {getEntriesWithTag(agent, selectedKnowledgeNode.title).map((entry) => (
                                  <div
                                    key={entry.id}
                                    className="text-sm py-1 px-2 rounded hover:bg-purple-800/30 cursor-pointer flex items-center"
                                    onClick={(e) => handleSelectKnowledgeEntry(agent, entry, e)}
                                  >
                                    <span className="w-1 h-1 bg-purple-400 rounded-full mr-2"></span>
                                    <span>{entry.title}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : (
                            <div className="mt-2">
                              <p className="text-sm text-muted-foreground mb-1">
                                {getEntriesWithTitle(agent, selectedKnowledgeNode.title).length} entries with this
                                title:
                              </p>
                              <div className="space-y-1 ml-2">
                                {getEntriesWithTitle(agent, selectedKnowledgeNode.title).map((entry) => (
                                  <div
                                    key={entry.id}
                                    className="text-sm py-1 px-2 rounded hover:bg-purple-800/30 cursor-pointer flex items-center"
                                    onClick={(e) => handleSelectKnowledgeEntry(agent, entry, e)}
                                  >
                                    <span className="w-1 h-1 bg-purple-400 rounded-full mr-2"></span>
                                    <span>Created: {formatTimestamp(entry.timestamp)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground">
                      No agents have this {selectedKnowledgeNode.type === "entry" ? "knowledge entry" : "tag"}.
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : selectedAgent ? (
        <div className="flex-1 flex flex-col">
          <div className="px-4 pt-4">
            <div className="flex justify-between items-center">
              <Select
                value={selectedView}
                onValueChange={(value) => {
                  setSelectedView(value)
                }}
              >
                <SelectTrigger className="w-full bg-purple-950 border-purple-700 text-white">
                  <SelectValue placeholder="Select view" />
                </SelectTrigger>
                <SelectContent className="bg-purple-950 border-purple-700 text-white">
                  <SelectItem value="biography">Biography</SelectItem>
                  <SelectItem value="conversations">Conversations</SelectItem>
                  <SelectItem value="knowledge">Knowledge</SelectItem>
                  <SelectItem value="graph">Graph</SelectItem>
                  <SelectItem value="system">System Prompt</SelectItem>
                  <SelectItem value="tools">Tools</SelectItem>
                </SelectContent>
              </Select>

              {selectedAgent && (
                <Button
                  onClick={() => exportAgentKnowledge(selectedAgent)}
                  className="ml-2 bg-purple-700 hover:bg-purple-600 text-white"
                  size="sm"
                >
                  <Save size={16} className="mr-2" />
                  Export Knowledge
                </Button>
              )}
            </div>
          </div>

          <div className="flex-1 p-4 pt-2 overflow-hidden">
            {selectedView === "biography" && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{selectedAgent.name}'s Profile</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label htmlFor="agent-biography" className="text-sm font-medium">
                        Biography
                      </label>
                      <Textarea
                        id="agent-biography"
                        value={biography}
                        onChange={(e) => setBiography(e.target.value)}
                        className="min-h-[200px]"
                        placeholder="Enter agent biography..."
                      />
                    </div>
                    <Button onClick={handleSaveBiography} className="bg-purple-700 hover:bg-purple-600 text-white">
                      <Save size={16} className="mr-2" />
                      Save Biography
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedView === "conversations" && (
              <Card className="h-full flex flex-col">
                <CardHeader className="pb-2 flex-shrink-0">
                  <CardTitle className="text-base">Conversation History</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 p-0 overflow-hidden">
                  <ScrollArea
                    className="h-full py-4 px-6 conversation-history-scrollbar"
                    style={{
                      height: "calc(100vh - 220px)",
                    }}
                  >
                    {conversationHistory.filter((conv) => conv.participants.includes(selectedAgent.id)).length > 0 ? (
                      <div className="space-y-4">
                        {/* Conversation entries remain the same */}
                        {conversationHistory
                          .filter((conv) => conv.participants.includes(selectedAgent.id))
                          .sort((a, b) => b.startTime.getTime() - a.startTime.getTime())
                          .map((conv) => (
                            <Card key={conv.id} className="p-4">
                              <div className="mb-2">
                                <div className="flex justify-between items-center">
                                  <h3 className="font-medium">Conversation {formatTimestamp(conv.startTime)}</h3>
                                  <span className="text-xs text-muted-foreground">{conv.messages.length} messages</span>
                                </div>
                                <div className="text-xs text-muted-foreground mt-1">
                                  Participants:{" "}
                                  {conv.participants
                                    .map((id) => agents.find((a) => a.id === id)?.name || "Unknown")
                                    .join(", ")}
                                </div>
                              </div>
                              <div className="border-t pt-2 mt-2">
                                <div className="max-h-32 overflow-y-auto text-sm">
                                  {conv.messages.length > 0 ? (
                                    conv.messages.map((msg) => (
                                      <div key={msg.id} className="mb-2">
                                        <div className="flex items-center gap-1">
                                          <span className="font-medium">
                                            {msg.senderId === "user"
                                              ? "You"
                                              : agents.find((a) => a.id === msg.senderId)?.name || "Unknown"}
                                            :
                                          </span>
                                          <span className="text-xs text-muted-foreground">
                                            {new Date(msg.timestamp).toLocaleTimeString()}
                                          </span>
                                        </div>
                                        <p className="text-sm">{msg.content}</p>
                                      </div>
                                    ))
                                  ) : (
                                    <p className="text-muted-foreground">No messages in this conversation.</p>
                                  )}
                                </div>
                              </div>
                            </Card>
                          ))}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full text-muted-foreground">
                        No conversation history for this agent.
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            )}

            {selectedView === "knowledge" && (
              <Card className="h-full">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Knowledge Base</CardTitle>
                </CardHeader>
                <CardContent className="h-[calc(100%-60px)] p-0">
                  <div className="h-full flex flex-col">
                    <div className="px-6 py-2 border-b">
                      <Select value={knowledgeTab} onValueChange={setKnowledgeTab}>
                        <SelectTrigger className="w-full bg-purple-950 border-purple-700 text-white">
                          <SelectValue placeholder="Select view" />
                        </SelectTrigger>
                        <SelectContent className="bg-purple-950 border-purple-700 text-white">
                          <SelectItem value="browse">Browse</SelectItem>
                          <SelectItem value="add">Add Knowledge</SelectItem>
                          <SelectItem value="inference">Inference</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex-1 overflow-hidden">
                      {knowledgeTab === "browse" && (
                        <div className="h-full p-0 m-0">
                          <div className="grid grid-cols-3 h-full">
                            <div className="col-span-1 border-r h-full">
                              <div className="p-4 border-b">
                                <div className="flex items-center gap-2 mb-2">
                                  <Search size={14} className="text-muted-foreground" />
                                  <Input
                                    placeholder="Search knowledge..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="h-8"
                                  />
                                  {(searchQuery || selectedTag || sortBy !== "newest") && (
                                    <Button variant="ghost" size="sm" onClick={clearFilters} className="h-8 w-8 p-0">
                                      <X size={14} />
                                    </Button>
                                  )}
                                </div>

                                <div className="flex flex-wrap gap-2 mb-2">
                                  {uniqueTags.length > 0 && (
                                    <Select value={selectedTag} onValueChange={setSelectedTag}>
                                      <SelectTrigger className="h-8 w-full bg-purple-950 border-purple-700 text-white">
                                        <SelectValue placeholder="Filter by tag" />
                                      </SelectTrigger>
                                      <SelectContent className="bg-purple-950 border-purple-700 text-white">
                                        <SelectItem value="all_tags">All tags</SelectItem>
                                        {uniqueTags.map((tag) => (
                                          <SelectItem key={tag} value={tag}>
                                            {tag}
                                          </SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  )}
                                </div>

                                <div className="flex justify-between items-center">
                                  <span className="text-xs text-muted-foreground">
                                    {filteredKnowledge.length} entries
                                  </span>
                                  <Select value={sortBy} onValueChange={(value) => setSortBy(value as any)}>
                                    <SelectTrigger className="h-7 text-xs w-28 bg-purple-950 border-purple-700 text-white">
                                      <SelectValue placeholder="Sort by" />
                                    </SelectTrigger>
                                    <SelectContent className="bg-purple-950 border-purple-700 text-white">
                                      <SelectItem value="newest">Newest</SelectItem>
                                      <SelectItem value="oldest">Oldest</SelectItem>
                                      <SelectItem value="title">Title</SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                              </div>

                              <ScrollArea
                                className="p-4"
                                type="always"
                                style={{
                                  height: "calc(100vh - 300px)",
                                  maxHeight: "calc(100vh - 300px)",
                                  overflow: "hidden",
                                }}
                              >
                                {filteredKnowledge.length > 0 ? (
                                  <div className="space-y-2">
                                    {filteredKnowledge.map((k) => (
                                      <div
                                        key={k.id}
                                        className={`p-2 rounded border cursor-pointer ${
                                          selectedKnowledge?.id === k.id
                                            ? "border-primary bg-primary/10"
                                            : "border-border hover:bg-muted"
                                        }`}
                                        onClick={() => {
                                          setSelectedKnowledge(k)
                                          setEditingKnowledge(false)
                                        }}
                                      >
                                        <h4 className="font-medium text-sm truncate">{k.title}</h4>
                                        <p className="text-xs text-muted-foreground mt-1">
                                          {formatTimestamp(k.timestamp)}
                                        </p>
                                        {k.tags.length > 0 && (
                                          <div className="flex flex-wrap gap-1 mt-1">
                                            {k.tags.slice(0, 2).map((tag) => (
                                              <Badge key={tag} variant="secondary" className="text-xs py-0 h-5">
                                                {tag}
                                              </Badge>
                                            ))}
                                            {k.tags.length > 2 && (
                                              <Badge variant="outline" className="text-xs py-0 h-5">
                                                +{k.tags.length - 2}
                                              </Badge>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="text-center text-muted-foreground py-8">
                                    {selectedAgent.knowledge.length > 0
                                      ? "No matching knowledge entries found."
                                      : "No knowledge entries yet."}
                                  </div>
                                )}
                              </ScrollArea>
                            </div>

                            <div className="col-span-2 h-full">
                              <ScrollArea
                                className="p-4"
                                type="always"
                                style={{
                                  height: "calc(100vh - 300px)",
                                  maxHeight: "calc(100vh - 300px)",
                                  overflow: "hidden",
                                }}
                              >
                                {selectedKnowledge ? (
                                  <div className="flex flex-col">
                                    <div className="flex justify-between items-center mb-2">
                                      <h3 className="font-medium">{selectedKnowledge.title}</h3>
                                      <div className="flex gap-2">
                                        <Button
                                          variant="outline"
                                          size="sm"
                                          onClick={() => setEditingKnowledge(!editingKnowledge)}
                                          className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                                        >
                                          <Edit size={14} className="mr-1" />
                                          {editingKnowledge ? "Cancel" : "Edit"}
                                        </Button>
                                        <Button
                                          variant="destructive"
                                          size="sm"
                                          onClick={() => {
                                            setKnowledgeToDelete(selectedKnowledge)
                                            setIsDeleteDialogOpen(true)
                                          }}
                                        >
                                          <Trash size={14} className="mr-1" />
                                          Delete
                                        </Button>
                                      </div>
                                    </div>

                                    {editingKnowledge ? (
                                      <div className="flex flex-col">
                                        <div className="mb-4">
                                          <label
                                            htmlFor="edit-knowledge-title"
                                            className="text-sm font-medium block mb-1"
                                          >
                                            Title
                                          </label>
                                          <Input
                                            id="edit-knowledge-title"
                                            value={editedKnowledgeTitle}
                                            onChange={(e) => setEditedKnowledgeTitle(e.target.value)}
                                            placeholder="Knowledge title..."
                                          />
                                        </div>
                                        <div className="flex-1 flex flex-col">
                                          <label
                                            htmlFor="edit-knowledge-content"
                                            className="text-sm font-medium block mb-1"
                                          >
                                            Content
                                          </label>
                                          <Textarea
                                            id="edit-knowledge-content"
                                            value={editedKnowledgeContent}
                                            onChange={(e) => setEditedKnowledgeContent(e.target.value)}
                                            className="flex-1 min-h-[200px]"
                                            placeholder="Knowledge content..."
                                          />
                                        </div>
                                        <div className="mt-2 text-xs text-muted-foreground">
                                          Use [[tag]] syntax to add tags to your knowledge.
                                        </div>
                                        <Button
                                          className="mt-4 bg-purple-700 hover:bg-purple-600 text-white"
                                          onClick={handleSaveKnowledgeChanges}
                                        >
                                          <Save size={16} className="mr-2" />
                                          Save Changes
                                        </Button>
                                      </div>
                                    ) : (
                                      <div>
                                        <div
                                          className="prose max-w-none"
                                          dangerouslySetInnerHTML={{
                                            __html: renderMarkdownWithTags(selectedKnowledge.content),
                                          }}
                                          onClick={handleTagClick}
                                        />
                                        <p className="text-xs text-muted-foreground mt-2">
                                          Created: {formatTimestamp(selectedKnowledge.timestamp)}
                                        </p>
                                      </div>
                                    )}
                                  </div>
                                ) : (
                                  <div className="text-center text-muted-foreground py-8">
                                    Select a knowledge entry to view its contents.
                                  </div>
                                )}
                              </ScrollArea>
                            </div>
                          </div>
                        </div>
                      )}

                      {knowledgeTab === "add" && (
                        <div className="p-4">
                          <div className="space-y-4">
                            <div className="space-y-2">
                              <label htmlFor="new-knowledge-title" className="text-sm font-medium">
                                Title
                              </label>
                              <Input
                                id="new-knowledge-title"
                                value={newKnowledgeTitle}
                                onChange={(e) => setNewKnowledgeTitle(e.target.value)}
                                placeholder="Knowledge title..."
                              />
                            </div>
                            <div className="space-y-2">
                              <label htmlFor="new-knowledge-content" className="text-sm font-medium">
                                Content
                              </label>
                              <Textarea
                                id="new-knowledge-content"
                                value={newKnowledgeContent}
                                onChange={(e) => setNewKnowledgeContent(e.target.value)}
                                className="min-h-[200px]"
                                placeholder="Knowledge content..."
                              />
                              <div className="text-xs text-muted-foreground">
                                Use [[tag]] syntax to add tags to your knowledge.
                              </div>
                            </div>
                            <Button
                              onClick={handleAddKnowledge}
                              className="bg-purple-700 hover:bg-purple-600 text-white"
                            >
                              <Save size={16} className="mr-2" />
                              Add Knowledge
                            </Button>
                          </div>
                        </div>
                      )}

                      {knowledgeTab === "inference" && (
                        <div className="flex flex-col h-full">
                          <div className="px-6 py-2 border-b">
                            <Select value={inferenceTab} onValueChange={setInferenceTab}>
                              <SelectTrigger className="w-full bg-purple-950 border-purple-700 text-white">
                                <SelectValue placeholder="Select view" />
                              </SelectTrigger>
                              <SelectContent className="bg-purple-950 border-purple-700 text-white">
                                <SelectItem value="prompt">Prompt</SelectItem>
                                <SelectItem value="results">Results</SelectItem>
                                <SelectItem value="preview">Preview</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {inferenceTab === "prompt" && (
                            <div className="p-4">
                              <div className="space-y-4">
                                <div className="space-y-2">
                                  <label htmlFor="beliefs-prompt" className="text-sm font-medium">
                                    Beliefs Prompt
                                  </label>
                                  <Textarea
                                    id="beliefs-prompt"
                                    value={beliefsPrompt}
                                    onChange={(e) => setBeliefsPrompt(e.target.value)}
                                    className="min-h-[100px]"
                                    placeholder="Enter prompt for extracting beliefs..."
                                  />
                                </div>

                                <div className="space-y-2">
                                  <label htmlFor="conversation-select" className="text-sm font-medium">
                                    Select Conversation
                                  </label>
                                  <Select value={selectedConversationId} onValueChange={setSelectedConversationId}>
                                    <SelectTrigger className="w-full bg-purple-950 border-purple-700 text-white">
                                      <SelectValue placeholder="Select a conversation" />
                                    </SelectTrigger>
                                    <SelectContent className="bg-purple-950 border-purple-700 text-white">
                                      {relevantConversations.length > 0 ? (
                                        relevantConversations.map((conv) => (
                                          <SelectItem key={conv.id} value={conv.id}>
                                            Conversation {formatTimestamp(conv.startTime)}
                                          </SelectItem>
                                        ))
                                      ) : (
                                        <SelectItem disabled value="no-conversations">
                                          No conversations available
                                        </SelectItem>
                                      )}
                                    </SelectContent>
                                  </Select>
                                </div>

                                <Button
                                  onClick={handleUpdateBeliefs}
                                  className="bg-purple-700 hover:bg-purple-600 text-white"
                                  disabled={isExtractingBeliefs}
                                >
                                  {isExtractingBeliefs ? (
                                    <>Extracting Beliefs... ({extractionProgress}%)</>
                                  ) : (
                                    <>
                                      <Search size={16} className="mr-2" />
                                      Extract Beliefs
                                    </>
                                  )}
                                </Button>
                              </div>
                            </div>
                          )}

                          {inferenceTab === "results" && (
                            <div className="flex-1 p-4 overflow-auto">
                              {extractionStep === "idle" && (
                                <div className="text-center text-muted-foreground py-8">No beliefs extracted yet.</div>
                              )}

                              {extractionStep === "extracting" && (
                                <div className="text-center py-8 space-y-4">
                                  <p className="text-muted-foreground">Extracting beliefs...</p>
                                  <div className="w-full max-w-md mx-auto bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                                    <div
                                      className="bg-purple-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
                                      style={{ width: `${extractionProgress}%` }}
                                    ></div>
                                  </div>
                                  <p className="text-sm text-muted-foreground">{extractionProgress}% complete</p>
                                </div>
                              )}

                              {extractionStep === "refining" && (
                                <div className="text-center py-8 space-y-4">
                                  <p className="text-muted-foreground">Refining beliefs...</p>
                                  <div className="w-full max-w-md mx-auto bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                                    <div
                                      className="bg-purple-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
                                      style={{ width: `${extractionProgress}%` }}
                                    ></div>
                                  </div>
                                  <p className="text-sm text-muted-foreground">{extractionProgress}% complete</p>
                                </div>
                              )}

                              {extractionStep === "complete" && (
                                <div className="space-y-4">
                                  {extractedBeliefs.length > 0 ? (
                                    extractedBeliefs.map((belief, index) => (
                                      <Card key={index} className="p-4">
                                        <div className="flex items-center justify-between">
                                          <h3 className="font-medium truncate max-w-[80%]" title={belief.title}>
                                            {belief.title || "Untitled belief"}
                                            {belief.title && belief.title.length > 30 ? "..." : ""}
                                          </h3>
                                          <label className="inline-flex items-center space-x-2 cursor-pointer">
                                            <input
                                              type="checkbox"
                                              className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                                              checked={belief.selected !== false}
                                              onChange={() => handleToggleBelief(index)}
                                            />
                                            <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                                              Select
                                            </span>
                                          </label>
                                        </div>
                                        <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                                          {belief.refined_content || "No content available"}
                                        </p>
                                      </Card>
                                    ))
                                  ) : (
                                    <div className="text-center text-muted-foreground py-8">No beliefs extracted.</div>
                                  )}

                                  <Button
                                    onClick={handleAddSelectedBeliefs}
                                    className="bg-purple-700 hover:bg-purple-600 text-white"
                                    disabled={extractedBeliefs.length === 0}
                                  >
                                    Add Selected Beliefs to Knowledge
                                  </Button>
                                </div>
                              )}
                            </div>
                          )}

                          {inferenceTab === "preview" && (
                            <div className="p-4">
                              <div className="space-y-4">
                                <div className="space-y-2">
                                  <label htmlFor="raw-beliefs" className="text-sm font-medium">
                                    Raw Beliefs
                                  </label>
                                  <Textarea
                                    id="raw-beliefs"
                                    value={JSON.stringify(rawBeliefs, null, 2)}
                                    className="min-h-[100px]"
                                    readOnly
                                  />
                                </div>

                                <div className="space-y-2">
                                  <label htmlFor="extracted-beliefs" className="text-sm font-medium">
                                    Extracted Beliefs
                                  </label>
                                  <Textarea
                                    id="extracted-beliefs"
                                    value={JSON.stringify(extractedBeliefs, null, 2)}
                                    className="min-h-[100px]"
                                    readOnly
                                  />
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedView === "system" && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">System Prompt</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label htmlFor="system-prompt-name" className="text-sm font-medium">
                        Prompt Name
                      </label>
                      <Input
                        id="system-prompt-name"
                        value={systemPromptName}
                        onChange={(e) => setSystemPromptName(e.target.value)}
                        placeholder="Enter prompt name..."
                      />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="system-prompt" className="text-sm font-medium">
                        System Prompt
                      </label>
                      <Textarea
                        id="system-prompt"
                        value={systemPrompt}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                        className="min-h-[200px]"
                        placeholder="Enter system prompt..."
                      />
                    </div>
                    <Button
                      onClick={handleSaveSystemPrompt}
                      className="bg-purple-700 hover:bg-purple-600 text-white"
                      disabled={isProcessing}
                    >
                      <Save size={16} className="mr-2" />
                      Save System Prompt
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedView === "tools" && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Agent Tools</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Information Access Tools */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Information Access Tools</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.internetSearch}
                            onChange={(e) => handleToolChange("internetSearch", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Internet Search
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.webScraping}
                            onChange={(e) => handleToolChange("webScraping", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Web Scraping
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.wikipediaAccess}
                            onChange={(e) => handleToolChange("wikipediaAccess", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Wikipedia Access
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.newsApi}
                            onChange={(e) => handleToolChange("newsApi", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            News API
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.academicSearch}
                            onChange={(e) => handleToolChange("academicSearch", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Academic Search
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.documentRetrieval}
                            onChange={(e) => handleToolChange("documentRetrieval", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Document Retrieval
                          </span>
                        </label>
                      </div>
                    </div>

                    {/* Content Generation & Processing */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Content Generation &amp; Processing</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.imageGeneration}
                            onChange={(e) => handleToolChange("imageGeneration", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Image Generation
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.textSummarization}
                            onChange={(e) => handleToolChange("textSummarization", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Text Summarization
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.translation}
                            onChange={(e) => handleToolChange("translation", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Translation
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.codeExecution}
                            onChange={(e) => handleToolChange("codeExecution", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Code Execution
                          </span>
                        </label>
                      </div>
                    </div>

                    {/* Knowledge & Reasoning Tools */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Knowledge &amp; Reasoning Tools</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.calculator}
                            onChange={(e) => handleToolChange("calculator", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Calculator
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.knowledgeGraphQuery}
                            onChange={(e) => handleToolChange("knowledgeGraphQuery", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Knowledge Graph Query
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.factChecking}
                            onChange={(e) => handleToolChange("factChecking", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Fact Checking
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.timelineGenerator}
                            onChange={(e) => handleToolChange("timelineGenerator", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Timeline Generator
                          </span>
                        </label>
                      </div>
                    </div>

                    {/* External Integrations */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">External Integrations</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.weatherData}
                            onChange={(e) => handleToolChange("weatherData", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Weather Data
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.mapLocationData}
                            onChange={(e) => handleToolChange("mapLocationData", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Map Location Data
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.financialData}
                            onChange={(e) => handleToolChange("financialData", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Financial Data
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.publicDatasets}
                            onChange={(e) => handleToolChange("publicDatasets", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Public Datasets
                          </span>
                        </label>
                      </div>
                    </div>

                    {/* Agent-Specific Tools */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Agent-Specific Tools</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.memorySearch}
                            onChange={(e) => handleToolChange("memorySearch", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Memory Search
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.crossAgentKnowledge}
                            onChange={(e) => handleToolChange("crossAgentKnowledge", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Cross-Agent Knowledge
                          </span>
                        </label>
                        <label className="inline-flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            className="h-5 w-5 rounded-sm border-gray-700 text-purple-500 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed peer"
                            checked={toolPermissions.conversationAnalysis}
                            onChange={(e) => handleToolChange("conversationAnalysis", e.target.checked)}
                          />
                          <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Conversation Analysis
                          </span>
                        </label>
                      </div>
                    </div>

                    <Button
                      onClick={handleSaveToolSettings}
                      className="bg-purple-700 hover:bg-purple-600 text-white"
                      disabled={!hasToolChanges || isSavingTools}
                    >
                      <Save size={16} className="mr-2" />
                      Save Tool Settings
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-muted-foreground">
          Select an agent to view memory.
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      {isDeleteDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <Card className="max-w-md w-full">
            <CardHeader>
              <CardTitle>Delete Knowledge</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Are you sure you want to delete "{knowledgeToDelete?.title}"?</p>
              <div className="mt-4 flex justify-end gap-2">
                <Button variant="ghost" onClick={() => setIsDeleteDialogOpen(false)}>
                  Cancel
                </Button>
                <Button variant="destructive" onClick={handleDeleteKnowledge}>
                  Delete
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
