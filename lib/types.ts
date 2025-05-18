export interface Position {
  x: number
  y: number
}

export interface KnowledgeEntry {
  id: string
  title: string
  content: string
  timestamp: Date
  tags: string[]
}

// Define the tool permissions interface
export interface AgentToolPermissions {
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

export interface Agent {
  id: string
  name: string
  biography: string
  inConversation: boolean
  position: Position
  color: string
  knowledge: KnowledgeEntry[]
  autonomyEnabled: boolean // New property to track if agent can engage in autonomous conversations
  toolPermissions?: AgentToolPermissions // Optional to maintain backward compatibility
}

export interface Message {
  id: string
  content: string
  senderId: string
  timestamp: Date
  metadata?: {
    isGeneratedByLLM?: boolean
    knowledgeSources?: Array<{ id: string; title: string }>
    isSystemMessage?: boolean
    type?: string
    respondingTo?: string
    [key: string]: any
  }
}

export interface Conversation {
  id: string
  participants: string[]
  messages: Message[]
  startTime: Date
  endTime: Date | null
  isAutonomous?: boolean // New property to track if conversation was autonomously initiated
  trigger?: string // What triggered this autonomous conversation
  topic?: string // The topic of the conversation (especially for knowledge-based triggers)
}

export interface SystemPrompt {
  id: string
  name: string
  content: string
}
