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
  biography?: string
  class?: string // Agent class/type (Explorer, Merchant, Scholar, Guardian, etc.)
  avatar?: string // Optional avatar image URL
  inConversation: boolean
  position: Position
  color: string
  knowledge: KnowledgeEntry[]
  autonomyEnabled: boolean // New property to track if agent can engage in autonomous conversations
  toolPermissions?: AgentToolPermissions // Optional to maintain backward compatibility
}

// Enhanced Message interface for real-time conversation monitoring
export interface Message {
  id: string
  content: string
  senderId: string
  timestamp: Date

  // Enhanced metadata for conversation monitoring
  metadata?: {
    // Core message properties
    isGeneratedByLLM?: boolean
    isSystemMessage?: boolean
    type?: 'user' | 'agent' | 'system' | 'conversation_starter' | 'conversation_prompt' | 'action' | 'tool_result' | 'typing'

    // Thread relationships for conversation flow tracking
    respondingTo?: string // ID of message this is responding to
    threadId?: string // Thread identifier for grouped conversations
    parentMessageId?: string // For nested responses
    childMessageIds?: string[] // For tracking responses to this message

    // Agent attribution and context
    agentType?: 'explorer' | 'merchant' | 'scholar' | 'guardian' | 'custom'
    agentRole?: string // Role in current conversation
    confidence?: number // Agent's confidence in this message (0-1)

    // Message processing and delivery
    processingTime?: number // Time taken to generate (ms)
    deliveryStatus?: 'pending' | 'delivered' | 'failed' | 'retrying'
    retryCount?: number

    // Knowledge and reasoning
    knowledgeSources?: Array<{ id: string; title: string; relevance?: number }>
    reasoningTrace?: Array<{ step: string; confidence: number }>

    // Conversation dynamics
    priority?: 'low' | 'normal' | 'high' | 'urgent'
    expectations?: string[] // What responses are expected
    conversationTurn?: number // Turn number in conversation

    // Rich content support
    attachments?: Array<{ type: string; url: string; metadata?: any }>
    embeddedContent?: { type: string; data: any }

    // Real-time monitoring
    readBy?: Array<{ agentId: string; timestamp: Date }>
    reactions?: Array<{ agentId: string; type: string; timestamp: Date }>

    // Analytics and tracking
    sentiment?: { polarity: number; subjectivity: number }
    topics?: string[]
    entities?: Array<{ type: string; value: string; confidence: number }>

    // System and debugging
    debugInfo?: any
    performanceMetrics?: {
      generationTime?: number
      tokens?: { input: number; output: number }
      modelUsed?: string
    }

    // Extensible metadata
    [key: string]: any
  }
}

// Message queue status for tracking pending responses
export interface MessageQueueStatus {
  pendingMessages: Array<{
    messageId: string
    agentId: string
    estimatedTime?: number
    priority: 'low' | 'normal' | 'high' | 'urgent'
  }>
  processingMessages: Array<{
    messageId: string
    agentId: string
    startTime: Date
    progress?: number // 0-1
  }>
  failedMessages: Array<{
    messageId: string
    agentId: string
    error: string
    retryCount: number
  }>
}

// Thread information for conversation structure
export interface ConversationThread {
  id: string
  parentMessageId?: string
  participantIds: string[]
  topic?: string
  startTime: Date
  lastActivity: Date
  messageCount: number
  isActive: boolean
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

  // Enhanced conversation properties for monitoring
  threads?: ConversationThread[]
  activeParticipants?: string[] // Currently active/online participants
  messageQueue?: MessageQueueStatus
  conversationMetrics?: {
    totalMessages: number
    averageResponseTime: number
    participationRates: Record<string, number> // agentId -> participation percentage
    topicDrift?: number // How much the topic has changed
    engagementLevel?: number // Overall engagement score
  }
}

export interface SystemPrompt {
  id: string
  name: string
  content: string
}

// Knowledge Graph Visualization Interfaces for Task 45
export interface KnowledgeNode {
  id: string;
  title: string;
  type: 'concept' | 'fact' | 'belief' | 'agent' | 'entity' | 'relationship' | 'pattern';
  content?: string;

  // Positioning and physics
  x: number;
  y: number;
  vx?: number; // velocity x for physics simulation
  vy?: number; // velocity y for physics simulation
  fx?: number | null; // fixed position x (for pinned nodes)
  fy?: number | null; // fixed position y (for pinned nodes)

  // Visual properties
  radius: number;
  color: string;
  opacity?: number;
  strokeColor?: string;
  strokeWidth?: number;

  // Agent association
  agentId?: string;
  agentIds?: string[]; // for shared knowledge
  ownerType: 'individual' | 'collective' | 'shared';

  // Knowledge properties
  confidence: number; // 0.0 to 1.0
  importance: number; // 0.0 to 1.0
  lastUpdated: Date;
  createdAt: Date;
  accessCount?: number;

  // Belief-specific properties
  supporting_evidence?: string[];
  contradicting_evidence?: string[];
  belief_strength?: number;

  // Metadata
  tags?: string[];
  category?: string;
  source?: string;
  verified?: boolean;

  // Interaction state
  isSelected?: boolean;
  isHovered?: boolean;
  isPinned?: boolean;
  isVisible?: boolean;

  // Additional data for extensions
  metadata?: Record<string, any>;
}

export interface KnowledgeEdge {
  id: string;
  source: string; // source node id
  target: string; // target node id

  // Relationship properties
  type: 'supports' | 'contradicts' | 'relates_to' | 'causes' | 'prevents' | 'similar_to' | 'derived_from' | 'contains' | 'depends_on';
  strength: number; // 0.0 to 1.0
  confidence: number; // 0.0 to 1.0
  bidirectional?: boolean;

  // Visual properties
  color: string;
  width?: number;
  opacity?: number;
  style?: 'solid' | 'dashed' | 'dotted';

  // Temporal properties
  createdAt: Date;
  lastUpdated: Date;

  // Agent context
  agentId?: string; // which agent created this relationship
  agentIds?: string[]; // agents that recognize this relationship

  // Interaction state
  isSelected?: boolean;
  isHovered?: boolean;
  isVisible?: boolean;

  // Additional data
  metadata?: Record<string, any>;
}

// Graph layer types for dual-layer visualization
export interface KnowledgeGraphLayer {
  id: string;
  name: string;
  type: 'individual' | 'collective';
  agentId?: string; // for individual layers
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  isVisible: boolean;
  opacity: number;
  color?: string; // layer tint color
}

// Complete knowledge graph structure
export interface KnowledgeGraph {
  id: string;
  name: string;
  description?: string;
  layers: KnowledgeGraphLayer[];

  // Graph-level metadata
  createdAt: Date;
  lastUpdated: Date;
  version: string;

  // Display settings
  layout: 'force-directed' | 'hierarchical' | 'circular' | 'grid';
  renderer: 'd3' | 'threejs' | 'auto';

  // Performance settings
  maxNodes: number;
  lodEnabled: boolean;
  clusteringEnabled: boolean;

  // Filter state
  filters: KnowledgeGraphFilters;

  // Interaction state
  selectedNodes: string[];
  selectedEdges: string[];
  hoveredNode?: string;
  hoveredEdge?: string;

  // View state
  zoom: number;
  pan: { x: number; y: number };

  metadata?: Record<string, any>;
}

// Filter configuration for knowledge graphs
export interface KnowledgeGraphFilters {
  // Node filters
  nodeTypes: string[];
  confidenceRange: [number, number];
  importanceRange: [number, number];
  timeRange?: [Date, Date];
  agentIds: string[];
  tags: string[];

  // Edge filters
  edgeTypes: string[];
  strengthRange: [number, number];

  // Text search
  searchQuery?: string;

  // Visual filters
  showOnlyConnected: boolean;
  hideIsolatedNodes: boolean;
  maxConnections?: number;
}

// Real-time update events for WebSocket integration
export interface KnowledgeGraphUpdate {
  type: 'node_added' | 'node_updated' | 'node_removed' | 'edge_added' | 'edge_updated' | 'edge_removed' | 'graph_reset';
  graphId: string;
  layerId?: string;
  agentId?: string;
  timestamp: Date;

  // Update data
  node?: KnowledgeNode;
  edge?: KnowledgeEdge;
  nodes?: KnowledgeNode[];
  edges?: KnowledgeEdge[];

  // Change details
  changes?: Record<string, any>;
  previousValue?: any;

  metadata?: Record<string, any>;
}

// Performance and analytics
export interface KnowledgeGraphMetrics {
  nodeCount: number;
  edgeCount: number;
  layerCount: number;

  // Performance metrics
  renderTime: number;
  frameRate: number;
  memoryUsage?: number;

  // Graph analytics
  averageConnectivity: number;
  clusteringCoefficient: number;
  centralityScores: Record<string, number>;

  // User interaction metrics
  interactionCount: number;
  lastInteraction: Date;

  timestamp: Date;
}

// Export configuration
export interface KnowledgeGraphExport {
  format: 'png' | 'svg' | 'json' | 'graphml' | 'gexf';
  includeMetadata: boolean;
  includeFilters: boolean;
  resolution?: number; // for raster formats
  quality?: number; // 0.0 to 1.0

  // Layer selection
  layerIds?: string[];
  includeAllLayers: boolean;

  // Node/edge selection
  nodeIds?: string[];
  edgeIds?: string[];
  includeAllElements: boolean;

  // Visual settings
  backgroundColor?: string;
  includeLabels: boolean;
  labelFontSize?: number;

  metadata?: Record<string, any>;
}

// Conversation Orchestration Control Panel Types for Task 48
export interface ConversationPreset {
  // Preset metadata
  id: string;
  name: string;
  description: string;
  category: 'conservative' | 'balanced' | 'aggressive' | 'custom';
  version: string;
  createdAt: string;
  updatedAt: string;
  createdBy: string;

  // Response Dynamics Parameters
  responseDynamics: {
    // Turn-taking behavior
    turnTaking: {
      enabled: boolean;
      maxConcurrentResponses: number; // 1-5
      responseThreshold: number; // 0.0-1.0, probability threshold for responding
      mentionResponseProbability: number; // 0.0-1.0, higher chance when mentioned
      conversationStarterResponseRate: number; // 0.0-1.0, response rate to conversation starters
    };

    // Agent selection dynamics
    agentSelection: {
      autoSelectRespondents: boolean;
      selectionStrategy: 'random' | 'round_robin' | 'expertise_based' | 'engagement_based';
      diversityBonus: number; // 0.0-1.0, bonus for agents who haven't spoken recently
      expertiseWeight: number; // 0.0-1.0, weight given to domain expertise
      maxSpeakersPerTurn: number; // 1-10
    };

    // Response generation parameters
    responseGeneration: {
      maxKnowledgeEntries: number; // 0-50
      includeAgentKnowledge: boolean;
      streamResponse: boolean;
      responseLength: 'short' | 'medium' | 'long';
      creativityLevel: number; // 0.0-1.0, affects response diversity
      coherenceWeight: number; // 0.0-1.0, weight for maintaining conversation coherence
    };
  };

  // Timing Controls
  timingControls: {
    // Response delays
    responseDelay: {
      type: 'fixed' | 'range' | 'adaptive';
      fixedDelay: number; // milliseconds
      minDelay: number; // milliseconds
      maxDelay: number; // milliseconds
      adaptiveFactors: {
        messageLength: boolean; // longer messages = longer delay
        agentProcessingTime: boolean; // simulate thinking time
        conversationPace: boolean; // adapt to conversation speed
      };
    };

    // Conversation flow
    conversationFlow: {
      maxAutonomousMessages: number; // 5-100
      stallDetectionTimeout: number; // milliseconds
      stallRecoveryStrategy: 'prompt_random' | 'prompt_expert' | 'end_conversation';
      turnTimeoutDuration: number; // milliseconds
      pauseBetweenTurns: number; // milliseconds
    };

    // Real-time controls
    realTimeControls: {
      enableTypingIndicators: boolean;
      typingIndicatorDelay: number; // milliseconds
      messagePreviewEnabled: boolean;
      ghostMessageDuration: number; // milliseconds
    };
  };

  // Advanced Parameters (Expert Mode)
  advancedParameters: {
    // Conversation dynamics
    conversationDynamics: {
      topicDriftAllowance: number; // 0.0-1.0
      contextWindowSize: number; // number of previous messages to consider
      semanticCoherenceThreshold: number; // 0.0-1.0
      emotionalToneConsistency: number; // 0.0-1.0
    };

    // Agent behavior modifiers
    agentBehavior: {
      personalityInfluence: number; // 0.0-1.0
      expertiseBoost: number; // 0.0-1.0
      randomnessInjection: number; // 0.0-1.0
      memoryRetentionFactor: number; // 0.0-1.0
    };

    // Quality controls
    qualityControls: {
      minimumResponseQuality: number; // 0.0-1.0
      duplicateDetectionSensitivity: number; // 0.0-1.0
      relevanceThreshold: number; // 0.0-1.0
      factualAccuracyWeight: number; // 0.0-1.0
    };

    // Performance optimization
    performanceOptimization: {
      enableCaching: boolean;
      cacheExpirationTime: number; // milliseconds
      maxConcurrentGenerations: number; // 1-10
      resourceThrottling: boolean;
    };
  };

  // A/B Testing Configuration
  abTestingConfig?: {
    enabled: boolean;
    testId: string;
    variant: 'A' | 'B';
    comparisonMetrics: string[];
    sampleSize: number;
    confidenceLevel: number; // 0.90, 0.95, 0.99
  };

  // Safety and Validation
  safetyConstraints: {
    enableSafetyChecks: boolean;
    maxResponseLength: number; // characters
    contentFiltering: boolean;
    rateLimiting: {
      enabled: boolean;
      maxRequestsPerMinute: number;
      maxRequestsPerHour: number;
    };
    emergencyStopConditions: string[];
  };

  // Monitoring and Analytics
  monitoring: {
    enableMetrics: boolean;
    trackPerformance: boolean;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
    metricsRetentionDays: number;
    alertThresholds: {
      responseTimeMs: number;
      errorRate: number; // 0.0-1.0
      qualityScore: number; // 0.0-1.0
    };
  };
}

export interface ConversationPresetValidation {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  estimatedPerformanceImpact: 'minimal' | 'moderate' | 'significant' | 'severe';
}

export interface ConversationPresetDiff {
  presetId: string;
  changes: Array<{
    path: string;
    oldValue: any;
    newValue: any;
    changeType: 'added' | 'modified' | 'removed';
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
  }>;
  summary: {
    totalChanges: number;
    riskDistribution: Record<string, number>;
    affectedCategories: string[];
  };
}

export interface ConversationPresetHistory {
  id: string;
  presetId: string;
  version: string;
  changes: ConversationPresetDiff;
  appliedAt: string;
  appliedBy: string;
  rollbackAvailable: boolean;
  performanceMetrics?: {
    responseTime: number;
    qualityScore: number;
    userSatisfaction: number;
    errorRate: number;
  };
}

export interface ConversationPresetTemplate {
  id: string;
  name: string;
  description: string;
  category: ConversationPreset['category'];
  basePreset: Partial<ConversationPreset>;
  customizableFields: string[];
  recommendedUseCase: string;
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert';
}

export interface ConversationParameterConstraints {
  [key: string]: {
    min?: number;
    max?: number;
    allowedValues?: any[];
    required?: boolean;
    dependencies?: string[];
    validationRules?: string[];
  };
}

export interface ConversationPresetExport {
  format: 'json' | 'yaml' | 'csv';
  includeHistory: boolean;
  includeMetrics: boolean;
  dateRange?: {
    startDate: string;
    endDate: string;
  };
  compression: boolean;
}

// Real-time Preview Types
export interface ConversationPreview {
  presetId: string;
  simulationId: string;
  ghostMessages: GhostMessage[];
  probabilityIndicators: ProbabilityIndicator[];
  estimatedOutcomes: ConversationOutcome[];
  performanceProjections: PerformanceProjection;
}

export interface GhostMessage {
  id: string;
  agentId: string;
  agentName: string;
  content: string;
  probability: number; // 0.0-1.0
  estimatedDelay: number; // milliseconds
  confidence: number; // 0.0-1.0
  isVisible: boolean;
  fadeOutTime: number; // milliseconds
}

export interface ProbabilityIndicator {
  agentId: string;
  agentName: string;
  responseprobability: number; // 0.0-1.0
  estimatedResponseTime: number; // milliseconds
  factors: Array<{
    name: string;
    weight: number;
    contribution: number;
  }>;
}

export interface ConversationOutcome {
  scenarioName: string;
  probability: number; // 0.0-1.0
  estimatedDuration: number; // milliseconds
  messageCount: number;
  participantEngagement: Record<string, number>;
  qualityScore: number; // 0.0-1.0
}

export interface PerformanceProjection {
  estimatedResponseTime: number; // milliseconds
  resourceUtilization: number; // 0.0-1.0
  qualityScore: number; // 0.0-1.0
  throughput: number; // messages per minute
  errorProbability: number; // 0.0-1.0
}

// Control Panel State Management
export interface ConversationOrchestrationState {
  currentPreset: ConversationPreset | null;
  isPreviewMode: boolean;
  hasUnsavedChanges: boolean;
  isAdvancedMode: boolean;
  activePreview: ConversationPreview | null;
  validationResult: ConversationPresetValidation | null;
  history: ConversationPresetHistory[];
  abTestResults?: ABTestResults;
}

export interface ABTestResults {
  testId: string;
  startDate: string;
  endDate: string;
  variants: {
    A: ConversationPreset;
    B: ConversationPreset;
  };
  metrics: {
    variant: 'A' | 'B';
    sampleSize: number;
    averageResponseTime: number;
    qualityScore: number;
    userSatisfaction: number;
    errorRate: number;
    engagementLevel: number;
  }[];
  statisticalSignificance: number; // 0.0-1.0
  recommendation: 'A' | 'B' | 'inconclusive';
  confidenceInterval: {
    lower: number;
    upper: number;
    metric: string;
  };
}

// Safety validation types
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  suggestions?: string[];
}

export interface SafetyCheckResult {
  passed: boolean;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  issues: string[];
  recommendations: string[];
}

export type ExpertReviewStatus = 'pending' | 'approved' | 'rejected' | 'requires-changes';
