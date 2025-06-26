import { validateSession } from '@/lib/api-key-storage'
import { rateLimit } from '@/lib/rate-limit'
import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'

// Active Inference mathematical parameter schemas
const BeliefStateSchema = z.object({
  beliefs: z.array(z.number().min(0).max(1)), // q(s) - belief distribution
  entropy: z.number().min(0), // H[q(s)]
  confidence: z.number().min(0).max(1), // 1 - normalized entropy
  mostLikelyState: z.number().int().min(0),
  timestamp: z.number()
})

const GenerativeModelSchema = z.object({
  A: z.array(z.array(z.number())), // Observation model matrix
  B: z.array(z.array(z.array(z.number()))), // Transition model tensor
  C: z.array(z.number()), // Prior preferences
  D: z.array(z.number()) // Initial beliefs
})

const PrecisionParametersSchema = z.object({
  sensory: z.number().min(0.1).max(100), // γ - sensory precision
  policy: z.number().min(0.1).max(100), // β - policy precision
  state: z.number().min(0.1).max(100) // α - state precision
})

const ActiveInferenceConfigSchema = z.object({
  template: z.enum(['explorer', 'guardian', 'merchant', 'scholar']),
  stateLabels: z.array(z.string()),
  numStates: z.number().int().min(1).max(20),
  numObservations: z.number().int().min(1).max(20),
  numActions: z.number().int().min(1).max(10),
  generativeModel: GenerativeModelSchema,
  precisionParameters: PrecisionParametersSchema,
  beliefState: BeliefStateSchema.optional(),
  mathematicalConstraints: z.object({
    normalizedBeliefs: z.boolean().default(true),
    stochasticMatrices: z.boolean().default(true),
    precisionBounds: z.boolean().default(true)
  }).optional()
})

// Extended agent creation schema
const CreateAgentSchema = z.object({
  name: z.string().min(1).max(100),

  // Legacy personality support for backward compatibility
  personality: z.object({
    openness: z.number().min(0).max(1),
    conscientiousness: z.number().min(0).max(1),
    extraversion: z.number().min(0).max(1),
    agreeableness: z.number().min(0).max(1),
    neuroticism: z.number().min(0).max(1)
  }).optional(),

  // New Active Inference configuration
  activeInference: ActiveInferenceConfigSchema.optional(),

  capabilities: z.array(z.enum([
    'movement', 'perception', 'communication', 'planning',
    'learning', 'memory', 'resource_management', 'social_interaction'
  ])).optional(),
  initialPosition: z.object({
    x: z.number(),
    y: z.number(),
    z: z.number().optional()
  }).optional(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional()
})

const GetAgentsQuerySchema = z.object({
  status: z.enum(['idle', 'moving', 'interacting', 'planning', 'executing', 'learning', 'error', 'offline']).optional(),
  capability: z.string().optional(),
  tag: z.string().optional(),
  limit: z.coerce.number().min(1).max(100).default(20),
  offset: z.coerce.number().min(0).default(0),
  sortBy: z.enum(['created_at', 'updated_at', 'name', 'status']).default('created_at'),
  sortOrder: z.enum(['asc', 'desc']).default('desc')
})

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
})

// GET /api/agents - List agents with filtering and pagination
export async function GET(request: NextRequest) {
  try {
    // Check rate limit
    const identifier = request.ip ?? 'anonymous'
    const { success } = await limiter.check(identifier, 10)

    if (!success) {
      return NextResponse.json(
        { error: 'Rate limit exceeded' },
        { status: 429 }
      )
    }

    // Validate session
    const sessionId = request.cookies.get('session')?.value
    const isValid = sessionId ? await validateSession('session', sessionId) : false

    if (!isValid) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    // Parse query parameters
    const searchParams = Object.fromEntries(request.nextUrl.searchParams)
    const query = GetAgentsQuerySchema.parse(searchParams)

    // TODO: In a real implementation, fetch from database
    // For now, return mock data
    const mockAgents = [
      {
        id: 'agent-1',
        name: 'Explorer Alpha',
        status: 'idle',
        personality: {
          openness: 0.8,
          conscientiousness: 0.7,
          extraversion: 0.6,
          agreeableness: 0.75,
          neuroticism: 0.3
        },
        capabilities: ['movement', 'perception', 'communication', 'planning'],
        position: { x: 10, y: 20, z: 0 },
        resources: {
          energy: 85,
          health: 100,
          memory_used: 2048,
          memory_capacity: 8192
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 'agent-2',
        name: 'Social Beta',
        status: 'interacting',
        personality: {
          openness: 0.6,
          conscientiousness: 0.8,
          extraversion: 0.9,
          agreeableness: 0.85,
          neuroticism: 0.2
        },
        capabilities: ['communication', 'social_interaction', 'memory'],
        position: { x: 25, y: 30, z: 0 },
        resources: {
          energy: 60,
          health: 95,
          memory_used: 4096,
          memory_capacity: 8192
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ]

    // Apply filters
    let filteredAgents = mockAgents

    if (query.status) {
      filteredAgents = filteredAgents.filter(agent => agent.status === query.status)
    }

    if (query.capability) {
      filteredAgents = filteredAgents.filter(agent =>
        agent.capabilities.includes(query.capability as any)
      )
    }

    // Apply pagination
    const total = filteredAgents.length
    const agents = filteredAgents.slice(
      query.offset,
      query.offset + query.limit
    )

    return NextResponse.json({
      agents,
      pagination: {
        total,
        limit: query.limit,
        offset: query.offset,
        hasMore: query.offset + query.limit < total
      }
    })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request parameters', details: error.errors },
        { status: 400 }
      )
    }

    console.error('Failed to list agents:', error)
    return NextResponse.json(
      { error: 'Failed to list agents' },
      { status: 500 }
    )
  }
}

// POST /api/agents - Create a new agent
export async function POST(request: NextRequest) {
  try {
    // Check rate limit
    const identifier = request.ip ?? 'anonymous'
    const { success } = await limiter.check(identifier, 5)

    if (!success) {
      return NextResponse.json(
        { error: 'Rate limit exceeded' },
        { status: 429 }
      )
    }

    // Validate session
    const sessionId = request.cookies.get('session')?.value
    const isValid = sessionId ? await validateSession('session', sessionId) : false

    if (!isValid) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    // Parse and validate request body
    const body = await request.json()
    const data = CreateAgentSchema.parse(body)

    // Validate Active Inference mathematical constraints
    if (data.activeInference) {
      const ai = data.activeInference;

      // Validate belief distribution normalization
      if (ai.beliefState) {
        const beliefSum = ai.beliefState.beliefs.reduce((sum, b) => sum + b, 0);
        if (Math.abs(beliefSum - 1.0) > 1e-10) {
          return NextResponse.json(
            { error: 'Invalid belief state: beliefs must sum to 1.0' },
            { status: 400 }
          );
        }
      }

      // Validate matrix dimensions
      const { A, B, C, D } = ai.generativeModel;
      if (A.length !== ai.numObservations || A[0]?.length !== ai.numStates) {
        return NextResponse.json(
          { error: 'Invalid A matrix dimensions' },
          { status: 400 }
        );
      }

      // Validate stochastic matrices (rows sum to 1)
      if (ai.mathematicalConstraints?.stochasticMatrices !== false) {
        for (const row of A) {
          const rowSum = row.reduce((sum, val) => sum + val, 0);
          if (Math.abs(rowSum - 1.0) > 1e-10) {
            return NextResponse.json(
              { error: 'Invalid A matrix: rows must be stochastic (sum to 1)' },
              { status: 400 }
            );
          }
        }
      }
    }

    // TODO: In a real implementation:
    // 1. Create agent in database with Active Inference configuration
    // 2. Initialize agent state with belief distribution
    // 3. Generate Agent template from ActiveInference config
    // 4. Start agent lifecycle with pymdp integration

    // Create agent response with Active Inference support
    const agentId = `agent-${Date.now()}`;
    const newAgent = {
      id: agentId,
      name: data.name,
      status: 'idle' as const,

      // Legacy personality for backward compatibility
      personality: data.personality || {
        openness: 0.7,
        conscientiousness: 0.7,
        extraversion: 0.5,
        agreeableness: 0.7,
        neuroticism: 0.3
      },

      // Active Inference configuration
      activeInference: data.activeInference ? {
        ...data.activeInference,
        // Initialize belief state if not provided
        beliefState: data.activeInference.beliefState || {
          beliefs: Array(data.activeInference.numStates).fill(1 / data.activeInference.numStates),
          entropy: Math.log(data.activeInference.numStates),
          confidence: 0.0,
          mostLikelyState: 0,
          timestamp: Date.now()
        }
      } : null,

      capabilities: data.capabilities || ['movement', 'perception', 'communication'],
      position: data.initialPosition || { x: 0, y: 0, z: 0 },
      resources: {
        energy: 100,
        health: 100,
        memory_used: 0,
        memory_capacity: 8192
      },
      tags: data.tags || [],
      metadata: {
        ...data.metadata || {},
        // Add mathematical validation metadata
        mathematicallyValidated: !!data.activeInference,
        templateType: data.activeInference?.template || 'legacy',
        precision: data.activeInference?.precisionParameters || null
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    return NextResponse.json(
      { agent: newAgent },
      { status: 201 }
    )
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request body', details: error.errors },
        { status: 400 }
      )
    }

    console.error('Failed to create agent:', error)
    return NextResponse.json(
      { error: 'Failed to create agent' },
      { status: 500 }
    )
  }
}
