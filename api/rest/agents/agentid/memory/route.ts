import { validateSession } from '@/lib/api-key-storage'
import { rateLimit } from '@/lib/rate-limit'
import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'

// Request schemas
const AddMemorySchema = z.object({
  type: z.enum(['event', 'interaction', 'location', 'pattern', 'general']),
  content: z.string().min(1).max(1000),
  importance: z.number().min(0).max(1).optional(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional()
})

const QueryMemorySchema = z.object({
  type: z.enum(['event', 'interaction', 'location', 'pattern', 'general']).optional(),
  query: z.string().optional(),
  tags: z.array(z.string()).optional(),
  min_importance: z.number().min(0).max(1).optional(),
  limit: z.number().min(1).max(100).default(20),
  offset: z.number().min(0).default(0)
})

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
})

// GET /api/agents/[agentId]/memory - Query agent memories
export async function GET(
  request: NextRequest,
  { params }: { params: { agentId: string } }
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? 'anonymous'
    const { success } = await limiter.check(identifier, 20)

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

    const agentId = params.agentId
    const searchParams = Object.fromEntries(request.nextUrl.searchParams)
    const query = QueryMemorySchema.parse(searchParams)

    // TODO: In a real implementation, query from memory system
    // For now, return mock memories
    const mockMemories = [
      {
        id: 'mem-1',
        type: 'location',
        content: 'Found abundant resources at coordinates (45, 67)',
        importance: 0.9,
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        access_count: 5,
        last_accessed: new Date(Date.now() - 3600000).toISOString(),
        tags: ['resources', 'exploration'],
        metadata: {
          location: { x: 45, y: 67 },
          resource_type: 'energy_crystal'
        }
      },
      {
        id: 'mem-2',
        type: 'interaction',
        content: 'Agent-3 shared information about safe zones',
        importance: 0.8,
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        access_count: 3,
        last_accessed: new Date(Date.now() - 1800000).toISOString(),
        tags: ['social', 'information'],
        metadata: {
          agent_id: 'agent-3',
          trust_increase: 0.1
        }
      },
      {
        id: 'mem-3',
        type: 'pattern',
        content: 'Resources regenerate every 2 hours in explored areas',
        importance: 0.85,
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        access_count: 12,
        last_accessed: new Date(Date.now() - 600000).toISOString(),
        tags: ['pattern', 'resources'],
        metadata: {
          confidence: 0.9,
          observations: 8
        }
      }
    ]

    // Apply filters
    let filteredMemories = mockMemories

    if (query.type) {
      filteredMemories = filteredMemories.filter(mem => mem.type === query.type)
    }

    if (query.min_importance) {
      filteredMemories = filteredMemories.filter(mem => mem.importance >= query.min_importance)
    }

    if (query.tags && query.tags.length > 0) {
      filteredMemories = filteredMemories.filter(mem =>
        query.tags!.some(tag => mem.tags.includes(tag))
      )
    }

    const total = filteredMemories.length
    const memories = filteredMemories.slice(query.offset, query.offset + query.limit)

    return NextResponse.json({
      agent_id: agentId,
      memories,
      memory_stats: {
        total_memories: total,
        total_capacity: 1000,
        used_capacity: total * 10, // Mock capacity usage
        consolidation_pending: false
      },
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
        { error: 'Invalid query parameters', details: error.errors },
        { status: 400 }
      )
    }

    console.error('Failed to query memories:', error)
    return NextResponse.json(
      { error: 'Failed to query memories' },
      { status: 500 }
    )
  }
}

// POST /api/agents/[agentId]/memory - Add a new memory
export async function POST(
  request: NextRequest,
  { params }: { params: { agentId: string } }
) {
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

    const agentId = params.agentId
    const body = await request.json()
    const memoryData = AddMemorySchema.parse(body)

    // TODO: In a real implementation:
    // 1. Check agent memory capacity
    // 2. Run memory consolidation if needed
    // 3. Store memory
    // 4. Update indices
    // 5. Trigger learning processes

    const newMemory = {
      id: `mem-${Date.now()}`,
      agent_id: agentId,
      type: memoryData.type,
      content: memoryData.content,
      importance: memoryData.importance || 0.5,
      timestamp: new Date().toISOString(),
      access_count: 0,
      last_accessed: new Date().toISOString(),
      tags: memoryData.tags || [],
      metadata: memoryData.metadata || {}
    }

    return NextResponse.json(
      { memory: newMemory },
      { status: 201 }
    )
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request body', details: error.errors },
        { status: 400 }
      )
    }

    console.error('Failed to add memory:', error)
    return NextResponse.json(
      { error: 'Failed to add memory' },
      { status: 500 }
    )
  }
}

// DELETE /api/agents/[agentId]/memory/[memoryId] - Delete a memory
export async function DELETE(
  request: NextRequest,
  { params }: { params: { agentId: string, memoryId: string } }
) {
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

    const { agentId, memoryId } = params

    // TODO: In a real implementation:
    // 1. Check if memory exists
    // 2. Check if memory can be deleted (not core memory)
    // 3. Remove from storage
    // 4. Update indices

    return NextResponse.json({
      message: `Memory ${memoryId} deleted successfully`,
      agent_id: agentId,
      deleted_at: new Date().toISOString()
    })
  } catch (error) {
    console.error('Failed to delete memory:', error)
    return NextResponse.json(
      { error: 'Failed to delete memory' },
      { status: 500 }
    )
  }
}

// POST /api/agents/[agentId]/memory/consolidate - Trigger memory consolidation
export async function CONSOLIDATE(
  request: NextRequest,
  { params }: { params: { agentId: string } }
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? 'anonymous'
    const { success } = await limiter.check(identifier, 2)

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

    const agentId = params.agentId

    // TODO: In a real implementation:
    // 1. Check if consolidation is needed
    // 2. Identify similar memories
    // 3. Merge/compress memories
    // 4. Update importance scores
    // 5. Remove redundant memories

    return NextResponse.json({
      agent_id: agentId,
      consolidation_started: new Date().toISOString(),
      estimated_duration: 5000, // milliseconds
      memories_before: 150,
      estimated_memories_after: 120,
      status: 'in_progress'
    })
  } catch (error) {
    console.error('Failed to consolidate memories:', error)
    return NextResponse.json(
      { error: 'Failed to consolidate memories' },
      { status: 500 }
    )
  }
}
