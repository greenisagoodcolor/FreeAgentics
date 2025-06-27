import { validateSession } from "@/lib/api-key-storage";
import { rateLimit } from "@/lib/rate-limit";
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

// Request schemas
const UpdateAgentSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  status: z
    .enum([
      "idle",
      "moving",
      "interacting",
      "planning",
      "executing",
      "learning",
      "error",
      "offline",
    ])
    .optional(),
  position: z
    .object({
      x: z.number(),
      y: z.number(),
      z: z.number().optional(),
    })
    .optional(),
  resources: z
    .object({
      energy: z.number().min(0).max(100).optional(),
      health: z.number().min(0).max(100).optional(),
    })
    .optional(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional(),
});

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

// GET /api/agents/[agentId] - Get a specific agent
export async function GET(
  request: NextRequest,
  { params }: { params: { agentId: string } },
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? "anonymous";
    const { success } = await limiter.check(identifier, 20);

    if (!success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429 },
      );
    }

    // Validate session
    const sessionId = request.cookies.get("session")?.value;
    const isValid = sessionId
      ? await validateSession("session", sessionId)
      : false;

    if (!isValid) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const agentId = params.agentId;

    // TODO: In a real implementation, fetch from database
    // For now, return mock data
    const mockAgent = {
      id: agentId,
      name: "Explorer Alpha",
      status: "idle",
      personality: {
        openness: 0.8,
        conscientiousness: 0.7,
        extraversion: 0.6,
        agreeableness: 0.75,
        neuroticism: 0.3,
      },
      capabilities: ["movement", "perception", "communication", "planning"],
      position: { x: 10, y: 20, z: 0 },
      resources: {
        energy: 85,
        health: 100,
        memory_used: 2048,
        memory_capacity: 8192,
      },
      beliefs: [
        {
          id: "belief-1",
          content: "Resource at location (15, 25)",
          confidence: 0.8,
        },
        {
          id: "belief-2",
          content: "Safe zone at location (0, 0)",
          confidence: 0.95,
        },
      ],
      goals: [
        {
          id: "goal-1",
          description: "Explore unknown areas",
          priority: 0.7,
          deadline: null,
        },
        {
          id: "goal-2",
          description: "Find resources",
          priority: 0.9,
          deadline: "2025-06-19T00:00:00Z",
        },
      ],
      relationships: [
        {
          agent_id: "agent-2",
          trust_level: 0.8,
          last_interaction: "2025-06-18T12:00:00Z",
        },
      ],
      tags: ["explorer", "autonomous"],
      metadata: {
        version: "1.0.0",
        created_by: "system",
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    return NextResponse.json({ agent: mockAgent });
  } catch (error) {
    console.error("Failed to get agent:", error);
    return NextResponse.json({ error: "Failed to get agent" }, { status: 500 });
  }
}

// PUT /api/agents/[agentId] - Update an agent
export async function PUT(
  request: NextRequest,
  { params }: { params: { agentId: string } },
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? "anonymous";
    const { success } = await limiter.check(identifier, 10);

    if (!success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429 },
      );
    }

    // Validate session
    const sessionId = request.cookies.get("session")?.value;
    const isValid = sessionId
      ? await validateSession("session", sessionId)
      : false;

    if (!isValid) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const agentId = params.agentId;
    const body = await request.json();
    const updates = UpdateAgentSchema.parse(body);

    // TODO: In a real implementation:
    // 1. Fetch agent from database
    // 2. Apply updates
    // 3. Validate state transitions
    // 4. Save to database
    // 5. Notify agent system of changes

    // For now, return mock updated agent
    const updatedAgent = {
      id: agentId,
      name: updates.name || "Explorer Alpha",
      status: updates.status || "idle",
      personality: {
        openness: 0.8,
        conscientiousness: 0.7,
        extraversion: 0.6,
        agreeableness: 0.75,
        neuroticism: 0.3,
      },
      capabilities: ["movement", "perception", "communication", "planning"],
      position: updates.position || { x: 10, y: 20, z: 0 },
      resources: {
        energy: updates.resources?.energy ?? 85,
        health: updates.resources?.health ?? 100,
        memory_used: 2048,
        memory_capacity: 8192,
      },
      tags: updates.tags || ["explorer", "autonomous"],
      metadata: updates.metadata || { version: "1.0.0", created_by: "system" },
      created_at: "2025-06-18T12:00:00Z",
      updated_at: new Date().toISOString(),
    };

    return NextResponse.json({ agent: updatedAgent });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request body", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Failed to update agent:", error);
    return NextResponse.json(
      { error: "Failed to update agent" },
      { status: 500 },
    );
  }
}

// PATCH /api/agents/[agentId] - Partially update an agent (alias for PUT)
export const PATCH = PUT;

// DELETE /api/agents/[agentId] - Delete an agent
export async function DELETE(
  request: NextRequest,
  { params }: { params: { agentId: string } },
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? "anonymous";
    const { success } = await limiter.check(identifier, 5);

    if (!success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429 },
      );
    }

    // Validate session
    const sessionId = request.cookies.get("session")?.value;
    const isValid = sessionId
      ? await validateSession("session", sessionId)
      : false;

    if (!isValid) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const agentId = params.agentId;

    // TODO: In a real implementation:
    // 1. Check if agent exists
    // 2. Check if agent can be deleted (no active operations)
    // 3. Clean up agent resources
    // 4. Remove from database
    // 5. Notify other systems

    // For now, return success
    return NextResponse.json({
      message: `Agent ${agentId} deleted successfully`,
      deleted_at: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Failed to delete agent:", error);
    return NextResponse.json(
      { error: "Failed to delete agent" },
      { status: 500 },
    );
  }
}
