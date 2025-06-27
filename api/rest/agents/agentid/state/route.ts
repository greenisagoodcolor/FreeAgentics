import { validateSession } from "@/lib/api-key-storage";
import { rateLimit } from "@/lib/rate-limit";
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

// Request schemas
const UpdateStateSchema = z.object({
  status: z.enum([
    "idle",
    "moving",
    "interacting",
    "planning",
    "executing",
    "learning",
    "error",
    "offline",
  ]),
  force: z.boolean().optional(),
});

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

// GET /api/agents/[agentId]/state - Get agent state history
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
    const limit = parseInt(request.nextUrl.searchParams.get("limit") || "10");
    const offset = parseInt(request.nextUrl.searchParams.get("offset") || "0");

    // TODO: In a real implementation, fetch from state history
    // For now, return mock state history
    const mockStateHistory = [
      {
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        from_state: "idle",
        to_state: "moving",
        reason: "Task assigned: Explore sector 5",
        metadata: { task_id: "task-123" },
      },
      {
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        from_state: "moving",
        to_state: "interacting",
        reason: "Encountered agent-2",
        metadata: { agent_id: "agent-2" },
      },
      {
        timestamp: new Date().toISOString(),
        from_state: "interacting",
        to_state: "idle",
        reason: "Interaction completed",
        metadata: { duration: 300 },
      },
    ];

    const total = mockStateHistory.length;
    const history = mockStateHistory.slice(offset, offset + limit);

    return NextResponse.json({
      agent_id: agentId,
      current_state: "idle",
      state_history: history,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
    });
  } catch (error) {
    console.error("Failed to get agent state history:", error);
    return NextResponse.json(
      { error: "Failed to get agent state history" },
      { status: 500 },
    );
  }
}

// PUT /api/agents/[agentId]/state - Update agent state
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
    const { status, force } = UpdateStateSchema.parse(body);

    // TODO: In a real implementation:
    // 1. Fetch current agent state
    // 2. Validate state transition
    // 3. Apply state change
    // 4. Record in history
    // 5. Notify observers

    // For now, simulate state transition validation
    const validTransitions: Record<string, string[]> = {
      idle: ["moving", "interacting", "planning", "learning", "offline"],
      moving: ["idle", "interacting", "offline", "error"],
      interacting: ["idle", "moving", "offline", "error"],
      planning: ["idle", "moving", "executing", "offline", "error"],
      executing: [
        "idle",
        "moving",
        "interacting",
        "planning",
        "offline",
        "error",
      ],
      learning: ["idle", "planning", "offline", "error"],
      error: ["idle", "offline"],
      offline: ["idle"],
    };

    const currentState = "idle"; // Mock current state
    const isValidTransition =
      validTransitions[currentState]?.includes(status) || force;

    if (!isValidTransition) {
      return NextResponse.json(
        {
          error: "Invalid state transition",
          details: {
            current_state: currentState,
            requested_state: status,
            valid_transitions: validTransitions[currentState],
          },
        },
        { status: 400 },
      );
    }

    return NextResponse.json({
      agent_id: agentId,
      previous_state: currentState,
      current_state: status,
      transition_time: new Date().toISOString(),
      forced: force || false,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request body", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Failed to update agent state:", error);
    return NextResponse.json(
      { error: "Failed to update agent state" },
      { status: 500 },
    );
  }
}
