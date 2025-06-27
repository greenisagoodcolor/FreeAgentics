import { validateSession } from "@/lib/api-key-storage";
import { rateLimit } from "@/lib/rate-limit";
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

// Request schemas
const ExecuteCommandSchema = z.object({
  command: z.enum(["move", "interact", "observe", "plan", "learn", "rest"]),
  parameters: z.record(z.any()),
  async: z.boolean().optional(),
});

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

// GET /api/agents/[agentId]/commands - Get command history
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
    const status = request.nextUrl.searchParams.get("status");

    // TODO: In a real implementation, fetch from command history
    // For now, return mock command history
    const mockCommandHistory = [
      {
        id: "cmd-1",
        command: "move",
        parameters: { target: { x: 20, y: 30 }, speed: "normal" },
        status: "completed",
        issued_at: new Date(Date.now() - 3600000).toISOString(),
        started_at: new Date(Date.now() - 3600000 + 1000).toISOString(),
        completed_at: new Date(Date.now() - 3000000).toISOString(),
        result: { success: true, distance_moved: 14.14 },
      },
      {
        id: "cmd-2",
        command: "observe",
        parameters: { radius: 10 },
        status: "completed",
        issued_at: new Date(Date.now() - 2000000).toISOString(),
        started_at: new Date(Date.now() - 2000000 + 500).toISOString(),
        completed_at: new Date(Date.now() - 1999000).toISOString(),
        result: { success: true, entities_detected: 3 },
      },
      {
        id: "cmd-3",
        command: "interact",
        parameters: { target_agent: "agent-2", action: "greet" },
        status: "in_progress",
        issued_at: new Date(Date.now() - 300000).toISOString(),
        started_at: new Date(Date.now() - 299000).toISOString(),
        completed_at: null,
        result: null,
      },
    ];

    let filteredCommands = mockCommandHistory;
    if (status) {
      filteredCommands = filteredCommands.filter(
        (cmd) => cmd.status === status,
      );
    }

    const total = filteredCommands.length;
    const commands = filteredCommands.slice(offset, offset + limit);

    return NextResponse.json({
      agent_id: agentId,
      commands,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
    });
  } catch (error) {
    console.error("Failed to get command history:", error);
    return NextResponse.json(
      { error: "Failed to get command history" },
      { status: 500 },
    );
  }
}

// POST /api/agents/[agentId]/commands - Execute a command
export async function POST(
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
    const {
      command,
      parameters,
      async = false,
    } = ExecuteCommandSchema.parse(body);

    // TODO: In a real implementation:
    // 1. Validate agent exists and is active
    // 2. Validate command parameters
    // 3. Check if agent can execute command (capabilities, resources)
    // 4. Queue command for execution
    // 5. Return command ID and status

    const commandId = `cmd-${Date.now()}`;
    const commandRecord = {
      id: commandId,
      agent_id: agentId,
      command,
      parameters,
      status: async ? "queued" : "executing",
      issued_at: new Date().toISOString(),
      started_at: async ? null : new Date().toISOString(),
      completed_at: null,
      result: null,
    };

    if (!async) {
      // Simulate synchronous command execution
      await new Promise((resolve) => setTimeout(resolve, 1000));

      commandRecord.status = "completed";
      commandRecord.completed_at = new Date().toISOString();
      commandRecord.result = {
        success: true,
        message: `Command ${command} executed successfully`,
      };
    }

    return NextResponse.json(
      {
        command: commandRecord,
        async,
        status_url: `/api/agents/${agentId}/commands/${commandId}`,
      },
      { status: async ? 202 : 200 },
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request body", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Failed to execute command:", error);
    return NextResponse.json(
      { error: "Failed to execute command" },
      { status: 500 },
    );
  }
}

// GET /api/agents/[agentId]/commands/[commandId] - Get command status
export async function GET_COMMAND(
  request: NextRequest,
  { params }: { params: { agentId: string; commandId: string } },
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? "anonymous";
    const { success } = await limiter.check(identifier, 30);

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

    const { agentId, commandId } = params;

    // TODO: In a real implementation, fetch command status
    // For now, return mock status
    const mockCommand = {
      id: commandId,
      agent_id: agentId,
      command: "move",
      parameters: { target: { x: 20, y: 30 }, speed: "normal" },
      status: "completed",
      issued_at: new Date(Date.now() - 10000).toISOString(),
      started_at: new Date(Date.now() - 9000).toISOString(),
      completed_at: new Date(Date.now() - 5000).toISOString(),
      result: {
        success: true,
        distance_moved: 14.14,
        final_position: { x: 20, y: 30 },
      },
    };

    return NextResponse.json({ command: mockCommand });
  } catch (error) {
    console.error("Failed to get command status:", error);
    return NextResponse.json(
      { error: "Failed to get command status" },
      { status: 500 },
    );
  }
}
