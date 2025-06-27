import { validateSession } from "@/lib/api-key-storage";
import { rateLimit } from "@/lib/rate-limit";
import { NextRequest, NextResponse } from "next/server";

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

// GET /api/experiments/[experimentid] - Get experiment export details
export async function GET(
  request: NextRequest,
  { params }: { params: { experimentid: string } },
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

    const experimentId = params.experimentid;

    // TODO: In a real implementation, fetch from database or file system
    // For now, return mock data
    if (experimentId === "exp_a1b2c3d4") {
      return NextResponse.json({
        id: "exp_a1b2c3d4",
        name: "Coalition Formation Experiment",
        description:
          "Baseline experiment for coalition formation with 5 agents",
        createdAt: new Date("2025-06-20T14:30:00Z").toISOString(),
        createdBy: "researcher@example.com",
        components: [
          "Agents",
          "Conversations",
          "Knowledge Graphs",
          "Coalitions",
        ],
        statistics: {
          totalAgents: 5,
          totalConversations: 12,
          totalMessages: 156,
          totalKnowledgeNodes: 48,
        },
        fileSizeMb: 2.4,
        filePath: "/exports/experiment_a1b2c3d4.json",
        status: "completed",
      });
    }

    // Return 404 if experiment not found
    return NextResponse.json(
      { error: "Experiment export not found" },
      { status: 404 },
    );
  } catch (error) {
    console.error("Error fetching experiment details:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

// DELETE /api/experiments/[experimentid] - Delete an experiment export
export async function DELETE(
  request: NextRequest,
  { params }: { params: { experimentid: string } },
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

    const experimentId = params.experimentid;

    // TODO: In a real implementation, delete from database or file system
    // For now, simulate successful deletion

    return NextResponse.json({
      success: true,
      message: `Experiment export ${experimentId} deleted successfully`,
    });
  } catch (error) {
    console.error("Error deleting experiment export:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
