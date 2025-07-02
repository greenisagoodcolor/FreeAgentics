import { NextRequest } from "next/server";

// Mock WebSocket API route for fallback when backend isn't available
export async function GET(request: NextRequest) {
  // Check if this is a WebSocket upgrade request
  const upgrade = request.headers.get("upgrade");

  if (upgrade !== "websocket") {
    return new Response("Expected WebSocket", { status: 400 });
  }

  // In a real implementation, this would handle WebSocket upgrades
  // For now, return a mock response indicating service unavailable
  return new Response(
    JSON.stringify({
      error: "Backend WebSocket service unavailable",
      message: "Using fallback mode - limited functionality",
      fallback: true,
    }),
    {
      status: 503,
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
}

// Handle POST requests for direct API calls when WebSocket fails
export async function POST(request: NextRequest) {
  const body = await request.json();

  // Mock responses for common WebSocket message types
  switch (body.type) {
    case "ping":
      return Response.json({
        type: "pong",
        clientTime: body.clientTime,
        serverTime: Date.now(),
        fallback: true,
      });

    case "agent_status_change":
      return Response.json({
        type: "agent_status_changed",
        agent_id: body.agent_id,
        status: body.status,
        fallback: true,
      });

    case "message_send":
      return Response.json({
        type: "message_sent",
        conversation_id: body.conversation_id,
        message_id: `msg_${Date.now()}`,
        fallback: true,
      });

    default:
      return Response.json({
        type: "ack",
        original_type: body.type,
        fallback: true,
      });
  }
}
