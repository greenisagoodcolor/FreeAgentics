import { validateSession } from "@/lib/api-key-storage";
import { rateLimit } from "@/lib/rate-limit";
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

// Shared link schemas
const CreateSharedLinkSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  accessLevel: z.enum(["view", "comment", "edit"]).default("view"),
  expiresIn: z.string().default("7d"),
  requireAuth: z.boolean().default(false),
});

const UpdateSharedLinkSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  description: z.string().max(500).optional(),
  accessLevel: z.enum(["view", "comment", "edit"]).optional(),
  isActive: z.boolean().optional(),
});

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

// GET /api/experiments/[id]/sharing - Get sharing information
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } },
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

    const experimentId = params.id;
    const searchParams = Object.fromEntries(request.nextUrl.searchParams);
    const type = searchParams.type || "all"; // 'links', 'versions', 'changelog', 'all'

    try {
      // Mock data for now - would come from database/service
      const sharingData = {
        sharedLinks:
          type === "all" || type === "links"
            ? [
                {
                  id: "link_001",
                  url: `https://freeagentics.com/shared/${experimentId}/view?token=abc123def456`,
                  name: "Research Team Access",
                  description:
                    "Shared with research team for collaborative analysis",
                  createdAt: "2024-01-15T10:30:00Z",
                  expiresAt: "2024-01-22T10:30:00Z",
                  accessLevel: "view",
                  isActive: true,
                  accessCount: 12,
                  lastAccessed: "2024-01-16T14:20:00Z",
                },
              ]
            : undefined,

        versions:
          type === "all" || type === "versions"
            ? [
                {
                  id: "v1.0.0",
                  name: "Initial Export",
                  createdAt: "2024-01-15T10:30:00Z",
                  createdBy: "researcher@example.com",
                  changesSummary: "Initial export with all components",
                  statistics: {
                    totalAgents: 5,
                    totalConversations: 12,
                    totalMessages: 89,
                    totalKnowledgeNodes: 234,
                  },
                },
                {
                  id: "v1.1.0",
                  name: "Updated Knowledge Graphs",
                  createdAt: "2024-01-16T14:20:00Z",
                  createdBy: "analyst@example.com",
                  changesSummary:
                    "Added new knowledge graph connections and updated agent behaviors",
                  statistics: {
                    totalAgents: 5,
                    totalConversations: 15,
                    totalMessages: 127,
                    totalKnowledgeNodes: 298,
                  },
                },
              ]
            : undefined,

        changeLog:
          type === "all" || type === "changelog"
            ? [
                {
                  id: "change_001",
                  timestamp: "2024-01-16T14:20:00Z",
                  author: "analyst@example.com",
                  action: "modified",
                  component: "Knowledge Graphs",
                  description: "Added 64 new nodes and 12 edges",
                  details:
                    "Enhanced agent decision-making pathways with additional concept relationships",
                },
                {
                  id: "change_002",
                  timestamp: "2024-01-16T09:15:00Z",
                  author: "researcher@example.com",
                  action: "shared",
                  component: "Export",
                  description: "Shared experiment with research team",
                  details:
                    "Created read-only access link for collaborative analysis",
                },
                {
                  id: "change_003",
                  timestamp: "2024-01-15T10:30:00Z",
                  author: "researcher@example.com",
                  action: "created",
                  component: "Export",
                  description: "Created initial experiment export",
                  details:
                    "Exported complete experiment state including all agents, conversations, and knowledge graphs",
                },
              ]
            : undefined,
      };

      return NextResponse.json(sharingData);
    } catch (error) {
      console.error("Error fetching sharing data:", error);
      return NextResponse.json(
        { error: "Failed to fetch sharing data" },
        { status: 500 },
      );
    }
  } catch (error) {
    console.error("Error in sharing GET:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

// POST /api/experiments/[id]/sharing - Create shared link
export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } },
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

    const experimentId = params.id;

    // Parse and validate request body
    const body = await request.json();
    const linkRequest = CreateSharedLinkSchema.parse(body);

    try {
      // Generate expiration date
      const getExpirationDate = (expiresIn: string): string => {
        const now = new Date();
        const expirationMap: Record<string, number> = {
          "1h": 1000 * 60 * 60,
          "1d": 1000 * 60 * 60 * 24,
          "7d": 1000 * 60 * 60 * 24 * 7,
          "30d": 1000 * 60 * 60 * 24 * 30,
          never: 0,
        };

        const duration = expirationMap[expiresIn];
        if (duration === 0) return "";

        return new Date(now.getTime() + duration).toISOString();
      };

      // Create new shared link
      const newLink = {
        id: `link_${Date.now()}`,
        url: `https://freeagentics.com/shared/${experimentId}/view?token=${Math.random().toString(36).substring(2, 15)}`,
        name: linkRequest.name,
        description: linkRequest.description || "",
        createdAt: new Date().toISOString(),
        expiresAt: getExpirationDate(linkRequest.expiresIn),
        accessLevel: linkRequest.accessLevel,
        isActive: true,
        accessCount: 0,
        lastAccessed: null,
      };

      // In production, this would be saved to database
      // await saveLinkToDatabase(experimentId, newLink)

      // Log the sharing action
      const changeLogEntry = {
        id: `change_${Date.now()}`,
        timestamp: new Date().toISOString(),
        author: "current_user@example.com", // Would come from session
        action: "shared" as const,
        component: "Export",
        description: `Created shared link: ${linkRequest.name}`,
        details: `Access level: ${linkRequest.accessLevel}, Expires: ${linkRequest.expiresIn === "never" ? "Never" : linkRequest.expiresIn}`,
      };

      return NextResponse.json(
        {
          link: newLink,
          changeLogEntry,
          message: "Shared link created successfully",
        },
        { status: 201 },
      );
    } catch (error) {
      console.error("Error creating shared link:", error);
      return NextResponse.json(
        { error: "Failed to create shared link" },
        { status: 500 },
      );
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request parameters", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Error in sharing POST:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

// PUT /api/experiments/[id]/sharing - Update shared link
export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } },
) {
  try {
    // Check rate limit
    const identifier = request.ip ?? "anonymous";
    const { success } = await limiter.check(identifier, 15);

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

    const experimentId = params.id;
    const searchParams = Object.fromEntries(request.nextUrl.searchParams);
    const linkId = searchParams.linkId;

    if (!linkId) {
      return NextResponse.json(
        { error: "Link ID is required" },
        { status: 400 },
      );
    }

    // Parse and validate request body
    const body = await request.json();
    const linkUpdate = UpdateSharedLinkSchema.parse(body);

    try {
      // In production, this would update the database
      // const updatedLink = await updateLinkInDatabase(experimentId, linkId, linkUpdate)

      // Mock updated link
      const updatedLink = {
        id: linkId,
        url: `https://freeagentics.com/shared/${experimentId}/view?token=abc123def456`,
        ...linkUpdate,
        updatedAt: new Date().toISOString(),
      };

      // Log the update action
      const changeLogEntry = {
        id: `change_${Date.now()}`,
        timestamp: new Date().toISOString(),
        author: "current_user@example.com",
        action: "modified" as const,
        component: "Shared Link",
        description: `Updated shared link: ${linkUpdate.name || "settings"}`,
        details: `Updated properties: ${Object.keys(linkUpdate).join(", ")}`,
      };

      return NextResponse.json({
        link: updatedLink,
        changeLogEntry,
        message: "Shared link updated successfully",
      });
    } catch (error) {
      console.error("Error updating shared link:", error);
      return NextResponse.json(
        { error: "Failed to update shared link" },
        { status: 500 },
      );
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request parameters", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Error in sharing PUT:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

// DELETE /api/experiments/[id]/sharing - Delete shared link
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } },
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

    const experimentId = params.id;
    const searchParams = Object.fromEntries(request.nextUrl.searchParams);
    const linkId = searchParams.linkId;

    if (!linkId) {
      return NextResponse.json(
        { error: "Link ID is required" },
        { status: 400 },
      );
    }

    try {
      // In production, this would delete from database
      // await deleteLinkFromDatabase(experimentId, linkId)

      // Log the deletion action
      const changeLogEntry = {
        id: `change_${Date.now()}`,
        timestamp: new Date().toISOString(),
        author: "current_user@example.com",
        action: "deleted" as const,
        component: "Shared Link",
        description: `Deleted shared link: ${linkId}`,
        details: "Shared link permanently removed",
      };

      return NextResponse.json({
        changeLogEntry,
        message: "Shared link deleted successfully",
      });
    } catch (error) {
      console.error("Error deleting shared link:", error);
      return NextResponse.json(
        { error: "Failed to delete shared link" },
        { status: 500 },
      );
    }
  } catch (error) {
    console.error("Error in sharing DELETE:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
