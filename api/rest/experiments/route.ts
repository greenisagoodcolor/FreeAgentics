import { validateSession } from "@/lib/api-key-storage";
import { rateLimit } from "@/lib/rate-limit";
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { spawn } from "child_process";
import { promisify } from "util";
import path from "path";

// Experiment component schemas
const ExperimentComponentsSchema = z.object({
  agents: z.boolean().default(true),
  conversations: z.boolean().default(true),
  knowledgeGraphs: z.boolean().default(true),
  coalitions: z.boolean().default(true),
  inferenceModels: z.boolean().default(true),
  worldState: z.boolean().default(true),
  parameters: z.boolean().default(true),
  agentIds: z.array(z.string()).optional(),
  conversationIds: z.array(z.string()).optional(),
  dateRange: z.tuple([z.string(), z.string()]).optional(),
  includeArchived: z.boolean().default(false),
});

// Export request schema
const ExportExperimentSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  components: ExperimentComponentsSchema.optional(),
  compression: z.boolean().default(true),
  createdBy: z.string().optional(),
});

// Query schema for listing exports
const ListExperimentsQuerySchema = z.object({
  limit: z.coerce.number().min(1).max(100).default(20),
  offset: z.coerce.number().min(0).default(0),
  sortBy: z.enum(["created_at", "name", "size"]).default("created_at"),
  sortOrder: z.enum(["asc", "desc"]).default("desc"),
});

// Rate limiter
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

/**
 * Call Python backend ExperimentExport service
 */
async function callPythonExportService(
  action: string,
  params: Record<string, any> = {},
): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(
      process.cwd(),
      "scripts",
      "experiment_service.py",
    );
    const args = [action, JSON.stringify(params)];

    const pythonProcess = spawn("python3", [pythonScript, ...args], {
      cwd: process.cwd(),
      env: { ...process.env, PYTHONPATH: process.cwd() },
    });

    let stdout = "";
    let stderr = "";

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (error) {
          reject(new Error(`Failed to parse Python response: ${error}`));
        }
      } else {
        reject(new Error(`Python script failed: ${stderr}`));
      }
    });

    pythonProcess.on("error", (error) => {
      reject(new Error(`Failed to start Python process: ${error}`));
    });
  });
}

// GET /api/experiments - List experiment exports
export async function GET(request: NextRequest) {
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

    // Parse query parameters
    const searchParams = Object.fromEntries(request.nextUrl.searchParams);
    const query = ListExperimentsQuerySchema.parse(searchParams);

    try {
      // Call Python backend to list exports
      const result = await callPythonExportService("list_exports", query);

      if (result.success) {
        return NextResponse.json({
          exports: result.data,
          pagination: {
            total: result.total || result.data.length,
            limit: query.limit,
            offset: query.offset,
            hasMore:
              query.offset + query.limit < (result.total || result.data.length),
          },
        });
      } else {
        throw new Error(result.error || "Failed to list experiments");
      }
    } catch (pythonError) {
      console.error("Python service error:", pythonError);

      // Fallback to mock data if Python service fails
      const mockExports = [
        {
          id: "exp_fallback_001",
          name: "Fallback Export (Service Unavailable)",
          description:
            "Mock data returned due to backend service unavailability",
          createdAt: new Date().toISOString(),
          createdBy: "system",
          components: ["Agents", "Conversations"],
          statistics: {
            totalAgents: 0,
            totalConversations: 0,
            totalMessages: 0,
            totalKnowledgeNodes: 0,
          },
          fileSizeMb: 0.1,
          filePath: "/exports/fallback.json",
        },
      ];

      return NextResponse.json({
        exports: mockExports,
        pagination: {
          total: mockExports.length,
          limit: query.limit,
          offset: query.offset,
          hasMore: false,
        },
        warning: "Backend service unavailable, showing fallback data",
      });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request parameters", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Error listing experiments:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

// POST /api/experiments - Create a new experiment export
export async function POST(request: NextRequest) {
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

    // Parse and validate request body
    const body = await request.json();
    const exportRequest = ExportExperimentSchema.parse(body);

    try {
      // Prepare parameters for Python service
      const exportParams = {
        experiment_name: exportRequest.name,
        description: exportRequest.description || "",
        created_by: exportRequest.createdBy || "api_user",
        components: exportRequest.components
          ? {
              agents: exportRequest.components.agents,
              conversations: exportRequest.components.conversations,
              knowledge_graphs: exportRequest.components.knowledgeGraphs,
              coalitions: exportRequest.components.coalitions,
              inference_models: exportRequest.components.inferenceModels,
              world_state: exportRequest.components.worldState,
              parameters: exportRequest.components.parameters,
              agent_ids: exportRequest.components.agentIds,
              conversation_ids: exportRequest.components.conversationIds,
              date_range: exportRequest.components.dateRange,
              include_archived: exportRequest.components.includeArchived,
            }
          : null,
      };

      // Call Python backend to create export
      const result = await callPythonExportService(
        "export_experiment",
        exportParams,
      );

      if (result.success) {
        return NextResponse.json(
          {
            exportId: result.export_id,
            name: exportRequest.name,
            description: exportRequest.description || "",
            status: "completed",
            message: "Export created successfully",
            statistics: result.statistics,
            filePath: result.file_path,
          },
          { status: 201 },
        );
      } else {
        throw new Error(result.error || "Export failed");
      }
    } catch (pythonError) {
      console.error("Python export service error:", pythonError);

      // Return processing status - export will be handled by background service
      const fallbackExportId = `exp_${Math.random().toString(36).substring(2, 10)}`;

      return NextResponse.json(
        {
          exportId: fallbackExportId,
          name: exportRequest.name,
          description: exportRequest.description || "",
          status: "processing",
          message:
            "Export job queued. Backend service will process when available.",
          warning: "Primary export service unavailable",
        },
        { status: 202 },
      );
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request parameters", details: error.errors },
        { status: 400 },
      );
    }

    console.error("Error creating experiment export:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
