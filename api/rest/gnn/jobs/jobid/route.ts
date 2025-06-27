import { validateApiKey } from "@/lib/api-key-service-server";
import { cancelJob, getJobStatus } from "@/lib/gnn/job-manager";
import { getServerSession } from "next-auth";
import { NextRequest, NextResponse } from "next/server";

interface RouteParams {
  params: {
    jobId: string;
  };
}

// GET /api/gnn/jobs/[jobId] - Get job status
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    // Validate API key or session
    const apiKey = request.headers.get("x-api-key");
    const session = await getServerSession();

    if (!apiKey && !session) {
      return NextResponse.json(
        {
          error: "Unauthorized",
          message: "API key or valid session required",
        },
        { status: 401 },
      );
    }

    if (apiKey) {
      const isValid = await validateApiKey(apiKey);
      if (!isValid) {
        return NextResponse.json(
          {
            error: "Invalid API key",
            message: "The provided API key is invalid or expired",
          },
          { status: 401 },
        );
      }
    }

    const { jobId } = params;

    if (!jobId) {
      return NextResponse.json(
        {
          error: "Invalid request",
          message: "Job ID is required",
        },
        { status: 400 },
      );
    }

    const jobStatus = await getJobStatus(jobId);

    if (!jobStatus) {
      return NextResponse.json(
        {
          error: "Not found",
          message: `Job ${jobId} not found`,
        },
        { status: 404 },
      );
    }

    return NextResponse.json({
      jobId: jobStatus.id,
      status: jobStatus.status,
      progress: jobStatus.progress,
      createdAt: jobStatus.createdAt,
      updatedAt: jobStatus.updatedAt,
      completedAt: jobStatus.completedAt,
      error: jobStatus.error,
      metadata: {
        graphNodes: jobStatus.metadata?.graphNodes,
        graphEdges: jobStatus.metadata?.graphEdges,
        modelArchitecture: jobStatus.metadata?.modelArchitecture,
        task: jobStatus.metadata?.task,
      },
      links: {
        self: `/api/gnn/jobs/${jobId}`,
        results:
          jobStatus.status === "completed"
            ? `/api/gnn/jobs/${jobId}/results`
            : null,
        cancel:
          jobStatus.status === "queued" || jobStatus.status === "processing"
            ? `/api/gnn/jobs/${jobId}`
            : null,
      },
    });
  } catch (error) {
    console.error("Get job status error:", error);

    return NextResponse.json(
      {
        error: "Internal server error",
        message: "An unexpected error occurred",
      },
      { status: 500 },
    );
  }
}

// DELETE /api/gnn/jobs/[jobId] - Cancel job
export async function DELETE(request: NextRequest, { params }: RouteParams) {
  try {
    // Validate API key or session
    const apiKey = request.headers.get("x-api-key");
    const session = await getServerSession();

    if (!apiKey && !session) {
      return NextResponse.json(
        {
          error: "Unauthorized",
          message: "API key or valid session required",
        },
        { status: 401 },
      );
    }

    if (apiKey) {
      const isValid = await validateApiKey(apiKey);
      if (!isValid) {
        return NextResponse.json(
          {
            error: "Invalid API key",
            message: "The provided API key is invalid or expired",
          },
          { status: 401 },
        );
      }
    }

    const { jobId } = params;

    if (!jobId) {
      return NextResponse.json(
        {
          error: "Invalid request",
          message: "Job ID is required",
        },
        { status: 400 },
      );
    }

    const result = await cancelJob(jobId);

    if (!result.success) {
      return NextResponse.json(
        {
          error: "Cancellation failed",
          message: result.message,
        },
        { status: result.status || 400 },
      );
    }

    return NextResponse.json({
      success: true,
      message: `Job ${jobId} cancelled successfully`,
      jobId,
    });
  } catch (error) {
    console.error("Cancel job error:", error);

    return NextResponse.json(
      {
        error: "Internal server error",
        message: "An unexpected error occurred",
      },
      { status: 500 },
    );
  }
}
