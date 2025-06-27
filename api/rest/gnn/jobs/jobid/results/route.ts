import { validateApiKey } from "@/lib/api-key-service-server";
import { getJobResults } from "@/lib/gnn/job-manager";
import { getServerSession } from "next-auth";
import { NextRequest, NextResponse } from "next/server";

interface RouteParams {
  params: {
    jobId: string;
  };
}

// GET /api/gnn/jobs/[jobId]/results - Get job results
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

    const results = await getJobResults(jobId);

    if (!results) {
      return NextResponse.json(
        {
          error: "Not found",
          message: `Results for job ${jobId} not found. The job may still be processing or may have failed.`,
        },
        { status: 404 },
      );
    }

    // Check if results are ready
    if (results.status !== "completed") {
      return NextResponse.json(
        {
          error: "Results not ready",
          message: `Job ${jobId} is still ${results.status}. Results are only available for completed jobs.`,
          status: results.status,
          links: {
            status: `/api/gnn/jobs/${jobId}`,
          },
        },
        { status: 202 },
      );
    }

    // Format results based on task type
    const formattedResults = formatResults(results);

    return NextResponse.json({
      jobId,
      status: "completed",
      task: results.task,
      model: {
        architecture: results.modelArchitecture,
        config: results.modelConfig,
      },
      results: formattedResults,
      metadata: {
        processingTime: results.processingTime,
        graphStats: {
          nodes: results.graphNodes,
          edges: results.graphEdges,
        },
        timestamp: results.completedAt,
      },
      links: {
        self: `/api/gnn/jobs/${jobId}/results`,
        status: `/api/gnn/jobs/${jobId}`,
      },
    });
  } catch (error) {
    console.error("Get job results error:", error);

    return NextResponse.json(
      {
        error: "Internal server error",
        message: "An unexpected error occurred",
      },
      { status: 500 },
    );
  }
}

// Helper function to format results based on task type
function formatResults(results: any) {
  const { task, predictions, embeddings, attentionWeights, metrics } = results;

  switch (task) {
    case "node_classification":
      return {
        predictions: predictions
          ? {
              nodes: predictions.nodes || [],
              classes: predictions.classes || [],
              probabilities: predictions.probabilities || [],
            }
          : null,
        embeddings: embeddings || null,
        attentionWeights: attentionWeights || null,
        metrics: metrics || {
          accuracy: null,
          precision: null,
          recall: null,
          f1Score: null,
        },
      };

    case "graph_classification":
      return {
        prediction: predictions
          ? {
              class: predictions.class,
              probability: predictions.probability,
              allProbabilities: predictions.allProbabilities || [],
            }
          : null,
        embeddings: embeddings || null,
        metrics: metrics || {
          confidence: predictions?.probability || null,
        },
      };

    case "link_prediction":
      return {
        predictions: predictions
          ? {
              links: predictions.links || [],
              scores: predictions.scores || [],
              threshold: predictions.threshold || 0.5,
            }
          : null,
        embeddings: embeddings || null,
        metrics: metrics || {
          auc: null,
          precision: null,
          recall: null,
        },
      };

    default:
      return {
        raw: results.predictions,
        embeddings: embeddings || null,
        attentionWeights: attentionWeights || null,
        metrics: metrics || {},
      };
  }
}
