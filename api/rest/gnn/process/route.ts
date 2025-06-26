import { validateApiKey } from '@/lib/api-key-service-server';
import { processGraph } from '@/lib/gnn/process-graph';
import { rateLimit } from '@/lib/rate-limit';
import { getServerSession } from 'next-auth';
import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

// Request validation schema
const processRequestSchema = z.object({
  graph: z.object({
    nodes: z.array(z.object({
      id: z.string(),
      features: z.record(z.any()),
      position: z.object({
        lat: z.number().optional(),
        lon: z.number().optional(),
        x: z.number().optional(),
        y: z.number().optional(),
        z: z.number().optional(),
      }).optional(),
    })),
    edges: z.array(z.object({
      source: z.string(),
      target: z.string(),
      weight: z.number().optional(),
      type: z.string().optional(),
      attributes: z.record(z.any()).optional(),
    })),
  }),
  model: z.object({
    architecture: z.enum(['GCN', 'GAT', 'SAGE', 'GIN', 'auto']).default('auto'),
    task: z.enum(['node_classification', 'graph_classification', 'link_prediction']),
    config: z.object({
      hidden_dims: z.array(z.number()).optional(),
      dropout: z.number().min(0).max(1).optional(),
      num_heads: z.number().optional(),
      aggregation: z.enum(['mean', 'max', 'sum']).optional(),
    }).optional(),
  }),
  options: z.object({
    batch_size: z.number().min(1).max(1000).default(32),
    return_embeddings: z.boolean().default(false),
    return_attention_weights: z.boolean().default(false),
  }).optional(),
});

// Rate limiter instance
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});

export async function POST(request: NextRequest) {
  try {
    // Check rate limiting
    const identifier = request.headers.get('x-api-key') ||
                      request.headers.get('x-forwarded-for') ||
                      'anonymous';

    try {
      await limiter.check(10, identifier); // 10 requests per minute
    } catch {
      return NextResponse.json(
        {
          error: 'Too many requests',
          message: 'Rate limit exceeded. Please try again later.',
        },
        { status: 429 }
      );
    }

    // Validate API key or session
    const apiKey = request.headers.get('x-api-key');
    const session = await getServerSession();

    if (!apiKey && !session) {
      return NextResponse.json(
        {
          error: 'Unauthorized',
          message: 'API key or valid session required',
        },
        { status: 401 }
      );
    }

    if (apiKey) {
      const isValid = await validateApiKey(apiKey);
      if (!isValid) {
        return NextResponse.json(
          {
            error: 'Invalid API key',
            message: 'The provided API key is invalid or expired',
          },
          { status: 401 }
        );
      }
    }

    // Parse and validate request body
    const body = await request.json();
    const validationResult = processRequestSchema.safeParse(body);

    if (!validationResult.success) {
      return NextResponse.json(
        {
          error: 'Invalid request',
          message: 'Request validation failed',
          details: validationResult.error.errors,
        },
        { status: 400 }
      );
    }

    const { graph, model, options } = validationResult.data;

    // Validate graph structure
    if (graph.nodes.length === 0) {
      return NextResponse.json(
        {
          error: 'Invalid graph',
          message: 'Graph must contain at least one node',
        },
        { status: 400 }
      );
    }

    // Validate edges reference valid nodes
    const nodeIds = new Set(graph.nodes.map(n => n.id));
    for (const edge of graph.edges) {
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
        return NextResponse.json(
          {
            error: 'Invalid graph',
            message: `Edge references non-existent node: ${edge.source} -> ${edge.target}`,
          },
          { status: 400 }
        );
      }
    }

    // Process the graph
    try {
      const result = await processGraph({
        graph,
        model,
        options: options || {},
      });

      return NextResponse.json({
        success: true,
        jobId: result.jobId,
        status: result.status,
        message: 'Graph processing initiated successfully',
        estimatedTime: result.estimatedTime,
        links: {
          status: `/api/gnn/jobs/${result.jobId}`,
          results: `/api/gnn/jobs/${result.jobId}/results`,
        },
      });
    } catch (error) {
      console.error('Graph processing error:', error);

      return NextResponse.json(
        {
          error: 'Processing failed',
          message: error instanceof Error ? error.message : 'An unexpected error occurred',
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('API error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error',
        message: 'An unexpected error occurred',
      },
      { status: 500 }
    );
  }
}

// API documentation
export async function GET(request: NextRequest) {
  return NextResponse.json({
    endpoint: '/api/gnn/process',
    method: 'POST',
    description: 'Process a graph using GNN models',
    authentication: 'API key (header: x-api-key) or session required',
    rateLimits: {
      requests: 10,
      window: '1 minute',
    },
    request: {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': 'Your API key (optional if authenticated)',
      },
      body: {
        graph: {
          nodes: [
            {
              id: 'string',
              features: 'object (key-value pairs)',
              position: {
                lat: 'number (optional)',
                lon: 'number (optional)',
                x: 'number (optional)',
                y: 'number (optional)',
                z: 'number (optional)',
              },
            },
          ],
          edges: [
            {
              source: 'string (node id)',
              target: 'string (node id)',
              weight: 'number (optional, default: 1.0)',
              type: 'string (optional)',
              attributes: 'object (optional)',
            },
          ],
        },
        model: {
          architecture: 'GCN | GAT | SAGE | GIN | auto (default: auto)',
          task: 'node_classification | graph_classification | link_prediction',
          config: {
            hidden_dims: 'number[] (optional)',
            dropout: 'number (0-1, optional)',
            num_heads: 'number (for GAT, optional)',
            aggregation: 'mean | max | sum (for SAGE, optional)',
          },
        },
        options: {
          batch_size: 'number (1-1000, default: 32)',
          return_embeddings: 'boolean (default: false)',
          return_attention_weights: 'boolean (default: false)',
        },
      },
    },
    response: {
      success: {
        status: 200,
        body: {
          success: true,
          jobId: 'string',
          status: 'queued | processing | completed | failed',
          message: 'string',
          estimatedTime: 'number (seconds)',
          links: {
            status: 'string (URL)',
            results: 'string (URL)',
          },
        },
      },
      errors: {
        400: 'Invalid request or graph structure',
        401: 'Unauthorized - API key or session required',
        429: 'Rate limit exceeded',
        500: 'Internal server error',
      },
    },
    examples: {
      nodeClassification: {
        graph: {
          nodes: [
            { id: '1', features: { degree: 3, pagerank: 0.15 } },
            { id: '2', features: { degree: 2, pagerank: 0.10 } },
          ],
          edges: [
            { source: '1', target: '2', weight: 0.8 },
          ],
        },
        model: {
          architecture: 'GCN',
          task: 'node_classification',
          config: {
            hidden_dims: [64, 32],
            dropout: 0.5,
          },
        },
      },
    },
  });
}
