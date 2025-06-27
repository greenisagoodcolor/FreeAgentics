import { NextRequest, NextResponse } from "next/server";

// GET /api/gnn/models - List available GNN models
export async function GET(request: NextRequest) {
  return NextResponse.json({
    models: [
      {
        id: "gcn",
        name: "Graph Convolutional Network (GCN)",
        description:
          "Classic GNN architecture that aggregates neighbor features through spectral graph convolutions",
        tasks: ["node_classification", "graph_classification"],
        parameters: {
          hidden_dims: {
            type: "array",
            description: "Hidden layer dimensions",
            default: [64, 32],
            example: [128, 64, 32],
          },
          dropout: {
            type: "number",
            description: "Dropout rate for regularization",
            default: 0.5,
            min: 0,
            max: 1,
          },
          activation: {
            type: "string",
            description: "Activation function",
            default: "relu",
            options: ["relu", "tanh", "sigmoid"],
          },
        },
        strengths: [
          "Simple and efficient",
          "Good for homophilic graphs",
          "Well-studied theoretical properties",
        ],
        limitations: [
          "Limited expressiveness",
          "Assumes equal importance of neighbors",
          "Can suffer from over-smoothing",
        ],
      },
      {
        id: "gat",
        name: "Graph Attention Network (GAT)",
        description:
          "Uses attention mechanisms to learn different weights for different neighbors",
        tasks: [
          "node_classification",
          "graph_classification",
          "link_prediction",
        ],
        parameters: {
          hidden_dims: {
            type: "array",
            description: "Hidden layer dimensions",
            default: [64, 32],
          },
          num_heads: {
            type: "number",
            description: "Number of attention heads",
            default: 4,
            min: 1,
            max: 16,
          },
          dropout: {
            type: "number",
            description: "Dropout rate",
            default: 0.6,
            min: 0,
            max: 1,
          },
          attention_dropout: {
            type: "number",
            description: "Dropout for attention weights",
            default: 0.6,
            min: 0,
            max: 1,
          },
        },
        strengths: [
          "Adaptive neighbor importance",
          "Interpretable attention weights",
          "Good for heterophilic graphs",
        ],
        limitations: [
          "Higher computational cost",
          "More parameters to tune",
          "Can be unstable to train",
        ],
      },
      {
        id: "sage",
        name: "GraphSAGE",
        description:
          "Inductive learning framework that samples and aggregates neighbor features",
        tasks: [
          "node_classification",
          "graph_classification",
          "link_prediction",
        ],
        parameters: {
          hidden_dims: {
            type: "array",
            description: "Hidden layer dimensions",
            default: [64, 32],
          },
          aggregation: {
            type: "string",
            description: "Aggregation method",
            default: "mean",
            options: ["mean", "max", "sum", "lstm"],
          },
          dropout: {
            type: "number",
            description: "Dropout rate",
            default: 0.5,
            min: 0,
            max: 1,
          },
          neighbor_samples: {
            type: "array",
            description: "Number of neighbors to sample per layer",
            default: [25, 10],
            example: [20, 10, 5],
          },
        },
        strengths: [
          "Inductive learning capability",
          "Scalable to large graphs",
          "Flexible aggregation functions",
        ],
        limitations: [
          "Sampling can introduce variance",
          "May miss important distant neighbors",
          "Requires careful hyperparameter tuning",
        ],
      },
      {
        id: "gin",
        name: "Graph Isomorphism Network (GIN)",
        description:
          "Maximally expressive GNN architecture based on the Weisfeiler-Lehman test",
        tasks: ["graph_classification", "node_classification"],
        parameters: {
          hidden_dims: {
            type: "array",
            description: "Hidden layer dimensions",
            default: [64, 32],
          },
          epsilon: {
            type: "number",
            description: "Initial epsilon value",
            default: 0,
            min: -1,
            max: 1,
          },
          dropout: {
            type: "number",
            description: "Dropout rate",
            default: 0.5,
            min: 0,
            max: 1,
          },
          train_epsilon: {
            type: "boolean",
            description: "Whether to make epsilon trainable",
            default: true,
          },
        },
        strengths: [
          "Maximum expressive power",
          "Strong theoretical guarantees",
          "Good for graph-level tasks",
        ],
        limitations: [
          "Can be prone to overfitting",
          "Less interpretable",
          "May require more training data",
        ],
      },
    ],
    taskDescriptions: {
      node_classification: {
        name: "Node Classification",
        description: "Predict labels for individual nodes in the graph",
        examples: [
          "Social network user classification",
          "Protein function prediction",
          "Document categorization in citation networks",
        ],
        inputRequirements: {
          nodeFeatures: "Required - Features for each node",
          nodeLabels: "Required for training - Ground truth labels",
          edges: "Required - Graph connectivity",
        },
        outputFormat: {
          predictions: "Class predictions for each node",
          probabilities: "Class probabilities for each node",
          embeddings: "Optional - Node embeddings",
        },
      },
      graph_classification: {
        name: "Graph Classification",
        description: "Predict labels for entire graphs",
        examples: [
          "Molecular property prediction",
          "Social network community detection",
          "Program analysis and bug detection",
        ],
        inputRequirements: {
          nodeFeatures: "Required - Features for each node",
          graphLabel: "Required for training - Ground truth label",
          edges: "Required - Graph connectivity",
        },
        outputFormat: {
          prediction: "Class prediction for the graph",
          probability: "Confidence score",
          embeddings: "Optional - Graph-level embedding",
        },
      },
      link_prediction: {
        name: "Link Prediction",
        description: "Predict missing or future edges in the graph",
        examples: [
          "Friend recommendation in social networks",
          "Drug-target interaction prediction",
          "Knowledge graph completion",
        ],
        inputRequirements: {
          nodeFeatures: "Required - Features for each node",
          edges: "Required - Known edges",
          candidatePairs: "Optional - Specific pairs to evaluate",
        },
        outputFormat: {
          predictions: "Predicted links with scores",
          scores: "Link probability scores",
          threshold: "Recommended threshold for binary prediction",
        },
      },
    },
    autoSelection: {
      description:
        'When architecture is set to "auto", the system selects the best model based on:',
      criteria: [
        "Graph size and density",
        "Feature dimensionality",
        "Task type",
        "Presence of node/edge attributes",
        "Graph structural properties",
      ],
      heuristics: {
        small_dense: "GCN - Efficient for small, densely connected graphs",
        large_sparse: "GraphSAGE - Scalable sampling-based approach",
        heterophilic: "GAT - Attention helps with diverse neighborhoods",
        graph_level: "GIN - Maximum expressiveness for whole-graph tasks",
      },
    },
  });
}
