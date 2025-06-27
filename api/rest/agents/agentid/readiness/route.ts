import { NextRequest, NextResponse } from "next/server";

// Mock readiness data for demonstration
const mockReadinessScores: Record<string, any> = {
  agent_123: {
    agent_id: "agent_123",
    timestamp: new Date().toISOString(),
    scores: {
      knowledge_maturity: 0.85,
      goal_achievement: 0.92,
      model_stability: 0.88,
      collaboration: 0.79,
      resource_management: 0.83,
      overall: 0.854,
    },
    is_ready: true,
    metrics: {
      knowledge: {
        experience_count: 1200,
        pattern_count: 60,
        avg_pattern_confidence: 0.87,
      },
      goals: {
        total_attempted: 100,
        successful: 92,
        success_rate: 0.92,
        complex_completed: 6,
      },
      model_stability: {
        update_count: 150,
        is_converged: true,
        stable_iterations: 120,
      },
      collaboration: {
        total_interactions: 50,
        successful_interactions: 45,
        knowledge_shared: 15,
        unique_collaborators: 3,
      },
      resources: {
        energy_efficiency: 0.85,
        resource_efficiency: 0.85,
        sustainability_score: 0.88,
      },
    },
    recommendations: [],
  },
  agent_456: {
    agent_id: "agent_456",
    timestamp: new Date().toISOString(),
    scores: {
      knowledge_maturity: 0.65,
      goal_achievement: 0.72,
      model_stability: 0.58,
      collaboration: 0.45,
      resource_management: 0.68,
      overall: 0.616,
    },
    is_ready: false,
    metrics: {
      knowledge: {
        experience_count: 650,
        pattern_count: 30,
        avg_pattern_confidence: 0.72,
      },
      goals: {
        total_attempted: 50,
        successful: 36,
        success_rate: 0.72,
        complex_completed: 2,
      },
      model_stability: {
        update_count: 80,
        is_converged: false,
        stable_iterations: 45,
      },
      collaboration: {
        total_interactions: 20,
        successful_interactions: 12,
        knowledge_shared: 5,
        unique_collaborators: 2,
      },
      resources: {
        energy_efficiency: 0.7,
        resource_efficiency: 0.65,
        sustainability_score: 0.69,
      },
    },
    recommendations: [
      "Gain more experience: 650/1000 experiences",
      "Extract more patterns: 30/50 patterns",
      "Model has not converged - continue training",
      "Need more stable iterations: 45/100",
      "Engage in more successful collaborations",
      "Share more knowledge with other agents",
    ],
  },
};

export async function GET(
  request: NextRequest,
  { params }: { params: { agentId: string } },
) {
  try {
    const agentId = params.agentId;

    // In a real implementation, this would:
    // 1. Fetch the agent from database
    // 2. Run the readiness evaluator
    // 3. Return the results

    // For now, return mock data
    const readinessScore = mockReadinessScores[agentId] || {
      agent_id: agentId,
      timestamp: new Date().toISOString(),
      scores: {
        knowledge_maturity: Math.random() * 0.5 + 0.5,
        goal_achievement: Math.random() * 0.5 + 0.5,
        model_stability: Math.random() * 0.5 + 0.5,
        collaboration: Math.random() * 0.5 + 0.5,
        resource_management: Math.random() * 0.5 + 0.5,
        overall: 0,
      },
      is_ready: false,
      metrics: {
        knowledge: {
          experience_count: Math.floor(Math.random() * 1000),
          pattern_count: Math.floor(Math.random() * 50),
          avg_pattern_confidence: Math.random() * 0.3 + 0.6,
        },
        goals: {
          total_attempted: Math.floor(Math.random() * 100),
          successful: Math.floor(Math.random() * 80),
          success_rate: Math.random() * 0.3 + 0.6,
          complex_completed: Math.floor(Math.random() * 5),
        },
        model_stability: {
          update_count: Math.floor(Math.random() * 100 + 50),
          is_converged: Math.random() > 0.5,
          stable_iterations: Math.floor(Math.random() * 100),
        },
        collaboration: {
          total_interactions: Math.floor(Math.random() * 50),
          successful_interactions: Math.floor(Math.random() * 40),
          knowledge_shared: Math.floor(Math.random() * 20),
          unique_collaborators: Math.floor(Math.random() * 5),
        },
        resources: {
          energy_efficiency: Math.random() * 0.3 + 0.6,
          resource_efficiency: Math.random() * 0.3 + 0.6,
          sustainability_score: Math.random() * 0.3 + 0.6,
        },
      },
      recommendations: [],
    };

    // Calculate overall score if not set
    if (readinessScore.scores.overall === 0) {
      const scores = readinessScore.scores;
      const weights = [0.25, 0.2, 0.2, 0.2, 0.15];
      scores.overall =
        scores.knowledge_maturity * weights[0] +
        scores.goal_achievement * weights[1] +
        scores.model_stability * weights[2] +
        scores.collaboration * weights[3] +
        scores.resource_management * weights[4];

      readinessScore.is_ready = scores.overall >= 0.85;
    }

    // Generate recommendations if not ready
    if (
      !readinessScore.is_ready &&
      readinessScore.recommendations.length === 0
    ) {
      const recs = [];

      if (readinessScore.scores.knowledge_maturity < 0.8) {
        recs.push(
          "Increase knowledge maturity through more experiences and pattern extraction",
        );
      }
      if (readinessScore.scores.goal_achievement < 0.8) {
        recs.push("Improve goal success rate and complete more complex goals");
      }
      if (readinessScore.scores.model_stability < 0.8) {
        recs.push(
          "Continue training until model converges with stable iterations",
        );
      }
      if (readinessScore.scores.collaboration < 0.8) {
        recs.push(
          "Engage in more collaborative interactions and knowledge sharing",
        );
      }
      if (readinessScore.scores.resource_management < 0.8) {
        recs.push("Optimize resource usage for better efficiency");
      }

      readinessScore.recommendations = recs;
    }

    return NextResponse.json(readinessScore);
  } catch (error) {
    console.error("Failed to get readiness score:", error);
    return NextResponse.json(
      { error: "Failed to get readiness score" },
      { status: 500 },
    );
  }
}
