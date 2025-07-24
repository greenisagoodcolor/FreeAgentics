"use client";

import { useState } from "react";
import Link from "next/link";

interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  type: "explorer" | "optimizer" | "predictor";
  complexity: "simple" | "moderate" | "complex";
}

const agentTemplates: AgentTemplate[] = [
  {
    id: "basic-explorer",
    name: "Basic Explorer",
    description: "Simple agent that explores the environment using active inference",
    type: "explorer",
    complexity: "simple",
  },
  {
    id: "goal-optimizer",
    name: "Goal Optimizer",
    description: "Agent focused on optimizing specific objectives",
    type: "optimizer",
    complexity: "moderate",
  },
  {
    id: "pattern-predictor",
    name: "Pattern Predictor",
    description: "Advanced agent that learns and predicts environmental patterns",
    type: "predictor",
    complexity: "complex",
  },
];

export default function AgentsPage() {
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [agentName, setAgentName] = useState("");
  const [creating, setCreating] = useState(false);

  const handleCreateAgent = async () => {
    if (!selectedTemplate || !agentName) return;

    setCreating(true);
    // Simulate agent creation
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // In real implementation, this would call the API
    alert(`Agent "${agentName}" would be created with template: ${selectedTemplate}`);
    setCreating(false);
    setAgentName("");
    setSelectedTemplate(null);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Agent Explorer</h1>
        <Link href="/" className="text-blue-600 hover:text-blue-800 underline">
          ← Back to Home
        </Link>
      </div>

      {/* Create Agent Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Create New Agent</h2>

        {/* Agent Name Input */}
        <div className="mb-6">
          <label htmlFor="agent-name" className="block text-sm font-medium text-gray-700 mb-2">
            Agent Name
          </label>
          <input
            id="agent-name"
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter agent name..."
          />
        </div>

        {/* Template Selection */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Select Template</h3>
          <div className="grid md:grid-cols-3 gap-4">
            {agentTemplates.map((template) => (
              <button
                key={template.id}
                onClick={() => setSelectedTemplate(template.id)}
                className={`p-4 border rounded-lg text-left transition-all ${
                  selectedTemplate === template.id
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <h4 className="font-semibold text-gray-900 mb-1">{template.name}</h4>
                <p className="text-sm text-gray-600 mb-2">{template.description}</p>
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      template.type === "explorer"
                        ? "bg-green-100 text-green-800"
                        : template.type === "optimizer"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-purple-100 text-purple-800"
                    }`}
                  >
                    {template.type}
                  </span>
                  <span
                    className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      template.complexity === "simple"
                        ? "bg-gray-100 text-gray-800"
                        : template.complexity === "moderate"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-red-100 text-red-800"
                    }`}
                  >
                    {template.complexity}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Create Button */}
        <button
          onClick={handleCreateAgent}
          disabled={!selectedTemplate || !agentName || creating}
          className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${
            selectedTemplate && agentName && !creating
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "bg-gray-300 text-gray-500 cursor-not-allowed"
          }`}
        >
          {creating ? "Creating Agent..." : "Create Agent"}
        </button>
      </div>

      {/* Existing Agents */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Existing Agents</h2>
        <div className="text-center py-8 text-gray-500">
          <p>No agents have been created yet</p>
        </div>
      </div>

      {/* Development Notice */}
      <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-800">
          <strong>Note:</strong> Agent creation is currently in demo mode. Active Inference engine
          integration is in progress.
        </p>
      </div>
    </div>
  );
}
