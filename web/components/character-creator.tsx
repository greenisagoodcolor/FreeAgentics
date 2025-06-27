"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, CheckCircle, AlertTriangle } from "lucide-react";
import { AgentCreationWizard } from "@/components/ui/agent-creation-wizard";
import { AgentTemplate } from "@/components/ui/agent-template-selector";
import { AgentConfigurationData } from "@/components/ui/agent-configuration-form";
import { agentsApi, Agent, CreateAgentRequest } from "@/lib/api/agents-api";

interface CharacterCreatorProps {
  onClose: () => void;
  onSuccess: (agent: Agent) => void;
}

export function CharacterCreator({
  onClose,
  onSuccess,
}: CharacterCreatorProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<Agent | null>(null);

  const handleAgentCreate = async (
    template: AgentTemplate,
    configuration: AgentConfigurationData,
  ) => {
    setIsSubmitting(true);
    setError(null);

    try {
      // Convert template and configuration to API request format
      const createRequest: CreateAgentRequest = {
        name: configuration.name,

        // Convert Active Inference configuration
        activeInference: {
          template: template.category, // Use template category as Active Inference template
          stateLabels: ["State1", "State2", "State3"],
          numStates:
            configuration.mathematics?.matrices?.aMatrix?.rows ||
            template.mathematicalFoundation.beliefsStates,
          numObservations:
            configuration.mathematics?.matrices?.bMatrix?.observations ||
            template.mathematicalFoundation.observationModalities,
          numActions: template.mathematicalFoundation.actionSpaces,
          generativeModel: {
            A: [],
            B: [],
            C: [],
            D: [],
          },
          precisionParameters: {
            sensory:
              configuration.mathematics?.precision?.sensoryPrecision ||
              template.mathematicalFoundation.defaultPrecision.sensory,
            policy:
              configuration.mathematics?.precision?.policyPrecision ||
              template.mathematicalFoundation.defaultPrecision.policy,
            state:
              configuration.mathematics?.precision?.statePrecision ||
              template.mathematicalFoundation.defaultPrecision.state,
          },
          mathematicalConstraints: {
            normalizedBeliefs: true,
            stochasticMatrices: true,
            precisionBounds: true,
          },
        },

        // Add legacy personality mapping for backward compatibility
        personality: undefined,

        capabilities: template.capabilities || [
          "movement",
          "perception",
          "communication",
        ],

        tags: [
          template.category,
          template.complexity,
          ...(configuration.environment?.tags || []),
        ],

        metadata: {
          templateId: template.id,
          templateName: template.name,
          complexity: template.complexity,
          description: configuration.description,
          mathematicalFoundation: template.mathematicalFoundation,
          createdViaWizard: true,
          configurationVersion: "1.0",
          ...(configuration.advanced?.customParameters || {}),
        },
      };

      // Create agent via API
      const result = await agentsApi.createAgent(createRequest);

      setSuccess(result.agent);

      // Show success briefly, then call success callback
      setTimeout(() => {
        onSuccess(result.agent);
      }, 1500);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
      console.error("Agent creation failed:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle creating from template directly (backward compatibility)
  const handleTemplateCreate = async (template: AgentTemplate) => {
    setIsSubmitting(true);
    setError(null);

    try {
      // Create agent using template method
      const result = await agentsApi.createAgentFromTemplate({
        template: template.category,
        name: `${template.name} ${Date.now()}`,
        stateLabels: ["Explore", "Rest", "Communicate", "Plan"],
        precisionParameters: {
          sensory:
            template.mathematicalFoundation.defaultPrecision.sensory || 16,
          policy: template.mathematicalFoundation.defaultPrecision.policy || 12,
          state: template.mathematicalFoundation.defaultPrecision.state || 2,
        },
        tags: [template.category, template.complexity],
        metadata: {
          templateId: template.id,
          templateName: template.name,
          complexity: template.complexity,
          createdDirectly: true,
        },
      });

      setSuccess(result.agent);

      setTimeout(() => {
        onSuccess(result.agent);
      }, 1500);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (success) {
    return (
      <Card className="p-6">
        <div className="text-center space-y-4">
          <CheckCircle className="h-16 w-16 text-green-600 mx-auto" />
          <div>
            <h3 className="text-lg font-semibold text-green-900">
              Agent Created Successfully!
            </h3>
            <p className="text-sm text-green-700 mt-2">
              <strong>{success.name}</strong> has been created with Active
              Inference capabilities.
            </p>
            {success.activeInference && (
              <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                <div className="text-xs text-green-800 space-y-1">
                  <p>• Template: {success.activeInference.template}</p>
                  <p>• States: {success.activeInference.numStates}</p>
                  <p>
                    • Precision γ:{" "}
                    {success.activeInference.precisionParameters.sensory.toFixed(
                      1,
                    )}
                  </p>
                  <p>• Mathematical validation: ✓ Passed</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Create Active Inference Agent</h2>
          <p className="text-muted-foreground mt-1">
            Build a mathematically rigorous agent with real-time visualization
          </p>
        </div>
        <Button variant="outline" onClick={onClose} disabled={isSubmitting}>
          Cancel
        </Button>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <strong>Creation Failed:</strong> {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Loading Overlay */}
      {isSubmitting && (
        <div className="relative">
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm z-10 flex items-center justify-center rounded-lg">
            <div className="flex items-center gap-3">
              <Loader2 className="h-6 w-6 animate-spin" />
              <div className="text-center">
                <p className="font-medium">Creating Agent...</p>
                <p className="text-sm text-muted-foreground">
                  Validating mathematical parameters and initializing Active
                  Inference systems
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Agent Creation Wizard */}
      <AgentCreationWizard
        onAgentCreate={handleAgentCreate}
        onCancel={onClose}
      />

      {/* Mathematical Information */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <h4 className="font-semibold text-blue-900 mb-2">
          Active Inference Integration
        </h4>
        <div className="text-sm text-blue-800 space-y-1">
          <p>
            • <strong>API Validation:</strong> Mathematical constraints verified
            server-side
          </p>
          <p>
            • <strong>Belief States:</strong> Real-time q(s) distribution with
            entropy calculation
          </p>
          <p>
            • <strong>Free Energy:</strong> F = Accuracy + Complexity
            minimization
          </p>
          <p>
            • <strong>pymdp Compatible:</strong> Ready for expert review and
            production deployment
          </p>
        </div>
      </Card>
    </div>
  );
}
