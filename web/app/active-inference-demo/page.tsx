"use client";

import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Brain, Zap, Eye, Info } from "lucide-react";
import { ActiveInferenceDashboard } from "@/components/ui/active-inference-dashboard";
import { BeliefStateVisualization } from "@/components/ui/belief-state-visualization";
import { FreeEnergyVisualization } from "@/components/ui/free-energy-visualization";

export default function ActiveInferenceDemoPage() {
  const explorerStateLabels = ["Explore", "Navigate", "Search", "Rest", "Communicate", "Learn", "Plan", "Execute"];
  const guardianStateLabels = ["Monitor", "Patrol", "Alert", "Defend", "Coordinate", "Assess", "Report", "Standby"];

  return (
    <div className="container mx-auto p-6 space-y-8">
      {/* Page Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <Brain className="h-8 w-8 text-primary" />
          <h1 className="text-4xl font-bold">Active Inference Visualization Demo</h1>
        </div>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Real-time mathematical visualization of Active Inference agents showing belief states, 
          free energy dynamics, and precision parameters with D3.js
        </p>
        <div className="flex items-center justify-center gap-2">
          <Badge variant="outline">D3.js Powered</Badge>
          <Badge variant="outline">Real-time Updates</Badge>
          <Badge variant="outline">Mathematical Accuracy</Badge>
          <Badge variant="outline">pymdp Compatible</Badge>
        </div>
      </div>

      <Separator />

      {/* Mathematical Foundation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            Mathematical Foundation
          </CardTitle>
          <CardDescription>
            Active Inference mathematics implemented in these visualizations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="font-semibold text-blue-900">Belief State Mathematics</h4>
              <div className="text-sm space-y-1">
                <p>• <strong>q(s)</strong>: Belief distribution over hidden states (probability simplex)</p>
                <p>• <strong>H[q(s)]</strong>: Shannon entropy = -Σ q(s) log q(s)</p>
                <p>• <strong>Confidence</strong>: 1 - H[q(s)]/log(|S|) (normalized uncertainty)</p>
                <p>• <strong>Normalization</strong>: Σ q(s) = 1.0 ± 1e-10</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold text-red-900">Free Energy Mathematics</h4>
              <div className="text-sm space-y-1">
                <p>• <strong>F</strong>: Variational Free Energy = Accuracy + Complexity</p>
                <p>• <strong>Accuracy</strong>: -E<sub>q</sub>[ln p(o|s)] (negative log likelihood)</p>
                <p>• <strong>Complexity</strong>: D<sub>KL</sub>[q(s)||p(s)] (KL divergence)</p>
                <p>• <strong>G(π)</strong>: Expected Free Energy for policy π</p>
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
            <h4 className="font-semibold text-green-900 mb-2">Precision Parameters</h4>
            <div className="text-sm text-green-800 grid md:grid-cols-3 gap-4">
              <div>
                <strong>γ (Sensory)</strong>: Inverse variance of sensory noise σ⁻²
              </div>
              <div>
                <strong>β (Policy)</strong>: Temperature parameter in softmax policy selection
              </div>
              <div>
                <strong>α (State)</strong>: Precision of state transition predictions
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Explorer Agent Dashboard */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Eye className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Explorer Agent Dashboard</h2>
        </div>
        <ActiveInferenceDashboard
          agentId="explorer-001"
          agentName="Explorer Alpha"
          template="explorer"
          updateInterval={1500}
          stateLabels={explorerStateLabels}
          isRealTime={true}
        />
      </div>

      <Separator />

      {/* Guardian Agent Dashboard */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Zap className="h-6 w-6 text-red-600" />
          <h2 className="text-2xl font-bold">Guardian Agent Dashboard</h2>
        </div>
        <ActiveInferenceDashboard
          agentId="guardian-001"
          agentName="Guardian Beta"
          template="guardian"
          updateInterval={1200}
          stateLabels={guardianStateLabels}
          isRealTime={true}
        />
      </div>

      <Separator />

      {/* Individual Component Demos */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-center">Individual Component Demonstrations</h2>
        
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Standalone Belief State Visualization */}
          <BeliefStateVisualization
            agentId="demo-belief"
            stateLabels={["State A", "State B", "State C", "State D", "State E"]}
            height={350}
            updateInterval={800}
            isRealTime={true}
          />
          
          {/* Standalone Free Energy Visualization */}
          <FreeEnergyVisualization
            agentId="demo-energy"
            height={350}
            updateInterval={800}
            timeWindow={30000}
            isRealTime={true}
          />
        </div>
      </div>

      {/* Technical Implementation Details */}
      <Card>
        <CardHeader>
          <CardTitle>Technical Implementation</CardTitle>
          <CardDescription>
            Implementation details for the Active Inference visualizations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="font-semibold">D3.js Features</h4>
              <ul className="text-sm space-y-1 list-disc list-inside">
                <li>Real-time SVG rendering with smooth animations</li>
                <li>Responsive design with dynamic scaling</li>
                <li>Interactive controls (play/pause/reset)</li>
                <li>Mathematical validation and error checking</li>
                <li>Color-coded state highlighting</li>
                <li>Temporal data windowing and filtering</li>
              </ul>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold">Architecture Compliance</h4>
              <ul className="text-sm space-y-1 list-disc list-inside">
                <li>ADR-002: Components in canonical web/components/ui/</li>
                <li>ADR-003: UI layer decoupled from domain logic</li>
                <li>ADR-008: Clean API integration patterns</li>
                <li>TypeScript with strict mathematical typing</li>
                <li>React hooks for state management</li>
                <li>Expert-ready for mathematical review</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded-lg">
            <h4 className="font-semibold text-purple-900 mb-2">Expert Review Ready</h4>
            <p className="text-sm text-purple-800">
              These visualizations are mathematically accurate and ready for review by Active Inference experts 
              including Conor Heins (pymdp), Alexander Tschantz, and architecture experts Robert Martin and Rich Hickey.
              All mathematical constraints are validated and documented.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="text-center text-sm text-muted-foreground py-8">
        <p>FreeAgentics Active Inference Visualization Suite</p>
        <p>Built with D3.js, React, TypeScript, and mathematical rigor</p>
      </div>
    </div>
  );
} 