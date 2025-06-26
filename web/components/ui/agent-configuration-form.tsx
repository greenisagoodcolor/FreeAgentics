"use client";

import React, { useState, useEffect } from "react";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, Brain, Settings, AlertTriangle } from "lucide-react";
import { AgentTemplate } from "./agent-template-selector";

// Mathematical validation schemas following Active Inference constraints
const MatrixConfigSchema = z.object({
  // A matrix (transition model) - must be stochastic
  aMatrix: z.object({
    rows: z.number().min(2).max(1024),
    cols: z.number().min(2).max(1024),
    stochastic: z.boolean().default(true),
    sparsity: z.number().min(0).max(1).default(0.1),
  }),
  
  // B matrix (observation model) - must be stochastic
  bMatrix: z.object({
    observations: z.number().min(2).max(256),
    states: z.number().min(2).max(1024),
    stochastic: z.boolean().default(true),
    sparsity: z.number().min(0).max(1).default(0.2),
  }),
});

const PrecisionSchema = z.object({
  // Precision parameters (must be positive)
  sensoryPrecision: z.number().min(0.1).max(256).default(16),
  policyPrecision: z.number().min(0.1).max(256).default(16),
  statePrecision: z.number().min(0.1).max(256).default(1),
  
  // Learning rates
  beliefLearningRate: z.number().min(0.001).max(1.0).default(0.1),
  policyLearningRate: z.number().min(0.001).max(1.0).default(0.05),
});

const AgentConfigurationSchema = z.object({
  // Basic agent properties
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  
  // Mathematical configuration
  mathematics: z.object({
    matrices: MatrixConfigSchema,
    precision: PrecisionSchema,
    
    // Prior beliefs (must sum to 1)
    priorBeliefs: z.array(z.number().min(0).max(1)).optional(),
    
    // Time horizon for planning
    planningHorizon: z.number().min(1).max(100).default(5),
    
    // Advanced options
    useHierarchical: z.boolean().default(false),
    enableLearning: z.boolean().default(true),
    memoryCapacity: z.number().min(100).max(100000).default(10000),
  }),
  
  // Environment configuration
  environment: z.object({
    initialPosition: z.object({
      x: z.number(),
      y: z.number(),
      z: z.number().default(0),
    }).optional(),
    energyLevel: z.number().min(0).max(100).default(100),
    tags: z.array(z.string()).default([]),
  }),
  
  // Advanced configuration
  advanced: z.object({
    enableDebugMode: z.boolean().default(false),
    customParameters: z.record(z.any()).optional(),
  }),
});

type AgentConfigurationData = z.infer<typeof AgentConfigurationSchema>;

interface AgentConfigurationFormProps {
  template: AgentTemplate;
  onSubmit: (data: AgentConfigurationData) => void;
  onCancel: () => void;
  isLoading?: boolean;
  className?: string;
}

export function AgentConfigurationForm({
  template,
  onSubmit,
  onCancel,
  isLoading = false,
  className,
}: AgentConfigurationFormProps) {
  const [activeSection, setActiveSection] = useState<string>("basic");
  const [validationIssues, setValidationIssues] = useState<string[]>([]);

  // Initialize form with template defaults
  const form = useForm<AgentConfigurationData>({
    resolver: zodResolver(AgentConfigurationSchema),
    defaultValues: {
      name: `${template.name} ${Math.floor(Math.random() * 1000)}`,
      description: `An ${template.name.toLowerCase()} optimized for ${template.description.toLowerCase()}`,
      mathematics: {
        matrices: {
          aMatrix: {
            rows: template.mathematicalFoundation.beliefsStates,
            cols: template.mathematicalFoundation.beliefsStates,
            stochastic: true,
            sparsity: template.category === "explorer" ? 0.1 : 0.2,
          },
          bMatrix: {
            observations: template.mathematicalFoundation.observationModalities,
            states: template.mathematicalFoundation.beliefsStates,
            stochastic: true,
            sparsity: 0.2,
          },
        },
        precision: {
          sensoryPrecision: template.mathematicalFoundation.defaultPrecision.sensory,
          policyPrecision: template.mathematicalFoundation.defaultPrecision.policy,
          statePrecision: template.mathematicalFoundation.defaultPrecision.state,
          beliefLearningRate: 0.1,
          policyLearningRate: 0.05,
        },
        planningHorizon: template.category === "explorer" ? 3 : 5,
        useHierarchical: template.complexity === "advanced",
        enableLearning: true,
        memoryCapacity: template.mathematicalFoundation.beliefsStates * 100,
      },
      environment: {
        energyLevel: 100,
        tags: [template.category, template.complexity],
      },
      advanced: {
        enableDebugMode: false,
      },
    },
  });

  // Watch for changes to validate mathematical constraints
  const mathematicsData = useWatch({
    control: form.control,
    name: "mathematics",
  });

  // Validate mathematical constraints
  useEffect(() => {
    const issues: string[] = [];
    
    if (mathematicsData) {
      // Check matrix dimensions compatibility
      const aRows = mathematicsData.matrices?.aMatrix.rows;
      const aCols = mathematicsData.matrices?.aMatrix.cols;
      const bStates = mathematicsData.matrices?.bMatrix.states;
      
      if (aRows !== aCols) {
        issues.push("A matrix must be square (rows = columns) for valid transitions");
      }
      
      if (aRows !== bStates) {
        issues.push("A matrix dimensions must match B matrix state space");
      }
      
      // Check precision parameters
      const sensoryPrec = mathematicsData.precision?.sensoryPrecision;
      const policyPrec = mathematicsData.precision?.policyPrecision;
      
      if (sensoryPrec && policyPrec && sensoryPrec > policyPrec * 10) {
        issues.push("Very high sensory precision relative to policy precision may cause instability");
      }
    }
    
    setValidationIssues(issues);
  }, [mathematicsData]);

  const handleSubmit = (data: AgentConfigurationData) => {
    // Additional validation before submission
    if (validationIssues.length > 0) {
      console.warn("Submitting with validation issues:", validationIssues);
    }
    
    onSubmit(data);
  };

  const sections = [
    { id: "basic", label: "Basic Configuration", icon: Settings },
    { id: "mathematics", label: "Mathematical Parameters", icon: Brain },
    { id: "advanced", label: "Advanced Options", icon: Info },
  ];

  return (
    <div className={className}>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
          {/* Header */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Configure {template.name}</h2>
                <p className="text-muted-foreground">
                  Set up mathematical parameters and agent properties
                </p>
              </div>
              <Badge variant="outline" className="px-3 py-1">
                {template.complexity}
              </Badge>
            </div>

            {/* Template Info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  {template.icon}
                  {template.name} Template
                </CardTitle>
                <CardDescription>{template.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Belief States:</span>{" "}
                    {template.mathematicalFoundation.beliefsStates}
                  </div>
                  <div>
                    <span className="font-medium">Modalities:</span>{" "}
                    {template.mathematicalFoundation.observationModalities}
                  </div>
                  <div>
                    <span className="font-medium">Actions:</span>{" "}
                    {template.mathematicalFoundation.actionSpaces}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Validation Issues */}
          {validationIssues.length > 0 && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <div className="space-y-1">
                  <p className="font-semibold">Mathematical Validation Issues:</p>
                  <ul className="list-disc list-inside">
                    {validationIssues.map((issue, index) => (
                      <li key={index} className="text-sm">{issue}</li>
                    ))}
                  </ul>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Section Navigation */}
          <div className="flex space-x-1 border-b">
            {sections.map((section) => (
              <Button
                key={section.id}
                type="button"
                variant={activeSection === section.id ? "default" : "ghost"}
                className="flex items-center gap-2"
                onClick={() => setActiveSection(section.id)}
              >
                <section.icon className="h-4 w-4" />
                {section.label}
              </Button>
            ))}
          </div>

          {/* Basic Configuration */}
          {activeSection === "basic" && (
            <Card>
              <CardHeader>
                <CardTitle>Basic Configuration</CardTitle>
                <CardDescription>
                  Set up the agent&apos;s name, description, and initial environment
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Agent Name</FormLabel>
                      <FormControl>
                        <Input placeholder="Enter agent name" {...field} />
                      </FormControl>
                      <FormDescription>
                        A unique identifier for this agent instance
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description (Optional)</FormLabel>
                      <FormControl>
                        <Textarea
                          placeholder="Describe the agent's purpose and behavior"
                          className="min-h-[80px]"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="environment.initialPosition.x"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Initial X Position</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="0"
                            {...field}
                            onChange={(e) => field.onChange(Number(e.target.value))}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="environment.initialPosition.y"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Initial Y Position</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="0"
                            {...field}
                            onChange={(e) => field.onChange(Number(e.target.value))}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="environment.energyLevel"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Initial Energy Level: {field.value}%</FormLabel>
                      <FormControl>
                        <Slider
                          min={0}
                          max={100}
                          step={1}
                          value={[field.value]}
                          onValueChange={(value) => field.onChange(value[0])}
                          className="w-full"
                        />
                      </FormControl>
                      <FormDescription>
                        Starting energy level for the agent (0-100%)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>
            </Card>
          )}

          {/* Mathematical Parameters */}
          {activeSection === "mathematics" && (
            <div className="space-y-6">
              {/* Precision Parameters */}
              <Card>
                <CardHeader>
                  <CardTitle>Precision Parameters</CardTitle>
                  <CardDescription>
                    Control the agent&apos;s confidence in sensory observations, policies, and state estimates
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <FormField
                    control={form.control}
                    name="mathematics.precision.sensoryPrecision"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Sensory Precision (γ): {field.value.toFixed(1)}</FormLabel>
                        <FormControl>
                          <Slider
                            min={0.1}
                            max={128}
                            step={0.1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="w-full"
                          />
                        </FormControl>
                        <FormDescription>
                          Higher values increase confidence in sensory observations
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="mathematics.precision.policyPrecision"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Policy Precision (β): {field.value.toFixed(1)}</FormLabel>
                        <FormControl>
                          <Slider
                            min={0.1}
                            max={128}
                            step={0.1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="w-full"
                          />
                        </FormControl>
                        <FormDescription>
                          Higher values make policy selection more deterministic
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="mathematics.precision.statePrecision"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>State Precision (α): {field.value.toFixed(1)}</FormLabel>
                        <FormControl>
                          <Slider
                            min={0.1}
                            max={32}
                            step={0.1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="w-full"
                          />
                        </FormControl>
                        <FormDescription>
                          Controls confidence in state transition predictions
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </CardContent>
              </Card>

              {/* Matrix Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle>Generative Model Configuration</CardTitle>
                  <CardDescription>
                    Configure the A and B matrices for the agent&apos;s generative model
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="mathematics.matrices.aMatrix.rows"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>A Matrix Size</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              min={2}
                              max={1024}
                              {...field}
                              onChange={(e) => field.onChange(Number(e.target.value))}
                            />
                          </FormControl>
                          <FormDescription>
                            Transition model dimensions (must be square)
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="mathematics.matrices.bMatrix.observations"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>B Matrix Observations</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              min={2}
                              max={256}
                              {...field}
                              onChange={(e) => field.onChange(Number(e.target.value))}
                            />
                          </FormControl>
                          <FormDescription>
                            Number of observation modalities
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>

                  <FormField
                    control={form.control}
                    name="mathematics.planningHorizon"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Planning Horizon: {field.value} steps</FormLabel>
                        <FormControl>
                          <Slider
                            min={1}
                            max={20}
                            step={1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="w-full"
                          />
                        </FormControl>
                        <FormDescription>
                          Number of time steps to plan ahead
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </CardContent>
              </Card>
            </div>
          )}

          {/* Advanced Options */}
          {activeSection === "advanced" && (
            <Card>
              <CardHeader>
                <CardTitle>Advanced Options</CardTitle>
                <CardDescription>
                  Configure advanced features and debugging options
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <FormField
                  control={form.control}
                  name="mathematics.useHierarchical"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
                      <div className="space-y-0.5">
                        <FormLabel>Hierarchical Active Inference</FormLabel>
                        <FormDescription>
                          Enable hierarchical belief updating for complex environments
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="mathematics.enableLearning"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
                      <div className="space-y-0.5">
                        <FormLabel>Enable Learning</FormLabel>
                        <FormDescription>
                          Allow the agent to update its generative model parameters
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="advanced.enableDebugMode"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
                      <div className="space-y-0.5">
                        <FormLabel>Debug Mode</FormLabel>
                        <FormDescription>
                          Enable detailed logging of belief updates and policy selection
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="mathematics.memoryCapacity"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Memory Capacity: {field.value.toLocaleString()} items</FormLabel>
                      <FormControl>
                        <Slider
                          min={100}
                          max={100000}
                          step={100}
                          value={[field.value]}
                          onValueChange={(value) => field.onChange(value[0])}
                          className="w-full"
                        />
                      </FormControl>
                      <FormDescription>
                        Maximum number of belief states to retain in memory
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex justify-between">
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading || validationIssues.length > 0}
              className="min-w-[120px]"
            >
              {isLoading ? "Creating..." : "Create Agent"}
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}

export { AgentConfigurationSchema };
export type { AgentConfigurationData }; 