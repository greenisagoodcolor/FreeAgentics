"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { usePromptProcessor } from "../hooks/use-prompt-processor";
import { AgentVisualization } from "./agent-visualization";
import { KnowledgeGraphView } from "./knowledge-graph-view";
import { SuggestionsList } from "./suggestions-list";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Alert } from "./ui/alert";
import { Card } from "./ui/card";

interface PromptInterfaceProps {
  className?: string;
}

export function PromptInterface({ className = "" }: PromptInterfaceProps) {
  const [prompt, setPrompt] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    isLoading,
    error,
    agents,
    knowledgeGraph,
    suggestions,
    submitPrompt,
    retry,
    fetchSuggestions,
    iterationContext,
    resetConversation,
  } = usePromptProcessor();

  // Check for mobile viewport
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  // Handle prompt submission
  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();

      if (!prompt.trim() || isLoading) return;

      const trimmedPrompt = prompt.trim();
      await submitPrompt(trimmedPrompt);
      setPrompt("");
      setShowSuggestions(false);
    },
    [prompt, isLoading, submitPrompt],
  );

  // Handle input changes with debounced suggestions
  const handleInputChange = useCallback(
    (value: string) => {
      setPrompt(value);

      if (value.trim().length > 2) {
        setShowSuggestions(true);
        fetchSuggestions(value);
      } else {
        setShowSuggestions(false);
      }
    },
    [fetchSuggestions],
  );

  // Handle suggestion selection
  const handleSuggestionSelect = useCallback((suggestion: string) => {
    setPrompt(suggestion);
    setShowSuggestions(false);
    textareaRef.current?.focus();
  }, []);

  // Handle keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const containerClasses = `
    flex ${isMobile ? "flex-col" : "flex-row"}
    gap-4 p-4 min-h-screen bg-gray-50
    ${className}
  `;

  return (
    <div className={containerClasses} data-testid="prompt-interface-container">
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Iteration Context Card */}
        {iterationContext && (
          <Card className="p-4 bg-blue-50 border-blue-200">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-sm font-medium text-blue-900">
                  Iteration {iterationContext.iteration_number} of Conversation
                </h3>
                <p className="text-xs text-blue-700">
                  {iterationContext.total_agents} agents created • {iterationContext.kg_nodes}{" "}
                  knowledge nodes
                </p>
                {iterationContext.conversation_summary.belief_evolution && (
                  <p className="text-xs text-blue-600">
                    Belief stability:{" "}
                    {(
                      iterationContext.conversation_summary.belief_evolution.stability * 100
                    ).toFixed(0)}
                    % • Trend: {iterationContext.conversation_summary.belief_evolution.trend}
                  </p>
                )}
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={resetConversation}
                className="text-blue-700 hover:text-blue-900"
              >
                New Conversation
              </Button>
            </div>
          </Card>
        )}

        {/* Prompt Input Card */}
        <Card className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <label htmlFor="prompt-input" className="sr-only">
                Enter your prompt
              </label>
              <Textarea
                id="prompt-input"
                ref={textareaRef}
                value={prompt}
                onChange={(e) => handleInputChange(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="What would you like to explore today?"
                className="min-h-[100px] resize-none pr-24"
                aria-label="Enter your prompt"
                aria-describedby={error ? "error-message" : undefined}
                disabled={isLoading}
              />

              {/* Submit Button */}
              <div className="absolute bottom-2 right-2">
                <Button
                  type="submit"
                  disabled={!prompt.trim() || isLoading}
                  aria-label="Send prompt"
                  size={isMobile ? "sm" : "default"}
                >
                  {isLoading ? "Processing..." : "Send"}
                </Button>
              </div>
            </div>

            {/* Suggestions Dropdown */}
            {showSuggestions && suggestions.length > 0 && (
              <SuggestionsList
                suggestions={suggestions}
                onSelect={handleSuggestionSelect}
                onClose={() => setShowSuggestions(false)}
              />
            )}
          </form>

          {/* Error Display */}
          {error && (
            <Alert id="error-message" role="alert" className="mt-4" variant="destructive">
              <div className="flex items-center justify-between">
                <span>{error}</span>
                <Button onClick={retry} variant="outline" size="sm" aria-label="Retry">
                  Retry
                </Button>
              </div>
            </Alert>
          )}

          {/* Loading State */}
          {isLoading && (
            <div role="status" aria-live="polite" className="mt-4">
              <div className="flex items-center justify-center p-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-sm text-muted-foreground">Processing your prompt...</p>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Intelligent Suggestions from Iterative Controller */}
        {suggestions.length > 0 && !showSuggestions && (
          <Card className="p-4 bg-green-50 border-green-200">
            <h3 className="text-sm font-medium text-green-900 mb-2">Suggested Next Actions</h3>
            <div className="space-y-2">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionSelect(suggestion)}
                  className="w-full text-left text-sm p-2 rounded hover:bg-green-100 transition-colors text-green-800"
                >
                  • {suggestion}
                </button>
              ))}
            </div>
          </Card>
        )}

        {/* Agent Visualization */}
        <Card className="flex-1 p-6">
          <div role="region" aria-label="Agent visualization" className="h-full min-h-[400px]">
            <h2 className="text-xl font-semibold mb-4">Active Agents</h2>
            {agents.length > 0 ? (
              <AgentVisualization agents={agents} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                {isLoading ? "Initializing agents..." : "No active agents yet"}
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Knowledge Graph Sidebar */}
      <div className={`${isMobile ? "w-full" : "w-96"} flex flex-col gap-4`}>
        <Card className="flex-1 p-6">
          <div role="region" aria-label="Knowledge graph" className="h-full min-h-[400px]">
            <h2 className="text-xl font-semibold mb-4">Knowledge Graph</h2>
            {knowledgeGraph ? (
              <KnowledgeGraphView graph={knowledgeGraph} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                {isLoading ? "Building knowledge graph..." : "No knowledge graph yet"}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
