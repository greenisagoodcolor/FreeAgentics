"use client";

import React, { useState, useEffect } from "react";
import { Send, Loader2, Target, Brain, Sparkles, AlertCircle, CheckCircle2 } from "lucide-react";

interface GoalStatus {
  agentsReceived: number;
  totalAgents: number;
  inferenceActive: boolean;
  propagationStatus: 'idle' | 'propagating' | 'completed' | 'error';
}

export default function GoalPanel() {
  const [goal, setGoal] = useState("");
  const [currentGoal, setCurrentGoal] = useState({
    text: "Initialize a distributed knowledge graph that maps emergent coalition behaviors, optimize for maximum information gain while maintaining agent autonomy constraints.",
    setAt: Date.now() - 3600000,
    progress: 0.67
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [goalStatus, setGoalStatus] = useState<GoalStatus>({
    agentsReceived: 4,
    totalAgents: 4,
    inferenceActive: true,
    propagationStatus: 'completed'
  });
  const [showSuccess, setShowSuccess] = useState(false);

  // Fetch current goal on mount
  useEffect(() => {
    fetch('/api/goals')
      .then(res => res.json())
      .then(data => {
        setCurrentGoal({
          text: data.goal.text,
          setAt: data.goal.setAt,
          progress: data.goal.progress
        });
        setIsFetching(false);
      })
      .catch(err => {
        console.error('Failed to fetch goal:', err);
        setIsFetching(false);
      });
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (goal.trim() && !isLoading) {
      setIsLoading(true);
      setGoalStatus(prev => ({ ...prev, propagationStatus: 'propagating' }));
      
      try {
        const response = await fetch('/api/goals', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: goal }),
        });
        
        const data = await response.json();
        setCurrentGoal({
          text: data.goal.text,
          setAt: Date.now(),
          progress: 0
        });
        setGoal("");
        setShowSuccess(true);
        setTimeout(() => setShowSuccess(false), 3000);
        
        // Simulate goal propagation
        setGoalStatus({
          agentsReceived: 0,
          totalAgents: 4,
          inferenceActive: false,
          propagationStatus: 'propagating'
        });
        
        // Simulate agents receiving the goal
        for (let i = 1; i <= 4; i++) {
          setTimeout(() => {
            setGoalStatus(prev => ({
              ...prev,
              agentsReceived: i,
              inferenceActive: i === 4
            }));
          }, i * 300);
        }
        
        setTimeout(() => {
          setGoalStatus(prev => ({ ...prev, propagationStatus: 'completed' }));
        }, 1500);
        
      } catch (error) {
        console.error('Failed to update goal:', error);
        setGoalStatus(prev => ({ ...prev, propagationStatus: 'error' }));
      } finally {
        setIsLoading(false);
      }
    }
  };

  const getTimeAgo = (timestamp: number) => {
    const minutes = Math.floor((Date.now() - timestamp) / 60000);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  };

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Current Goal Display */}
      <div className="flex-1">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-amber-500" />
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Active Coalition Goal
            </h3>
          </div>
          <div className="flex items-center gap-3 text-xs">
            {/* Goal Status Indicators */}
            <div className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${goalStatus.inferenceActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
              <span className="text-gray-400">Active Inference</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Brain className="w-3 h-3 text-purple-400" />
              <span className="text-gray-400">PyMDP ↔ GNN</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-blue-400" />
              <span className="text-gray-400">{goalStatus.agentsReceived}/{goalStatus.totalAgents} Agents</span>
            </div>
          </div>
        </div>
        
        {/* Goal Text */}
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
          {isFetching ? (
            <div className="flex items-center gap-2 text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading current goal...</span>
            </div>
          ) : (
            <>
              <p className="text-gray-100 leading-relaxed mb-3">
                {currentGoal.text}
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">
                  Set {getTimeAgo(currentGoal.setAt)}
                </span>
                <div className="flex items-center gap-2">
                  <div className="text-xs text-gray-400">Progress</div>
                  <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-500"
                      style={{ width: `${currentGoal.progress * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400">{Math.round(currentGoal.progress * 100)}%</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Goal Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="flex-1 relative">
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="Enter a new goal directive for the agent coalition..."
            disabled={isLoading}
            rows={2}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-sm placeholder-gray-400 
                     focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 disabled:opacity-50
                     resize-none leading-relaxed"
          />
          {showSuccess && (
            <div className="absolute -top-8 left-0 flex items-center gap-2 text-green-400 text-sm animate-fade-in">
              <CheckCircle2 className="w-4 h-4" />
              <span>Goal successfully propagated to all agents</span>
            </div>
          )}
        </div>
        <button
          type="submit"
          disabled={isLoading || !goal.trim()}
          className="px-4 py-2 bg-amber-500 hover:bg-amber-600 disabled:bg-gray-700 disabled:text-gray-500 
                   text-gray-900 rounded-lg flex items-center gap-2 transition-all duration-200
                   hover:shadow-lg hover:shadow-amber-500/20"
        >
          {isLoading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm font-medium">Propagating...</span>
            </>
          ) : (
            <>
              <Send size={16} />
              <span className="text-sm font-medium">Set Goal</span>
            </>
          )}
        </button>
      </form>

      {/* Propagation Status */}
      {goalStatus.propagationStatus === 'propagating' && (
        <div className="flex items-center gap-2 text-xs text-amber-400 animate-pulse">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Broadcasting goal to agent coalition...</span>
        </div>
      )}
      
      {goalStatus.propagationStatus === 'error' && (
        <div className="flex items-center gap-2 text-xs text-red-400">
          <AlertCircle className="w-3 h-3" />
          <span>Failed to propagate goal. Please try again.</span>
        </div>
      )}
    </div>
  );
}