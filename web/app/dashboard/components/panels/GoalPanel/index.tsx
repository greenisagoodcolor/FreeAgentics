"use client";

import React, { useState } from "react";
import { Send } from "lucide-react";

interface GoalPanelProps {
  view: string;
}

export default function GoalPanel({ view }: GoalPanelProps) {
  const [goal, setGoal] = useState("");
  const [currentGoal, setCurrentGoal] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (goal.trim()) {
      setCurrentGoal(goal);
      // TODO: Send goal to agents
    }
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] p-4">
      <form onSubmit={handleSubmit} className="flex gap-2 mb-4">
        <input
          type="text"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Enter goal for agents..."
          className="flex-1 px-4 py-2 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-tertiary)] focus:outline-none focus:border-[var(--primary-amber)] font-mono text-sm"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-[var(--primary-amber)] hover:bg-[var(--primary-amber-hover)] rounded-lg text-[var(--bg-primary)] font-mono font-bold text-sm transition-colors flex items-center gap-2"
        >
          <Send className="w-4 h-4" />
          SET GOAL
        </button>
      </form>
      
      {currentGoal && (
        <div className="flex-1 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded-lg p-4">
          <div className="text-xs font-mono text-[var(--text-secondary)] mb-2">CURRENT GOAL</div>
          <div className="text-lg font-mono text-[var(--primary-amber)]">{currentGoal}</div>
        </div>
      )}
    </div>
  );
}