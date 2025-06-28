import React, { useState, useMemo } from 'react';
import { Agent } from '@/lib/types';

interface AgentListProps {
  agents: Agent[];
  onAgentSelect?: (agentId: string) => void;
  showPerformance?: boolean;
}

export const AgentList: React.FC<AgentListProps> = ({
  agents,
  onAgentSelect,
  showPerformance = false
}) => {
  const [filter, setFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('name');

  const filteredAgents = useMemo(() => {
    let filtered = agents;
    
    if (filter !== 'all') {
      filtered = agents.filter(agent => agent.status === filter);
    }

    if (sortBy === 'performance') {
      filtered = [...filtered].sort((a, b) => 
        (b.performance?.taskCompletion || 0) - (a.performance?.taskCompletion || 0)
      );
    } else {
      filtered = [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    }

    return filtered;
  }, [agents, filter, sortBy]);

  return (
    <div className="agent-list">
      <div className="controls">
        <label htmlFor="status-filter">Filter by status</label>
        <select
          id="status-filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="all">All</option>
          <option value="active">Active</option>
          <option value="idle">Idle</option>
          <option value="offline">Offline</option>
        </select>

        <label htmlFor="sort-by">Sort by</label>
        <select
          id="sort-by"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
        >
          <option value="name">Name</option>
          <option value="performance">Performance</option>
        </select>
      </div>

      <div className="agents">
        {filteredAgents.map(agent => (
          <article
            key={agent.id}
            onClick={() => onAgentSelect?.(agent.id)}
            className="agent-item"
          >
            <h3>{agent.name}</h3>
            <span className={`badge-${agent.type}`}>{agent.type}</span>
            <span className="status">{agent.status}</span>
            {showPerformance && agent.performance && (
              <div className="performance">
                <span>{Math.round((agent.performance.taskCompletion || 0) * 100)}%</span>
                <span>{Math.round((agent.performance.collaborationScore || 0) * 100)}%</span>
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  );
};