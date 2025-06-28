/**
 * Stub components for testing
 * These provide minimal implementations of components that don't exist yet
 */

import React from 'react';
import { Agent } from '@/lib/types';

// AgentList stub
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
  const [filter, setFilter] = React.useState<string>('all');
  const [sortBy, setSortBy] = React.useState<string>('name');

  const filteredAgents = React.useMemo(() => {
    let filtered = agents;
    
    if (filter !== 'all') {
      filtered = agents.filter(agent => (agent as any).status === filter);
    }

    if (sortBy === 'performance') {
      filtered = [...filtered].sort((a, b) => 
        ((b as any).performance?.taskCompletion || 0) - ((a as any).performance?.taskCompletion || 0)
      );
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
            role="article"
          >
            <h3>{agent.name}</h3>
            <span className={`badge-${(agent as any).type}`}>{(agent as any).type}</span>
            <span className="status">{(agent as any).status}</span>
            {showPerformance && (agent as any).performance && (
              <div className="performance">
                <span>{Math.round(((agent as any).performance.taskCompletion || 0) * 100)}%</span>
                <span>{Math.round(((agent as any).performance.collaborationScore || 0) * 100)}%</span>
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  );
};

// AgentBeliefVisualizer stub
interface AgentBeliefVisualizerProps {
  agent: Agent;
  history?: any[];
  previousBeliefs?: Record<string, number>;
  editable?: boolean;
  onBeliefChange?: (agentId: string, beliefs: Record<string, number>) => void;
}

export const AgentBeliefVisualizer: React.FC<AgentBeliefVisualizerProps> = ({
  agent,
  history,
  previousBeliefs,
  editable = false,
  onBeliefChange
}) => {
  const [showTimeline, setShowTimeline] = React.useState(false);

  const getBeliefClass = (belief: string, value: number) => {
    if (!previousBeliefs) return '';
    const prev = previousBeliefs[belief];
    if (value > prev) return 'belief-increased';
    if (value < prev) return 'belief-decreased';
    return '';
  };

  return (
    <div className="belief-visualizer">
      {Object.entries((agent as any).beliefs || {}).map(([belief, value]) => (
        <div key={belief} className="belief-item">
          <span>{belief}: {String(value)}</span>
          <div 
            data-testid={`belief-${belief}`}
            className={getBeliefClass(belief, value as number)}
          >
            {editable ? (
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={value as number}
                aria-label={belief}
                onChange={(e) => onBeliefChange?.(agent.id, {
                  ...(agent as any).beliefs,
                  [belief]: parseFloat(e.target.value)
                })}
              />
            ) : (
              <div className="belief-bar" style={{ width: `${(value as number) * 100}%` }} />
            )}
          </div>
        </div>
      ))}
      
      <button onClick={() => setShowTimeline(!showTimeline)}>
        {showTimeline ? 'Hide' : 'Show'} Timeline
      </button>
      
      {showTimeline && <div data-testid="belief-timeline">Timeline visualization</div>}
    </div>
  );
};

// CharacterCreator stub
interface CharacterCreatorProps {
  onCreate?: (character: any) => void;
}

export const CharacterCreator: React.FC<CharacterCreatorProps> = ({ onCreate }) => {
  const [name, setName] = React.useState('');
  const [type, setType] = React.useState('');
  const [capabilities, setCapabilities] = React.useState<string[]>([]);
  const [template, setTemplate] = React.useState('');
  const [exploration, setExploration] = React.useState(0.5);
  const [showPreview, setShowPreview] = React.useState(false);
  const [errors, setErrors] = React.useState<Record<string, string>>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const newErrors: Record<string, string> = {};
    if (!name) newErrors.name = 'Name is required';
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    onCreate?.({
      name,
      type,
      capabilities,
      beliefs: { exploration }
    });
  };

  const handleTemplateChange = (value: string) => {
    setTemplate(value);
    if (value === 'researcher') {
      setName('Research Agent');
      setType('explorer');
      setCapabilities(['reasoning', 'learning', 'analysis']);
    }
  };

  const toggleCapability = (cap: string) => {
    setCapabilities(prev => 
      prev.includes(cap) 
        ? prev.filter(c => c !== cap)
        : [...prev, cap]
    );
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="agent-name">Agent Name</label>
        <input
          id="agent-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        {errors.name && <span>{errors.name}</span>}
      </div>

      <div>
        <label htmlFor="agent-type">Agent Type</label>
        <select
          id="agent-type"
          value={type}
          onChange={(e) => setType(e.target.value)}
        >
          <option value="">Select type</option>
          <option value="explorer">Explorer</option>
          <option value="coordinator">Coordinator</option>
        </select>
      </div>

      <div>
        <label htmlFor="template">Use Template</label>
        <select
          id="template"
          value={template}
          onChange={(e) => handleTemplateChange(e.target.value)}
        >
          <option value="">None</option>
          <option value="researcher">Researcher</option>
        </select>
      </div>

      <div>
        <span>Select Capabilities</span>
        {['reasoning', 'learning', 'analysis', 'communication', 'negotiation', 'planning', 'coordination'].map(cap => (
          <label key={cap}>
            <input
              type="checkbox"
              checked={capabilities.includes(cap)}
              onChange={() => toggleCapability(cap)}
            />
            {cap}
          </label>
        ))}
      </div>

      <div>
        <label htmlFor="exploration">Initial Exploration</label>
        <input
          id="exploration"
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={exploration}
          onChange={(e) => setExploration(parseFloat(e.target.value))}
        />
      </div>

      <button type="button" onClick={() => setShowPreview(!showPreview)}>
        Preview
      </button>

      {showPreview && (
        <div data-testid="agent-preview">
          {name || 'Preview Agent'}
        </div>
      )}

      <button type="submit">Create Agent</button>
    </form>
  );
};