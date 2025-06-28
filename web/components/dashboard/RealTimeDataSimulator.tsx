"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "@/store";

// =============================================================================
// REAL-TIME DATA TYPES
// =============================================================================

interface MetricsData {
  timestamp: number;
  cpuUsage: number;
  memoryUsage: number;
  networkLatency: number;
  activeConnections: number;
  systemLoad: number;
}

interface AgentActivity {
  agentId: string;
  action: 'thinking' | 'responding' | 'idle' | 'error';
  timestamp: number;
  duration?: number;
  message?: string;
}

interface KnowledgeGraphUpdate {
  type: 'node_added' | 'node_updated' | 'edge_added' | 'edge_removed';
  nodeId?: string;
  edgeId?: string;
  data: any;
  timestamp: number;
}

// =============================================================================
// REAL-TIME DATA CONTEXT
// =============================================================================

interface RealTimeContextType {
  isConnected: boolean;
  connectionQuality: 'excellent' | 'good' | 'poor' | 'disconnected';
  latency: number;
  lastUpdate: number;
  metricsData: MetricsData[];
  agentActivities: AgentActivity[];
  knowledgeUpdates: KnowledgeGraphUpdate[];
  startSimulation: () => void;
  stopSimulation: () => void;
  pauseSimulation: () => void;
  resumeSimulation: () => void;
  setUpdateInterval: (interval: number) => void;
}

const RealTimeContext = React.createContext<RealTimeContextType | null>(null);

export const useRealTimeData = () => {
  const context = React.useContext(RealTimeContext);
  if (!context) {
    throw new Error('useRealTimeData must be used within RealTimeDataProvider');
  }
  return context;
};

// =============================================================================
// REAL-TIME DATA PROVIDER
// =============================================================================

export const RealTimeDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionQuality, setConnectionQuality] = useState<'excellent' | 'good' | 'poor' | 'disconnected'>('disconnected');
  const [latency, setLatency] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(0);
  const [metricsData, setMetricsData] = useState<MetricsData[]>([]);
  const [agentActivities, setAgentActivities] = useState<AgentActivity[]>([]);
  const [knowledgeUpdates, setKnowledgeUpdates] = useState<KnowledgeGraphUpdate[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [updateInterval, setUpdateIntervalState] = useState(1000); // 1 second

  const intervalRef = useRef<number>();
  const metricsIntervalRef = useRef<number>();
  const agentIntervalRef = useRef<number>();
  const knowledgeIntervalRef = useRef<number>();

  // Simulate connection quality based on latency
  const updateConnectionQuality = useCallback((currentLatency: number) => {
    if (currentLatency < 50) {
      setConnectionQuality('excellent');
    } else if (currentLatency < 150) {
      setConnectionQuality('good');
    } else if (currentLatency < 500) {
      setConnectionQuality('poor');
    } else {
      setConnectionQuality('disconnected');
    }
  }, []);

  // Generate realistic metrics data
  const generateMetricsData = useCallback((): MetricsData => {
    const now = Date.now();
    const baseLatency = 12 + Math.random() * 20; // 12-32ms base
    const jitter = (Math.random() - 0.5) * 10; // ±5ms jitter
    const currentLatency = Math.max(1, baseLatency + jitter);

    return {
      timestamp: now,
      cpuUsage: Math.max(0, Math.min(100, 20 + Math.sin(now / 10000) * 15 + Math.random() * 10)),
      memoryUsage: Math.max(0, Math.min(100, 45 + Math.sin(now / 15000) * 20 + Math.random() * 5)),
      networkLatency: currentLatency,
      activeConnections: Math.floor(150 + Math.sin(now / 20000) * 50 + Math.random() * 20),
      systemLoad: Math.max(0, Math.min(5, 1.2 + Math.sin(now / 12000) * 0.8 + Math.random() * 0.3))
    };
  }, []);

  // Generate agent activity
  const generateAgentActivity = useCallback((): AgentActivity | null => {
    if (Math.random() > 0.3) return null; // 30% chance of activity

    const agents = ['agent-1', 'agent-2', 'agent-3', 'agent-4'];
    const actions: AgentActivity['action'][] = ['thinking', 'responding', 'idle', 'error'];
    const weights = [0.4, 0.3, 0.25, 0.05]; // Weighted probabilities

    const randomAgent = agents[Math.floor(Math.random() * agents.length)];
    
    let selectedAction: AgentActivity['action'] = 'idle';
    const random = Math.random();
    let cumulativeWeight = 0;
    
    for (let i = 0; i < actions.length; i++) {
      cumulativeWeight += weights[i];
      if (random <= cumulativeWeight) {
        selectedAction = actions[i];
        break;
      }
    }

    const messages = {
      thinking: ['Analyzing data patterns...', 'Processing user query...', 'Evaluating options...'],
      responding: ['Generating response...', 'Formatting output...', 'Sending message...'],
      idle: ['Waiting for input...', 'Monitoring system...', 'Standing by...'],
      error: ['Connection timeout', 'Processing error', 'Invalid input detected']
    };

    return {
      agentId: randomAgent,
      action: selectedAction,
      timestamp: Date.now(),
      duration: selectedAction === 'thinking' ? 2000 + Math.random() * 3000 : undefined,
      message: messages[selectedAction][Math.floor(Math.random() * messages[selectedAction].length)]
    };
  }, []);

  // Generate knowledge graph updates
  const generateKnowledgeUpdate = useCallback((): KnowledgeGraphUpdate | null => {
    if (Math.random() > 0.15) return null; // 15% chance of update

    const types: KnowledgeGraphUpdate['type'][] = ['node_added', 'node_updated', 'edge_added', 'edge_removed'];
    const type = types[Math.floor(Math.random() * types.length)];

    const nodeIds = ['belief-1', 'fact-2', 'hypothesis-3', 'concept-4', 'relation-5'];
    const edgeIds = ['edge-1', 'edge-2', 'edge-3', 'edge-4'];

    let data: any = {};

    switch (type) {
      case 'node_added':
      case 'node_updated':
        data = {
          id: `node-${Date.now()}`,
          label: `New Concept ${Math.floor(Math.random() * 1000)}`,
          type: ['belief', 'fact', 'hypothesis'][Math.floor(Math.random() * 3)],
          confidence: Math.random(),
          agent: `agent-${Math.floor(Math.random() * 4) + 1}`
        };
        break;
      case 'edge_added':
        data = {
          id: `edge-${Date.now()}`,
          source: nodeIds[Math.floor(Math.random() * nodeIds.length)],
          target: nodeIds[Math.floor(Math.random() * nodeIds.length)],
          type: ['supports', 'contradicts', 'related'][Math.floor(Math.random() * 3)],
          strength: Math.random()
        };
        break;
      case 'edge_removed':
        data = {
          id: edgeIds[Math.floor(Math.random() * edgeIds.length)]
        };
        break;
    }

    return {
      type,
      nodeId: type.includes('node') ? data.id : undefined,
      edgeId: type.includes('edge') ? data.id : undefined,
      data,
      timestamp: Date.now()
    };
  }, []);

  // Start simulation
  const startSimulation = useCallback(() => {
    if (intervalRef.current) return; // Already running

    setIsConnected(true);
    setIsPaused(false);

    // Metrics update loop
    metricsIntervalRef.current = window.setInterval(() => {
      if (isPaused) return;

      const newMetrics = generateMetricsData();
      setLatency(newMetrics.networkLatency);
      setLastUpdate(newMetrics.timestamp);
      updateConnectionQuality(newMetrics.networkLatency);

      setMetricsData(prev => {
        const updated = [...prev, newMetrics];
        return updated.slice(-50); // Keep last 50 data points
      });
    }, updateInterval);

    // Agent activity loop
    agentIntervalRef.current = window.setInterval(() => {
      if (isPaused) return;

      const activity = generateAgentActivity();
      if (activity) {
        setAgentActivities(prev => {
          const updated = [...prev, activity];
          return updated.slice(-20); // Keep last 20 activities
        });
      }
    }, 2000);

    // Knowledge graph update loop
    knowledgeIntervalRef.current = window.setInterval(() => {
      if (isPaused) return;

      const update = generateKnowledgeUpdate();
      if (update) {
        setKnowledgeUpdates(prev => {
          const updated = [...prev, update];
          return updated.slice(-10); // Keep last 10 updates
        });
      }
    }, 5000);

  }, [generateMetricsData, generateAgentActivity, generateKnowledgeUpdate, updateConnectionQuality, updateInterval, isPaused]);

  // Stop simulation
  const stopSimulation = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (metricsIntervalRef.current) clearInterval(metricsIntervalRef.current);
    if (agentIntervalRef.current) clearInterval(agentIntervalRef.current);
    if (knowledgeIntervalRef.current) clearInterval(knowledgeIntervalRef.current);

    intervalRef.current = undefined;
    metricsIntervalRef.current = undefined;
    agentIntervalRef.current = undefined;
    knowledgeIntervalRef.current = undefined;

    setIsConnected(false);
    setConnectionQuality('disconnected');
    setLatency(0);
  }, []);

  // Pause simulation
  const pauseSimulation = useCallback(() => {
    setIsPaused(true);
  }, []);

  // Resume simulation
  const resumeSimulation = useCallback(() => {
    setIsPaused(false);
  }, []);

  // Set update interval
  const setUpdateInterval = useCallback((interval: number) => {
    setUpdateIntervalState(interval);
    
    // Restart with new interval if currently running
    if (intervalRef.current) {
      stopSimulation();
      setTimeout(startSimulation, 100);
    }
  }, [startSimulation, stopSimulation]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSimulation();
    };
  }, [stopSimulation]);

  // Auto-start simulation
  useEffect(() => {
    startSimulation();
    return stopSimulation;
  }, [startSimulation, stopSimulation]);

  const value: RealTimeContextType = {
    isConnected,
    connectionQuality,
    latency,
    lastUpdate,
    metricsData,
    agentActivities,
    knowledgeUpdates,
    startSimulation,
    stopSimulation,
    pauseSimulation,
    resumeSimulation,
    setUpdateInterval
  };

  return (
    <RealTimeContext.Provider value={value}>
      {children}
    </RealTimeContext.Provider>
  );
};

// =============================================================================
// CONNECTION STATUS COMPONENT
// =============================================================================

export const ConnectionStatus: React.FC = () => {
  const { isConnected, connectionQuality, latency, lastUpdate } = useRealTimeData();

  const getStatusColor = () => {
    switch (connectionQuality) {
      case 'excellent': return 'var(--success)';
      case 'good': return 'var(--warning)';
      case 'poor': return 'var(--error)';
      case 'disconnected': return 'var(--text-muted)';
      default: return 'var(--text-muted)';
    }
  };

  const getStatusText = () => {
    if (!isConnected) return 'DISCONNECTED';
    return connectionQuality.toUpperCase();
  };

  return (
    <div className="connection-status flex items-center gap-2 text-sm font-mono">
      <div 
        className="status-dot"
        style={{ 
          background: getStatusColor(),
          boxShadow: isConnected ? `0 0 8px ${getStatusColor()}` : 'none'
        }}
      />
      <span style={{ color: getStatusColor() }}>
        {getStatusText()}
      </span>
      {isConnected && (
        <>
          <span className="text-[var(--text-tertiary)]">•</span>
          <span className="text-[var(--text-secondary)]">
            {Math.round(latency)}ms
          </span>
        </>
      )}
    </div>
  );
};

// =============================================================================
// REAL-TIME METRICS DISPLAY
// =============================================================================

export const RealTimeMetrics: React.FC = () => {
  const { metricsData, isConnected } = useRealTimeData();
  const latestMetrics = metricsData[metricsData.length - 1];

  if (!isConnected || !latestMetrics) {
    return (
      <div className="real-time-metrics opacity-50">
        <div className="text-sm text-[var(--text-secondary)]">No real-time data</div>
      </div>
    );
  }

  return (
    <div className="real-time-metrics grid grid-cols-2 md:grid-cols-4 gap-4 text-sm font-mono">
      <div className="metric">
        <div className="text-[var(--text-tertiary)]">CPU</div>
        <div className="text-[var(--text-primary)]">{latestMetrics.cpuUsage.toFixed(1)}%</div>
      </div>
      <div className="metric">
        <div className="text-[var(--text-tertiary)]">MEM</div>
        <div className="text-[var(--text-primary)]">{latestMetrics.memoryUsage.toFixed(1)}%</div>
      </div>
      <div className="metric">
        <div className="text-[var(--text-tertiary)]">LAT</div>
        <div className="text-[var(--text-primary)]">{Math.round(latestMetrics.networkLatency)}ms</div>
      </div>
      <div className="metric">
        <div className="text-[var(--text-tertiary)]">CONN</div>
        <div className="text-[var(--text-primary)]">{latestMetrics.activeConnections}</div>
      </div>
    </div>
  );
};

// =============================================================================
// AGENT ACTIVITY FEED
// =============================================================================

export const AgentActivityFeed: React.FC = () => {
  const { agentActivities, isConnected } = useRealTimeData();

  if (!isConnected) {
    return (
      <div className="agent-activity-feed opacity-50">
        <div className="text-sm text-[var(--text-secondary)]">Activity feed offline</div>
      </div>
    );
  }

  return (
    <div className="agent-activity-feed max-h-48 overflow-y-auto space-y-2">
      {agentActivities.slice(-5).reverse().map((activity, index) => (
        <div key={`${activity.agentId}-${activity.timestamp}-${index}`} className="activity-item flex items-center gap-3 p-2 rounded bg-[var(--bg-tertiary)]">
          <div className={`status-dot ${activity.action}`} />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-[var(--text-primary)]">
              {activity.agentId.toUpperCase()}
            </div>
            <div className="text-xs text-[var(--text-secondary)] truncate">
              {activity.message}
            </div>
          </div>
          <div className="text-xs text-[var(--text-tertiary)] font-mono">
            {new Date(activity.timestamp).toLocaleTimeString()}
          </div>
        </div>
      ))}
      {agentActivities.length === 0 && (
        <div className="text-sm text-[var(--text-secondary)] text-center py-4">
          No recent activity
        </div>
      )}
    </div>
  );
};

export default {
  RealTimeDataProvider,
  useRealTimeData,
  ConnectionStatus,
  RealTimeMetrics,
  AgentActivityFeed
};
