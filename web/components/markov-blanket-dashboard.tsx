"use client";

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Separator } from './ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';
import { 
  AlertTriangle, 
  Activity, 
  Wifi, 
  WifiOff, 
  Play, 
  Pause, 
  RotateCcw,
  Settings,
  Download,
  Bell,
  BellOff
} from 'lucide-react';

import { MarkovBlanketVisualization, type BoundaryViolationEvent } from './markov-blanket-visualization';
import { 
  useMarkovBlanketWebSocket,
  type MarkovBlanketEvent,
  type BoundaryViolation,
  type MonitoringStatus,
  type ConnectionStats,
  type MarkovBlanketSubscription
} from '../hooks/useMarkovBlanketWebSocket';

interface MarkovBlanketDashboardProps {
  initialAgentIds?: string[];
  autoStartMonitoring?: boolean;
  showAdvancedControls?: boolean;
  enableNotifications?: boolean;
}

export const MarkovBlanketDashboard: React.FC<MarkovBlanketDashboardProps> = ({
  initialAgentIds = [],
  autoStartMonitoring = true,
  showAdvancedControls = false,
  enableNotifications = true
}) => {
  // State management
  const [selectedAgentId, setSelectedAgentId] = useState<string>(initialAgentIds[0] || '');
  const [newAgentId, setNewAgentId] = useState('');
  const [monitoredAgents, setMonitoredAgents] = useState<Set<string>>(new Set(initialAgentIds));
  const [eventLog, setEventLog] = useState<MarkovBlanketEvent[]>([]);
  const [notificationsEnabled, setNotificationsEnabled] = useState(enableNotifications);
  const [soundAlertsEnabled, setSoundAlertsEnabled] = useState(false);
  const [autoAcknowledgeViolations, setAutoAcknowledgeViolations] = useState(false);
  const [maxLogEntries, setMaxLogEntries] = useState(1000);

  // Real-time data state
  const [agentDimensions, setAgentDimensions] = useState<Record<string, any>>({});
  const [agentMetrics, setAgentMetrics] = useState<Record<string, any>>({});
  const [agentPositions, setAgentPositions] = useState<Record<string, any>>({});
  const [boundaryThresholds, setBoundaryThresholds] = useState({
    internal: 0.8,
    sensory: 0.8,
    active: 0.8,
    external: 0.8
  });

  // WebSocket subscription configuration
  const subscription: MarkovBlanketSubscription = useMemo(() => ({
    agent_ids: Array.from(monitoredAgents),
    event_types: [
      'boundary_violation',
      'state_update',
      'integrity_update',
      'threshold_breach',
      'monitoring_error'
    ],
    severity_levels: ['info', 'warning', 'error', 'critical'],
    include_mathematical_proofs: showAdvancedControls,
    include_detailed_metrics: true,
    violation_alerts_only: false,
    real_time_updates: true
  }), [monitoredAgents, showAdvancedControls]);

  // WebSocket connection
  const {
    isConnected,
    isConnecting,
    error: wsError,
    lastEventTime,
    connectionStats,
    monitoringStatus,
    violations,
    connect,
    disconnect,
    updateSubscription,
    registerAgent,
    unregisterAgent,
    startMonitoring,
    stopMonitoring,
    getMonitoringStatus,
    getAgentViolations,
    getConnectionStats,
    getComplianceReport,
    ping
  } = useMarkovBlanketWebSocket({
    autoConnect: true,
    subscription,
    onEvent: handleMarkovBlanketEvent,
    onConnect: handleWebSocketConnect,
    onDisconnect: handleWebSocketDisconnect,
    onError: handleWebSocketError,
    onViolation: handleBoundaryViolation
  });

  // Event handlers
  function handleMarkovBlanketEvent(event: MarkovBlanketEvent) {
    // Add to event log
    setEventLog(prev => {
      const newLog = [event, ...prev].slice(0, maxLogEntries);
      return newLog;
    });

    // Update agent-specific data based on event type
    switch (event.type) {
      case 'state_update':
        if (event.data.dimensions) {
          setAgentDimensions(prev => ({
            ...prev,
            [event.agent_id]: event.data.dimensions
          }));
        }
        if (event.data.metrics) {
          setAgentMetrics(prev => ({
            ...prev,
            [event.agent_id]: event.data.metrics
          }));
        }
        if (event.data.position) {
          setAgentPositions(prev => ({
            ...prev,
            [event.agent_id]: event.data.position
          }));
        }
        break;

      case 'integrity_update':
        setAgentMetrics(prev => ({
          ...prev,
          [event.agent_id]: {
            ...prev[event.agent_id],
            boundary_integrity: event.data.boundary_integrity,
            conditional_independence: event.data.conditional_independence
          }
        }));
        break;

      case 'boundary_violation':
        // Show notification if enabled
        if (notificationsEnabled && 'Notification' in window) {
          new Notification('Boundary Violation Detected', {
            body: `Agent ${event.agent_id}: ${event.data.violation_type}`,
            icon: '/favicon.ico'
          });
        }

        // Play sound alert if enabled
        if (soundAlertsEnabled) {
          // Create audio context and play alert sound
          const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
          const oscillator = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          
          oscillator.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
          gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
          
          oscillator.start();
          oscillator.stop(audioContext.currentTime + 0.2);
        }
        break;
    }
  }

  function handleWebSocketConnect() {
    console.log('Connected to Markov Blanket monitoring');
    // Register all monitored agents
    monitoredAgents.forEach(agentId => {
      registerAgent(agentId);
    });
    
    // Start monitoring if auto-start is enabled
    if (autoStartMonitoring) {
      startMonitoring();
    }

    // Get initial status
    getMonitoringStatus();
    getConnectionStats();
  }

  function handleWebSocketDisconnect() {
    console.log('Disconnected from Markov Blanket monitoring');
  }

  function handleWebSocketError(error: string) {
    console.error('Markov Blanket WebSocket error:', error);
  }

  function handleBoundaryViolation(violation: BoundaryViolation) {
    console.log('Boundary violation detected:', violation);
    
    // Auto-acknowledge if enabled
    if (autoAcknowledgeViolations) {
      setTimeout(() => {
        // In a real implementation, this would call an acknowledgment API
        console.log('Auto-acknowledging violation:', violation.agent_id);
      }, 5000);
    }
  }

  // Agent management
  const handleAddAgent = useCallback(() => {
    if (newAgentId.trim() && !monitoredAgents.has(newAgentId.trim())) {
      const agentId = newAgentId.trim();
      setMonitoredAgents(prev => new Set([...prev, agentId]));
      setNewAgentId('');
      
      if (isConnected) {
        registerAgent(agentId);
      }
    }
  }, [newAgentId, monitoredAgents, isConnected, registerAgent]);

  const handleRemoveAgent = useCallback((agentId: string) => {
    setMonitoredAgents(prev => {
      const newSet = new Set(prev);
      newSet.delete(agentId);
      return newSet;
    });
    
    if (isConnected) {
      unregisterAgent(agentId);
    }
    
    // Clear agent data
    setAgentDimensions(prev => {
      const newData = { ...prev };
      delete newData[agentId];
      return newData;
    });
    setAgentMetrics(prev => {
      const newData = { ...prev };
      delete newData[agentId];
      return newData;
    });
    setAgentPositions(prev => {
      const newData = { ...prev };
      delete newData[agentId];
      return newData;
    });

    // Select different agent if current one was removed
    if (selectedAgentId === agentId) {
      const remainingAgents = Array.from(monitoredAgents).filter(id => id !== agentId);
      setSelectedAgentId(remainingAgents[0] || '');
    }
  }, [isConnected, unregisterAgent, selectedAgentId, monitoredAgents]);

  // Threshold management
  const handleBoundaryThresholdChange = useCallback((dimension: string, value: number) => {
    setBoundaryThresholds(prev => ({
      ...prev,
      [dimension]: value
    }));
  }, []);

  // Violation acknowledgment
  const handleViolationAcknowledge = useCallback((violationId: string) => {
    // In a real implementation, this would call an API to acknowledge the violation
    console.log('Acknowledging violation:', violationId);
  }, []);

  // Data export
  const handleExportData = useCallback(() => {
    const exportData = {
      timestamp: new Date().toISOString(),
      agents: Array.from(monitoredAgents),
      eventLog: eventLog.slice(0, 100), // Last 100 events
      connectionStats,
      monitoringStatus,
      violations: violations.slice(0, 50) // Last 50 violations
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `markov-blanket-data-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [monitoredAgents, eventLog, connectionStats, monitoringStatus, violations]);

  // Request notification permission
  useEffect(() => {
    if (notificationsEnabled && 'Notification' in window) {
      Notification.requestPermission();
    }
  }, [notificationsEnabled]);

  // Update subscription when monitored agents change
  useEffect(() => {
    if (isConnected) {
      updateSubscription(subscription);
    }
  }, [isConnected, subscription, updateSubscription]);

  // Periodic status updates
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      getMonitoringStatus();
      getConnectionStats();
      ping();
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, [isConnected, getMonitoringStatus, getConnectionStats, ping]);

  // Get current agent data
  const currentAgentDimensions = selectedAgentId ? agentDimensions[selectedAgentId] : null;
  const currentAgentMetrics = selectedAgentId ? agentMetrics[selectedAgentId] : null;
  const currentAgentPosition = selectedAgentId ? agentPositions[selectedAgentId] : null;
  const currentAgentViolations = violations
    .filter(v => v.agent_id === selectedAgentId)
    .map((v, index): BoundaryViolationEvent => ({
      event_id: `event-${Date.now()}-${index}`,
      agent_id: v.agent_id,
      violation_type: v.violation_type,
      timestamp: v.timestamp,
      severity: parseFloat(v.severity) || 0.5,
      independence_measure: v.independence_measure,
      threshold_violated: v.threshold,
      free_energy: 0,
      expected_free_energy: 0,
      kl_divergence: 0,
      acknowledged: false,
      mitigated: false
    }));

  // Generate mock data if no real data available (for demonstration)
  const mockDimensions = {
    internal_states: [0.3, 0.7, 0.2],
    sensory_states: [0.8, 0.4, 0.6],
    active_states: [0.5, 0.9],
    external_states: [0.2, 0.3, 0.8, 0.1],
    internal_dimension: 0.4,
    sensory_dimension: 0.6,
    active_dimension: 0.7,
    external_dimension: 0.3
  };

  const mockMetrics = {
    free_energy: 2.34,
    expected_free_energy: 1.89,
    kl_divergence: 0.45,
    boundary_integrity: 0.87,
    conditional_independence: 0.03,
    stability_over_time: 0.92,
    violation_count: violations.length,
    last_violation_time: violations[0]?.timestamp
  };

  const mockPosition = {
    agent_id: selectedAgentId,
    position: {
      internal: 0.4,
      sensory: 0.6,
      active: 0.7,
      external: 0.3
    },
    boundary_distance: 0.15,
    is_within_boundary: true
  };

  return (
    <div className="space-y-6">
      {/* Header with Connection Status */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Markov Blanket Monitoring</h2>
          <p className="text-muted-foreground">
            Real-time boundary violation detection and agent state monitoring
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <Wifi className="h-4 w-4 text-green-500" />
            ) : isConnecting ? (
              <Activity className="h-4 w-4 text-yellow-500 animate-spin" />
            ) : (
              <WifiOff className="h-4 w-4 text-red-500" />
            )}
            <span className="text-sm">
              {isConnected ? 'Connected' : isConnecting ? 'Connecting...' : 'Disconnected'}
            </span>
          </div>

          {/* Monitoring Status */}
          <Badge variant={monitoringStatus?.monitoring_active ? "default" : "secondary"}>
            {monitoringStatus?.monitoring_active ? 'Monitoring Active' : 'Monitoring Inactive'}
          </Badge>

          {/* Controls */}
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              variant="outline"
              onClick={monitoringStatus?.monitoring_active ? stopMonitoring : startMonitoring}
              disabled={!isConnected}
            >
              {monitoringStatus?.monitoring_active ? (
                <>
                  <Pause className="h-4 w-4 mr-2" />
                  Stop
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start
                </>
              )}
            </Button>

            <Button
              size="sm"
              variant="outline"
              onClick={handleExportData}
              disabled={!isConnected}
            >
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </div>

      {/* Connection Error Alert */}
      {wsError && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Connection Error</AlertTitle>
          <AlertDescription>{wsError}</AlertDescription>
        </Alert>
      )}

      {/* Main Dashboard */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Agent Management */}
        <Card>
          <CardHeader>
            <CardTitle>Monitored Agents</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Add Agent */}
            <div className="flex space-x-2">
              <Input
                placeholder="Agent ID"
                value={newAgentId}
                onChange={(e) => setNewAgentId(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddAgent()}
              />
              <Button size="sm" onClick={handleAddAgent}>
                Add
              </Button>
            </div>

            {/* Agent List */}
            <ScrollArea className="h-32">
              <div className="space-y-2">
                {Array.from(monitoredAgents).map(agentId => (
                  <div key={agentId} className="flex items-center justify-between p-2 border rounded">
                    <div className="flex items-center space-x-2">
                      <div 
                        className={`w-2 h-2 rounded-full ${
                          agentPositions[agentId]?.is_within_boundary !== false 
                            ? 'bg-green-500' 
                            : 'bg-red-500'
                        }`}
                      />
                      <span className="text-sm">{agentId}</span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleRemoveAgent(agentId)}
                    >
                      ×
                    </Button>
                  </div>
                ))}
              </div>
            </ScrollArea>

            {/* Agent Selection */}
            {monitoredAgents.size > 0 && (
              <Select value={selectedAgentId} onValueChange={setSelectedAgentId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select agent to view" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from(monitoredAgents).map(agentId => (
                    <SelectItem key={agentId} value={agentId}>
                      {agentId}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </CardContent>
        </Card>

        {/* Main Visualization */}
        <div className="xl:col-span-3">
          {selectedAgentId && (
            <MarkovBlanketVisualization
              agentId={selectedAgentId}
              dimensions={currentAgentDimensions || mockDimensions}
              metrics={currentAgentMetrics || mockMetrics}
              violations={currentAgentViolations}
              agentPosition={currentAgentPosition || mockPosition}
              boundaryThresholds={boundaryThresholds}
              realTimeUpdates={isConnected}
              showViolations={true}
              showMetrics={true}
              onViolationAcknowledge={handleViolationAcknowledge}
              onBoundaryThresholdChange={handleBoundaryThresholdChange}
            />
          )}
        </div>
      </div>

      {/* Additional Tabs */}
      <Tabs defaultValue="events" className="w-full">
        <TabsList>
          <TabsTrigger value="events">Event Log</TabsTrigger>
          <TabsTrigger value="violations">Violations</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="events">
          <Card>
            <CardHeader>
              <CardTitle>Real-Time Event Log</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                <div className="space-y-2">
                  {eventLog.slice(0, 50).map((event, index) => (
                    <div key={index} className="p-2 border rounded text-sm">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Badge variant={
                            event.severity === 'critical' ? 'destructive' :
                            event.severity === 'error' ? 'destructive' :
                            event.severity === 'warning' ? 'default' : 'secondary'
                          }>
                            {event.severity}
                          </Badge>
                          <span className="font-medium">{event.type}</span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="mt-1">
                        <span className="text-xs">Agent: {event.agent_id}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="violations">
          <Card>
            <CardHeader>
              <CardTitle>Boundary Violations</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                <div className="space-y-2">
                  {violations.slice(0, 20).map((violation, index) => (
                    <div key={index} className="p-3 border rounded">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{violation.violation_type}</div>
                          <div className="text-sm text-muted-foreground">
                            Agent: {violation.agent_id}
                          </div>
                        </div>
                        <Badge variant="destructive">
                          {violation.severity}
                        </Badge>
                      </div>
                      <div className="mt-2 text-xs">
                        <div>Independence: {violation.independence_measure.toFixed(4)}</div>
                        <div>Threshold: {violation.threshold.toFixed(4)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="stats">
          <Card>
            <CardHeader>
              <CardTitle>Connection Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              {connectionStats && (
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>Total Connections:</div>
                  <div>{connectionStats.total_connections}</div>
                  
                  <div>Events Sent:</div>
                  <div>{connectionStats.total_events_sent}</div>
                  
                  <div>Active Violations:</div>
                  <div>{connectionStats.active_violations}</div>
                  
                  <div>Monitored Agents:</div>
                  <div>{connectionStats.monitored_agents}</div>
                  
                  <div>System Uptime:</div>
                  <div>{Math.round(connectionStats.system_uptime)}s</div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings">
          <Card>
            <CardHeader>
              <CardTitle>Dashboard Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={notificationsEnabled}
                  onCheckedChange={setNotificationsEnabled}
                />
                <Label>Browser Notifications</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  checked={soundAlertsEnabled}
                  onCheckedChange={setSoundAlertsEnabled}
                />
                <Label>Sound Alerts</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  checked={autoAcknowledgeViolations}
                  onCheckedChange={setAutoAcknowledgeViolations}
                />
                <Label>Auto-acknowledge Violations</Label>
              </div>

              <div className="space-y-2">
                <Label>Max Log Entries</Label>
                <Input
                  type="number"
                  value={maxLogEntries}
                  onChange={(e) => setMaxLogEntries(parseInt(e.target.value) || 1000)}
                  min={100}
                  max={10000}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MarkovBlanketDashboard;