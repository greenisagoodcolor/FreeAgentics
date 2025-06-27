'use client';

import React, { useEffect, useState } from 'react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Brain,
  Users,
  MessageSquare,
  Network,
  Activity,
  Play,
  Pause,
  Zap,
  BarChart3,
  Settings,
  Download,
  Upload,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

// Import dashboard components
import { AgentTemplateSelector } from '@/components/dashboard/AgentTemplateSelector';
import { ActiveAgentsList } from '@/components/dashboard/ActiveAgentsList';
import { ConversationFeed } from '@/components/dashboard/ConversationFeed';
import { KnowledgeGraphVisualization } from '@/components/dashboard/KnowledgeGraphVisualization';
import { AnalyticsWidgetGrid } from '@/components/dashboard/AnalyticsWidgetGrid';
import { ConversationOrchestration } from '@/components/dashboard/ConversationOrchestration';
import { SpatialGrid } from '@/components/dashboard/SpatialGrid';
import { BeliefExtractionPanel } from '@/components/dashboard/BeliefExtractionPanel';

// Redux hooks
import { useAppDispatch, useAppSelector } from '@/store';
import { socketService } from '@/services/socketService';

export default function MVPDashboard() {
  const dispatch = useAppDispatch();
  const [isClient, setIsClient] = useState(false);
  const [currentTime, setCurrentTime] = useState<string>('');

  // Redux state
  const agents = useAppSelector(state => state.agents.agents);
  const agentOrder = useAppSelector(state => state.agents.agentOrder);
  const conversations = useAppSelector(state => state.conversations.conversations);
  const activeConversationId = useAppSelector(state => state.conversations.activeConversationId);
  const connectionStatus = useAppSelector(state => state.connection.status);
  const knowledgeGraph = useAppSelector(state => state.knowledge.graph);
  const activeView = useAppSelector(state => state.ui.activeView);
  const panels = useAppSelector(state => state.ui.panels);

  // Simulation state
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);

  // Initialize client-side state
  useEffect(() => {
    setIsClient(true);
    setCurrentTime(new Date().toLocaleTimeString());
    
    // Update time every second
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);

    // Connect to WebSocket
    socketService.connect();
    
    return () => {
      clearInterval(timeInterval);
      socketService.disconnect();
    };
  }, []);

  // Calculate statistics
  const stats = {
    totalAgents: agentOrder.length,
    activeAgents: Object.values(agents).filter(a => a.status === 'active').length,
    typingAgents: Object.values(agents).filter(a => a.status === 'typing').length,
    totalMessages: Object.values(conversations).reduce((sum, conv) => sum + conv.messages.length, 0),
    messageRate: conversations[activeConversationId || 'main']?.messageRate || 0,
    knowledgeNodes: Object.keys(knowledgeGraph.nodes).length,
    knowledgeEdges: Object.keys(knowledgeGraph.edges).length,
  };

  return (
    <div className="h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] font-[var(--font-ui)] flex flex-col">
      {/* Bloomberg Terminal Header */}
      <header className="border-b border-[var(--border-primary)] bg-[var(--bg-primary)] px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-[var(--accent-primary)]" />
              <h1 className="text-lg font-bold">FREEAGENTICS MVP</h1>
              <Badge variant="outline" className="text-[var(--success)] border-[var(--success)]">
                LIVE
              </Badge>
            </div>
            <div className="text-xs text-[var(--text-secondary)] font-mono">
              Multi-Agent Research Dashboard v3.0
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="text-xs text-[var(--text-secondary)] font-mono">
              {isClient ? `${currentTime} EST` : '-- EST'}
            </div>
            <Button
              onClick={() => setIsSimulationRunning(!isSimulationRunning)}
              variant={isSimulationRunning ? "default" : "outline"}
              size="sm"
              className="gap-2"
            >
              {isSimulationRunning ? (
                <>
                  <Pause className="h-4 w-4" />
                  PAUSE
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  START
                </>
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="border-b border-[var(--border-primary)] bg-[var(--bg-primary)] px-6 py-2">
        <div className="flex items-center gap-8 text-xs font-mono">
          <div className="flex items-center gap-2">
            <Users className="h-3 w-3 text-[var(--accent-primary)]" />
            <span className="text-[var(--text-secondary)]">AGENTS:</span>
            <span className="text-[var(--success)]">{stats.totalAgents}</span>
            <span className="text-[var(--text-tertiary)]">
              ({stats.activeAgents} active, {stats.typingAgents} typing)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <MessageSquare className="h-3 w-3 text-[var(--accent-primary)]" />
            <span className="text-[var(--text-secondary)]">MESSAGES:</span>
            <span className="text-[var(--success)]">{stats.totalMessages}</span>
            <span className="text-[var(--text-tertiary)]">
              ({stats.messageRate}/min)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Network className="h-3 w-3 text-[var(--accent-primary)]" />
            <span className="text-[var(--text-secondary)]">KNOWLEDGE:</span>
            <span className="text-[var(--success)]">{stats.knowledgeNodes}</span>
            <span className="text-[var(--text-tertiary)]">
              nodes, {stats.knowledgeEdges} edges
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="h-3 w-3 text-[var(--accent-primary)]" />
            <span className="text-[var(--text-secondary)]">STATUS:</span>
            <span className={isSimulationRunning ? "text-[var(--success)]" : "text-[var(--warning)]"}>
              {isSimulationRunning ? 'RUNNING' : 'PAUSED'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="h-3 w-3 text-[var(--accent-primary)]" />
            <span className="text-[var(--text-secondary)]">CONNECTION:</span>
            <span className={
              connectionStatus.websocket === 'connected' ? "text-[var(--success)]" : 
              connectionStatus.websocket === 'connecting' ? "text-[var(--warning)]" : "text-[var(--error)]"
            }>
              {connectionStatus.websocket.toUpperCase()}
            </span>
            {connectionStatus.latency && (
              <span className="text-[var(--text-tertiary)]">
                ({connectionStatus.latency}ms)
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Three-Column Resizable Layout */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left Panel: Agent Management & Controls */}
        <ResizablePanel 
          defaultSize={panels.left.size} 
          minSize={panels.left.minSize} 
          maxSize={panels.left.maxSize}
        >
          <div className="h-full border-r border-[var(--border-primary)] bg-[var(--bg-primary)] overflow-hidden flex flex-col">
            <Tabs defaultValue="agents" className="flex-1 flex flex-col">
              <TabsList className="w-full justify-start rounded-none border-b border-[var(--border-primary)] bg-transparent h-10 p-0">
                <TabsTrigger value="agents" className="rounded-none data-[state=active]:bg-[var(--bg-secondary)]">
                  <Users className="h-3 w-3 mr-1" />
                  Agents
                </TabsTrigger>
                <TabsTrigger value="controls" className="rounded-none data-[state=active]:bg-[var(--bg-secondary)]">
                  <Settings className="h-3 w-3 mr-1" />
                  Controls
                </TabsTrigger>
                <TabsTrigger value="spatial" className="rounded-none data-[state=active]:bg-[var(--bg-secondary)]">
                  <Network className="h-3 w-3 mr-1" />
                  Spatial
                </TabsTrigger>
              </TabsList>

              <TabsContent value="agents" className="flex-1 p-4 space-y-4 overflow-y-auto">
                <AgentTemplateSelector />
                <ActiveAgentsList />
              </TabsContent>

              <TabsContent value="controls" className="flex-1 p-4">
                <ConversationOrchestration />
              </TabsContent>

              <TabsContent value="spatial" className="flex-1 p-4">
                <SpatialGrid />
              </TabsContent>
            </Tabs>
          </div>
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-[var(--border-primary)] hover:bg-[var(--border-secondary)]" />

        {/* Center Panel: Main Content Area */}
        <ResizablePanel 
          defaultSize={panels.center.size} 
          minSize={panels.center.minSize}
        >
          <div className="h-full flex flex-col bg-[var(--bg-primary)]">
            <Tabs value={activeView} className="flex-1 flex flex-col">
              <TabsList className="w-full justify-start rounded-none border-b border-[var(--border-primary)] bg-transparent h-10 p-0">
                <TabsTrigger value="dashboard" className="rounded-none data-[state=active]:bg-[var(--bg-secondary)]">
                  <MessageSquare className="h-3 w-3 mr-1" />
                  Conversations
                </TabsTrigger>
                <TabsTrigger value="knowledge" className="rounded-none data-[state=active]:bg-[var(--bg-secondary)]">
                  <Network className="h-3 w-3 mr-1" />
                  Knowledge Graph
                </TabsTrigger>
                <TabsTrigger value="analytics" className="rounded-none data-[state=active]:bg-[var(--bg-secondary)]">
                  <BarChart3 className="h-3 w-3 mr-1" />
                  Analytics
                </TabsTrigger>
              </TabsList>

              <TabsContent value="dashboard" className="flex-1 overflow-hidden">
                <ConversationFeed />
              </TabsContent>

              <TabsContent value="knowledge" className="flex-1 overflow-hidden">
                <KnowledgeGraphVisualization />
              </TabsContent>

              <TabsContent value="analytics" className="flex-1 overflow-hidden">
                <AnalyticsWidgetGrid />
              </TabsContent>
            </Tabs>
          </div>
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-[var(--border-primary)] hover:bg-[var(--border-secondary)]" />

        {/* Right Panel: Context & Details */}
        <ResizablePanel 
          defaultSize={panels.right.size} 
          minSize={panels.right.minSize} 
          maxSize={panels.right.maxSize}
        >
          <div className="h-full border-l border-[var(--border-primary)] bg-[var(--bg-primary)]">
            <BeliefExtractionPanel />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
} 