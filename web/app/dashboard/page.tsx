'use client';

import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { 
  Brain, Settings, Maximize2, Minimize2, Monitor, 
  Users, BarChart3, Network, MessageSquare, Zap 
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { socketService } from '@/services/socketService';

// Layout imports
import BloombergLayout from './layouts/BloombergLayout';
import ResizableLayout from './layouts/ResizableLayout';
import KnowledgeLayout from './layouts/KnowledgeLayout';

// Panel imports (lazy loaded for performance)
const AgentPanel = React.lazy(() => import('./components/panels/AgentPanel'));
const ConversationPanel = React.lazy(() => import('./components/panels/ConversationPanel'));
const AnalyticsPanel = React.lazy(() => import('./components/panels/AnalyticsPanel'));
const KnowledgePanel = React.lazy(() => import('./components/panels/KnowledgePanel'));
const ControlPanel = React.lazy(() => import('./components/panels/ControlPanel'));
const MetricsPanel = React.lazy(() => import('./components/panels/MetricsPanel'));

// Dashboard view configurations
export interface DashboardView {
  id: string;
  name: string;
  description: string;
  layout: 'bloomberg' | 'resizable' | 'knowledge' | 'compact';
  panels: string[];
  permissions: string[];
  theme: {
    primaryColor: string;
    backgroundColor: string;
    fontFamily: string;
  };
}

const DASHBOARD_VIEWS: Record<string, DashboardView> = {
  executive: {
    id: 'executive',
    name: 'Executive Dashboard',
    description: 'CEO-ready Bloomberg terminal aesthetic with key metrics',
    layout: 'bloomberg',
    panels: ['metrics', 'agents', 'knowledge', 'controls'],
    permissions: ['view_all'],
    theme: {
      primaryColor: 'var(--accent-primary)',
      backgroundColor: 'var(--bg-primary)',
      fontFamily: 'var(--font-ui)'
    }
  },
  technical: {
    id: 'technical', 
    name: 'Technical Dashboard',
    description: 'Developer-focused with detailed controls and debugging',
    layout: 'resizable',
    panels: ['agents', 'conversation', 'analytics', 'controls'],
    permissions: ['view_all', 'modify_agents', 'debug'],
    theme: {
      primaryColor: '#3B82F6',
      backgroundColor: '#0F172A',
      fontFamily: 'JetBrains Mono'
    }
  },
  research: {
    id: 'research',
    name: 'Research Dashboard', 
    description: 'Knowledge-focused interface for researchers and analysts',
    layout: 'knowledge',
    panels: ['knowledge', 'analytics', 'agents'],
    permissions: ['view_knowledge', 'export_data', 'analyze'],
    theme: {
      primaryColor: '#8B5CF6',
      backgroundColor: '#1E1B4B',
      fontFamily: 'Inter'
    }
  },
  minimal: {
    id: 'minimal',
    name: 'Minimal Dashboard',
    description: 'Clean, focused interface for specific tasks',
    layout: 'compact',
    panels: ['conversation', 'agents'],
    permissions: ['view_basic'],
    theme: {
      primaryColor: '#10B981',
      backgroundColor: '#FFFFFF',
      fontFamily: 'Inter'
    }
  }
};

// Panel component mapping
const PANEL_COMPONENTS = {
  agents: AgentPanel,
  conversation: ConversationPanel,
  analytics: AnalyticsPanel,
  knowledge: KnowledgePanel,
  controls: ControlPanel,
  metrics: MetricsPanel,
};

// Layout component mapping
const LAYOUT_COMPONENTS = {
  bloomberg: BloombergLayout,
  resizable: ResizableLayout,
  knowledge: KnowledgeLayout,
  compact: ResizableLayout, // Use resizable for compact
};

export default function UnifiedDashboard() {
  const dispatch = useAppDispatch();
  const searchParams = useSearchParams();
  
  // Get view from URL params or default to executive
  const viewParam = searchParams?.get('view') || 'executive';
  const [currentView, setCurrentView] = useState<string>(
    DASHBOARD_VIEWS[viewParam] ? viewParam : 'executive'
  );
  
  // Dashboard state
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [currentTime, setCurrentTime] = useState<string>('');
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  
  // Redux state
  const agents = useAppSelector(state => state.agents?.agents) || {};
  const agentOrder = useAppSelector(state => state.agents?.agentOrder) || [];
  const conversations = useAppSelector(state => state.conversations?.conversations) || {};
  const activeConversationId = useAppSelector(state => state.conversations?.activeConversationId) || 'main';
  const knowledgeGraph = useAppSelector(state => state.knowledge?.graph) || { nodes: {}, edges: {} };
  const analytics = useAppSelector(state => state.analytics) || { metrics: {}, snapshots: [] };
  
  // Get current view configuration
  const viewConfig = DASHBOARD_VIEWS[currentView];
  const LayoutComponent = LAYOUT_COMPONENTS[viewConfig.layout];
  
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
    setConnectionStatus('connecting');
    
    // Simulate connection status (replace with actual WebSocket events)
    setTimeout(() => setConnectionStatus('connected'), 1000);
    
    return () => {
      clearInterval(timeInterval);
      socketService.disconnect();
    };
  }, []);

  // Handle view changes
  const handleViewChange = useCallback((newView: string) => {
    if (DASHBOARD_VIEWS[newView]) {
      setCurrentView(newView);
      // Update URL without page reload
      const url = new URL(window.location.href);
      url.searchParams.set('view', newView);
      window.history.pushState({}, '', url.toString());
    }
  }, []);

  // Handle fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Loading state for SSR
  if (!isClient) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-[var(--accent-primary)] mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold mb-2">Initializing FreeAgentics Dashboard</h2>
          <p className="text-[var(--text-secondary)]">Loading unified multi-agent interface...</p>
        </div>
      </div>
    );
  }

  // Get statistics for header
  const stats = {
    totalAgents: Object.keys(agents).length,
    activeAgents: Object.values(agents).filter((a: any) => a.status === 'active').length,
    totalMessages: conversations[activeConversationId]?.messages?.length || 0,
    knowledgeNodes: Object.keys(knowledgeGraph.nodes).length,
  };

  return (
    <div 
      className="min-h-screen font-ui transition-all duration-300"
      style={{ 
        backgroundColor: viewConfig.theme.backgroundColor,
        fontFamily: viewConfig.theme.fontFamily 
      }}
    >
      {/* Unified Dashboard Header */}
      <header className="bg-[var(--bg-secondary)] border-b border-[var(--bg-tertiary)] px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left: Branding & View Info */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <Brain className="w-8 h-8" style={{ color: viewConfig.theme.primaryColor }} />
              <div>
                <h1 className="text-2xl font-bold text-[var(--text-primary)]">
                  FREEAGENTICS
                </h1>
                <p className="text-sm text-[var(--text-secondary)]">
                  {viewConfig.name} • {viewConfig.description}
                </p>
              </div>
            </div>
          </div>
          
          {/* Center: View Selector */}
          <div className="flex items-center gap-4">
            <Select value={currentView} onValueChange={handleViewChange}>
              <SelectTrigger className="w-48 bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.values(DASHBOARD_VIEWS).map((view) => (
                  <SelectItem key={view.id} value={view.id}>
                    <div className="flex items-center gap-2">
                      {view.id === 'executive' && <BarChart3 className="w-4 h-4" />}
                      {view.id === 'technical' && <Settings className="w-4 h-4" />}
                      {view.id === 'research' && <Network className="w-4 h-4" />}
                      {view.id === 'minimal' && <Monitor className="w-4 h-4" />}
                      <span>{view.name}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {/* Right: Status & Controls */}
          <div className="flex items-center gap-6 text-sm font-mono">
            <div className="flex items-center gap-2">
              <div 
                className={`w-2 h-2 rounded-full animate-pulse ${
                  connectionStatus === 'connected' ? 'bg-[var(--success)]' :
                  connectionStatus === 'connecting' ? 'bg-[var(--warning)]' : 'bg-[var(--error)]'
                }`}
              />
              <span className="text-[var(--text-secondary)]">
                {connectionStatus.toUpperCase()}
              </span>
            </div>
            
            <div className="text-[var(--text-secondary)]">{currentTime}</div>
            
            <div className="flex items-center gap-4">
              <span className="text-[var(--text-secondary)]">AGENTS: {stats.totalAgents}</span>
              <span className="text-[var(--text-secondary)]">ACTIVE: {stats.activeAgents}</span>
              <span className="text-[var(--text-secondary)]">MESSAGES: {stats.totalMessages}</span>
              <span className="text-[var(--text-secondary)]">KNOWLEDGE: {stats.knowledgeNodes}</span>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleFullscreen}
              className="text-[var(--text-secondary)]"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Dashboard Content */}
      <main className="h-[calc(100vh-80px)]">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            <LayoutComponent
              view={viewConfig}
              panels={viewConfig.panels.map(panelId => ({
                id: panelId,
                component: PANEL_COMPONENTS[panelId as keyof typeof PANEL_COMPONENTS],
                title: panelId.charAt(0).toUpperCase() + panelId.slice(1),
              }))}
            />
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Loading Suspense Fallback */}
      <Suspense fallback={
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[var(--bg-secondary)] rounded-lg p-6 flex items-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-[var(--accent-primary)]"></div>
            <span>Loading panel...</span>
          </div>
        </div>
      }>
        {/* Panels will be lazy loaded here */}
      </Suspense>
    </div>
  );
} 