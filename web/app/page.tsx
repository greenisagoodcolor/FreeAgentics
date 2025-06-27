"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { motion } from "framer-motion";
import {
  Brain,
  Users,
  MessageSquare,
  Plus,
  Play,
  Pause,
  Activity,
  Clock,
  Zap,
  TrendingUp,
  Network,
} from "lucide-react";

export default function MultiAgentDashboard() {
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [currentTime, setCurrentTime] = useState<string>('');
  const [isClient, setIsClient] = useState(false);

  // Initialize client-side only state
  useEffect(() => {
    setIsClient(true);
    setCurrentTime(new Date().toLocaleTimeString());
    
    // Update time every second
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);

    return () => {
      clearInterval(timeInterval);
    };
  }, []);

  if (!isClient) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-[var(--accent-primary)]"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)]">
      {/* Header */}
      <div className="border-b border-[var(--bg-tertiary)] bg-[var(--bg-secondary)]">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-8 w-8 text-[var(--accent-primary)]" />
              <h1 className="text-2xl font-bold">FreeAgentics MVP</h1>
            </div>
            <Badge variant="outline" className="text-xs">
              {currentTime}
            </Badge>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              Connected
            </div>
            <Button
              variant={isSimulationRunning ? "destructive" : "default"}
              size="sm"
              onClick={() => setIsSimulationRunning(!isSimulationRunning)}
              className="gap-2"
            >
              {isSimulationRunning ? (
                <>
                  <Pause className="h-4 w-4" />
                  Pause Simulation
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Start Simulation
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Dashboard */}
      <div className="h-[calc(100vh-80px)]">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* Left Panel - Agents */}
          <ResizablePanel defaultSize={25} minSize={20} maxSize={35}>
            <div className="h-full flex flex-col bg-[var(--bg-secondary)]">
              <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
                <h2 className="font-semibold flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  Active Agents (0)
                </h2>
                <Button size="sm" variant="outline" className="gap-2">
                  <Plus className="h-4 w-4" />
                  Add Agent
                </Button>
              </div>
              
              <div className="flex-1 p-4">
                <div className="text-center text-[var(--text-secondary)] py-8">
                  <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="text-sm">No agents created yet</p>
                  <p className="text-xs mt-1">Create your first agent to get started</p>
                </div>
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Center Panel - Conversations */}
          <ResizablePanel defaultSize={45} minSize={40}>
            <div className="h-full flex flex-col bg-[var(--bg-primary)]">
              <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
                <h2 className="font-semibold flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Live Conversations
                </h2>
                <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
                  <Activity className="h-4 w-4" />
                  0 messages
                </div>
              </div>
              
              <div className="flex-1 p-4">
                <div className="text-center text-[var(--text-secondary)] py-12">
                  <MessageSquare className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium mb-2">Ready for Conversations</p>
                  <p className="text-sm">Start the simulation to see agents interact</p>
                </div>
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right Panel - Analytics & Knowledge */}
          <ResizablePanel defaultSize={30} minSize={25} maxSize={40}>
            <div className="h-full flex flex-col bg-[var(--bg-secondary)]">
              <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
                <h2 className="font-semibold flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Analytics
                </h2>
              </div>
              
              <div className="flex-1 p-4 space-y-4">
                {/* Quick Stats */}
                <div className="grid grid-cols-2 gap-3">
                  <Card className="p-3 bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)]">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-[var(--accent-primary)]" />
                      <span className="text-sm font-medium">Uptime</span>
                    </div>
                    <p className="text-lg font-bold mt-1">0:00:00</p>
                  </Card>
                  
                  <Card className="p-3 bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)]">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-[var(--warning)]" />
                      <span className="text-sm font-medium">Activity</span>
                    </div>
                    <p className="text-lg font-bold mt-1">0%</p>
                  </Card>
                </div>

                {/* Knowledge Graph Placeholder */}
                <Card className="p-4 bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)]">
                  <div className="flex items-center gap-2 mb-3">
                    <Network className="h-4 w-4" />
                    <span className="font-medium">Knowledge Graph</span>
                  </div>
                  <div className="text-center text-[var(--text-secondary)] py-8">
                    <Network className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">No knowledge nodes yet</p>
                  </div>
                </Card>
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
