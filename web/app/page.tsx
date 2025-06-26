"use client";

import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
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
  Search,
  ShoppingCart,
  BookOpen,
  Shield,
  Bot,
  Send,
  MoreHorizontal
} from "lucide-react";
import { cn } from "@/lib/utils";
import { FixedSizeList as List } from "react-window";

// Type Definitions from PRD
interface AgentTemplate {
  id: string;
  name: string;
  category: 'researcher' | 'student' | 'expert' | 'generalist' | 'contrarian';
  description: string;
  avatar: string;
  color: string;
  icon: React.ComponentType<any>;
}

interface Message {
  id: string;
  conversationId: string;
  agentId: string;
  content: string;
  timestamp: number;
  type: 'agent' | 'system' | 'user';
  metadata?: {
    respondingTo?: string;
    processingTime?: number;
    confidence?: number;
  };
  status: 'sending' | 'delivered' | 'failed';
}

interface KnowledgeNode {
  id: string;
  label: string;
  type: 'belief' | 'fact' | 'hypothesis';
  confidence: number;
  agents: string[];
  x?: number;
  y?: number;
}

interface Agent {
  id: string;
  name: string;
  template: string;
  status: 'active' | 'typing' | 'idle';
  avatar: string;
  color: string;
}

// Agent Templates (PRD Specification)
const AGENT_TEMPLATES: AgentTemplate[] = [
  {
    id: 'explorer',
    name: 'Explorer',
    category: 'researcher',
    description: 'Discovers new territories and maps environments',
    avatar: 'EX',
    color: '#10b981',
    icon: Search,
  },
  {
    id: 'merchant',
    name: 'Merchant', 
    category: 'expert',
    description: 'Optimizes resource trading and market dynamics',
    avatar: 'ME',
    color: '#3b82f6',
    icon: ShoppingCart,
  },
  {
    id: 'scholar',
    name: 'Scholar',
    category: 'student',
    description: 'Analyzes patterns and synthesizes knowledge',
    avatar: 'SC',
    color: '#8b5cf6',
    icon: BookOpen,
  },
  {
    id: 'guardian',
    name: 'Guardian',
    category: 'expert',
    description: 'Protects systems and responds to threats',
    avatar: 'GU',
    color: '#ef4444',
    icon: Shield,
  },
  {
    id: 'generalist',
    name: 'Generalist',
    category: 'generalist',
    description: 'Adaptable problem solver with broad capabilities',
    avatar: 'GE',
    color: '#f59e0b',
    icon: Brain,
  },
];

export default function MultiAgentDashboard() {
  // State Management
  const [agents, setAgents] = useState<Agent[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [knowledgeNodes, setKnowledgeNodes] = useState<KnowledgeNode[]>([]);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [typingAgents, setTypingAgents] = useState<Set<string>>(new Set());
  const [autoScroll, setAutoScroll] = useState(true);
  const [backendStatus, setBackendStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<List>(null);

  // Check backend connectivity
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status');
        if (response.ok) {
          setBackendStatus('connected');
        } else {
          setBackendStatus('disconnected');
        }
      } catch (error) {
        setBackendStatus('disconnected');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Real-time simulation
  useEffect(() => {
    if (!isSimulationRunning) return;

    const interval = setInterval(() => {
      // Simulate agent typing
      if (agents.length > 0 && Math.random() > 0.7) {
        const randomAgent = agents[Math.floor(Math.random() * agents.length)];
                 setTypingAgents(prev => new Set([...Array.from(prev), randomAgent.id]));
        
        // After 2-3 seconds, send message
        setTimeout(() => {
          const sampleMessages = [
            "I've discovered an interesting pattern in the resource distribution.",
            "The market conditions suggest we should adjust our trading strategy.",
            "My analysis indicates a correlation between agent cooperation and success rates.",
            "Security protocols are functioning optimally. No threats detected.",
            "I recommend forming a coalition to tackle this complex problem.",
            "The knowledge graph shows emerging consensus on this topic.",
            "Resource scarcity in sector 7 requires immediate attention.",
            "Fascinating! This finding contradicts our previous assumptions.",
          ];
          
          const newMessage: Message = {
            id: Date.now().toString(),
            conversationId: 'main',
            agentId: randomAgent.id,
            content: sampleMessages[Math.floor(Math.random() * sampleMessages.length)],
            timestamp: Date.now(),
            type: 'agent',
            status: 'delivered',
            metadata: {
              confidence: 0.7 + Math.random() * 0.3,
              processingTime: 500 + Math.random() * 1500,
            }
          };
          
          setMessages(prev => [...prev, newMessage]);
          setTypingAgents(prev => {
            const next = new Set(prev);
            next.delete(randomAgent.id);
            return next;
          });
          
          // Simulate knowledge graph updates
          if (Math.random() > 0.6) {
            const newNode: KnowledgeNode = {
              id: Date.now().toString(),
              label: `Belief ${knowledgeNodes.length + 1}`,
              type: ['belief', 'fact', 'hypothesis'][Math.floor(Math.random() * 3)] as any,
              confidence: 0.5 + Math.random() * 0.5,
              agents: [randomAgent.id],
              x: Math.random() * 400,
              y: Math.random() * 300,
            };
            setKnowledgeNodes(prev => [...prev, newNode]);
          }
        }, 2000 + Math.random() * 3000);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isSimulationRunning, agents, knowledgeNodes.length]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  // Agent creation
  const handleCreateAgent = (template: AgentTemplate) => {
    const newAgent: Agent = {
      id: Date.now().toString(),
      name: `${template.name} ${agents.length + 1}`,
      template: template.id,
      status: 'idle',
      avatar: template.avatar,
      color: template.color,
    };
    setAgents(prev => [...prev, newAgent]);
  };

  // Message renderer for virtualization
  const MessageRow = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const message = messages[index];
    const agent = agents.find(a => a.id === message.agentId);
    
    return (
      <div style={style} className="px-4 py-2">
        <div className="flex items-start gap-3 group">
          <Avatar className="h-8 w-8 border-2" style={{ borderColor: agent?.color }}>
            <AvatarFallback 
              className="text-xs font-bold"
              style={{ backgroundColor: agent?.color + '20', color: agent?.color }}
            >
              {agent?.avatar || 'AI'}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-sm" style={{ color: agent?.color }}>
                {agent?.name || 'Unknown Agent'}
              </span>
              <span className="text-xs text-muted-foreground font-mono">
                {new Date(message.timestamp).toLocaleTimeString()}
              </span>
              {message.metadata?.confidence && (
                <Badge variant="outline" className="text-xs">
                  {Math.round(message.metadata.confidence * 100)}% confidence
                </Badge>
              )}
            </div>
            <p className="text-sm text-foreground leading-relaxed">
              {message.content}
            </p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen bg-[#0A0A0B] text-white font-['Inter'] flex flex-col">
      {/* Bloomberg Terminal Header */}
      <header className="border-b border-zinc-800 bg-[#0A0A0B] px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-[#4F46E5]" />
              <h1 className="text-lg font-bold">FREEAGENTICS</h1>
              <Badge variant="outline" className="text-[#10B981] border-[#10B981]">
                LIVE
              </Badge>
            </div>
            <div className="text-xs text-zinc-400 font-mono">
              Multi-Agent Research Dashboard v2.1.0
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="text-xs text-zinc-400 font-mono">
              {new Date().toLocaleTimeString()} EST
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
      <div className="border-b border-zinc-800 bg-[#0A0A0B] px-6 py-2">
        <div className="flex items-center gap-8 text-xs font-mono">
          <div className="flex items-center gap-2">
            <Users className="h-3 w-3 text-[#4F46E5]" />
            <span className="text-zinc-400">AGENTS:</span>
            <span className="text-[#10B981]">{agents.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <MessageSquare className="h-3 w-3 text-[#4F46E5]" />
            <span className="text-zinc-400">MESSAGES:</span>
            <span className="text-[#10B981]">{messages.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <Network className="h-3 w-3 text-[#4F46E5]" />
            <span className="text-zinc-400">KNOWLEDGE:</span>
            <span className="text-[#10B981]">{knowledgeNodes.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="h-3 w-3 text-[#4F46E5]" />
            <span className="text-zinc-400">STATUS:</span>
            <span className={isSimulationRunning ? "text-[#10B981]" : "text-[#F59E0B]"}>
              {isSimulationRunning ? 'ACTIVE' : 'STANDBY'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="h-3 w-3 text-[#4F46E5]" />
            <span className="text-zinc-400">BACKEND:</span>
            <span className={
              backendStatus === 'connected' ? "text-[#10B981]" : 
              backendStatus === 'connecting' ? "text-[#F59E0B]" : "text-[#EF4444]"
            }>
              {backendStatus.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {/* Three-Column Resizable Layout */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left Panel: Agent Management */}
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className="h-full border-r border-zinc-800 bg-[#0A0A0B]">
            <div className="p-4 border-b border-zinc-800">
              <h2 className="font-semibold text-sm text-zinc-300 mb-3">AGENT TEMPLATES</h2>
              <div className="space-y-2">
                {AGENT_TEMPLATES.map((template) => (
                  <Card 
                    key={template.id}
                    className="bg-zinc-900 border-zinc-700 hover:border-zinc-600 cursor-pointer transition-all p-3"
                    onClick={() => handleCreateAgent(template)}
                  >
                    <div className="flex items-center gap-3">
                      <div 
                        className="p-2 rounded-md"
                        style={{ backgroundColor: template.color + '20' }}
                      >
                        <template.icon className="h-4 w-4" style={{ color: template.color }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{template.name}</div>
                        <div className="text-xs text-zinc-400 truncate">
                          {template.description}
                        </div>
                      </div>
                      <Plus className="h-4 w-4 text-zinc-500" />
                    </div>
                  </Card>
                ))}
              </div>
            </div>
            
            <div className="p-4">
              <h2 className="font-semibold text-sm text-zinc-300 mb-3">ACTIVE AGENTS</h2>
              <div className="space-y-2">
                {agents.map((agent) => (
                  <div key={agent.id} className="flex items-center gap-3 p-2 rounded bg-zinc-900">
                    <Avatar className="h-6 w-6 border" style={{ borderColor: agent.color }}>
                      <AvatarFallback 
                        className="text-xs"
                        style={{ backgroundColor: agent.color + '20', color: agent.color }}
                      >
                        {agent.avatar}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium truncate">{agent.name}</div>
                      <div className="text-xs text-zinc-500 flex items-center gap-1">
                        {typingAgents.has(agent.id) ? (
                          <>
                            <div className="animate-pulse w-1 h-1 bg-[#10B981] rounded-full"></div>
                            typing...
                          </>
                        ) : (
                          'idle'
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-zinc-800 hover:bg-zinc-700" />

        {/* Center Panel: Conversation Feed */}
        <ResizablePanel defaultSize={50} minSize={30}>
          <div className="h-full flex flex-col bg-[#0A0A0B]">
            <div className="p-4 border-b border-zinc-800">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-sm text-zinc-300">CONVERSATION FEED</h2>
                <div className="flex items-center gap-2">
                  {typingAgents.size > 0 && (
                    <div className="text-xs text-zinc-400 flex items-center gap-1">
                      <div className="animate-pulse w-1 h-1 bg-[#10B981] rounded-full"></div>
                      {typingAgents.size} agent{typingAgents.size > 1 ? 's' : ''} typing...
                    </div>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setAutoScroll(!autoScroll)}
                    className={cn("text-xs", autoScroll ? "text-[#10B981]" : "text-zinc-400")}
                  >
                    {autoScroll ? "LIVE" : "PAUSED"}
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="flex-1 relative">
              {messages.length > 0 ? (
                                 <List
                   ref={listRef}
                   height={600}
                   width="100%"
                   itemCount={messages.length}
                   itemSize={80}
                   className="scrollbar-thin scrollbar-track-zinc-800 scrollbar-thumb-zinc-600"
                 >
                   {MessageRow}
                 </List>
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-500">
                  <div className="text-center">
                    <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No conversations yet</p>
                    <p className="text-xs text-zinc-600 mt-1">Create agents and start simulation</p>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-zinc-800 hover:bg-zinc-700" />

        {/* Right Panel: Knowledge Graph */}
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className="h-full border-l border-zinc-800 bg-[#0A0A0B]">
            <div className="p-4 border-b border-zinc-800">
              <h2 className="font-semibold text-sm text-zinc-300">KNOWLEDGE GRAPH</h2>
            </div>
            
            <div className="p-4">
              {knowledgeNodes.length > 0 ? (
                <div className="space-y-2">
                  {knowledgeNodes.slice(-10).map((node) => (
                    <div 
                      key={node.id} 
                      className="p-2 rounded bg-zinc-900 border border-zinc-700 hover:border-zinc-600 cursor-pointer"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium">{node.label}</span>
                        <Badge 
                          variant="outline" 
                          className="text-xs"
                          style={{ 
                            borderColor: node.confidence > 0.8 ? '#10B981' : node.confidence > 0.6 ? '#F59E0B' : '#EF4444',
                            color: node.confidence > 0.8 ? '#10B981' : node.confidence > 0.6 ? '#F59E0B' : '#EF4444' 
                          }}
                        >
                          {Math.round(node.confidence * 100)}%
                        </Badge>
                      </div>
                      <div className="text-xs text-zinc-400">
                        {node.type} • {node.agents.length} agent{node.agents.length > 1 ? 's' : ''}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-32 text-zinc-500">
                  <div className="text-center">
                    <Network className="h-6 w-6 mx-auto mb-2 opacity-50" />
                    <p className="text-xs">No knowledge nodes yet</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
