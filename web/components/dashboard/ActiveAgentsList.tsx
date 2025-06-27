'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Activity, Brain, Users, Zap } from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  type: 'Explorer' | 'Guardian' | 'Merchant' | 'Scholar';
  status: 'active' | 'idle' | 'processing';
  lastActivity: string;
  performance: number;
}

const mockAgents: Agent[] = [
  {
    id: '1',
    name: 'Explorer-Alpha',
    type: 'Explorer',
    status: 'active',
    lastActivity: '2 min ago',
    performance: 94
  },
  {
    id: '2', 
    name: 'Guardian-Beta',
    type: 'Guardian',
    status: 'processing',
    lastActivity: '5 min ago',
    performance: 87
  },
  {
    id: '3',
    name: 'Merchant-Gamma',
    type: 'Merchant', 
    status: 'active',
    lastActivity: '1 min ago',
    performance: 91
  },
  {
    id: '4',
    name: 'Scholar-Delta',
    type: 'Scholar',
    status: 'idle',
    lastActivity: '15 min ago',
    performance: 96
  }
];

const getStatusColor = (status: Agent['status']) => {
  switch (status) {
    case 'active': return 'bg-green-500';
    case 'processing': return 'bg-yellow-500';
    case 'idle': return 'bg-gray-400';
    default: return 'bg-gray-400';
  }
};

const getAgentIcon = (type: Agent['type']) => {
  switch (type) {
    case 'Explorer': return <Zap className="h-4 w-4" />;
    case 'Guardian': return <Users className="h-4 w-4" />;
    case 'Merchant': return <Activity className="h-4 w-4" />;
    case 'Scholar': return <Brain className="h-4 w-4" />;
    default: return <Activity className="h-4 w-4" />;
  }
};

export function ActiveAgentsList() {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          Active Agents ({mockAgents.filter(a => a.status === 'active').length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {mockAgents.map((agent) => (
            <div key={agent.id} className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <Avatar className="h-10 w-10">
                  <AvatarFallback className="bg-primary/10">
                    {getAgentIcon(agent.type)}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <div className="font-medium">{agent.name}</div>
                  <div className="text-sm text-muted-foreground">{agent.type}</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="text-right">
                  <div className="text-sm font-medium">{agent.performance}%</div>
                  <div className="text-xs text-muted-foreground">{agent.lastActivity}</div>
                </div>
                <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`} />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
} 