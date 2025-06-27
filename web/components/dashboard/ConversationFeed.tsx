'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MessageSquare, Clock } from 'lucide-react';

interface Conversation {
  id: string;
  agents: string[];
  topic: string;
  status: 'active' | 'completed' | 'pending';
  timestamp: string;
  messageCount: number;
}

const mockConversations: Conversation[] = [
  {
    id: '1',
    agents: ['Explorer-Alpha', 'Scholar-Delta'],
    topic: 'Market Analysis Collaboration',
    status: 'active',
    timestamp: '2 min ago',
    messageCount: 23
  },
  {
    id: '2',
    agents: ['Guardian-Beta', 'Merchant-Gamma'],
    topic: 'Risk Assessment Protocol',
    status: 'completed',
    timestamp: '15 min ago', 
    messageCount: 45
  }
];

export function ConversationFeed() {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Active Conversations
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {mockConversations.map((conv) => (
            <div key={conv.id} className="p-3 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium">{conv.topic}</div>
                <Badge variant={conv.status === 'active' ? 'default' : 'secondary'}>
                  {conv.status}
                </Badge>
              </div>
              <div className="text-sm text-muted-foreground mb-2">
                {conv.agents.join(' ↔ ')}
              </div>
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{conv.messageCount} messages</span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {conv.timestamp}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
} 