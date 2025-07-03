import { NextResponse } from 'next/server';

// Mock agent data
const mockAgents = [
  {
    id: 'agent-1',
    name: 'Explorer Alpha',
    status: 'active',
    templateId: 'explorer',
    biography: 'An adventurous agent that discovers new territories and maps unknown environments. Specializes in exploration and discovery.',
    messageCount: 156,
    lastActive: Date.now() - 300000,
  },
  {
    id: 'agent-2',
    name: 'Scholar Beta',
    status: 'active',
    templateId: 'scholar',
    biography: 'A learned agent that analyzes patterns and synthesizes knowledge. Dedicated to understanding and teaching.',
    messageCount: 243,
    lastActive: Date.now() - 120000,
  },
  {
    id: 'agent-3',
    name: 'Merchant Gamma',
    status: 'idle',
    templateId: 'merchant',
    biography: 'A savvy trader that optimizes resource allocation and market dynamics. Expert in negotiations and value assessment.',
    messageCount: 89,
    lastActive: Date.now() - 600000,
  },
  {
    id: 'agent-4',
    name: 'Guardian Delta',
    status: 'active',
    templateId: 'guardian',
    biography: 'A protective agent that safeguards systems and responds to threats. Specializes in security and defense.',
    messageCount: 312,
    lastActive: Date.now() - 60000,
  },
];

export async function GET() {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 100));
  
  return NextResponse.json({
    agents: mockAgents,
    timestamp: Date.now(),
  });
}

export async function POST(request: Request) {
  const body = await request.json();
  
  // Mock creating a new agent
  const newAgent = {
    id: `agent-${Date.now()}`,
    name: body.name || 'New Agent',
    status: 'idle',
    templateId: body.templateId || 'explorer',
    biography: body.biography || 'A newly created agent ready to explore.',
    messageCount: 0,
    lastActive: Date.now(),
  };
  
  return NextResponse.json({
    agent: newAgent,
    message: 'Agent created successfully',
  });
}