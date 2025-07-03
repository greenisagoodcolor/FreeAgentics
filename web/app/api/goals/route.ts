import { NextResponse } from 'next/server';

// Current goal state
let currentGoal = {
  id: 'goal-1',
  text: 'Optimize resource allocation across agent networks',
  setAt: Date.now() - 3600000,
  setBy: 'system',
  progress: 0.67,
  status: 'in_progress',
};

export async function GET() {
  return NextResponse.json({
    goal: currentGoal,
    timestamp: Date.now(),
  });
}

export async function POST(request: Request) {
  const body = await request.json();
  
  // Update the current goal
  currentGoal = {
    id: `goal-${Date.now()}`,
    text: body.text || 'New goal',
    setAt: Date.now(),
    setBy: body.setBy || 'user',
    progress: 0,
    status: 'pending',
  };
  
  // Simulate broadcasting to agents
  await new Promise(resolve => setTimeout(resolve, 200));
  
  return NextResponse.json({
    goal: currentGoal,
    message: 'Goal updated and broadcast to all agents',
    affectedAgents: 4,
  });
}