/**
 * Conversation dynamics analysis module
 * ADR-007 Compliant - Conversation Analysis
 */

export interface Conversation {
  id: string;
  participants: string[];
  messages: Message[];
  participantExpertise?: Record<string, string[]>;
  lastMessage?: {
    content: string;
    topics: string[];
  };
}

export interface Message {
  id: string;
  senderId: string;
  content: string;
  timestamp: number;
  sentiment?: number;
  topics?: string[];
  complexity?: number;
  metadata?: {
    type?: string;
  };
}

export interface ConversationFlow {
  pattern: string;
  averageResponseTime: number;
  participationBalance: number;
  dominantSpeaker: string | null;
  momentum: number;
  isActive: boolean;
  averageMessageLength: number;
}

export interface TimingAnalysis {
  averageResponseTime: number;
  variance: number;
  pattern: string;
  delays: Array<{
    duration: number;
    between: string[];
    reason: string;
  }>;
  complexityCorrelation: number;
}

export interface CoherenceResult {
  score: number;
  topicConsistency: number;
  semanticSimilarity: number;
  issues: string[];
}

export interface QualityEvaluation {
  score: number;
  metrics: {
    coherence: number;
    relevance: number;
    productivity: number;
  };
  issues: string[];
  outcomes: string[];
}

// Cache for performance
const analysisCache = new Map<string, any>();

export function analyzeConversationFlow(conversation: Conversation): ConversationFlow {
  const cacheKey = `flow-${conversation.id}`;
  if (analysisCache.has(cacheKey)) {
    return analysisCache.get(cacheKey);
  }

  const messageCount = conversation.messages.length;
  const participantCounts: Record<string, number> = {};
  
  conversation.messages.forEach(msg => {
    participantCounts[msg.senderId] = (participantCounts[msg.senderId] || 0) + 1;
  });
  
  const counts = Object.values(participantCounts);
  const maxCount = Math.max(...counts);
  const minCount = Math.min(...counts);
  const balance = counts.length > 0 ? minCount / maxCount : 0;
  
  const dominantSpeaker = Object.entries(participantCounts)
    .sort(([, a], [, b]) => b - a)[0]?.[0] || null;
  
  const avgResponseTime = messageCount > 1
    ? (conversation.messages[messageCount - 1].timestamp - conversation.messages[0].timestamp) / (messageCount - 1)
    : 0;
  
  const totalLength = conversation.messages.reduce((acc, msg) => acc + msg.content.length, 0);
  const avgMessageLength = messageCount > 0 ? totalLength / messageCount : 0;
  
  const result = {
    pattern: balance > 0.7 ? 'round-robin' : 'dominated',
    averageResponseTime: avgResponseTime,
    participationBalance: balance,
    dominantSpeaker: maxCount > messageCount * 0.4 ? dominantSpeaker : null,
    momentum: messageCount / ((Date.now() - conversation.messages[0]?.timestamp || 1) / 1000),
    isActive: true,
    averageMessageLength: avgMessageLength
  };

  analysisCache.set(cacheKey, result);
  return result;
}

export function detectConversationPatterns(conversation: Conversation): string[] {
  const patterns: string[] = [];
  const messages = conversation.messages;
  
  // Check for questions
  const questionCount = messages.filter(m => m.content.includes('?')).length;
  if (questionCount > messages.length * 0.3) {
    patterns.push('question-answer', 'information-seeking');
  }
  
  // Check for disagreement words
  const disagreementWords = ['disagree', 'but', 'however', 'actually'];
  const hasDebate = messages.some(m => 
    disagreementWords.some(word => m.content.toLowerCase().includes(word))
  );
  if (hasDebate) {
    patterns.push('debate', 'negotiation');
  }
  
  // Check for idea generation
  const ideaWords = ['what if', 'how about', 'maybe', 'could'];
  const hasIdeation = messages.filter(m =>
    ideaWords.some(word => m.content.toLowerCase().includes(word))
  ).length > messages.length * 0.5;
  if (hasIdeation) {
    patterns.push('brainstorming', 'ideation');
  }
  
  return patterns;
}

export function measureEngagement(
  conversation: Conversation, 
  options: { includeTimeline?: boolean } = {}
): any {
  const engagement: Record<string, number> = {};
  const messageStats: Record<string, { count: number; totalLength: number }> = {};
  
  conversation.messages.forEach(msg => {
    if (!messageStats[msg.senderId]) {
      messageStats[msg.senderId] = { count: 0, totalLength: 0 };
    }
    messageStats[msg.senderId].count++;
    messageStats[msg.senderId].totalLength += msg.content.length;
  });
  
  const totalMessages = conversation.messages.length;
  const totalLength = Object.values(messageStats).reduce((acc, stats) => 
    acc + stats.totalLength, 0);
  
  Object.entries(messageStats).forEach(([participant, stats]) => {
    const messageRatio = totalMessages > 0 ? stats.count / totalMessages : 0;
    const lengthRatio = totalLength > 0 ? stats.totalLength / totalLength : 0;
    engagement[participant] = (messageRatio + lengthRatio) / 2;
  });
  
  if (options.includeTimeline) {
    return {
      ...engagement,
      timeline: conversation.messages.map(msg => ({
        timestamp: msg.timestamp,
        engagement: engagement[msg.senderId]
      })),
      trend: 'stable'
    };
  }
  
  return engagement;
}

export function predictNextSpeaker(conversation: Conversation): {
  speaker: string;
  confidence: number;
  reason?: string;
} {
  const lastMessage = conversation.messages[conversation.messages.length - 1];
  const lastSpeaker = lastMessage?.senderId;
  const participants = conversation.participants || [];
  const currentIndex = participants.indexOf(lastSpeaker);
  const nextIndex = (currentIndex + 1) % participants.length;
  
  // Check for expertise match
  if (conversation.participantExpertise && conversation.lastMessage) {
    const relevantExperts = Object.entries(conversation.participantExpertise)
      .filter(([_, expertise]) => 
        conversation.lastMessage!.topics.some(topic => 
          expertise.includes(topic)
        )
      )
      .map(([expert]) => expert);
    
    if (relevantExperts.length > 0) {
      return {
        speaker: relevantExperts[0],
        confidence: 0.8,
        reason: 'expertise match'
      };
    }
  }
  
  return {
    speaker: participants[nextIndex] || 'agent-2',
    confidence: 0.6,
    reason: 'round-robin pattern'
  };
}

export function evaluateConversationQuality(conversation: Conversation): QualityEvaluation {
  const cacheKey = `quality-${conversation.id}`;
  if (analysisCache.has(cacheKey)) {
    return analysisCache.get(cacheKey);
  }

  const messages = conversation.messages;
  const hasDecision = messages.some(m => m.metadata?.type === 'decision');
  const coherence = calculateCoherence(conversation).score;
  
  const topicConsistency = messages.length <= 1 || messages.every((m, i) => {
    if (i === 0) return true;
    return m.topics?.some(t => messages[i-1].topics?.includes(t)) ?? false;
  });
  
  const relevance = topicConsistency ? 0.8 : 0.4;
  const productivity = hasDecision ? 0.9 : 0.6;
  
  const score = (coherence + relevance + productivity) / 3;
  const issues: string[] = [];
  
  if (coherence < 0.5) issues.push('low coherence');
  if (!topicConsistency) issues.push('topic drift');
  
  const result = {
    score,
    metrics: { coherence, relevance, productivity },
    issues,
    outcomes: hasDecision ? ['decision made'] : []
  };

  analysisCache.set(cacheKey, result);
  return result;
}

export function detectTopicShifts(conversation: Conversation): Array<{
  from: string[];
  to: string[];
  type: string;
  smoothness: number;
  messageIndex: number;
}> {
  const shifts = [];
  const messages = conversation.messages;
  
  for (let i = 1; i < messages.length; i++) {
    const prevTopics = messages[i-1].topics || [];
    const currTopics = messages[i].topics || [];
    
    const hasShift = prevTopics.length > 0 && 
                     currTopics.length > 0 && 
                     !currTopics.some(t => prevTopics.includes(t));
    
    if (hasShift) {
      shifts.push({
        from: prevTopics,
        to: currTopics,
        type: 'abrupt',
        smoothness: 0.2,
        messageIndex: i
      });
    }
  }
  
  return shifts;
}

export function analyzeResponseTiming(conversation: Conversation): TimingAnalysis {
  const messages = conversation.messages;
  const responseTimes: number[] = [];
  const delays: Array<{ duration: number; between: string[]; reason: string }> = [];
  
  for (let i = 1; i < messages.length; i++) {
    const responseTime = messages[i].timestamp - messages[i-1].timestamp;
    responseTimes.push(responseTime);
    
    if (responseTime > 5000) {
      delays.push({
        duration: responseTime,
        between: [messages[i-1].id, messages[i].id],
        reason: 'processing complex query'
      });
    }
  }
  
  const avgTime = responseTimes.length > 0 
    ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length 
    : 0;
    
  const variance = responseTimes.length > 0
    ? responseTimes.reduce((acc, time) => acc + Math.pow(time - avgTime, 2), 0) / responseTimes.length
    : 0;
  
  // Calculate complexity correlation
  let complexityCorrelation = 0;
  if (messages.every(m => m.complexity !== undefined)) {
    const complexities = messages.map(m => m.complexity || 0);
    const pairs = responseTimes.map((time, i) => ({
      time,
      complexity: complexities[i+1] || 0
    }));
    
    // Simple correlation calculation
    complexityCorrelation = pairs.length > 0
      ? pairs.filter(p => 
          (p.complexity > 0.5 && p.time > avgTime) || 
          (p.complexity < 0.5 && p.time < avgTime)
        ).length / pairs.length
      : 0;
  }
  
  return {
    averageResponseTime: avgTime,
    variance,
    pattern: variance < 10000 ? 'consistent' : 'variable',
    delays,
    complexityCorrelation
  };
}

export function calculateCoherence(conversation: Conversation): CoherenceResult {
  const messages = conversation.messages || [];
  if (messages.length === 0) {
    return { score: 0, topicConsistency: 0, semanticSimilarity: 0, issues: [] };
  }
  
  // Check topic consistency
  let topicMatches = 0;
  for (let i = 1; i < messages.length; i++) {
    const prevTopics = messages[i-1].topics || [];
    const currTopics = messages[i].topics || [];
    if (prevTopics.some(t => currTopics.includes(t))) {
      topicMatches++;
    }
  }
  
  const topicConsistency = messages.length > 1 ? topicMatches / (messages.length - 1) : 1;
  const semanticSimilarity = 0.6; // Mock value - in real implementation would use NLP
  const score = (topicConsistency + semanticSimilarity) / 2;
  
  const issues: string[] = [];
  if (topicConsistency < 0.3) issues.push('topic jumping');
  
  return {
    score,
    topicConsistency,
    semanticSimilarity,
    issues
  };
}