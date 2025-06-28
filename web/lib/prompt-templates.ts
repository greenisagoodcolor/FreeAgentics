import type { Agent, Message, KnowledgeEntry } from "@/lib/types";

export interface PromptTemplate {
  systemPrompt: string;
  userPromptPrefix?: string;
  userPromptSuffix?: string;
  maxHistoryMessages?: number;
}

// Helper function to replace template variables
function replaceVariables(
  text: string,
  variables: Record<string, string>,
): string {
  return Object.entries(variables).reduce(
    (result, [key, value]) =>
      result.replace(new RegExp(`{{${key}}}`, "g"), value),
    text,
  );
}

// Format conversation history for inclusion in prompts
export function formatConversationHistory(
  messages: Message[],
  agents: Map<string, Agent>,
  maxMessages = 10,
): string {
  // Take the most recent messages up to maxMessages
  const recentMessages = messages.slice(-maxMessages);

  return recentMessages
    .map((msg) => {
      const sender =
        msg.senderId === "user"
          ? "User"
          : agents.get(msg.senderId)?.name || "Unknown Agent";

      return `${sender}: ${msg.content}`;
    })
    .join("\n\n");
}

// Format knowledge entries for inclusion in prompts
export function formatKnowledgeForPrompt(
  entries: KnowledgeEntry[],
  includeMetadata = false,
): string {
  if (!entries.length) return "No relevant knowledge available.";

  return entries
    .map((entry) => {
      let formatted = `KNOWLEDGE ENTRY: ${entry.title}\n${entry.content}`;

      if (includeMetadata) {
        formatted += `\nTags: ${entry.tags.join(", ")}`;
        formatted += `\nTimestamp: ${entry.timestamp.toISOString()}`;
      }

      return formatted;
    })
    .join("\n\n");
}

// Add a function to format the participants list for the prompt

// Format conversation participants for inclusion in prompts
export function formatParticipantsList(
  agents: Map<string, Agent>,
  currentAgentId: string,
): string {
  return Array.from(agents.values())
    .map((agent) => {
      const isCurrentAgent = agent.id === currentAgentId;
      return `- ${agent.name}${isCurrentAgent ? " (you)" : ""}: ${agent.biography.split(".")[0]}.`;
    })
    .join("\n");
}

// Assemble a complete prompt from template and variables
export function assemblePrompt(
  template: PromptTemplate,
  variables: Record<string, string>,
  conversationHistory: Message[],
  agents: Map<string, Agent>,
  relevantKnowledge?: KnowledgeEntry[],
): { systemPrompt: string; userPrompt: string } {
  // Replace variables in the system prompt
  const systemPrompt = replaceVariables(template.systemPrompt, variables);

  // Format conversation history
  const historyText = formatConversationHistory(
    conversationHistory,
    agents,
    template.maxHistoryMessages,
  );

  // Format knowledge if provided
  const knowledgeText = relevantKnowledge
    ? formatKnowledgeForPrompt(relevantKnowledge)
    : "";

  // Assemble user prompt with optional prefix and suffix
  let userPrompt = "";

  if (template.userPromptPrefix) {
    userPrompt +=
      replaceVariables(template.userPromptPrefix, variables) + "\n\n";
  }

  if (knowledgeText) {
    userPrompt += "RELEVANT KNOWLEDGE:\n" + knowledgeText + "\n\n";
  }

  userPrompt += "CONVERSATION HISTORY:\n" + historyText;

  if (template.userPromptSuffix) {
    userPrompt +=
      "\n\n" + replaceVariables(template.userPromptSuffix, variables);
  }

  return { systemPrompt, userPrompt };
}

// Define standard templates for different purposes

// Template for agent responses in conversation
export const agentResponseTemplate: PromptTemplate = {
  systemPrompt: `You are {{agentName}}, with the following biography: {{agentBiography}}

You are participating in a multi-agent conversation with the following participants:
{{participantsList}}

Your responses should be consistent with your character's knowledge, personality, and background.
You should respond naturally as if you are having a conversation with multiple participants.

IMPORTANT: Always start your response with "{{agentName}}:" followed by your message.

When a message is clearly directed at another agent (e.g., addressed by name), you should:
1. Only respond if you have something valuable to add
2. Acknowledge that the message was primarily for another agent
3. Keep your response brief and relevant

When a message is directed at you specifically, provide a complete and helpful response.
When a message is directed at everyone or no one specific, respond naturally.

You have access to your own knowledge base which will be provided in the prompt if relevant.
Only reference knowledge that is explicitly provided to you.`,

  userPromptSuffix: `Based on the conversation history and your knowledge, provide a response as {{agentName}}.
Your response should be a single message in a conversational tone.
Remember to start your response with "{{agentName}}:" followed by your message.
If the message was clearly directed at another agent and you don't have anything valuable to add, respond with "SKIP_RESPONSE" and I will not include your message.`,

  maxHistoryMessages: 10,
};

// Template for extracting beliefs from conversations
export const beliefExtractionTemplate: PromptTemplate = {
  systemPrompt: `You are an AI assistant that analyzes conversations and extracts potential new knowledge or beliefs.
Your task is to identify information, facts, or beliefs that should be added to an agent's knowledge base.
Focus on extracting factual information, preferences, opinions, and relationships mentioned in the conversation.

IMPORTANT: Format your response using Obsidian-style markdown. Use [[double brackets]] around important concepts, entities, and categories that should be tagged.`,

  userPromptPrefix: `The following is a conversation involving {{agentName}}.
Extract potential new knowledge or beliefs that {{agentName}} should remember from this conversation.
Pay special attention to: {{extractionPriorities}}`,

  userPromptSuffix: `List the extracted beliefs in bullet points. Each belief should be a concise statement of fact or opinion.
For each belief:
1. Use [[double brackets]] around key concepts that should be tagged
2. Indicate the confidence level (High/Medium/Low) based on how explicitly it was stated
3. Format the belief as a complete, well-structured markdown note

Example format:
- {{agentName}} believes that [[quantum computing]] will revolutionize [[cryptography]] within the next decade. (High)
- {{agentName}} seems to prefer [[coffee]] over [[tea]] based on their ordering habits. (Medium)`,

  maxHistoryMessages: 20,
};

// Template for relationship analysis
export const relationshipAnalysisTemplate: PromptTemplate = {
  systemPrompt: `You are an AI assistant that analyzes conversations to determine the relationship dynamics between participants.
Your task is to assess how {{agentName}} relates to other participants in the conversation.`,

  userPromptPrefix: `The following is a conversation involving {{agentName}} and other participants.
Analyze the conversation to determine {{agentName}}'s relationship with each other participant.`,

  userPromptSuffix: `For each participant that {{agentName}} interacted with, provide:
1. A sentiment score from -5 (very negative) to +5 (very positive)
2. A brief description of the relationship dynamic
3. Key moments in the conversation that support your analysis`,

  maxHistoryMessages: 15,
};
