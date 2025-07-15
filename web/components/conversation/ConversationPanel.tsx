// Temporary ConversationPanel component to fix TypeScript errors

export interface ConversationMessage {
  id: string;
  agentId: string;
  agent_id: string; // snake_case for compatibility
  user_id?: string;
  conversation_id?: string; // for compatibility
  message: string;
  content: string; // content field for compatibility
  timestamp: string;
  type: "user" | "agent" | "system";
  message_type: string; // snake_case for compatibility
  metadata?: Record<string, any>;
}

export interface ConversationPanelProps {
  messages: ConversationMessage[];
  onSendMessage?: (message: string) => void;
  isLoading?: boolean;
}

// Placeholder component - implement actual UI as needed
export default function ConversationPanel({
  messages,
  onSendMessage,
  isLoading = false,
}: ConversationPanelProps) {
  return (
    <div className="conversation-panel">
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.type}`}>
            <span className="timestamp">{msg.timestamp}</span>
            <span className="content">{msg.content || msg.message}</span>
          </div>
        ))}
      </div>
      {isLoading && <div>Loading...</div>}
    </div>
  );
}
