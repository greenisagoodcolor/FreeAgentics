"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { Agent, Conversation, Message } from "@/lib/types";
import {
  Send,
  X,
  Loader2,
  CornerDownRight,
  AlertTriangle,
  Upload,
} from "lucide-react";
import { useConversationOrchestrator } from "@/hooks/useConversationorchestrator";
import { createLogger } from "@/lib/debug-logger";

// Create a logger for this component
const logger = createLogger("ChatWindow");

interface ChatWindowProps {
  conversation: Conversation | null;
  agents: Agent[];
  onSendMessage: (content: string, senderId: string) => void;
  onEndConversation: () => void;
}

export default function ChatWindow({
  conversation,
  agents,
  onSendMessage,
  onEndConversation,
}: ChatWindowProps) {
  const [message, setMessage] = useState("");
  const [userAgentId, setUserAgentId] = useState<string | "user">("user");
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [processingMessageMap, setProcessingMessageMap] = useState<
    Record<string, boolean>
  >({});
  const processedMessageRef = useRef<string | null>(null);

  // Use the conversation orchestrator with the onSendMessage callback
  const {
    queueAgentResponse,
    processNewMessage,
    cancelAllResponses,
    processingAgents,
    queuedAgents,
    typingAgents,
    processingMessageIds,
    isProcessing,
    error: orchestratorError,
  } = useConversationOrchestrator(
    conversation,
    agents,
    {
      autoSelectRespondents: true,
      responseDelay: [800, 2000],
    },
    onSendMessage,
  );

  // Update the processing message map to prevent UI flashing
  useEffect(() => {
    // Create a stable map of which messages are being processed
    setProcessingMessageMap((prev) => {
      const newMap = { ...prev };

      // Add new processing messages
      for (const messageId of processingMessageIds) {
        newMap[messageId] = true;
      }

      // Remove messages that are no longer being processed
      // Only if there are no typing agents (to prevent flashing)
      if (Object.keys(typingAgents).length === 0 && !isProcessing) {
        for (const messageId of Object.keys(newMap)) {
          if (!processingMessageIds.includes(messageId)) {
            delete newMap[messageId];
          }
        }
      }

      return newMap;
    });
  }, [processingMessageIds, typingAgents, isProcessing]);

  // Combine local and orchestrator errors
  const error = localError || orchestratorError;

  // Handle sending a message
  const handleSendMessage = () => {
    if (message.trim() && conversation && !isSending) {
      setIsSending(true);
      setLocalError(null); // Clear any previous local errors

      // Create the new message
      const newMessage: Message = {
        id: `msg-${Date.now()}`,
        content: message,
        senderId: userAgentId,
        timestamp: new Date(),
      };

      // Clear the input first to provide immediate feedback
      const messageContent = message;
      setMessage("");

      try {
        // Send the message (this updates the conversation state)
        onSendMessage(messageContent, userAgentId);

        // Wait a brief moment to ensure the conversation state updates
        setTimeout(() => {
          try {
            // Double-check that conversation still exists
            if (!conversation) {
              throw new Error("Conversation no longer exists");
            }

            logger.log(
              `Calling processNewMessage for message: ${newMessage.id}`,
            );

            // Process all messages, including conversation starters
            processNewMessage(newMessage);
          } catch (err) {
            logger.error("Error processing message:", err);
            setLocalError(
              `Failed to process message: ${err instanceof Error ? err.message : String(err)}`,
            );
          } finally {
            setIsSending(false);
          }
        }, 100);
      } catch (err) {
        logger.error("Error sending message:", err);
        setLocalError(
          `Failed to send message: ${err instanceof Error ? err.message : String(err)}`,
        );
        setIsSending(false);
        // Restore the message in the input field so the user doesn't lose their text
        setMessage(messageContent);
      }
    }
  };

  // Check for direct mentions to prompt immediate responses
  useEffect(() => {
    if (!conversation || !conversation.messages) return;

    const latestMessage =
      conversation.messages[conversation?.messages.length - 1];
    if (!latestMessage) return;

    // Create a ref to track if we've already processed this message
    if (processedMessageRef.current === latestMessage.id) return;

    // Log conversation starter messages
    if (latestMessage.metadata?.type === "conversation_starter") {
      logger.log("Detected conversation starter message:", {
        messageId: latestMessage.id,
        content: latestMessage.content.substring(0, 30) + "...",
        senderId: latestMessage.senderId,
        metadata: latestMessage.metadata,
      });

      // If we're not already processing, trigger responses
      if (!isProcessing && !isSending) {
        logger.log("Triggering responses to conversation starter message");

        // Get all agents in the conversation except the sender
        const respondingAgents = agents.filter(
          (agent) =>
            conversation.participants.includes(agent.id) &&
            agent.id !== latestMessage.senderId,
        );

        logger.log(
          `Found ${respondingAgents.length} agents to respond to conversation starter:`,
          respondingAgents.map((a) => a.name),
        );

        // Queue responses from all agents with slight delays
        respondingAgents.forEach((agent, index) => {
          logger.log(
            `Queueing response from ${agent.name} to conversation starter`,
          );
          queueAgentResponse(agent.id, {
            messageToRespondTo: latestMessage,
            responseDelay: 500 + index * 1000, // Stagger responses
            force: true, // Force response regardless of dynamics
          });
        });

        // Mark this message as processed
        processedMessageRef.current = latestMessage.id;
      } else {
        logger.log(
          `Not triggering responses to conversation starter: isProcessing=${isProcessing}, isSending=${isSending}`,
        );
      }
    }

    // Also handle system prompts that are trying to restart a stalled conversation
    if (
      latestMessage.metadata?.type === "conversation_prompt" &&
      latestMessage.senderId === "system"
    ) {
      logger.log("Detected conversation prompt message:", {
        messageId: latestMessage.id,
        content: latestMessage.content,
      });

      // Extract the agent name from the message (format: "Agent X, what do you think...")
      const agentNameMatch = latestMessage.content.match(/^([^,]+),/);
      if (agentNameMatch) {
        const agentName = agentNameMatch[1].trim();
        const agent = agents.find((a) => a.name === agentName);

        if (agent) {
          logger.log(
            `Queueing response from ${agent.name} to conversation prompt`,
          );
          queueAgentResponse(agent.id, {
            messageToRespondTo: latestMessage,
            responseDelay: 500,
            force: true,
          });

          // Mark this message as processed
          processedMessageRef.current = latestMessage.id;
        }
      }
    }

    // Skip if it's not a user message or we're already processing
    if (latestMessage.senderId !== "user" || isSending || isProcessing) return;

    // Check for direct mentions like "Agent 2, [message]" or "@Agent 2 [message]"
    const mentionMatch = latestMessage.content.match(/^(?:@?(.+?),?\s+)/i);
    if (!mentionMatch) return;

    const mentionedName = mentionMatch[1];

    // Find the mentioned agent
    const mentionedAgent = agents.find(
      (agent) =>
        agent.name.toLowerCase() === mentionedName.toLowerCase() ||
        agent.name.toLowerCase().startsWith(mentionedName.toLowerCase()),
    );

    // If we found a matching agent and they're in the conversation, prioritize their response
    if (
      mentionedAgent &&
      conversation.participants.includes(mentionedAgent.id)
    ) {
      queueAgentResponse(mentionedAgent.id, {
        messageToRespondTo: latestMessage,
        responseDelay: 300, // Quick response for direct mentions
        force: true, // Force response regardless of dynamics
      });

      // Mark this message as processed
      processedMessageRef.current = latestMessage.id;
    }
  }, [
    conversation,
    agents,
    isSending,
    isProcessing,
    queueAgentResponse,
  ]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [conversation?.messages, typingAgents]);

  // Get agent by ID helper
  const getAgentById = (id: string) => {
    return agents.find((agent) => agent.id === id);
  };

  // Find message by ID
  const getMessageById = (id: string) => {
    return conversation?.messages?.find((msg) => msg.id === id);
  };

  // Clear error after 5 seconds
  useEffect(() => {
    if (localError) {
      const timer = setTimeout(() => {
        setLocalError(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [localError]);

  // Force agents to respond if they haven't after a timeout
  useEffect(() => {
    if (
      !conversation ||
      !conversation.messages ||
      conversation.messages.length === 0 ||
      isProcessing ||
      isSending
    )
      return;

    // Get the latest message
    const latestMessage =
      conversation.messages[conversation.messages.length - 1];
    if (!latestMessage) return;

    // Only apply this to user-initiated conversations (not autonomous ones)
    // And only when the latest message is from the user
    if (
      latestMessage.senderId === "user" &&
      !isProcessing &&
      !isSending &&
      !conversation.isAutonomous
    ) {
      const timer = setTimeout(() => {
        // Only proceed if we're still not processing and the conversation exists
        if (
          !isProcessing &&
          !isSending &&
          conversation &&
          conversation.participants &&
          conversation.participants.length > 0
        ) {
          logger.log(
            "No automatic responses detected, forcing agent responses",
          );

          // Get all agents in the conversation except the sender
          const respondingAgents = agents.filter(
            (agent) =>
              conversation.participants.includes(agent.id) &&
              agent.id !== latestMessage.senderId,
          );

          // Queue responses from all agents
          respondingAgents.forEach((agent) => {
            queueAgentResponse(agent.id, {
              messageToRespondTo: latestMessage,
              responseDelay: 500 + Math.random() * 1000, // Stagger responses
              force: true,
            });
          });
        }
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [
    conversation?.messages,
    isProcessing,
    isSending,
    agents,
    conversation,
    queueAgentResponse,
  ]);

  return (
    <div className="flex flex-col h-full">
      {/* Header - fixed height */}
      <div className="p-4 border-b border-purple-800 bg-gradient-to-r from-purple-900/50 to-indigo-900/50 flex-shrink-0">
        <h2 className="text-xl font-semibold text-white">Chat</h2>
        {conversation && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => {
              cancelAllResponses();
              onEndConversation();
            }}
            className="flex items-center gap-1 mt-2"
          >
            <X size={16} />
            End Conversation
          </Button>
        )}
      </div>

      {/* Messages container - THIS IS THE KEY PART */}
      <div
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4"
        style={{
          height: "calc(100vh - 200px)", // Fixed height calculation
          maxHeight: "calc(100vh - 200px)", // Max height to ensure scrolling
        }}
      >
        {conversation && conversation.messages ? (
          conversation.messages.length > 0 ? (
            <div className="space-y-4">
              {conversation.messages.map((msg) => {
                // Skip rendering messages that contain SKIP_RESPONSE
                if (msg.content.includes("SKIP_RESPONSE")) {
                  return null;
                }

                // Handle system messages differently
                if (msg.metadata?.isSystemMessage) {
                  return (
                    <div key={msg.id} className="flex flex-col">
                      <div className="py-2 px-3 bg-purple-900/30 rounded-md text-purple-200 text-sm text-center">
                        {msg.content}
                      </div>
                    </div>
                  );
                }

                // Determine the sender name
                let senderName = "Unknown";
                let senderColor = "#888";

                if (msg.senderId === "user") {
                  senderName = "You";
                  senderColor = "#ffffff";
                } else if (msg.senderId === "system") {
                  senderName = "Environment";
                  senderColor = "#9333ea";
                } else {
                  const agent = getAgentById(msg.senderId);
                  if (agent) {
                    senderName = agent.name;
                    senderColor = agent.color;
                  }
                }

                const isBeingRespondedTo = processingMessageMap[msg.id];

                return (
                  <div key={msg.id} className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: senderColor }}
                      />
                      <span className="font-semibold text-purple-100">
                        {senderName}
                      </span>
                      <span className="text-xs text-purple-300">
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </span>
                      {msg.metadata?.isGeneratedByLLM && (
                        <span className="text-xs bg-purple-800/50 px-1.5 py-0.5 rounded text-purple-200">
                          AI
                        </span>
                      )}
                      {msg.metadata?.respondingTo && (
                        <div className="flex items-center text-xs text-purple-300">
                          <CornerDownRight size={12} className="mr-1" />
                          responding to:{" "}
                          {getMessageById(msg.metadata.respondingTo)
                            ? getMessageById(
                                msg.metadata.respondingTo,
                              )?.content.substring(0, 20) + "..."
                            : "a previous message"}
                        </div>
                      )}
                    </div>
                    <p className="ml-5 mt-1 text-gray-100">{msg.content}</p>

                    {isBeingRespondedTo && (
                      <div className="ml-5 mt-1 text-xs text-purple-300 flex items-center">
                        <Loader2 size={10} className="animate-spin mr-1" />
                        Agents are responding to this message...
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Typing indicators */}
              {Object.entries(typingAgents).map(
                ([agentId, { text, messageId }]) => {
                  const agent = getAgentById(agentId);
                  if (!agent) return null;

                  const respondingToMessage = getMessageById(messageId);

                  return (
                    <div
                      key={`typing-${agentId}`}
                      className="flex flex-col opacity-80"
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: agent.color }}
                        />
                        <span className="font-semibold">{agent.name}</span>
                        <Loader2
                          size={14}
                          className="animate-spin text-purple-300"
                        />
                        <span className="text-xs text-purple-300">
                          typing...
                        </span>

                        {respondingToMessage && (
                          <div className="flex items-center text-xs text-purple-300">
                            <CornerDownRight size={12} className="mx-1" />
                            re: {respondingToMessage.content.substring(0, 20)}
                            ...
                          </div>
                        )}
                      </div>
                      <p className="ml-5 mt-1 text-gray-100">{text || "..."}</p>
                    </div>
                  );
                },
              )}
            </div>
          ) : (
            <div className="text-center text-purple-300 py-8">
              No messages yet. Start the conversation by sending a message!
            </div>
          )
        ) : (
          <div className="text-center text-purple-300 py-8">
            No active conversation. Add agents to start one.
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded-md text-red-200 text-sm">
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle size={16} className="text-red-300" />
              <p className="font-semibold">Error:</p>
            </div>
            <p>{error}</p>
          </div>
        )}
      </div>

      {/* Input area - fixed height */}
      {conversation && (
        <div className="p-4 border-t border-purple-800 bg-black/20 flex-shrink-0">
          <div className="flex gap-2 mb-2 items-center">
            <select
              className="p-2 text-sm border border-purple-700 rounded-md bg-purple-950 text-white"
              value={userAgentId}
              onChange={(e) => setUserAgentId(e.target.value)}
              disabled={isProcessing || isSending}
            >
              <option value="user">You</option>
              {agents
                .filter(
                  (agent) =>
                    conversation.participants &&
                    conversation.participants.includes(agent.id),
                )
                .map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name}
                  </option>
                ))}
            </select>
            <Button
              variant="outline"
              size="sm"
              disabled={true}
              className="flex items-center gap-1 border-purple-700 bg-purple-950/30 text-purple-300 hover:bg-purple-900/50 hover:text-purple-200"
              title="Upload files (coming soon)"
            >
              <Upload size={14} />
              Upload
            </Button>
          </div>
          <div className="flex gap-2">
            <Input
              placeholder="Type your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={isProcessing || isSending}
              className="shadow-sm bg-purple-950/50 border-purple-700 text-white"
            />
            <Button
              onClick={handleSendMessage}
              disabled={isProcessing || isSending || !message.trim()}
              className="shadow-sm bg-purple-700 hover:bg-purple-600"
            >
              {isSending ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Send size={18} />
              )}
            </Button>
          </div>
          {(isProcessing || isSending) && (
            <div className="mt-2 text-xs text-purple-300 flex items-center">
              <Loader2 size={12} className="animate-spin mr-1" />
              {isSending
                ? "Sending message..."
                : `Agents are responding (${processingAgents.length + queuedAgents.length} in queue)...`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
