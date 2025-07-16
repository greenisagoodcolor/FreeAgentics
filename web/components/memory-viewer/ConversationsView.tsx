import React, { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Users, Clock } from "lucide-react";
import { formatTimestamp } from "@/lib/utils";
import { getAgentInitials } from "@/lib/memory-viewer-utils";
import type { Agent, Conversation } from "@/lib/types";

interface ConversationsViewProps {
  selectedAgent: Agent | null;
  conversationHistory: Conversation[];
  agents: Agent[];
}

export function ConversationsView({
  selectedAgent,
  conversationHistory,
  agents,
}: ConversationsViewProps) {
  // Create agent lookup map
  const agentMap = useMemo(
    () =>
      agents.reduce(
        (acc, agent) => {
          acc[agent.id] = agent;
          return acc;
        },
        {} as Record<string, Agent>,
      ),
    [agents],
  );

  // Filter conversations involving the selected agent
  const relevantConversations = useMemo(() => {
    if (!selectedAgent) return conversationHistory;

    return conversationHistory.filter((conv) => conv.participants.includes(selectedAgent.id));
  }, [conversationHistory, selectedAgent]);

  // Sort conversations by most recent
  const sortedConversations = useMemo(() => {
    return [...relevantConversations].sort((a, b) => {
      const aTime = a.endTime || a.startTime;
      const bTime = b.endTime || b.startTime;
      return new Date(bTime).getTime() - new Date(aTime).getTime();
    });
  }, [relevantConversations]);

  if (!selectedAgent) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent>
          <p className="text-center text-muted-foreground">
            Select an agent to view their conversations
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="h-full space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Conversation History
        </h3>
        <Badge variant="outline">{sortedConversations.length} conversations</Badge>
      </div>

      <ScrollArea className="h-[500px] pr-4">
        <div className="space-y-4">
          {sortedConversations.map((conversation) => (
            <Card key={conversation.id}>
              <CardContent className="p-4 space-y-4">
                {/* Conversation header */}
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium">
                        {conversation.participants.length} participants
                      </span>
                      {conversation.isAutonomous && (
                        <Badge variant="secondary" className="text-xs">
                          Autonomous
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>
                        {formatTimestamp(conversation.startTime)}
                        {conversation.endTime && ` - ${formatTimestamp(conversation.endTime)}`}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Conversation metadata */}
                <div className="flex flex-wrap gap-2">
                  {conversation.topic && (
                    <Badge variant="outline" className="text-xs">
                      Topic: {conversation.topic}
                    </Badge>
                  )}
                  {conversation.trigger && (
                    <Badge variant="outline" className="text-xs">
                      Trigger: {conversation.trigger}
                    </Badge>
                  )}
                  <Badge variant="outline" className="text-xs">
                    {conversation.messages.length} messages
                  </Badge>
                </div>

                {/* Participants */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Participants:</span>
                  <div className="flex -space-x-2">
                    {conversation.participants.map((participantId) => {
                      const participant = agentMap[participantId];
                      return (
                        <Avatar key={participantId} className="h-6 w-6 border-2 border-background">
                          <AvatarImage src={participant?.avatar || participant?.avatarUrl} />
                          <AvatarFallback className="text-xs">
                            {participant ? getAgentInitials(participant.name) : "?"}
                          </AvatarFallback>
                        </Avatar>
                      );
                    })}
                  </div>
                </div>

                {/* Messages preview */}
                <div className="space-y-2 max-h-[200px] overflow-y-auto">
                  {conversation.messages.slice(0, 5).map((message) => {
                    const agent = agentMap[message.senderId];
                    const isOwnMessage = message.senderId === selectedAgent?.id;

                    return (
                      <div
                        key={message.id}
                        className={`flex items-start gap-2 ${
                          isOwnMessage ? "flex-row-reverse" : ""
                        }`}
                      >
                        <Avatar className="h-8 w-8">
                          <AvatarImage src={agent?.avatar || agent?.avatarUrl} />
                          <AvatarFallback className="text-xs">
                            {agent ? getAgentInitials(agent.name) : "?"}
                          </AvatarFallback>
                        </Avatar>
                        <div className={`flex-1 space-y-1 ${isOwnMessage ? "text-right" : ""}`}>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{agent?.name || "Unknown"}</span>
                            <span className="text-xs text-muted-foreground">
                              {formatTimestamp(message.timestamp)}
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground">{message.content}</p>
                        </div>
                      </div>
                    );
                  })}

                  {conversation.messages.length > 5 && (
                    <p className="text-xs text-center text-muted-foreground">
                      ... and {conversation.messages.length - 5} more messages
                    </p>
                  )}
                </div>

                {/* Conversation metrics */}
                {conversation.conversationMetrics && (
                  <div className="flex flex-wrap gap-4 pt-2 border-t text-xs text-muted-foreground">
                    <span>
                      Avg response time:{" "}
                      {Math.round(conversation.conversationMetrics.averageResponseTime)}
                      ms
                    </span>
                    {conversation.conversationMetrics.engagementLevel !== undefined && (
                      <span>
                        Engagement:{" "}
                        {Math.round(conversation.conversationMetrics.engagementLevel * 100)}%
                      </span>
                    )}
                    {conversation.conversationMetrics.topicDrift !== undefined && (
                      <span>
                        Topic drift: {Math.round(conversation.conversationMetrics.topicDrift * 100)}
                        %
                      </span>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}

          {sortedConversations.length === 0 && (
            <Card>
              <CardContent className="p-8 text-center text-muted-foreground">
                No conversations found for this agent
              </CardContent>
            </Card>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
