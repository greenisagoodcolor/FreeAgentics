"use client";

import { Card } from "@/components/ui/card";

interface ConversationViewProps {
  conversationId: string;
  isLive: boolean;
}

export function ConversationView({ conversationId, isLive }: ConversationViewProps) {
  return (
    <Card className="w-full h-full p-6">
      <h2 className="text-2xl font-semibold mb-4">
        Conversation {conversationId}
      </h2>
      <p className="text-muted-foreground">
        {isLive ? "Live" : "Recorded"} conversation interface coming soon...
      </p>
    </Card>
  );
}
