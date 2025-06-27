"use client";

import React from "react";
import { DashboardView } from "../../../page";
import { MessageSquare } from "lucide-react";

interface ConversationPanelProps {
  view: DashboardView;
}

export default function ConversationPanel({ view }: ConversationPanelProps) {
  return (
    <div className="h-full bg-[var(--bg-primary)] p-4 flex items-center justify-center">
      <div className="text-center">
        <MessageSquare className="w-12 h-12 text-[var(--text-tertiary)] mx-auto mb-3" />
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">
          Conversation Panel
        </h3>
        <p className="text-[var(--text-secondary)] text-sm">
          Real-time conversation interface coming soon
        </p>
      </div>
    </div>
  );
}
