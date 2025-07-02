"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useAppDispatch } from "@/store/hooks";
import { setDemoAgent, clearAgents } from "@/store/slices/agentSlice";
import {
  setDemoConversation,
  clearConversations,
} from "@/store/slices/conversationSlice";
import {
  setDemoKnowledgeGraph,
  clearKnowledgeGraph,
} from "@/store/slices/knowledgeSlice";
import { demoDataService } from "@/lib/demo-data-service";

interface DemoDataContextType {
  isDemoMode: boolean;
  toggleDemoMode: () => void;
  refreshDemoData: () => void;
  analytics: any;
}

const DemoDataContext = createContext<DemoDataContextType>({
  isDemoMode: false,
  toggleDemoMode: () => {},
  refreshDemoData: () => {},
  analytics: {},
});

export const useDemoData = () => useContext(DemoDataContext);

interface DemoDataProviderProps {
  children: React.ReactNode;
}

export function DemoDataProvider({ children }: DemoDataProviderProps) {
  const [isDemoMode, setIsDemoMode] = useState(true); // Start in demo mode by default
  const [analytics, setAnalytics] = useState({});
  const dispatch = useAppDispatch();

  const loadDemoDataToStore = () => {
    console.log("Loading demo data to Redux store...");

    // Load agents with demo data
    const agents = demoDataService.getAgents();
    agents.forEach((agent) => {
      dispatch(setDemoAgent(agent));
    });

    // Load conversations with demo data
    const conversations = demoDataService.getConversations();
    conversations.forEach((conversation) => {
      dispatch(setDemoConversation(conversation));
    });

    // Load knowledge graph with demo data
    const nodes = demoDataService.getKnowledgeNodes();
    const edges = demoDataService.getKnowledgeEdges();

    dispatch(setDemoKnowledgeGraph({ nodes, edges }));

    // Load analytics
    setAnalytics(demoDataService.getAnalytics());
  };

  const clearDemoData = () => {
    console.log("Clearing demo data from Redux store...");
    dispatch(clearAgents());
    dispatch(clearConversations());
    dispatch(clearKnowledgeGraph());
    setAnalytics({});
  };

  const toggleDemoMode = () => {
    const newMode = !isDemoMode;
    setIsDemoMode(newMode);

    if (newMode) {
      loadDemoDataToStore();
      demoDataService.startSimulation();
    } else {
      clearDemoData();
      demoDataService.stopSimulation();
    }
  };

  const refreshDemoData = () => {
    if (isDemoMode) {
      loadDemoDataToStore();
      setAnalytics(demoDataService.getAnalytics());
    }
  };

  useEffect(() => {
    if (isDemoMode) {
      // Load initial demo data
      loadDemoDataToStore();

      // Start simulation
      demoDataService.startSimulation();

      // Set up real-time updates
      const unsubscribe = demoDataService.onUpdate(() => {
        // Update analytics in real-time
        setAnalytics(demoDataService.getAnalytics());

        // Reload all demo data to Redux to pick up simulation changes
        loadDemoDataToStore();
      });

      return () => {
        unsubscribe();
        demoDataService.stopSimulation();
      };
    }
  }, [isDemoMode, dispatch]);

  // Update analytics every 5 seconds when in demo mode
  useEffect(() => {
    if (!isDemoMode) return;

    const interval = setInterval(() => {
      setAnalytics(demoDataService.getAnalytics());
    }, 5000);

    return () => clearInterval(interval);
  }, [isDemoMode]);

  const contextValue: DemoDataContextType = {
    isDemoMode,
    toggleDemoMode,
    refreshDemoData,
    analytics,
  };

  return (
    <DemoDataContext.Provider value={contextValue}>
      {children}
    </DemoDataContext.Provider>
  );
}
