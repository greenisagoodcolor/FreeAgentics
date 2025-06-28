"use client";

import React, { useEffect, useState } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";

interface WebSocketProviderProps {
  children: React.ReactNode;
  autoConnect?: boolean;
  showConnectionStatus?: boolean;
}

export function WebSocketProvider({ 
  children, 
  autoConnect = true, 
  showConnectionStatus = false 
}: WebSocketProviderProps) {
  const [mounted, setMounted] = useState(false);
  
  const websocket = useWebSocket({
    autoConnect,
    onConnect: () => {
      console.log("🟢 WebSocket connected successfully");
    },
    onDisconnect: () => {
      console.log("🔴 WebSocket disconnected");
    },
    onError: (error) => {
      console.error("🟠 WebSocket error:", error);
    },
  });

  // Handle client-side mounting
  useEffect(() => {
    setMounted(true);
  }, []);

  // Don't render on server side
  if (!mounted) {
    return <>{children}</>;
  }

  return (
    <>
      {showConnectionStatus && (
        <div className="fixed top-4 right-4 z-50">
          <ConnectionStatusIndicator
            isConnected={websocket.isConnected}
            isConnecting={websocket.isConnecting}
            error={websocket.error}
            latency={websocket.latency}
            reconnectAttempts={websocket.reconnectAttempts}
          />
        </div>
      )}
      {children}
    </>
  );
}

interface ConnectionStatusIndicatorProps {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  latency: number | null;
  reconnectAttempts: number;
}

function ConnectionStatusIndicator({
  isConnected,
  isConnecting,
  error,
  latency,
  reconnectAttempts,
}: ConnectionStatusIndicatorProps) {
  const getStatusColor = () => {
    if (isConnected) return "bg-green-500";
    if (isConnecting) return "bg-yellow-500";
    if (error) return "bg-red-500";
    return "bg-gray-500";
  };

  const getStatusText = () => {
    if (isConnected) return `Connected ${latency ? `(${latency}ms)` : ""}`;
    if (isConnecting) return "Connecting...";
    if (error) return `Error: ${error}`;
    return "Disconnected";
  };

  return (
    <div className="flex items-center space-x-2 px-3 py-2 bg-gray-900 rounded-lg border border-gray-700 text-sm">
      <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
      <span className="text-gray-200">{getStatusText()}</span>
      {reconnectAttempts > 0 && (
        <span className="text-yellow-400 text-xs">
          (Retry {reconnectAttempts})
        </span>
      )}
    </div>
  );
}

export default WebSocketProvider; 