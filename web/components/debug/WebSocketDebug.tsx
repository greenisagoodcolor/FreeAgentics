"use client";

import { useEffect, useState } from "react";
import { getWebSocketUrl } from "@/utils/websocket-url";

export function WebSocketDebug() {
  const [debugInfo, setDebugInfo] = useState<{
    envUrl: string | undefined;
    backendUrl: string | undefined;
    constructedUrl: string;
    error?: string;
  } | null>(null);

  useEffect(() => {
    try {
      const envUrl = process.env.NEXT_PUBLIC_WS_URL;
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
      const constructedUrl = getWebSocketUrl('dev');
      
      setDebugInfo({
        envUrl,
        backendUrl,
        constructedUrl,
      });
    } catch (error) {
      setDebugInfo({
        envUrl: process.env.NEXT_PUBLIC_WS_URL,
        backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL,
        constructedUrl: 'ERROR',
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, []);

  if (!debugInfo) {
    return <div>Loading debug info...</div>;
  }

  return (
    <div style={{ 
      position: 'fixed', 
      top: '10px', 
      right: '10px', 
      background: 'rgba(0,0,0,0.8)', 
      color: 'white', 
      padding: '10px',
      borderRadius: '4px',
      fontSize: '12px',
      fontFamily: 'monospace',
      zIndex: 9999,
      maxWidth: '300px'
    }}>
      <div><strong>WebSocket Debug:</strong></div>
      <div>ENV_WS_URL: {debugInfo.envUrl || 'undefined'}</div>
      <div>ENV_BACKEND_URL: {debugInfo.backendUrl || 'undefined'}</div>
      <div>CONSTRUCTED_URL: {debugInfo.constructedUrl}</div>
      {debugInfo.error && <div style={{ color: 'red' }}>ERROR: {debugInfo.error}</div>}
    </div>
  );
}