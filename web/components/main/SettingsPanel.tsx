import React from "react";

export function SettingsPanel() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">API Configuration</h3>
        <p className="text-sm text-muted-foreground">
          Configure your API settings and authentication
        </p>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium">API Key</label>
          <input
            type="password"
            placeholder="Enter your API key"
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
        
        <div>
          <label className="text-sm font-medium">API URL</label>
          <input
            type="text"
            defaultValue="http://localhost:8000"
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
      </div>
      
      <div className="pt-6">
        <button className="px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90">
          Save Settings
        </button>
      </div>
    </div>
  );
}