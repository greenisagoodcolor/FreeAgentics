"use client";

import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

export default function SettingsModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [key, setKey] = useState("");

  useEffect(() => {
    if (typeof window !== "undefined") {
      setKey(localStorage.getItem("OPENAI_API_KEY") || "");
    }
  }, []);

  const handleSave = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("OPENAI_API_KEY", key);
      setIsOpen(false);
      // Optional: Show toast or confirmation
    }
  };

  if (!isOpen) {
    return (
      <Button
        variant="secondary"
        onClick={() => setIsOpen(true)}
        className="fixed top-4 right-4 z-50"
        title="Open Settings to Configure OpenAI API Key"
      >
        ⚙️ Settings
      </Button>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg border-2 border-gray-300 shadow-lg p-8 w-full max-w-lg mx-4">
        <h2 className="text-2xl font-bold mb-6 text-gray-900">Development Settings</h2>
        
        <div className="space-y-6">
          <div>
            <label htmlFor="openai" className="block text-lg font-semibold mb-3 text-gray-900">
              OpenAI API Key (Optional)
            </label>
            <p className="text-sm text-gray-600 mb-3">
              Paste your OpenAI API key to use real AI responses instead of mock data
            </p>
            <Input
              id="openai"
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="sk-proj-... (your OpenAI API key)"
              className="w-full"
            />
          </div>
        </div>

        <div className="flex gap-4 mt-8">
          <Button onClick={handleSave} className="flex-1">
            💾 Save API Key
          </Button>
          <Button
            variant="outline"
            onClick={() => setIsOpen(false)}
            className="flex-1"
          >
            ❌ Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}