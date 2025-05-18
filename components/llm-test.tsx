"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { useLLM } from "@/contexts/llm-context"

export default function LLMTest() {
  const { client, isProcessing, setIsProcessing, settings } = useLLM()
  const [prompt, setPrompt] = useState<string>("Explain quantum computing in simple terms.")
  const [response, setResponse] = useState<string>("")
  const [debugInfo, setDebugInfo] = useState<string>("")
  const [sessionDebug, setSessionDebug] = useState<string>("")

  // Add debug info on mount and when settings change
  useEffect(() => {
    const checkSessionId = async () => {
      try {
        // Get session ID from localStorage
        const localSessionId = localStorage.getItem(`api_session_${settings.provider}`)

        // Debug session info
        setSessionDebug(`
Provider: ${settings.provider}
Settings has apiKeySessionId: ${!!settings.apiKeySessionId}
Settings apiKeySessionId: ${settings.apiKeySessionId || "undefined"}
Local storage sessionId: ${localSessionId || "undefined"}
Settings keys: ${Object.keys(settings).join(", ")}
`)
      } catch (error) {
        console.error("Error checking session ID:", error)
        setSessionDebug(`Error checking session ID: ${error instanceof Error ? error.message : "Unknown error"}`)
      }
    }

    checkSessionId()
  }, [settings])

  const handleTest = async () => {
    if (!prompt.trim() || isProcessing) return

    try {
      setIsProcessing(true)
      setResponse("")
      setDebugInfo("")

      // Log detailed debug info about settings and session
      console.log("Test settings debug:", {
        provider: settings.provider,
        hasApiKeySessionId: !!settings.apiKeySessionId,
        apiKeySessionId: settings.apiKeySessionId,
        settingsKeys: Object.keys(settings),
        localStorageSessionId: localStorage.getItem(`api_session_${settings.provider}`),
      })

      // Check if API key session ID is available
      const sessionId = settings.apiKeySessionId || localStorage.getItem(`api_session_${settings.provider}`)

      if (!sessionId) {
        throw new Error(
          `API key is required for ${settings.provider} provider. Please add your API key in the Settings tab.`,
        )
      }

      // If we have a session ID in localStorage but not in settings, update the settings
      if (!settings.apiKeySessionId && sessionId) {
        console.log("Found session ID in localStorage but not in settings, updating settings")
        client.updateSettings({
          ...settings,
          apiKeySessionId: sessionId,
        })
      }

      // Add debug info
      setDebugInfo(`Provider: ${settings.provider}
Model: ${settings.model}
API key: [Securely Stored]
API key session ID: ${sessionId}
Temperature: ${settings.temperature}
Max tokens: ${settings.maxTokens}
Top P: ${settings.topP}
Frequency penalty: ${settings.frequencyPenalty}
Presence penalty: ${settings.presencePenalty}
System fingerprint: ${settings.systemFingerprint}`)

      // Force update the client settings before making the call
      client.updateSettings({
        ...settings,
        apiKeySessionId: sessionId,
      })

      // Get the current settings directly from the client to ensure we're using the latest
      const currentSettings = client.getSettings()
      console.log("Current settings before test:", {
        provider: currentSettings.provider,
        model: currentSettings.model,
        hasApiKeySessionId: !!currentSettings.apiKeySessionId,
        apiKeySessionId: currentSettings.apiKeySessionId,
      })

      try {
        const result = await client.generateResponse(
          "You are a helpful AI assistant that explains complex topics in simple terms.",
          prompt,
        )
        setResponse(result)
      } catch (error) {
        console.error("Error in client.generateResponse:", error)
        setResponse(`Error: ${error instanceof Error ? error.message : "Unknown error"}`)
      }
    } catch (error) {
      console.error("Error testing LLM:", error)
      setResponse(`Error: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>LLM Integration Test</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {sessionDebug && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Session Debug:</h3>
              <div className="p-4 bg-gray-800 rounded-md whitespace-pre-wrap text-xs font-mono text-white">
                {sessionDebug}
              </div>
            </div>
          )}

          <div className="space-y-2">
            <label htmlFor="prompt" className="text-sm font-medium">
              Prompt
            </label>
            <Textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a prompt to test the LLM integration..."
              className="min-h-[100px]"
            />
          </div>

          <Button onClick={handleTest} disabled={isProcessing || !prompt.trim()} className="w-full">
            {isProcessing ? (
              <>
                <Spinner size={16} className="mr-2" />
                Processing...
              </>
            ) : (
              "Test LLM Integration"
            )}
          </Button>

          {debugInfo && (
            <div className="space-y-2 mt-4">
              <h3 className="text-sm font-medium">Debug Info:</h3>
              <div className="p-4 bg-gray-800 rounded-md whitespace-pre-wrap text-xs font-mono text-white">
                {debugInfo}
              </div>
            </div>
          )}

          {response && (
            <div className="space-y-2 mt-4">
              <h3 className="text-sm font-medium">Response:</h3>
              <div className="p-4 bg-muted rounded-md whitespace-pre-wrap">{response}</div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
