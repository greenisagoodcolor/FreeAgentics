"use client"

import { useState, useEffect } from "react"
import { type LLMClient, llmClient } from "./llm-client"

export function useLLM() {
  const [client, setClient] = useState<LLMClient>(llmClient)

  useEffect(() => {
    // Initialize the client with the initial settings
    setClient(llmClient)
  }, [])

  return { client }
}
