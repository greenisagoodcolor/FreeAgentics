"use client"

import type React from "react"
import { createContext, useContext, useState } from "react"

interface IsSendingContextType {
  isSending: boolean
  setIsSending: (isSending: boolean) => void
}

// Create a default implementation that won't throw errors
const defaultSetIsSending = () => {
  console.log("Default setIsSending called - context not yet initialized")
}

// Initialize with safe defaults
const IsSendingContext = createContext<IsSendingContextType>({
  isSending: false,
  setIsSending: defaultSetIsSending,
})

export function IsSendingProvider({ children }: { children: React.ReactNode }) {
  console.log("IsSendingProvider rendering")
  // Initialize state here to avoid hydration issues
  const [isSending, setIsSending] = useState<boolean>(false)

  // Create the context value object only once per render
  const contextValue = {
    isSending,
    setIsSending: (value: boolean) => {
      console.log(`setIsSending called with value: ${value}`)
      try {
        setIsSending(value)
      } catch (error) {
        console.error("Error in setIsSending:", error)
      }
    },
  }

  console.log("IsSendingProvider created context value:", { isSending })
  return <IsSendingContext.Provider value={contextValue}>{children}</IsSendingContext.Provider>
}

export function useIsSending() {
  console.log("useIsSending hook called")
  // Add safety check for SSR/hydration
  const context = useContext(IsSendingContext)

  console.log("useIsSending context retrieved:", {
    isSending: context?.isSending,
    setIsSendingType: typeof context?.setIsSending,
  })

  // Ensure we never return undefined functions
  if (typeof context?.setIsSending !== "function") {
    console.error("useIsSending: setIsSending is not a function!")
    return {
      isSending: context?.isSending || false,
      setIsSending: defaultSetIsSending,
    }
  }

  return context
}
