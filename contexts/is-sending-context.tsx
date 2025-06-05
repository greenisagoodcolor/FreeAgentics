"use client"

import React, { createContext, useContext, useState } from "react"
import { createLogger } from "@/lib/debug-logger"

const logger = createLogger("IS-SENDING-CONTEXT")

interface IsSendingContextType {
  isSending: boolean
  setIsSending: (value: boolean) => void
}

// Create default setIsSending function
const defaultSetIsSending = (value: boolean) => {
  logger.warn("Default setIsSending called - context not yet initialized")
}

// Initialize with safe defaults
const IsSendingContext = createContext<IsSendingContextType>({
  isSending: false,
  setIsSending: defaultSetIsSending,
})

export function IsSendingProvider({ children }: { children: React.ReactNode }) {
  logger.info("IsSendingProvider rendering")
  // Initialize state here to avoid hydration issues
  const [isSending, setIsSending] = useState<boolean>(false)

  // Create the context value object only once per render
  const contextValue = {
    isSending,
    setIsSending: (value: boolean) => {
      logger.info(`setIsSending called with value: ${value}`)
      try {
        setIsSending(value)
      } catch (error) {
        logger.error("Error in setIsSending:", error)
      }
    },
  }

  logger.info("IsSendingProvider created context value:", { isSending })
  return <IsSendingContext.Provider value={contextValue}>{children}</IsSendingContext.Provider>
}

export function useIsSending() {
  logger.info("useIsSending hook called")
  // Add safety check for SSR/hydration
  const context = useContext(IsSendingContext)

  logger.info("useIsSending context retrieved:", {
    isSending: context?.isSending,
    setIsSendingType: typeof context?.setIsSending,
  })

  // Ensure we never return undefined functions
  if (typeof context?.setIsSending !== "function") {
    logger.error("useIsSending: setIsSending is not a function!")
    return {
      isSending: context?.isSending || false,
      setIsSending: defaultSetIsSending,
    }
  }

  return context
}
