/**
 * Utility for structured logging of conversation events
 */
export const ConversationLogger = {
  init: (conversationId: string) => {
    console.log(`[CONV:${conversationId}] Initializing conversation logger`);
    return {
      log: (stage: string, message: string, data?: any) => {
        console.log(
          `[CONV:${conversationId}][${stage}] ${message}`,
          data || "",
        );
      },
      error: (stage: string, message: string, error?: any) => {
        console.error(
          `[CONV:${conversationId}][${stage}] ERROR: ${message}`,
          error || "",
        );
      },
      warn: (stage: string, message: string, data?: any) => {
        console.warn(
          `[CONV:${conversationId}][${stage}] WARNING: ${message}`,
          data || "",
        );
      },
      debug: (stage: string, message: string, data?: any) => {
        console.debug(
          `[CONV:${conversationId}][${stage}] ${message}`,
          data || "",
        );
      },
    };
  },

  // Static methods for logging without a specific conversation context
  system: {
    log: (component: string, message: string, data?: any) => {
      console.log(`[SYSTEM:${component}] ${message}`, data || "");
    },
    error: (component: string, message: string, error?: any) => {
      console.error(`[SYSTEM:${component}] ERROR: ${message}`, error || "");
    },
    warn: (component: string, message: string, data?: any) => {
      console.warn(`[SYSTEM:${component}] WARNING: ${message}`, data || "");
    },
    debug: (component: string, message: string, data?: any) => {
      console.debug(`[SYSTEM:${component}] ${message}`, data || "");
    },
  },

  // Message-specific logging
  message: (messageId: string) => {
    return {
      log: (stage: string, message: string, data?: any) => {
        console.log(`[MSG:${messageId}][${stage}] ${message}`, data || "");
      },
      error: (stage: string, message: string, error?: any) => {
        console.error(
          `[MSG:${messageId}][${stage}] ERROR: ${message}`,
          error || "",
        );
      },
      warn: (stage: string, message: string, data?: any) => {
        console.warn(
          `[MSG:${messageId}][${stage}] WARNING: ${message}`,
          data || "",
        );
      },
    };
  },

  // Agent-specific logging
  agent: (agentId: string) => {
    return {
      log: (stage: string, message: string, data?: any) => {
        console.log(`[AGENT:${agentId}][${stage}] ${message}`, data || "");
      },
      error: (stage: string, message: string, error?: any) => {
        console.error(
          `[AGENT:${agentId}][${stage}] ERROR: ${message}`,
          error || "",
        );
      },
      warn: (stage: string, message: string, data?: any) => {
        console.warn(
          `[AGENT:${agentId}][${stage}] WARNING: ${message}`,
          data || "",
        );
      },
    };
  },
};
