// lib/debug-logger.ts
const DEBUG = true; // Or use an environment variable

export const createLogger = (namespace: string) => {
  return {
    log: (message: string, ...args: any[]) => {
      if (DEBUG) {
        console.log(`[${namespace}] ${message}`, ...args);
      }
    },
    info: (message: string, ...args: any[]) => {
      if (DEBUG) {
        console.info(`[${namespace}] ${message}`, ...args);
      }
    },
    warn: (message: string, ...args: any[]) => {
      if (DEBUG) {
        console.warn(`[${namespace}] ${message}`, ...args);
      }
    },
    error: (message: string, ...args: any[]) => {
      if (DEBUG) {
        console.error(`[${namespace}] ${message}`, ...args);
      }
    },
    debug: (message: string, ...args: any[]) => {
      if (DEBUG) {
        console.debug(`[${namespace}] ${message}`, ...args);
      }
    },
  };
};

export const debugLog = (message: string, ...args: any[]) => {
  if (DEBUG) {
    console.log(`[DEBUG] ${message}`, ...args);
  }
};
