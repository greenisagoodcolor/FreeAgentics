export interface LLMProvider {
  id: string;
  name: string;
  type: "openai" | "anthropic" | "openrouter";
  enabled: boolean;
  priority: number;
  apiKey?: string;
  endpoint?: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  status: {
    isHealthy: boolean;
    lastHealthCheck: Date;
    responseTimeMs: number;
    errorCount: number;
    errorRate?: number;
  };
  usage: {
    totalRequests: number;
    totalTokens?: number;
    totalCost: number;
    lastUsed?: Date;
  };
  rateLimits?: {
    requestsPerMinute?: number;
    tokensPerMinute?: number;
    concurrentRequests?: number;
  };
  customHeaders?: Record<string, string>;
  timeout?: number;
  retryConfig?: {
    maxRetries: number;
    retryDelay: number;
    backoffMultiplier?: number;
  };
}

export interface FailoverRule {
  id: string;
  name: string;
  enabled: boolean;
  priority: number;
  condition: {
    type: "error" | "latency" | "rate_limit" | "cost" | "availability";
    threshold?: number;
    errorCodes?: string[];
    timeWindowMs?: number;
  };
  action: {
    type: "failover" | "retry" | "circuit_break" | "rate_limit";
    targetProviderId?: string;
    retryCount?: number;
    cooldownMs?: number;
    maxDuration?: number;
  };
  metadata?: {
    description?: string;
    createdAt: Date;
    updatedAt: Date;
    triggeredCount: number;
    lastTriggered?: Date;
  };
}

export interface CredentialFormData {
  [key: string]: string;
}

export interface EncryptionResult {
  encryptedData: string;
  keyId: string;
  algorithm: string;
  timestamp: number;
}

export interface ProviderMonitoringData {
  providerId: string;
  metrics: UsageMetrics;
  health: HealthStatus;
  timestamp: Date;
}

export interface UsageMetrics {
  requestCount: number;
  tokenCount: number;
  costEstimate: number;
  averageLatency: number;
  errorRate: number;
  successRate: number;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy" | "unknown";
  lastCheck: Date;
  uptime: number;
  incidents: HealthIncident[];
}

export interface HealthIncident {
  id: string;
  type: "outage" | "degradation" | "rate_limit" | "error";
  severity: "low" | "medium" | "high" | "critical";
  startTime: Date;
  endTime?: Date;
  duration?: number;
  impact: string;
  resolution?: string;
}

export class CredentialCrypto {
  static async encryptCredentials(credentials: CredentialFormData): Promise<EncryptionResult> {
    // This would be implemented with actual encryption logic
    const timestamp = Date.now();
    return {
      encryptedData: btoa(JSON.stringify(credentials)),
      keyId: `key_${timestamp}`,
      algorithm: "AES-256-GCM",
      timestamp
    };
  }

  static async decryptCredentials(encrypted: EncryptionResult): Promise<CredentialFormData> {
    // This would be implemented with actual decryption logic
    return JSON.parse(atob(encrypted.encryptedData));
  }

  static async initializeSecureSession(): Promise<void> {
    // Initialize encryption session
    return Promise.resolve();
  }
}