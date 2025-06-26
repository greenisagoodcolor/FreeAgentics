export type { ProviderMonitoringData, UsageMetrics, HealthStatus } from '@/types/llm-providers';

export function ProviderMonitoringDashboard() {
  return null;
}

export interface ProviderMetrics {
  callCount: number;
  errorCount: number;
  avgLatency: number;
  cost: number;
}

export interface ProviderHealth {
  status: 'healthy' | 'degraded' | 'down';
  lastCheck: Date;
  uptime: number;
}