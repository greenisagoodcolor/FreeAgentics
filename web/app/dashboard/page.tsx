"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface SystemMetrics {
  activeAgents: number;
  totalInferences: number;
  avgResponseTime: number;
  memoryUsage: number;
}

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    activeAgents: 0,
    totalInferences: 0,
    avgResponseTime: 0,
    memoryUsage: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate fetching metrics
    setTimeout(() => {
      setMetrics({
        activeAgents: 0,
        totalInferences: 42,
        avgResponseTime: 120,
        memoryUsage: 45.2,
      });
      setLoading(false);
    }, 1000);
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Agent Dashboard</h1>
        <Link href="/" className="text-blue-600 hover:text-blue-800 underline">
          ← Back to Home
        </Link>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Active Agents"
          value={metrics.activeAgents}
          unit=""
          loading={loading}
          color="blue"
        />
        <MetricCard
          title="Total Inferences"
          value={metrics.totalInferences}
          unit=""
          loading={loading}
          color="green"
        />
        <MetricCard
          title="Avg Response Time"
          value={metrics.avgResponseTime}
          unit="ms"
          loading={loading}
          color="yellow"
        />
        <MetricCard
          title="Memory Usage"
          value={metrics.memoryUsage}
          unit="%"
          loading={loading}
          color="purple"
        />
      </div>

      {/* Agent List Placeholder */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Active Agents</h2>
        <div className="text-center py-12 text-gray-500">
          <p className="mb-4">No agents are currently active</p>
          <Link
            href="/agents"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
          >
            Create First Agent
          </Link>
        </div>
      </div>

      {/* Development Notice */}
      <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-800">
          <strong>Note:</strong> The dashboard is currently showing demo data.
          Agent functionality is being implemented.
        </p>
      </div>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: number;
  unit: string;
  loading: boolean;
  color: "blue" | "green" | "yellow" | "purple";
}

function MetricCard({ title, value, unit, loading, color }: MetricCardProps) {
  const colorClasses = {
    blue: "bg-blue-50 text-blue-700",
    green: "bg-green-50 text-green-700",
    yellow: "bg-yellow-50 text-yellow-700",
    purple: "bg-purple-50 text-purple-700",
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-sm font-medium text-gray-500 mb-2">{title}</h3>
      {loading ? (
        <div className="h-8 bg-gray-200 rounded animate-pulse"></div>
      ) : (
        <div className="flex items-baseline">
          <span className="text-2xl font-bold text-gray-900">{value}</span>
          {unit && <span className="ml-1 text-sm text-gray-500">{unit}</span>}
        </div>
      )}
      <div
        className={`mt-2 inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${colorClasses[color]}`}
      >
        {loading ? "Loading..." : "Live"}
      </div>
    </div>
  );
}
