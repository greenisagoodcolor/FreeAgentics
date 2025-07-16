"use client";

import { useState } from "react";
import Link from "next/link";

export default function HomePage() {
  const [systemStatus, setSystemStatus] = useState<"checking" | "online" | "offline">("checking");

  // Check backend status on mount
  useState(() => {
    fetch("/api/health")
      .then((res) => (res.ok ? setSystemStatus("online") : setSystemStatus("offline")))
      .catch(() => setSystemStatus("offline"));
  });

  return (
    <main className="container mx-auto px-4 py-16">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">FreeAgentics</h1>
          <p className="text-xl text-gray-600 mb-2">Active Inference Multi-Agent Platform</p>
          <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800">
            v0.1-alpha - Under Development
          </div>
        </div>

        {/* System Status */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">System Status</h2>
          <div className="flex items-center justify-between">
            <span className="text-gray-700">Backend API</span>
            <span
              className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                systemStatus === "online"
                  ? "bg-green-100 text-green-800"
                  : systemStatus === "offline"
                    ? "bg-red-100 text-red-800"
                    : "bg-gray-100 text-gray-800"
              }`}
            >
              {systemStatus === "checking" ? "Checking..." : systemStatus}
            </span>
          </div>
        </div>

        {/* Quick Links */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          <Link href="/dashboard" className="block">
            <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
              <h3 className="text-xl font-semibold mb-2">Agent Dashboard</h3>
              <p className="text-gray-600">Monitor and control active agents in the system</p>
            </div>
          </Link>

          <Link href="/agents" className="block">
            <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
              <h3 className="text-xl font-semibold mb-2">Agent Explorer</h3>
              <p className="text-gray-600">Create and configure new active inference agents</p>
            </div>
          </Link>
        </div>

        {/* Development Notice */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-yellow-900 mb-2">🚧 Development Notice</h3>
          <p className="text-yellow-800 mb-4">
            This is an early alpha release. Core features are still being implemented:
          </p>
          <ul className="list-disc list-inside text-yellow-700 space-y-1">
            <li>Active Inference engine integration (15% complete)</li>
            <li>Multi-agent coalition formation (0% complete)</li>
            <li>Graph Neural Network inference (40% complete)</li>
            <li>Spatial world simulation (0% complete)</li>
          </ul>
          <div className="mt-4">
            <a
              href="https://github.com/FreeAgentics/freeagentics"
              className="text-blue-600 hover:text-blue-800 underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              View progress on GitHub →
            </a>
          </div>
        </div>

        {/* Technical Stack */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>Built with PyTorch • FastAPI • Next.js • Active Inference (PyMDP)</p>
        </div>
      </div>
    </main>
  );
}
