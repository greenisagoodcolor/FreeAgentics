"use client";

import React from "react";
import { motion } from "framer-motion";
import { Play, Clock, Users, TrendingUp, ArrowRight } from "lucide-react";

interface CEODemoLandingProps {
  onStartDemo: () => void;
}

export default function CEODemoLanding({ onStartDemo }: CEODemoLandingProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white flex items-center justify-center p-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto text-center"
      >
        {/* Header */}
        <motion.div
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-12"
        >
          <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mb-6 mx-auto">
            <Play className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            FreeAgentics CEO Demo
          </h1>
          <p className="text-xl text-gray-300 mb-8 leading-relaxed">
            Experience how autonomous AI agents transform business operations in
            just 3 minutes
          </p>
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-3 gap-8 mb-12"
        >
          <div className="text-center">
            <Clock className="w-8 h-8 text-blue-400 mx-auto mb-3" />
            <h3 className="font-semibold mb-2">3 Minute Demo</h3>
            <p className="text-sm text-gray-400">
              Quick, focused presentation designed for busy executives
            </p>
          </div>
          <div className="text-center">
            <TrendingUp className="w-8 h-8 text-green-400 mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Real ROI Metrics</h3>
            <p className="text-sm text-gray-400">
              See actual cost savings and productivity gains
            </p>
          </div>
          <div className="text-center">
            <Users className="w-8 h-8 text-purple-400 mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Live Collaboration</h3>
            <p className="text-sm text-gray-400">
              Watch AI agents work together in real-time
            </p>
          </div>
        </motion.div>

        {/* What You'll See */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 mb-8 border border-gray-700"
        >
          <h3 className="text-xl font-semibold mb-4">What You&apos;ll See:</h3>
          <div className="grid grid-cols-2 gap-6 text-left">
            <div>
              <h4 className="font-medium text-blue-400 mb-2">Business Value</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• 80% cost reduction metrics</li>
                <li>• 24/7 operational capacity</li>
                <li>• 3x faster execution times</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-purple-400 mb-2">
                Live Demonstration
              </h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• AI research team collaboration</li>
                <li>• Autonomous problem solving</li>
                <li>• Real-time decision making</li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="space-y-4"
        >
          <button
            onClick={onStartDemo}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-lg text-lg font-semibold flex items-center gap-3 mx-auto transition-all transform hover:scale-105"
          >
            <Play className="w-5 h-5" />
            Start CEO Demo
            <ArrowRight className="w-5 h-5" />
          </button>

          <p className="text-sm text-gray-400">
            No signup required • 3 minutes • Full interactive experience
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}
