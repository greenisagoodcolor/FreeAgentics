"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  TrendingUp,
  Users,
  Brain,
  Zap,
  Target,
  ArrowRight,
  Play,
  CheckCircle,
  DollarSign,
  Clock,
  Shield,
} from "lucide-react";
import { DashboardView } from "../../page";

interface CEODemoLayoutProps {
  view: DashboardView;
}

// Demo stages for guided experience
type DemoStage =
  | "welcome"
  | "value-props"
  | "live-demo"
  | "results"
  | "next-steps";

export default function CEODemoLayout({ view }: CEODemoLayoutProps) {
  const [currentStage, setCurrentStage] = useState<DemoStage>("welcome");
  const [isPlaying, setIsPlaying] = useState(false);

  const startDemo = () => {
    setIsPlaying(true);
    setCurrentStage("value-props");
  };

  const resetDemo = () => {
    setIsPlaying(false);
    setCurrentStage("welcome");
  };

  return (
    <div className="ceo-demo-layout h-full bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      {/* Progress Bar */}
      <div className="fixed top-0 left-0 right-0 z-50 bg-gray-900/90 backdrop-blur-sm">
        <div className="h-1 bg-gray-700">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            initial={{ width: "0%" }}
            animate={{
              width:
                currentStage === "welcome"
                  ? "0%"
                  : currentStage === "value-props"
                    ? "25%"
                    : currentStage === "live-demo"
                      ? "50%"
                      : currentStage === "results"
                        ? "75%"
                        : "100%",
            }}
            transition={{ duration: 0.5 }}
          />
        </div>

        {/* Demo Controls */}
        <div className="flex items-center justify-between px-8 py-4">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold">FreeAgentics</h1>
            <span className="text-sm text-gray-400">CEO Demo Experience</span>
          </div>

          <div className="flex items-center gap-3">
            {isPlaying ? (
              <button
                onClick={resetDemo}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors"
              >
                Reset Demo
              </button>
            ) : (
              <button
                onClick={startDemo}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Start Demo
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="pt-20 h-full">
        <AnimatePresence mode="wait">
          {currentStage === "welcome" && <WelcomeStage onStart={startDemo} />}
          {currentStage === "value-props" && (
            <ValuePropsStage onNext={() => setCurrentStage("live-demo")} />
          )}
          {currentStage === "live-demo" && (
            <LiveDemoStage onNext={() => setCurrentStage("results")} />
          )}
          {currentStage === "results" && (
            <ResultsStage onNext={() => setCurrentStage("next-steps")} />
          )}
          {currentStage === "next-steps" && (
            <NextStepsStage onReset={resetDemo} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// Welcome Stage
function WelcomeStage({ onStart }: { onStart: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex flex-col items-center justify-center h-full text-center max-w-4xl mx-auto px-8"
    >
      <motion.div
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2 }}
        className="mb-8"
      >
        <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mb-6 mx-auto">
          <Brain className="w-12 h-12 text-white" />
        </div>
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          FREEAGENTICS PLATFORM
        </h1>
        <p className="text-xl text-gray-300 mb-8">
          Enterprise AI agent orchestration delivering 80% cost reduction and
          24/7 autonomous operations
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="grid grid-cols-3 gap-8 mb-12"
      >
        <div className="text-center">
          <DollarSign className="w-8 h-8 text-green-400 mx-auto mb-2" />
          <h3 className="font-semibold mb-1">Cost Reduction</h3>
          <p className="text-sm text-gray-400">Up to 80% operational savings</p>
        </div>
        <div className="text-center">
          <Clock className="w-8 h-8 text-blue-400 mx-auto mb-2" />
          <h3 className="font-semibold mb-1">24/7 Operations</h3>
          <p className="text-sm text-gray-400">Never stop working</p>
        </div>
        <div className="text-center">
          <Shield className="w-8 h-8 text-purple-400 mx-auto mb-2" />
          <h3 className="font-semibold mb-1">Enterprise Ready</h3>
          <p className="text-sm text-gray-400">Secure, scalable, compliant</p>
        </div>
      </motion.div>

      <motion.button
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        onClick={onStart}
        className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-lg text-lg font-semibold flex items-center gap-3 transition-all transform hover:scale-105"
      >
        <Play className="w-5 h-5" />
        See It In Action
        <ArrowRight className="w-5 h-5" />
      </motion.button>
    </motion.div>
  );
}

// Value Props Stage
function ValuePropsStage({ onNext }: { onNext: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onNext, 4000);
    return () => clearTimeout(timer);
  }, [onNext]);

  const valueProps = [
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "COST OPTIMIZATION",
      description:
        "80% reduction in operational expenses through intelligent automation",
      metric: "3x faster execution",
      color: "from-green-500 to-emerald-500",
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "PARALLEL PROCESSING",
      description:
        "Multi-agent collaborative workflows with zero coordination overhead",
      metric: "Zero coordination overhead",
      color: "from-blue-500 to-cyan-500",
    },
    {
      icon: <Brain className="w-8 h-8" />,
      title: "CONTINUOUS OPERATIONS",
      description:
        "24/7 autonomous operations with self-improving system capabilities",
      metric: "Self-improving systems",
      color: "from-purple-500 to-pink-500",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col items-center justify-center max-w-6xl mx-auto px-8"
    >
      <motion.h2
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-4xl font-bold mb-4 text-center"
      >
        ENTERPRISE INFRASTRUCTURE
      </motion.h2>
      <p className="text-lg text-gray-300 mb-12 text-center">
        Proven at scale by Fortune 500 companies
      </p>

      <div className="grid grid-cols-3 gap-8 w-full">
        {valueProps.map((prop, index) => (
          <motion.div
            key={prop.title}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2 }}
            className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all"
          >
            <div
              className={`w-16 h-16 bg-gradient-to-r ${prop.color} rounded-lg flex items-center justify-center mb-4 text-white`}
            >
              {prop.icon}
            </div>
            <h3 className="text-xl font-semibold mb-3">{prop.title}</h3>
            <p className="text-gray-300 mb-4">{prop.description}</p>
            <div
              className={`text-sm font-semibold bg-gradient-to-r ${prop.color} bg-clip-text text-transparent`}
            >
              {prop.metric}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// Live Demo Stage
function LiveDemoStage({ onNext }: { onNext: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onNext, 4000);
    return () => clearTimeout(timer);
  }, [onNext]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col items-center justify-center max-w-4xl mx-auto px-8 text-center"
    >
      <motion.h2
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-4xl font-bold mb-8"
      >
        SYSTEM METRICS
      </motion.h2>
      <p className="text-lg text-gray-300 mb-8">
        Real-time operational intelligence driving 80% cost reduction
      </p>

      <div className="bg-gray-800/30 rounded-xl p-8 border border-gray-700 w-full mb-6">
        <Brain className="w-16 h-16 text-blue-400 mx-auto mb-4" />
        <h3 className="text-2xl font-semibold mb-2">KNOWLEDGE GRAPH</h3>
        <p className="text-gray-300 mb-6">
          Autonomous AI agents collaborating to solve complex business problems
        </p>

        <div className="bg-gray-900/50 rounded-lg p-6 text-left">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              <span>Agent 1: Market trend analysis complete</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
              <span>Agent 2: Competitive intelligence gathered</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse"></div>
              <span>Agent 3: Strategic insights synthesized</span>
            </div>
          </div>

          <div className="mt-6 p-4 bg-green-900/30 border border-green-700 rounded-lg">
            <CheckCircle className="w-5 h-5 text-green-400 inline mr-2" />
            <span className="text-green-400 font-semibold">
              AGENT CONTROL: Task completed in 3 minutes vs 3 hours manually
            </span>
          </div>
        </div>
      </div>

      <div className="text-sm text-gray-400">
        Professional enterprise dashboard with 24/7 autonomous operations
      </div>
    </motion.div>
  );
}

// Results Stage
function ResultsStage({ onNext }: { onNext: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onNext, 4000);
    return () => clearTimeout(timer);
  }, [onNext]);

  const results = [
    {
      metric: "80%",
      label: "Cost Reduction",
      icon: <DollarSign className="w-6 h-6" />,
    },
    { metric: "24/7", label: "Uptime", icon: <Clock className="w-6 h-6" /> },
    {
      metric: "3x",
      label: "Faster Output",
      icon: <TrendingUp className="w-6 h-6" />,
    },
    { metric: "100%", label: "Accuracy", icon: <Target className="w-6 h-6" /> },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col items-center justify-center max-w-4xl mx-auto px-8 text-center"
    >
      <motion.h2
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-4xl font-bold mb-4"
      >
        BUSINESS INTELLIGENCE METRICS
      </motion.h2>
      <p className="text-lg text-gray-300 mb-8">
        Proven ROI and operational efficiency for enterprise clients
      </p>

      <div className="grid grid-cols-4 gap-8 mb-12">
        {results.map((result, index) => (
          <motion.div
            key={result.label}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className="text-center"
          >
            <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4 text-white">
              {result.icon}
            </div>
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {result.metric}
            </div>
            <div className="text-gray-300">{result.label}</div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-green-900/30 border border-green-700 rounded-lg p-6"
      >
        <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-3" />
        <h3 className="text-xl font-semibold mb-2 text-green-400">
          DEPLOYMENT READY
        </h3>
        <p className="text-gray-300">
          Secure, scalable, and compliant with your existing infrastructure
        </p>
      </motion.div>
    </motion.div>
  );
}

// Next Steps Stage
function NextStepsStage({ onReset }: { onReset: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col items-center justify-center max-w-4xl mx-auto px-8 text-center"
    >
      <motion.h2
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-4xl font-bold mb-4"
      >
        DEPLOYMENT OPTIONS
      </motion.h2>
      <p className="text-lg text-gray-300 mb-8">
        Choose your implementation pathway for immediate business transformation
      </p>

      <div className="grid grid-cols-2 gap-8 mb-12 w-full max-w-2xl">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-blue-900/30 border border-blue-700 rounded-lg p-6"
        >
          <h3 className="text-xl font-semibold mb-2 text-blue-400">
            TRIAL ENVIRONMENT
          </h3>
          <p className="text-sm text-gray-400 mb-3">
            30-day full-access trial with white-glove setup
          </p>
          <p className="text-gray-300 mb-4">
            Proof of concept deployment with your data and use cases
          </p>
          <button className="w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors">
            Start Trial
          </button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-purple-900/30 border border-purple-700 rounded-lg p-6"
        >
          <h3 className="text-xl font-semibold mb-2 text-purple-400">
            ENTERPRISE INTEGRATION
          </h3>
          <p className="text-sm text-gray-400 mb-3">
            Custom demonstration with your data and use cases
          </p>
          <p className="text-gray-300 mb-4">
            Full-scale deployment planning and technical architecture review
          </p>
          <button className="w-full py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition-colors">
            Schedule Demo
          </button>
        </motion.div>
      </div>

      <motion.button
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        onClick={onReset}
        className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
      >
        Watch Demo Again
      </motion.button>
    </motion.div>
  );
}
