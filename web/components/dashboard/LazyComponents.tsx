"use client";

import React, { Suspense } from "react";
import { motion } from "framer-motion";

// Lazy load heavy dashboard components
const KnowledgeGraphVisualization = React.lazy(() => 
  import("./KnowledgeGraphVisualization").then(module => ({
    default: module.default
  }))
);

const AnalyticsWidgetSystem = React.lazy(() => 
  import("./AnalyticsWidgetSystem").then(module => ({
    default: module.default
  }))
);

// Loading skeleton component
const LoadingSkeleton: React.FC<{ 
  height?: string; 
  className?: string;
  children?: React.ReactNode;
}> = ({ 
  height = "200px", 
  className = "",
  children 
}) => (
  <motion.div
    className={`skeleton-enhanced ${className}`}
    style={{ height }}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    transition={{ duration: 0.2 }}
  >
    <div className="skeleton-shimmer" />
    {children && (
      <div className="flex items-center justify-center h-full text-sm text-gray-500">
        {children}
      </div>
    )}
  </motion.div>
);

// Lazy component exports
export const LazyKnowledgeGraph: React.FC<any> = (props) => (
  <Suspense fallback={<LoadingSkeleton height="400px" />}>
    <KnowledgeGraphVisualization {...props} />
  </Suspense>
);

export const LazyAnalyticsSystem: React.FC<any> = (props) => (
  <Suspense fallback={<LoadingSkeleton height="300px" />}>
    <AnalyticsWidgetSystem {...props} />
  </Suspense>
);

// Preload utilities
export const preloadComponent = (componentName: string) => {
  switch (componentName) {
    case 'knowledge-graph':
      import("./KnowledgeGraphVisualization");
      break;
    case 'analytics':
      import("./AnalyticsWidgetSystem");
      break;
  }
};

export default {
  LazyKnowledgeGraph,
  LazyAnalyticsSystem,
  preloadComponent
};
