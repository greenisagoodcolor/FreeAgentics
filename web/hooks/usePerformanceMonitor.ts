"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface PerformanceMetrics {
  renderTime: number;
  componentMounts: number;
  reRenders: number;
  memoryUsage: number;
  lastRenderTime: number;
  averageRenderTime: number;
  slowRenders: number;
  cacheHitRate: number;
  optimizationSuggestions: string[];
}

export interface PerformanceThresholds {
  slowRenderThreshold: number; // ms
  memoryWarningThreshold: number; // MB
  reRenderWarningThreshold: number;
  cacheHitRateMinimum: number; // percentage
}

export interface UsePerformanceMonitorOptions {
  componentName: string;
  enabled?: boolean;
  thresholds?: Partial<PerformanceThresholds>;
  trackMemory?: boolean;
  trackCacheHits?: boolean;
  onSlowRender?: (renderTime: number) => void;
  onMemoryWarning?: (usage: number) => void;
  onOptimizationSuggestion?: (suggestions: string[]) => void;
}

const defaultThresholds: PerformanceThresholds = {
  slowRenderThreshold: 16, // 60 FPS = 16.67ms per frame
  memoryWarningThreshold: 50, // 50MB
  reRenderWarningThreshold: 5, // 5 re-renders per second
  cacheHitRateMinimum: 80, // 80% cache hit rate
};

export function usePerformanceMonitor(options: UsePerformanceMonitorOptions) {
  const {
    componentName,
    enabled = true,
    thresholds = {},
    trackMemory = true,
    trackCacheHits = true,
    onSlowRender,
    onMemoryWarning,
    onOptimizationSuggestion,
  } = options;

  const finalThresholds = { ...defaultThresholds, ...thresholds };

  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    componentMounts: 0,
    reRenders: 0,
    memoryUsage: 0,
    lastRenderTime: 0,
    averageRenderTime: 0,
    slowRenders: 0,
    cacheHitRate: 100,
    optimizationSuggestions: [],
  });

  const renderStartTime = useRef<number>(0);
  const renderTimes = useRef<number[]>([]);
  const mountTime = useRef<number>(0);
  const reRenderCount = useRef<number>(0);
  const cacheRequests = useRef<number>(0);
  const cacheHits = useRef<number>(0);
  const lastReRenderTime = useRef<number>(0);
  const memoryCheckInterval = useRef<NodeJS.Timeout | null>(null);

  // Track component mount
  useEffect(() => {
    if (!enabled) return;

    mountTime.current = performance.now();
    setMetrics((prev) => ({
      ...prev,
      componentMounts: prev.componentMounts + 1,
    }));

    return () => {
      if (memoryCheckInterval.current) {
        clearInterval(memoryCheckInterval.current);
      }
    };
  }, [enabled]);

  // Memory monitoring
  useEffect(() => {
    if (!enabled || !trackMemory) return;

    const checkMemory = () => {
      if ("memory" in performance) {
        const memInfo = (performance as any).memory;
        const usageInMB = memInfo.usedJSHeapSize / (1024 * 1024);

        setMetrics((prev) => ({
          ...prev,
          memoryUsage: usageInMB,
        }));

        if (usageInMB > finalThresholds.memoryWarningThreshold) {
          onMemoryWarning?.(usageInMB);
        }
      }
    };

    checkMemory();
    memoryCheckInterval.current = setInterval(checkMemory, 5000); // Check every 5 seconds

    return () => {
      if (memoryCheckInterval.current) {
        clearInterval(memoryCheckInterval.current);
      }
    };
  }, [
    enabled,
    trackMemory,
    finalThresholds.memoryWarningThreshold,
    onMemoryWarning,
  ]);

  // Render performance tracking
  const startRender = useCallback(() => {
    if (!enabled) return;
    renderStartTime.current = performance.now();
  }, [enabled]);

  const endRender = useCallback(() => {
    if (!enabled || renderStartTime.current === 0) return;

    const renderTime = performance.now() - renderStartTime.current;
    renderTimes.current.push(renderTime);

    // Keep only last 100 render times for average calculation
    if (renderTimes.current.length > 100) {
      renderTimes.current.shift();
    }

    const averageRenderTime =
      renderTimes.current.reduce((sum, time) => sum + time, 0) /
      renderTimes.current.length;
    const isSlowRender = renderTime > finalThresholds.slowRenderThreshold;

    // Track re-renders
    const now = Date.now();
    if (now - lastReRenderTime.current < 1000) {
      reRenderCount.current++;
    } else {
      reRenderCount.current = 1;
    }
    lastReRenderTime.current = now;

    setMetrics((prev) => ({
      ...prev,
      renderTime,
      lastRenderTime: renderTime,
      averageRenderTime,
      reRenders: prev.reRenders + 1,
      slowRenders: prev.slowRenders + (isSlowRender ? 1 : 0),
    }));

    if (isSlowRender) {
      onSlowRender?.(renderTime);
    }

    renderStartTime.current = 0;
  }, [enabled, finalThresholds.slowRenderThreshold, onSlowRender]);

  // Cache performance tracking
  const trackCacheRequest = useCallback(
    (isHit: boolean = false) => {
      if (!enabled || !trackCacheHits) return;

      cacheRequests.current++;
      if (isHit) {
        cacheHits.current++;
      }

      const hitRate = (cacheHits.current / cacheRequests.current) * 100;

      setMetrics((prev) => ({
        ...prev,
        cacheHitRate: hitRate,
      }));
    },
    [enabled, trackCacheHits],
  );

  // Generate optimization suggestions
  const generateOptimizationSuggestions = useCallback(() => {
    const suggestions: string[] = [];

    if (metrics.averageRenderTime > finalThresholds.slowRenderThreshold) {
      suggestions.push(
        `Consider memoizing ${componentName} - average render time is ${metrics.averageRenderTime.toFixed(2)}ms`,
      );
    }

    if (reRenderCount.current > finalThresholds.reRenderWarningThreshold) {
      suggestions.push(
        `High re-render frequency detected in ${componentName} - consider optimizing dependencies`,
      );
    }

    if (metrics.memoryUsage > finalThresholds.memoryWarningThreshold) {
      suggestions.push(
        `High memory usage detected (${metrics.memoryUsage.toFixed(2)}MB) - check for memory leaks`,
      );
    }

    if (metrics.cacheHitRate < finalThresholds.cacheHitRateMinimum) {
      suggestions.push(
        `Low cache hit rate (${metrics.cacheHitRate.toFixed(1)}%) - optimize caching strategy`,
      );
    }

    if (metrics.slowRenders > 5) {
      suggestions.push(
        `${metrics.slowRenders} slow renders detected - consider code splitting or virtualization`,
      );
    }

    return suggestions;
  }, [metrics, finalThresholds, componentName, reRenderCount]);

  // Update suggestions periodically
  useEffect(() => {
    if (!enabled) return;

    const suggestions = generateOptimizationSuggestions();
    if (
      suggestions.length !== metrics.optimizationSuggestions.length ||
      suggestions.some((s, i) => s !== metrics.optimizationSuggestions[i])
    ) {
      setMetrics((prev) => ({
        ...prev,
        optimizationSuggestions: suggestions,
      }));

      if (suggestions.length > 0) {
        onOptimizationSuggestion?.(suggestions);
      }
    }
  }, [
    enabled,
    generateOptimizationSuggestions,
    metrics.optimizationSuggestions,
    onOptimizationSuggestion,
  ]);

  // Performance profiler hooks
  const profileRender = useCallback(
    (renderFn: () => void) => {
      startRender();
      try {
        renderFn();
      } finally {
        endRender();
      }
    },
    [startRender, endRender],
  );

  // Get performance report
  const getPerformanceReport = useCallback(() => {
    const report = {
      componentName,
      timestamp: new Date().toISOString(),
      metrics: { ...metrics },
      thresholds: finalThresholds,
      renderTimesHistory: [...renderTimes.current],
      suggestions: generateOptimizationSuggestions(),
      healthScore: calculateHealthScore(),
    };

    return report;
  }, [
    componentName,
    metrics,
    finalThresholds,
    generateOptimizationSuggestions,
  ]);

  // Calculate overall health score (0-100)
  const calculateHealthScore = useCallback(() => {
    let score = 100;

    // Deduct points for performance issues
    if (metrics.averageRenderTime > finalThresholds.slowRenderThreshold) {
      score -= Math.min(
        30,
        (metrics.averageRenderTime - finalThresholds.slowRenderThreshold) * 2,
      );
    }

    if (metrics.memoryUsage > finalThresholds.memoryWarningThreshold) {
      score -= Math.min(
        25,
        (metrics.memoryUsage - finalThresholds.memoryWarningThreshold) * 0.5,
      );
    }

    if (metrics.cacheHitRate < finalThresholds.cacheHitRateMinimum) {
      score -= Math.min(
        20,
        (finalThresholds.cacheHitRateMinimum - metrics.cacheHitRate) * 0.5,
      );
    }

    if (reRenderCount.current > finalThresholds.reRenderWarningThreshold) {
      score -= Math.min(
        15,
        (reRenderCount.current - finalThresholds.reRenderWarningThreshold) * 3,
      );
    }

    if (metrics.slowRenders > 0) {
      score -= Math.min(10, metrics.slowRenders);
    }

    return Math.max(0, Math.round(score));
  }, [metrics, finalThresholds, reRenderCount]);

  // Reset metrics
  const resetMetrics = useCallback(() => {
    renderTimes.current = [];
    reRenderCount.current = 0;
    cacheRequests.current = 0;
    cacheHits.current = 0;

    setMetrics({
      renderTime: 0,
      componentMounts: 0,
      reRenders: 0,
      memoryUsage: 0,
      lastRenderTime: 0,
      averageRenderTime: 0,
      slowRenders: 0,
      cacheHitRate: 100,
      optimizationSuggestions: [],
    });
  }, []);

  return {
    metrics,
    startRender,
    endRender,
    trackCacheRequest,
    profileRender,
    getPerformanceReport,
    resetMetrics,
    healthScore: calculateHealthScore(),
    isEnabled: enabled,
  };
}
