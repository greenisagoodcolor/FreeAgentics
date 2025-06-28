/**
 * Performance Optimization Tests
 * 
 * Tests for performance monitoring, optimization, and benchmarking
 * following ADR-007 testing requirements.
 */

import { jest } from '@jest/globals';

// Mock performance monitoring utilities
class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();
  private thresholds: Map<string, number> = new Map();
  private observers: Set<Function> = new Set();

  recordMetric(name: string, value: number): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(value);
    
    // Check thresholds
    const threshold = this.thresholds.get(name);
    if (threshold && value > threshold) {
      this.notifyObservers({
        type: 'threshold_exceeded',
        metric: name,
        value,
        threshold,
        timestamp: performance.now(),
      });
    }
  }

  setThreshold(metric: string, threshold: number): void {
    this.thresholds.set(metric, threshold);
  }

  getAverageMetric(name: string): number {
    const values = this.metrics.get(name) || [];
    return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0;
  }

  getMetricPercentile(name: string, percentile: number): number {
    const values = this.metrics.get(name) || [];
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  clearMetrics(name?: string): void {
    if (name) {
      this.metrics.delete(name);
    } else {
      this.metrics.clear();
    }
  }

  subscribe(observer: Function): void {
    this.observers.add(observer);
  }

  unsubscribe(observer: Function): void {
    this.observers.delete(observer);
  }

  private notifyObservers(event: any): void {
    this.observers.forEach(observer => observer(event));
  }

  getReport(): any {
    const report: any = {};
    
    this.metrics.forEach((values, name) => {
      report[name] = {
        count: values.length,
        average: this.getAverageMetric(name),
        min: Math.min(...values),
        max: Math.max(...values),
        p50: this.getMetricPercentile(name, 50),
        p95: this.getMetricPercentile(name, 95),
        p99: this.getMetricPercentile(name, 99),
      };
    });
    
    return report;
  }
}

// Mock optimization utilities
class MemoryOptimizer {
  private caches: Map<string, Map<string, any>> = new Map();
  private cacheConfig: Map<string, { maxSize: number; ttl: number }> = new Map();

  createCache(name: string, config: { maxSize?: number; ttl?: number } = {}): void {
    this.caches.set(name, new Map());
    this.cacheConfig.set(name, {
      maxSize: config.maxSize || 100,
      ttl: config.ttl || 60000, // 1 minute
    });
  }

  set(cacheName: string, key: string, value: any): void {
    const cache = this.caches.get(cacheName);
    if (!cache) return;
    
    const config = this.cacheConfig.get(cacheName)!;
    
    // Evict if cache is full
    if (cache.size >= config.maxSize) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
    
    cache.set(key, {
      value,
      timestamp: Date.now(),
    });
  }

  get(cacheName: string, key: string): any {
    const cache = this.caches.get(cacheName);
    if (!cache) return null;
    
    const entry = cache.get(key);
    if (!entry) return null;
    
    const config = this.cacheConfig.get(cacheName)!;
    
    // Check TTL
    if (Date.now() - entry.timestamp > config.ttl) {
      cache.delete(key);
      return null;
    }
    
    return entry.value;
  }

  clear(cacheName: string): void {
    const cache = this.caches.get(cacheName);
    if (cache) {
      cache.clear();
    }
  }

  getCacheStats(cacheName: string): any {
    const cache = this.caches.get(cacheName);
    if (!cache) return null;
    
    const config = this.cacheConfig.get(cacheName)!;
    const now = Date.now();
    let expiredCount = 0;
    
    cache.forEach((entry) => {
      if (now - entry.timestamp > config.ttl) {
        expiredCount++;
      }
    });
    
    return {
      size: cache.size,
      maxSize: config.maxSize,
      expiredEntries: expiredCount,
      hitRate: cache.size > 0 ? ((cache.size - expiredCount) / cache.size) : 0,
    };
  }

  getAllCacheStats(): any {
    const stats: any = {};
    this.caches.forEach((_, name) => {
      stats[name] = this.getCacheStats(name);
    });
    return stats;
  }
}

// Mock performance benchmarking
class BenchmarkSuite {
  private benchmarks: Map<string, Function> = new Map();
  private results: Map<string, any> = new Map();

  addBenchmark(name: string, fn: Function): void {
    this.benchmarks.set(name, fn);
  }

  async runBenchmark(name: string, iterations: number = 1000): Promise<any> {
    const benchmark = this.benchmarks.get(name);
    if (!benchmark) throw new Error(`Benchmark '${name}' not found`);
    
    const times: number[] = [];
    
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await benchmark();
      const end = performance.now();
      times.push(end - start);
    }
    
    const result = {
      name,
      iterations,
      totalTime: times.reduce((sum, time) => sum + time, 0),
      averageTime: times.reduce((sum, time) => sum + time, 0) / times.length,
      minTime: Math.min(...times),
      maxTime: Math.max(...times),
      standardDeviation: this.calculateStandardDeviation(times),
      operationsPerSecond: 1000 / (times.reduce((sum, time) => sum + time, 0) / times.length),
    };
    
    this.results.set(name, result);
    return result;
  }

  async runAllBenchmarks(iterations: number = 1000): Promise<Map<string, any>> {
    const results = new Map();
    
    for (const [name] of this.benchmarks) {
      const result = await this.runBenchmark(name, iterations);
      results.set(name, result);
    }
    
    return results;
  }

  compare(name1: string, name2: string): any {
    const result1 = this.results.get(name1);
    const result2 = this.results.get(name2);
    
    if (!result1 || !result2) {
      throw new Error('Both benchmarks must be run before comparison');
    }
    
    return {
      faster: result1.averageTime < result2.averageTime ? name1 : name2,
      speedupFactor: Math.max(result1.averageTime, result2.averageTime) / 
                     Math.min(result1.averageTime, result2.averageTime),
      timeDifference: Math.abs(result1.averageTime - result2.averageTime),
    };
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDifferences = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDifferences.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
  }

  getResults(): Map<string, any> {
    return new Map(this.results);
  }

  clear(): void {
    this.results.clear();
  }
}

// Mock resource monitoring
class ResourceMonitor {
  private interval: NodeJS.Timeout | null = null;
  private listeners: Set<Function> = new Set();
  private isMonitoring = false;

  startMonitoring(intervalMs: number = 1000): void {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    this.interval = setInterval(() => {
      const stats = this.collectStats();
      this.listeners.forEach(listener => listener(stats));
    }, intervalMs);
  }

  stopMonitoring(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.isMonitoring = false;
  }

  addListener(listener: Function): void {
    this.listeners.add(listener);
  }

  removeListener(listener: Function): void {
    this.listeners.delete(listener);
  }

  collectStats(): any {
    return {
      timestamp: Date.now(),
      memory: {
        used: Math.random() * 100, // Mock memory usage MB
        total: 512,
      },
      cpu: {
        usage: Math.random() * 100, // Mock CPU usage percentage
      },
      network: {
        bytesIn: Math.random() * 1024,
        bytesOut: Math.random() * 1024,
      },
      performance: {
        fps: 60 - Math.random() * 10, // Mock FPS
        frameTime: 16 + Math.random() * 5, // Mock frame time ms
      },
    };
  }

  getAverageStats(duration: number = 60000): Promise<any> {
    return new Promise((resolve) => {
      const stats: any[] = [];
      const startTime = Date.now();
      
      const listener = (stat: any) => {
        stats.push(stat);
        
        if (Date.now() - startTime >= duration) {
          this.removeListener(listener);
          
          const averages = {
            memory: {
              used: stats.reduce((sum, s) => sum + s.memory.used, 0) / stats.length,
              total: stats[0]?.memory.total || 0,
            },
            cpu: {
              usage: stats.reduce((sum, s) => sum + s.cpu.usage, 0) / stats.length,
            },
            network: {
              bytesIn: stats.reduce((sum, s) => sum + s.network.bytesIn, 0) / stats.length,
              bytesOut: stats.reduce((sum, s) => sum + s.network.bytesOut, 0) / stats.length,
            },
            performance: {
              fps: stats.reduce((sum, s) => sum + s.performance.fps, 0) / stats.length,
              frameTime: stats.reduce((sum, s) => sum + s.performance.frameTime, 0) / stats.length,
            },
            sampleCount: stats.length,
          };
          
          resolve(averages);
        }
      };
      
      this.addListener(listener);
    });
  }
}

describe('Performance Optimization', () => {
  describe('PerformanceMonitor', () => {
    let monitor: PerformanceMonitor;
    
    beforeEach(() => {
      monitor = new PerformanceMonitor();
    });

    describe('Metric Recording', () => {
      it('records and retrieves metrics', () => {
        monitor.recordMetric('response_time', 150);
        monitor.recordMetric('response_time', 200);
        monitor.recordMetric('response_time', 100);
        
        const average = monitor.getAverageMetric('response_time');
        expect(average).toBe(150); // (150 + 200 + 100) / 3
      });

      it('calculates percentiles correctly', () => {
        const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        values.forEach(val => monitor.recordMetric('test_metric', val));
        
        expect(monitor.getMetricPercentile('test_metric', 50)).toBe(50);
        expect(monitor.getMetricPercentile('test_metric', 95)).toBe(100);
        expect(monitor.getMetricPercentile('test_metric', 99)).toBe(100);
      });

      it('handles empty metrics gracefully', () => {
        expect(monitor.getAverageMetric('nonexistent')).toBe(0);
        expect(monitor.getMetricPercentile('nonexistent', 50)).toBe(0);
      });

      it('clears specific metrics', () => {
        monitor.recordMetric('metric1', 100);
        monitor.recordMetric('metric2', 200);
        
        monitor.clearMetrics('metric1');
        
        expect(monitor.getAverageMetric('metric1')).toBe(0);
        expect(monitor.getAverageMetric('metric2')).toBe(200);
      });

      it('clears all metrics', () => {
        monitor.recordMetric('metric1', 100);
        monitor.recordMetric('metric2', 200);
        
        monitor.clearMetrics();
        
        expect(monitor.getAverageMetric('metric1')).toBe(0);
        expect(monitor.getAverageMetric('metric2')).toBe(0);
      });
    });

    describe('Threshold Monitoring', () => {
      it('triggers alerts when thresholds are exceeded', () => {
        const alertHandler = jest.fn();
        monitor.subscribe(alertHandler);
        monitor.setThreshold('response_time', 100);
        
        monitor.recordMetric('response_time', 150); // Exceeds threshold
        
        expect(alertHandler).toHaveBeenCalledWith({
          type: 'threshold_exceeded',
          metric: 'response_time',
          value: 150,
          threshold: 100,
          timestamp: expect.any(Number),
        });
      });

      it('does not trigger alerts for values below threshold', () => {
        const alertHandler = jest.fn();
        monitor.subscribe(alertHandler);
        monitor.setThreshold('response_time', 100);
        
        monitor.recordMetric('response_time', 50); // Below threshold
        
        expect(alertHandler).not.toHaveBeenCalled();
      });

      it('allows unsubscribing from alerts', () => {
        const alertHandler = jest.fn();
        monitor.subscribe(alertHandler);
        monitor.unsubscribe(alertHandler);
        monitor.setThreshold('response_time', 100);
        
        monitor.recordMetric('response_time', 150);
        
        expect(alertHandler).not.toHaveBeenCalled();
      });
    });

    describe('Performance Reports', () => {
      it('generates comprehensive performance reports', () => {
        monitor.recordMetric('api_latency', 50);
        monitor.recordMetric('api_latency', 100);
        monitor.recordMetric('api_latency', 75);
        
        monitor.recordMetric('memory_usage', 60);
        monitor.recordMetric('memory_usage', 80);
        
        const report = monitor.getReport();
        
        expect(report).toEqual({
          api_latency: {
            count: 3,
            average: 75,
            min: 50,
            max: 100,
            p50: 75,
            p95: 100,
            p99: 100,
          },
          memory_usage: {
            count: 2,
            average: 70,
            min: 60,
            max: 80,
            p50: 80,
            p95: 80,
            p99: 80,
          },
        });
      });

      it('handles empty report generation', () => {
        const report = monitor.getReport();
        expect(report).toEqual({});
      });
    });
  });

  describe('MemoryOptimizer', () => {
    let optimizer: MemoryOptimizer;
    
    beforeEach(() => {
      optimizer = new MemoryOptimizer();
    });

    describe('Cache Management', () => {
      it('creates and uses cache', () => {
        optimizer.createCache('test_cache');
        optimizer.set('test_cache', 'key1', 'value1');
        
        const value = optimizer.get('test_cache', 'key1');
        expect(value).toBe('value1');
      });

      it('respects cache size limits', () => {
        optimizer.createCache('small_cache', { maxSize: 2 });
        
        optimizer.set('small_cache', 'key1', 'value1');
        optimizer.set('small_cache', 'key2', 'value2');
        optimizer.set('small_cache', 'key3', 'value3'); // Should evict key1
        
        expect(optimizer.get('small_cache', 'key1')).toBeNull();
        expect(optimizer.get('small_cache', 'key2')).toBe('value2');
        expect(optimizer.get('small_cache', 'key3')).toBe('value3');
      });

      it('respects TTL (time to live)', () => {
        optimizer.createCache('ttl_cache', { ttl: 100 }); // 100ms TTL
        optimizer.set('ttl_cache', 'key1', 'value1');
        
        expect(optimizer.get('ttl_cache', 'key1')).toBe('value1');
        
        // Wait for TTL to expire
        return new Promise<void>((resolve) => {
          setTimeout(() => {
            expect(optimizer.get('ttl_cache', 'key1')).toBeNull();
            resolve();
          }, 150);
        });
      });

      it('clears cache', () => {
        optimizer.createCache('clear_cache');
        optimizer.set('clear_cache', 'key1', 'value1');
        optimizer.set('clear_cache', 'key2', 'value2');
        
        optimizer.clear('clear_cache');
        
        expect(optimizer.get('clear_cache', 'key1')).toBeNull();
        expect(optimizer.get('clear_cache', 'key2')).toBeNull();
      });

      it('handles non-existent cache gracefully', () => {
        optimizer.set('nonexistent', 'key', 'value');
        expect(optimizer.get('nonexistent', 'key')).toBeNull();
      });
    });

    describe('Cache Statistics', () => {
      it('provides cache statistics', () => {
        optimizer.createCache('stats_cache', { maxSize: 10, ttl: 60000 });
        optimizer.set('stats_cache', 'key1', 'value1');
        optimizer.set('stats_cache', 'key2', 'value2');
        
        const stats = optimizer.getCacheStats('stats_cache');
        
        expect(stats).toEqual({
          size: 2,
          maxSize: 10,
          expiredEntries: 0,
          hitRate: 1,
        });
      });

      it('provides statistics for all caches', () => {
        optimizer.createCache('cache1');
        optimizer.createCache('cache2');
        optimizer.set('cache1', 'key', 'value');
        
        const allStats = optimizer.getAllCacheStats();
        
        expect(allStats).toHaveProperty('cache1');
        expect(allStats).toHaveProperty('cache2');
        expect(allStats.cache1.size).toBe(1);
        expect(allStats.cache2.size).toBe(0);
      });

      it('returns null for non-existent cache stats', () => {
        const stats = optimizer.getCacheStats('nonexistent');
        expect(stats).toBeNull();
      });
    });
  });

  describe('BenchmarkSuite', () => {
    let benchmarkSuite: BenchmarkSuite;
    
    beforeEach(() => {
      benchmarkSuite = new BenchmarkSuite();
    });

    describe('Benchmark Execution', () => {
      it('runs synchronous benchmarks', async () => {
        benchmarkSuite.addBenchmark('simple_math', () => {
          let result = 0;
          for (let i = 0; i < 1000; i++) {
            result += Math.sqrt(i);
          }
          return result;
        });
        
        const result = await benchmarkSuite.runBenchmark('simple_math', 10);
        
        expect(result).toMatchObject({
          name: 'simple_math',
          iterations: 10,
          totalTime: expect.any(Number),
          averageTime: expect.any(Number),
          minTime: expect.any(Number),
          maxTime: expect.any(Number),
          standardDeviation: expect.any(Number),
          operationsPerSecond: expect.any(Number),
        });
        
        expect(result.totalTime).toBeGreaterThan(0);
        expect(result.averageTime).toBeGreaterThan(0);
        expect(result.operationsPerSecond).toBeGreaterThan(0);
      });

      it('runs asynchronous benchmarks', async () => {
        benchmarkSuite.addBenchmark('async_operation', async () => {
          await new Promise(resolve => setTimeout(resolve, 1));
        });
        
        const result = await benchmarkSuite.runBenchmark('async_operation', 5);
        
        expect(result.name).toBe('async_operation');
        expect(result.iterations).toBe(5);
        expect(result.averageTime).toBeGreaterThan(1); // At least 1ms
      });

      it('throws error for non-existent benchmark', async () => {
        await expect(
          benchmarkSuite.runBenchmark('nonexistent')
        ).rejects.toThrow("Benchmark 'nonexistent' not found");
      });

      it('runs all benchmarks', async () => {
        benchmarkSuite.addBenchmark('test1', () => Math.random());
        benchmarkSuite.addBenchmark('test2', () => Math.random());
        
        const results = await benchmarkSuite.runAllBenchmarks(5);
        
        expect(results.size).toBe(2);
        expect(results.has('test1')).toBe(true);
        expect(results.has('test2')).toBe(true);
      });
    });

    describe('Benchmark Comparison', () => {
      it('compares benchmark results', async () => {
        benchmarkSuite.addBenchmark('fast_operation', () => {
          // Minimal work
          return 1 + 1;
        });
        
        benchmarkSuite.addBenchmark('slow_operation', () => {
          // More work
          let result = 0;
          for (let i = 0; i < 100; i++) {
            result += Math.sqrt(i);
          }
          return result;
        });
        
        await benchmarkSuite.runBenchmark('fast_operation', 100);
        await benchmarkSuite.runBenchmark('slow_operation', 100);
        
        const comparison = benchmarkSuite.compare('fast_operation', 'slow_operation');
        
        expect(comparison.faster).toBe('fast_operation');
        expect(comparison.speedupFactor).toBeGreaterThan(1);
        expect(comparison.timeDifference).toBeGreaterThan(0);
      });

      it('throws error when comparing unrun benchmarks', () => {
        benchmarkSuite.addBenchmark('test1', () => {});
        benchmarkSuite.addBenchmark('test2', () => {});
        
        expect(() => {
          benchmarkSuite.compare('test1', 'test2');
        }).toThrow('Both benchmarks must be run before comparison');
      });
    });

    describe('Results Management', () => {
      it('stores and retrieves results', async () => {
        benchmarkSuite.addBenchmark('test', () => Math.random());
        await benchmarkSuite.runBenchmark('test', 10);
        
        const results = benchmarkSuite.getResults();
        expect(results.has('test')).toBe(true);
        
        const testResult = results.get('test');
        expect(testResult.name).toBe('test');
        expect(testResult.iterations).toBe(10);
      });

      it('clears results', async () => {
        benchmarkSuite.addBenchmark('test', () => Math.random());
        await benchmarkSuite.runBenchmark('test', 5);
        
        benchmarkSuite.clear();
        
        const results = benchmarkSuite.getResults();
        expect(results.size).toBe(0);
      });
    });
  });

  describe('ResourceMonitor', () => {
    let resourceMonitor: ResourceMonitor;
    
    beforeEach(() => {
      resourceMonitor = new ResourceMonitor();
    });

    afterEach(() => {
      resourceMonitor.stopMonitoring();
    });

    describe('Monitoring Control', () => {
      it('starts and stops monitoring', () => {
        expect(resourceMonitor['isMonitoring']).toBe(false);
        
        resourceMonitor.startMonitoring(100);
        expect(resourceMonitor['isMonitoring']).toBe(true);
        
        resourceMonitor.stopMonitoring();
        expect(resourceMonitor['isMonitoring']).toBe(false);
      });

      it('prevents multiple monitoring sessions', () => {
        resourceMonitor.startMonitoring(100);
        const firstInterval = resourceMonitor['interval'];
        
        resourceMonitor.startMonitoring(200); // Should not create new interval
        const secondInterval = resourceMonitor['interval'];
        
        expect(firstInterval).toBe(secondInterval);
        
        resourceMonitor.stopMonitoring();
      });
    });

    describe('Stats Collection', () => {
      it('collects system statistics', () => {
        const stats = resourceMonitor.collectStats();
        
        expect(stats).toMatchObject({
          timestamp: expect.any(Number),
          memory: {
            used: expect.any(Number),
            total: expect.any(Number),
          },
          cpu: {
            usage: expect.any(Number),
          },
          network: {
            bytesIn: expect.any(Number),
            bytesOut: expect.any(Number),
          },
          performance: {
            fps: expect.any(Number),
            frameTime: expect.any(Number),
          },
        });
        
        expect(stats.memory.used).toBeGreaterThanOrEqual(0);
        expect(stats.memory.used).toBeLessThanOrEqual(100);
        expect(stats.cpu.usage).toBeGreaterThanOrEqual(0);
        expect(stats.cpu.usage).toBeLessThanOrEqual(100);
      });
    });

    describe('Listener Management', () => {
      it('adds and removes listeners', (done) => {
        const listener = jest.fn();
        
        resourceMonitor.addListener(listener);
        resourceMonitor.startMonitoring(50);
        
        setTimeout(() => {
          expect(listener).toHaveBeenCalled();
          
          resourceMonitor.removeListener(listener);
          const callCount = listener.mock.calls.length;
          
          setTimeout(() => {
            // Should not have been called again after removal
            expect(listener).toHaveBeenCalledTimes(callCount);
            done();
          }, 100);
        }, 100);
      });

      it('provides average statistics over time', async () => {
        resourceMonitor.startMonitoring(10);
        
        const averages = await resourceMonitor.getAverageStats(100);
        
        expect(averages).toMatchObject({
          memory: {
            used: expect.any(Number),
            total: expect.any(Number),
          },
          cpu: {
            usage: expect.any(Number),
          },
          network: {
            bytesIn: expect.any(Number),
            bytesOut: expect.any(Number),
          },
          performance: {
            fps: expect.any(Number),
            frameTime: expect.any(Number),
          },
          sampleCount: expect.any(Number),
        });
        
        expect(averages.sampleCount).toBeGreaterThan(0);
      });
    });
  });

  describe('Integration Performance Tests', () => {
    it('measures end-to-end performance', async () => {
      const monitor = new PerformanceMonitor();
      const benchmark = new BenchmarkSuite();
      
      // Set up performance monitoring
      monitor.setThreshold('operation_time', 100);
      
      // Create benchmark for complex operation
      benchmark.addBenchmark('complex_operation', () => {
        const start = performance.now();
        
        // Simulate complex operation
        let result = 0;
        for (let i = 0; i < 10000; i++) {
          result += Math.sqrt(i) * Math.random();
        }
        
        const duration = performance.now() - start;
        monitor.recordMetric('operation_time', duration);
        
        return result;
      });
      
      // Run benchmark
      const result = await benchmark.runBenchmark('complex_operation', 50);
      
      // Verify performance metrics
      expect(result.averageTime).toBeGreaterThan(0);
      expect(monitor.getAverageMetric('operation_time')).toBeGreaterThan(0);
      
      // Check if we can optimize
      const p95Time = monitor.getMetricPercentile('operation_time', 95);
      expect(p95Time).toBeGreaterThan(0);
    });

    it('tests memory optimization under load', () => {
      const optimizer = new MemoryOptimizer();
      const monitor = new PerformanceMonitor();
      
      optimizer.createCache('load_test', { maxSize: 1000, ttl: 30000 });
      
      // Simulate high load
      const startTime = performance.now();
      
      for (let i = 0; i < 10000; i++) {
        const key = `key_${i % 1000}`; // Cycle through keys
        const value = { data: new Array(100).fill(i) };
        
        optimizer.set('load_test', key, value);
        const retrieved = optimizer.get('load_test', key);
        
        if (retrieved) {
          monitor.recordMetric('cache_hit', 1);
        } else {
          monitor.recordMetric('cache_miss', 1);
        }
      }
      
      const endTime = performance.now();
      monitor.recordMetric('load_test_duration', endTime - startTime);
      
      // Verify cache performance
      const stats = optimizer.getCacheStats('load_test');
      expect(stats?.size).toBeLessThanOrEqual(1000);
      expect(stats?.hitRate).toBeGreaterThan(0);
      
      // Verify timing
      expect(monitor.getAverageMetric('load_test_duration')).toBeGreaterThan(0);
    });

    it('monitors resource usage during intensive operations', async () => {
      const resourceMonitor = new ResourceMonitor();
      const stats: any[] = [];
      
      const listener = (stat: any) => stats.push(stat);
      resourceMonitor.addListener(listener);
      resourceMonitor.startMonitoring(20);
      
      // Perform intensive operations
      const operations = Array.from({ length: 100 }, (_, i) => 
        new Promise<void>((resolve) => {
          setTimeout(() => {
            // Simulate CPU-intensive work
            let result = 0;
            for (let j = 0; j < 1000; j++) {
              result += Math.sqrt(j) * Math.sin(j);
            }
            resolve();
          }, i * 2);
        })
      );
      
      await Promise.all(operations);
      
      // Stop monitoring and analyze
      resourceMonitor.stopMonitoring();
      
      expect(stats.length).toBeGreaterThan(0);
      expect(stats[0]).toHaveProperty('memory');
      expect(stats[0]).toHaveProperty('cpu');
      expect(stats[0]).toHaveProperty('performance');
    });
  });
});
