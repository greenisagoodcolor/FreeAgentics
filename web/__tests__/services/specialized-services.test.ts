/**
 * Specialized Services Tests
 * 
 * Tests for compression workers, provider monitoring, knowledge graph utilities,
 * and other specialized service components following ADR-007 requirements.
 */

import { jest } from '@jest/globals';

// Mock Worker for compression service
global.Worker = jest.fn(() => ({
  postMessage: jest.fn(),
  terminate: jest.fn(),
  onmessage: null,
  onerror: null,
})) as any;

// Mock IndexedDB for storage services
global.indexedDB = {
  open: jest.fn(() => ({
    onsuccess: null,
    onerror: null,
    onupgradeneeded: null,
    result: {
      createObjectStore: jest.fn(() => ({
        createIndex: jest.fn(),
      })),
      transaction: jest.fn(() => ({
        objectStore: jest.fn(() => ({
          add: jest.fn(() => ({ onsuccess: null, onerror: null })),
          get: jest.fn(() => ({ onsuccess: null, onerror: null })),
          delete: jest.fn(() => ({ onsuccess: null, onerror: null })),
          put: jest.fn(() => ({ onsuccess: null, onerror: null })),
          getAll: jest.fn(() => ({ onsuccess: null, onerror: null })),
          clear: jest.fn(() => ({ onsuccess: null, onerror: null })),
        })),
      })),
    },
  })),
  deleteDatabase: jest.fn(),
} as any;

// Compression Worker Service
interface CompressionOptions {
  algorithm: 'gzip' | 'deflate' | 'brotli';
  level: number;
  chunkSize: number;
}

class CompressionWorker {
  private worker: Worker;
  private pendingJobs: Map<string, { resolve: Function; reject: Function }> = new Map();

  constructor() {
    this.worker = new Worker('/workers/compression.js');
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = this.handleError.bind(this);
  }

  private handleMessage(event: MessageEvent): void {
    const { id, result, error } = event.data;
    const job = this.pendingJobs.get(id);
    
    if (job) {
      if (error) {
        job.reject(new Error(error));
      } else {
        job.resolve(result);
      }
      this.pendingJobs.delete(id);
    }
  }

  private handleError(error: ErrorEvent): void {
    console.error('Compression worker error:', error);
    // Reject all pending jobs
    this.pendingJobs.forEach(({ reject }) => reject(error));
    this.pendingJobs.clear();
  }

  async compress(data: string | ArrayBuffer, options: CompressionOptions): Promise<ArrayBuffer> {
    const id = Math.random().toString(36);
    
    return new Promise((resolve, reject) => {
      this.pendingJobs.set(id, { resolve, reject });
      this.worker.postMessage({
        id,
        action: 'compress',
        data,
        options,
      });
    });
  }

  async decompress(data: ArrayBuffer, algorithm: string): Promise<string> {
    const id = Math.random().toString(36);
    
    return new Promise((resolve, reject) => {
      this.pendingJobs.set(id, { resolve, reject });
      this.worker.postMessage({
        id,
        action: 'decompress',
        data,
        algorithm,
      });
    });
  }

  terminate(): void {
    this.worker.terminate();
    this.pendingJobs.clear();
  }
}

// Provider Monitoring Service
interface ProviderMetrics {
  responseTime: number;
  successRate: number;
  errorRate: number;
  throughput: number;
  availability: number;
  cost: number;
}

interface ProviderAlert {
  id: string;
  providerId: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: Date;
  resolved: boolean;
}

class ProviderMonitoringService {
  private metrics: Map<string, ProviderMetrics[]> = new Map();
  private alerts: ProviderAlert[] = [];
  private thresholds = {
    responseTime: 1000,
    errorRate: 0.05,
    availability: 0.95,
  };

  recordMetric(providerId: string, metric: ProviderMetrics): void {
    if (!this.metrics.has(providerId)) {
      this.metrics.set(providerId, []);
    }
    
    const providerMetrics = this.metrics.get(providerId)!;
    providerMetrics.push(metric);
    
    // Keep only last 100 metrics
    if (providerMetrics.length > 100) {
      providerMetrics.shift();
    }
    
    this.checkThresholds(providerId, metric);
  }

  private checkThresholds(providerId: string, metric: ProviderMetrics): void {
    const alerts: ProviderAlert[] = [];
    
    if (metric.responseTime > this.thresholds.responseTime) {
      alerts.push({
        id: Math.random().toString(36),
        providerId,
        severity: 'warning',
        message: `High response time: ${metric.responseTime}ms`,
        timestamp: new Date(),
        resolved: false,
      });
    }
    
    if (metric.errorRate > this.thresholds.errorRate) {
      alerts.push({
        id: Math.random().toString(36),
        providerId,
        severity: 'error',
        message: `High error rate: ${(metric.errorRate * 100).toFixed(2)}%`,
        timestamp: new Date(),
        resolved: false,
      });
    }
    
    if (metric.availability < this.thresholds.availability) {
      alerts.push({
        id: Math.random().toString(36),
        providerId,
        severity: 'critical',
        message: `Low availability: ${(metric.availability * 100).toFixed(2)}%`,
        timestamp: new Date(),
        resolved: false,
      });
    }
    
    this.alerts.push(...alerts);
  }

  getMetrics(providerId: string): ProviderMetrics[] {
    return this.metrics.get(providerId) || [];
  }

  getAggregatedMetrics(providerId: string): Partial<ProviderMetrics> {
    const metrics = this.getMetrics(providerId);
    if (metrics.length === 0) return {};
    
    return {
      responseTime: metrics.reduce((sum, m) => sum + m.responseTime, 0) / metrics.length,
      successRate: metrics.reduce((sum, m) => sum + m.successRate, 0) / metrics.length,
      errorRate: metrics.reduce((sum, m) => sum + m.errorRate, 0) / metrics.length,
      throughput: metrics.reduce((sum, m) => sum + m.throughput, 0) / metrics.length,
      availability: metrics.reduce((sum, m) => sum + m.availability, 0) / metrics.length,
      cost: metrics.reduce((sum, m) => sum + m.cost, 0),
    };
  }

  getAlerts(providerId?: string): ProviderAlert[] {
    return providerId 
      ? this.alerts.filter(alert => alert.providerId === providerId)
      : this.alerts;
  }

  resolveAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.resolved = true;
      return true;
    }
    return false;
  }

  clearResolvedAlerts(): void {
    this.alerts = this.alerts.filter(alert => !alert.resolved);
  }
}

// Knowledge Graph Export Service
interface KnowledgeNode {
  id: string;
  type: string;
  properties: Record<string, any>;
  metadata: {
    created: Date;
    updated: Date;
    confidence: number;
  };
}

interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  properties: Record<string, any>;
}

interface ExportOptions {
  format: 'json' | 'csv' | 'gexf' | 'graphml';
  includeMetadata: boolean;
  filterByType?: string[];
  filterByConfidence?: number;
  compressed?: boolean;
}

class KnowledgeGraphExportService {
  private compressionWorker?: CompressionWorker;

  constructor() {
    this.compressionWorker = new CompressionWorker();
  }

  async exportGraph(
    nodes: KnowledgeNode[], 
    edges: KnowledgeEdge[], 
    options: ExportOptions
  ): Promise<string | ArrayBuffer> {
    // Filter nodes and edges
    let filteredNodes = this.filterNodes(nodes, options);
    let filteredEdges = this.filterEdges(edges, options);
    
    // Generate export data based on format
    let exportData: string;
    
    switch (options.format) {
      case 'json':
        exportData = this.exportToJSON(filteredNodes, filteredEdges, options);
        break;
      case 'csv':
        exportData = this.exportToCSV(filteredNodes, filteredEdges, options);
        break;
      case 'gexf':
        exportData = this.exportToGEXF(filteredNodes, filteredEdges, options);
        break;
      case 'graphml':
        exportData = this.exportToGraphML(filteredNodes, filteredEdges, options);
        break;
      default:
        throw new Error(`Unsupported export format: ${options.format}`);
    }
    
    // Compress if requested
    if (options.compressed && this.compressionWorker) {
      return await this.compressionWorker.compress(exportData, {
        algorithm: 'gzip',
        level: 6,
        chunkSize: 1024 * 1024,
      });
    }
    
    return exportData;
  }

  private filterNodes(nodes: KnowledgeNode[], options: ExportOptions): KnowledgeNode[] {
    let filtered = nodes;
    
    if (options.filterByType) {
      filtered = filtered.filter(node => options.filterByType!.includes(node.type));
    }
    
    if (options.filterByConfidence !== undefined) {
      filtered = filtered.filter(node => node.metadata.confidence >= options.filterByConfidence!);
    }
    
    return filtered;
  }

  private filterEdges(edges: KnowledgeEdge[], options: ExportOptions): KnowledgeEdge[] {
    return edges; // Simple implementation
  }

  private exportToJSON(nodes: KnowledgeNode[], edges: KnowledgeEdge[], options: ExportOptions): string {
    const data = {
      nodes: options.includeMetadata ? nodes : nodes.map(({ metadata, ...node }) => node),
      edges: edges,
      exportInfo: {
        timestamp: new Date().toISOString(),
        nodeCount: nodes.length,
        edgeCount: edges.length,
        options,
      },
    };
    
    return JSON.stringify(data, null, 2);
  }

  private exportToCSV(nodes: KnowledgeNode[], edges: KnowledgeEdge[], options: ExportOptions): string {
    const nodeHeaders = ['id', 'type', 'properties'];
    if (options.includeMetadata) {
      nodeHeaders.push('created', 'updated', 'confidence');
    }
    
    const edgeHeaders = ['id', 'source', 'target', 'type', 'weight', 'properties'];
    
    let csv = 'NODES\n';
    csv += nodeHeaders.join(',') + '\n';
    
    nodes.forEach(node => {
      const row = [
        node.id,
        node.type,
        JSON.stringify(node.properties).replace(/"/g, '""'),
      ];
      
      if (options.includeMetadata) {
        row.push(
          node.metadata.created.toISOString(),
          node.metadata.updated.toISOString(),
          node.metadata.confidence.toString()
        );
      }
      
      csv += row.join(',') + '\n';
    });
    
    csv += '\nEDGES\n';
    csv += edgeHeaders.join(',') + '\n';
    
    edges.forEach(edge => {
      const row = [
        edge.id,
        edge.source,
        edge.target,
        edge.type,
        edge.weight.toString(),
        JSON.stringify(edge.properties).replace(/"/g, '""'),
      ];
      
      csv += row.join(',') + '\n';
    });
    
    return csv;
  }

  private exportToGEXF(nodes: KnowledgeNode[], edges: KnowledgeEdge[], options: ExportOptions): string {
    let gexf = '<?xml version="1.0" encoding="UTF-8"?>\n';
    gexf += '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n';
    gexf += '  <graph mode="static" defaultedgetype="directed">\n';
    
    // Nodes
    gexf += '    <nodes>\n';
    nodes.forEach(node => {
      gexf += `      <node id="${node.id}" label="${node.type}"/>\n`;
    });
    gexf += '    </nodes>\n';
    
    // Edges
    gexf += '    <edges>\n';
    edges.forEach(edge => {
      gexf += `      <edge id="${edge.id}" source="${edge.source}" target="${edge.target}" weight="${edge.weight}"/>\n`;
    });
    gexf += '    </edges>\n';
    
    gexf += '  </graph>\n';
    gexf += '</gexf>';
    
    return gexf;
  }

  private exportToGraphML(nodes: KnowledgeNode[], edges: KnowledgeEdge[], options: ExportOptions): string {
    let graphml = '<?xml version="1.0" encoding="UTF-8"?>\n';
    graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n';
    graphml += '  <graph id="knowledge-graph" edgedefault="directed">\n';
    
    // Nodes
    nodes.forEach(node => {
      graphml += `    <node id="${node.id}"/>\n`;
    });
    
    // Edges
    edges.forEach(edge => {
      graphml += `    <edge source="${edge.source}" target="${edge.target}"/>\n`;
    });
    
    graphml += '  </graph>\n';
    graphml += '</graphml>';
    
    return graphml;
  }

  terminate(): void {
    if (this.compressionWorker) {
      this.compressionWorker.terminate();
    }
  }
}

// Storage Service with IndexedDB
class AdvancedStorageService {
  private dbName = 'FreeAgenticsDB';
  private version = 1;
  private db?: IDBDatabase;

  async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object stores
        if (!db.objectStoreNames.contains('agents')) {
          const agentsStore = db.createObjectStore('agents', { keyPath: 'id' });
          agentsStore.createIndex('type', 'type', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('conversations')) {
          const conversationsStore = db.createObjectStore('conversations', { keyPath: 'id' });
          conversationsStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('knowledge')) {
          const knowledgeStore = db.createObjectStore('knowledge', { keyPath: 'id' });
          knowledgeStore.createIndex('type', 'type', { unique: false });
        }
      };
    });
  }

  async store(storeName: string, data: any): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.add(data);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async retrieve(storeName: string, id: string): Promise<any> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(id);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
  }

  async retrieveAll(storeName: string): Promise<any[]> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
  }

  async update(storeName: string, data: any): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(data);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async delete(storeName: string, id: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(id);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async clear(storeName: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }
}

// Tests
describe('Specialized Services', () => {
  describe('CompressionWorker', () => {
    let compressionWorker: CompressionWorker;

    beforeEach(() => {
      compressionWorker = new CompressionWorker();
    });

    afterEach(() => {
      compressionWorker.terminate();
    });

    test('should create worker instance', () => {
      expect(compressionWorker).toBeDefined();
      expect(Worker).toHaveBeenCalledWith('/workers/compression.js');
    });

    test('should compress data', async () => {
      const testData = 'Hello, World!';
      const options: CompressionOptions = {
        algorithm: 'gzip',
        level: 6,
        chunkSize: 1024,
      };

      // Mock successful compression
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: new ArrayBuffer(8),
          },
        });
      }, 10);

      const result = await compressionWorker.compress(testData, options);
      expect(result).toBeInstanceOf(ArrayBuffer);
    });

    test('should decompress data', async () => {
      const testData = new ArrayBuffer(8);
      const algorithm = 'gzip';

      // Mock successful decompression
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: 'Hello, World!',
          },
        });
      }, 10);

      const result = await compressionWorker.decompress(testData, algorithm);
      expect(result).toBe('Hello, World!');
    });

    test('should handle compression errors', async () => {
      const testData = 'Hello, World!';
      const options: CompressionOptions = {
        algorithm: 'gzip',
        level: 6,
        chunkSize: 1024,
      };

      // Mock compression error
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            error: 'Compression failed',
          },
        });
      }, 10);

      await expect(compressionWorker.compress(testData, options))
        .rejects.toThrow('Compression failed');
    });

    test('should handle worker errors', () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      const errorEvent = new ErrorEvent('error', { message: 'Worker crashed' });
      
      expect(() => {
        mockWorker.onerror(errorEvent);
      }).not.toThrow();
    });
  });

  describe('ProviderMonitoringService', () => {
    let monitoringService: ProviderMonitoringService;

    beforeEach(() => {
      monitoringService = new ProviderMonitoringService();
    });

    test('should record metrics', () => {
      const metric: ProviderMetrics = {
        responseTime: 500,
        successRate: 0.95,
        errorRate: 0.05,
        throughput: 100,
        availability: 0.99,
        cost: 0.001,
      };

      monitoringService.recordMetric('provider1', metric);
      const metrics = monitoringService.getMetrics('provider1');
      
      expect(metrics).toHaveLength(1);
      expect(metrics[0]).toEqual(metric);
    });

    test('should generate alerts for threshold violations', () => {
      const highLatencyMetric: ProviderMetrics = {
        responseTime: 2000, // Above threshold
        successRate: 0.95,
        errorRate: 0.05,
        throughput: 100,
        availability: 0.99,
        cost: 0.001,
      };

      monitoringService.recordMetric('provider1', highLatencyMetric);
      const alerts = monitoringService.getAlerts('provider1');
      
      expect(alerts).toHaveLength(1);
      expect(alerts[0].severity).toBe('warning');
      expect(alerts[0].message).toContain('High response time');
    });

    test('should calculate aggregated metrics', () => {
      const metrics: ProviderMetrics[] = [
        {
          responseTime: 400,
          successRate: 0.95,
          errorRate: 0.05,
          throughput: 100,
          availability: 0.99,
          cost: 0.001,
        },
        {
          responseTime: 600,
          successRate: 0.98,
          errorRate: 0.02,
          throughput: 120,
          availability: 0.97,
          cost: 0.002,
        },
      ];

      metrics.forEach(metric => {
        monitoringService.recordMetric('provider1', metric);
      });

      const aggregated = monitoringService.getAggregatedMetrics('provider1');
      
      expect(aggregated.responseTime).toBe(500); // Average
      expect(aggregated.successRate).toBeCloseTo(0.965, 3);
      expect(aggregated.cost).toBe(0.003); // Sum
    });

    test('should resolve alerts', () => {
      const metric: ProviderMetrics = {
        responseTime: 2000,
        successRate: 0.95,
        errorRate: 0.1, // High error rate
        throughput: 100,
        availability: 0.90, // Low availability
        cost: 0.001,
      };

      monitoringService.recordMetric('provider1', metric);
      const alerts = monitoringService.getAlerts('provider1');
      
      expect(alerts.length).toBeGreaterThan(0);
      
      const alertId = alerts[0].id;
      const resolved = monitoringService.resolveAlert(alertId);
      
      expect(resolved).toBe(true);
      expect(alerts[0].resolved).toBe(true);
    });

    test('should clear resolved alerts', () => {
      const metric: ProviderMetrics = {
        responseTime: 2000,
        successRate: 0.95,
        errorRate: 0.1,
        throughput: 100,
        availability: 0.90,
        cost: 0.001,
      };

      monitoringService.recordMetric('provider1', metric);
      const alerts = monitoringService.getAlerts();
      
      // Resolve all alerts
      alerts.forEach(alert => {
        monitoringService.resolveAlert(alert.id);
      });

      monitoringService.clearResolvedAlerts();
      const remainingAlerts = monitoringService.getAlerts();
      
      expect(remainingAlerts).toHaveLength(0);
    });
  });

  describe('KnowledgeGraphExportService', () => {
    let exportService: KnowledgeGraphExportService;
    let mockNodes: KnowledgeNode[];
    let mockEdges: KnowledgeEdge[];

    beforeEach(() => {
      exportService = new KnowledgeGraphExportService();
      
      mockNodes = [
        {
          id: 'node1',
          type: 'agent',
          properties: { name: 'Agent 1' },
          metadata: {
            created: new Date('2023-01-01'),
            updated: new Date('2023-01-02'),
            confidence: 0.9,
          },
        },
        {
          id: 'node2',
          type: 'concept',
          properties: { name: 'Concept 1' },
          metadata: {
            created: new Date('2023-01-01'),
            updated: new Date('2023-01-02'),
            confidence: 0.8,
          },
        },
      ];

      mockEdges = [
        {
          id: 'edge1',
          source: 'node1',
          target: 'node2',
          type: 'knows',
          weight: 0.7,
          properties: { strength: 'strong' },
        },
      ];
    });

    afterEach(() => {
      exportService.terminate();
    });

    test('should export to JSON format', async () => {
      const options: ExportOptions = {
        format: 'json',
        includeMetadata: true,
      };

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      
      expect(typeof result).toBe('string');
      const parsed = JSON.parse(result as string);
      
      expect(parsed.nodes).toHaveLength(2);
      expect(parsed.edges).toHaveLength(1);
      expect(parsed.nodes[0].metadata).toBeDefined();
    });

    test('should export to CSV format', async () => {
      const options: ExportOptions = {
        format: 'csv',
        includeMetadata: false,
      };

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      
      expect(typeof result).toBe('string');
      expect(result).toContain('NODES');
      expect(result).toContain('EDGES');
      expect(result).toContain('node1,agent');
    });

    test('should export to GEXF format', async () => {
      const options: ExportOptions = {
        format: 'gexf',
        includeMetadata: false,
      };

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      
      expect(typeof result).toBe('string');
      expect(result).toContain('<?xml version="1.0"');
      expect(result).toContain('<gexf xmlns=');
      expect(result).toContain('<node id="node1"');
    });

    test('should export to GraphML format', async () => {
      const options: ExportOptions = {
        format: 'graphml',
        includeMetadata: false,
      };

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      
      expect(typeof result).toBe('string');
      expect(result).toContain('<?xml version="1.0"');
      expect(result).toContain('<graphml xmlns=');
      expect(result).toContain('<node id="node1"');
    });

    test('should filter by node type', async () => {
      const options: ExportOptions = {
        format: 'json',
        includeMetadata: false,
        filterByType: ['agent'],
      };

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      const parsed = JSON.parse(result as string);
      
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.nodes[0].type).toBe('agent');
    });

    test('should filter by confidence', async () => {
      const options: ExportOptions = {
        format: 'json',
        includeMetadata: true,
        filterByConfidence: 0.85,
      };

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      const parsed = JSON.parse(result as string);
      
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.nodes[0].metadata.confidence).toBeGreaterThanOrEqual(0.85);
    });

    test('should handle compression', async () => {
      const options: ExportOptions = {
        format: 'json',
        includeMetadata: true,
        compressed: true,
      };

      // Mock compression worker
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: new ArrayBuffer(100),
          },
        });
      }, 10);

      const result = await exportService.exportGraph(mockNodes, mockEdges, options);
      
      expect(result).toBeInstanceOf(ArrayBuffer);
    });

    test('should handle unsupported format', async () => {
      const options: ExportOptions = {
        format: 'unsupported' as any,
        includeMetadata: false,
      };

      await expect(
        exportService.exportGraph(mockNodes, mockEdges, options)
      ).rejects.toThrow('Unsupported export format');
    });
  });

  describe('AdvancedStorageService', () => {
    let storageService: AdvancedStorageService;

    beforeEach(async () => {
      storageService = new AdvancedStorageService();
      
      // Mock successful initialization
      const mockRequest = {
        onsuccess: null,
        onerror: null,
        onupgradeneeded: null,
        result: {
          objectStoreNames: {
            contains: jest.fn(() => false),
          },
          createObjectStore: jest.fn(() => ({
            createIndex: jest.fn(),
          })),
          transaction: jest.fn(() => ({
            objectStore: jest.fn(() => ({
              add: jest.fn(() => ({ onsuccess: null, onerror: null })),
              get: jest.fn(() => ({ onsuccess: null, onerror: null })),
              getAll: jest.fn(() => ({ onsuccess: null, onerror: null })),
              put: jest.fn(() => ({ onsuccess: null, onerror: null })),
              delete: jest.fn(() => ({ onsuccess: null, onerror: null })),
              clear: jest.fn(() => ({ onsuccess: null, onerror: null })),
            })),
          })),
        },
      };

      (global.indexedDB.open as jest.Mock).mockReturnValue(mockRequest);
      
      // Simulate successful initialization
      setTimeout(() => {
        if (mockRequest.onsuccess) {
          mockRequest.onsuccess();
        }
      }, 0);

      await storageService.initialize();
    });

    test('should initialize database', async () => {
      expect(global.indexedDB.open).toHaveBeenCalledWith('FreeAgenticsDB', 1);
    });

    test('should store data', async () => {
      const testData = { id: 'test1', name: 'Test Agent' };
      
      // Mock successful store operation
      const mockTransaction = {
        objectStore: jest.fn(() => ({
          add: jest.fn(() => {
            const request = { onsuccess: null, onerror: null };
            setTimeout(() => {
              if (request.onsuccess) request.onsuccess();
            }, 0);
            return request;
          }),
        })),
      };

      (storageService as any).db = {
        transaction: jest.fn(() => mockTransaction),
      };

      await expect(storageService.store('agents', testData)).resolves.toBeUndefined();
    });

    test('should retrieve data', async () => {
      const testId = 'test1';
      const expectedData = { id: 'test1', name: 'Test Agent' };
      
      // Mock successful retrieve operation
      const mockTransaction = {
        objectStore: jest.fn(() => ({
          get: jest.fn(() => {
            const request = { onsuccess: null, onerror: null, result: expectedData };
            setTimeout(() => {
              if (request.onsuccess) request.onsuccess();
            }, 0);
            return request;
          }),
        })),
      };

      (storageService as any).db = {
        transaction: jest.fn(() => mockTransaction),
      };

      const result = await storageService.retrieve('agents', testId);
      expect(result).toEqual(expectedData);
    });

    test('should retrieve all data', async () => {
      const expectedData = [
        { id: 'test1', name: 'Test Agent 1' },
        { id: 'test2', name: 'Test Agent 2' },
      ];
      
      // Mock successful retrieveAll operation
      const mockTransaction = {
        objectStore: jest.fn(() => ({
          getAll: jest.fn(() => {
            const request = { onsuccess: null, onerror: null, result: expectedData };
            setTimeout(() => {
              if (request.onsuccess) request.onsuccess();
            }, 0);
            return request;
          }),
        })),
      };

      (storageService as any).db = {
        transaction: jest.fn(() => mockTransaction),
      };

      const result = await storageService.retrieveAll('agents');
      expect(result).toEqual(expectedData);
    });

    test('should update data', async () => {
      const testData = { id: 'test1', name: 'Updated Agent' };
      
      // Mock successful update operation
      const mockTransaction = {
        objectStore: jest.fn(() => ({
          put: jest.fn(() => {
            const request = { onsuccess: null, onerror: null };
            setTimeout(() => {
              if (request.onsuccess) request.onsuccess();
            }, 0);
            return request;
          }),
        })),
      };

      (storageService as any).db = {
        transaction: jest.fn(() => mockTransaction),
      };

      await expect(storageService.update('agents', testData)).resolves.toBeUndefined();
    });

    test('should delete data', async () => {
      const testId = 'test1';
      
      // Mock successful delete operation
      const mockTransaction = {
        objectStore: jest.fn(() => ({
          delete: jest.fn(() => {
            const request = { onsuccess: null, onerror: null };
            setTimeout(() => {
              if (request.onsuccess) request.onsuccess();
            }, 0);
            return request;
          }),
        })),
      };

      (storageService as any).db = {
        transaction: jest.fn(() => mockTransaction),
      };

      await expect(storageService.delete('agents', testId)).resolves.toBeUndefined();
    });

    test('should clear store', async () => {
      // Mock successful clear operation
      const mockTransaction = {
        objectStore: jest.fn(() => ({
          clear: jest.fn(() => {
            const request = { onsuccess: null, onerror: null };
            setTimeout(() => {
              if (request.onsuccess) request.onsuccess();
            }, 0);
            return request;
          }),
        })),
      };

      (storageService as any).db = {
        transaction: jest.fn(() => mockTransaction),
      };

      await expect(storageService.clear('agents')).resolves.toBeUndefined();
    });

    test('should handle database not initialized error', async () => {
      const uninitializedService = new AdvancedStorageService();
      
      await expect(uninitializedService.store('agents', {}))
        .rejects.toThrow('Database not initialized');
    });
  });
});