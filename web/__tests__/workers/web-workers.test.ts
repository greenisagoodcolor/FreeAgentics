/**
 * Web Workers Tests
 * 
 * Tests for web worker implementations, background processing,
 * and worker communication following ADR-007 requirements.
 */

import { jest } from '@jest/globals';

// Mock Worker API
global.Worker = jest.fn(() => ({
  postMessage: jest.fn(),
  terminate: jest.fn(),
  onmessage: null,
  onerror: null,
  onmessageerror: null,
})) as any;

// Mock SharedWorker API
global.SharedWorker = jest.fn(() => ({
  port: {
    postMessage: jest.fn(),
    onmessage: null,
    onmessageerror: null,
    start: jest.fn(),
    close: jest.fn(),
  },
  onerror: null,
})) as any;

// Mock ServiceWorker API
global.ServiceWorker = jest.fn(() => ({
  postMessage: jest.fn(),
  state: 'activated',
  onstatechange: null,
  onerror: null,
})) as any;

global.navigator = {
  ...global.navigator,
  serviceWorker: {
    register: jest.fn(() => Promise.resolve({
      installing: null,
      waiting: null,
      active: {
        postMessage: jest.fn(),
        state: 'activated',
      },
      scope: '/test-scope/',
      update: jest.fn(),
      unregister: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    })),
    ready: Promise.resolve({
      installing: null,
      waiting: null,
      active: {
        postMessage: jest.fn(),
        state: 'activated',
      },
      scope: '/test-scope/',
      update: jest.fn(),
      unregister: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    }),
    controller: null,
    getRegistration: jest.fn(),
    getRegistrations: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
  },
} as any;

// Data Processing Worker
interface ProcessingTask {
  id: string;
  type: 'filter' | 'map' | 'reduce' | 'sort' | 'aggregate';
  data: any[];
  options?: any;
}

interface ProcessingResult {
  id: string;
  result: any;
  processingTime: number;
  error?: string;
}

class DataProcessingWorker {
  private worker: Worker;
  private pendingTasks: Map<string, { resolve: Function; reject: Function; startTime: number }> = new Map();

  constructor(workerScript: string = '/workers/data-processing.js') {
    this.worker = new Worker(workerScript);
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = this.handleError.bind(this);
  }

  private handleMessage(event: MessageEvent<ProcessingResult>): void {
    const { id, result, processingTime, error } = event.data;
    const task = this.pendingTasks.get(id);
    
    if (task) {
      if (error) {
        task.reject(new Error(error));
      } else {
        task.resolve({ result, processingTime });
      }
      this.pendingTasks.delete(id);
    }
  }

  private handleError(error: ErrorEvent): void {
    console.error('Data processing worker error:', error);
    
    // Reject all pending tasks
    this.pendingTasks.forEach(({ reject }) => {
      reject(new Error('Worker encountered an error'));
    });
    this.pendingTasks.clear();
  }

  async processData(task: ProcessingTask): Promise<{ result: any; processingTime: number }> {
    const taskId = task.id || Math.random().toString(36);
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
      this.pendingTasks.set(taskId, { resolve, reject, startTime });
      
      this.worker.postMessage({
        ...task,
        id: taskId,
      });
      
      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingTasks.has(taskId)) {
          this.pendingTasks.delete(taskId);
          reject(new Error('Task timeout'));
        }
      }, 30000);
    });
  }

  async filterData(data: any[], predicate: string, options?: any): Promise<any[]> {
    const task: ProcessingTask = {
      id: Math.random().toString(36),
      type: 'filter',
      data,
      options: { predicate, ...options },
    };
    
    const result = await this.processData(task);
    return result.result;
  }

  async mapData(data: any[], mapper: string, options?: any): Promise<any[]> {
    const task: ProcessingTask = {
      id: Math.random().toString(36),
      type: 'map',
      data,
      options: { mapper, ...options },
    };
    
    const result = await this.processData(task);
    return result.result;
  }

  async reduceData(data: any[], reducer: string, initialValue?: any): Promise<any> {
    const task: ProcessingTask = {
      id: Math.random().toString(36),
      type: 'reduce',
      data,
      options: { reducer, initialValue },
    };
    
    const result = await this.processData(task);
    return result.result;
  }

  async sortData(data: any[], compareFn?: string): Promise<any[]> {
    const task: ProcessingTask = {
      id: Math.random().toString(36),
      type: 'sort',
      data,
      options: { compareFn },
    };
    
    const result = await this.processData(task);
    return result.result;
  }

  async aggregateData(data: any[], aggregations: Record<string, string>): Promise<any> {
    const task: ProcessingTask = {
      id: Math.random().toString(36),
      type: 'aggregate',
      data,
      options: { aggregations },
    };
    
    const result = await this.processData(task);
    return result.result;
  }

  getQueueSize(): number {
    return this.pendingTasks.size;
  }

  terminate(): void {
    this.worker.terminate();
    this.pendingTasks.clear();
  }
}

// Image Processing Worker
interface ImageProcessingTask {
  id: string;
  type: 'resize' | 'filter' | 'crop' | 'rotate' | 'brightness' | 'contrast';
  imageData: ImageData | string; // Base64 or ImageData
  options: any;
}

class ImageProcessingWorker {
  private worker: Worker;
  private pendingTasks: Map<string, { resolve: Function; reject: Function }> = new Map();

  constructor() {
    this.worker = new Worker('/workers/image-processing.js');
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = this.handleError.bind(this);
  }

  private handleMessage(event: MessageEvent): void {
    const { id, result, error } = event.data;
    const task = this.pendingTasks.get(id);
    
    if (task) {
      if (error) {
        task.reject(new Error(error));
      } else {
        task.resolve(result);
      }
      this.pendingTasks.delete(id);
    }
  }

  private handleError(error: ErrorEvent): void {
    console.error('Image processing worker error:', error);
    
    this.pendingTasks.forEach(({ reject }) => {
      reject(new Error('Worker encountered an error'));
    });
    this.pendingTasks.clear();
  }

  async processImage(task: ImageProcessingTask): Promise<ImageData | string> {
    const taskId = task.id || Math.random().toString(36);
    
    return new Promise((resolve, reject) => {
      this.pendingTasks.set(taskId, { resolve, reject });
      
      this.worker.postMessage({
        ...task,
        id: taskId,
      });
      
      // Timeout after 60 seconds for image processing
      setTimeout(() => {
        if (this.pendingTasks.has(taskId)) {
          this.pendingTasks.delete(taskId);
          reject(new Error('Image processing timeout'));
        }
      }, 60000);
    });
  }

  async resizeImage(imageData: ImageData, width: number, height: number): Promise<ImageData> {
    const result = await this.processImage({
      id: Math.random().toString(36),
      type: 'resize',
      imageData,
      options: { width, height },
    });
    
    return result as ImageData;
  }

  async applyFilter(imageData: ImageData, filterType: string, intensity: number = 1): Promise<ImageData> {
    const result = await this.processImage({
      id: Math.random().toString(36),
      type: 'filter',
      imageData,
      options: { filterType, intensity },
    });
    
    return result as ImageData;
  }

  async cropImage(imageData: ImageData, x: number, y: number, width: number, height: number): Promise<ImageData> {
    const result = await this.processImage({
      id: Math.random().toString(36),
      type: 'crop',
      imageData,
      options: { x, y, width, height },
    });
    
    return result as ImageData;
  }

  async rotateImage(imageData: ImageData, angle: number): Promise<ImageData> {
    const result = await this.processImage({
      id: Math.random().toString(36),
      type: 'rotate',
      imageData,
      options: { angle },
    });
    
    return result as ImageData;
  }

  async adjustBrightness(imageData: ImageData, brightness: number): Promise<ImageData> {
    const result = await this.processImage({
      id: Math.random().toString(36),
      type: 'brightness',
      imageData,
      options: { brightness },
    });
    
    return result as ImageData;
  }

  async adjustContrast(imageData: ImageData, contrast: number): Promise<ImageData> {
    const result = await this.processImage({
      id: Math.random().toString(36),
      type: 'contrast',
      imageData,
      options: { contrast },
    });
    
    return result as ImageData;
  }

  terminate(): void {
    this.worker.terminate();
    this.pendingTasks.clear();
  }
}

// Shared Worker for Cross-Tab Communication
interface TabMessage {
  type: 'sync' | 'broadcast' | 'request' | 'response';
  data: any;
  tabId?: string;
  timestamp: number;
}

class CrossTabCommunicator {
  private sharedWorker: SharedWorker;
  private port: MessagePort;
  private tabId: string;
  private messageHandlers: Map<string, (data: any) => void> = new Map();

  constructor() {
    this.tabId = Math.random().toString(36);
    this.sharedWorker = new SharedWorker('/workers/cross-tab.js');
    this.port = this.sharedWorker.port;
    
    this.port.onmessage = this.handleMessage.bind(this);
    this.port.onmessageerror = this.handleMessageError.bind(this);
    this.port.start();
    
    // Register this tab
    this.sendMessage({
      type: 'sync',
      data: { action: 'register', tabId: this.tabId },
      timestamp: Date.now(),
    });
  }

  private handleMessage(event: MessageEvent<TabMessage>): void {
    const { type, data, tabId, timestamp } = event.data;
    
    // Don't handle our own messages
    if (tabId === this.tabId) return;
    
    const handler = this.messageHandlers.get(type);
    if (handler) {
      handler(data);
    }
    
    // Emit general message event
    const generalHandler = this.messageHandlers.get('*');
    if (generalHandler) {
      generalHandler({ type, data, tabId, timestamp });
    }
  }

  private handleMessageError(error: MessageEvent): void {
    console.error('Cross-tab communication error:', error);
  }

  private sendMessage(message: TabMessage): void {
    this.port.postMessage({
      ...message,
      tabId: this.tabId,
    });
  }

  broadcast(data: any): void {
    this.sendMessage({
      type: 'broadcast',
      data,
      timestamp: Date.now(),
    });
  }

  sync(data: any): void {
    this.sendMessage({
      type: 'sync',
      data,
      timestamp: Date.now(),
    });
  }

  request(data: any): Promise<any> {
    const requestId = Math.random().toString(36);
    
    return new Promise((resolve, reject) => {
      const responseHandler = (responseData: any) => {
        if (responseData.requestId === requestId) {
          this.off('response', responseHandler);
          resolve(responseData.data);
        }
      };
      
      this.on('response', responseHandler);
      
      this.sendMessage({
        type: 'request',
        data: { ...data, requestId },
        timestamp: Date.now(),
      });
      
      // Timeout after 10 seconds
      setTimeout(() => {
        this.off('response', responseHandler);
        reject(new Error('Request timeout'));
      }, 10000);
    });
  }

  respond(requestId: string, data: any): void {
    this.sendMessage({
      type: 'response',
      data: { requestId, data },
      timestamp: Date.now(),
    });
  }

  on(messageType: string, handler: (data: any) => void): void {
    this.messageHandlers.set(messageType, handler);
  }

  off(messageType: string, handler?: (data: any) => void): void {
    if (handler) {
      const currentHandler = this.messageHandlers.get(messageType);
      if (currentHandler === handler) {
        this.messageHandlers.delete(messageType);
      }
    } else {
      this.messageHandlers.delete(messageType);
    }
  }

  disconnect(): void {
    this.sendMessage({
      type: 'sync',
      data: { action: 'unregister', tabId: this.tabId },
      timestamp: Date.now(),
    });
    
    this.port.close();
    this.messageHandlers.clear();
  }
}

// Service Worker Manager
interface CacheConfig {
  name: string;
  maxAge: number;
  maxEntries: number;
  strategy: 'cache-first' | 'network-first' | 'stale-while-revalidate';
}

class ServiceWorkerManager {
  private registration?: ServiceWorkerRegistration;
  private isRegistered: boolean = false;

  async register(scriptUrl: string = '/sw.js'): Promise<ServiceWorkerRegistration> {
    if (!('serviceWorker' in navigator)) {
      throw new Error('Service Worker not supported');
    }

    try {
      this.registration = await navigator.serviceWorker.register(scriptUrl);
      this.isRegistered = true;
      
      this.registration.addEventListener('updatefound', this.handleUpdateFound.bind(this));
      
      return this.registration;
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      throw error;
    }
  }

  private handleUpdateFound(): void {
    if (!this.registration) return;
    
    const newWorker = this.registration.installing;
    if (newWorker) {
      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          console.log('New Service Worker available');
          // Could trigger update notification to user
        }
      });
    }
  }

  async unregister(): Promise<boolean> {
    if (!this.registration) {
      return false;
    }

    try {
      const result = await this.registration.unregister();
      this.isRegistered = false;
      return result;
    } catch (error) {
      console.error('Service Worker unregistration failed:', error);
      return false;
    }
  }

  async update(): Promise<void> {
    if (!this.registration) {
      throw new Error('Service Worker not registered');
    }

    await this.registration.update();
  }

  postMessage(message: any): void {
    if (!this.registration?.active) {
      throw new Error('No active Service Worker');
    }

    this.registration.active.postMessage(message);
  }

  async configureCache(configs: CacheConfig[]): Promise<void> {
    this.postMessage({
      type: 'configure-cache',
      configs,
    });
  }

  async clearCache(cacheName?: string): Promise<void> {
    this.postMessage({
      type: 'clear-cache',
      cacheName,
    });
  }

  async getCacheStats(): Promise<any> {
    return new Promise((resolve, reject) => {
      const channel = new MessageChannel();
      
      channel.port1.onmessage = (event) => {
        if (event.data.error) {
          reject(new Error(event.data.error));
        } else {
          resolve(event.data);
        }
      };
      
      this.postMessage({
        type: 'get-cache-stats',
        port: channel.port2,
      });
      
      setTimeout(() => {
        reject(new Error('Cache stats request timeout'));
      }, 5000);
    });
  }

  getRegistration(): ServiceWorkerRegistration | undefined {
    return this.registration;
  }

  isServiceWorkerRegistered(): boolean {
    return this.isRegistered;
  }
}

// Background Sync Manager
interface SyncTask {
  id: string;
  type: string;
  data: any;
  retryCount: number;
  maxRetries: number;
  createdAt: number;
}

class BackgroundSyncManager {
  private tasks: Map<string, SyncTask> = new Map();
  private serviceWorkerManager: ServiceWorkerManager;

  constructor(serviceWorkerManager: ServiceWorkerManager) {
    this.serviceWorkerManager = serviceWorkerManager;
  }

  async scheduleSync(type: string, data: any, options: { maxRetries?: number } = {}): Promise<string> {
    const taskId = Math.random().toString(36);
    const task: SyncTask = {
      id: taskId,
      type,
      data,
      retryCount: 0,
      maxRetries: options.maxRetries || 3,
      createdAt: Date.now(),
    };

    this.tasks.set(taskId, task);

    // Send to service worker for background processing
    this.serviceWorkerManager.postMessage({
      type: 'schedule-sync',
      task,
    });

    return taskId;
  }

  async cancelSync(taskId: string): Promise<boolean> {
    const task = this.tasks.get(taskId);
    if (!task) return false;

    this.tasks.delete(taskId);
    
    this.serviceWorkerManager.postMessage({
      type: 'cancel-sync',
      taskId,
    });

    return true;
  }

  async retryFailedTasks(): Promise<void> {
    const failedTasks = Array.from(this.tasks.values())
      .filter(task => task.retryCount < task.maxRetries);

    for (const task of failedTasks) {
      task.retryCount++;
      
      this.serviceWorkerManager.postMessage({
        type: 'retry-sync',
        task,
      });
    }
  }

  getPendingTasks(): SyncTask[] {
    return Array.from(this.tasks.values());
  }

  getTaskStatus(taskId: string): SyncTask | undefined {
    return this.tasks.get(taskId);
  }

  clearCompletedTasks(): void {
    // This would typically be called after receiving success notifications
    // from the service worker
    const completedTasks = Array.from(this.tasks.entries())
      .filter(([_, task]) => task.retryCount >= task.maxRetries);

    completedTasks.forEach(([taskId]) => {
      this.tasks.delete(taskId);
    });
  }
}

// Tests
describe('Web Workers', () => {
  describe('DataProcessingWorker', () => {
    let worker: DataProcessingWorker;

    beforeEach(() => {
      worker = new DataProcessingWorker();
    });

    afterEach(() => {
      worker.terminate();
    });

    test('should create worker instance', () => {
      expect(Worker).toHaveBeenCalledWith('/workers/data-processing.js');
      expect(worker.getQueueSize()).toBe(0);
    });

    test('should process data with result', async () => {
      const testData = [1, 2, 3, 4, 5];
      const task: ProcessingTask = {
        id: 'test1',
        type: 'filter',
        data: testData,
        options: { predicate: 'x => x > 2' },
      };

      // Mock worker response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: 'test1',
            result: [3, 4, 5],
            processingTime: 100,
          },
        });
      }, 10);

      const result = await worker.processData(task);
      expect(result.result).toEqual([3, 4, 5]);
      expect(result.processingTime).toBe(100);
    });

    test('should handle worker errors', async () => {
      const testData = [1, 2, 3];
      const task: ProcessingTask = {
        id: 'test2',
        type: 'map',
        data: testData,
      };

      // Mock worker error response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: 'test2',
            error: 'Processing failed',
          },
        });
      }, 10);

      await expect(worker.processData(task)).rejects.toThrow('Processing failed');
    });

    test('should filter data', async () => {
      const testData = [1, 2, 3, 4, 5];

      // Mock worker response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: [4, 5],
            processingTime: 50,
          },
        });
      }, 10);

      const result = await worker.filterData(testData, 'x => x > 3');
      expect(result).toEqual([4, 5]);
    });

    test('should map data', async () => {
      const testData = [1, 2, 3];

      // Mock worker response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: [2, 4, 6],
            processingTime: 30,
          },
        });
      }, 10);

      const result = await worker.mapData(testData, 'x => x * 2');
      expect(result).toEqual([2, 4, 6]);
    });

    test('should reduce data', async () => {
      const testData = [1, 2, 3, 4];

      // Mock worker response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: 10,
            processingTime: 20,
          },
        });
      }, 10);

      const result = await worker.reduceData(testData, '(acc, val) => acc + val', 0);
      expect(result).toBe(10);
    });

    test('should sort data', async () => {
      const testData = [3, 1, 4, 1, 5];

      // Mock worker response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: [1, 1, 3, 4, 5],
            processingTime: 25,
          },
        });
      }, 10);

      const result = await worker.sortData(testData);
      expect(result).toEqual([1, 1, 3, 4, 5]);
    });

    test('should aggregate data', async () => {
      const testData = [
        { category: 'A', value: 10 },
        { category: 'B', value: 20 },
        { category: 'A', value: 15 },
      ];

      // Mock worker response
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: { A: 25, B: 20 },
            processingTime: 40,
          },
        });
      }, 10);

      const result = await worker.aggregateData(testData, { sum: 'value' });
      expect(result).toEqual({ A: 25, B: 20 });
    });

    test('should handle task timeout', async () => {
      const testData = [1, 2, 3];
      const task: ProcessingTask = {
        id: 'timeout-test',
        type: 'filter',
        data: testData,
      };

      // Don't mock any response to trigger timeout

      await expect(worker.processData(task)).rejects.toThrow('Task timeout');
    }, 35000);

    test('should handle queue size tracking', () => {
      expect(worker.getQueueSize()).toBe(0);
      
      // Start a task (won't complete without mock response)
      worker.processData({
        id: 'queue-test',
        type: 'map',
        data: [1, 2, 3],
      }).catch(() => {}); // Ignore the error

      expect(worker.getQueueSize()).toBe(1);
    });
  });

  describe('ImageProcessingWorker', () => {
    let worker: ImageProcessingWorker;
    let mockImageData: ImageData;

    beforeEach(() => {
      worker = new ImageProcessingWorker();
      
      // Mock ImageData
      mockImageData = {
        data: new Uint8ClampedArray([255, 0, 0, 255]), // Red pixel
        width: 1,
        height: 1,
        colorSpace: 'srgb',
      } as ImageData;
    });

    afterEach(() => {
      worker.terminate();
    });

    test('should create image processing worker', () => {
      expect(Worker).toHaveBeenCalledWith('/workers/image-processing.js');
    });

    test('should resize image', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: {
              data: new Uint8ClampedArray([255, 0, 0, 255, 255, 0, 0, 255]),
              width: 2,
              height: 1,
            },
          },
        });
      }, 10);

      const result = await worker.resizeImage(mockImageData, 2, 1);
      expect(result.width).toBe(2);
      expect(result.height).toBe(1);
    });

    test('should apply filter to image', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: {
              data: new Uint8ClampedArray([128, 128, 128, 255]), // Grayscale
              width: 1,
              height: 1,
            },
          },
        });
      }, 10);

      const result = await worker.applyFilter(mockImageData, 'grayscale', 1);
      expect(result.data[0]).toBe(128); // Should be grayscale
    });

    test('should crop image', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: mockImageData,
          },
        });
      }, 10);

      const result = await worker.cropImage(mockImageData, 0, 0, 1, 1);
      expect(result).toBeDefined();
    });

    test('should rotate image', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: mockImageData,
          },
        });
      }, 10);

      const result = await worker.rotateImage(mockImageData, 90);
      expect(result).toBeDefined();
    });

    test('should adjust brightness', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: {
              data: new Uint8ClampedArray([255, 100, 100, 255]), // Brighter
              width: 1,
              height: 1,
            },
          },
        });
      }, 10);

      const result = await worker.adjustBrightness(mockImageData, 1.5);
      expect(result.data[1]).toBe(100); // Modified brightness
    });

    test('should adjust contrast', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            result: {
              data: new Uint8ClampedArray([255, 50, 50, 255]), // Higher contrast
              width: 1,
              height: 1,
            },
          },
        });
      }, 10);

      const result = await worker.adjustContrast(mockImageData, 2);
      expect(result.data[1]).toBe(50); // Modified contrast
    });

    test('should handle processing errors', async () => {
      const mockWorker = (Worker as jest.Mock).mock.results[0].value;
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            id: expect.any(String),
            error: 'Invalid image format',
          },
        });
      }, 10);

      await expect(worker.resizeImage(mockImageData, 0, 0))
        .rejects.toThrow('Invalid image format');
    });

    test('should handle processing timeout', async () => {
      // Don't mock any response to trigger timeout
      await expect(worker.resizeImage(mockImageData, 100, 100))
        .rejects.toThrow('Image processing timeout');
    }, 65000);
  });

  describe('CrossTabCommunicator', () => {
    let communicator: CrossTabCommunicator;

    beforeEach(() => {
      communicator = new CrossTabCommunicator();
    });

    afterEach(() => {
      communicator.disconnect();
    });

    test('should create shared worker for cross-tab communication', () => {
      expect(SharedWorker).toHaveBeenCalledWith('/workers/cross-tab.js');
    });

    test('should broadcast messages', () => {
      const testData = { message: 'Hello from tab!' };
      
      communicator.broadcast(testData);
      
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'broadcast',
          data: testData,
          tabId: expect.any(String),
          timestamp: expect.any(Number),
        })
      );
    });

    test('should sync data', () => {
      const syncData = { state: 'updated' };
      
      communicator.sync(syncData);
      
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'sync',
          data: syncData,
        })
      );
    });

    test('should handle request-response pattern', async () => {
      const requestData = { query: 'getUserData' };
      const responseData = { user: { id: 1, name: 'Test User' } };

      // Mock response
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      setTimeout(() => {
        communicator.on('response', (data) => {
          if (data.requestId) {
            // Simulate response from another tab
          }
        });
        
        // Simulate receiving response
        if (mockPort.onmessage) {
          mockPort.onmessage({
            data: {
              type: 'response',
              data: { requestId: expect.any(String), data: responseData },
              tabId: 'other-tab',
              timestamp: Date.now(),
            },
          });
        }
      }, 10);

      const result = await communicator.request(requestData);
      expect(result).toEqual(responseData);
    });

    test('should handle message handlers', () => {
      const handler = jest.fn();
      
      communicator.on('broadcast', handler);
      
      // Simulate receiving message
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      if (mockPort.onmessage) {
        mockPort.onmessage({
          data: {
            type: 'broadcast',
            data: { test: 'data' },
            tabId: 'other-tab',
            timestamp: Date.now(),
          },
        });
      }

      expect(handler).toHaveBeenCalledWith({ test: 'data' });
    });

    test('should remove message handlers', () => {
      const handler = jest.fn();
      
      communicator.on('broadcast', handler);
      communicator.off('broadcast', handler);
      
      // Simulate receiving message
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      if (mockPort.onmessage) {
        mockPort.onmessage({
          data: {
            type: 'broadcast',
            data: { test: 'data' },
            tabId: 'other-tab',
            timestamp: Date.now(),
          },
        });
      }

      expect(handler).not.toHaveBeenCalled();
    });

    test('should ignore own messages', () => {
      const handler = jest.fn();
      communicator.on('broadcast', handler);
      
      // Simulate receiving our own message
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      const tabId = (communicator as any).tabId;
      
      if (mockPort.onmessage) {
        mockPort.onmessage({
          data: {
            type: 'broadcast',
            data: { test: 'data' },
            tabId: tabId,
            timestamp: Date.now(),
          },
        });
      }

      expect(handler).not.toHaveBeenCalled();
    });

    test('should respond to requests', () => {
      const requestId = 'test-request-123';
      const responseData = { result: 'success' };
      
      communicator.respond(requestId, responseData);
      
      const mockPort = (SharedWorker as jest.Mock).mock.results[0].value.port;
      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'response',
          data: { requestId, data: responseData },
        })
      );
    });
  });

  describe('ServiceWorkerManager', () => {
    let manager: ServiceWorkerManager;

    beforeEach(() => {
      manager = new ServiceWorkerManager();
    });

    test('should register service worker', async () => {
      const registration = await manager.register('/test-sw.js');
      
      expect(navigator.serviceWorker.register).toHaveBeenCalledWith('/test-sw.js');
      expect(registration).toBeDefined();
      expect(manager.isServiceWorkerRegistered()).toBe(true);
    });

    test('should unregister service worker', async () => {
      await manager.register();
      const result = await manager.unregister();
      
      expect(result).toBe(true);
      expect(manager.isServiceWorkerRegistered()).toBe(false);
    });

    test('should update service worker', async () => {
      await manager.register();
      await manager.update();
      
      const registration = manager.getRegistration();
      expect(registration?.update).toHaveBeenCalled();
    });

    test('should post message to service worker', async () => {
      await manager.register();
      const message = { type: 'test', data: 'hello' };
      
      manager.postMessage(message);
      
      const registration = manager.getRegistration();
      expect(registration?.active?.postMessage).toHaveBeenCalledWith(message);
    });

    test('should configure cache', async () => {
      await manager.register();
      const configs: CacheConfig[] = [
        {
          name: 'api-cache',
          maxAge: 3600000,
          maxEntries: 100,
          strategy: 'network-first',
        },
      ];
      
      await manager.configureCache(configs);
      
      const registration = manager.getRegistration();
      expect(registration?.active?.postMessage).toHaveBeenCalledWith({
        type: 'configure-cache',
        configs,
      });
    });

    test('should clear cache', async () => {
      await manager.register();
      
      await manager.clearCache('test-cache');
      
      const registration = manager.getRegistration();
      expect(registration?.active?.postMessage).toHaveBeenCalledWith({
        type: 'clear-cache',
        cacheName: 'test-cache',
      });
    });

    test('should get cache stats', async () => {
      await manager.register();
      
      // This would timeout in real implementation without proper response
      await expect(manager.getCacheStats()).rejects.toThrow('Cache stats request timeout');
    });

    test('should throw error when service worker not supported', async () => {
      // Mock unsupported environment
      const originalServiceWorker = (global.navigator as any).serviceWorker;
      delete (global.navigator as any).serviceWorker;
      
      const unsupportedManager = new ServiceWorkerManager();
      
      await expect(unsupportedManager.register()).rejects.toThrow('Service Worker not supported');
      
      // Restore
      (global.navigator as any).serviceWorker = originalServiceWorker;
    });

    test('should throw error when posting message without active worker', async () => {
      expect(() => {
        manager.postMessage({ test: 'data' });
      }).toThrow('No active Service Worker');
    });
  });

  describe('BackgroundSyncManager', () => {
    let syncManager: BackgroundSyncManager;
    let serviceWorkerManager: ServiceWorkerManager;

    beforeEach(async () => {
      serviceWorkerManager = new ServiceWorkerManager();
      await serviceWorkerManager.register();
      syncManager = new BackgroundSyncManager(serviceWorkerManager);
    });

    test('should schedule sync task', async () => {
      const taskData = { action: 'uploadFile', fileId: '123' };
      
      const taskId = await syncManager.scheduleSync('file-upload', taskData);
      
      expect(taskId).toBeDefined();
      expect(syncManager.getPendingTasks()).toHaveLength(1);
      expect(serviceWorkerManager.getRegistration()?.active?.postMessage)
        .toHaveBeenCalledWith(expect.objectContaining({
          type: 'schedule-sync',
          task: expect.objectContaining({
            id: taskId,
            type: 'file-upload',
            data: taskData,
          }),
        }));
    });

    test('should cancel sync task', async () => {
      const taskId = await syncManager.scheduleSync('test-sync', { data: 'test' });
      
      const cancelled = await syncManager.cancelSync(taskId);
      
      expect(cancelled).toBe(true);
      expect(syncManager.getPendingTasks()).toHaveLength(0);
      expect(serviceWorkerManager.getRegistration()?.active?.postMessage)
        .toHaveBeenCalledWith(expect.objectContaining({
          type: 'cancel-sync',
          taskId,
        }));
    });

    test('should retry failed tasks', async () => {
      const taskId = await syncManager.scheduleSync('test-sync', { data: 'test' });
      
      await syncManager.retryFailedTasks();
      
      const task = syncManager.getTaskStatus(taskId);
      expect(task?.retryCount).toBe(1);
      expect(serviceWorkerManager.getRegistration()?.active?.postMessage)
        .toHaveBeenCalledWith(expect.objectContaining({
          type: 'retry-sync',
          task: expect.objectContaining({
            id: taskId,
            retryCount: 1,
          }),
        }));
    });

    test('should get task status', async () => {
      const taskId = await syncManager.scheduleSync('test-sync', { data: 'test' });
      
      const status = syncManager.getTaskStatus(taskId);
      
      expect(status).toBeDefined();
      expect(status?.id).toBe(taskId);
      expect(status?.type).toBe('test-sync');
    });

    test('should clear completed tasks', async () => {
      const taskId = await syncManager.scheduleSync('test-sync', { data: 'test' }, { maxRetries: 0 });
      
      // Simulate task failure by setting retry count to max
      const task = syncManager.getTaskStatus(taskId);
      if (task) {
        task.retryCount = task.maxRetries;
      }
      
      syncManager.clearCompletedTasks();
      
      expect(syncManager.getTaskStatus(taskId)).toBeUndefined();
    });

    test('should handle task with custom max retries', async () => {
      const taskId = await syncManager.scheduleSync('test-sync', { data: 'test' }, { maxRetries: 5 });
      
      const task = syncManager.getTaskStatus(taskId);
      expect(task?.maxRetries).toBe(5);
    });

    test('should not cancel non-existent task', async () => {
      const cancelled = await syncManager.cancelSync('non-existent-id');
      
      expect(cancelled).toBe(false);
    });
  });
});