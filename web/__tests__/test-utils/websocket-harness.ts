import { vi } from "vitest";
import { act } from "@testing-library/react";

export class WebSocketTestHarness {
  private mockSocket: any;
  private eventHandlers: Map<string, Set<Function>> = new Map();
  private messageQueue: any[] = [];

  constructor() {
    this.mockSocket = {
      readyState: WebSocket.CONNECTING,
      send: vi.fn((data: any) => this.messageQueue.push(data)),
      close: vi.fn(() => {
        this.mockSocket.readyState = WebSocket.CLOSED;
        this.emit("close", { code: 1000 });
      }),
      addEventListener: (event: string, handler: Function) => {
        if (!this.eventHandlers.has(event)) {
          this.eventHandlers.set(event, new Set());
        }
        this.eventHandlers.get(event)!.add(handler);
      },
      removeEventListener: (event: string, handler: Function) => {
        this.eventHandlers.get(event)?.delete(handler);
      },
    };
  }

  async connect(): Promise<void> {
    await act(async () => {
      this.mockSocket.readyState = WebSocket.OPEN;
      this.emit("open", {});
    });
  }

  async disconnect(code: number = 1000): Promise<void> {
    await act(async () => {
      this.mockSocket.readyState = WebSocket.CLOSED;
      this.emit("close", { code });
    });
  }

  async receiveMessage(data: any): Promise<void> {
    await act(async () => {
      this.emit("message", { data: JSON.stringify(data) });
    });
  }

  private emit(event: string, data: any): void {
    this.eventHandlers.get(event)?.forEach((handler) => handler(data));
  }

  getSocket(): any {
    return this.mockSocket;
  }

  getSentMessages(): any[] {
    return [...this.messageQueue];
  }
}
