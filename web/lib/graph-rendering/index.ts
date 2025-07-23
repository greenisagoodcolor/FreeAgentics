/**
 * Graph rendering utilities and managers
 */

import {
  GraphData,
  GraphRenderingOptions,
  GraphRenderingManager as IGraphRenderingManager,
  RenderingConfig,
  GraphNode,
  GraphEdge,
} from "./types";

/**
 * Default rendering configuration
 */
const DEFAULT_CONFIG: RenderingConfig = {
  width: 800,
  height: 600,
  nodeRadius: 8,
  linkDistance: 100,
  chargeStrength: -300,
  alphaDecay: 0.0228,
  backgroundColor: "#ffffff",
  showLabels: true,
  enableZoom: true,
  enablePan: true,
};

/**
 * Simple graph rendering manager implementation
 * This is a basic implementation that can be extended with D3.js or other libraries
 */
class GraphRenderingManager implements IGraphRenderingManager {
  private container: HTMLElement;
  private config: RenderingConfig;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private data: GraphData = { nodes: [], edges: [] };
  private options: GraphRenderingOptions;

  constructor(options: GraphRenderingOptions) {
    this.options = options;
    this.config = { ...DEFAULT_CONFIG, ...options.config };

    // Get container element
    if (typeof options.container === "string") {
      const element = document.querySelector(options.container);
      if (!element) {
        throw new Error(`Container element not found: ${options.container}`);
      }
      this.container = element as HTMLElement;
    } else {
      this.container = options.container;
    }

    this.initializeCanvas();
  }

  private initializeCanvas(): void {
    // Create canvas element
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.config.width;
    this.canvas.height = this.config.height;
    this.canvas.style.backgroundColor = this.config.backgroundColor || "#ffffff";

    // Get 2D context
    this.ctx = this.canvas.getContext("2d");
    if (!this.ctx) {
      throw new Error("Failed to get 2D canvas context");
    }

    // Add to container
    this.container.appendChild(this.canvas);

    // Add event listeners for interactions if enabled
    if (this.config.enableZoom || this.config.enablePan) {
      this.addEventListeners();
    }
  }

  private addEventListeners(): void {
    if (!this.canvas) return;

    // Basic click handling for nodes
    this.canvas.addEventListener("click", (event) => {
      const rect = this.canvas!.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Find clicked node
      const clickedNode = this.findNodeAt(x, y);
      if (clickedNode && this.options.onNodeClick) {
        this.options.onNodeClick(clickedNode);
      }
    });

    // Basic hover handling
    this.canvas.addEventListener("mousemove", (event) => {
      const rect = this.canvas!.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const hoveredNode = this.findNodeAt(x, y);
      if (this.options.onNodeHover) {
        this.options.onNodeHover(hoveredNode);
      }
    });
  }

  private findNodeAt(x: number, y: number): GraphNode | null {
    for (const node of this.data.nodes) {
      if (node.x !== undefined && node.y !== undefined) {
        const distance = Math.sqrt(Math.pow(x - node.x, 2) + Math.pow(y - node.y, 2));
        if (distance <= (this.config.nodeRadius || 8)) {
          return node;
        }
      }
    }
    return null;
  }

  render(data: GraphData): void {
    this.data = data;
    this.layoutNodes();
    this.draw();
  }

  update(data: GraphData): void {
    this.data = data;
    this.draw();
  }

  private layoutNodes(): void {
    // Simple circular layout if nodes don't have positions
    const centerX = this.config.width / 2;
    const centerY = this.config.height / 2;
    const radius = Math.min(centerX, centerY) * 0.7;

    this.data.nodes.forEach((node, index) => {
      if (node.x === undefined || node.y === undefined) {
        const angle = (2 * Math.PI * index) / this.data.nodes.length;
        node.x = centerX + radius * Math.cos(angle);
        node.y = centerY + radius * Math.sin(angle);
      }
    });
  }

  private draw(): void {
    if (!this.ctx || !this.canvas) return;

    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw edges
    this.ctx.strokeStyle = "#999";
    this.ctx.lineWidth = 1;

    for (const edge of this.data.edges) {
      const sourceNode =
        typeof edge.source === "string"
          ? this.data.nodes.find((n) => n.id === edge.source)
          : edge.source;
      const targetNode =
        typeof edge.target === "string"
          ? this.data.nodes.find((n) => n.id === edge.target)
          : edge.target;

      if (
        sourceNode &&
        targetNode &&
        sourceNode.x !== undefined &&
        sourceNode.y !== undefined &&
        targetNode.x !== undefined &&
        targetNode.y !== undefined
      ) {
        this.ctx.beginPath();
        this.ctx.moveTo(sourceNode.x, sourceNode.y);
        this.ctx.lineTo(targetNode.x, targetNode.y);
        this.ctx.stroke();
      }
    }

    // Draw nodes
    for (const node of this.data.nodes) {
      if (node.x !== undefined && node.y !== undefined) {
        this.ctx.fillStyle = node.color || "#3498db";
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.size || this.config.nodeRadius || 8, 0, 2 * Math.PI);
        this.ctx.fill();

        // Draw labels if enabled
        if (this.config.showLabels && node.label) {
          this.ctx.fillStyle = "#333";
          this.ctx.font = "12px Arial";
          this.ctx.textAlign = "center";
          this.ctx.fillText(
            node.label,
            node.x,
            node.y + (node.size || this.config.nodeRadius || 8) + 15,
          );
        }
      }
    }
  }

  resize(width: number, height: number): void {
    if (this.canvas) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.config.width = width;
      this.config.height = height;
      this.draw();
    }
  }

  destroy(): void {
    if (this.canvas && this.container.contains(this.canvas)) {
      this.container.removeChild(this.canvas);
    }
    this.canvas = null;
    this.ctx = null;
  }

  zoomToFit(): void {
    // Simple implementation - would be enhanced with proper zoom/pan
    this.layoutNodes();
    this.draw();
  }

  centerNode(nodeId: string): void {
    const node = this.data.nodes.find((n) => n.id === nodeId);
    if (node) {
      node.x = this.config.width / 2;
      node.y = this.config.height / 2;
      this.draw();
    }
  }

  highlightNode(nodeId: string, highlight: boolean): void {
    const node = this.data.nodes.find((n) => n.id === nodeId);
    if (node) {
      node.color = highlight ? "#e74c3c" : "#3498db";
      this.draw();
    }
  }

  getConfig(): RenderingConfig {
    return { ...this.config };
  }

  setConfig(config: Partial<RenderingConfig>): void {
    this.config = { ...this.config, ...config };
    if (this.canvas) {
      if (config.width) this.canvas.width = config.width;
      if (config.height) this.canvas.height = config.height;
      if (config.backgroundColor) this.canvas.style.backgroundColor = config.backgroundColor;
    }
    this.draw();
  }
}

/**
 * Factory function to create a graph renderer
 */
export function createGraphRenderer(options: GraphRenderingOptions): GraphRenderingManager {
  return new GraphRenderingManager(options);
}

/**
 * Re-export types
 */
export type {
  GraphData,
  GraphNode,
  GraphEdge,
  RenderingConfig,
  GraphRenderingOptions,
  GraphRenderingManager as IGraphRenderingManager,
} from "./types";

export { GraphRenderingManager };
