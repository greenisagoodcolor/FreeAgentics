/**
 * D3 Force Layout Engine for Knowledge Graphs
 * This provides a simulation-based layout using force-directed positioning
 */

import {
  RenderableGraphData,
  RenderableGraphNode,
  RenderableGraphEdge,
  RenderingConfig,
  LayoutEngine,
} from "../types";

export class D3ForceLayoutEngine implements LayoutEngine {
  private nodes: RenderableGraphNode[] = [];
  private edges: RenderableGraphEdge[] = [];
  private config: RenderingConfig;
  private simulation: unknown = null; // Would be d3.Simulation in a real D3 implementation
  private isRunning = false;

  constructor() {
    this.config = {
      width: 800,
      height: 600,
      nodeRadius: 8,
      linkDistance: 100,
      chargeStrength: -300,
      alphaDecay: 0.0228,
      centerForce: 0.1,
      collisionRadius: 12,
    };
  }

  initialize(data: RenderableGraphData, config: RenderingConfig): void {
    this.nodes = [...data.nodes];
    this.edges = [...data.edges];
    this.config = { ...this.config, ...config };

    // Initialize node positions if not set
    this.nodes.forEach((node, _index) => {
      if (node.x === undefined || node.y === undefined) {
        // Start with random positions
        node.x = Math.random() * this.config.width;
        node.y = Math.random() * this.config.height;
      }

      // Initialize velocities
      if (node.vx === undefined) node.vx = 0;
      if (node.vy === undefined) node.vy = 0;
    });

    this.setupSimulation();
  }

  private setupSimulation(): void {
    // This is a simplified force simulation
    // In a real implementation, this would use D3's force simulation
    this.isRunning = true;
    this.simulateForces();
  }

  private simulateForces(): void {
    if (!this.isRunning) return;

    const alpha = 0.1; // Simulation strength
    const iterations = 10; // Number of force iterations per tick

    for (let i = 0; i < iterations; i++) {
      // Apply center force
      this.applyCenterForce();

      // Apply repulsion force between nodes
      this.applyChargeForce();

      // Apply link forces
      this.applyLinkForces();

      // Apply collision detection
      this.applyCollisionForce();

      // Update positions
      this.updatePositions(alpha);
    }

    // Continue simulation
    if (this.isRunning) {
      requestAnimationFrame(() => this.simulateForces());
    }
  }

  private applyCenterForce(): void {
    const centerX = this.config.width / 2;
    const centerY = this.config.height / 2;
    const strength = this.config.centerForce || 0.1;

    this.nodes.forEach((node) => {
      if (node.fx === null || node.fx === undefined) {
        node.vx = (node.vx || 0) + (centerX - (node.x || 0)) * strength;
      }
      if (node.fy === null || node.fy === undefined) {
        node.vy = (node.vy || 0) + (centerY - (node.y || 0)) * strength;
      }
    });
  }

  private applyChargeForce(): void {
    const strength = this.config.chargeStrength || -300;

    for (let i = 0; i < this.nodes.length; i++) {
      const nodeA = this.nodes[i];

      for (let j = i + 1; j < this.nodes.length; j++) {
        const nodeB = this.nodes[j];

        const dx = (nodeB.x || 0) - (nodeA.x || 0);
        const dy = (nodeB.y || 0) - (nodeA.y || 0);
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = strength / (distance * distance);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;

        nodeA.vx = (nodeA.vx || 0) - fx;
        nodeA.vy = (nodeA.vy || 0) - fy;
        nodeB.vx = (nodeB.vx || 0) + fx;
        nodeB.vy = (nodeB.vy || 0) + fy;
      }
    }
  }

  private applyLinkForces(): void {
    const distance = this.config.linkDistance || 100;

    this.edges.forEach((edge) => {
      const source =
        typeof edge.source === "string"
          ? this.nodes.find((n) => n.id === edge.source)
          : edge.source;
      const target =
        typeof edge.target === "string"
          ? this.nodes.find((n) => n.id === edge.target)
          : edge.target;

      if (source && target) {
        const dx = (target.x || 0) - (source.x || 0);
        const dy = (target.y || 0) - (source.y || 0);
        const currentDistance = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = (currentDistance - distance) * 0.1;
        const fx = (dx / currentDistance) * force;
        const fy = (dy / currentDistance) * force;

        source.vx = (source.vx || 0) + fx;
        source.vy = (source.vy || 0) + fy;
        target.vx = (target.vx || 0) - fx;
        target.vy = (target.vy || 0) - fy;
      }
    });
  }

  private applyCollisionForce(): void {
    const radius = this.config.collisionRadius || 12;

    for (let i = 0; i < this.nodes.length; i++) {
      const nodeA = this.nodes[i];

      for (let j = i + 1; j < this.nodes.length; j++) {
        const nodeB = this.nodes[j];

        const dx = (nodeB.x || 0) - (nodeA.x || 0);
        const dy = (nodeB.y || 0) - (nodeA.y || 0);
        const distance = Math.sqrt(dx * dx + dy * dy);
        const minDistance = radius * 2;

        if (distance < minDistance && distance > 0) {
          const overlap = minDistance - distance;
          const fx = (dx / distance) * overlap * 0.5;
          const fy = (dy / distance) * overlap * 0.5;

          nodeA.vx = (nodeA.vx || 0) - fx;
          nodeA.vy = (nodeA.vy || 0) - fy;
          nodeB.vx = (nodeB.vx || 0) + fx;
          nodeB.vy = (nodeB.vy || 0) + fy;
        }
      }
    }
  }

  private updatePositions(alpha: number): void {
    this.nodes.forEach((node) => {
      // Apply velocity damping
      node.vx = (node.vx || 0) * 0.99;
      node.vy = (node.vy || 0) * 0.99;

      // Update positions (only if not fixed)
      if (node.fx === null || node.fx === undefined) {
        node.x = (node.x || 0) + (node.vx || 0) * alpha;
      }
      if (node.fy === null || node.fy === undefined) {
        node.y = (node.y || 0) + (node.vy || 0) * alpha;
      }

      // Keep nodes within bounds
      node.x = Math.max(10, Math.min(this.config.width - 10, node.x || 0));
      node.y = Math.max(10, Math.min(this.config.height - 10, node.y || 0));
    });
  }

  tick(): void {
    // Single step of the simulation
    this.applyCenterForce();
    this.applyChargeForce();
    this.applyLinkForces();
    this.applyCollisionForce();
    this.updatePositions(0.1);
  }

  restart(): void {
    this.isRunning = true;
    this.simulateForces();
  }

  stop(): void {
    this.isRunning = false;
  }

  alpha(): number;
  alpha(alpha: number): void;
  alpha(alpha?: number): number | void {
    if (alpha === undefined) {
      return this.isRunning ? 0.1 : 0;
    } else {
      // Set alpha would control simulation intensity
      // For simplicity, we just use it to start/stop
      if (alpha > 0) {
        this.restart();
      } else {
        this.stop();
      }
    }
  }
}
