/**
 * Knowledge graph specific type definitions
 */

export interface RenderableGraphNode {
  id: string;
  label: string;
  type: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
  size?: number;
  color?: string;
  properties?: Record<string, unknown>;
}

export interface RenderableGraphEdge {
  id?: string;
  source: string | RenderableGraphNode;
  target: string | RenderableGraphNode;
  relationship: string;
  weight?: number;
  color?: string;
  width?: number;
}

export interface RenderableGraphData {
  nodes: RenderableGraphNode[];
  edges: RenderableGraphEdge[];
}

export interface RenderingConfig {
  width: number;
  height: number;
  nodeRadius?: number;
  linkDistance?: number;
  chargeStrength?: number;
  alphaDecay?: number;
  backgroundColor?: string;
  showLabels?: boolean;
  enableZoom?: boolean;
  enablePan?: boolean;
  centerForce?: number;
  collisionRadius?: number;
}

export interface LayoutEngine {
  initialize(data: RenderableGraphData, config: RenderingConfig): void;
  tick(): void;
  restart(): void;
  stop(): void;
  alpha(): number;
  alpha(alpha: number): void;
}

export interface KnowledgeGraphRenderer {
  render(data: RenderableGraphData): void;
  update(data: RenderableGraphData): void;
  resize(width: number, height: number): void;
  destroy(): void;
  zoomToFit(): void;
  centerNode(nodeId: string): void;
  highlightPath(nodeIds: string[]): void;
  filterByType(types: string[]): void;
  searchNodes(query: string): RenderableGraphNode[];
}
