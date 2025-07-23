/**
 * Graph rendering type definitions
 */

export interface GraphNode {
  id: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
  label?: string;
  type?: string;
  size?: number;
  color?: string;
  properties?: Record<string, unknown>;
}

export interface GraphEdge {
  id?: string;
  source: string | GraphNode;
  target: string | GraphNode;
  relationship?: string;
  weight?: number;
  color?: string;
  width?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
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
}

export interface GraphRenderingOptions {
  container: HTMLElement | string;
  config?: Partial<RenderingConfig>;
  onNodeClick?: (node: GraphNode) => void;
  onNodeHover?: (node: GraphNode | null) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
}

export interface GraphRenderingManager {
  render(data: GraphData): void;
  update(data: GraphData): void;
  resize(width: number, height: number): void;
  destroy(): void;
  zoomToFit(): void;
  centerNode(nodeId: string): void;
  highlightNode(nodeId: string, highlight: boolean): void;
  getConfig(): RenderingConfig;
  setConfig(config: Partial<RenderingConfig>): void;
}
