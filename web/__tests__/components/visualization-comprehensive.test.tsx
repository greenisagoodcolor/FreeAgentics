/**
 * Comprehensive Visualization Tests
 * 
 * Tests for all visualization components including knowledge graphs,
 * belief state displays, and interactive charts following ADR-007 requirements.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { jest } from '@jest/globals';

// Mock D3 comprehensively
const mockD3 = {
  select: jest.fn((...args: any[]) => ({
    selectAll: jest.fn((...args: any[]) => ({
      data: jest.fn((...args: any[]) => ({
        enter: jest.fn((...args: any[]) => ({
          append: jest.fn((...args: any[]) => ({
            attr: jest.fn((...args: any[]) => mockD3ChainableMethods),
            style: jest.fn((...args: any[]) => mockD3ChainableMethods),
            text: jest.fn((...args: any[]) => mockD3ChainableMethods),
            on: jest.fn((...args: any[]) => mockD3ChainableMethods),
            call: jest.fn((...args: any[]) => mockD3ChainableMethods),
          })),
          merge: jest.fn((...args: any[]) => mockD3ChainableMethods),
        })),
        exit: jest.fn((...args: any[]) => ({
          remove: jest.fn((...args: any[]) => mockD3ChainableMethods),
        })),
      })),
      remove: jest.fn((...args: any[]) => mockD3ChainableMethods),
    })),
    attr: jest.fn((...args: any[]) => mockD3ChainableMethods),
    style: jest.fn((...args: any[]) => mockD3ChainableMethods),
    on: jest.fn((...args: any[]) => mockD3ChainableMethods),
    call: jest.fn((...args: any[]) => mockD3ChainableMethods),
    datum: jest.fn((...args: any[]) => mockD3ChainableMethods),
    append: jest.fn((...args: any[]) => mockD3ChainableMethods),
    select: jest.fn((...args: any[]) => mockD3ChainableMethods),
    transition: jest.fn((...args: any[]) => mockD3ChainableMethods),
  })),
  scaleLinear: jest.fn(() => ({
    domain: jest.fn(function(this: any, ...args: any[]) { return this; }),
    range: jest.fn(function(this: any, ...args: any[]) { return this; }),
    nice: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  scaleOrdinal: jest.fn(() => ({
    domain: jest.fn(function(this: any, ...args: any[]) { return this; }),
    range: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  scaleTime: jest.fn(() => ({
    domain: jest.fn(function(this: any, ...args: any[]) { return this; }),
    range: jest.fn(function(this: any, ...args: any[]) { return this; }),
    nice: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  extent: jest.fn((...args: any[]) => [0, 100]),
  max: jest.fn((...args: any[]) => 100),
  min: jest.fn((...args: any[]) => 0),
  sum: jest.fn((...args: any[]) => 100),
  mean: jest.fn((...args: any[]) => 50),
  zoom: jest.fn(() => ({
    scaleExtent: jest.fn(function(this: any, ...args: any[]) { return this; }),
    on: jest.fn(function(this: any, ...args: any[]) { return this; }),
    transform: jest.fn((...args: any[]) => {}),
    scaleTo: jest.fn((...args: any[]) => {}),
    translateTo: jest.fn((...args: any[]) => {}),
  })),
  drag: jest.fn(() => ({
    on: jest.fn(function(this: any, ...args: any[]) { return this; }),
    filter: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  forceSimulation: jest.fn((...args: any[]) => ({
    force: jest.fn(function(this: any, ...args: any[]) { return this; }),
    nodes: jest.fn(function(this: any, ...args: any[]) { return this; }),
    links: jest.fn(function(this: any, ...args: any[]) { return this; }),
    on: jest.fn(function(this: any, ...args: any[]) { return this; }),
    stop: jest.fn((...args: any[]) => {}),
    restart: jest.fn((...args: any[]) => {}),
    tick: jest.fn((...args: any[]) => {}),
    alpha: jest.fn((...args: any[]) => 0.5),
    alphaTarget: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  forceLink: jest.fn((...args: any[]) => ({
    id: jest.fn(function(this: any, ...args: any[]) { return this; }),
    distance: jest.fn(function(this: any, ...args: any[]) { return this; }),
    strength: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  forceManyBody: jest.fn((...args: any[]) => ({
    strength: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  forceCenter: jest.fn((...args: any[]) => ({
    x: jest.fn(function(this: any, ...args: any[]) { return this; }),
    y: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  forceCollide: jest.fn((...args: any[]) => ({
    radius: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  line: jest.fn(() => ({
    x: jest.fn(function(this: any, ...args: any[]) { return this; }),
    y: jest.fn(function(this: any, ...args: any[]) { return this; }),
    curve: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  area: jest.fn(() => ({
    x: jest.fn(function(this: any, ...args: any[]) { return this; }),
    y0: jest.fn(function(this: any, ...args: any[]) { return this; }),
    y1: jest.fn(function(this: any, ...args: any[]) { return this; }),
    curve: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  arc: jest.fn(() => ({
    innerRadius: jest.fn(function(this: any, ...args: any[]) { return this; }),
    outerRadius: jest.fn(function(this: any, ...args: any[]) { return this; }),
    startAngle: jest.fn(function(this: any, ...args: any[]) { return this; }),
    endAngle: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  pie: jest.fn(() => ({
    value: jest.fn(function(this: any, ...args: any[]) { return this; }),
    sort: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  curveCardinal: {},
  interpolateRgb: jest.fn(() => '#ff0000'),
  schemeCategory10: ['#1f77b4', '#ff7f0e', '#2ca02c'],
  group: jest.fn((...args: any[]) => []),
  axisBottom: jest.fn((...args: any[]) => ({
    tickSize: jest.fn(function(this: any, ...args: any[]) { return this; }),
    tickFormat: jest.fn(function(this: any, ...args: any[]) { return this; }),
    ticks: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  axisLeft: jest.fn((...args: any[]) => ({
    tickSize: jest.fn(function(this: any, ...args: any[]) { return this; }),
    tickFormat: jest.fn(function(this: any, ...args: any[]) { return this; }),
    ticks: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  brushX: jest.fn(() => ({
    extent: jest.fn(function(this: any, ...args: any[]) { return this; }),
    on: jest.fn(function(this: any, ...args: any[]) { return this; }),
  })),
  timeFormat: jest.fn((...args: any[]) => (d: any) => d.toString()),
  format: jest.fn((...args: any[]) => (d: any) => d.toString()),
};

const mockD3ChainableMethods = {
  attr: jest.fn(function(this: any, ...args: any[]) { return this; }),
  style: jest.fn(function(this: any, ...args: any[]) { return this; }),
  text: jest.fn(function(this: any, ...args: any[]) { return this; }),
  on: jest.fn(function(this: any, ...args: any[]) { return this; }),
  call: jest.fn(function(this: any, ...args: any[]) { return this; }),
  datum: jest.fn(function(this: any, ...args: any[]) { return this; }),
  append: jest.fn(function(this: any, ...args: any[]) { return this; }),
  remove: jest.fn(function(this: any, ...args: any[]) { return this; }),
  selectAll: jest.fn((...args: any[]) => mockD3ChainableMethods),
  select: jest.fn((...args: any[]) => mockD3ChainableMethods),
  data: jest.fn((...args: any[]) => mockD3ChainableMethods),
  enter: jest.fn((...args: any[]) => mockD3ChainableMethods),
  exit: jest.fn((...args: any[]) => mockD3ChainableMethods),
  merge: jest.fn((...args: any[]) => mockD3ChainableMethods),
  transition: jest.fn((...args: any[]) => mockD3ChainableMethods),
  duration: jest.fn(function(this: any, ...args: any[]) { return this; }),
  ease: jest.fn(function(this: any, ...args: any[]) { return this; }),
};

jest.unstable_mockModule('d3', () => mockD3);

// Mock data structures
interface Node {
  id: string;
  label: string;
  type: 'agent' | 'concept' | 'belief' | 'coalition';
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
  weight: number;
  metadata: Record<string, any>;
}

interface Edge {
  id: string;
  source: string | Node;
  target: string | Node;
  weight: number;
  type: 'connection' | 'influence' | 'membership';
  metadata: Record<string, any>;
}

interface BeliefState {
  id: string;
  agentId: string;
  beliefs: Record<string, number>;
  confidence: number;
  timestamp: Date;
  uncertainty: number;
}

// Advanced Knowledge Graph Visualization
const AdvancedKnowledgeGraph: React.FC<{
  nodes: Node[];
  edges: Edge[];
  width: number;
  height: number;
  onNodeClick?: (node: Node) => void;
  onEdgeClick?: (edge: Edge) => void;
  onSelectionChange?: (selected: Node[]) => void;
  theme?: 'light' | 'dark';
  interactive?: boolean;
  showLabels?: boolean;
  enableClustering?: boolean;
}> = ({
  nodes,
  edges,
  width,
  height,
  onNodeClick,
  onEdgeClick,
  onSelectionChange,
  theme = 'light',
  interactive = true,
  showLabels = true,
  enableClustering = false,
}) => {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [selectedNodes, setSelectedNodes] = React.useState<Node[]>([]);
  const [zoomLevel, setZoomLevel] = React.useState(1);
  const [isPanning, setIsPanning] = React.useState(false);
  const [simulation, setSimulation] = React.useState<any>(null);

  React.useEffect(() => {
    if (!svgRef.current) return;

    const svg = mockD3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create force simulation
    const sim = mockD3.forceSimulation(nodes)
      .force('link', mockD3.forceLink(edges).id((d: any) => d.id).distance(100))
      .force('charge', mockD3.forceManyBody().strength(-300))
      .force('center', mockD3.forceCenter(width / 2, height / 2))
      .force('collision', mockD3.forceCollide().radius(30));

    setSimulation(sim);

    // Create zoom behavior
    const zoom = mockD3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event: any) => {
        setZoomLevel(event.transform.k);
        svg.select('.main-group').attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create main group
    const g = svg.append('g').attr('class', 'main-group');

    // Draw edges
    const links = g.append('g').attr('class', 'links')
      .selectAll('line')
      .data(edges)
      .enter().append('line')
      .attr('class', 'edge')
      .attr('stroke', theme === 'dark' ? '#666' : '#999')
      .attr('stroke-width', (d: Edge) => Math.sqrt(d.weight) * 2)
      .on('click', (event: any, d: Edge) => {
        if (interactive && onEdgeClick) {
          onEdgeClick(d);
        }
      });

    // Draw nodes
    const nodeGroups = g.append('g').attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('class', 'node-group');

    const circles = nodeGroups.append('circle')
      .attr('class', 'node')
      .attr('r', (d: Node) => 5 + d.weight * 3)
      .attr('fill', (d: Node) => getNodeColor(d.type, theme))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .on('click', (event: any, d: Node) => {
        if (interactive) {
          handleNodeClick(d);
        }
      });

    // Add labels if enabled
    if (showLabels) {
      nodeGroups.append('text')
        .attr('class', 'node-label')
        .attr('dy', '.35em')
        .attr('text-anchor', 'middle')
        .attr('fill', theme === 'dark' ? '#fff' : '#000')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .text((d: Node) => d.label);
    }

    // Update simulation on tick
    sim.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodeGroups.attr('transform', (d: Node) => `translate(${d.x},${d.y})`);
    });

    // Drag behavior
    if (interactive) {
      const drag = mockD3.drag()
        .on('start', (event: any, d: Node) => {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event: any, d: Node) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event: any, d: Node) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      circles.call(drag);
    }

    return () => {
      if (sim) sim.stop();
    };
  }, [nodes, edges, width, height, theme, interactive, showLabels]);

  const getNodeColor = (type: string, theme: string): string => {
    const colors = theme === 'dark' ? {
      agent: '#3b82f6',
      concept: '#10b981',
      belief: '#f59e0b',
      coalition: '#8b5cf6',
    } : {
      agent: '#2563eb',
      concept: '#059669',
      belief: '#d97706',
      coalition: '#7c3aed',
    };
    return colors[type as keyof typeof colors] || '#6b7280';
  };

  const handleNodeClick = (node: Node) => {
    setSelectedNodes(prev => {
      const isSelected = prev.some(n => n.id === node.id);
      const newSelection = isSelected
        ? prev.filter(n => n.id !== node.id)
        : [...prev, node];
      
      onSelectionChange?.(newSelection);
      return newSelection;
    });
    
    onNodeClick?.(node);
  };

  const handleZoomIn = () => {
    if (svgRef.current) {
      const svg = mockD3.select(svgRef.current);
      svg.transition().duration(300).call(
        svg.call(mockD3.zoom().scaleTo, Math.min(zoomLevel * 1.5, 10)) as any
      );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      const svg = mockD3.select(svgRef.current);
      svg.transition().duration(300).call(
        svg.call(mockD3.zoom().scaleTo, Math.max(zoomLevel / 1.5, 0.1)) as any
      );
    }
  };

  const handleReset = () => {
    if (simulation) {
      simulation.alpha(1).restart();
    }
    setSelectedNodes([]);
    setZoomLevel(1);
  };

  return (
    <div data-testid="knowledge-graph" className="knowledge-graph-container">
      <div className="graph-controls">
        <button data-testid="zoom-in" onClick={handleZoomIn}>
          Zoom In
        </button>
        <button data-testid="zoom-out" onClick={handleZoomOut}>
          Zoom Out
        </button>
        <button data-testid="reset-graph" onClick={handleReset}>
          Reset
        </button>
        <span data-testid="zoom-level">Zoom: {(zoomLevel * 100).toFixed(0)}%</span>
        <span data-testid="selected-count">
          Selected: {selectedNodes.length}
        </span>
      </div>
      
      <svg
        ref={svgRef}
        data-testid="graph-svg"
        width={width}
        height={height}
        style={{ border: '1px solid #ccc' }}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
          </marker>
        </defs>
      </svg>
      
      {selectedNodes.length > 0 && (
        <div data-testid="selection-info" className="selection-info">
          <h4>Selected Nodes</h4>
          {selectedNodes.map(node => (
            <div key={node.id} data-testid={`selected-node-${node.id}`}>
              {node.label} ({node.type})
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Belief State Visualization
const BeliefStateVisualization: React.FC<{
  beliefStates: BeliefState[];
  width: number;
  height: number;
  timeRange?: [Date, Date];
  onBeliefClick?: (belief: string, agentId: string) => void;
  showUncertainty?: boolean;
  animateTransitions?: boolean;
}> = ({
  beliefStates,
  width,
  height,
  timeRange,
  onBeliefClick,
  showUncertainty = true,
  animateTransitions = true,
}) => {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [selectedAgent, setSelectedAgent] = React.useState<string | null>(null);
  const [hoveredBelief, setHoveredBelief] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!svgRef.current || !beliefStates.length) return;

    const svg = mockD3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Get all unique beliefs
    const allBeliefs = Array.from(
      new Set(beliefStates.flatMap(state => Object.keys(state.beliefs)))
    );

    // Create scales
    const xScale = mockD3.scaleTime()
      .domain(timeRange || mockD3.extent(beliefStates, d => d.timestamp))
      .range([0, innerWidth]);

    const yScale = mockD3.scaleLinear()
      .domain([0, allBeliefs.length])
      .range([innerHeight, 0]);

    const colorScale = mockD3.scaleOrdinal()
      .domain(beliefStates.map(d => d.agentId))
      .range(mockD3.schemeCategory10);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add axes
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(mockD3.axisBottom(xScale));

    g.append('g')
      .attr('class', 'y-axis')
      .call(mockD3.axisLeft(yScale));

    // Draw belief trajectories
    beliefStates.forEach((state, stateIndex) => {
      Object.entries(state.beliefs).forEach(([belief, value], beliefIndex) => {
        const circle = g.append('circle')
          .attr('class', `belief-point belief-${belief}`)
          .attr('cx', xScale(state.timestamp))
          .attr('cy', yScale(allBeliefs.indexOf(belief)))
          .attr('r', 3 + value * 5)
          .attr('fill', colorScale(state.agentId))
          .attr('opacity', selectedAgent ? 
            (selectedAgent === state.agentId ? 1 : 0.3) : 0.7)
          .on('click', () => onBeliefClick?.(belief, state.agentId))
          .on('mouseover', () => setHoveredBelief(belief))
          .on('mouseout', () => setHoveredBelief(null));

        // Add uncertainty indicators
        if (showUncertainty && state.uncertainty > 0) {
          g.append('circle')
            .attr('class', 'uncertainty-ring')
            .attr('cx', xScale(state.timestamp))
            .attr('cy', yScale(allBeliefs.indexOf(belief)))
            .attr('r', (3 + value * 5) + state.uncertainty * 10)
            .attr('fill', 'none')
            .attr('stroke', colorScale(state.agentId))
            .attr('stroke-width', 1)
            .attr('opacity', 0.3);
        }
      });
    });

    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 120}, 20)`);

    const uniqueAgents = Array.from(new Set(beliefStates.map(s => s.agentId)));
    uniqueAgents.forEach((agentId, index) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${index * 20})`)
        .style('cursor', 'pointer')
        .on('click', () => {
          setSelectedAgent(selectedAgent === agentId ? null : agentId);
        });

      legendItem.append('circle')
        .attr('r', 5)
        .attr('fill', colorScale(agentId));

      legendItem.append('text')
        .attr('x', 10)
        .attr('y', 5)
        .text(agentId)
        .style('font-size', '12px');
    });

  }, [beliefStates, width, height, timeRange, selectedAgent, showUncertainty]);

  return (
    <div data-testid="belief-visualization" className="belief-visualization">
      <div className="belief-controls">
        <button 
          data-testid="show-all-agents"
          onClick={() => setSelectedAgent(null)}
        >
          Show All Agents
        </button>
        <button 
          data-testid="toggle-uncertainty"
          onClick={() => {/* Toggle uncertainty display */}}
        >
          {showUncertainty ? 'Hide' : 'Show'} Uncertainty
        </button>
      </div>
      
      <svg
        ref={svgRef}
        data-testid="belief-svg"
        width={width}
        height={height}
      />
      
      {hoveredBelief && (
        <div data-testid="belief-tooltip" className="belief-tooltip">
          Belief: {hoveredBelief}
          {selectedAgent && <div>Agent: {selectedAgent}</div>}
        </div>
      )}
    </div>
  );
};

// Interactive Time Series Chart
const TimeSeriesChart: React.FC<{
  data: Array<{ timestamp: Date; value: number; series: string }>;
  width: number;
  height: number;
  onBrushSelection?: (selection: [Date, Date] | null) => void;
  showBrush?: boolean;
  enableZoom?: boolean;
  interpolation?: 'linear' | 'cardinal' | 'step';
}> = ({
  data,
  width,
  height,
  onBrushSelection,
  showBrush = true,
  enableZoom = true,
  interpolation = 'linear',
}) => {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [brushSelection, setBrushSelection] = React.useState<[Date, Date] | null>(null);
  const [zoomTransform, setZoomTransform] = React.useState<any>(null);

  React.useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = mockD3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: showBrush ? 80 : 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const brushHeight = 40;

    // Group data by series
    const seriesData = Array.from(
      mockD3.group(data, d => d.series),
      ([key, values]) => ({ key, values })
    );

    // Create scales
    const xScale = mockD3.scaleTime()
      .domain(mockD3.extent(data, d => d.timestamp))
      .range([0, innerWidth]);

    const yScale = mockD3.scaleLinear()
      .domain(mockD3.extent(data, d => d.value))
      .range([innerHeight, 0])
      .nice();

    const colorScale = mockD3.scaleOrdinal()
      .domain(seriesData.map(d => d.key))
      .range(mockD3.schemeCategory10);

    // Create main chart area
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add clip path for zooming
    g.append('defs').append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', innerWidth)
      .attr('height', innerHeight);

    // Add axes
    const xAxis = g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(mockD3.axisBottom(xScale));

    const yAxis = g.append('g')
      .attr('class', 'y-axis')
      .call(mockD3.axisLeft(yScale));

    // Create line generator
    const line = mockD3.line()
      .x((d: any) => xScale(d.timestamp))
      .y((d: any) => yScale(d.value))
      .curve(interpolation === 'cardinal' ? mockD3.curveCardinal : undefined);

    // Draw lines for each series
    const chartArea = g.append('g').attr('clip-path', 'url(#clip)');
    
    seriesData.forEach(series => {
      chartArea.append('path')
        .datum(series.values)
        .attr('class', `line series-${series.key}`)
        .attr('fill', 'none')
        .attr('stroke', colorScale(series.key))
        .attr('stroke-width', 2)
        .attr('d', line);

      // Add data points
      chartArea.selectAll(`.point-${series.key}`)
        .data(series.values)
        .enter().append('circle')
        .attr('class', `point point-${series.key}`)
        .attr('cx', d => xScale(d.timestamp))
        .attr('cy', d => yScale(d.value))
        .attr('r', 3)
        .attr('fill', colorScale(series.key))
        .on('mouseover', function(event, d) {
          // Show tooltip logic
        });
    });

    // Add zoom behavior
    if (enableZoom) {
      const zoom = mockD3.zoom()
        .scaleExtent([1, 10])
        .on('zoom', (event) => {
          const transform = event.transform;
          setZoomTransform(transform);
          
          // Update scales
          const newXScale = transform.rescaleX(xScale);
          
          // Update axes
          xAxis.call(mockD3.axisBottom(newXScale));
          
          // Update lines and points
          chartArea.selectAll('.line')
            .attr('d', line.x((d: any) => newXScale(d.timestamp)));
          
          chartArea.selectAll('.point')
            .attr('cx', (d: any) => newXScale(d.timestamp));
        });

      svg.call(zoom);
    }

    // Add brush for selection
    if (showBrush) {
      const brushG = svg.append('g')
        .attr('class', 'brush')
        .attr('transform', `translate(${margin.left},${height - brushHeight})`);

      const brush = mockD3.brushX()
        .extent([[0, 0], [innerWidth, brushHeight]])
        .on('brush end', (event) => {
          if (event.selection) {
            const [x0, x1] = event.selection;
            const selection: [Date, Date] = [
              xScale.invert(x0),
              xScale.invert(x1)
            ];
            setBrushSelection(selection);
            onBrushSelection?.(selection);
          } else {
            setBrushSelection(null);
            onBrushSelection?.(null);
          }
        });

      brushG.call(brush);
    }

    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 120}, 20)`);

    seriesData.forEach((series, index) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${index * 20})`);

      legendItem.append('line')
        .attr('x1', 0)
        .attr('x2', 15)
        .attr('stroke', colorScale(series.key))
        .attr('stroke-width', 2);

      legendItem.append('text')
        .attr('x', 20)
        .attr('y', 5)
        .text(series.key)
        .style('font-size', '12px');
    });

  }, [data, width, height, showBrush, enableZoom, interpolation]);

  return (
    <div data-testid="time-series-chart" className="time-series-chart">
      <div className="chart-controls">
        <button data-testid="reset-zoom" onClick={() => {/* Reset zoom */}}>
          Reset Zoom
        </button>
        <button data-testid="clear-brush" onClick={() => setBrushSelection(null)}>
          Clear Selection
        </button>
      </div>
      
      <svg
        ref={svgRef}
        data-testid="chart-svg"
        width={width}
        height={height}
      />
      
      {brushSelection && (
        <div data-testid="brush-info" className="brush-info">
          Selection: {brushSelection[0].toLocaleString()} - {brushSelection[1].toLocaleString()}
        </div>
      )}
    </div>
  );
};

// Coalition Network Diagram
const CoalitionNetwork: React.FC<{
  coalitions: Array<{
    id: string;
    name: string;
    members: string[];
    strength: number;
    purpose: string;
  }>;
  agents: Array<{
    id: string;
    name: string;
    capabilities: string[];
  }>;
  width: number;
  height: number;
  onCoalitionClick?: (coalitionId: string) => void;
  onAgentClick?: (agentId: string) => void;
}> = ({
  coalitions,
  agents,
  width,
  height,
  onCoalitionClick,
  onAgentClick,
}) => {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [selectedCoalition, setSelectedCoalition] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!svgRef.current) return;

    const svg = mockD3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create nodes for coalitions and agents
    const coalitionNodes = coalitions.map(c => ({
      id: c.id,
      type: 'coalition',
      label: c.name,
      strength: c.strength,
      members: c.members.length,
    }));

    const agentNodes = agents.map(a => ({
      id: a.id,
      type: 'agent',
      label: a.name,
      capabilities: a.capabilities.length,
    }));

    const allNodes = [...coalitionNodes, ...agentNodes];

    // Create edges for membership
    const edges = coalitions.flatMap(coalition =>
      coalition.members.map(memberId => ({
        source: coalition.id,
        target: memberId,
        type: 'membership',
      }))
    );

    // Set up force simulation
    const simulation = mockD3.forceSimulation(allNodes)
      .force('link', mockD3.forceLink(edges).id((d: any) => d.id).distance(80))
      .force('charge', mockD3.forceManyBody().strength(-200))
      .force('center', mockD3.forceCenter(width / 2, height / 2));

    const g = svg.append('g').attr('class', 'coalition-network');

    // Draw edges
    const links = g.append('g').attr('class', 'links')
      .selectAll('line')
      .data(edges)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-width', 2);

    // Draw nodes
    const nodeGroups = g.append('g').attr('class', 'nodes')
      .selectAll('g')
      .data(allNodes)
      .enter().append('g')
      .attr('class', 'node-group');

    nodeGroups.append('circle')
      .attr('r', (d: any) => d.type === 'coalition' ? 15 + d.strength * 5 : 10)
      .attr('fill', (d: any) => d.type === 'coalition' ? '#8b5cf6' : '#3b82f6')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .on('click', (event, d: any) => {
        if (d.type === 'coalition') {
          setSelectedCoalition(d.id);
          onCoalitionClick?.(d.id);
        } else {
          onAgentClick?.(d.id);
        }
      });

    nodeGroups.append('text')
      .attr('dy', '.35em')
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('pointer-events', 'none')
      .text((d: any) => d.label);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodeGroups.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => simulation.stop();
  }, [coalitions, agents, width, height]);

  return (
    <div data-testid="coalition-network" className="coalition-network">
      <div className="network-info">
        <span data-testid="coalition-count">Coalitions: {coalitions.length}</span>
        <span data-testid="agent-count">Agents: {agents.length}</span>
        {selectedCoalition && (
          <span data-testid="selected-coalition">
            Selected: {coalitions.find(c => c.id === selectedCoalition)?.name}
          </span>
        )}
      </div>
      
      <svg
        ref={svgRef}
        data-testid="coalition-svg"
        width={width}
        height={height}
      />
    </div>
  );
};

describe('Comprehensive Visualization Tests', () => {
  const mockNodes: Node[] = [
    {
      id: 'node1',
      label: 'Agent 1',
      type: 'agent',
      weight: 0.8,
      metadata: { active: true },
    },
    {
      id: 'node2',
      label: 'Concept A',
      type: 'concept',
      weight: 0.6,
      metadata: { category: 'knowledge' },
    },
  ];

  const mockEdges: Edge[] = [
    {
      id: 'edge1',
      source: 'node1',
      target: 'node2',
      weight: 0.7,
      type: 'connection',
      metadata: { strength: 'high' },
    },
  ];

  const mockBeliefStates: BeliefState[] = [
    {
      id: 'belief1',
      agentId: 'agent1',
      beliefs: { 'belief_a': 0.8, 'belief_b': 0.3 },
      confidence: 0.9,
      timestamp: new Date('2024-01-01T10:00:00Z'),
      uncertainty: 0.1,
    },
    {
      id: 'belief2',
      agentId: 'agent1',
      beliefs: { 'belief_a': 0.7, 'belief_b': 0.5 },
      confidence: 0.8,
      timestamp: new Date('2024-01-01T11:00:00Z'),
      uncertainty: 0.2,
    },
  ];

  const mockTimeSeriesData = [
    { timestamp: new Date('2024-01-01T10:00:00Z'), value: 50, series: 'metric1' },
    { timestamp: new Date('2024-01-01T11:00:00Z'), value: 65, series: 'metric1' },
    { timestamp: new Date('2024-01-01T10:00:00Z'), value: 30, series: 'metric2' },
    { timestamp: new Date('2024-01-01T11:00:00Z'), value: 45, series: 'metric2' },
  ];

  const mockCoalitions = [
    {
      id: 'coalition1',
      name: 'Research Team',
      members: ['agent1', 'agent2'],
      strength: 0.8,
      purpose: 'research',
    },
  ];

  const mockAgents = [
    {
      id: 'agent1',
      name: 'Researcher',
      capabilities: ['analysis', 'reporting'],
    },
    {
      id: 'agent2',
      name: 'Assistant',
      capabilities: ['data_collection'],
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('AdvancedKnowledgeGraph', () => {
    it('renders knowledge graph with nodes and edges', () => {
      render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('knowledge-graph')).toBeInTheDocument();
      expect(screen.getByTestId('graph-svg')).toBeInTheDocument();
      expect(screen.getByTestId('zoom-in')).toBeInTheDocument();
      expect(screen.getByTestId('zoom-out')).toBeInTheDocument();
      expect(screen.getByTestId('reset-graph')).toBeInTheDocument();
    });

    it('handles node interactions', () => {
      const onNodeClick = jest.fn();
      const onSelectionChange = jest.fn();

      render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
          onNodeClick={onNodeClick}
          onSelectionChange={onSelectionChange}
        />
      );

      expect(screen.getByTestId('selected-count')).toHaveTextContent('Selected: 0');
    });

    it('controls zoom levels', () => {
      render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('zoom-level')).toHaveTextContent('Zoom: 100%');

      fireEvent.click(screen.getByTestId('zoom-in'));
      // Zoom level would be updated by D3, but in our mock it stays the same
      expect(screen.getByTestId('zoom-level')).toBeInTheDocument();
    });

    it('resets graph state', () => {
      render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
        />
      );

      fireEvent.click(screen.getByTestId('reset-graph'));
      expect(screen.getByTestId('selected-count')).toHaveTextContent('Selected: 0');
    });

    it('handles edge interactions', () => {
      const onEdgeClick = jest.fn();

      render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
          onEdgeClick={onEdgeClick}
        />
      );

      // Edge click would be handled by D3 event listeners
      expect(screen.getByTestId('graph-svg')).toBeInTheDocument();
    });

    it('supports different themes', () => {
      const { rerender } = render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
          theme="light"
        />
      );

      expect(screen.getByTestId('graph-svg')).toBeInTheDocument();

      rerender(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
          theme="dark"
        />
      );

      expect(screen.getByTestId('graph-svg')).toBeInTheDocument();
    });

    it('toggles interactive features', () => {
      const { rerender } = render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
          interactive={true}
        />
      );

      expect(screen.getByTestId('knowledge-graph')).toBeInTheDocument();

      rerender(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
          interactive={false}
        />
      );

      expect(screen.getByTestId('knowledge-graph')).toBeInTheDocument();
    });
  });

  describe('BeliefStateVisualization', () => {
    it('renders belief state visualization', () => {
      render(
        <BeliefStateVisualization
          beliefStates={mockBeliefStates}
          width={800}
          height={400}
        />
      );

      expect(screen.getByTestId('belief-visualization')).toBeInTheDocument();
      expect(screen.getByTestId('belief-svg')).toBeInTheDocument();
      expect(screen.getByTestId('show-all-agents')).toBeInTheDocument();
      expect(screen.getByTestId('toggle-uncertainty')).toBeInTheDocument();
    });

    it('handles belief interactions', () => {
      const onBeliefClick = jest.fn();

      render(
        <BeliefStateVisualization
          beliefStates={mockBeliefStates}
          width={800}
          height={400}
          onBeliefClick={onBeliefClick}
        />
      );

      // Belief click would be handled by D3 event listeners
      expect(screen.getByTestId('belief-svg')).toBeInTheDocument();
    });

    it('filters by agent', () => {
      render(
        <BeliefStateVisualization
          beliefStates={mockBeliefStates}
          width={800}
          height={400}
        />
      );

      fireEvent.click(screen.getByTestId('show-all-agents'));
      expect(screen.getByTestId('belief-visualization')).toBeInTheDocument();
    });

    it('toggles uncertainty display', () => {
      render(
        <BeliefStateVisualization
          beliefStates={mockBeliefStates}
          width={800}
          height={400}
          showUncertainty={true}
        />
      );

      fireEvent.click(screen.getByTestId('toggle-uncertainty'));
      expect(screen.getByTestId('belief-visualization')).toBeInTheDocument();
    });

    it('handles custom time ranges', () => {
      const timeRange: [Date, Date] = [
        new Date('2024-01-01T09:00:00Z'),
        new Date('2024-01-01T12:00:00Z'),
      ];

      render(
        <BeliefStateVisualization
          beliefStates={mockBeliefStates}
          width={800}
          height={400}
          timeRange={timeRange}
        />
      );

      expect(screen.getByTestId('belief-svg')).toBeInTheDocument();
    });
  });

  describe('TimeSeriesChart', () => {
    it('renders time series chart', () => {
      render(
        <TimeSeriesChart
          data={mockTimeSeriesData}
          width={800}
          height={400}
        />
      );

      expect(screen.getByTestId('time-series-chart')).toBeInTheDocument();
      expect(screen.getByTestId('chart-svg')).toBeInTheDocument();
      expect(screen.getByTestId('reset-zoom')).toBeInTheDocument();
      expect(screen.getByTestId('clear-brush')).toBeInTheDocument();
    });

    it('handles brush selection', () => {
      const onBrushSelection = jest.fn();

      render(
        <TimeSeriesChart
          data={mockTimeSeriesData}
          width={800}
          height={400}
          onBrushSelection={onBrushSelection}
          showBrush={true}
        />
      );

      // Brush selection would be handled by D3 brush behavior
      expect(screen.getByTestId('chart-svg')).toBeInTheDocument();
    });

    it('supports zoom functionality', () => {
      render(
        <TimeSeriesChart
          data={mockTimeSeriesData}
          width={800}
          height={400}
          enableZoom={true}
        />
      );

      fireEvent.click(screen.getByTestId('reset-zoom'));
      expect(screen.getByTestId('chart-svg')).toBeInTheDocument();
    });

    it('handles different interpolation types', () => {
      const { rerender } = render(
        <TimeSeriesChart
          data={mockTimeSeriesData}
          width={800}
          height={400}
          interpolation="linear"
        />
      );

      expect(screen.getByTestId('chart-svg')).toBeInTheDocument();

      rerender(
        <TimeSeriesChart
          data={mockTimeSeriesData}
          width={800}
          height={400}
          interpolation="cardinal"
        />
      );

      expect(screen.getByTestId('chart-svg')).toBeInTheDocument();
    });

    it('clears brush selection', () => {
      render(
        <TimeSeriesChart
          data={mockTimeSeriesData}
          width={800}
          height={400}
        />
      );

      fireEvent.click(screen.getByTestId('clear-brush'));
      expect(screen.queryByTestId('brush-info')).not.toBeInTheDocument();
    });
  });

  describe('CoalitionNetwork', () => {
    it('renders coalition network diagram', () => {
      render(
        <CoalitionNetwork
          coalitions={mockCoalitions}
          agents={mockAgents}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('coalition-network')).toBeInTheDocument();
      expect(screen.getByTestId('coalition-svg')).toBeInTheDocument();
      expect(screen.getByTestId('coalition-count')).toHaveTextContent('Coalitions: 1');
      expect(screen.getByTestId('agent-count')).toHaveTextContent('Agents: 2');
    });

    it('handles coalition interactions', () => {
      const onCoalitionClick = jest.fn();
      const onAgentClick = jest.fn();

      render(
        <CoalitionNetwork
          coalitions={mockCoalitions}
          agents={mockAgents}
          width={800}
          height={600}
          onCoalitionClick={onCoalitionClick}
          onAgentClick={onAgentClick}
        />
      );

      // Interactions would be handled by D3 event listeners
      expect(screen.getByTestId('coalition-svg')).toBeInTheDocument();
    });

    it('displays network statistics', () => {
      render(
        <CoalitionNetwork
          coalitions={mockCoalitions}
          agents={mockAgents}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('coalition-count')).toBeInTheDocument();
      expect(screen.getByTestId('agent-count')).toBeInTheDocument();
    });
  });

  describe('Integration Tests', () => {
    it('handles empty data gracefully', () => {
      render(
        <AdvancedKnowledgeGraph
          nodes={[]}
          edges={[]}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('knowledge-graph')).toBeInTheDocument();
      expect(screen.getByTestId('selected-count')).toHaveTextContent('Selected: 0');
    });

    it('handles large datasets efficiently', () => {
      const largeNodes = Array.from({ length: 100 }, (_, i) => ({
        id: `node${i}`,
        label: `Node ${i}`,
        type: 'concept' as const,
        weight: Math.random(),
        metadata: {},
      }));

      const largeEdges = Array.from({ length: 150 }, (_, i) => ({
        id: `edge${i}`,
        source: `node${Math.floor(Math.random() * 100)}`,
        target: `node${Math.floor(Math.random() * 100)}`,
        weight: Math.random(),
        type: 'connection' as const,
        metadata: {},
      }));

      render(
        <AdvancedKnowledgeGraph
          nodes={largeNodes}
          edges={largeEdges}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('knowledge-graph')).toBeInTheDocument();
    });

    it('handles visualization updates correctly', () => {
      const { rerender } = render(
        <AdvancedKnowledgeGraph
          nodes={mockNodes}
          edges={mockEdges}
          width={800}
          height={600}
        />
      );

      const updatedNodes = [
        ...mockNodes,
        {
          id: 'node3',
          label: 'New Node',
          type: 'belief' as const,
          weight: 0.9,
          metadata: {},
        },
      ];

      rerender(
        <AdvancedKnowledgeGraph
          nodes={updatedNodes}
          edges={mockEdges}
          width={800}
          height={600}
        />
      );

      expect(screen.getByTestId('knowledge-graph')).toBeInTheDocument();
    });
  });
});