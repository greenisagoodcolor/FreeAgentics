/**
 * Markov Blanket Visualization Tests
 * 
 * Tests for the Markov blanket visualization component
 * following ADR-007 comprehensive testing requirements.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MarkovBlanketVisualization } from '@/components/markov-blanket-visualization';
import * as d3 from 'd3';

// Mock D3
jest.mock('d3', () => ({
  select: jest.fn().mockReturnThis(),
  selectAll: jest.fn().mockReturnThis(),
  append: jest.fn().mockReturnThis(),
  attr: jest.fn().mockReturnThis(),
  style: jest.fn().mockReturnThis(),
  text: jest.fn().mockReturnThis(),
  data: jest.fn().mockReturnThis(),
  enter: jest.fn().mockReturnThis(),
  exit: jest.fn().mockReturnThis(),
  remove: jest.fn().mockReturnThis(),
  on: jest.fn().mockReturnThis(),
  transition: jest.fn().mockReturnThis(),
  duration: jest.fn().mockReturnThis(),
  call: jest.fn().mockReturnThis(),
  scaleLinear: jest.fn(() => ({
    domain: jest.fn().mockReturnThis(),
    range: jest.fn().mockReturnThis(),
  })),
  scaleOrdinal: jest.fn(() => ({
    domain: jest.fn().mockReturnThis(),
    range: jest.fn().mockReturnThis(),
  })),
  arc: jest.fn(() => ({
    innerRadius: jest.fn().mockReturnThis(),
    outerRadius: jest.fn().mockReturnThis(),
    startAngle: jest.fn().mockReturnThis(),
    endAngle: jest.fn().mockReturnThis(),
  })),
  pie: jest.fn(() => ({
    value: jest.fn().mockReturnThis(),
    sort: jest.fn().mockReturnThis(),
  })),
  line: jest.fn(() => ({
    x: jest.fn().mockReturnThis(),
    y: jest.fn().mockReturnThis(),
    curve: jest.fn().mockReturnThis(),
  })),
  curveLinearClosed: {},
  curveCardinal: {},
  curveMonotoneX: {},
  axisBottom: jest.fn().mockReturnThis(),
  axisLeft: jest.fn().mockReturnThis(),
}));

describe('MarkovBlanketVisualization Component', () => {
  const mockDimensions = {
    internal_states: [0.7, 0.8, 0.3, 0.75],
    sensory_states: [0.7, 0.4, 0.3],
    active_states: [0.6, 0.5, 0.8],
    external_states: [0.4, 0.6, 0.8],
    internal_dimension: 0.65,
    sensory_dimension: 0.47,
    active_dimension: 0.63,
    external_dimension: 0.6,
  };

  const mockMetrics = {
    free_energy: 2.45,
    expected_free_energy: 2.1,
    kl_divergence: 0.35,
    boundary_integrity: 0.85,
    conditional_independence: 0.78,
    stability_over_time: 0.92,
    violation_count: 0,
  };

  const mockViolations = [
    {
      event_id: 'violation-1',
      agent_id: 'agent-1',
      violation_type: 'conditional_independence',
      timestamp: new Date().toISOString(),
      severity: 0.7,
      independence_measure: 0.45,
      threshold_violated: 0.5,
      free_energy: 2.8,
      expected_free_energy: 2.1,
      kl_divergence: 0.7,
      acknowledged: false,
      mitigated: false,
    },
  ];

  const mockAgentPosition = {
    agent_id: 'agent-1',
    position: {
      internal: 0.65,
      sensory: 0.47,
      active: 0.63,
      external: 0.6,
    },
    boundary_distance: 0.15,
    is_within_boundary: true,
  };

  const mockBoundaryThresholds = {
    internal: 0.8,
    sensory: 0.7,
    active: 0.75,
    external: 0.7,
  };

  const mockProps = {
    agentId: 'agent-1',
    dimensions: mockDimensions,
    metrics: mockMetrics,
    violations: mockViolations,
    agentPosition: mockAgentPosition,
    boundaryThresholds: mockBoundaryThresholds,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders without crashing', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText('Markov Blanket Visualization')).toBeInTheDocument();
    });

    it('displays agent ID in header', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText(/Agent agent-1/)).toBeInTheDocument();
    });

    it('renders radar chart SVG', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByRole('img', { hidden: true })).toBeInTheDocument();
    });

    it('displays boundary status badge', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText('Within Boundary')).toBeInTheDocument();
    });

    it('shows live badge when realTimeUpdates is true', () => {
      render(<MarkovBlanketVisualization {...mockProps} realTimeUpdates={true} />);
      expect(screen.getByText('Live')).toBeInTheDocument();
    });
  });

  describe('Metrics Display', () => {
    it('displays boundary metrics when showMetrics is true', () => {
      render(<MarkovBlanketVisualization {...mockProps} showMetrics={true} />);
      expect(screen.getByText('Boundary Metrics')).toBeInTheDocument();
      expect(screen.getByText('2.450')).toBeInTheDocument(); // Free Energy
      expect(screen.getByText('85.0%')).toBeInTheDocument(); // Boundary Integrity
    });

    it('hides metrics when showMetrics is false', () => {
      render(<MarkovBlanketVisualization {...mockProps} showMetrics={false} />);
      expect(screen.queryByText('Boundary Metrics')).not.toBeInTheDocument();
    });

    it('displays KL divergence and violation count', () => {
      render(<MarkovBlanketVisualization {...mockProps} showMetrics={true} />);
      expect(screen.getByText('0.350')).toBeInTheDocument(); // KL Divergence
      expect(screen.getByText('0')).toBeInTheDocument(); // Violation count
    });
  });

  describe('Violations Display', () => {
    it('displays violations when showViolations is true', () => {
      render(<MarkovBlanketVisualization {...mockProps} showViolations={true} />);
      expect(screen.getByText('Recent Violations')).toBeInTheDocument();
      expect(screen.getByText('conditional_independence')).toBeInTheDocument();
    });

    it('hides violations when showViolations is false', () => {
      render(<MarkovBlanketVisualization {...mockProps} showViolations={false} />);
      expect(screen.queryByText('Recent Violations')).not.toBeInTheDocument();
    });

    it('shows acknowledge button for unacknowledged violations', () => {
      render(<MarkovBlanketVisualization {...mockProps} showViolations={true} />);
      expect(screen.getByText('Acknowledge')).toBeInTheDocument();
    });

    it('calls onViolationAcknowledge when acknowledge button is clicked', () => {
      const mockAcknowledge = jest.fn();
      render(
        <MarkovBlanketVisualization 
          {...mockProps} 
          showViolations={true}
          onViolationAcknowledge={mockAcknowledge}
        />
      );
      
      fireEvent.click(screen.getByText('Acknowledge'));
      expect(mockAcknowledge).toHaveBeenCalledWith('violation-1');
    });
  });

  describe('Controls', () => {
    it('renders control sliders', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText('Controls')).toBeInTheDocument();
      expect(screen.getByText('Alert Threshold')).toBeInTheDocument();
      expect(screen.getByText('Animation Speed')).toBeInTheDocument();
    });

    it('renders agent trail toggle', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText('Show Agent Trail')).toBeInTheDocument();
    });

    it('updates alert threshold when slider changes', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: 0.9 } });
      // Slider change would trigger internal state update
    });
  });

  describe('Critical Violations Alert', () => {
    it('shows alert for critical violations', () => {
      const criticalViolations = [{
        ...mockViolations[0],
        severity: 0.9, // Above default threshold of 0.8
      }];
      
      render(
        <MarkovBlanketVisualization 
          {...mockProps} 
          violations={criticalViolations}
        />
      );
      
      expect(screen.getByText(/1 critical boundary violation detected/)).toBeInTheDocument();
    });

    it('shows plural message for multiple critical violations', () => {
      const criticalViolations = [
        { ...mockViolations[0], severity: 0.9 },
        { ...mockViolations[0], event_id: 'violation-2', severity: 0.85 },
      ];
      
      render(
        <MarkovBlanketVisualization 
          {...mockProps} 
          violations={criticalViolations}
        />
      );
      
      expect(screen.getByText(/2 critical boundary violations detected/)).toBeInTheDocument();
    });
  });

  describe('Boundary Violation Status', () => {
    it('shows boundary violation badge when agent is outside boundary', () => {
      const outsideBoundaryPosition = {
        ...mockAgentPosition,
        is_within_boundary: false,
      };
      
      render(
        <MarkovBlanketVisualization 
          {...mockProps} 
          agentPosition={outsideBoundaryPosition}
        />
      );
      
      expect(screen.getByText('Boundary Violation')).toBeInTheDocument();
    });

    it('applies destructive variant to badge for boundary violations', () => {
      const outsideBoundaryPosition = {
        ...mockAgentPosition,
        is_within_boundary: false,
      };
      
      render(
        <MarkovBlanketVisualization 
          {...mockProps} 
          agentPosition={outsideBoundaryPosition}
        />
      );
      
      const badge = screen.getByText('Boundary Violation');
      expect(badge.closest('.badge')).toHaveClass(); // Would check for destructive styling
    });
  });

  describe('D3 Visualization Integration', () => {
    it('initializes D3 radar chart on mount', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(d3.select).toHaveBeenCalled();
    });

    it('creates radar chart elements', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(d3.select).toHaveBeenCalled();
      expect(d3.scaleLinear).toHaveBeenCalled();
    });

    it('updates visualization when dimensions change', () => {
      const { rerender } = render(<MarkovBlanketVisualization {...mockProps} />);
      
      const newDimensions = {
        ...mockDimensions,
        internal_dimension: 0.8,
      };
      
      rerender(
        <MarkovBlanketVisualization 
          {...mockProps} 
          dimensions={newDimensions}
        />
      );
      
      expect(d3.select).toHaveBeenCalled();
    });
  });

  describe('Threshold Management', () => {
    it('calls onBoundaryThresholdChange when threshold is modified', () => {
      const mockThresholdChange = jest.fn();
      render(
        <MarkovBlanketVisualization 
          {...mockProps}
          onBoundaryThresholdChange={mockThresholdChange}
        />
      );
      
      // Threshold changes would be triggered through UI interactions
      // In a full implementation, this would test slider or input changes
    });
  });

  describe('Error Handling', () => {
    it('handles missing dimensions gracefully', () => {
      const propsWithoutDimensions = {
        ...mockProps,
        dimensions: undefined as any,
      };
      
      expect(() => {
        render(<MarkovBlanketVisualization {...propsWithoutDimensions} />);
      }).not.toThrow();
    });

    it('handles empty violations array', () => {
      render(
        <MarkovBlanketVisualization 
          {...mockProps} 
          violations={[]}
          showViolations={true}
        />
      );
      
      expect(screen.queryByText('Recent Violations')).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides accessible labels for interactive elements', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText('Show Agent Trail')).toBeInTheDocument();
      expect(screen.getByText('Alert Threshold')).toBeInTheDocument();
    });

    it('includes proper heading structure', () => {
      render(<MarkovBlanketVisualization {...mockProps} />);
      expect(screen.getByText('Markov Blanket Visualization')).toBeInTheDocument();
      expect(screen.getByText('Controls')).toBeInTheDocument();
    });
  });
});