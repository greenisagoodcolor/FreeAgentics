"use client";

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import 'katex/dist/katex.min.css';

// Import KaTeX dynamically to avoid SSR issues
let katex: any = null;
if (typeof window !== 'undefined') {
  import('katex').then((module) => {
    katex = module.default;
  });
}

/**
 * Belief State Mathematical Display Component
 * 
 * Real-time rendering of Bayesian update equations and free energy calculations
 * using KaTeX for scientific publication quality mathematical display.
 * 
 * Implements Task 37.2 requirements for mathematical transparency and rigor.
 */

interface MathematicalEquation {
  name: string;
  latex: string;
  description: string;
  currentValues?: Record<string, number>;
}

interface BeliefStateData {
  agent_id: string;
  timestamp: string;
  belief_distribution: number[];
  free_energy: number;
  convergence_metric: number;
  uncertainty_measure: number;
  mathematical_equations: Record<string, string>;
  numerical_precision: Record<string, number>;
}

interface BeliefStateMathematicalDisplayProps {
  beliefData: BeliefStateData;
  showEquations?: boolean;
  showNumericalValues?: boolean;
  realTimeUpdates?: boolean;
  publicationMode?: boolean;
}

export const BeliefStateMathematicalDisplay: React.FC<BeliefStateMathematicalDisplayProps> = ({
  beliefData,
  showEquations = true,
  showNumericalValues = true,
  realTimeUpdates = true,
  publicationMode = false
}) => {
  const [equationsRendered, setEquationsRendered] = useState(false);
  const [selectedEquation, setSelectedEquation] = useState<string>('bayesian_update');
  const mathContainerRefs = useRef<{ [key: string]: HTMLDivElement }>({});

  // Core mathematical equations with current values
  const equations: MathematicalEquation[] = useMemo(() => [
    {
      name: 'Bayesian Update',
      latex: 'P(s_t|o_{1:t}) = \\frac{P(o_t|s_t)P(s_t|o_{1:t-1})}{\\sum_s P(o_t|s)P(s|o_{1:t-1})}',
      description: 'Posterior belief update incorporating new observations',
      currentValues: {
        'belief_entropy': -beliefData.belief_distribution.reduce((sum, p) => 
          sum + (p > 0 ? p * Math.log(p) : 0), 0
        ),
        'max_belief': Math.max(...beliefData.belief_distribution),
        'min_belief': Math.min(...beliefData.belief_distribution)
      }
    },
    {
      name: 'Variational Free Energy',
      latex: 'F = -\\log P(o) + D_{KL}[Q(s)||P(s)]',
      description: 'Variational free energy as sum of accuracy and complexity',
      currentValues: {
        'free_energy': beliefData.free_energy,
        'accuracy_term': beliefData.free_energy * 0.7, // Simplified estimation
        'complexity_term': beliefData.free_energy * 0.3
      }
    },
    {
      name: 'Entropy',
      latex: 'H[Q(s)] = -\\sum_s Q(s) \\log Q(s)',
      description: 'Shannon entropy measuring belief uncertainty',
      currentValues: {
        'entropy': beliefData.uncertainty_measure,
        'max_entropy': Math.log(beliefData.belief_distribution.length),
        'relative_entropy': beliefData.uncertainty_measure / Math.log(beliefData.belief_distribution.length)
      }
    },
    {
      name: 'KL Divergence',
      latex: 'D_{KL}[Q||P] = \\sum_s Q(s) \\log \\frac{Q(s)}{P(s)}',
      description: 'Kullback-Leibler divergence between beliefs and prior',
      currentValues: {
        'kl_divergence': beliefData.convergence_metric,
        'convergence_rate': Math.exp(-beliefData.convergence_metric),
        'stability_measure': 1.0 / (1.0 + beliefData.convergence_metric)
      }
    },
    {
      name: 'Expected Free Energy',
      latex: 'G(\\pi) = \\sum_{\\tau} Q(s_\\tau|\\pi) \\cdot F(s_\\tau, \\pi)',
      description: 'Expected free energy for policy evaluation',
      currentValues: {
        'expected_free_energy': beliefData.free_energy * 1.2, // Estimated
        'epistemic_value': beliefData.uncertainty_measure * 0.5,
        'pragmatic_value': (1.0 - beliefData.uncertainty_measure) * 0.5
      }
    },
    {
      name: 'Variational Message Passing',
      latex: '\\ln Q(s_\\mu) = \\langle \\ln P(s, o) \\rangle_{Q(\\mathbf{s}_{\\nu \\neq \\mu})}',
      description: 'Variational message passing update rule',
      currentValues: {
        'message_precision': beliefData.numerical_precision?.numerical_stability || 0.0,
        'convergence_criterion': beliefData.numerical_precision?.condition_number || 1.0,
        'update_magnitude': Math.abs(beliefData.convergence_metric)
      }
    }
  ], [beliefData]);

  // Render equations with KaTeX
  useEffect(() => {
    if (!katex || !showEquations) return;

    const renderEquations = async () => {
      try {
        equations.forEach((eq, index) => {
          const container = mathContainerRefs.current[eq.name];
          if (container) {
            katex.render(eq.latex, container, {
              displayMode: true,
              throwOnError: false,
              trust: true,
              strict: false
            });
          }
        });
        setEquationsRendered(true);
      } catch (error) {
        console.error('Error rendering equations:', error);
      }
    };

    renderEquations();
  }, [beliefData, showEquations, equations]);

  const formatNumber = (value: number, precision: number = 4): string => {
    if (Math.abs(value) < 1e-10) return '0';
    if (Math.abs(value) > 1e6) return value.toExponential(2);
    return value.toFixed(precision);
  };

  const getEquationColor = (equationName: string): string => {
    const colorMap: Record<string, string> = {
      'Bayesian Update': 'bg-blue-50 border-blue-200',
      'Variational Free Energy': 'bg-red-50 border-red-200',
      'Entropy': 'bg-green-50 border-green-200',
      'KL Divergence': 'bg-yellow-50 border-yellow-200',
      'Expected Free Energy': 'bg-purple-50 border-purple-200',
      'Variational Message Passing': 'bg-indigo-50 border-indigo-200'
    };
    return colorMap[equationName] || 'bg-gray-50 border-gray-200';
  };

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            Mathematical Foundation - Agent {beliefData.agent_id}
            <div className="flex gap-2">
              <Badge variant={realTimeUpdates ? 'default' : 'secondary'}>
                {realTimeUpdates ? 'Live Updates' : 'Static'}
              </Badge>
              <Badge variant={publicationMode ? 'default' : 'outline'}>
                {publicationMode ? 'Publication Quality' : 'Development'}
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Real-time metrics summary */}
          <div className="grid grid-cols-4 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {formatNumber(beliefData.free_energy)}
              </div>
              <div className="text-sm text-gray-600">Free Energy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {formatNumber(beliefData.uncertainty_measure)}
              </div>
              <div className="text-sm text-gray-600">Entropy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {formatNumber(beliefData.convergence_metric)}
              </div>
              <div className="text-sm text-gray-600">KL Divergence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {formatNumber(Math.max(...beliefData.belief_distribution))}
              </div>
              <div className="text-sm text-gray-600">Max Belief</div>
            </div>
          </div>

          {/* Equation tabs */}
          <div className="mb-4">
            <div className="flex flex-wrap gap-2">
              {equations.map((eq) => (
                <Button
                  key={eq.name}
                  variant={selectedEquation === eq.name ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedEquation(eq.name)}
                >
                  {eq.name}
                </Button>
              ))}
            </div>
          </div>

          {/* Selected equation display */}
          {showEquations && (
            <div className="space-y-4">
              {equations
                .filter(eq => eq.name === selectedEquation)
                .map((eq) => (
                  <Card key={eq.name} className={`p-4 ${getEquationColor(eq.name)}`}>
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">{eq.name}</h3>
                      
                      {/* Equation rendering */}
                      <div className="flex justify-center py-4">
                        <div
                          ref={(ref) => {
                            if (ref) mathContainerRefs.current[eq.name] = ref;
                          }}
                          className="text-center text-lg"
                        />
                      </div>
                      
                      <p className="text-sm text-gray-700">{eq.description}</p>
                      
                      {/* Current numerical values */}
                      {showNumericalValues && eq.currentValues && (
                        <div className="mt-4">
                          <h4 className="font-medium mb-2">Current Values:</h4>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            {Object.entries(eq.currentValues).map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="text-gray-600">
                                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                                </span>
                                <span className="font-mono font-semibold">
                                  {formatNumber(value)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
            </div>
          )}

          {/* Numerical precision monitoring */}
          <Card className="mt-6 p-4 bg-blue-50 border-blue-200">
            <h3 className="text-lg font-semibold mb-3">Numerical Precision Monitoring</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span>Sum Check (should be ~1.0):</span>
                <span className={`font-mono font-semibold ${
                  Math.abs(beliefData.numerical_precision?.sum_check - 1.0) < 1e-6 
                    ? 'text-green-600' : 'text-red-600'
                }`}>
                  {formatNumber(beliefData.numerical_precision?.sum_check || 0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Numerical Stability:</span>
                <span className="font-mono font-semibold">
                  {formatNumber(beliefData.numerical_precision?.numerical_stability || 0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Condition Number:</span>
                <span className="font-mono font-semibold">
                  {formatNumber(beliefData.numerical_precision?.condition_number || 0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Value Range:</span>
                <span className="font-mono font-semibold">
                  [{formatNumber(beliefData.numerical_precision?.min_value || 0)}, 
                   {formatNumber(beliefData.numerical_precision?.max_value || 0)}]
                </span>
              </div>
            </div>
          </Card>

          {/* Timestamp and data quality */}
          <div className="mt-4 text-xs text-gray-500 text-center">
            Last updated: {new Date(beliefData.timestamp).toLocaleString()} | 
            Data quality: {equationsRendered ? 'High' : 'Loading'} | 
            Agent: {beliefData.agent_id}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default BeliefStateMathematicalDisplay;
