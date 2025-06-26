"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { ConversationPresetHistory, ABTestResults } from '@/lib/types';
import { History, RotateCcw, GitBranch, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

interface ChangeHistoryProps {
  history: ConversationPresetHistory[];
  abTestResults?: ABTestResults;
  onRollback?: (historyItem: ConversationPresetHistory) => void;
  onStartABTest?: () => void;
  className?: string;
}

export function ChangeHistory({
  history,
  abTestResults,
  onRollback,
  onStartABTest,
  className = ""
}: ChangeHistoryProps) {
  const [selectedHistory, setSelectedHistory] = useState<ConversationPresetHistory | null>(null);

  /**
   * Format timestamp for display
   */
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  /**
   * Get risk level color
   */
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'high': return 'text-orange-600';
      case 'critical': return 'text-red-600';
      default: return 'text-muted-foreground';
    }
  };

  /**
   * Get performance trend icon
   */
  const getPerformanceTrend = (metrics: any) => {
    if (!metrics) return null;
    
    const score = (metrics.responseTime + metrics.qualityScore + metrics.userSatisfaction) / 3;
    if (score > 0.7) return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (score < 0.3) return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Change History
          <Badge variant="outline" className="ml-auto">
            {history.length} changes
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* A/B Test Results */}
        {abTestResults && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <GitBranch className="h-4 w-4" />
              <h4 className="font-medium text-sm">A/B Test Results</h4>
              <Badge variant={abTestResults.recommendation === 'A' ? 'default' : 'secondary'}>
                Recommend: {abTestResults.recommendation}
              </Badge>
            </div>
            
            <Card className="border-dashed">
              <CardContent className="p-4 space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  {abTestResults.metrics.map((metric, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Badge variant={metric.variant === 'A' ? 'default' : 'secondary'}>
                          Variant {metric.variant}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {metric.sampleSize} samples
                        </span>
                      </div>
                      
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span>Response Time:</span>
                          <span>{metric.averageResponseTime.toFixed(0)}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Quality Score:</span>
                          <span>{(metric.qualityScore * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Satisfaction:</span>
                          <span>{(metric.userSatisfaction * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Error Rate:</span>
                          <span>{(metric.errorRate * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between text-xs">
                  <span>Statistical Significance:</span>
                  <Badge variant={abTestResults.statisticalSignificance > 0.95 ? 'default' : 'outline'}>
                    {(abTestResults.statisticalSignificance * 100).toFixed(1)}%
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between text-xs">
                  <span>Confidence Interval:</span>
                  <span>
                    {abTestResults.confidenceInterval.lower.toFixed(2)} - {abTestResults.confidenceInterval.upper.toFixed(2)}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Start A/B Test Button */}
        {onStartABTest && !abTestResults && (
          <Button
            variant="outline"
            size="sm"
            onClick={onStartABTest}
            className="w-full"
          >
            <GitBranch className="h-4 w-4 mr-2" />
            Start A/B Test
          </Button>
        )}

        {/* History List */}
        <div className="space-y-3">
          <h4 className="font-medium text-sm">Recent Changes</h4>
          
          {history.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              No changes recorded yet
            </div>
          ) : (
            <ScrollArea className="h-64">
              <div className="space-y-3">
                {history.map((item) => (
                  <Card 
                    key={item.id}
                    className={`cursor-pointer transition-all ${
                      selectedHistory?.id === item.id ? 'ring-2 ring-primary' : ''
                    }`}
                    onClick={() => setSelectedHistory(item)}
                  >
                    <CardContent className="p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            v{item.version}
                          </Badge>
                          {getPerformanceTrend(item.performanceMetrics)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {formatTimestamp(item.appliedAt)}
                        </div>
                      </div>
                      
                      <div className="text-sm">
                        <div className="flex items-center justify-between mb-1">
                          <span>Changes:</span>
                          <Badge variant="secondary" className="text-xs">
                            {item.changes.changes.length}
                          </Badge>
                        </div>
                        
                        <div className="space-y-1">
                          {item.changes.changes.slice(0, 2).map((change, index) => (
                            <div key={index} className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground truncate">
                                {change.path}
                              </span>
                              <Badge 
                                variant="outline" 
                                className={`text-xs ${getRiskColor(change.riskLevel)}`}
                              >
                                {change.riskLevel}
                              </Badge>
                            </div>
                          ))}
                          {item.changes.changes.length > 2 && (
                            <div className="text-xs text-muted-foreground">
                              +{item.changes.changes.length - 2} more changes
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {item.performanceMetrics && (
                        <div className="pt-2 border-t space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span>Quality Score:</span>
                            <span>{(item.performanceMetrics.qualityScore * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex items-center justify-between text-xs">
                            <span>Response Time:</span>
                            <span>{item.performanceMetrics.responseTime.toFixed(0)}ms</span>
                          </div>
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between pt-2">
                        <Badge variant="outline" className="text-xs">
                          {item.appliedBy}
                        </Badge>
                        {item.rollbackAvailable && onRollback && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              onRollback(item);
                            }}
                            className="h-6 text-xs"
                          >
                            <RotateCcw className="h-3 w-3 mr-1" />
                            Rollback
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          )}
        </div>

        {/* Selected History Details */}
        {selectedHistory && (
          <div className="space-y-3">
            <Separator />
            <h4 className="font-medium text-sm">Change Details</h4>
            
            <Card className="border-dashed">
              <CardContent className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <Badge variant="outline">
                    Version {selectedHistory.version}
                  </Badge>
                  <div className="text-xs text-muted-foreground">
                    {formatTimestamp(selectedHistory.appliedAt)}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h5 className="font-medium text-xs">Summary</h5>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>Total Changes: {selectedHistory.changes.summary.totalChanges}</div>
                    <div>Affected Categories: {selectedHistory.changes.summary.affectedCategories.length}</div>
                  </div>
                  
                  <div className="space-y-1">
                    {Object.entries(selectedHistory.changes.summary.riskDistribution).map(([risk, count]) => (
                      <div key={risk} className="flex items-center justify-between text-xs">
                        <span className={getRiskColor(risk)}>{risk} risk:</span>
                        <span>{count} changes</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                {selectedHistory.performanceMetrics && (
                  <div className="space-y-2">
                    <h5 className="font-medium text-xs">Performance Impact</h5>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>Response Time: {selectedHistory.performanceMetrics.responseTime.toFixed(0)}ms</div>
                      <div>Quality Score: {(selectedHistory.performanceMetrics.qualityScore * 100).toFixed(0)}%</div>
                      <div>User Satisfaction: {(selectedHistory.performanceMetrics.userSatisfaction * 100).toFixed(0)}%</div>
                      <div>Error Rate: {(selectedHistory.performanceMetrics.errorRate * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 