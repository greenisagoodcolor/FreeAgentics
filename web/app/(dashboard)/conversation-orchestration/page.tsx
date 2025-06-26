"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { 
  Settings, 
  Play, 
  Pause, 
  RotateCcw, 
  Save, 
  Upload, 
  Download,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff,
  Zap,
  Clock,
  Users,
  Brain
} from 'lucide-react';
import { 
  ConversationPreset, 
  ConversationOrchestrationState,
  ConversationPresetValidation 
} from '@/lib/types';
import { ConversationPresetValidator } from '@/lib/conversation-preset-validator';
import PresetSelector from '@/components/conversation-orchestration/preset-selector';
import { ResponseDynamicsControls } from '@/components/conversation-orchestration/response-dynamics-controls';
import { TimingControls } from '@/components/conversation-orchestration/timing-controls';
import { AdvancedControls } from '@/components/conversation-orchestration/advanced-controls';
import { RealTimePreview } from '@/components/conversation-orchestration/real-time-preview';
import { ChangeHistory } from '@/components/conversation-orchestration/change-history';
import { useDebounce } from '@/hooks/useDebounce';

// Conversation Orchestration Control Panel
// Implements comprehensive real-time control for conversation parameters

export default function ConversationOrchestrationPage() {
  // State management
  const [state, setState] = useState<ConversationOrchestrationState>({
    currentPreset: null,
    isPreviewMode: false,
    hasUnsavedChanges: false,
    isAdvancedMode: false,
    activePreview: null,
    validationResult: null,
    history: [],
    abTestResults: undefined
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Debounced validation to prevent excessive validation calls
  const debouncedPreset = useDebounce(state.currentPreset, 500);

  /**
   * Create a complete preset from partial data
   */
  const createCompletePreset = (partial: Partial<ConversationPreset>, category: string): ConversationPreset => {
    const now = new Date().toISOString();
    return {
      id: `preset-${Date.now()}`,
      name: `${category.charAt(0).toUpperCase() + category.slice(1)} Preset`,
      description: `${category} conversation orchestration preset`,
      category: category as ConversationPreset['category'],
      version: '1.0.0',
      createdAt: now,
      updatedAt: now,
      createdBy: 'user',
      
      // Default values with overrides from partial
      responseDynamics: {
        turnTaking: {
          enabled: true,
          maxConcurrentResponses: 2,
          responseThreshold: 0.6,
          mentionResponseProbability: 0.9,
          conversationStarterResponseRate: 0.85,
          ...partial.responseDynamics?.turnTaking
        },
        agentSelection: {
          autoSelectRespondents: true,
          selectionStrategy: 'engagement_based',
          diversityBonus: 0.5,
          expertiseWeight: 0.6,
          maxSpeakersPerTurn: 2,
          ...partial.responseDynamics?.agentSelection
        },
        responseGeneration: {
          maxKnowledgeEntries: 20,
          includeAgentKnowledge: true,
          streamResponse: true,
          responseLength: 'medium',
          creativityLevel: 0.5,
          coherenceWeight: 0.7,
          ...partial.responseDynamics?.responseGeneration
        },
        ...partial.responseDynamics
      },
      
      timingControls: {
        responseDelay: {
          type: 'range',
          fixedDelay: 800,
          minDelay: 500,
          maxDelay: 2000,
          adaptiveFactors: {
            messageLength: true,
            agentProcessingTime: true,
            conversationPace: true
          },
          ...partial.timingControls?.responseDelay
        },
        conversationFlow: {
          maxAutonomousMessages: 25,
          stallDetectionTimeout: 8000,
          stallRecoveryStrategy: 'prompt_random',
          turnTimeoutDuration: 20000,
          pauseBetweenTurns: 300,
          ...partial.timingControls?.conversationFlow
        },
        realTimeControls: {
          enableTypingIndicators: true,
          typingIndicatorDelay: 500,
          messagePreviewEnabled: true,
          ghostMessageDuration: 5000,
          ...partial.timingControls?.realTimeControls
        },
        ...partial.timingControls
      },
      
      advancedParameters: {
        conversationDynamics: {
          topicDriftAllowance: 0.3,
          contextWindowSize: 10,
          semanticCoherenceThreshold: 0.7,
          emotionalToneConsistency: 0.6,
          ...partial.advancedParameters?.conversationDynamics
        },
        agentBehavior: {
          personalityInfluence: 0.7,
          expertiseBoost: 0.5,
          randomnessInjection: 0.2,
          memoryRetentionFactor: 0.8,
          ...partial.advancedParameters?.agentBehavior
        },
        qualityControls: {
          minimumResponseQuality: 0.6,
          duplicateDetectionSensitivity: 0.7,
          relevanceThreshold: 0.8,
          factualAccuracyWeight: 0.9,
          ...partial.advancedParameters?.qualityControls
        },
        performanceOptimization: {
          enableCaching: true,
          cacheExpirationTime: 300000,
          maxConcurrentGenerations: 3,
          resourceThrottling: true,
          ...partial.advancedParameters?.performanceOptimization
        },
        ...partial.advancedParameters
      },
      
      safetyConstraints: {
        enableSafetyChecks: true,
        maxResponseLength: 2000,
        contentFiltering: true,
        rateLimiting: {
          enabled: true,
          maxRequestsPerMinute: 60,
          maxRequestsPerHour: 1000
        },
        emergencyStopConditions: ['high_error_rate', 'quality_degradation', 'rate_limit_exceeded'],
        ...partial.safetyConstraints
      },
      
      monitoring: {
        enableMetrics: true,
        trackPerformance: true,
        logLevel: 'info',
        metricsRetentionDays: 30,
        alertThresholds: {
          responseTimeMs: 5000,
          errorRate: 0.1,
          qualityScore: 0.5
        },
        ...partial.monitoring
      },
      
      ...partial
    };
  };

  /**
   * Load default preset
   */
  const loadDefaultPreset = useCallback(async () => {
    setIsLoading(true);
    try {
      // In production, this would load from API
      const defaultPresets = ConversationPresetValidator.getDefaultPresets();
      const balancedPreset = createCompletePreset(defaultPresets.balanced!, 'balanced');
      
      setState(prev => ({
        ...prev,
        currentPreset: balancedPreset,
        hasUnsavedChanges: false
      }));
    } catch (err) {
      setError('Failed to load default preset');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load initial preset
  useEffect(() => {
    loadDefaultPreset();
  }, [loadDefaultPreset]);

  // Validate preset when it changes (debounced)
  useEffect(() => {
    const validatePreset = async () => {
      if (debouncedPreset) {
        const validation = await ConversationPresetValidator.validatePreset(debouncedPreset);
        setState(prev => ({
          ...prev,
          validationResult: validation
        }));
      }
    };
    
    validatePreset();
  }, [debouncedPreset]);

  /**
   * Update preset with new values
   */
  const updatePreset = useCallback((updates: Partial<ConversationPreset>) => {
    setState(prev => {
      if (!prev.currentPreset) return prev;
      
      const updatedPreset = {
        ...prev.currentPreset,
        ...updates,
        updatedAt: new Date().toISOString()
      };
      
      return {
        ...prev,
        currentPreset: updatedPreset,
        hasUnsavedChanges: true
      };
    });
  }, []);

  /**
   * Save current preset
   */
  const savePreset = useCallback(async () => {
    if (!state.currentPreset) return;
    
    setIsLoading(true);
    try {
      // In production, this would save to API
      console.log('Saving preset:', state.currentPreset);
      
      setState(prev => ({
        ...prev,
        hasUnsavedChanges: false
      }));
      
      setSuccessMessage('Preset saved successfully');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError('Failed to save preset');
    } finally {
      setIsLoading(false);
    }
  }, [state.currentPreset]);

  /**
   * Toggle preview mode
   */
  const togglePreviewMode = useCallback(() => {
    setState(prev => ({
      ...prev,
      isPreviewMode: !prev.isPreviewMode
    }));
  }, []);

  /**
   * Toggle advanced mode
   */
  const toggleAdvancedMode = useCallback(() => {
    setState(prev => ({
      ...prev,
      isAdvancedMode: !prev.isAdvancedMode
    }));
  }, []);

  /**
   * Reset to default preset
   */
  const resetPreset = useCallback(() => {
    loadDefaultPreset();
  }, [loadDefaultPreset]);

  /**
   * Get validation status color and icon
   */
  const getValidationStatus = () => {
    if (!state.validationResult) return { color: 'gray', icon: Settings };
    
    if (!state.validationResult.isValid) {
      return { color: 'red', icon: AlertTriangle };
    }
    
    if (state.validationResult.warnings.length > 0) {
      return { color: 'yellow', icon: AlertTriangle };
    }
    
    return { color: 'green', icon: CheckCircle };
  };

  const validationStatus = getValidationStatus();

  return (
    <div className="conversation-orchestration-page p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Conversation Orchestration</h1>
          <p className="text-muted-foreground">
            Fine-tune conversation parameters and dynamics in real-time
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Advanced Mode Toggle */}
          <div className="flex items-center gap-2">
            <Label htmlFor="advanced-mode">Advanced Mode</Label>
            <Switch
              id="advanced-mode"
              checked={state.isAdvancedMode}
              onCheckedChange={toggleAdvancedMode}
            />
          </div>
          
          {/* Preview Mode Toggle */}
          <div className="flex items-center gap-2">
            <Label htmlFor="preview-mode">Preview Mode</Label>
            <Switch
              id="preview-mode"
              checked={state.isPreviewMode}
              onCheckedChange={togglePreviewMode}
            />
            {state.isPreviewMode ? (
              <Eye className="h-4 w-4 text-blue-500" />
            ) : (
              <EyeOff className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between mb-6 p-4 bg-muted/50 rounded-lg">
        <div className="flex items-center gap-4">
          {/* Validation Status */}
          <div className="flex items-center gap-2">
            <validationStatus.icon 
              className={`h-5 w-5 ${
                validationStatus.color === 'red' ? 'text-red-500' :
                validationStatus.color === 'yellow' ? 'text-yellow-500' :
                validationStatus.color === 'green' ? 'text-green-500' :
                'text-muted-foreground'
              }`} 
            />
            <span className="text-sm font-medium">
              {state.validationResult?.isValid ? 'Valid Configuration' : 'Invalid Configuration'}
            </span>
            {state.validationResult && (
              <Badge variant={state.validationResult.isValid ? 'default' : 'destructive'}>
                {state.validationResult.riskLevel}
              </Badge>
            )}
          </div>
          
          {/* Unsaved Changes Indicator */}
          {state.hasUnsavedChanges && (
            <Badge variant="outline" className="text-orange-600">
              Unsaved Changes
            </Badge>
          )}
          
          {/* Current Preset */}
          {state.currentPreset && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Preset:</span>
              <Badge variant="secondary">{state.currentPreset.name}</Badge>
            </div>
          )}
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={resetPreset}
            disabled={isLoading}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            disabled={isLoading}
          >
            <Upload className="h-4 w-4 mr-2" />
            Import
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            disabled={isLoading || !state.currentPreset}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          
          <Button
            onClick={savePreset}
            disabled={isLoading || !state.hasUnsavedChanges}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Save className="h-4 w-4 mr-2" />
            Save Preset
          </Button>
        </div>
      </div>

      {/* Error and Success Messages */}
      {error && (
        <Alert className="mb-6" variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      {successMessage && (
        <Alert className="mb-6">
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>Success</AlertTitle>
          <AlertDescription>{successMessage}</AlertDescription>
        </Alert>
      )}

      {/* Validation Errors and Warnings */}
      {state.validationResult && (state.validationResult.errors.length > 0 || state.validationResult.warnings.length > 0) && (
        <div className="mb-6 space-y-2">
          {state.validationResult.errors.map((error, index) => (
            <Alert key={`error-${index}`} variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ))}
          
          {state.validationResult.warnings.map((warning, index) => (
            <Alert key={`warning-${index}`}>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{warning}</AlertDescription>
            </Alert>
          ))}
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Panels */}
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="presets" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="presets">Presets</TabsTrigger>
              <TabsTrigger value="response">Response</TabsTrigger>
              <TabsTrigger value="timing">Timing</TabsTrigger>
              {state.isAdvancedMode && <TabsTrigger value="advanced">Advanced</TabsTrigger>}
              <TabsTrigger value="history">History</TabsTrigger>
            </TabsList>
            
            <TabsContent value="presets" className="space-y-4">
              <PresetSelector
                currentPreset={state.currentPreset}
                onPresetChange={(preset) => {
                  setState(prev => ({
                    ...prev,
                    currentPreset: preset,
                    hasUnsavedChanges: false
                  }));
                }}
                onSavePreset={(name: string) => {
                  console.log('Save preset:', name);
                }}
                onLoadPreset={(preset) => {
                  setState(prev => ({
                    ...prev,
                    currentPreset: preset,
                    hasUnsavedChanges: false
                  }));
                }}
              />
            </TabsContent>
            
            <TabsContent value="response" className="space-y-4">
              {state.currentPreset && (
                <ResponseDynamicsControls
                  preset={state.currentPreset}
                  onUpdate={updatePreset}
                />
              )}
            </TabsContent>
            
            <TabsContent value="timing" className="space-y-4">
              {state.currentPreset && (
                <TimingControls
                  preset={state.currentPreset}
                  onUpdate={updatePreset}
                />
              )}
            </TabsContent>
            
            {state.isAdvancedMode && (
              <TabsContent value="advanced" className="space-y-4">
                {state.currentPreset && (
                  <AdvancedControls
                    preset={state.currentPreset}
                    onUpdate={updatePreset}
                  />
                )}
              </TabsContent>
            )}
            
            <TabsContent value="history" className="space-y-4">
              <ChangeHistory
                history={state.history}
              />
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Real-time Preview */}
        <div className="space-y-6">
          {state.currentPreset && (
            <RealTimePreview
              preview={state.activePreview}
              preset={state.currentPreset}
              isPreviewMode={state.isPreviewMode}
              onRollback={() => {
                // Reset to last saved state
                loadDefaultPreset();
                setState(prev => ({
                  ...prev,
                  hasUnsavedChanges: false
                }));
              }}
            />
          )}
          
          {/* Quick Stats */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Quick Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {state.currentPreset && (
                <>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Max Concurrent</span>
                    <Badge variant="outline">
                      {state.currentPreset.responseDynamics.turnTaking.maxConcurrentResponses}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Response Threshold</span>
                    <Badge variant="outline">
                      {(state.currentPreset.responseDynamics.turnTaking.responseThreshold * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Max Messages</span>
                    <Badge variant="outline">
                      {state.currentPreset.timingControls.conversationFlow.maxAutonomousMessages}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Delay Range</span>
                    <Badge variant="outline">
                      {state.currentPreset.timingControls.responseDelay.minDelay}-
                      {state.currentPreset.timingControls.responseDelay.maxDelay}ms
                    </Badge>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 