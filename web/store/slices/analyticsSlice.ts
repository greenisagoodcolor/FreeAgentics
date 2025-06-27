import { createSlice, PayloadAction } from '@reduxjs/toolkit';

// Types from PRD
export interface AnalyticsWidget {
  id: string;
  type: 'metric' | 'chart' | 'heatmap' | 'timeline' | 'sankey' | 'network';
  title: string;
  size: 'small' | 'medium' | 'large';
  position: { x: number; y: number };
  config: Record<string, any>;
  refreshInterval?: number;
  dataSource: string; // Which metric to display
  isLoading: boolean;
  error?: string;
}

export interface MetricData {
  id: string;
  name: string;
  value: number | string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: number;
  unit?: string;
  history?: Array<{ timestamp: number; value: number }>;
  lastUpdated: number;
}

export interface AnalyticsSnapshot {
  timestamp: number;
  metrics: Record<string, MetricData>;
  agentMetrics: Record<string, Record<string, number>>; // agentId -> metricName -> value
  conversationMetrics: {
    totalMessages: number;
    messagesPerMinute: number;
    activeConversations: number;
    averageResponseTime: number;
  };
  knowledgeMetrics: {
    totalNodes: number;
    totalEdges: number;
    averageConfidence: number;
    knowledgeDiversity: number; // Shannon entropy
  };
}

interface AnalyticsState {
  widgets: Record<string, AnalyticsWidget>;
  metrics: Record<string, MetricData>;
  snapshots: AnalyticsSnapshot[];
  widgetLayout: string[]; // Widget IDs in order
  gridColumns: number;
  isRecording: boolean;
  recordingStartTime: number | null;
  selectedTimeRange: {
    start: number;
    end: number;
  };
}

// Default widgets from PRD
const defaultWidgets: Record<string, AnalyticsWidget> = {
  conversationRate: {
    id: 'conversationRate',
    type: 'chart',
    title: 'Conversation Rate',
    size: 'small',
    position: { x: 0, y: 0 },
    config: {
      chartType: 'line',
      showTrend: true,
      color: '#10B981',
    },
    dataSource: 'messagesPerMinute',
    isLoading: false,
  },
  activeAgents: {
    id: 'activeAgents',
    type: 'chart',
    title: 'Active Agents',
    size: 'small',
    position: { x: 4, y: 0 },
    config: {
      chartType: 'pie',
      showLegend: true,
    },
    dataSource: 'agentStates',
    isLoading: false,
  },
  knowledgeDiversity: {
    id: 'knowledgeDiversity',
    type: 'metric',
    title: 'Knowledge Diversity',
    size: 'small',
    position: { x: 8, y: 0 },
    config: {
      format: 'percentage',
      showSparkline: true,
    },
    dataSource: 'knowledgeDiversity',
    isLoading: false,
  },
  beliefConfidence: {
    id: 'beliefConfidence',
    type: 'chart',
    title: 'Belief Confidence Distribution',
    size: 'medium',
    position: { x: 0, y: 1 },
    config: {
      chartType: 'histogram',
      bins: 10,
    },
    dataSource: 'beliefConfidence',
    isLoading: false,
  },
  responseTime: {
    id: 'responseTime',
    type: 'chart',
    title: 'Response Time by Agent',
    size: 'medium',
    position: { x: 6, y: 1 },
    config: {
      chartType: 'boxplot',
      showOutliers: true,
    },
    dataSource: 'agentResponseTimes',
    isLoading: false,
  },
  turnTaking: {
    id: 'turnTaking',
    type: 'sankey',
    title: 'Conversation Flow',
    size: 'large',
    position: { x: 0, y: 2 },
    config: {
      nodeWidth: 15,
      nodePadding: 10,
    },
    dataSource: 'conversationFlow',
    isLoading: false,
  },
};

const initialState: AnalyticsState = {
  widgets: defaultWidgets,
  metrics: {},
  snapshots: [],
  widgetLayout: Object.keys(defaultWidgets),
  gridColumns: 12,
  isRecording: false,
  recordingStartTime: null,
  selectedTimeRange: {
    start: Date.now() - 3600000, // Last hour
    end: Date.now(),
  },
};

const analyticsSlice = createSlice({
  name: 'analytics',
  initialState,
  reducers: {
    // Widget management
    addWidget: (state, action: PayloadAction<Omit<AnalyticsWidget, 'isLoading'>>) => {
      const widget: AnalyticsWidget = {
        ...action.payload,
        isLoading: false,
      };
      state.widgets[widget.id] = widget;
      state.widgetLayout.push(widget.id);
    },

    removeWidget: (state, action: PayloadAction<string>) => {
      const widgetId = action.payload;
      delete state.widgets[widgetId];
      state.widgetLayout = state.widgetLayout.filter(id => id !== widgetId);
    },

    updateWidget: (state, action: PayloadAction<{
      id: string;
      updates: Partial<AnalyticsWidget>;
    }>) => {
      const { id, updates } = action.payload;
      if (state.widgets[id]) {
        state.widgets[id] = {
          ...state.widgets[id],
          ...updates,
        };
      }
    },

    moveWidget: (state, action: PayloadAction<{
      id: string;
      position: { x: number; y: number };
    }>) => {
      const { id, position } = action.payload;
      if (state.widgets[id]) {
        state.widgets[id].position = position;
      }
    },

    resizeWidget: (state, action: PayloadAction<{
      id: string;
      size: AnalyticsWidget['size'];
    }>) => {
      const { id, size } = action.payload;
      if (state.widgets[id]) {
        state.widgets[id].size = size;
      }
    },

    // Widget layout
    updateWidgetLayout: (state, action: PayloadAction<string[]>) => {
      state.widgetLayout = action.payload;
    },

    resetWidgetLayout: (state) => {
      state.widgets = defaultWidgets;
      state.widgetLayout = Object.keys(defaultWidgets);
    },

    // Metrics
    updateMetric: (state, action: PayloadAction<MetricData>) => {
      state.metrics[action.payload.id] = action.payload;
    },

    batchUpdateMetrics: (state, action: PayloadAction<MetricData[]>) => {
      action.payload.forEach(metric => {
        state.metrics[metric.id] = metric;
      });
    },

    // Recording
    startRecording: (state) => {
      state.isRecording = true;
      state.recordingStartTime = Date.now();
    },

    stopRecording: (state) => {
      state.isRecording = false;
    },

    addSnapshot: (state, action: PayloadAction<Omit<AnalyticsSnapshot, 'timestamp'>>) => {
      const snapshot: AnalyticsSnapshot = {
        ...action.payload,
        timestamp: Date.now(),
      };
      state.snapshots.push(snapshot);
      
      // Keep only last 1000 snapshots
      if (state.snapshots.length > 1000) {
        state.snapshots = state.snapshots.slice(-1000);
      }
    },

    clearSnapshots: (state) => {
      state.snapshots = [];
    },

    // Time range
    setTimeRange: (state, action: PayloadAction<{
      start: number;
      end: number;
    }>) => {
      state.selectedTimeRange = action.payload;
    },

    // Widget loading states
    setWidgetLoading: (state, action: PayloadAction<{
      id: string;
      isLoading: boolean;
    }>) => {
      const { id, isLoading } = action.payload;
      if (state.widgets[id]) {
        state.widgets[id].isLoading = isLoading;
      }
    },

    setWidgetError: (state, action: PayloadAction<{
      id: string;
      error: string | undefined;
    }>) => {
      const { id, error } = action.payload;
      if (state.widgets[id]) {
        state.widgets[id].error = error;
      }
    },

    // Grid configuration
    setGridColumns: (state, action: PayloadAction<number>) => {
      state.gridColumns = action.payload;
    },

    // Batch operations
    importWidgetConfiguration: (state, action: PayloadAction<{
      widgets: Record<string, AnalyticsWidget>;
      layout: string[];
    }>) => {
      state.widgets = action.payload.widgets;
      state.widgetLayout = action.payload.layout;
    },
  },
});

export const {
  addWidget,
  removeWidget,
  updateWidget,
  moveWidget,
  resizeWidget,
  updateWidgetLayout,
  resetWidgetLayout,
  updateMetric,
  batchUpdateMetrics,
  startRecording,
  stopRecording,
  addSnapshot,
  clearSnapshots,
  setTimeRange,
  setWidgetLoading,
  setWidgetError,
  setGridColumns,
  importWidgetConfiguration,
} = analyticsSlice.actions;

export default analyticsSlice.reducer; 