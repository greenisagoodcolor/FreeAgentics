import React, { useState, useEffect, useRef } from 'react';
import { motion, Reorder } from 'framer-motion';
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, 
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';
import { useAppSelector } from '@/store/hooks';
import { 
  Activity, Users, Brain, Clock, TrendingUp, Zap, 
  MoreVertical, Maximize2, RotateCcw, Download, Settings
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface WidgetConfig {
  id: string;
  type: 'metric' | 'chart' | 'heatmap' | 'timeline';
  title: string;
  size: 'small' | 'medium' | 'large';
  position: { x: number; y: number };
  refreshInterval?: number;
}

interface MetricData {
  id: string;
  name: string;
  value: number;
  lastUpdated: number;
  trend?: number;
  unit?: string;
}

const AnalyticsWidgetSystem: React.FC = () => {
  const [widgets, setWidgets] = useState<WidgetConfig[]>([
    { id: 'conversation-rate', type: 'metric', title: 'Conversation Rate', size: 'small', position: { x: 0, y: 0 } },
    { id: 'active-agents', type: 'chart', title: 'Active Agents', size: 'medium', position: { x: 1, y: 0 } },
    { id: 'knowledge-diversity', type: 'chart', title: 'Knowledge Diversity', size: 'medium', position: { x: 2, y: 0 } },
    { id: 'belief-confidence', type: 'chart', title: 'Belief Confidence', size: 'large', position: { x: 0, y: 1 } },
    { id: 'response-time', type: 'chart', title: 'Response Time', size: 'medium', position: { x: 2, y: 1 } },
    { id: 'turn-taking', type: 'chart', title: 'Turn Taking', size: 'large', position: { x: 0, y: 2 } },
  ]);

  const [expandedWidget, setExpandedWidget] = useState<string | null>(null);
  const [refreshTimestamp, setRefreshTimestamp] = useState(Date.now());

  // Redux state
  const agents = useAppSelector(state => state.agents.agents);
  const conversations = useAppSelector(state => state.conversations.conversations);
  const knowledgeGraph = useAppSelector(state => state.knowledge.graph);
  const analytics = useAppSelector(state => state.analytics);

  // Auto-refresh effect
  useEffect(() => {
    const interval = setInterval(() => {
      setRefreshTimestamp(Date.now());
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, []);

  // Calculate real-time metrics
  const calculateMetrics = (): Record<string, MetricData> => {
    const totalAgents = Object.keys(agents).length;
    const activeAgents = Object.values(agents).filter(a => a.status === 'active').length;
    const totalMessages = Object.values(conversations).reduce((sum, conv) => sum + conv.messages.length, 0);
    const totalKnowledge = Object.keys(knowledgeGraph.nodes).length;
    
    // Calculate message rate (messages per minute)
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const recentMessages = Object.values(conversations)
      .flatMap(conv => conv.messages)
      .filter(msg => msg.timestamp > oneMinuteAgo);
    
    // Calculate average confidence
    const confidences = Object.values(knowledgeGraph.nodes).map(node => node.confidence);
    const avgConfidence = confidences.length > 0 ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0;
    
    // Calculate Shannon entropy for knowledge diversity
    const typeCount = Object.values(knowledgeGraph.nodes).reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const total = Object.values(typeCount).reduce((a, b) => a + b, 0);
    const entropy = total > 0 ? -Object.values(typeCount)
      .map(count => count / total)
      .filter(p => p > 0)
      .reduce((acc, p) => acc + p * Math.log2(p), 0) : 0;

    return {
      conversationRate: {
        id: 'conversation-rate',
        name: 'Messages/Min',
        value: recentMessages.length,
        lastUpdated: now,
        unit: 'msg/min',
        trend: Math.random() * 20 - 10 // Mock trend
      },
      activeAgents: {
        id: 'active-agents',
        name: 'Active Agents',
        value: activeAgents,
        lastUpdated: now,
        unit: 'agents'
      },
      totalMessages: {
        id: 'total-messages',
        name: 'Total Messages',
        value: totalMessages,
        lastUpdated: now,
        unit: 'messages'
      },
      knowledgeDiversity: {
        id: 'knowledge-diversity',
        name: 'Knowledge Entropy',
        value: entropy,
        lastUpdated: now,
        unit: 'bits'
      },
      avgConfidence: {
        id: 'avg-confidence',
        name: 'Avg Confidence',
        value: avgConfidence,
        lastUpdated: now,
        unit: '%'
      },
      totalKnowledge: {
        id: 'total-knowledge',
        name: 'Knowledge Nodes',
        value: totalKnowledge,
        lastUpdated: now,
        unit: 'nodes'
      }
    };
  };

  const metrics = calculateMetrics();

  // Generate chart data
  const generateChartData = (widgetId: string) => {
    switch (widgetId) {
      case 'active-agents':
        return Object.values(agents).reduce((acc, agent) => {
          acc[agent.status] = (acc[agent.status] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);

      case 'knowledge-diversity':
        return Object.values(knowledgeGraph.nodes).reduce((acc, node) => {
          acc[node.type] = (acc[node.type] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);

      case 'belief-confidence':
        const confidenceBuckets = { 'Low (0-0.3)': 0, 'Medium (0.3-0.7)': 0, 'High (0.7-1.0)': 0 };
        Object.values(knowledgeGraph.nodes).forEach(node => {
          if (node.confidence <= 0.3) confidenceBuckets['Low (0-0.3)']++;
          else if (node.confidence <= 0.7) confidenceBuckets['Medium (0.3-0.7)']++;
          else confidenceBuckets['High (0.7-1.0)']++;
        });
        return confidenceBuckets;

      case 'response-time':
        // Generate mock response time data
        return Array.from({ length: 10 }, (_, i) => ({
          time: `${i * 6}:00`,
          avgResponse: Math.random() * 2000 + 500,
          p95Response: Math.random() * 5000 + 1000,
        }));

      case 'turn-taking':
        // Generate mock turn-taking flow data
        const agentNames = Object.values(agents).slice(0, 5).map(a => a.name);
        return agentNames.map((name, i) => ({
          agent: name,
          initiates: Math.floor(Math.random() * 50) + 10,
          responds: Math.floor(Math.random() * 80) + 20,
        }));

      default:
        return {};
    }
  };

  // Widget components
  const MetricWidget: React.FC<{ config: WidgetConfig; metric: MetricData }> = ({ config, metric }) => (
    <Card className="widget-container h-full">
      <CardHeader className="pb-2">
        <CardTitle className="widget-title flex items-center justify-between">
          {config.title}
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
              <MoreVertical className="h-3 w-3" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center">
          <div className="text-3xl font-bold font-mono text-[var(--text-primary)]">
            {metric.value.toFixed(metric.unit === '%' ? 1 : 0)}
          </div>
          <div className="text-sm text-[var(--text-secondary)]">
            {metric.unit}
          </div>
          {metric.trend && (
            <div className={`text-xs mt-1 ${metric.trend > 0 ? 'text-[var(--success)]' : 'text-[var(--error)]'}`}>
              {metric.trend > 0 ? '+' : ''}{metric.trend.toFixed(1)}%
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );

  const ChartWidget: React.FC<{ config: WidgetConfig }> = ({ config }) => {
    const data = generateChartData(config.id);
    
    const renderChart = () => {
      switch (config.id) {
        case 'active-agents':
        case 'knowledge-diversity':
        case 'belief-confidence':
          const pieData = Object.entries(data).map(([key, value], index) => ({
            name: key,
            value: value as number,
            fill: ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'][index % 5]
          }));
          
          return (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={config.size === 'small' ? 30 : 40}
                  outerRadius={config.size === 'small' ? 60 : 80}
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                  fontSize={12}
                  fill="#8884d8"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          );

        case 'response-time':
          return (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={data as any[]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--bg-secondary)', 
                    border: '1px solid var(--bg-tertiary)',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="avgResponse" stroke="#4F46E5" strokeWidth={2} name="Avg Response" />
                <Line type="monotone" dataKey="p95Response" stroke="#F59E0B" strokeWidth={2} name="95th Percentile" />
              </LineChart>
            </ResponsiveContainer>
          );

        case 'turn-taking':
          return (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data as any[]} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" stroke="#666" fontSize={12} />
                <YAxis dataKey="agent" type="category" stroke="#666" fontSize={12} width={80} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--bg-secondary)', 
                    border: '1px solid var(--bg-tertiary)',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Bar dataKey="initiates" fill="#4F46E5" name="Initiates" />
                <Bar dataKey="responds" fill="#10B981" name="Responds" />
              </BarChart>
            </ResponsiveContainer>
          );

        default:
          return <div className="text-center text-[var(--text-secondary)]">No data available</div>;
      }
    };

    return (
      <Card className="widget-container h-full">
        <CardHeader className="pb-2">
          <CardTitle className="widget-title flex items-center justify-between">
            {config.title}
            <div className="flex items-center gap-1">
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-6 w-6 p-0"
                onClick={() => setExpandedWidget(expandedWidget === config.id ? null : config.id)}
              >
                <Maximize2 className="h-3 w-3" />
              </Button>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                <MoreVertical className="h-3 w-3" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {renderChart()}
        </CardContent>
      </Card>
    );
  };

  const getWidgetSizeClass = (size: string) => {
    switch (size) {
      case 'small': return 'col-span-1 row-span-1';
      case 'medium': return 'col-span-2 row-span-1';
      case 'large': return 'col-span-3 row-span-2';
      default: return 'col-span-1 row-span-1';
    }
  };

  return (
    <div className="p-6 h-full bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="font-ui text-xl font-semibold text-[var(--text-primary)]">
            Analytics Dashboard
          </h2>
          <p className="font-ui text-sm text-[var(--text-secondary)] mt-1">
            Real-time system metrics and insights
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]">
            <Activity className="w-3 h-3 mr-1" />
            Live
          </Badge>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setRefreshTimestamp(Date.now())}
            className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Widget Grid */}
      <div className="grid grid-cols-4 gap-4 h-[calc(100%-120px)] auto-rows-fr">
        {widgets.map((widget) => (
          <motion.div
            key={widget.id}
            className={getWidgetSizeClass(widget.size)}
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            {widget.type === 'metric' ? (
              <MetricWidget 
                config={widget} 
                metric={metrics[widget.id.replace('-', '')] || metrics.conversationRate} 
              />
            ) : (
              <ChartWidget config={widget} />
            )}
          </motion.div>
        ))}
      </div>

      {/* Expanded Widget Modal */}
      {expandedWidget && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-8"
          onClick={() => setExpandedWidget(null)}
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0.9 }}
            className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded-lg p-6 max-w-4xl w-full h-[80vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-[var(--text-primary)]">
                {widgets.find(w => w.id === expandedWidget)?.title}
              </h3>
              <Button 
                variant="ghost"
                onClick={() => setExpandedWidget(null)}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              >
                ✕
              </Button>
            </div>
            <div className="h-[calc(100%-60px)]">
              <ChartWidget 
                config={widgets.find(w => w.id === expandedWidget)!} 
              />
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default AnalyticsWidgetSystem; 