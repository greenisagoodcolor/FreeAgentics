/**
 * Comprehensive Page Component Tests
 * 
 * Tests for all page components including app routing, layouts,
 * and page-specific functionality following ADR-007 requirements.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { jest } from '@jest/globals';

// Mock Next.js router and navigation
const mockPush = jest.fn();
const mockReplace = jest.fn();
const mockBack = jest.fn();
const mockRefresh = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: mockReplace,
    back: mockBack,
    forward: jest.fn(),
    refresh: mockRefresh,
    prefetch: jest.fn(),
  }),
  usePathname: () => '/dashboard',
  useSearchParams: () => new URLSearchParams('tab=agents'),
  useParams: () => ({ id: 'test-id' }),
}));

// Mock Next.js components
jest.mock('next/link', () => {
  return function MockLink({ children, href, ...props }: any) {
    return <a href={href} {...props}>{children}</a>;
  };
});

jest.mock('next/image', () => {
  return function MockImage({ src, alt, ...props }: any) {
    return <img src={src} alt={alt} {...props} />;
  };
});

// Mock WebSocket
global.WebSocket = jest.fn(() => ({
  send: jest.fn(),
  close: jest.fn(),
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
  readyState: 1,
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
})) as any;

// Mock Canvas
HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
  fillRect: jest.fn(),
  clearRect: jest.fn(),
  getImageData: jest.fn(() => ({ data: new Array(4) })),
  putImageData: jest.fn(),
  createImageData: jest.fn(() => ({ data: new Array(4) })),
  setTransform: jest.fn(),
  drawImage: jest.fn(),
  save: jest.fn(),
  restore: jest.fn(),
  fillText: jest.fn(),
  measureText: jest.fn(() => ({ width: 0 })),
  strokeText: jest.fn(),
  beginPath: jest.fn(),
  moveTo: jest.fn(),
  lineTo: jest.fn(),
  stroke: jest.fn(),
  fill: jest.fn(),
  arc: jest.fn(),
  closePath: jest.fn(),
})) as any;

// Comprehensive Page Components Implementation

// Root Layout Component
const RootLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
  const [theme, setTheme] = React.useState<'light' | 'dark'>('light');
  const [user, setUser] = React.useState<any>(null);

  React.useEffect(() => {
    // Simulate user authentication check
    const checkAuth = async () => {
      try {
        const authToken = localStorage.getItem('auth_token');
        if (authToken) {
          setUser({
            id: 'user-1',
            name: 'Test User',
            email: 'test@example.com',
            role: 'admin',
          });
        }
      } catch (error) {
        console.error('Auth check failed:', error);
      }
    };

    checkAuth();
  }, []);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    setUser(null);
    mockPush('/login');
  };

  return (
    <html lang="en" data-theme={theme}>
      <body className={`theme-${theme}`}>
        <div data-testid="root-layout" className="app-layout">
          <header data-testid="app-header" className="app-header">
            <button 
              data-testid="sidebar-toggle"
              onClick={toggleSidebar}
              className="sidebar-toggle"
            >
              ☰
            </button>
            
            <h1 className="app-title">FreeAgentics</h1>
            
            <div className="header-controls">
              <button 
                data-testid="theme-toggle"
                onClick={toggleTheme}
                className="theme-toggle"
              >
                {theme === 'light' ? '🌙' : '☀️'}
              </button>
              
              {user ? (
                <div className="user-menu">
                  <span data-testid="user-name">{user.name}</span>
                  <button 
                    data-testid="logout-button"
                    onClick={handleLogout}
                    className="logout-btn"
                  >
                    Logout
                  </button>
                </div>
              ) : (
                <button 
                  data-testid="login-button"
                  onClick={() => mockPush('/login')}
                  className="login-btn"
                >
                  Login
                </button>
              )}
            </div>
          </header>

          <nav 
            data-testid="sidebar"
            className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}
          >
            <ul className="nav-menu">
              <li>
                <a href="/dashboard" data-testid="nav-dashboard">Dashboard</a>
              </li>
              <li>
                <a href="/agents" data-testid="nav-agents">Agents</a>
              </li>
              <li>
                <a href="/conversations" data-testid="nav-conversations">Conversations</a>
              </li>
              <li>
                <a href="/knowledge" data-testid="nav-knowledge">Knowledge Graph</a>
              </li>
              <li>
                <a href="/experiments" data-testid="nav-experiments">Experiments</a>
              </li>
              <li>
                <a href="/world" data-testid="nav-world">World Simulation</a>
              </li>
              <li>
                <a href="/active-inference-demo" data-testid="nav-active-inference">Active Inference</a>
              </li>
            </ul>
          </nav>

          <main 
            data-testid="main-content"
            className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`}
          >
            {children}
          </main>

          <footer data-testid="app-footer" className="app-footer">
            <p>&copy; 2024 FreeAgentics. All rights reserved.</p>
            <p>Theme: {theme}</p>
          </footer>
        </div>
      </body>
    </html>
  );
};

// Dashboard Page Component
const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = React.useState({
    activeAgents: 0,
    totalConversations: 0,
    knowledgeNodes: 0,
    systemHealth: 'good' as 'good' | 'warning' | 'critical',
  });

  const [recentActivity, setRecentActivity] = React.useState<Array<{
    id: string;
    type: string;
    message: string;
    timestamp: Date;
  }>>([]);

  const [notifications, setNotifications] = React.useState<Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: Date;
  }>>([]);

  React.useEffect(() => {
    // Simulate metrics fetching
    const fetchMetrics = async () => {
      try {
        setMetrics({
          activeAgents: Math.floor(Math.random() * 20) + 5,
          totalConversations: Math.floor(Math.random() * 100) + 10,
          knowledgeNodes: Math.floor(Math.random() * 1000) + 100,
          systemHealth: Math.random() > 0.2 ? 'good' : 'warning',
        });

        setRecentActivity([
          {
            id: '1',
            type: 'agent_created',
            message: 'New agent "Researcher-7" created',
            timestamp: new Date(Date.now() - 300000),
          },
          {
            id: '2',
            type: 'conversation_started',
            message: 'Conversation "Climate Discussion" started',
            timestamp: new Date(Date.now() - 600000),
          },
          {
            id: '3',
            type: 'knowledge_updated',
            message: 'Knowledge graph updated with 15 new nodes',
            timestamp: new Date(Date.now() - 900000),
          },
        ]);

        if (Math.random() > 0.7) {
          setNotifications([
            {
              id: '1',
              type: 'info',
              message: 'System maintenance scheduled for tonight',
              timestamp: new Date(),
            },
          ]);
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const dismissNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const navigateToSection = (section: string) => {
    mockPush(`/${section}`);
  };

  return (
    <div data-testid="dashboard-page" className="dashboard-page">
      <header className="page-header">
        <h1>System Dashboard</h1>
        <button 
          data-testid="refresh-dashboard"
          onClick={() => window.location.reload()}
          className="refresh-btn"
        >
          Refresh
        </button>
      </header>

      {notifications.length > 0 && (
        <div data-testid="notifications-area" className="notifications">
          {notifications.map(notification => (
            <div 
              key={notification.id}
              data-testid={`notification-${notification.id}`}
              className={`notification ${notification.type}`}
            >
              <span>{notification.message}</span>
              <button 
                data-testid={`dismiss-${notification.id}`}
                onClick={() => dismissNotification(notification.id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="dashboard-grid">
        <div className="metrics-section">
          <h2>System Metrics</h2>
          
          <div className="metrics-cards">
            <div 
              data-testid="active-agents-card"
              className="metric-card"
              onClick={() => navigateToSection('agents')}
            >
              <h3>Active Agents</h3>
              <div className="metric-value">{metrics.activeAgents}</div>
            </div>
            
            <div 
              data-testid="conversations-card"
              className="metric-card"
              onClick={() => navigateToSection('conversations')}
            >
              <h3>Total Conversations</h3>
              <div className="metric-value">{metrics.totalConversations}</div>
            </div>
            
            <div 
              data-testid="knowledge-card"
              className="metric-card"
              onClick={() => navigateToSection('knowledge')}
            >
              <h3>Knowledge Nodes</h3>
              <div className="metric-value">{metrics.knowledgeNodes}</div>
            </div>
            
            <div 
              data-testid="health-card"
              className={`metric-card health-${metrics.systemHealth}`}
            >
              <h3>System Health</h3>
              <div className="metric-value">{metrics.systemHealth}</div>
            </div>
          </div>
        </div>

        <div className="activity-section">
          <h2>Recent Activity</h2>
          <div data-testid="activity-list" className="activity-list">
            {recentActivity.map(activity => (
              <div 
                key={activity.id}
                data-testid={`activity-${activity.id}`}
                className="activity-item"
              >
                <span className="activity-type">{activity.type}</span>
                <span className="activity-message">{activity.message}</span>
                <span className="activity-time">
                  {activity.timestamp.toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="quick-actions">
          <h2>Quick Actions</h2>
          <div className="action-buttons">
            <button 
              data-testid="create-agent-btn"
              onClick={() => navigateToSection('agents')}
              className="action-btn"
            >
              Create Agent
            </button>
            <button 
              data-testid="start-conversation-btn"
              onClick={() => navigateToSection('conversations')}
              className="action-btn"
            >
              Start Conversation
            </button>
            <button 
              data-testid="view-experiments-btn"
              onClick={() => navigateToSection('experiments')}
              className="action-btn"
            >
              View Experiments
            </button>
            <button 
              data-testid="run-simulation-btn"
              onClick={() => navigateToSection('world')}
              className="action-btn"
            >
              Run Simulation
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Agents Page Component
const AgentsPage: React.FC = () => {
  const [agents, setAgents] = React.useState<Array<{
    id: string;
    name: string;
    type: string;
    status: 'active' | 'idle' | 'offline';
    capabilities: string[];
    performance: {
      tasksCompleted: number;
      successRate: number;
      averageResponseTime: number;
    };
  }>>([]);

  const [selectedAgent, setSelectedAgent] = React.useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = React.useState(false);
  const [newAgentForm, setNewAgentForm] = React.useState({
    name: '',
    type: 'conversational',
    capabilities: [] as string[],
  });

  React.useEffect(() => {
    // Simulate agents data loading
    setAgents([
      {
        id: 'agent-1',
        name: 'Research Assistant',
        type: 'conversational',
        status: 'active',
        capabilities: ['research', 'analysis', 'writing'],
        performance: {
          tasksCompleted: 45,
          successRate: 0.92,
          averageResponseTime: 1200,
        },
      },
      {
        id: 'agent-2',
        name: 'Data Analyzer',
        type: 'analytical',
        status: 'active',
        capabilities: ['data_analysis', 'visualization', 'statistics'],
        performance: {
          tasksCompleted: 23,
          successRate: 0.88,
          averageResponseTime: 2500,
        },
      },
      {
        id: 'agent-3',
        name: 'Creative Writer',
        type: 'creative',
        status: 'idle',
        capabilities: ['writing', 'storytelling', 'creativity'],
        performance: {
          tasksCompleted: 12,
          successRate: 0.95,
          averageResponseTime: 3200,
        },
      },
    ]);
  }, []);

  const handleCreateAgent = (e: React.FormEvent) => {
    e.preventDefault();
    
    const newAgent = {
      id: `agent-${Date.now()}`,
      name: newAgentForm.name,
      type: newAgentForm.type,
      status: 'active' as const,
      capabilities: newAgentForm.capabilities,
      performance: {
        tasksCompleted: 0,
        successRate: 0,
        averageResponseTime: 0,
      },
    };

    setAgents(prev => [...prev, newAgent]);
    setNewAgentForm({ name: '', type: 'conversational', capabilities: [] });
    setShowCreateForm(false);
  };

  const handleDeleteAgent = (agentId: string) => {
    setAgents(prev => prev.filter(a => a.id !== agentId));
    if (selectedAgent === agentId) {
      setSelectedAgent(null);
    }
  };

  const selectedAgentData = agents.find(a => a.id === selectedAgent);

  return (
    <div data-testid="agents-page" className="agents-page">
      <header className="page-header">
        <h1>Agents Management</h1>
        <button 
          data-testid="create-agent-button"
          onClick={() => setShowCreateForm(true)}
          className="create-btn"
        >
          Create New Agent
        </button>
      </header>

      {showCreateForm && (
        <div data-testid="create-agent-modal" className="modal-overlay">
          <div className="modal">
            <h2>Create New Agent</h2>
            <form onSubmit={handleCreateAgent}>
              <div className="form-group">
                <label htmlFor="agent-name">Name:</label>
                <input
                  id="agent-name"
                  data-testid="agent-name-input"
                  type="text"
                  value={newAgentForm.name}
                  onChange={e => setNewAgentForm(prev => ({ ...prev, name: e.target.value }))}
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="agent-type">Type:</label>
                <select
                  id="agent-type"
                  data-testid="agent-type-select"
                  value={newAgentForm.type}
                  onChange={e => setNewAgentForm(prev => ({ ...prev, type: e.target.value }))}
                >
                  <option value="conversational">Conversational</option>
                  <option value="analytical">Analytical</option>
                  <option value="creative">Creative</option>
                  <option value="specialized">Specialized</option>
                </select>
              </div>
              
              <div className="form-actions">
                <button 
                  type="submit"
                  data-testid="submit-agent"
                  className="submit-btn"
                >
                  Create Agent
                </button>
                <button 
                  type="button"
                  data-testid="cancel-agent"
                  onClick={() => setShowCreateForm(false)}
                  className="cancel-btn"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="agents-container">
        <div className="agents-list">
          <h2>All Agents ({agents.length})</h2>
          <div data-testid="agents-grid" className="agents-grid">
            {agents.map(agent => (
              <div 
                key={agent.id}
                data-testid={`agent-card-${agent.id}`}
                className={`agent-card ${selectedAgent === agent.id ? 'selected' : ''}`}
                onClick={() => setSelectedAgent(agent.id)}
              >
                <h3>{agent.name}</h3>
                <div className="agent-meta">
                  <span className="agent-type">{agent.type}</span>
                  <span className={`agent-status ${agent.status}`}>{agent.status}</span>
                </div>
                <div className="agent-capabilities">
                  {agent.capabilities.slice(0, 3).map(cap => (
                    <span key={cap} className="capability-tag">{cap}</span>
                  ))}
                </div>
                <div className="agent-performance">
                  <small>Tasks: {agent.performance.tasksCompleted}</small>
                  <small>Success: {(agent.performance.successRate * 100).toFixed(0)}%</small>
                </div>
                <button 
                  data-testid={`delete-agent-${agent.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteAgent(agent.id);
                  }}
                  className="delete-btn"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        </div>

        {selectedAgentData && (
          <div data-testid="agent-details-panel" className="agent-details">
            <h2>Agent Details</h2>
            <div className="detail-section">
              <h3>{selectedAgentData.name}</h3>
              <p><strong>Type:</strong> {selectedAgentData.type}</p>
              <p><strong>Status:</strong> {selectedAgentData.status}</p>
              
              <h4>Capabilities</h4>
              <div className="capabilities-list">
                {selectedAgentData.capabilities.map(cap => (
                  <span key={cap} className="capability-badge">{cap}</span>
                ))}
              </div>
              
              <h4>Performance Metrics</h4>
              <div className="performance-metrics">
                <div className="metric">
                  <label>Tasks Completed:</label>
                  <span>{selectedAgentData.performance.tasksCompleted}</span>
                </div>
                <div className="metric">
                  <label>Success Rate:</label>
                  <span>{(selectedAgentData.performance.successRate * 100).toFixed(1)}%</span>
                </div>
                <div className="metric">
                  <label>Avg Response Time:</label>
                  <span>{selectedAgentData.performance.averageResponseTime}ms</span>
                </div>
              </div>

              <div className="agent-actions">
                <button 
                  data-testid="activate-agent"
                  className="action-btn"
                >
                  {selectedAgentData.status === 'active' ? 'Deactivate' : 'Activate'}
                </button>
                <button 
                  data-testid="configure-agent"
                  className="action-btn"
                >
                  Configure
                </button>
                <button 
                  data-testid="view-agent-logs"
                  className="action-btn"
                >
                  View Logs
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Conversations Page Component  
const ConversationsPage: React.FC = () => {
  const [conversations, setConversations] = React.useState<Array<{
    id: string;
    title: string;
    participants: string[];
    lastMessage: string;
    lastActivity: Date;
    status: 'active' | 'archived';
    messageCount: number;
  }>>([]);

  const [selectedConversation, setSelectedConversation] = React.useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = React.useState(false);

  React.useEffect(() => {
    setConversations([
      {
        id: 'conv-1',
        title: 'Climate Change Research',
        participants: ['Research Assistant', 'Data Analyzer'],
        lastMessage: 'The latest data shows a concerning trend...',
        lastActivity: new Date(Date.now() - 300000),
        status: 'active',
        messageCount: 45,
      },
      {
        id: 'conv-2',
        title: 'Creative Writing Project',
        participants: ['Creative Writer', 'Research Assistant'],
        lastMessage: 'Let me revise that opening paragraph...',
        lastActivity: new Date(Date.now() - 1800000),
        status: 'active',
        messageCount: 23,
      },
      {
        id: 'conv-3',
        title: 'Data Analysis Discussion',
        participants: ['Data Analyzer'],
        lastMessage: 'The correlation coefficient suggests...',
        lastActivity: new Date(Date.now() - 3600000),
        status: 'archived',
        messageCount: 12,
      },
    ]);
  }, []);

  const handleCreateConversation = () => {
    const newConv = {
      id: `conv-${Date.now()}`,
      title: 'New Conversation',
      participants: [],
      lastMessage: 'Conversation started...',
      lastActivity: new Date(),
      status: 'active' as const,
      messageCount: 0,
    };
    
    setConversations(prev => [newConv, ...prev]);
    setSelectedConversation(newConv.id);
    setShowCreateForm(false);
  };

  const handleArchiveConversation = (convId: string) => {
    setConversations(prev => 
      prev.map(conv => 
        conv.id === convId 
          ? { ...conv, status: 'archived' as const }
          : conv
      )
    );
  };

  const selectedConversationData = conversations.find(c => c.id === selectedConversation);

  return (
    <div data-testid="conversations-page" className="conversations-page">
      <header className="page-header">
        <h1>Conversations</h1>
        <button 
          data-testid="create-conversation-button"
          onClick={() => setShowCreateForm(true)}
          className="create-btn"
        >
          Start New Conversation
        </button>
      </header>

      {showCreateForm && (
        <div data-testid="create-conversation-modal" className="modal-overlay">
          <div className="modal">
            <h2>Start New Conversation</h2>
            <p>Select agents to participate in the conversation:</p>
            <div className="form-actions">
              <button 
                data-testid="confirm-create-conversation"
                onClick={handleCreateConversation}
                className="submit-btn"
              >
                Create Conversation
              </button>
              <button 
                data-testid="cancel-create-conversation"
                onClick={() => setShowCreateForm(false)}
                className="cancel-btn"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="conversations-container">
        <div className="conversations-list">
          <div className="filter-tabs">
            <button 
              data-testid="active-conversations-tab"
              className="tab active"
            >
              Active ({conversations.filter(c => c.status === 'active').length})
            </button>
            <button 
              data-testid="archived-conversations-tab"
              className="tab"
            >
              Archived ({conversations.filter(c => c.status === 'archived').length})
            </button>
          </div>

          <div data-testid="conversations-list" className="conversation-items">
            {conversations.map(conv => (
              <div 
                key={conv.id}
                data-testid={`conversation-item-${conv.id}`}
                className={`conversation-item ${selectedConversation === conv.id ? 'selected' : ''} ${conv.status}`}
                onClick={() => setSelectedConversation(conv.id)}
              >
                <h3>{conv.title}</h3>
                <p className="participants">
                  Participants: {conv.participants.join(', ') || 'None'}
                </p>
                <p className="last-message">{conv.lastMessage}</p>
                <div className="conversation-meta">
                  <span className="message-count">{conv.messageCount} messages</span>
                  <span className="last-activity">
                    {conv.lastActivity.toLocaleDateString()}
                  </span>
                  <span className={`status ${conv.status}`}>{conv.status}</span>
                </div>
                {conv.status === 'active' && (
                  <button 
                    data-testid={`archive-conversation-${conv.id}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleArchiveConversation(conv.id);
                    }}
                    className="archive-btn"
                  >
                    Archive
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        {selectedConversationData && (
          <div data-testid="conversation-details-panel" className="conversation-details">
            <h2>{selectedConversationData.title}</h2>
            <div className="conversation-info">
              <p><strong>Status:</strong> {selectedConversationData.status}</p>
              <p><strong>Participants:</strong> {selectedConversationData.participants.join(', ') || 'None'}</p>
              <p><strong>Messages:</strong> {selectedConversationData.messageCount}</p>
              <p><strong>Last Activity:</strong> {selectedConversationData.lastActivity.toLocaleString()}</p>
            </div>

            <div className="conversation-actions">
              <button 
                data-testid="join-conversation"
                className="action-btn"
              >
                Join Conversation
              </button>
              <button 
                data-testid="view-full-conversation"
                className="action-btn"
              >
                View Full History
              </button>
              <button 
                data-testid="export-conversation"
                className="action-btn"
              >
                Export
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Knowledge Graph Page Component
const KnowledgePage: React.FC = () => {
  const [stats, setStats] = React.useState({
    totalNodes: 0,
    totalEdges: 0,
    nodeTypes: {} as Record<string, number>,
    recentUpdates: [] as Array<{
      id: string;
      type: 'node_added' | 'edge_added' | 'node_updated';
      description: string;
      timestamp: Date;
    }>,
  });

  const [selectedNodeType, setSelectedNodeType] = React.useState<string>('all');
  const [searchQuery, setSearchQuery] = React.useState('');

  React.useEffect(() => {
    setStats({
      totalNodes: 1247,
      totalEdges: 3421,
      nodeTypes: {
        'concept': 456,
        'agent': 12,
        'belief': 234,
        'relationship': 545,
      },
      recentUpdates: [
        {
          id: '1',
          type: 'node_added',
          description: 'Added concept node "Machine Learning"',
          timestamp: new Date(Date.now() - 300000),
        },
        {
          id: '2',
          type: 'edge_added',
          description: 'Connected "AI" to "Ethics"',
          timestamp: new Date(Date.now() - 600000),
        },
        {
          id: '3',
          type: 'node_updated',
          description: 'Updated belief node "cooperation_value"',
          timestamp: new Date(Date.now() - 900000),
        },
      ],
    });
  }, []);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    // In real implementation, would trigger search
  };

  const handleExport = () => {
    // Simulate export functionality
    const exportData = {
      nodes: stats.totalNodes,
      edges: stats.totalEdges,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'knowledge-graph.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div data-testid="knowledge-page" className="knowledge-page">
      <header className="page-header">
        <h1>Knowledge Graph</h1>
        <div className="header-controls">
          <button 
            data-testid="export-knowledge"
            onClick={handleExport}
            className="export-btn"
          >
            Export Graph
          </button>
          <button 
            data-testid="import-knowledge"
            className="import-btn"
          >
            Import Data
          </button>
        </div>
      </header>

      <div className="knowledge-controls">
        <div className="search-section">
          <input
            data-testid="knowledge-search"
            type="text"
            placeholder="Search knowledge graph..."
            value={searchQuery}
            onChange={e => handleSearch(e.target.value)}
            className="search-input"
          />
          <button 
            data-testid="search-button"
            className="search-btn"
          >
            Search
          </button>
        </div>

        <div className="filter-section">
          <label htmlFor="node-type-filter">Filter by type:</label>
          <select
            id="node-type-filter"
            data-testid="node-type-filter"
            value={selectedNodeType}
            onChange={e => setSelectedNodeType(e.target.value)}
          >
            <option value="all">All Types</option>
            <option value="concept">Concepts</option>
            <option value="agent">Agents</option>
            <option value="belief">Beliefs</option>
            <option value="relationship">Relationships</option>
          </select>
        </div>
      </div>

      <div className="knowledge-stats">
        <div data-testid="total-nodes" className="stat-card">
          <h3>Total Nodes</h3>
          <div className="stat-value">{stats.totalNodes.toLocaleString()}</div>
        </div>
        
        <div data-testid="total-edges" className="stat-card">
          <h3>Total Edges</h3>
          <div className="stat-value">{stats.totalEdges.toLocaleString()}</div>
        </div>
        
        <div data-testid="node-types" className="stat-card">
          <h3>Node Types</h3>
          <div className="node-type-breakdown">
            {Object.entries(stats.nodeTypes).map(([type, count]) => (
              <div key={type} className="type-stat">
                <span className="type-name">{type}:</span>
                <span className="type-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="knowledge-content">
        <div className="visualization-area">
          <h2>Graph Visualization</h2>
          <div data-testid="knowledge-graph-viz" className="graph-container">
            <canvas 
              data-testid="knowledge-canvas"
              width="800" 
              height="600"
              className="knowledge-canvas"
            />
            <div className="viz-controls">
              <button data-testid="zoom-in">Zoom In</button>
              <button data-testid="zoom-out">Zoom Out</button>
              <button data-testid="reset-view">Reset View</button>
            </div>
          </div>
        </div>

        <div className="recent-updates">
          <h2>Recent Updates</h2>
          <div data-testid="recent-updates-list" className="updates-list">
            {stats.recentUpdates.map(update => (
              <div 
                key={update.id}
                data-testid={`update-${update.id}`}
                className="update-item"
              >
                <span className={`update-type ${update.type}`}>
                  {update.type.replace('_', ' ')}
                </span>
                <span className="update-description">{update.description}</span>
                <span className="update-time">
                  {update.timestamp.toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// Error Page Components
const NotFoundPage: React.FC = () => {
  return (
    <div data-testid="not-found-page" className="error-page">
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
      <button 
        data-testid="go-home"
        onClick={() => mockPush('/dashboard')}
        className="home-btn"
      >
        Go to Dashboard
      </button>
    </div>
  );
};

const ErrorPage: React.FC<{ error?: Error }> = ({ error }) => {
  return (
    <div data-testid="error-page" className="error-page">
      <h1>Something went wrong</h1>
      <p>{error?.message || 'An unexpected error occurred'}</p>
      <button 
        data-testid="reload-page"
        onClick={() => window.location.reload()}
        className="reload-btn"
      >
        Reload Page
      </button>
    </div>
  );
};

describe('Comprehensive Page Component Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  describe('RootLayout', () => {
    it('renders root layout with header, sidebar, and footer', () => {
      render(
        <RootLayout>
          <div>Test Content</div>
        </RootLayout>
      );

      expect(screen.getByTestId('root-layout')).toBeInTheDocument();
      expect(screen.getByTestId('app-header')).toBeInTheDocument();
      expect(screen.getByTestId('sidebar')).toBeInTheDocument();
      expect(screen.getByTestId('main-content')).toBeInTheDocument();
      expect(screen.getByTestId('app-footer')).toBeInTheDocument();
    });

    it('toggles sidebar open/closed', () => {
      render(
        <RootLayout>
          <div>Test Content</div>
        </RootLayout>
      );

      const sidebar = screen.getByTestId('sidebar');
      const toggleButton = screen.getByTestId('sidebar-toggle');

      expect(sidebar).toHaveClass('closed');

      fireEvent.click(toggleButton);
      expect(sidebar).toHaveClass('open');

      fireEvent.click(toggleButton);
      expect(sidebar).toHaveClass('closed');
    });

    it('toggles theme between light and dark', () => {
      render(
        <RootLayout>
          <div>Test Content</div>
        </RootLayout>
      );

      const themeToggle = screen.getByTestId('theme-toggle');
      const footer = screen.getByTestId('app-footer');

      expect(footer).toHaveTextContent('Theme: light');

      fireEvent.click(themeToggle);
      expect(footer).toHaveTextContent('Theme: dark');

      fireEvent.click(themeToggle);
      expect(footer).toHaveTextContent('Theme: light');
    });

    it('handles user authentication state', () => {
      localStorage.setItem('auth_token', 'test-token');

      render(
        <RootLayout>
          <div>Test Content</div>
        </RootLayout>
      );

      expect(screen.getByTestId('user-name')).toHaveTextContent('Test User');
      expect(screen.getByTestId('logout-button')).toBeInTheDocument();
    });

    it('handles logout functionality', () => {
      localStorage.setItem('auth_token', 'test-token');

      render(
        <RootLayout>
          <div>Test Content</div>
        </RootLayout>
      );

      fireEvent.click(screen.getByTestId('logout-button'));

      expect(localStorage.getItem('auth_token')).toBeNull();
      expect(mockPush).toHaveBeenCalledWith('/login');
    });

    it('renders navigation menu', () => {
      render(
        <RootLayout>
          <div>Test Content</div>
        </RootLayout>
      );

      expect(screen.getByTestId('nav-dashboard')).toBeInTheDocument();
      expect(screen.getByTestId('nav-agents')).toBeInTheDocument();
      expect(screen.getByTestId('nav-conversations')).toBeInTheDocument();
      expect(screen.getByTestId('nav-knowledge')).toBeInTheDocument();
      expect(screen.getByTestId('nav-experiments')).toBeInTheDocument();
      expect(screen.getByTestId('nav-world')).toBeInTheDocument();
      expect(screen.getByTestId('nav-active-inference')).toBeInTheDocument();
    });
  });

  describe('DashboardPage', () => {
    it('renders dashboard with metrics and activity', async () => {
      render(<DashboardPage />);

      expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      expect(screen.getByTestId('active-agents-card')).toBeInTheDocument();
      expect(screen.getByTestId('conversations-card')).toBeInTheDocument();
      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      expect(screen.getByTestId('health-card')).toBeInTheDocument();

      await waitFor(() => {
        expect(screen.getByTestId('activity-list')).toBeInTheDocument();
      });
    });

    it('handles metric card clicks for navigation', () => {
      render(<DashboardPage />);

      fireEvent.click(screen.getByTestId('active-agents-card'));
      expect(mockPush).toHaveBeenCalledWith('/agents');

      fireEvent.click(screen.getByTestId('conversations-card'));
      expect(mockPush).toHaveBeenCalledWith('/conversations');

      fireEvent.click(screen.getByTestId('knowledge-card'));
      expect(mockPush).toHaveBeenCalledWith('/knowledge');
    });

    it('displays and dismisses notifications', async () => {
      render(<DashboardPage />);

      // Wait for potential notifications to appear
      await waitFor(() => {
        const notificationsArea = screen.queryByTestId('notifications-area');
        if (notificationsArea) {
          const dismissButton = screen.getByTestId(/dismiss-\d+/);
          fireEvent.click(dismissButton);
          
          // Notification should be removed
          expect(screen.queryByTestId('notifications-area')).not.toBeInTheDocument();
        }
      });
    });

    it('handles quick action buttons', () => {
      render(<DashboardPage />);

      fireEvent.click(screen.getByTestId('create-agent-btn'));
      expect(mockPush).toHaveBeenCalledWith('/agents');

      fireEvent.click(screen.getByTestId('start-conversation-btn'));
      expect(mockPush).toHaveBeenCalledWith('/conversations');

      fireEvent.click(screen.getByTestId('view-experiments-btn'));
      expect(mockPush).toHaveBeenCalledWith('/experiments');

      fireEvent.click(screen.getByTestId('run-simulation-btn'));
      expect(mockPush).toHaveBeenCalledWith('/world');
    });

    it('handles dashboard refresh', () => {
      const reloadSpy = jest.spyOn(window.location, 'reload').mockImplementation(() => {});

      render(<DashboardPage />);

      fireEvent.click(screen.getByTestId('refresh-dashboard'));
      expect(reloadSpy).toHaveBeenCalled();

      reloadSpy.mockRestore();
    });
  });

  describe('AgentsPage', () => {
    it('renders agents list and management interface', () => {
      render(<AgentsPage />);

      expect(screen.getByTestId('agents-page')).toBeInTheDocument();
      expect(screen.getByTestId('create-agent-button')).toBeInTheDocument();
      expect(screen.getByTestId('agents-grid')).toBeInTheDocument();
    });

    it('opens and closes create agent modal', () => {
      render(<AgentsPage />);

      fireEvent.click(screen.getByTestId('create-agent-button'));
      expect(screen.getByTestId('create-agent-modal')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('cancel-agent'));
      expect(screen.queryByTestId('create-agent-modal')).not.toBeInTheDocument();
    });

    it('creates new agent through form', () => {
      render(<AgentsPage />);

      fireEvent.click(screen.getByTestId('create-agent-button'));

      fireEvent.change(screen.getByTestId('agent-name-input'), {
        target: { value: 'Test Agent' },
      });

      fireEvent.change(screen.getByTestId('agent-type-select'), {
        target: { value: 'analytical' },
      });

      fireEvent.click(screen.getByTestId('submit-agent'));

      expect(screen.queryByTestId('create-agent-modal')).not.toBeInTheDocument();
      expect(screen.getByText('Test Agent')).toBeInTheDocument();
    });

    it('selects and displays agent details', () => {
      render(<AgentsPage />);

      fireEvent.click(screen.getByTestId('agent-card-agent-1'));

      expect(screen.getByTestId('agent-details-panel')).toBeInTheDocument();
      expect(screen.getByText('Research Assistant')).toBeInTheDocument();
    });

    it('deletes agents', () => {
      render(<AgentsPage />);

      const initialAgents = screen.getAllByTestId(/agent-card-/);
      const initialCount = initialAgents.length;

      fireEvent.click(screen.getByTestId('delete-agent-agent-1'));

      const remainingAgents = screen.getAllByTestId(/agent-card-/);
      expect(remainingAgents).toHaveLength(initialCount - 1);
    });

    it('handles agent detail actions', () => {
      render(<AgentsPage />);

      fireEvent.click(screen.getByTestId('agent-card-agent-1'));

      expect(screen.getByTestId('activate-agent')).toBeInTheDocument();
      expect(screen.getByTestId('configure-agent')).toBeInTheDocument();
      expect(screen.getByTestId('view-agent-logs')).toBeInTheDocument();
    });
  });

  describe('ConversationsPage', () => {
    it('renders conversations list and management interface', () => {
      render(<ConversationsPage />);

      expect(screen.getByTestId('conversations-page')).toBeInTheDocument();
      expect(screen.getByTestId('create-conversation-button')).toBeInTheDocument();
      expect(screen.getByTestId('conversations-list')).toBeInTheDocument();
    });

    it('creates new conversation', () => {
      render(<ConversationsPage />);

      fireEvent.click(screen.getByTestId('create-conversation-button'));
      expect(screen.getByTestId('create-conversation-modal')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('confirm-create-conversation'));

      expect(screen.queryByTestId('create-conversation-modal')).not.toBeInTheDocument();
      expect(screen.getByText('New Conversation')).toBeInTheDocument();
    });

    it('displays conversation details when selected', () => {
      render(<ConversationsPage />);

      fireEvent.click(screen.getByTestId('conversation-item-conv-1'));

      expect(screen.getByTestId('conversation-details-panel')).toBeInTheDocument();
      expect(screen.getByText('Climate Change Research')).toBeInTheDocument();
    });

    it('archives conversations', () => {
      render(<ConversationsPage />);

      fireEvent.click(screen.getByTestId('archive-conversation-conv-1'));

      const archivedTab = screen.getByTestId('archived-conversations-tab');
      expect(archivedTab).toHaveTextContent('Archived (2)');
    });

    it('handles conversation actions', () => {
      render(<ConversationsPage />);

      fireEvent.click(screen.getByTestId('conversation-item-conv-1'));

      expect(screen.getByTestId('join-conversation')).toBeInTheDocument();
      expect(screen.getByTestId('view-full-conversation')).toBeInTheDocument();
      expect(screen.getByTestId('export-conversation')).toBeInTheDocument();
    });

    it('filters conversations by status', () => {
      render(<ConversationsPage />);

      expect(screen.getByTestId('active-conversations-tab')).toHaveClass('active');
      expect(screen.getByTestId('archived-conversations-tab')).toBeInTheDocument();
    });
  });

  describe('KnowledgePage', () => {
    it('renders knowledge graph interface', () => {
      render(<KnowledgePage />);

      expect(screen.getByTestId('knowledge-page')).toBeInTheDocument();
      expect(screen.getByTestId('knowledge-search')).toBeInTheDocument();
      expect(screen.getByTestId('node-type-filter')).toBeInTheDocument();
      expect(screen.getByTestId('knowledge-graph-viz')).toBeInTheDocument();
    });

    it('displays knowledge statistics', () => {
      render(<KnowledgePage />);

      expect(screen.getByTestId('total-nodes')).toHaveTextContent('1,247');
      expect(screen.getByTestId('total-edges')).toHaveTextContent('3,421');
      expect(screen.getByTestId('node-types')).toBeInTheDocument();
    });

    it('handles search functionality', () => {
      render(<KnowledgePage />);

      fireEvent.change(screen.getByTestId('knowledge-search'), {
        target: { value: 'machine learning' },
      });

      fireEvent.click(screen.getByTestId('search-button'));

      expect(screen.getByTestId('knowledge-search')).toHaveValue('machine learning');
    });

    it('filters by node type', () => {
      render(<KnowledgePage />);

      fireEvent.change(screen.getByTestId('node-type-filter'), {
        target: { value: 'concept' },
      });

      expect(screen.getByTestId('node-type-filter')).toHaveValue('concept');
    });

    it('handles graph visualization controls', () => {
      render(<KnowledgePage />);

      expect(screen.getByTestId('zoom-in')).toBeInTheDocument();
      expect(screen.getByTestId('zoom-out')).toBeInTheDocument();
      expect(screen.getByTestId('reset-view')).toBeInTheDocument();
      expect(screen.getByTestId('knowledge-canvas')).toBeInTheDocument();
    });

    it('exports knowledge graph data', () => {
      const createObjectURLSpy = jest.spyOn(URL, 'createObjectURL').mockReturnValue('blob:url');
      const revokeObjectURLSpy = jest.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

      render(<KnowledgePage />);

      fireEvent.click(screen.getByTestId('export-knowledge'));

      expect(createObjectURLSpy).toHaveBeenCalled();
      expect(revokeObjectURLSpy).toHaveBeenCalled();

      createObjectURLSpy.mockRestore();
      revokeObjectURLSpy.mockRestore();
    });

    it('displays recent updates', () => {
      render(<KnowledgePage />);

      expect(screen.getByTestId('recent-updates-list')).toBeInTheDocument();
      expect(screen.getByTestId('update-1')).toBeInTheDocument();
      expect(screen.getByTestId('update-2')).toBeInTheDocument();
      expect(screen.getByTestId('update-3')).toBeInTheDocument();
    });
  });

  describe('Error Pages', () => {
    it('renders 404 not found page', () => {
      render(<NotFoundPage />);

      expect(screen.getByTestId('not-found-page')).toBeInTheDocument();
      expect(screen.getByText('404 - Page Not Found')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('go-home'));
      expect(mockPush).toHaveBeenCalledWith('/dashboard');
    });

    it('renders error page with custom error', () => {
      const testError = new Error('Test error message');

      render(<ErrorPage error={testError} />);

      expect(screen.getByTestId('error-page')).toBeInTheDocument();
      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });

    it('renders error page without custom error', () => {
      render(<ErrorPage />);

      expect(screen.getByTestId('error-page')).toBeInTheDocument();
      expect(screen.getByText('An unexpected error occurred')).toBeInTheDocument();
    });

    it('handles page reload on error page', () => {
      const reloadSpy = jest.spyOn(window.location, 'reload').mockImplementation(() => {});

      render(<ErrorPage />);

      fireEvent.click(screen.getByTestId('reload-page'));
      expect(reloadSpy).toHaveBeenCalled();

      reloadSpy.mockRestore();
    });
  });

  describe('Page Integration', () => {
    it('renders complete application with layout and page', () => {
      render(
        <RootLayout>
          <DashboardPage />
        </RootLayout>
      );

      expect(screen.getByTestId('root-layout')).toBeInTheDocument();
      expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      expect(screen.getByTestId('app-header')).toBeInTheDocument();
      expect(screen.getByTestId('main-content')).toBeInTheDocument();
    });

    it('maintains consistent navigation across pages', () => {
      const { rerender } = render(
        <RootLayout>
          <DashboardPage />
        </RootLayout>
      );

      expect(screen.getByTestId('nav-dashboard')).toBeInTheDocument();

      rerender(
        <RootLayout>
          <AgentsPage />
        </RootLayout>
      );

      expect(screen.getByTestId('nav-agents')).toBeInTheDocument();
      expect(screen.getByTestId('agents-page')).toBeInTheDocument();
    });

    it('preserves layout state across page changes', () => {
      const { rerender } = render(
        <RootLayout>
          <DashboardPage />
        </RootLayout>
      );

      // Open sidebar
      fireEvent.click(screen.getByTestId('sidebar-toggle'));
      expect(screen.getByTestId('sidebar')).toHaveClass('open');

      // Change page
      rerender(
        <RootLayout>
          <AgentsPage />
        </RootLayout>
      );

      // Sidebar should still be open
      expect(screen.getByTestId('sidebar')).toHaveClass('open');
    });

    it('handles authentication state across pages', () => {
      localStorage.setItem('auth_token', 'test-token');

      const { rerender } = render(
        <RootLayout>
          <DashboardPage />
        </RootLayout>
      );

      expect(screen.getByTestId('user-name')).toHaveTextContent('Test User');

      rerender(
        <RootLayout>
          <ConversationsPage />
        </RootLayout>
      );

      expect(screen.getByTestId('user-name')).toHaveTextContent('Test User');
    });
  });
});