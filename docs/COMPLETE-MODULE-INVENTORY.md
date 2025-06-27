# 📚 FreeAgentics Complete Module Inventory

> **Comprehensive catalog of all programmed functionality in backend and frontend**

Last Updated: 2025-06-26

---

## 🔧 BACKEND MODULES (Python)

### 1. **AGENTS MODULE** (`agents/`)

#### **Core Agent Framework** (`agents/base/`)

- **`agent.py`**: Base Agent class - handles lifecycle, state, actions
- **`agent_factory.py`**: Factory pattern for creating different agent types
- **`agent_template.py`**: Template system for reusable agent configurations
- **`data_model.py`**: Core data structures (Agent, Position, Resources, Status)
- **`interfaces.py`**: Protocol definitions for agent contracts

#### **Agent Capabilities** (`agents/base/`)

- **`decision_making.py`**: Decision systems (utility-based, goal-oriented, reactive, hybrid, active inference)
- **`perception.py`**: Sensory input processing, stimulus detection, percept generation
- **`movement.py`**: Movement controllers, path planning, spatial navigation
- **`memory.py`**: Short/long-term memory systems, episodic memory, memory retrieval
- **`communication.py`**: Message passing, communication protocols, language processing
- **`interaction.py`**: Agent-to-agent interactions, social behaviors
- **`behaviors.py`**: Behavior trees, action selection, behavioral patterns

#### **Advanced Systems** (`agents/base/`)

- **`active_inference_integration.py`**: PyMDP integration for Active Inference
- **`belief_synchronization.py`**: Multi-agent belief alignment
- **`epistemic_value_engine.py`**: Information gain calculations, exploration vs exploitation
- **`personality_system.py`**: Agent personality traits and behavioral modifiers
- **`resource_business_model.py`**: Resource management, economic behaviors
- **`markov_blanket.py`**: Statistical separation, boundary enforcement
- **`state_manager.py`**: Agent state persistence and transitions
- **`world_integration.py`**: World interaction interfaces

#### **Active Inference** (`agents/active_inference/`)

- **`generative_model.py`**: Generative models for Active Inference
- **`precision.py`**: Precision weighting and attention mechanisms

#### **Agent Types**

- **Explorer** (`agents/explorer/`): Discovery-focused agents
  - `explorer.py`: Main explorer implementation
  - `explorer_behavior.py`: Exploration strategies

- **Guardian** (`agents/guardian/`): Protective/defensive agents
  - `guardian.py`: Guardian implementation
  - `guardian_behavior.py`: Protection strategies

- **Merchant** (`agents/merchant/`): Trading/economic agents
  - `merchant.py`: Merchant implementation
  - `merchant_behavior.py`: Trading strategies

- **Scholar** (`agents/scholar/`): Knowledge-focused agents
  - `scholar.py`: Scholar implementation
  - `scholar_behavior.py`: Learning strategies

#### **Templates** (`agents/templates/`)

- **`base_template.py`**: Base template structure
- **`explorer_template.py`**: Explorer-specific template
- **`pymdp_integration.py`**: PyMDP template integration

#### **Testing Framework** (`agents/testing/`)

- **`agent_test_framework.py`**: Comprehensive agent testing utilities

---

### 2. **COALITIONS MODULE** (`coalitions/`)

#### **Coalition Core** (`coalitions/coalition/`)

- **`coalition_models.py`**: Data models (Coalition, CoalitionMember, CoalitionGoal)
- **`coalition_criteria.py`**: Formation criteria and triggers
- **`business_opportunities.py`**: Business opportunity detection and validation

#### **Formation Algorithms** (`coalitions/formation/`)

- **`coalition_formation_algorithms.py`**: Multiple formation strategies
  - Active Inference-based formation
  - Capability-based matching
  - Resource optimization
  - Social preference matching
- **`coalition_builder.py`**: Coalition construction utilities
- **`preference_matching.py`**: Agent preference algorithms
- **`stability_analysis.py`**: Coalition stability metrics
- **`business_value_engine.py`**: Economic value calculations
- **`expert_committee_validation.py`**: Validation protocols
- **`monitoring_integration.py`**: Real-time coalition monitoring

#### **Contracts** (`coalitions/contracts/`)

- **`coalition_contract.py`**: Smart contract implementations
- **`resource_sharing.py`**: Resource sharing agreements

#### **Readiness Assessment** (`coalitions/readiness/`)

- **`readiness_evaluator.py`**: Overall readiness scoring
- **`technical_readiness_validator.py`**: Technical capability validation
- **`business_readiness_assessor.py`**: Business viability checks
- **`safety_compliance_verifier.py`**: Safety protocol verification
- **`comprehensive_readiness_integrator.py`**: Integrated readiness assessment

#### **Deployment** (`coalitions/deployment/`)

- **`deployment_manifest.py`**: Deployment configuration
- **`edge_packager.py`**: Edge device packaging

---

### 3. **INFERENCE MODULE** (`inference/`)

#### **Core Engine** (`inference/engine/`)

- **`active_inference.py`**: Main Active Inference implementation
  - Variational Message Passing
  - Gradient Descent methods
  - Free Energy minimization
- **`belief_state.py`**: Belief state representations and operations
- **`belief_update.py`**: Belief updating algorithms
- **`policy_selection.py`**: Policy selection mechanisms
- **`generative_model.py`**: Generative model implementations
- **`precision.py`**: Precision matrices and attention

#### **Advanced Features** (`inference/engine/`)

- **`hierarchical_inference.py`**: Multi-level inference
- **`temporal_planning.py`**: Time-based planning
- **`active_learning.py`**: Learning from experience
- **`parameter_learning.py`**: Model parameter adaptation
- **`uncertainty_quantification.py`**: Uncertainty metrics
- **`computational_optimization.py`**: Performance optimizations
- **`diagnostics.py`**: Debugging and diagnostics
- **`belief_visualization_interface.py`**: Visualization APIs

#### **PyMDP Integration** (`inference/engine/`)

- **`pymdp_generative_model.py`**: PyMDP model wrapper
- **`pymdp_policy_selector.py`**: PyMDP policy interface

#### **Graph Neural Networks** (`inference/gnn/`)

- **`model.py`**: Core GNN model architecture
- **`layers.py`**: GNN layer implementations (GCN, GAT, SAGE, GIN)
- **`parser.py`**: Natural language to GNN parsing
- **`generator.py`**: GNN model generation
- **`validator.py`**: Model validation
- **`executor.py`**: Model execution engine

#### **GNN Infrastructure** (`inference/gnn/`)

- **`feature_extractor.py`**: Feature extraction pipelines
- **`edge_processor.py`**: Edge feature processing
- **`batch_processor.py`**: Batch processing utilities
- **`cache_manager.py`**: Model and data caching
- **`performance_optimizer.py`**: Performance tuning
- **`metrics_collector.py`**: Performance metrics
- **`monitoring.py`**: Real-time monitoring
- **`monitoring_dashboard.py`**: Monitoring UI backend
- **`alerting.py`**: Alert management system

#### **LLM Integration** (`inference/llm/`)

- **`provider_interface.py`**: Abstract LLM provider interface
- **`local_llm_manager.py`**: Local LLM management
- **`ollama_integration.py`**: Ollama model integration
- **`model_quantization.py`**: Model compression
- **`belief_integration.py`**: LLM-belief integration
- **`fallback_mechanisms.py`**: Fallback strategies

---

### 4. **INFRASTRUCTURE MODULE** (`infrastructure/`)

#### **Database** (`infrastructure/database/`)

- **`models.py`**: SQLAlchemy ORM models
- **`connection.py`**: Database connection management
- **`seed.py`**: Database seeding utilities
- **`manage.py`**: Database management commands
- **`alembic/`**: Database migrations

#### **Hardware Abstraction** (`infrastructure/hardware/`)

- **`hal_core.py`**: Hardware abstraction layer
- **`device_discovery.py`**: Device detection and enumeration

#### **Safety Systems** (`infrastructure/safety/`)

- **`safety_protocols.py`**: Core safety protocol definitions
- **`markov_blanket_verification.py`**: Boundary verification
- **`boundary_monitoring_service.py`**: Real-time boundary monitoring
- **`risk_mitigation_metrics.py`**: Risk analysis and metrics

#### **Export & Deployment** (`infrastructure/export/`)

- **`export_builder.py`**: Export package builder
- **`experiment_export.py`**: Experiment state export
- **`coalition_packaging.py`**: Coalition deployment packages
- **`business_model_exporter.py`**: Business model exports
- **`hardware_config.py`**: Hardware configuration
- **`model_compression.py`**: Model optimization
- **`deployment_scripts.py`**: Deployment automation

#### **Deployment Validation** (`infrastructure/deployment/`)

- **`deployment_verification.py`**: Deployment health checks
- **`export_validator.py`**: Export package validation
- **`hardware_compatibility.py`**: Hardware compatibility testing

---

### 5. **KNOWLEDGE MODULE** (`knowledge/`)

- **`knowledge_graph.py`**: Knowledge graph implementation
  - Node and edge management
  - Graph queries and traversal
  - Knowledge persistence

---

### 6. **WORLD MODULE** (`world/`)

#### **Spatial Systems** (`world/grid/`)

- **`hex_world.py`**: Hexagonal grid implementation
- **`spatial_index.py`**: Spatial indexing and queries

#### **Core World**

- **`h3_world.py`**: H3 hexagonal hierarchical spatial index
- **`grid_position.py`**: Position management

#### **Simulation** (`world/simulation/`)

- **`engine.py`**: Main simulation engine
  - World state management
  - Agent orchestration
  - Event processing

#### **Spatial API** (`world/spatial/`)

- **`spatial_api.py`**: RESTful spatial query API

---

### 7. **API MODULE** (`api/`)

#### **GraphQL** (`api/graphql/`)

- **`schema.py`**: GraphQL schema definitions

#### **WebSocket** (`api/websocket/`)

- **`real_time_updates.py`**: Real-time state streaming
- **`coalition_monitoring.py`**: Coalition event streaming
- **`markov_blanket_monitoring.py`**: Boundary monitoring streams

---

## 🎨 FRONTEND MODULES (TypeScript/React)

### 1. **APPLICATION STRUCTURE** (`web/app/`)

#### **Pages**

- **`page.tsx`**: Landing page
- **`agents/page.tsx`**: Agent management interface
- **`world/page.tsx`**: World visualization
- **`experiments/page.tsx`**: Experiment management
- **`knowledge/page.tsx`**: Knowledge graph interface
- **`conversations/page.tsx`**: Agent conversations
- **`active-inference-demo/page.tsx`**: Active Inference visualization

#### **Dashboard** (`web/app/(dashboard)/`)

- **`dashboard/page.tsx`**: Main dashboard
- **`conversation-orchestration/page.tsx`**: Conversation control panel
- **`layout.tsx`**: Dashboard layout wrapper

---

### 2. **COMPONENTS** (`web/components/`)

#### **Core UI Components** (`web/components/ui/`)

- **`button.tsx`**: Button component with variants
- **`badge.tsx`**: Status badges
- **`card.tsx`**: Card containers
- **`dialog.tsx`**: Modal dialogs
- **`dropdown-menu.tsx`**: Dropdown menus
- **`input.tsx`**: Form inputs
- **`label.tsx`**: Form labels
- **`progress.tsx`**: Progress indicators
- **`select.tsx`**: Select dropdowns
- **`slider.tsx`**: Range sliders
- **`switch.tsx`**: Toggle switches
- **`tabs.tsx`**: Tab navigation
- **`textarea.tsx`**: Multiline text input
- **`toast.tsx`**: Toast notifications
- **`tooltip.tsx`**: Hover tooltips
- **`spinner.tsx`**: Loading spinner
- **`separator.tsx`**: Visual separators
- **`accordion.tsx`**: Collapsible sections
- **`alert.tsx`**: Alert messages
- **`avatar.tsx`**: User avatars
- **`checkbox.tsx`**: Checkboxes
- **`radio-group.tsx`**: Radio buttons
- **`scroll-area.tsx`**: Scrollable containers
- **`sheet.tsx`**: Side panels
- **`table.tsx`**: Data tables
- **`context-menu.tsx`**: Right-click menus
- **`hover-card.tsx`**: Hover information cards
- **`popover.tsx`**: Popover containers
- **`collapsible.tsx`**: Collapsible content
- **`command.tsx`**: Command palette
- **`menubar.tsx`**: Application menubar
- **`navigation-menu.tsx`**: Navigation menus
- **`aspect-ratio.tsx`**: Aspect ratio containers
- **`skeleton.tsx`**: Loading skeletons
- **`sonner.tsx`**: Toast notification system
- **`resizable.tsx`**: Resizable panels
- **`calendar.tsx`**: Date picker calendar
- **`date-range-picker.tsx`**: Date range selection
- **`pagination.tsx`**: Page navigation
- **`breadcrumb.tsx`**: Breadcrumb navigation
- **`toggle-group.tsx`**: Toggle button groups
- **`toggle.tsx`**: Toggle buttons

#### **Agent Components**

- **`agent-card.tsx`**: Agent display card
- **`agent-details.tsx`**: Detailed agent view
- **`agent-list.tsx`**: Agent list view
- **`agent-status.tsx`**: Agent status indicator
- **`agent-creation-wizard.tsx`**: Agent creation flow
- **`agent-configuration-form.tsx`**: Agent configuration
- **`agent-template-selector.tsx`**: Template selection
- **`agent-instantiation-modal.tsx`**: Agent instantiation dialog
- **`agent-export-dialog.tsx`**: Agent export functionality
- **`agent-activity-timeline.tsx`**: Agent activity history
- **`character-creator.tsx`**: Character-based agent creation
- **`backend-agent-list.tsx`**: Backend-connected agent list

#### **Active Inference Visualizations**

- **`active-inference-dashboard.tsx`**: Main Active Inference dashboard
- **`belief-state-visualization.tsx`**: Belief state display
- **`belief-state-mathematical-display.tsx`**: Mathematical belief representation
- **`belief-trajectory-dashboard.tsx`**: Belief evolution over time
- **`free-energy-visualization.tsx`**: Free energy metrics
- **`free-energy-landscape-viz.tsx`**: 3D free energy landscape
- **`markov-blanket-visualization.tsx`**: Markov blanket display
- **`markov-blanket-dashboard.tsx`**: Markov blanket monitoring
- **`markov-blanket-configuration-ui.tsx`**: Blanket configuration
- **`precision-matrix-viz.tsx`**: Precision matrix visualization

#### **World & Spatial Components**

- **`world-visualization.tsx`**: Main world display
- **`backend-grid-world.tsx`**: Grid world visualization
- **`hex-grid.tsx`**: Hexagonal grid display
- **`spatial-mini-map.tsx`**: Minimap overlay
- **`coalition-geographic-viz.tsx`**: Coalition spatial display
- **`resource-heatmap.tsx`**: Resource distribution map

#### **Conversation Components** (`web/components/conversation/`)

- **`conversation-dashboard.tsx`**: Conversation management
- **`optimized-conversation-dashboard.tsx`**: Performance-optimized version
- **`virtualized-message-list.tsx`**: Virtualized message rendering
- **`message-components.tsx`**: Message display components
- **`conversation-search.tsx`**: Conversation search interface
- **`message-queue-visualization.tsx`**: Message queue display

#### **Conversation Orchestration** (`web/components/conversation-orchestration/`)

- **`advanced-controls.tsx`**: Advanced orchestration controls
- **`change-history.tsx`**: Configuration change history
- **`preset-selector.tsx`**: Conversation presets
- **`real-time-preview.tsx`**: Live conversation preview
- **`response-dynamics-controls.tsx`**: Response behavior controls
- **`timing-controls.tsx`**: Timing and delay controls

#### **Knowledge Components**

- **`knowledge-graph-viz.tsx`**: Knowledge graph visualization
- **`knowledge-graph-editor.tsx`**: Graph editing interface
- **`knowledge-search.tsx`**: Knowledge search

#### **Coalition Components**

- **`coalition-dashboard.tsx`**: Coalition overview
- **`coalition-formation-viz.tsx`**: Formation visualization
- **`coalition-list.tsx`**: Coalition listing
- **`coalition-metrics.tsx`**: Coalition performance metrics

#### **Experiment Components**

- **`experiment-card.tsx`**: Experiment display
- **`experiment-controls.tsx`**: Experiment controls
- **`experiment-dashboard.tsx`**: Experiment management
- **`experiment-export-modal.tsx`**: Export experiments
- **`experiment-import-modal.tsx`**: Import experiments
- **`experiment-sharing-modal.tsx`**: Share experiments
- **`experiment-timeline.tsx`**: Experiment history

#### **Monitoring Components**

- **`monitoring-dashboard.tsx`**: System monitoring
- **`readiness-panel.tsx`**: Readiness indicators
- **`performance-monitor.tsx`**: Performance metrics
- **`alert-panel.tsx`**: System alerts
- **`network-status.tsx`**: Network connectivity

#### **Other Components**

- **`ErrorBoundary.tsx`**: Error handling wrapper
- **`themeprovider.tsx`**: Theme management
- **`simulation-controls.tsx`**: Simulation control panel
- **`strategic-positioning-dashboard.tsx`**: Strategic overview
- **`AboutButton.tsx`**: About dialog trigger
- **`aboutmodal.tsx`**: About information modal

---

### 3. **HOOKS** (`web/hooks/`)

#### **WebSocket Hooks**

- **`useConversationWebSocket.ts`**: Conversation real-time updates
- **`useKnowledgeGraphWebSocket.ts`**: Knowledge graph updates
- **`useMarkovBlanketWebSocket.ts`**: Markov blanket monitoring

#### **State Management Hooks**

- **`useAgent.ts`**: Agent state management
- **`useWorld.ts`**: World state management
- **`useExperiment.ts`**: Experiment state

#### **Utility Hooks**

- **`useDebounce.ts`**: Input debouncing
- **`useAutoScroll.ts`**: Auto-scrolling behavior
- **`useLocalStorage.ts`**: Local storage persistence
- **`usePerformanceMonitor.ts`**: Performance tracking
- **`use-mobile.tsx`**: Mobile detection
- **`useToast.ts`**: Toast notifications

#### **Feature Hooks**

- **`useAutonomousconversations.ts`**: Autonomous conversation management
- **`useConversationorchestrator.ts`**: Conversation orchestration

---

### 4. **CONTEXTS** (`web/contexts/`)

- **`is-sending-context.tsx`**: Message sending state
- **`llm-context.tsx`**: LLM provider context

---

### 5. **LIB** (`lib/`)

- **`api-key-storage.ts`**: API key management
- **`audit-logger.ts`**: Audit trail logging
- **`auth-middleware.ts`**: Authentication middleware
- **`crypto-utils.ts`**: Encryption utilities
- **`performance-monitor.ts`**: Performance tracking
- **`rate-limiter.ts`**: API rate limiting
- **`websocket-client.ts`**: WebSocket connection management

---

### 6. **TYPES** (`web/types/`)

- **`agents.ts`**: Agent type definitions
- **`beliefs.ts`**: Belief state types

---

### 7. **API ROUTES** (`api/rest/`)

#### **Agent Routes** (`agents/`)

- **`route.ts`**: Agent CRUD operations
- **`agentid/route.ts`**: Individual agent operations
- **`agentid/commands/route.ts`**: Agent command execution
- **`agentid/evaluate/route.ts`**: Agent evaluation
- **`agentid/export/route.ts`**: Agent export
- **`agentid/memory/route.ts`**: Agent memory access
- **`agentid/readiness/route.ts`**: Readiness assessment
- **`agentid/state/route.ts`**: State management

#### **Other API Routes**

- **`experiments/`**: Experiment management
- **`gnn/`**: GNN model operations
- **`knowledge/`**: Knowledge graph API
- **`llm/`**: LLM integration endpoints
- **`spatial/`**: Spatial queries
- **`api-key/`**: API key management

---

## 🔄 Integration Points

### **Backend ↔ Frontend Communication**

1. **REST API**: TypeScript routes → Python handlers
2. **GraphQL**: Apollo Client → GraphQL schema
3. **WebSocket**: Real-time bidirectional updates
4. **File Export/Import**: JSON/Binary data exchange

### **Key Data Flows**

1. **Agent Creation**: UI → API → Agent Factory → Database
2. **Simulation**: Engine → World State → WebSocket → UI Updates
3. **Active Inference**: Belief Updates → GNN → Visualization
4. **Coalition Formation**: Algorithm → Monitoring → UI Events

### **State Management**

- **Backend**: SQLAlchemy ORM, In-memory caches
- **Frontend**: React hooks, Context API, Local storage
- **Real-time**: WebSocket event streams

---

## 📊 Functionality Coverage

### **✅ Implemented**

- Complete agent lifecycle management
- Active Inference with PyMDP integration
- Multi-agent coalition formation
- Real-time monitoring and visualization
- GNN-based model specification
- Hardware abstraction layer
- Safety protocols and boundaries
- Export/deployment system

### **🔧 Partially Implemented**

- LLM integration (local models ready, cloud providers pending)
- Advanced coalition contracts
- Full edge deployment pipeline
- Performance optimization (basic optimization done)

### **📋 Architecture Ready (Not Implemented)**

- Distributed simulation
- Multi-region deployment
- Advanced learning algorithms
- Blockchain integration for contracts

---

This comprehensive inventory shows a sophisticated multi-agent AI platform with strong mathematical foundations, real-time visualization, and production-ready infrastructure. The modular architecture allows for easy extension and integration of new capabilities.
