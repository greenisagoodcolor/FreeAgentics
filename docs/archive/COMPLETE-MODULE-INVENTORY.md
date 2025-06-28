# 📚 FreeAgentics Complete Module Inventory

> **Comprehensive catalog of all programmed functionality in backend and frontend**

Last Updated: 2025-06-28

---

## 🔧 BACKEND MODULES (Python)

### 1. **AGENTS MODULE** (`agents/`)

**Purpose**: Core autonomous agent system providing the foundation for intelligent, self-directed entities that can perceive, decide, and act within the simulation environment.

**Technical**: Built on Active Inference principles using PyMDP integration, with modular architecture supporting multiple agent types through inheritance and composition patterns.

**Intent**: Create a flexible agent framework that supports diverse behaviors while maintaining mathematical rigor through Active Inference formalism and Markov blanket boundaries.

**Implementation Status**: ✅ Fully implemented with 4 agent types (Explorer, Guardian, Merchant, Scholar), comprehensive testing framework, and production-ready state management.

**Potential TODOs**:
- Add reinforcement learning capabilities alongside Active Inference
- Implement agent breeding/evolution mechanisms
- Create visual agent designer UI
- Add more sophisticated communication protocols (e.g., contract negotiation)
- Implement agent skill trees and progression systems

#### **Core Agent Framework** (`agents/base/`)

- **`agent.py`**: Base Agent class - handles lifecycle, state, actions
- **`agent_factory.py`**: Factory pattern for creating different agent types
- **`agent_template.py`**: Template system for reusable agent configurations
- **`data_model.py`**: Core data structures (Agent, Position, Resources, Status)
- **`interfaces.py`**: Protocol definitions for agent contracts

#### **Agent Capabilities** (`agents/base/`)

**Purpose**: Modular capability system allowing agents to perceive, think, remember, communicate, and act in sophisticated ways.

**Technical**: Each capability is implemented as a separate module that can be mixed and matched, using dependency injection for flexibility.

**Implementation Status**: ✅ Core capabilities fully implemented with extensive unit testing.

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

**Purpose**: Enable autonomous agents to form temporary or permanent groups for achieving shared goals, resource optimization, and emergent collective behaviors.

**Technical**: Implements game-theoretic coalition formation algorithms, stability analysis using Shapley values, and Active Inference-based group decision making.

**Intent**: Support emergence of complex multi-agent behaviors including markets, governance structures, and collaborative problem-solving without central control.

**Implementation Status**: ✅ Core formation algorithms implemented, 🔧 Smart contracts partially done, ⏳ Distributed consensus mechanisms planned.

**Potential TODOs**:
- Implement blockchain-based coalition contracts
- Add coalition reputation systems
- Create coalition visualization dashboard
- Support hierarchical coalition structures (coalitions of coalitions)
- Add coalition dissolution strategies
- Implement voting mechanisms for coalition decisions

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

**Purpose**: Provide the mathematical and computational foundation for Active Inference, enabling agents to minimize free energy through perception and action.

**Technical**: Implements variational message passing, gradient descent on free energy, hierarchical inference, and integrates with PyMDP for established Active Inference algorithms.

**Intent**: Create a scientifically rigorous inference engine that can scale from simple grid worlds to complex multi-agent scenarios while maintaining theoretical coherence.

**Implementation Status**: ✅ Core Active Inference implemented, ✅ GNN integration complete, 🔧 LLM integration in progress.

**Potential TODOs**:
- Add GPU acceleration for large-scale inference
- Implement continuous state spaces (currently discrete)
- Add causal inference capabilities
- Create inference debugging and visualization tools
- Support for partially observable environments
- Implement meta-learning for generative model adaptation

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

**Purpose**: Provide robust foundation for deployment, persistence, safety, and hardware abstraction, enabling the system to run reliably in production environments.

**Technical**: Uses SQLAlchemy for data persistence, Alembic for migrations, hardware abstraction layer for edge deployment, and comprehensive safety protocols.

**Intent**: Ensure the platform can be deployed safely and reliably across diverse environments from cloud to edge devices while maintaining data integrity and system boundaries.

**Implementation Status**: ✅ Database and safety systems complete, ✅ Export functionality ready, 🔧 Edge deployment partially implemented.

**Potential TODOs**:
- Add Kubernetes deployment manifests
- Implement distributed database sharding
- Create automated backup and recovery
- Add hardware performance profiling
- Implement zero-downtime deployment strategies
- Add support for heterogeneous computing (CPU/GPU/TPU)

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

**Purpose**: Maintain a shared knowledge graph that agents can query and update, enabling collective intelligence and persistent learning.

**Technical**: Graph database implementation with efficient querying, versioning, and access control. Supports both local and distributed knowledge graphs.

**Intent**: Create a Wikipedia-like shared knowledge system that agents can collaboratively build and reference for decision-making.

**Implementation Status**: ✅ Basic knowledge graph implemented, ⏳ Advanced features planned.

**Potential TODOs**:
- Add knowledge graph embedding for similarity search
- Implement knowledge validation and trust scoring
- Create knowledge visualization interface
- Add federated knowledge graph support
- Implement knowledge pruning and compression
- Add semantic reasoning capabilities

- **`knowledge_graph.py`**: Knowledge graph implementation
  - Node and edge management
  - Graph queries and traversal
  - Knowledge persistence

---

### 6. **WORLD MODULE** (`world/`)

**Purpose**: Provide spatial environment and physics simulation where agents exist, move, and interact with resources and each other.

**Technical**: Uses H3 hexagonal hierarchical spatial indexing for efficient spatial queries, supports both 2D and 3D environments, and includes resource distribution systems.

**Intent**: Create rich, spatially-aware environments that support emergence of territorial behaviors, resource competition, and geographic coalition formation.

**Implementation Status**: ✅ Hexagonal grid world implemented, ✅ Spatial indexing complete, 🔧 3D environments in design phase.

**Potential TODOs**:
- Add physics simulation (gravity, collision)
- Implement weather and environmental effects
- Create procedural world generation
- Add support for multiple world instances
- Implement world persistence and snapshots
- Add terrain types and movement costs

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

**Purpose**: Expose platform functionality through multiple API paradigms (REST, GraphQL, WebSocket) for diverse client needs.

**Technical**: FastAPI for REST endpoints, GraphQL schema with resolvers, WebSocket for real-time updates, comprehensive API documentation with OpenAPI.

**Intent**: Provide flexible, well-documented APIs that support both simple CRUD operations and complex real-time simulations.

**Implementation Status**: ✅ REST API complete, ✅ WebSocket streaming ready, 🔧 GraphQL partially implemented.

**Potential TODOs**:
- Add gRPC support for high-performance scenarios
- Implement API versioning strategy
- Add API usage analytics
- Create API client SDKs for multiple languages
- Implement webhook system for external integrations
- Add batch operations support

#### **GraphQL** (`api/graphql/`)

- **`schema.py`**: GraphQL schema definitions

#### **WebSocket** (`api/websocket/`)

- **`real_time_updates.py`**: Real-time state streaming
- **`coalition_monitoring.py`**: Coalition event streaming
- **`markov_blanket_monitoring.py`**: Boundary monitoring streams

---

## 🎨 FRONTEND MODULES (TypeScript/React)

### 1. **APPLICATION STRUCTURE** (`web/app/`)

**Purpose**: Next.js 13+ app directory structure providing the main application pages, layouts, and routing for the multi-agent AI platform interface.

**Technical**: Uses Next.js App Router with server components, TypeScript for type safety, and modular page structure for different platform features.

**Intent**: Create an intuitive, high-performance web interface that makes complex AI concepts accessible through visualization and real-time interaction.

**Implementation Status**: ✅ Core pages implemented, ✅ Dashboard system complete, ✅ CEO demo ready, 🔧 Some advanced features in progress.

**Potential TODOs**:
- Add internationalization (i18n) support
- Implement PWA capabilities
- Create mobile-specific layouts
- Add A/B testing framework
- Implement analytics tracking
- Add offline mode support

#### **Pages**

- **`page.tsx`**: Landing page
- **`agents/page.tsx`**: Agent management interface
- **`world/page.tsx`**: World visualization
- **`experiments/page.tsx`**: Experiment management
- **`knowledge/page.tsx`**: Knowledge graph interface
- **`conversations/page.tsx`**: Agent conversations
- **`active-inference-demo/page.tsx`**: Active Inference visualization
- **`ceo-demo/page.tsx`**: CEO demonstration interface

#### **Dashboard** (`web/app/dashboard/`)

- **Dashboard Components** (`components/panels/`)
  - **`AgentPanel/index.tsx`**: Agent monitoring panel
  - **`AnalyticsPanel/index.tsx`**: Analytics and metrics panel
  - **`ControlPanel/index.tsx`**: System control panel
  - **`ConversationPanel/index.tsx`**: Conversation monitoring
  - **`KnowledgePanel/index.tsx`**: Knowledge graph panel
  - **`MetricsPanel/index.tsx`**: Performance metrics panel
  - **`GoalPanel/index.tsx`**: Goal tracking panel

- **Dashboard Layouts** (`layouts/`)
  - **`BloombergLayout.tsx`**: Bloomberg terminal style layout
  - **`BloombergTerminalLayout.tsx`**: Enhanced Bloomberg terminal
  - **`ImprovedBloombergLayout.tsx`**: Optimized Bloomberg layout
  - **`KnowledgeLayout.tsx`**: Knowledge-focused layout
  - **`ResizableLayout.tsx`**: Flexible resizable layout
  - **`CEODemoLayout.tsx`**: CEO demo specific layout

#### **Other App Features** (`web/app/`)

- **`conversation-orchestration/page.tsx`**: Conversation control panel
- **`layout.tsx`**: Root layout wrapper

---

### 2. **COMPONENTS** (`web/components/`)

**Purpose**: Comprehensive React component library providing reusable UI elements, visualizations, and domain-specific interfaces for agent management and monitoring.

**Technical**: Built with React 18+, TypeScript, Tailwind CSS for styling, and Radix UI for accessible primitives. Uses React hooks for state management.

**Intent**: Provide a consistent, accessible, and performant component system that can scale from simple forms to complex real-time visualizations.

**Implementation Status**: ✅ UI primitives complete, ✅ Domain components implemented, 🔧 Advanced visualizations ongoing.

**Potential TODOs**:
- Create Storybook documentation
- Add component performance profiling
- Implement component lazy loading strategies
- Create component theming system
- Add animation library integration
- Build component testing framework

#### **Core UI Components** (`web/components/ui/`)

**Purpose**: Foundational UI building blocks based on accessible patterns and consistent design system.

**Technical**: Built on Radix UI primitives with Tailwind styling, full TypeScript support, and ARIA compliance.

**Implementation Status**: ✅ All core components implemented with full accessibility support.

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

**Purpose**: Specialized components for creating, configuring, monitoring, and managing autonomous agents.

**Technical**: Integrates with backend agent APIs, uses WebSocket for real-time updates, and provides rich visualization of agent states.

**Implementation Status**: ✅ Core agent UI complete, 🔧 Advanced agent designer in development.

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

**Purpose**: Visualize complex mathematical concepts from Active Inference theory including beliefs, free energy, and Markov blankets.

**Technical**: Uses D3.js and Three.js for data visualization, WebGL for 3D landscapes, and real-time data streaming for live updates.

**Implementation Status**: ✅ Basic visualizations complete, 🔧 3D free energy landscapes in progress.

**Potential TODOs**:
- Add VR support for 3D visualizations
- Implement visualization recording/playback
- Create interactive tutorials
- Add GPU-accelerated rendering
- Support for multi-agent belief visualization

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

**Purpose**: Render and interact with spatial environments where agents live, including hexagonal grids and resource distributions.

**Technical**: Canvas-based rendering for performance, H3 library for hexagonal indexing, and efficient viewport culling for large worlds.

**Implementation Status**: ✅ 2D hex grid complete, 🔧 3D world visualization planned.

- **`world-visualization.tsx`**: Main world display
- **`backend-grid-world.tsx`**: Grid world visualization
- **`hex-grid.tsx`**: Hexagonal grid display
- **`spatial-mini-map.tsx`**: Minimap overlay
- **`coalition-geographic-viz.tsx`**: Coalition spatial display
- **`resource-heatmap.tsx`**: Resource distribution map

#### **Conversation Components** (`web/components/conversation/`)

**Purpose**: Enable and visualize agent-to-agent and human-to-agent conversations with rich formatting and real-time updates.

**Technical**: Virtual scrolling for performance, WebSocket for real-time messages, and Markdown rendering for rich text.

**Implementation Status**: ✅ Conversation UI complete, ✅ Performance optimizations done.

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

#### **Dashboard Components** (`web/components/dashboard/`)

- **`KnowledgeGraphVisualization.tsx`**: Interactive knowledge graph
- **`AccessibilityEnhancements.tsx`**: Accessibility features
- **`ErrorBoundary.tsx`**: Dashboard error boundary
- **`LazyComponents.tsx`**: Lazy-loaded dashboard components
- **`MobileEnhancements.tsx`**: Mobile-optimized features
- **`RealTimeDataSimulator.tsx`**: Real-time data simulation
- **`TiledPanel.tsx`**: Tiled panel component
- **`TilingWindowManager.tsx`**: Window tiling management

#### **Other Components**

- **`ErrorBoundary.tsx`**: Error handling wrapper
- **`themeprovider.tsx`**: Theme management
- **`simulation-controls.tsx`**: Simulation control panel
- **`strategic-positioning-dashboard.tsx`**: Strategic overview
- **`AboutButton.tsx`**: About dialog trigger
- **`aboutmodal.tsx`**: About information modal
- **`CEODemoLanding.tsx`**: CEO demo landing page
- **`WebSocketProvider.tsx`**: WebSocket context provider
- **`AgentList-stub.tsx`**: Agent list stub component

---

### 3. **HOOKS** (`web/hooks/`)

**Purpose**: Custom React hooks providing reusable logic for state management, API integration, and real-time features.

**Technical**: Built on React 18 hooks API, TypeScript for type safety, and follows React best practices for hook composition.

**Intent**: Encapsulate complex logic in reusable, testable units that can be composed into components.

**Implementation Status**: ✅ Core hooks implemented, ✅ WebSocket hooks tested, 🔧 Advanced state management ongoing.

**Potential TODOs**:
- Add hook testing utilities
- Create hook documentation generator
- Implement performance monitoring hooks
- Add error boundary hooks
- Create data fetching hook abstractions
- Add state persistence hooks

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
- **`useAsyncOperation.ts`**: Async operation management
- **`useWebSocket.ts`**: Generic WebSocket hook

#### **Feature Hooks**

- **`useAutonomousconversations.ts`**: Autonomous conversation management
- **`useConversationorchestrator.ts`**: Conversation orchestration

---

### 4. **CONTEXTS** (`web/contexts/`)

**Purpose**: React Context providers for global state management and cross-component communication.

**Technical**: Uses React Context API with TypeScript, implements provider patterns, and includes performance optimizations.

**Intent**: Manage global application state without prop drilling while maintaining performance.

**Implementation Status**: ✅ Core contexts implemented.

**Potential TODOs**:
- Add context devtools
- Implement context persistence
- Create context composition utilities
- Add context performance monitoring

- **`is-sending-context.tsx`**: Message sending state
- **`llm-context.tsx`**: LLM provider context

---

### 5. **LIB** (`web/lib/`)

**Purpose**: Utility libraries, services, and business logic that support the frontend application.

**Technical**: Pure TypeScript modules, service classes for API integration, and utility functions for common operations.

**Intent**: Separate business logic from UI components, provide reusable services, and maintain clean architecture.

**Implementation Status**: ✅ Core services implemented, ✅ LLM integration complete, 🔧 Advanced features ongoing.

**Potential TODOs**:
- Add service worker support
- Implement offline capabilities
- Create data caching layer
- Add request retry logic
- Implement API mocking for development
- Build error tracking service

#### **API & Authentication**
- **`api-key-migration.ts`**: API key migration utilities
- **`api-key-service-server.ts`**: Server-side API key service
- **`api-key-storage.ts`**: API key storage management
- **`encryption.ts`**: Encryption utilities
- **`session-management.ts`**: Session handling
- **`rate-limit.ts`**: Rate limiting implementation

#### **LLM Integration**
- **`llm-client.ts`**: LLM client implementation
- **`llm-constants.ts`**: LLM configuration constants
- **`llm-errors.ts`**: LLM error handling
- **`llm-providers.ts`**: Multiple LLM provider support
- **`llm-secure-client.ts`**: Secure LLM client
- **`llm-service.ts`**: LLM service layer
- **`llm-settings.ts`**: LLM settings management
- **`prompt-templates.ts`**: Prompt template system

#### **Conversation & Knowledge**
- **`autonomous-conversation.ts`**: Autonomous conversation logic
- **`belief-extraction.ts`**: Belief state extraction
- **`conversation-dynamics.ts`**: Conversation behavior dynamics
- **`conversation-logger.ts`**: Conversation logging
- **`conversation-orchestrator.ts`**: Conversation orchestration
- **`knowledge-export.ts`**: Knowledge graph export
- **`knowledge-import.ts`**: Knowledge graph import
- **`knowledge-retriever.ts`**: Knowledge retrieval service

#### **Utilities**
- **`browser-check.ts`**: Browser compatibility
- **`debug-logger.ts`**: Debug logging utilities
- **`feature-flags.ts`**: Feature flag management
- **`settings-export.ts`**: Settings export functionality
- **`utils.ts`**: General utilities

#### **Types**
- **`types.ts`**: Core type definitions
- **`types/agent-api.ts`**: Agent API types

---

### 6. **TYPES** (`web/types/`)

**Purpose**: TypeScript type definitions ensuring type safety across the frontend application.

**Technical**: Comprehensive type definitions, interfaces, and enums that match backend data models.

**Intent**: Provide compile-time type safety and excellent IDE support for development.

**Implementation Status**: ✅ Core types defined, continuously updated with API changes.

**Potential TODOs**:
- Generate types from OpenAPI spec
- Add runtime type validation
- Create type documentation
- Implement type versioning
- Add type testing utilities

- **`agents.ts`**: Agent type definitions
- **`beliefs.ts`**: Belief state types

---

### 7. **API ROUTES** (`api/rest/`)

**Purpose**: Next.js API routes providing backend functionality directly within the frontend application.

**Technical**: Server-side TypeScript handlers, integration with Python backend via HTTP, and middleware for auth/validation.

**Intent**: Provide a BFF (Backend for Frontend) layer that handles auth, caching, and API aggregation.

**Implementation Status**: ✅ Core routes implemented, ✅ WebSocket proxy ready.

**Potential TODOs**:
- Add request caching layer
- Implement rate limiting
- Create API documentation
- Add request logging
- Implement API versioning
- Add GraphQL federation support

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

**Purpose**: Define clear interfaces and data flows between system components to ensure modularity and maintainability.

**Technical**: Multiple communication protocols optimized for different use cases - REST for CRUD, WebSocket for real-time, GraphQL for complex queries.

**Intent**: Create a flexible architecture that supports both simple integrations and complex real-time scenarios while maintaining clear boundaries.

**Implementation Status**: ✅ Core integrations complete, 🔧 GraphQL being expanded.

**Potential TODOs**:
- Add message queue integration (RabbitMQ/Kafka)
- Implement event sourcing
- Add distributed tracing
- Create integration testing framework
- Add API gateway layer

### **Backend ↔ Frontend Communication**

1. **REST API**: TypeScript routes → Python handlers
   - **Purpose**: Standard CRUD operations and command execution
   - **Technical**: JSON over HTTP, OpenAPI documentation
   - **Status**: ✅ Fully implemented

2. **GraphQL**: Apollo Client → GraphQL schema
   - **Purpose**: Complex queries with nested relationships
   - **Technical**: GraphQL subscriptions for real-time updates
   - **Status**: 🔧 Partially implemented

3. **WebSocket**: Real-time bidirectional updates
   - **Purpose**: Live data streaming and event notifications
   - **Technical**: Socket.io with room-based channels
   - **Status**: ✅ Fully implemented
   - Conversation updates
   - Knowledge graph changes
   - Markov blanket monitoring
   - Agent state changes

4. **File Export/Import**: JSON/Binary data exchange
   - **Purpose**: Backup, sharing, and deployment packages
   - **Technical**: Compressed JSON with versioning
   - **Status**: ✅ Implemented

5. **Storage Services**: IndexedDB integration for client-side persistence
   - **Purpose**: Offline support and performance caching
   - **Technical**: Dexie.js wrapper around IndexedDB
   - **Status**: ✅ Implemented

6. **Compression Services**: Web Workers for data compression
   - **Purpose**: Reduce bandwidth for large datasets
   - **Technical**: LZ4 compression in background threads
   - **Status**: ✅ Implemented

### **Key Data Flows**

1. **Agent Creation Flow**
   - **Path**: UI Form → Validation → REST API → Agent Factory → Database → WebSocket Event
   - **Purpose**: Create and persist new agents with real-time UI updates
   - **Technical**: Transaction-wrapped creation with rollback on failure
   - **Status**: ✅ Fully implemented

2. **Simulation Loop**
   - **Path**: Simulation Engine → Agent Updates → World State → WebSocket → UI Renders
   - **Purpose**: Real-time simulation with synchronized visualization
   - **Technical**: 60 FPS update loop with delta compression
   - **Status**: ✅ Implemented

3. **Active Inference Pipeline**
   - **Path**: Sensory Input → Belief Updates → Free Energy Calc → Action Selection → GNN Processing → 3D Visualization
   - **Purpose**: Mathematical inference with visual feedback
   - **Technical**: PyMDP integration with GPU acceleration
   - **Status**: ✅ Core implemented, 🔧 GPU optimization ongoing

4. **Coalition Formation Workflow**
   - **Path**: Trigger Event → Candidate Selection → Formation Algorithm → Stability Check → Contract Creation → Monitoring Dashboard
   - **Purpose**: Autonomous group formation with human oversight
   - **Technical**: Game-theoretic algorithms with real-time visualization
   - **Status**: ✅ Algorithms ready, 🔧 Smart contracts in progress

### **State Management**

**Purpose**: Maintain consistent state across distributed components while optimizing for performance and user experience.

**Technical**: Multi-tier state management with appropriate persistence and synchronization strategies for each tier.

**Implementation Status**: ✅ Core state management implemented, 🔧 Distributed state sync in design.

- **Backend State**
  - **Persistent**: SQLAlchemy ORM with PostgreSQL
  - **Cache**: Redis for session and frequently accessed data
  - **In-Memory**: Agent state for active simulations
  - **Status**: ✅ Fully implemented

- **Frontend State**
  - **Component**: React hooks for local state
  - **Global**: Context API for app-wide state
  - **Persistent**: LocalStorage for user preferences
  - **Cache**: IndexedDB for offline data
  - **Status**: ✅ Implemented

- **Real-time Sync**
  - **Protocol**: WebSocket event streams
  - **Conflict Resolution**: Last-write-wins with optional CRDTs
  - **Recovery**: Automatic reconnection with state reconciliation
  - **Status**: ✅ Basic sync implemented, 🔧 Advanced conflict resolution planned

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
- Enhanced dashboard with multiple layouts (Bloomberg Terminal style)
- CEO demonstration interface
- Comprehensive WebSocket infrastructure
- Multi-provider LLM integration
- Goal tracking and analytics panels
- Mobile and accessibility enhancements

### **🔧 Partially Implemented**

- Advanced coalition contracts
- Full edge deployment pipeline
- Distributed simulation across multiple nodes
- Blockchain integration for contracts

### **📋 Architecture Ready (Not Implemented)**

- Multi-region deployment
- Advanced learning algorithms beyond Active Inference
- Real-time collaborative editing of experiments
- Advanced data compression for edge devices

---

This comprehensive inventory shows a sophisticated multi-agent AI platform with strong mathematical foundations, real-time visualization, and production-ready infrastructure. The modular architecture allows for easy extension and integration of new capabilities.

## 📝 Recent Updates (2025-06-28)

### **New Features**
- CEO Demo functionality with dedicated layout and landing page
- Goal Panel for tracking agent and system objectives
- Enhanced dashboard layouts including Bloomberg Terminal style interfaces
- Improved WebSocket infrastructure with dedicated provider
- Comprehensive LLM integration supporting multiple providers
- Mobile and accessibility enhancements for dashboard
- Real-time data simulation capabilities
- Tiling window manager for flexible layouts

### **Enhanced Components**
- Dashboard split into modular panels (Agent, Analytics, Control, Conversation, Knowledge, Metrics, Goal)
- Multiple dashboard layout options for different use cases
- Lazy loading and performance optimizations
- Error boundaries at component level
- Enhanced type safety with dedicated API types

### **Infrastructure Improvements**
- Session management and encryption utilities
- API key migration and secure storage
- Rate limiting implementation
- Feature flag system for gradual rollouts
- Debug logging infrastructure
- Browser compatibility checking

### **Documentation Restructure Initiative**
- Consolidated documentation into clearer top-level categories (Overview, Architecture, API, Guides, Tutorials, Examples).
- Removed duplicated files across `architecture/`, `gnn/`, and `platform/` by merging identical content and adding canonical references.
- Centralized ADRs under `architecture/adr/` and updated index files accordingly.
- Added `docs/README.md` table of contents that points to the new structure for easier navigation.
- Deprecated legacy tutorial paths in favour of a single `tutorials/` index with category anchors.
- Introduced lint script `docs:validate` (Markdown-lint + broken-link checker) to prevent future drift.
