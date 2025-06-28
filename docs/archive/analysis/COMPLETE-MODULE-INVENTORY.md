# FreeAgentics Complete Module Inventory

Last Updated: 2025-06-28

## Overview

This document provides a comprehensive inventory of all modules in the FreeAgentics codebase, organized by major functional areas.

## Python Backend Modules

### 1. Agent System (`/agents`)

#### Core Agent Infrastructure (`/agents/base`)
- `agent.py` - Base agent class and core functionality
- `agent_factory.py` - Factory pattern for agent creation
- `agent_template.py` - Template system for agent types
- `behaviors.py` - Behavior definitions and management
- `belief_synchronization.py` - Belief state synchronization
- `communication.py` - Inter-agent communication protocols
- `data_model.py` - Agent data structures
- `decision_making.py` - Decision-making framework
- `epistemic_value_engine.py` - Epistemic value calculations
- `interaction.py` - Agent interaction protocols
- `interfaces.py` - Interface definitions
- `markov_blanket.py` - Markov blanket implementation
- `memory.py` - Agent memory system
- `movement.py` - Movement capabilities
- `perception.py` - Perception system
- `persistence.py` - State persistence
- `personality_system.py` - Personality modeling
- `resource_business_model.py` - Resource management
- `state_manager.py` - State management
- `world_integration.py` - World environment integration
- `active_inference_integration.py` - Active inference integration

#### Active Inference Components (`/agents/active_inference`)
- `generative_model.py` - Generative model implementation
- `precision.py` - Precision weighting

#### Core Behaviors (`/agents/core`)
- `active_inference.py` - Core active inference
- `movement_perception.py` - Movement and perception integration

#### Agent Types
- **Explorer** (`/agents/explorer`)
  - `explorer.py` - Explorer agent implementation
  - `explorer_behavior.py` - Explorer-specific behaviors
- **Guardian** (`/agents/guardian`)
  - `guardian.py` - Guardian agent implementation
  - `guardian_behavior.py` - Guardian-specific behaviors
- **Merchant** (`/agents/merchant`)
  - `merchant.py` - Merchant agent implementation
  - `merchant_behavior.py` - Merchant-specific behaviors
- **Scholar** (`/agents/scholar`)
  - `scholar.py` - Scholar agent implementation
  - `scholar_behavior.py` - Scholar-specific behaviors

#### Templates (`/agents/templates`)
- `base_template.py` - Base template for agents
- `explorer_template.py` - Explorer template
- `pymdp_integration.py` - PyMDP integration

#### Testing Framework (`/agents/testing`)
- `agent_test_framework.py` - Agent testing utilities

### 2. Inference Engine (`/inference`)

#### Core Engine (`/inference/engine`)
- `active_inference.py` - Main active inference engine
- `active_learning.py` - Active learning capabilities
- `belief_state.py` - Belief state management
- `belief_update.py` - Belief update mechanisms
- `belief_visualization_interface.py` - Visualization interface
- `computational_optimization.py` - Optimization algorithms
- `diagnostics.py` - Diagnostic tools
- `generative_model.py` - Generative model core
- `gnn_integration.py` - GNN integration
- `graphnn_integration.py` - GraphNN integration
- `hierarchical_inference.py` - Hierarchical inference
- `parameter_learning.py` - Parameter learning
- `policy_selection.py` - Policy selection mechanisms
- `precision.py` - Precision calculations
- `pymdp_generative_model.py` - PyMDP generative model
- `pymdp_policy_selector.py` - PyMDP policy selector
- `temporal_planning.py` - Temporal planning
- `uncertainty_quantification.py` - Uncertainty quantification
- `utils.py` - Utility functions

#### GNN Module (`/inference/gnn`)
- `active_inference.py` - GNN active inference
- `alerting.py` - Alert system
- `batch_processor.py` - Batch processing
- `benchmark_datasets.py` - Benchmark datasets
- `cache_manager.py` - Cache management
- `edge_processor.py` - Edge processing
- `executor.py` - Execution engine
- `feature_extractor.py` - Feature extraction
- `generator.py` - Model generator
- `layers.py` - Neural network layers
- `metrics_collector.py` - Metrics collection
- `model.py` - Core GNN model
- `model_mapper.py` - Model mapping
- `monitoring.py` - Monitoring system
- `monitoring_dashboard.py` - Monitoring dashboard
- `parser.py` - Model parser
- `performance_optimizer.py` - Performance optimization
- `testing_framework.py` - Testing framework
- `validator.py` - Model validation

#### LLM Integration (`/inference/llm`)
- `belief_integration.py` - LLM belief integration
- `fallback_mechanisms.py` - Fallback handling
- `local_llm_manager.py` - Local LLM management
- `model_quantization.py` - Model quantization
- `ollama_integration.py` - Ollama integration
- `provider_interface.py` - Provider interface

#### Algorithms (`/inference/algorithms`)
- `variational_message_passing.py` - VMP implementation

### 3. Coalition System (`/coalitions`)

#### Coalition Core (`/coalitions/coalition`)
- `business_opportunities.py` - Business opportunity analysis
- `coalition_criteria.py` - Coalition criteria
- `coalition_models.py` - Coalition models

#### Formation (`/coalitions/formation`)
- `business_value_engine.py` - Business value calculations
- `coalition_builder.py` - Coalition building logic
- `coalition_formation_algorithms.py` - Formation algorithms
- `expert_committee_validation.py` - Expert validation
- `monitoring_integration.py` - Monitoring integration
- `preference_matching.py` - Preference matching
- `stability_analysis.py` - Stability analysis

#### Readiness Assessment (`/coalitions/readiness`)
- `business_readiness_assessor.py` - Business readiness
- `comprehensive_readiness_integrator.py` - Readiness integration
- `readiness_evaluator.py` - Readiness evaluation
- `safety_compliance_verifier.py` - Safety compliance
- `technical_readiness_validator.py` - Technical validation

#### Contracts (`/coalitions/contracts`)
- `coalition_contract.py` - Contract definitions
- `resource_sharing.py` - Resource sharing protocols

#### Deployment (`/coalitions/deployment`)
- `deployment_manifest.py` - Deployment manifests
- `edge_packager.py` - Edge packaging

### 4. World Environment (`/world`)

#### Grid System (`/world/grid`)
- `hex_world.py` - Hexagonal world grid
- `spatial_index.py` - Spatial indexing

#### Simulation (`/world/simulation`)
- `engine.py` - Simulation engine

#### Spatial (`/world/spatial`)
- `spatial_api.py` - Spatial API

#### Core World
- `grid_position.py` - Grid positioning
- `h3_world.py` - H3 world implementation

### 5. Infrastructure (`/infrastructure`)

#### Database (`/infrastructure/database`)
- `connection.py` - Database connections
- `models.py` - Database models
- `manage.py` - Database management
- `seed.py` - Data seeding
- `alembic/` - Database migrations

#### Deployment (`/infrastructure/deployment`)
- `deployment_verification.py` - Deployment verification
- `export_validator.py` - Export validation
- `hardware_compatibility.py` - Hardware compatibility

#### Export (`/infrastructure/export`)
- `business_model_exporter.py` - Business model export
- `coalition_packaging.py` - Coalition packaging
- `deployment_scripts.py` - Deployment scripts
- `experiment_export.py` - Experiment export
- `export_builder.py` - Export builder
- `hardware_config.py` - Hardware configuration
- `model_compression.py` - Model compression

#### Hardware (`/infrastructure/hardware`)
- `device_discovery.py` - Device discovery
- `hal_core.py` - Hardware abstraction layer

#### Safety (`/infrastructure/safety`)
- `boundary_monitoring_service.py` - Boundary monitoring
- `markov_blanket_verification.py` - Markov blanket verification
- `risk_mitigation_metrics.py` - Risk metrics
- `safety_protocols.py` - Safety protocols

#### Docker (`/infrastructure/docker`)
- `Dockerfile.api` - API container
- `Dockerfile.web` - Web container
- `docker-compose.yml` - Compose configuration

### 6. API Layer (`/api`)

#### GraphQL (`/api/graphql`)
- `schema.py` - GraphQL schema

#### REST API (`/api/rest`)
- `agents/` - Agent endpoints
- `api-key/` - API key management
- `experiments/` - Experiment endpoints
- `gnn/` - GNN endpoints
- `knowledge/` - Knowledge graph endpoints
- `llm/` - LLM endpoints
- `spatial/` - Spatial endpoints

#### WebSocket (`/api/websocket`)
- `coalition_monitoring.py` - Coalition monitoring
- `markov_blanket_monitoring.py` - Markov blanket monitoring
- `real_time_updates.py` - Real-time updates

#### Core API
- `main.py` - Main API application

## Frontend Modules

### 1. Next.js Application (`/web/app`)

#### Pages
- `page.tsx` - Home page
- `layout.tsx` - Root layout
- `active-inference-demo/page.tsx` - Active inference demo
- `agents/page.tsx` - Agents page
- `ceo-demo/page.tsx` - CEO demo (NEW)
- `conversations/page.tsx` - Conversations page
- `experiments/page.tsx` - Experiments page
- `knowledge/page.tsx` - Knowledge page
- `world/page.tsx` - World visualization

#### Dashboard (`/web/app/dashboard`)
- **Panels** (`components/panels/`)
  - `AgentPanel/` - Agent management panel
  - `AnalyticsPanel/` - Analytics panel
  - `ControlPanel/` - Control panel
  - `ConversationPanel/` - Conversation panel
  - `GoalPanel/` - Goal tracking panel (NEW)
  - `KnowledgePanel/` - Knowledge panel
  - `MetricsPanel/` - Metrics panel

- **Layouts** (`layouts/`)
  - `BloombergLayout.tsx` - Bloomberg-style layout
  - `BloombergTerminalLayout.tsx` - Terminal layout (NEW)
  - `CEODemoLayout.tsx` - CEO demo layout (NEW)
  - `ImprovedBloombergLayout.tsx` - Improved layout (NEW)
  - `KnowledgeLayout.tsx` - Knowledge-focused layout
  - `ResizableLayout.tsx` - Resizable layout

### 2. React Components (`/web/components`)

#### Core Components
- `AgentList.tsx` - Agent list display
- `AgentList-stub.tsx` - Agent list stub (NEW)
- `CEODemoLanding.tsx` - CEO demo landing (NEW)
- `GlobalKnowledgeGraph.tsx` - Global knowledge graph
- `KnowledgeGraph.tsx` - Knowledge graph component
- `KnowledgeGraph-viz.tsx` - Knowledge graph visualization
- `WebSocketProvider.tsx` - WebSocket provider (NEW)
- `aboutmodal.tsx` - About modal
- `AboutButton.tsx` - About button
- `navbar.tsx` - Navigation bar
- `themeprovider.tsx` - Theme provider
- `errorboundary.tsx` - Error boundary

#### Agent Components
- `agentcard.tsx` - Agent card display
- `agentdashboard.tsx` - Agent dashboard
- `agentbeliefvisualizer.tsx` - Belief visualization
- `agent-activity-timeline.tsx` - Activity timeline
- `agent-performance-chart.tsx` - Performance charts
- `agent-relationship-network.tsx` - Relationship network
- `backend-agent-list.tsx` - Backend agent list

#### Conversation Components (`/components/conversation`)
- `conversation-dashboard.tsx` - Conversation dashboard
- `conversation-search.tsx` - Search functionality
- `message-components.tsx` - Message components
- `message-queue-visualization.tsx` - Queue visualization
- `optimized-conversation-dashboard.tsx` - Optimized dashboard
- `virtualized-message-list.tsx` - Virtualized list
- `autonomous-conversation-manager.tsx` - Autonomous conversations

#### Conversation Orchestration (`/components/conversation-orchestration`)
- `advanced-controls.tsx` - Advanced controls
- `change-history.tsx` - Change history
- `preset-selector.tsx` - Preset selection
- `real-time-preview.tsx` - Real-time preview
- `response-dynamics-controls.tsx` - Response dynamics
- `timing-controls.tsx` - Timing controls

#### Visualization Components
- `belief-state-mathematical-display.tsx` - Mathematical display
- `belief-trajectory-dashboard.tsx` - Trajectory dashboard
- `coalition-geographic-viz.tsx` - Geographic visualization
- `collective-intelligence-dashboard.tsx` - Collective intelligence
- `dual-layer-knowledge-graph.tsx` - Dual-layer graph
- `free-energy-landscape-viz.tsx` - Free energy landscape
- `gridworld.tsx` - Grid world display
- `knowledge-graph-analytics.tsx` - Graph analytics
- `markov-blanket-visualization.tsx` - Markov blanket viz
- `markov-blanket-dashboard.tsx` - Markov blanket dashboard
- `markov-blanket-configuration-ui.tsx` - Configuration UI
- `multi-agent-network-viz.tsx` - Multi-agent network
- `strategic-positioning-dashboard.tsx` - Strategic positioning

#### Dashboard Components (`/components/dashboard`)
- `AccessibilityEnhancements.tsx` - Accessibility features (NEW)
- `ActiveAgentsList.tsx` - Active agents list
- `AgentTemplateSelector.tsx` - Template selector
- `AnalyticsWidgetGrid.tsx` - Analytics grid
- `AnalyticsWidgetSystem.tsx` - Widget system
- `BeliefExtractionPanel.tsx` - Belief extraction
- `ConversationFeed.tsx` - Conversation feed
- `ConversationOrchestration.tsx` - Orchestration
- `ErrorBoundary.tsx` - Error boundary (NEW)
- `KnowledgeGraphVisualization.tsx` - Knowledge viz
- `LazyComponents.tsx` - Lazy loading (NEW)
- `MobileEnhancements.tsx` - Mobile features (NEW)
- `RealTimeDataSimulator.tsx` - Data simulator (NEW)
- `SpatialGrid.tsx` - Spatial grid
- `TiledPanel.tsx` - Tiled panel (NEW)
- `TilingWindowManager.tsx` - Window manager (NEW)

#### UI Components (`/components/ui`)
- Basic UI elements (button, card, dialog, etc.)
- `active-inference-dashboard.tsx` - Active inference UI
- `agent-configuration-form.tsx` - Configuration form
- `agent-creation-wizard.tsx` - Creation wizard
- `agent-instantiation-modal.tsx` - Instantiation modal
- `agent-template-selector.tsx` - Template selector
- `belief-state-visualization.tsx` - Belief viz
- `experiment-export-modal.tsx` - Export modal
- `experiment-import-modal.tsx` - Import modal
- `experiment-sharing-modal.tsx` - Sharing modal
- `free-energy-visualization.tsx` - Free energy viz
- `horizontal-template-selector.tsx` - Horizontal selector
- `llm-provider-manager.tsx` - LLM provider management
- `provider-connection-test.tsx` - Connection testing
- `provider-monitoring-dashboard.tsx` - Provider monitoring
- `secure-credential-input.tsx` - Secure input
- `spatial-mini-map.tsx` - Mini map

#### Other Components
- `backend-grid-world.tsx` - Backend grid world
- `character-creator.tsx` - Character creation
- `chat-window.tsx` - Chat window
- `conversation-view.tsx` - Conversation view
- `llmtest.tsx` - LLM testing
- `memoryviewer.tsx` - Memory viewer
- `readiness-panel.tsx` - Readiness panel
- `safety-compliance-dashboard.tsx` - Safety compliance
- `simulation-controls.tsx` - Simulation controls
- `tools-tab.tsx` - Tools tab

### 3. Library Modules (`/web/lib`)

#### API Integration (`/lib/api`)
- `agents-api.ts` - Agents API client
- `dashboard-api.ts` - Dashboard API
- `dashboard-api-functions.ts` - API functions
- `knowledge-graph.ts` - Knowledge graph API

#### LLM Services
- `llm-client.ts` - LLM client
- `llm-constants.ts` - LLM constants
- `llm-errors.ts` - Error handling
- `llm-providers.ts` - Provider management
- `llm-secure-client.ts` - Secure client
- `llm-service.ts` - LLM service
- `llm-settings.ts` - Settings management

#### Conversation Management
- `autonomous-conversation.ts` - Autonomous conversations
- `conversation-dynamics.ts` - Conversation dynamics
- `conversation-logger.ts` - Logging
- `conversation-orchestrator.ts` - Orchestration
- `conversation-preset-persistence.ts` - Preset persistence
- `conversation-preset-safety-validator.ts` - Safety validation
- `conversation-preset-validator.ts` - Preset validation

#### Knowledge Management
- `knowledge-export.ts` - Export functionality
- `knowledge-graph-management.ts` - Graph management
- `knowledge-import.ts` - Import functionality
- `knowledge-retriever.ts` - Data retrieval

#### Security & Auth (`/lib/auth`)
- `route-protection.tsx` - Route protection
- `api-key-migration.ts` - API key migration
- `api-key-service-server.ts` - Server-side API keys
- `api-key-storage.ts` - API key storage
- `crypto-client.ts` - Crypto utilities
- `encryption.ts` - Encryption utilities
- `security.ts` - Security utilities
- `session-management.ts` - Session management

#### Utilities & Services
- `active-inference.ts` - Active inference utilities
- `agent-system.ts` - Agent system utilities
- `audit-logger.ts` - Audit logging
- `belief-extraction.ts` - Belief extraction
- `browser-check.ts` - Browser compatibility
- `debug-logger.ts` - Debug logging
- `feature-flags.ts` - Feature flags
- `markov-blanket.ts` - Markov blanket utilities
- `message-queue.ts` - Message queue
- `prompt-templates.ts` - Prompt templates
- `rate-limit.ts` - Rate limiting
- `settings-export.ts` - Settings export
- `types.ts` - TypeScript types
- `utils.ts` - General utilities

#### Hooks (`/lib/hooks`)
- `use-dashboard-data.ts` - Dashboard data hook
- `use-llm-providers.ts` - LLM providers hook
- `use-provider-monitoring.ts` - Provider monitoring

#### Performance (`/lib/performance`)
- `memoization.ts` - Memoization utilities
- `performance-monitor.ts` - Performance monitoring

#### Compliance (`/lib/compliance`)
- `adr-validator.ts` - ADR validation
- `task-44-compliance-report.ts` - Compliance reporting

#### Safety (`/lib/safety`)
- `data-validation.ts` - Data validation

#### Services (`/lib/services`)
- `agent-creation-service.ts` - Agent creation (NEW)
- `compression-service.ts` - Data compression (NEW)
- `provider-monitoring-service.ts` - Provider monitoring (NEW)

#### Storage (`/lib/storage`)
- `data-validation-storage.ts` - Validation storage (NEW)
- `indexeddb-storage.ts` - IndexedDB storage (NEW)

#### Stores (`/lib/stores`)
- `dashboard-store.ts` - Dashboard state store (NEW)

#### Types (`/lib/types`)
- `agent-api.ts` - Agent API types

#### Utils (`/lib/utils`)
- `knowledge-graph-export.ts` - Export utilities (NEW)
- `knowledge-graph-filters.ts` - Filter utilities (NEW)

#### Workers (`/lib/workers`)
- `compression-worker.ts` - Compression worker (NEW)

### 4. React Hooks (`/web/hooks`)

- `use-mobile.tsx` - Mobile detection
- `use-toast.ts` - Toast notifications
- `useAsyncOperation.ts` - Async operations (NEW)
- `useAutoScroll.ts` - Auto-scrolling
- `useAutonomousconversations.ts` - Autonomous conversations
- `useConversationWebSocket.ts` - Conversation WebSocket
- `useConversationorchestrator.ts` - Conversation orchestrator
- `useDebounce.ts` - Debouncing
- `useKnowledgeGraphWebSocket.ts` - Knowledge graph WebSocket
- `useMarkovBlanketMonitoring.ts` - Markov blanket monitoring
- `useMarkovBlanketWebSocket.ts` - Markov blanket WebSocket
- `usePerformanceMonitor.ts` - Performance monitoring
- `useToast.ts` - Toast hook
- `useWebSocket.ts` - Generic WebSocket (NEW)

### 5. Services (`/web/services`)

- `socketService.ts` - Socket.io service

## Test Modules

### Python Tests (`/tests`)

#### Unit Tests (`/tests/unit`)
- Active learning, belief state, GNN components
- Agent components (memory, perception, movement, etc.)
- Coalition framework
- Inference engine components
- `web/test_knowledge_graph_component.py` - Web component tests (NEW)

#### Integration Tests (`/tests/integration`)
- `test_agent_integration.py` - Agent integration
- `test_active_inference_integration.py` - Active inference
- `test_gnn_integration.py` - GNN integration
- `test_spatial_integration.py` - Spatial integration
- `test_system_integration.py` - System integration
- `test_websocket_integration.py` - WebSocket integration (NEW)
- `test_performance.py` - Performance tests
- `agents/test_agent_orchestration.py` - Agent orchestration

#### Behavior Tests (`/tests/behavior`)
- `test_agent_scenarios.py` - Agent scenarios
- `test_frontend_agent_scenarios.py` - Frontend scenarios (NEW)

#### Property Tests (`/tests/property`)
- `test_active_inference_invariants.py` - Active inference invariants
- `test_frontend_invariants.py` - Frontend invariants (NEW)

#### Feature Tests (`/tests/features`)
- `agent_exploration.feature` - Agent exploration
- `coalition_formation.feature` - Coalition formation (NEW)

#### Other Test Categories
- `chaos/` - Chaos engineering tests
- `compliance/` - Compliance tests
- `contract/` - API contract tests
- `security/` - Security tests

### Frontend Tests (`/web/__tests__`)

- Component tests
- Hook tests
- Integration tests
- Accessibility tests
- Performance tests
- Edge case tests
- Service tests
- Context tests
- Library module tests

### E2E Tests (`/web/e2e`)

- `active-inference.spec.ts` - Active inference E2E
- `agents.spec.ts` - Agents E2E
- `dashboard.spec.ts` - Dashboard E2E
- `knowledge-graph.spec.ts` - Knowledge graph E2E
- `performance.spec.ts` - Performance E2E
- `smoke.spec.ts` - Smoke tests
- `visual-tests.spec.ts` - Visual regression (NEW)
- `websocket.spec.ts` - WebSocket E2E (NEW)
- `helpers/test-utils.ts` - Test utilities (NEW)

## Configuration Files

### Root Configuration
- `pyproject.toml` - Python project configuration
- `package.json` - Node.js dependencies
- `Makefile` - Build automation
- `commitlint.config.js` - Commit linting
- `.taskmaster/` - Task management system

### Python Configuration
- `requirements.txt` - Python dependencies
- `requirements-dev.txt` - Development dependencies
- `mypy.ini` - Type checking configuration

### Frontend Configuration
- `next.config.js` - Next.js configuration
- `tsconfig.json` - TypeScript configuration
- `tailwind.config.ts` - Tailwind CSS configuration
- `jest.config.js` - Jest testing configuration
- `playwright.config.ts` - Playwright E2E configuration
- `.prettierignore` - Prettier ignore rules (NEW)

## Documentation

### Main Documentation (`/docs`)
- Architecture decisions (`/adr`)
- API documentation (`/api`)
- Development guides
- User guides
- Active inference documentation
- GNN documentation
- Platform documentation

## Scripts and Tools

### Infrastructure Scripts (`/infrastructure/scripts`)
- Development scripts (`/development`)
- Deployment scripts (`/deployment`)
- Setup scripts (`/setup`)
- Testing scripts (`/testing`)
- Demo scripts (`/demo`)

### Root Scripts
- `run_pre_production_tests.sh` - Pre-production testing
- `scripts/setup-hooks.sh` - Git hooks setup

## Recent Changes

### New Modules Added
- CEO Demo functionality (`web/app/ceo-demo/`)
- Goal Panel for dashboard (`web/app/dashboard/components/panels/GoalPanel/`)
- Multiple new dashboard layouts (Bloomberg Terminal, CEO Demo, Improved Bloomberg)
- WebSocket provider and hook improvements
- Frontend test infrastructure expansion
- Dashboard enhancements (accessibility, mobile, error boundaries, lazy loading)
- Storage services (IndexedDB, data validation)
- Compression services and workers
- Additional E2E tests (visual regression, WebSocket)

### Removed Modules
- Several Python utility scripts (flake8 fixers, etc.)
- Dashboard consolidation plan document
- Testing gaps analysis document
- Old dashboard page (moved to layout-based system)

### Modified Modules
Based on git status, significant updates to:
- Agent memory and perception systems
- Coalition builder
- Active inference engine
- GNN layers and parser
- Multiple test suites
- Dashboard panel components
- Frontend coverage improvements

## Module Dependencies

### Core Dependencies
- **Agents** depend on: Inference Engine, World Environment
- **Coalitions** depend on: Agents, Infrastructure
- **Inference** provides services to: Agents, Coalitions
- **Frontend** consumes: API endpoints, WebSocket connections
- **Infrastructure** supports: All modules

### External Dependencies
- PyMDP for active inference
- PyTorch for neural networks
- Next.js 14 for frontend
- Socket.io for real-time communication
- PostgreSQL for persistence
- Docker for containerization

## Notes

- The codebase follows a clear separation between backend (Python) and frontend (TypeScript/React)
- Active inference is central to the agent architecture
- The system supports multiple agent types with specialized behaviors
- Coalition formation is a key feature for multi-agent collaboration
- The frontend includes comprehensive dashboard and visualization capabilities
- Testing coverage spans unit, integration, behavior, and E2E tests
- Recent updates focus on dashboard improvements and testing infrastructure