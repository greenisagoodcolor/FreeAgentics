# Systematic Backend Coverage Improvement Plan
# Target: 80% Coverage Comprehensive Implementation

## Current Status Analysis
- **Total Backend Source Files:** 152
- **Current Unit Tests:** 67
- **Current Coverage:** 44.1% (67/152)
- **Target Coverage:** 80% (122/152)
- **Tests Needed:** 55 additional test files

## Phase 1: Critical Infrastructure (Week 1) - Priority 1
**Target:** Cover core system components first

### 1.1 Core Agent System (3 files)
```bash
# File: tests/unit/test_agent_core.py
# Covers: agents/base/agent.py (CRITICAL - core agent implementation)
```

### 1.2 API Infrastructure (2 files)  
```bash
# File: tests/unit/test_api_main.py
# Covers: api/main.py (CRITICAL - main API entry point)

# File: tests/unit/test_graphql_schema.py  
# Covers: api/graphql/schema.py
```

### 1.3 Simulation Engine (1 file)
```bash
# File: tests/unit/test_world_simulation_engine.py
# Covers: world/simulation/engine.py (CRITICAL - core simulation)
```

**Phase 1 Result:** 73/152 = 48.0% coverage

## Phase 2: Agent Specializations (Week 2) - Priority 2
**Target:** Complete agent system coverage

### 2.1 Agent Templates & Base Systems (6 files)
```bash
# File: tests/unit/test_agent_template.py
# Covers: agents/base/agent_template.py

# File: tests/unit/test_base_template.py  
# Covers: agents/templates/base_template.py

# File: tests/unit/test_explorer_template.py
# Covers: agents/templates/explorer_template.py

# File: tests/unit/test_pymdp_integration_templates.py
# Covers: agents/templates/pymdp_integration.py

# File: tests/unit/test_decision_making.py
# Covers: agents/base/decision_making.py

# File: tests/unit/test_interfaces.py
# Covers: agents/base/interfaces.py
```

### 2.2 Specialized Agent Types (8 files)
```bash
# File: tests/unit/test_explorer_agent.py
# Covers: agents/explorer/explorer.py, agents/explorer/explorer_behavior.py

# File: tests/unit/test_guardian_agent.py  
# Covers: agents/guardian/guardian.py, agents/guardian/guardian_behavior.py

# File: tests/unit/test_scholar_agent.py
# Covers: agents/scholar/scholar.py, agents/scholar/scholar_behavior.py

# File: tests/unit/test_merchant_behavior.py
# Covers: agents/merchant/merchant_behavior.py (merchant.py already covered)

# File: tests/unit/test_personality_system.py
# Covers: agents/base/personality_system.py

# File: tests/unit/test_belief_synchronization.py
# Covers: agents/base/belief_synchronization.py
```

**Phase 2 Result:** 87/152 = 57.2% coverage

## Phase 3: Inference & GNN Systems (Week 3) - Priority 2
**Target:** Complete inference engine coverage

### 3.1 Missing Engine Components (3 files)
```bash
# File: tests/unit/test_belief_visualization.py
# Covers: inference/engine/belief_visualization_interface.py

# File: tests/unit/test_gnn_integration_engine.py
# Covers: inference/engine/gnn_integration.py

# File: tests/unit/test_pymdp_generative_model_engine.py  
# Covers: inference/engine/pymdp_generative_model.py
```

### 3.2 GNN System Coverage (13 files - high impact)
```bash
# File: tests/unit/test_gnn_comprehensive.py
# Covers: inference/gnn/layers.py, inference/gnn/model.py, inference/gnn/cache_manager.py

# File: tests/unit/test_gnn_processing.py
# Covers: inference/gnn/executor.py, inference/gnn/generator.py, inference/gnn/validator.py

# File: tests/unit/test_gnn_monitoring.py  
# Covers: inference/gnn/monitoring.py, inference/gnn/monitoring_dashboard.py, inference/gnn/alerting.py

# File: tests/unit/test_gnn_datasets.py
# Covers: inference/gnn/benchmark_datasets.py, inference/gnn/testing_framework.py

# File: tests/unit/test_gnn_metrics.py
# Covers: inference/gnn/metrics_collector.py
```

### 3.3 LLM Integration (5 files)
```bash
# File: tests/unit/test_llm_integration.py
# Covers: inference/llm/belief_integration.py, inference/llm/local_llm_manager.py

# File: tests/unit/test_llm_quantization.py
# Covers: inference/llm/model_quantization.py, inference/llm/ollama_integration.py
```

**Phase 3 Result:** 108/152 = 71.1% coverage

## Phase 4: Coalition & Infrastructure (Week 4) - Priority 3
**Target:** Business logic and infrastructure

### 4.1 Coalition Core (8 files)
```bash
# File: tests/unit/test_coalition_business.py
# Covers: coalitions/coalition/business_opportunities.py, coalitions/coalition/coalition_criteria.py

# File: tests/unit/test_coalition_models.py
# Covers: coalitions/coalition/coalition_models.py

# File: tests/unit/test_coalition_contracts.py
# Covers: coalitions/contracts/coalition_contract.py, coalitions/contracts/resource_sharing.py

# File: tests/unit/test_coalition_deployment.py  
# Covers: coalitions/deployment/deployment_manifest.py, coalitions/deployment/edge_packager.py

# File: tests/unit/test_coalition_algorithms.py
# Covers: coalitions/formation/coalition_formation_algorithms.py, coalitions/formation/preference_matching.py

# File: tests/unit/test_expert_validation.py
# Covers: coalitions/formation/expert_committee_validation.py, coalitions/formation/stability_analysis.py
```

### 4.2 Infrastructure Export (7 files)
```bash
# File: tests/unit/test_infrastructure_export.py
# Covers: infrastructure/export/business_model_exporter.py, infrastructure/export/coalition_packaging.py

# File: tests/unit/test_deployment_scripts.py
# Covers: infrastructure/export/deployment_scripts.py, infrastructure/export/export_builder.py

# File: tests/unit/test_model_compression.py
# Covers: infrastructure/export/model_compression.py

# File: tests/unit/test_hardware_config.py
# Covers: infrastructure/export/hardware_config.py
```

**Phase 4 Result:** 122/152 = 80.3% coverage ✅

## Phase 5: Safety & Advanced Features (Week 5) - Priority 4
**Target:** Safety-critical and remaining components

### 5.1 Safety Systems (4 files)
```bash
# File: tests/unit/test_safety_protocols.py
# Covers: infrastructure/safety/safety_protocols.py

# File: tests/unit/test_boundary_monitoring.py
# Covers: infrastructure/safety/boundary_monitoring_service.py

# File: tests/unit/test_risk_mitigation.py  
# Covers: infrastructure/safety/risk_mitigation_metrics.py
```

### 5.2 World & Spatial Systems (5 files)
```bash
# File: tests/unit/test_world_systems.py
# Covers: world/grid_position.py, world/h3_world.py

# File: tests/unit/test_spatial_systems.py
# Covers: world/spatial/spatial_api.py, world/grid/hex_world.py, world/grid/spatial_index.py
```

### 5.3 API WebSocket (3 files)
```bash
# File: tests/unit/test_websocket_monitoring.py
# Covers: api/websocket/coalition_monitoring.py, api/websocket/markov_blanket_monitoring.py, api/websocket/real_time_updates.py
```

### 5.4 Deployment & Hardware (4 files)
```bash
# File: tests/unit/test_deployment_verification.py
# Covers: infrastructure/deployment/deployment_verification.py, infrastructure/deployment/export_validator.py

# File: tests/unit/test_hardware_compatibility.py
# Covers: infrastructure/deployment/hardware_compatibility.py

# File: tests/unit/test_hardware_infrastructure.py
# Covers: infrastructure/hardware/device_discovery.py, infrastructure/hardware/hal_core.py
```

**Phase 5 Result:** 138/152 = 90.8% coverage

## Implementation Integration with Makefile

### Update pyproject.toml for Systematic Testing
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers", 
    "--strict-config",
    "--cov=agents",
    "--cov=inference",
    "--cov=coalitions", 
    "--cov=infrastructure",
    "--cov=world",
    "--cov=api",
    "--cov=knowledge",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=80"
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests", 
    "unit: marks tests as unit tests",
    "critical: marks tests as critical infrastructure"
]
timeout = 300
```

### Makefile Integration Commands
```bash
# Add to existing Makefile
test-backend-coverage-phase1: setup
	@echo "$(BOLD)$(BLUE)🎯 Phase 1: Critical Infrastructure Coverage$(RESET)"
	. $(VENV_DIR)/bin/activate && pytest tests/unit/test_agent_core.py tests/unit/test_api_main.py tests/unit/test_world_simulation_engine.py -v --cov-report=term

test-backend-coverage-phase2: setup  
	@echo "$(BOLD)$(BLUE)🎯 Phase 2: Agent Specializations Coverage$(RESET)"
	. $(VENV_DIR)/bin/activate && pytest tests/unit/test_*agent*.py tests/unit/test_*template*.py -v --cov-report=term

test-backend-coverage-phase3: setup
	@echo "$(BOLD)$(BLUE)🎯 Phase 3: Inference & GNN Coverage$(RESET)" 
	. $(VENV_DIR)/bin/activate && pytest tests/unit/test_*gnn*.py tests/unit/test_*llm*.py tests/unit/test_belief_visualization.py -v --cov-report=term

test-backend-coverage-phase4: setup
	@echo "$(BOLD)$(BLUE)🎯 Phase 4: Coalition & Infrastructure Coverage$(RESET)"
	. $(VENV_DIR)/bin/activate && pytest tests/unit/test_coalition*.py tests/unit/test_infrastructure*.py -v --cov-report=term

test-backend-coverage-systematic: test-backend-coverage-phase1 test-backend-coverage-phase2 test-backend-coverage-phase3 test-backend-coverage-phase4
	@echo "$(BOLD)$(GREEN)✅ Systematic Backend Coverage Complete - Target: 80%$(RESET)"
	. $(VENV_DIR)/bin/activate && pytest --cov-report=html --cov-report=term-missing
```

## Success Metrics & Tracking
- **Phase 1:** 48.0% coverage (6 files added)
- **Phase 2:** 57.2% coverage (14 files added) 
- **Phase 3:** 71.1% coverage (21 files added)
- **Phase 4:** 80.3% coverage (15 files added) ✅ TARGET REACHED
- **Phase 5:** 90.8% coverage (16 files added) - Stretch goal

## Quality Standards for Each Test
1. **Comprehensive Function Coverage:** Test all public methods
2. **Edge Case Testing:** Include error conditions and boundary values
3. **Integration Testing:** Test module interactions where applicable
4. **Performance Verification:** Include basic performance assertions for critical paths
5. **Makefile Integration:** All tests must run through `make test-backend-coverage-systematic`

## No Tech Debt Policy
- Each test file must be complete and maintainable
- Use proper pytest fixtures and parametrization  
- Follow existing test patterns from well-covered modules
- Include proper docstrings and comments
- Integrate seamlessly with existing Makefile workflow
- No shortcuts or placeholder tests

This systematic approach will increase backend coverage from 44.1% to 80%+ while ensuring comprehensive testing of all critical components and seamless integration with the existing development workflow.