# Free Energy Metrics Scan Results

## Summary
This document captures all discovered free energy, belief, and Active Inference metrics already computed by PyMDP in the FreeAgentics codebase.

## Discovered Metrics

### 1. Free Energy (F) Values

| File | Line | Description | Sample Usage |
|------|------|-------------|--------------|
| `/agents/base_agent.py` | 1047-1048 | Computes average free energy from PyMDP agent | `self.metrics["avg_free_energy"] = float(np.mean(self.pymdp_agent.F))` |
| `/agents/base_agent.py` | 532, 931 | Initializes F value to 0.0 | `self.pymdp_agent.F = 0.0` |
| `/knowledge/models.py` | 134 | Database column for free energy | `free_energy = Column(Float, nullable=True)` |

### 2. Belief States (qs)

| File | Line | Description | Sample Usage |
|------|------|-------------|--------------|
| `/agents/base_agent.py` | 1105 | Stores belief state posterior | `self.beliefs["state_posterior"] = [q.tolist() for q in qs]` |
| `/agents/base_agent.py` | 1143-1144 | Access posterior beliefs | `qs = self.pymdp_agent.qs` |
| `/agents/resource_collector.py` | 524-525 | Checks and accesses beliefs | `if hasattr(self.pymdp_agent, "qs")` |

### 3. Agent Metrics in Examples

| File | Line | Description | Sample Output |
|------|------|-------------|---------------|
| `/examples/active_inference_demo.py` | 167 | Total free energy display | `F=1.234` |
| `/examples/active_inference_demo.py` | 270 | Free energy from metrics dict | `agent.metrics.get('total_free_energy', 0)` |
| `/examples/demo.py` | 122 | Expected free energy | `explorer.metrics['expected_free_energy']` |

### 4. API Endpoints

| Endpoint | File | Description | Response Format |
|----------|------|-------------|-----------------|
| `/api/v1/agents/{agent_id}/metrics` | `/api/v1/agents.py:299` | Get agent metrics | `AgentMetrics` model |
| `/api/v1/monitoring/agents/{agent_id}/beliefs` | `/api/v1/monitoring.py:371` | Get belief stats | Dict with belief data |

## Viable Candidates for UI Display

### Primary Candidate: Average Free Energy
- **Source**: `base_agent.py` line 1048
- **Update Frequency**: Every 10 steps (performance optimization)
- **Access**: Via `agent.metrics["avg_free_energy"]`
- **Format**: Single float value

### Secondary Candidate: Belief State Posterior
- **Source**: `base_agent.py` line 1105
- **Update Frequency**: Every inference cycle
- **Access**: Via `agent.beliefs["state_posterior"]`
- **Format**: List of probability distributions

## Integration Points

1. **Existing Metrics Hook**: `/web/hooks/use-metrics.ts` already fetches from `/api/v1/metrics`
2. **Agent State**: Could extend to fetch `/api/v1/agents/{id}/metrics` for individual agents
3. **WebSocket Updates**: Agent events already broadcast via WebSocket, could include F values

## Recommended Approach

The simplest integration would be to:
1. Extend the existing agent metrics endpoint to include `avg_free_energy`
2. Display it in the MetricsFooter component alongside CPU/Memory
3. Or add a small card in the Agent panel showing "Free Energy: X.XXX"