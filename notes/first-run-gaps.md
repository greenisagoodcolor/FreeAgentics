# First-Run E2E Gaps Analysis

## Module Status Summary

### ✅ Found & Implemented
1. **PromptBar** - `/web/components/main/PromptBar.tsx` - Has tests in `PromptBar.test.tsx`
2. **AgentCreator** - `/web/components/main/AgentCreatorPanel.tsx` - Has tests in `AgentCreatorPanel.test.tsx`
3. **Conversation** - `/web/components/main/ConversationWindow.tsx` - Has tests in `ConversationWindow.test.tsx`
4. **Demo WebSocket** - `/api/v1/websocket.py` has `/ws/demo` endpoint implemented
5. **Frontend WebSocket Hooks** - Both `use-websocket.ts` and `use-prompt-processor.ts` configured for demo endpoint

### ✅ Completed Components
1. **KnowledgeGraph** 
   - Found: `/web/components/main/KnowledgeGraphView.tsx` exists
   - Implemented: Node click opens details side-sheet
   - Tests: Added test for node click functionality
   
2. **GridWorld/SimulationGrid**
   - Found: `/web/components/main/SimulationGrid.tsx` exists
   - Tests: Created test for agent visibility and movement
   
3. **Metrics Endpoint**
   - Found: `/metrics` endpoint already exists in api/main.py
   - Tests: Created test to verify plain text prometheus format

## Action Items for Each Module

### PromptBar
- [ ] Verify Settings drawer opens on prompt submission
- [ ] Check history persistence

### AgentCreator
- [ ] Verify agent list refresh after create/destroy
- [ ] Check WebSocket integration for real-time updates

### Conversation
- [ ] Verify 3-message cycle rendering (goal, GMN, result)
- [ ] Check WebSocket message flow

### KnowledgeGraph
- [x] Write tests for node display
- [x] Implement click handler for details side-sheet
- [ ] Wire up to real data from WebSocket (already connected via useKnowledgeGraph hook)

### GridWorld (SimulationGrid)
- [x] Write tests for agent visibility
- [ ] Verify hex grid movement animation works smoothly
- [ ] Connect to agent position updates via WebSocket (already connected via useSimulation hook)

### Metrics
- [x] `/metrics` endpoint already exists in FastAPI
- [x] Returns plain text Prometheus-style metrics
- [ ] Verify MetricsFooter fetches from endpoint correctly

## WebSocket Integration Status
- Demo endpoint exists and is configured in frontend hooks
- Need to verify all components properly subscribe to WebSocket events
- May need to add more demo message types for missing features

## Test Organization
- Frontend tests are in `web/__tests__/` directory  
- Component tests created: KnowledgeGraphView.test.tsx, SimulationGrid.test.tsx
- Metrics endpoint test created: tests/unit/test_metrics_endpoint.py
- Tests should be run from respective directories (web/ for frontend, root for backend)