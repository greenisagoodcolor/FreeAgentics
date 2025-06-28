/**
 * Mock components for testing
 * These re-export default exports as named exports for easier testing
 */

import AgentDashboardDefault from '@/components/agentdashboard';
import AgentCardDefault from '@/components/agentcard';

export const AgentDashboard = AgentDashboardDefault;
export const AgentCard = AgentCardDefault;

// Export other components that might have similar issues
import AgentListDefault from '@/components/AgentList';
import AgentBeliefVisualizerDefault from '@/components/agentbeliefvisualizer';

export const AgentList = AgentListDefault;
export const AgentBeliefVisualizer = AgentBeliefVisualizerDefault;
export { CharacterCreator } from '@/components/character-creator';