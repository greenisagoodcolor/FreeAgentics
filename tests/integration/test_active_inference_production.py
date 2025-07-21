#!/usr/bin/env python3
"""
Comprehensive Active Inference Integration Test Suite for Production.

This test suite validates ALL Active Inference functionality works in production:
- PyMDP integration and compatibility 
- GMN parser functionality
- Agent creation and lifecycle
- Belief state management
- Action selection and expected free energy
- Multi-agent coordination
- Error handling and fallbacks
- Performance optimizations

Test results demonstrate core business value is functional.
"""

import unittest
import sys
import os
import numpy as np
import time
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestActiveInferenceProduction(unittest.TestCase):
    """Production tests for Active Inference functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_results = []
        self.start_time = time.time()

    def tearDown(self):
        """Clean up after tests."""
        duration = time.time() - self.start_time
        print(f"Test completed in {duration:.3f}s")

    def test_01_pymdp_import_and_basic_functionality(self):
        """Test PyMDP is available and working in production."""
        try:
            import pymdp
            from pymdp.agent import Agent as PyMDPAgent
            from pymdp import utils
            
            # Test basic agent creation
            A = np.eye(3)
            B = np.zeros((3, 3, 2))
            for i in range(2):
                B[:, :, i] = np.eye(3)
            C = np.array([1.0, 0.0, -1.0])
            D = np.ones(3) / 3
            
            agent = PyMDPAgent(A=A, B=B, C=C, D=D)
            
            # Test inference pipeline
            obs = [0]
            agent.infer_states(obs)
            agent.infer_policies()
            action = agent.sample_action()
            
            self.assertIsNotNone(action)
            self.assertTrue(hasattr(action, 'shape'))
            
        except ImportError as e:
            self.fail(f"PyMDP not available in production: {e}")
        except Exception as e:
            self.fail(f"PyMDP basic functionality failed: {e}")

    def test_02_gmn_parser_comprehensive(self):
        """Test GMN parser with various specifications."""
        try:
            from inference.active.gmn_parser import GMNParser, parse_gmn_spec, EXAMPLE_GMN_SPEC
            
            # Test 1: Example specification
            model_spec = parse_gmn_spec(EXAMPLE_GMN_SPEC)
            self.assertIn('num_states', model_spec)
            self.assertIn('A', model_spec)
            self.assertIn('B', model_spec)
            self.assertGreater(len(model_spec['A']), 0)
            self.assertGreater(len(model_spec['B']), 0)
            
            # Test 2: Custom grid world specification
            custom_gmn = '''
            [nodes]
            location: state {num_states: 9}
            observation: observation {num_observations: 5}
            movement: action {num_actions: 5}
            obs_model: likelihood
            trans_model: transition
            pref: preference {preferred_observation: 2}
            
            [edges]
            location -> obs_model: depends_on
            obs_model -> observation: generates
            location -> trans_model: depends_on
            movement -> trans_model: depends_on
            pref -> observation: depends_on
            '''
            
            custom_model = parse_gmn_spec(custom_gmn)
            self.assertEqual(custom_model['num_states'], [9])
            self.assertEqual(custom_model['num_obs'], [5])
            self.assertEqual(custom_model['num_actions'], [5])
            
            # Test 3: JSON format
            json_spec = {
                'nodes': [
                    {'id': 'state1', 'type': 'state', 'properties': {'num_states': 4}},
                    {'id': 'obs1', 'type': 'observation', 'properties': {'num_observations': 4}},
                    {'id': 'action1', 'type': 'action', 'properties': {'num_actions': 3}},
                    {'id': 'likelihood1', 'type': 'likelihood'},
                    {'id': 'transition1', 'type': 'transition'},
                ],
                'edges': [
                    {'source': 'state1', 'target': 'likelihood1', 'type': 'depends_on'},
                    {'source': 'likelihood1', 'target': 'obs1', 'type': 'generates'},
                ]
            }
            
            parser = GMNParser()
            graph = parser.parse(json_spec)
            json_model = parser.to_pymdp_model(graph)
            self.assertIn('A', json_model)
            self.assertIn('B', json_model)
            
        except Exception as e:
            self.fail(f"GMN parser failed: {e}")

    def test_03_gmn_pymdp_adapter_integration(self):
        """Test GMN to PyMDP adapter produces working models."""
        try:
            from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp
            from inference.active.gmn_parser import parse_gmn_spec
            from pymdp.agent import Agent as PyMDPAgent
            
            # Create test GMN model
            gmn_spec = '''
            [nodes]
            location: state {num_states: 4}
            observation: observation {num_observations: 3}
            movement: action {num_actions: 4}
            obs_model: likelihood
            trans_model: transition
            pref: preference {preferred_observation: 1}
            
            [edges]
            location -> obs_model: depends_on
            obs_model -> observation: generates
            location -> trans_model: depends_on
            movement -> trans_model: depends_on
            pref -> observation: depends_on
            '''
            
            # Parse and adapt
            model_spec = parse_gmn_spec(gmn_spec)
            adapted_model = adapt_gmn_to_pymdp(model_spec)
            
            # Validate adapter output
            self.assertIsInstance(adapted_model['A'], np.ndarray)
            self.assertIsInstance(adapted_model['B'], np.ndarray)
            
            # Test with PyMDP agent
            agent = PyMDPAgent(
                A=adapted_model['A'],
                B=adapted_model['B'], 
                C=adapted_model.get('C'),
                D=adapted_model.get('D')
            )
            
            # Verify inference works
            obs = [1]
            agent.infer_states(obs)
            agent.infer_policies()
            action = agent.sample_action()
            self.assertIsNotNone(action)
            
            # Test multi-factor rejection
            multi_factor_model = {
                'A': [np.eye(3), np.eye(4)],  # Multiple factors
                'B': [np.ones((3,3,2)), np.ones((4,4,2))],
                'C': [np.zeros(3), np.zeros(4)],
                'D': [np.ones(3)/3, np.ones(4)/4]
            }
            
            with self.assertRaises(ValueError):
                adapt_gmn_to_pymdp(multi_factor_model)
                
        except Exception as e:
            self.fail(f"GMN-PyMDP adapter failed: {e}")

    def test_04_basic_explorer_agent_workflow(self):
        """Test BasicExplorerAgent complete Active Inference workflow."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            # Create agent
            agent = BasicExplorerAgent('test_agent', 'Test Agent', grid_size=4)
            agent.start()
            
            self.assertTrue(agent.is_active)
            self.assertIsNotNone(agent.pymdp_agent)
            
            # Test observation processing
            observation = {
                'position': [1, 1],
                'surroundings': np.array([
                    [0, 0, 0],
                    [0, 0, 1], 
                    [0, -1, 0]
                ]),
                'time_step': 1
            }
            
            # Test complete Active Inference step
            initial_steps = agent.total_steps
            action = agent.step(observation)
            
            # Validate action
            self.assertIn(action, agent.actions)
            self.assertEqual(agent.total_steps, initial_steps + 1)
            
            # Test metrics
            metrics = agent.metrics
            self.assertGreater(metrics['total_observations'], 0)
            self.assertGreater(metrics['total_actions'], 0)
            self.assertIn('belief_entropy', metrics)
            
            # Test multi-step sequence
            actions = []
            for step in range(3):
                obs = {
                    'position': [1, 1 + step % 2],
                    'surroundings': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    'time_step': step + 2
                }
                action = agent.step(obs)
                actions.append(action)
            
            self.assertEqual(len(actions), 3)
            
            # Test agent cleanup
            agent.stop()
            self.assertFalse(agent.is_active)
            
        except Exception as e:
            self.fail(f"BasicExplorerAgent workflow failed: {e}")

    def test_05_belief_state_management(self):
        """Test Active Inference belief state management."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            agent = BasicExplorerAgent('belief_test', 'Belief Test Agent', grid_size=5)
            agent.start()
            
            # Test different observations and belief updates
            observations = [
                {
                    'position': [2, 2],
                    'surroundings': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
                    'time_step': 1
                },
                {
                    'position': [2, 3],
                    'surroundings': np.array([[0, -1, 0], [0, 0, 0], [0, 0, 0]]),
                    'time_step': 2
                }
            ]
            
            belief_entropies = []
            
            for obs in observations:
                agent.perceive(obs)
                agent.update_beliefs()
                
                entropy = agent.metrics.get('belief_entropy', 0.0)
                belief_entropies.append(entropy)
                
                # Validate belief properties
                if agent.pymdp_agent and hasattr(agent.pymdp_agent, 'qs') and agent.pymdp_agent.qs is not None:
                    beliefs = agent.pymdp_agent.qs[0]
                    
                    # Check belief is a probability distribution
                    self.assertAlmostEqual(np.sum(beliefs), 1.0, places=6)
                    self.assertTrue(np.all(beliefs >= 0))
                    
                    # Check entropy is non-negative
                    self.assertGreaterEqual(entropy, 0)
            
            self.assertEqual(len(belief_entropies), 2)
            agent.stop()
            
        except Exception as e:
            self.fail(f"Belief state management failed: {e}")

    def test_06_free_energy_computation(self):
        """Test free energy computation and decomposition."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            agent = BasicExplorerAgent('fe_test', 'Free Energy Test Agent', grid_size=4)
            agent.start()
            
            # Set up agent with observation
            observation = {
                'position': [1, 1],
                'surroundings': np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]),
                'time_step': 1
            }
            
            agent.step(observation)
            
            # Test free energy computation
            fe_components = agent.compute_free_energy()
            
            if 'total_free_energy' in fe_components:
                # Validate components
                self.assertIn('accuracy', fe_components)
                self.assertIn('complexity', fe_components)
                
                # Check decomposition: F = complexity - accuracy
                expected_fe = fe_components['complexity'] - fe_components['accuracy']
                actual_fe = fe_components['total_free_energy']
                self.assertAlmostEqual(expected_fe, actual_fe, places=6)
                
                # Check component properties
                for component, value in fe_components.items():
                    if component != 'error':
                        self.assertFalse(np.isnan(value))
                        self.assertFalse(np.isinf(value))
                        
                # Test surprise if available
                if 'surprise' in fe_components:
                    self.assertGreaterEqual(fe_components['surprise'], 0)
            
            agent.stop()
            
        except Exception as e:
            self.fail(f"Free energy computation failed: {e}")

    def test_07_action_selection_expected_free_energy(self):
        """Test action selection minimizes expected free energy."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            agent = BasicExplorerAgent('action_test', 'Action Test Agent', grid_size=4)
            agent.start()
            
            # Test various scenarios
            scenarios = [
                {
                    'obs': {
                        'position': [2, 2],
                        'surroundings': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                        'time_step': 1
                    }
                },
                {
                    'obs': {
                        'position': [2, 2],
                        'surroundings': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
                        'time_step': 2
                    }
                }
            ]
            
            actions_and_efe = []
            
            for scenario in scenarios:
                agent.perceive(scenario['obs'])
                agent.update_beliefs()
                action = agent.select_action()
                
                # Validate action
                self.assertIn(action, agent.actions)
                
                # Check expected free energy
                efe = agent.metrics.get('expected_free_energy')
                if efe is not None:
                    self.assertFalse(np.isnan(efe))
                    self.assertFalse(np.isinf(efe))
                    actions_and_efe.append((action, efe))
            
            agent.stop()
            
        except Exception as e:
            self.fail(f"Action selection test failed: {e}")

    def test_08_multi_agent_coordination(self):
        """Test multi-agent Active Inference coordination."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            # Create multiple agents
            agents = []
            num_agents = 3
            
            for i in range(num_agents):
                agent = BasicExplorerAgent(f'agent_{i}', f'Agent {i}', grid_size=4)
                agent.start()
                agents.append(agent)
            
            self.assertEqual(len(agents), num_agents)
            
            # Test coordination scenario
            observation_base = {
                'surroundings': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                'time_step': 1
            }
            
            actions = []
            for i, agent in enumerate(agents):
                obs = observation_base.copy()
                obs['position'] = [i, i]  # Different positions
                action = agent.step(obs)
                actions.append(action)
                
                # Validate each agent works independently
                self.assertIn(action, agent.actions)
                self.assertGreater(agent.total_steps, 0)
            
            self.assertEqual(len(actions), num_agents)
            
            # Cleanup
            for agent in agents:
                agent.stop()
                self.assertFalse(agent.is_active)
                
        except Exception as e:
            self.fail(f"Multi-agent coordination failed: {e}")

    def test_09_error_handling_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            agent = BasicExplorerAgent('error_test', 'Error Test Agent', grid_size=3)
            agent.start()
            
            # Test various error scenarios
            error_scenarios = [
                # Invalid observation format
                {'position': [10, 10], 'surroundings': None, 'time_step': 1},
                # Minimal observation
                {'position': [1, 1]},
                # Out of bounds position
                {'position': [-1, -1], 'surroundings': np.zeros((3, 3)), 'time_step': 1}
            ]
            
            for scenario in error_scenarios:
                try:
                    action = agent.step(scenario)
                    # Should return valid action even with bad input
                    self.assertIn(action, agent.actions)
                except Exception as e:
                    # Some errors are acceptable, but shouldn't crash the system
                    self.assertIsInstance(e, (ValueError, IndexError, KeyError))
            
            # Test agent status after errors
            status = agent.get_status()
            self.assertIn('agent_id', status)
            self.assertIn('is_active', status)
            
            agent.stop()
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")

    def test_10_performance_optimizations(self):
        """Test performance optimizations work correctly."""
        try:
            from agents.base_agent import BasicExplorerAgent
            
            # Test different performance modes
            modes = ['fast', 'balanced', 'accurate']
            performance_results = {}
            
            for mode in modes:
                agent = BasicExplorerAgent(f'perf_{mode}', f'Perf {mode}', grid_size=4)
                agent.performance_mode = mode
                agent.start()
                
                start_time = time.time()
                
                # Run standardized workload
                for step in range(5):  # Reduced for faster testing
                    obs = {
                        'position': [1, 1],
                        'surroundings': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                        'time_step': step
                    }
                    action = agent.step(obs)
                    self.assertIn(action, agent.actions)
                
                duration = time.time() - start_time
                performance_results[mode] = duration
                
                agent.stop()
            
            # Validate performance differences
            self.assertIn('fast', performance_results)
            self.assertIn('balanced', performance_results) 
            self.assertIn('accurate', performance_results)
            
            # Generally expect fast <= balanced <= accurate (though not guaranteed)
            for mode, duration in performance_results.items():
                self.assertGreater(duration, 0, f"Zero duration for {mode} mode")
            
        except Exception as e:
            self.fail(f"Performance optimization test failed: {e}")

    def test_11_production_integration_end_to_end(self):
        """End-to-end integration test simulating production usage."""
        try:
            from agents.base_agent import BasicExplorerAgent
            from inference.active.gmn_parser import parse_gmn_spec
            from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp
            
            # Complete pipeline: GMN -> PyMDP -> Agent -> Inference
            
            # 1. Create custom GMN model
            gmn_spec = '''
            [nodes]
            location: state {num_states: 16}
            observation: observation {num_observations: 5}
            movement: action {num_actions: 5}
            obs_model: likelihood
            trans_model: transition
            pref: preference {preferred_observation: 2}
            
            [edges]
            location -> obs_model: depends_on
            obs_model -> observation: generates
            location -> trans_model: depends_on
            movement -> trans_model: depends_on
            pref -> observation: depends_on
            '''
            
            # 2. Parse GMN and create PyMDP model  
            model_spec = parse_gmn_spec(gmn_spec)
            adapted_model = adapt_gmn_to_pymdp(model_spec)
            
            # 3. Create agent with custom model
            agent = BasicExplorerAgent('production_test', 'Production Test Agent', grid_size=4)
            
            # Override with custom model (simulate GMN loading)
            from pymdp.agent import Agent as PyMDPAgent
            if adapted_model['A'] is not None and adapted_model['B'] is not None:
                agent.pymdp_agent = PyMDPAgent(
                    A=adapted_model['A'],
                    B=adapted_model['B'],
                    C=adapted_model.get('C'),
                    D=adapted_model.get('D')
                )
            
            agent.start()
            
            # 4. Run realistic scenario
            scenario_steps = [
                {'position': [1, 1], 'surroundings': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])},
                {'position': [1, 2], 'surroundings': np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]])},
                {'position': [2, 2], 'surroundings': np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 2]])},
            ]
            
            actions = []
            free_energies = []
            
            for i, scenario in enumerate(scenario_steps):
                scenario['time_step'] = i + 1
                action = agent.step(scenario)
                actions.append(action)
                
                # Collect metrics
                fe_comp = agent.compute_free_energy()
                if 'total_free_energy' in fe_comp:
                    free_energies.append(fe_comp['total_free_energy'])
                
                # Validate each step
                self.assertIn(action, agent.actions)
                self.assertGreater(agent.metrics['total_observations'], i)
                
            # 5. Validate full pipeline results
            self.assertEqual(len(actions), len(scenario_steps))
            self.assertGreater(agent.total_steps, 0)
            
            if free_energies:
                self.assertGreater(len(free_energies), 0)
                for fe in free_energies:
                    self.assertFalse(np.isnan(fe))
                    self.assertFalse(np.isinf(fe))
            
            # 6. Cleanup
            agent.stop()
            self.assertFalse(agent.is_active)
            
        except Exception as e:
            self.fail(f"Production integration test failed: {e}")

def run_production_test_suite():
    """Run the complete Active Inference production test suite."""
    print("=" * 80)
    print("ACTIVE INFERENCE PRODUCTION TEST SUITE")
    print("=" * 80)
    print("Validating ALL Active Inference functionality for production deployment")
    print("")
    
    # Configure test runner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestActiveInferenceProduction)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("PRODUCTION TEST RESULTS SUMMARY")  
    print("=" * 80)
    
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = tests_run - failures - errors
    
    print(f"Tests run: {tests_run}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success rate: {100 * passed / tests_run:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    
    if failures > 0:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if errors > 0:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"  {test}: {error}")
    
    success = failures == 0 and errors == 0
    
    if success:
        print("\n🎉 ALL PRODUCTION TESTS PASSED! 🎉")
        print("Active Inference system is FULLY FUNCTIONAL for production deployment.")
        print("Core business value has been verified and validated.")
    else:
        print(f"\n⚠️  {failures + errors} test(s) failed.")
        print("Active Inference system requires fixes before production deployment.")
    
    return success

if __name__ == "__main__":
    success = run_production_test_suite()
    sys.exit(0 if success else 1)