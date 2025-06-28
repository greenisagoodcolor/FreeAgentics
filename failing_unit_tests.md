# Failing Unit Tests Summary

## Overview
Based on the pytest runs, we have identified the following failing tests in the unit test suite. Most other test files (test_memory.py, test_perception.py, test_hierarchical_inference.py, test_belief_state.py, test_generative_model.py, test_gnn_layers.py, etc.) are passing successfully.

## tests/unit/test_active_inference.py

### Import Issues (Fixed)
- ✅ Fixed imports by updating to use actual class names from active_inference.py
- ✅ Removed references to non-existent `FreeEnergyMinimization` class
- ✅ Removed references to standalone functions like `compute_free_energy`

### Current Failures/Errors

#### 1. TestInferenceConfig::test_config_validation (FAILED)
- Issue: InferenceConfig doesn't validate negative values as expected
- The test expects ValueError for negative num_iterations, learning_rate, and convergence_threshold

#### 2. ModelDimensions TypeError (Multiple tests affected)
- Error: `ModelDimensions.__init__() got an unexpected keyword argument 'time_steps'`
- Affected tests:
  - TestVariationalMessagePassing (all test methods)
  - TestBeliefPropagation::test_message_passing
  - TestGradientDescentInference (all test methods)
  - TestIntegration (all test methods)
  - Most tests using fixtures that create GenerativeModel with ModelDimensions

#### 3. Abstract Class Instantiation Errors
- TestBeliefPropagation::test_loopy_belief_propagation
  - Error: Can't instantiate abstract class CyclicModel
- TestPerformanceOptimization::test_sparse_computations
  - Error: Can't instantiate abstract class SparseModel
- TestVariationalMessagePassing::test_continuous_observation_inference
  - Error: Can't instantiate abstract class ContinuousGenerativeModel
- TestHierarchicalInference tests
  - Error: Can't instantiate abstract class HierarchicalGenerativeModel

#### 4. Missing Methods/Functions
- Tests are looking for standalone functions that are actually methods:
  - `compute_free_energy`
  - `compute_expected_free_energy`
  - `compute_variational_free_energy`

## Root Causes

1. **ModelDimensions API Change**: The `time_steps` parameter was likely removed or renamed
2. **Abstract Classes**: Test classes inheriting from GenerativeModel need to implement abstract methods
3. **API Mismatch**: Tests expect standalone functions but the implementation has them as methods
4. **Missing Validation**: InferenceConfig doesn't validate input parameters

## Recommended Fixes

1. Update fixture files to match current ModelDimensions constructor signature
2. Implement abstract methods in test model classes
3. Either create standalone functions or update tests to use methods
4. Add validation to InferenceConfig dataclass

## Summary Statistics

- **Total test files checked**: ~35 files in tests/unit/
- **Files with failures**: Primarily test_active_inference.py
- **Files passing**: test_memory.py, test_perception.py, test_hierarchical_inference.py, test_belief_state.py, test_belief_update.py, test_generative_model.py, test_gnn_layers.py, test_gnn_parser.py, test_batch_processor.py, test_agent_data_model.py, and many others
- **Main issue**: test_active_inference.py has ~20 failing tests due to API mismatches

The good news is that most of the unit test suite is passing. The failures are concentrated in test_active_inference.py and are mostly due to:
1. ModelDimensions constructor signature changes
2. Missing abstract method implementations in test classes
3. Differences between expected API (standalone functions) and actual API (class methods)