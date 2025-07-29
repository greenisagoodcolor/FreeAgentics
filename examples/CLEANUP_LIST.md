# Example Files Cleanup List

Based on CLAUDE.md's KISS/YAGNI principles and focusing on a working demo, here are the files to keep and remove:

## Files to KEEP ✅

1. **demo_full_pipeline.py** - Main integrated demo showing complete flow
2. **api_usage_examples.py** - Comprehensive API examples
3. **curl_examples.sh** - Shell command examples
4. **README.md** - Documentation
5. **TYPE_SAFETY_README.md** - Type safety documentation
6. **__init__.py** - Package marker

## Files to REMOVE ❌ (duplicate/obsolete)

1. **demo.py** - Console demo (replaced by web UI demo)
2. **demo_simple.py** - Duplicate of simple demo
3. **simple_demo.py** - Another duplicate
4. **active_inference_demo.py** - Likely duplicate functionality
5. **demo_agent_memory_lifecycle.py** - Specific feature demo (too granular)
6. **demo_efficient_data_structures.py** - Implementation detail
7. **demo_gmn.py** - Specific component demo
8. **demo_gmn_versioned_storage.py** - Implementation detail
9. **demo_llm_gmn_pipeline.py** - Partial pipeline (full pipeline exists)
10. **demo_matrix_pooling.py** - Implementation detail
11. **demo_memory_validation.py** - Testing/validation detail
12. **demo_results.json** - Output file
13. **error_handling_demo.py** - Implementation detail
14. **llm_error_handling_demo.py** - Implementation detail
15. **llm_integration_example.py** - Partial integration
16. **performance_benchmark_demo.py** - Benchmarking tool
17. **technical_spike.py** - Development artifact
18. **websocket_auth_demo.py** - Auth testing
19. **websocket_client.py** - Utility script
20. **websocket_secure_client.py** - Utility script
21. **test_examples.py** - This cleanup test file

## Rationale

Following CLAUDE.md section 11 (Demo Mode):
- Focus on the web UI demo experience
- Remove console-based demos that duplicate functionality
- Keep only comprehensive examples that show the full system
- Remove implementation details and test utilities

The goal is a clean examples directory that helps developers get started quickly without confusion.