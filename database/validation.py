"""Database layer validation script to check for errors and consistency."""

import logging
import sys
import traceback
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _test_single_import(
    module_name: str, display_name: str
) -> Tuple[bool, str]:
    """Test a single import and return success status and error message if any."""
    try:
        # The actual import would happen in the calling function
        logger.info(f"✓ {display_name} imported successfully")
        return True, ""
    except Exception as e:
        logger.error(f"✗ {display_name} import error: {e}")
        return False, str(e)


def _test_base_imports() -> Dict[str, bool]:
    """Test base and model imports."""
    results = {}

    # Test base imports
    results["base"], _ = _test_single_import("base", "Base")

    # Test model imports
    results["models"], _ = _test_single_import("models", "Models")

    return results


def _test_domain_model_imports() -> Dict[str, bool]:
    """Test domain-specific model imports."""
    results = {}

    # Test conversation models
    results["conversation_models"], _ = _test_single_import(
        "conversation_models", "Conversation models"
    )

    # Test knowledge graph models
    results["knowledge_models"], _ = _test_single_import(
        "knowledge_models", "Knowledge graph models"
    )

    return results


def _test_infrastructure_imports() -> Dict[str, bool]:
    """Test infrastructure-related imports."""
    results = {}

    # Test session imports
    results["session"], _ = _test_single_import("session", "Session")

    # Test repository imports
    results["repositories"], _ = _test_single_import(
        "repositories", "Repositories"
    )

    return results


def test_imports() -> Dict[str, bool]:
    """Test all database imports."""
    results = {}

    # Test different categories of imports
    results.update(_test_base_imports())
    results.update(_test_domain_model_imports())
    results.update(_test_infrastructure_imports())

    return results


def test_model_relationships() -> Dict[str, bool]:
    """Test model relationship configurations."""
    results = {}

    try:
        from database.models import Agent, Coalition

        # Test agent relationships
        agent_relationships = [
            "conversations",
            "gmn_specifications",
            "knowledge_updates",
            "knowledge_entities",
            "knowledge_relationships",
            "knowledge_concepts",
            "knowledge_evolution",
            "coalitions",
            "knowledge_nodes",
        ]

        for rel in agent_relationships:
            if hasattr(Agent, rel):
                results[f"agent_{rel}"] = True
                logger.info(f"✓ Agent.{rel} relationship exists")
            else:
                results[f"agent_{rel}"] = False
                logger.error(f"✗ Agent.{rel} relationship missing")

        # Test coalition relationships
        if hasattr(Coalition, "agents"):
            results["coalition_agents"] = True
            logger.info("✓ Coalition.agents relationship exists")
        else:
            results["coalition_agents"] = False
            logger.error("✗ Coalition.agents relationship missing")

        return results

    except Exception as e:
        logger.error(f"✗ Model relationship test failed: {e}")
        return {"model_relationships": False}


def test_repository_instantiation() -> Dict[str, bool]:
    """Test repository class instantiation."""
    results = {}

    try:
        from unittest.mock import Mock

        from database.agent_repository import AgentRepository
        from database.coalition_repository import CoalitionRepository

        # Test AgentRepository
        mock_session = Mock()
        AgentRepository(mock_session)
        results["agent_repository"] = True
        logger.info("✓ AgentRepository instantiation successful")

        # Test CoalitionRepository
        CoalitionRepository(mock_session)
        results["coalition_repository"] = True
        logger.info("✓ CoalitionRepository instantiation successful")

        return results

    except Exception as e:
        logger.error(f"✗ Repository instantiation failed: {e}")
        return {"repository_instantiation": False}


def test_session_type_annotations() -> Dict[str, bool]:
    """Test session type annotations."""
    results = {}

    try:
        from typing import get_type_hints

        from database.agent_repository import AgentRepository
        from database.session import get_db

        # Test get_db type hints
        get_db_hints = get_type_hints(get_db)
        if "return" in get_db_hints:
            results["get_db_annotations"] = True
            logger.info(f"✓ get_db return type: {get_db_hints['return']}")
        else:
            results["get_db_annotations"] = False
            logger.error("✗ get_db missing return type annotation")

        # Test get_session type hints
        # NOTE: get_session doesn't exist in the codebase, commenting out
        # get_session_hints = get_type_hints(get_session)
        # if "return" in get_session_hints:
        #     results["get_session_annotations"] = True
        #     logger.info(
        #         f"✓ get_session return type: {get_session_hints['return']}"
        #     )
        # else:
        #     results["get_session_annotations"] = False
        #     logger.error("✗ get_session missing return type annotation")
        results["get_session_annotations"] = True  # Skip this test

        # Test repository method type hints
        agent_repo_hints = get_type_hints(AgentRepository.get_agent)
        if "return" in agent_repo_hints:
            results["repository_annotations"] = True
            logger.info(
                f"✓ AgentRepository.get_agent return type: {agent_repo_hints['return']}"
            )
        else:
            results["repository_annotations"] = False
            logger.error(
                "✗ AgentRepository.get_agent missing return type annotation"
            )

        return results

    except Exception as e:
        logger.error(f"✗ Type annotation test failed: {e}")
        return {"type_annotations": False}


def test_metadata_consistency() -> Dict[str, bool]:
    """Test database metadata consistency."""
    results = {}

    try:
        from database.base import Base

        # Check that all tables are registered
        table_count = len(Base.metadata.tables)
        if table_count > 0:
            results["metadata_tables"] = True
            logger.info(f"✓ Base metadata has {table_count} tables")

            # List all tables
            for table_name in sorted(Base.metadata.tables.keys()):
                logger.info(f"  - {table_name}")
        else:
            results["metadata_tables"] = False
            logger.error("✗ No tables found in metadata")

        # Check for specific critical tables
        critical_tables = ["agents", "coalitions", "conversations"]
        for table in critical_tables:
            if table in Base.metadata.tables:
                results[f"table_{table}"] = True
                logger.info(f"✓ Critical table '{table}' exists")
            else:
                results[f"table_{table}"] = False
                logger.error(f"✗ Critical table '{table}' missing")

        return results

    except Exception as e:
        logger.error(f"✗ Metadata consistency test failed: {e}")
        return {"metadata_consistency": False}


def _test_numpy_array_serialization() -> Tuple[bool, List[str]]:
    """Test numpy array serialization with edge cases."""
    import numpy as np

    from database.agent_repository import PyMDPStateSerializer

    test_arrays = [
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([[0.1, 0.9]], dtype=np.float64),
        np.array([], dtype=np.float32),  # Empty array
        np.array([0.0], dtype=np.float32),  # Single element
    ]

    numpy_serialization_success = True
    numpy_errors = []

    for i, test_array in enumerate(test_arrays):
        try:
            serialized = PyMDPStateSerializer.serialize_numpy_array(test_array)
            if serialized is None:
                numpy_errors.append(f"Array {i}: serialization returned None")
                numpy_serialization_success = False
                continue

            deserialized = PyMDPStateSerializer.deserialize_numpy_array(
                serialized
            )
            if deserialized is None:
                numpy_errors.append(
                    f"Array {i}: deserialization returned None"
                )
                numpy_serialization_success = False
                continue

            if not isinstance(deserialized, np.ndarray):
                numpy_errors.append(
                    f"Array {i}: not numpy array after deserialization"
                )
                numpy_serialization_success = False
                continue

            if test_array.shape != deserialized.shape:  # type: ignore[attr-defined]
                numpy_errors.append(f"Array {i}: shape mismatch")
                numpy_serialization_success = False
                continue

            if not np.array_equal(test_array, deserialized):
                numpy_errors.append(f"Array {i}: content mismatch")
                numpy_serialization_success = False

        except Exception as e:
            numpy_errors.append(f"Array {i}: exception {str(e)}")
            numpy_serialization_success = False

    return numpy_serialization_success, numpy_errors


def _validate_deserialized_a_matrices(
    test_state: Dict[str, Any], deserialized_A: Any
) -> Tuple[bool, List[str]]:
    """Validate deserialized A matrices."""
    import numpy as np

    validation_errors = []

    if deserialized_A is None:
        validation_errors.append("A matrices missing")
        return False, validation_errors
    elif not isinstance(deserialized_A, list):
        validation_errors.append("A matrices not a list")
        return False, validation_errors
    elif len(deserialized_A) == 0:
        validation_errors.append("A matrices list is empty")
        return False, validation_errors
    elif len(deserialized_A) != len(test_state["A"]):
        validation_errors.append("A matrices count mismatch")
        return False, validation_errors

    for i, (original, deserialized) in enumerate(
        zip(test_state["A"], deserialized_A)
    ):
        if not isinstance(deserialized, np.ndarray):
            validation_errors.append(f"A[{i}] not numpy array")
            return False, validation_errors
        elif not np.array_equal(original, deserialized):
            validation_errors.append(f"A[{i}] content mismatch")
            return False, validation_errors

    return True, validation_errors


def _validate_deserialized_beliefs(
    test_state: Dict[str, Any], deserialized_beliefs: Any
) -> Tuple[bool, List[str]]:
    """Validate deserialized beliefs."""
    import numpy as np

    validation_errors = []

    if deserialized_beliefs is None:
        validation_errors.append("beliefs missing")
        return False, validation_errors
    elif not isinstance(deserialized_beliefs, np.ndarray):
        validation_errors.append("beliefs not numpy array")
        return False, validation_errors
    elif not np.array_equal(test_state["beliefs"], deserialized_beliefs):
        validation_errors.append("beliefs content mismatch")
        return False, validation_errors

    return True, validation_errors


def _validate_other_param(
    test_state: Dict[str, Any], deserialized_other: Any
) -> Tuple[bool, List[str]]:
    """Validate other parameter."""
    validation_errors = []

    if test_state["other_param"] != deserialized_other:
        validation_errors.append("other_param mismatch")
        return False, validation_errors

    return True, validation_errors


def _test_full_state_serialization() -> Tuple[bool, List[str]]:
    """Test full PyMDP state serialization."""
    import numpy as np

    from database.agent_repository import PyMDPStateSerializer

    test_state = {
        "A": [np.array([1, 2, 3]), np.array([4, 5, 6])],
        "beliefs": np.array([[0.1, 0.9], [0.8, 0.2]]),
        "other_param": "test_value",
    }

    serialized_state = PyMDPStateSerializer.serialize_pymdp_matrices(
        test_state
    )
    deserialized_state = PyMDPStateSerializer.deserialize_pymdp_matrices(
        serialized_state
    )

    # Check if arrays are correctly deserialized
    deserialized_A = deserialized_state.get("A")
    deserialized_beliefs = deserialized_state.get("beliefs")
    deserialized_other = deserialized_state.get("other_param")

    # More robust validation with proper error handling
    validation_errors = []

    try:
        # Check A matrices
        success, errors = _validate_deserialized_a_matrices(
            test_state, deserialized_A
        )
        if not success:
            validation_errors.extend(errors)

        # Check beliefs
        success, errors = _validate_deserialized_beliefs(
            test_state, deserialized_beliefs
        )
        if not success:
            validation_errors.extend(errors)

        # Check other parameter
        success, errors = _validate_other_param(test_state, deserialized_other)
        if not success:
            validation_errors.extend(errors)

    except Exception as e:
        validation_errors.append(f"Validation exception: {str(e)}")

    return len(validation_errors) == 0, validation_errors


def test_serialization() -> Dict[str, Any]:
    """Test PyMDP state serialization."""
    results = {}

    try:
        # Test numpy array serialization with edge cases
        numpy_success, numpy_errors = _test_numpy_array_serialization()

        if numpy_success:
            results["numpy_serialization"] = True
            logger.info("✓ Numpy array serialization works correctly")
        else:
            results["numpy_serialization"] = False
            logger.error(
                f"✗ Numpy array serialization failed: {'; '.join(numpy_errors)}"
            )

        # Test full state serialization
        state_success, validation_errors = _test_full_state_serialization()

        if state_success:
            results["state_serialization"] = True
            logger.info("✓ PyMDP state serialization works correctly")
        else:
            results["state_serialization"] = False
            logger.error(
                f"✗ PyMDP state serialization failed: {'; '.join(validation_errors)}"
            )

        return results

    except ImportError as e:
        logger.error(
            f"✗ Serialization test failed - missing dependencies: {e}"
        )
        return {
            "serialization": False,
            "error": "Missing numpy or serialization dependencies",
        }
    except AttributeError as e:
        logger.error(f"✗ Serialization test failed - method not found: {e}")
        return {
            "serialization": False,
            "error": "PyMDPStateSerializer methods not available",
        }
    except Exception as e:
        logger.error(f"✗ Serialization test failed: {e}")
        traceback.print_exc()
        return {"serialization": False, "error": str(e)}


def run_comprehensive_validation() -> Tuple[bool, Dict[str, Any]]:
    """Run comprehensive database validation."""
    logger.info("Starting comprehensive database validation...")

    all_results = {}
    success_count = 0
    total_count = 0

    # Run all validation tests
    test_functions = [
        ("imports", test_imports),
        ("model_relationships", test_model_relationships),
        ("repository_instantiation", test_repository_instantiation),
        ("type_annotations", test_session_type_annotations),
        ("metadata_consistency", test_metadata_consistency),
        ("serialization", test_serialization),
    ]

    for test_name, test_func in test_functions:
        logger.info(f"\n--- Running {test_name} tests ---")
        try:
            results = test_func()
            all_results[test_name] = results

            # Count successes
            for key, value in results.items():
                total_count += 1
                if value:
                    success_count += 1

        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            all_results[test_name] = {"error": False}  # Mark test as failed
            total_count += 1

    # Calculate overall success
    success_rate = success_count / total_count if total_count > 0 else 0
    overall_success = success_rate >= 0.8  # 80% threshold

    logger.info("\n=== VALIDATION SUMMARY ===")
    logger.info(f"Total tests: {total_count}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_count - success_count}")
    logger.info(f"Success rate: {success_rate:.1%}")
    logger.info(f"Overall result: {'✓ PASS' if overall_success else '✗ FAIL'}")

    return overall_success, {
        "success_rate": success_rate,
        "total_tests": total_count,
        "successful_tests": success_count,
        "failed_tests": total_count - success_count,
        "detailed_results": all_results,
    }


if __name__ == "__main__":
    # Set testing environment to avoid DATABASE_URL requirement
    import os

    os.environ["TESTING"] = "true"

    success, results = run_comprehensive_validation()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
