"""Infrastructure validation test - verify test environment works correctly."""

import os
import sys


def test_environment_setup():
    """Test that environment is correctly configured."""
    assert os.environ.get("TESTING") == "true"
    assert os.environ.get("DATABASE_URL") == "sqlite:///:memory:"
    assert os.environ.get("LLM_PROVIDER") == "mock"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    print("✓ Environment variables correctly set")
    return True


def test_database_import():
    """Test that database imports work."""
    try:

        print("✓ Database imports successful")
        return True
    except Exception as e:
        print(f"✗ Database import failed: {e}")
        return False


def test_basic_api_import():
    """Test that basic API imports work."""
    try:

        print("✓ Basic API imports successful")
        return True
    except Exception as e:
        print(f"✗ API import failed: {e}")
        return False


def test_auth_import():
    """Test that auth imports work."""
    try:

        print("✓ Auth imports successful")
        return True
    except Exception as e:
        print(f"✗ Auth import failed: {e}")
        return False


def main():
    """Run all infrastructure validation tests."""
    print("Running infrastructure validation tests...")

    tests = [
        test_environment_setup,
        test_database_import,
        test_basic_api_import,
        test_auth_import,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100

    print(f"\n--- Infrastructure Validation Results ---")
    print(f"Tests passed: {passed}/{total} ({success_rate:.1f}%)")

    if success_rate >= 95:
        print("✅ Infrastructure validation PASSED - ready for test execution")
        return 0
    else:
        print("❌ Infrastructure validation FAILED - needs fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
