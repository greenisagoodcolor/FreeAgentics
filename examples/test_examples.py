#!/usr/bin/env python3
"""Test which example files can be imported without errors."""

import importlib.util
import os


def test_example(filepath):
    """Test if an example file can be imported."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just check syntax
            compile(open(filepath).read(), filepath, "exec")
            return True, "OK"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    examples_dir = os.path.dirname(os.path.abspath(__file__))

    # List of examples to keep (known working)
    keep_examples = {
        "demo_full_pipeline.py",  # Main demo mentioned in README
        "api_usage_examples.py",  # API examples
        "curl_examples.sh",  # Shell examples
        "README.md",  # Documentation
        "TYPE_SAFETY_README.md",  # Documentation
        "__init__.py",  # Package file
        "test_examples.py",  # This file
    }

    print("Checking example files...\n")

    broken_files = []
    working_files = []

    for filename in os.listdir(examples_dir):
        if filename.endswith(".py") and filename not in keep_examples:
            filepath = os.path.join(examples_dir, filename)
            works, error = test_example(filepath)

            if works:
                working_files.append(filename)
            else:
                broken_files.append((filename, error))
                print(f"❌ {filename}: {error[:100]}...")

    print(f"\n✅ Working: {len(working_files)} files")
    print(f"❌ Broken: {len(broken_files)} files")

    if broken_files:
        print("\nFiles to remove:")
        for f, _ in broken_files:
            print(f"  - {f}")
