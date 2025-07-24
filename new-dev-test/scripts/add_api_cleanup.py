#!/usr/bin/env python3
"""
Add API Cleanup Endpoint.
As specified in subtask 15.1
"""

import sys
from pathlib import Path


def add_cleanup_endpoint():
    """Add cleanup endpoint to API system module."""

    # Check if system.py exists
    system_path = Path("api/v1/system.py")
    if not system_path.exists():
        print(f"❌ {system_path} not found")
        return False

    try:
        with open(system_path, "r") as f:
            content = f.read()

        # Check if cleanup endpoint already exists
        if "/cleanup" in content:
            print("✅ Cleanup endpoint already exists")
            return True

        # Add cleanup endpoint
        cleanup_code = '''

@router.post("/cleanup")
async def cleanup_resources():
    """
    Cleanup endpoint for container resources
    Removes temporary files, clears caches, resets connections
    """
    try:
        import tempfile
        import shutil
        import gc
        import psutil

        cleanup_results = {
            "status": "success",
            "cleaned": []
        }

        # Clean temporary files
        temp_dir = Path(tempfile.gettempdir())
        temp_files_cleaned = 0

        try:
            for temp_file in temp_dir.glob("tmp*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    temp_files_cleaned += 1
            cleanup_results["cleaned"].append(f"Removed {temp_files_cleaned} temporary files")
        except Exception as e:
            cleanup_results["cleaned"].append(f"Temporary file cleanup warning: {str(e)}")

        # Force garbage collection
        collected = gc.collect()
        cleanup_results["cleaned"].append(f"Garbage collected {collected} objects")

        # Clear Python caches
        try:
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            cleanup_results["cleaned"].append("Cleared Python type cache")
        except Exception as e:
            cleanup_results["cleaned"].append(f"Cache cleanup warning: {str(e)}")

        # Memory info
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cleanup_results["memory_usage_mb"] = memory_info.rss / 1024 / 1024
        except Exception:
            pass

        return cleanup_results

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@router.get("/cleanup/status")
async def cleanup_status():
    """
    Get cleanup status and resource usage
    """
    try:
        import psutil
        import tempfile

        status = {
            "status": "ok",
            "resources": {}
        }

        # Memory usage
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            status["resources"]["memory_mb"] = memory_info.rss / 1024 / 1024
            status["resources"]["memory_percent"] = process.memory_percent()
        except Exception:
            pass

        # Temporary files count
        try:
            temp_dir = Path(tempfile.gettempdir())
            temp_files = len(list(temp_dir.glob("tmp*")))
            status["resources"]["temp_files"] = temp_files
        except Exception:
            pass

        # CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            status["resources"]["cpu_percent"] = cpu_percent
        except Exception:
            pass

        return status

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
'''

        # Add the cleanup endpoints before the last line
        lines = content.split("\n")

        # Find the last meaningful line (not just whitespace)
        insert_index = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                insert_index = i + 1
                break

        # Insert cleanup code
        lines.insert(insert_index, cleanup_code)

        # Write back to file
        with open(system_path, "w") as f:
            f.write("\n".join(lines))

        print(f"✅ Added cleanup endpoints to {system_path}")
        return True

    except Exception as e:
        print(f"❌ Error adding cleanup endpoint: {e}")
        return False


def create_cleanup_test():
    """Create a test for the cleanup endpoint."""

    test_content = '''#!/usr/bin/env python3
"""
Test API Cleanup Endpoint
"""

import requests
import json
import time

def test_cleanup_endpoint():
    """Test the cleanup endpoint functionality"""

    base_url = "http://localhost:8000"

    try:
        # Test cleanup status endpoint
        print("Testing cleanup status endpoint...")
        response = requests.get(f"{base_url}/cleanup/status")

        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status endpoint working: {status}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False

        # Test cleanup endpoint
        print("Testing cleanup endpoint...")
        response = requests.post(f"{base_url}/cleanup")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cleanup endpoint working: {result}")
            return True
        else:
            print(f"❌ Cleanup endpoint failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if test_cleanup_endpoint():
        print("✅ Cleanup endpoint tests passed")
    else:
        print("❌ Cleanup endpoint tests failed")
'''

    test_path = Path("scripts/test_cleanup_endpoint.py")
    with open(test_path, "w") as f:
        f.write(test_content)

    print(f"✅ Created cleanup test at {test_path}")


def main():
    """Main function."""
    print("=== Adding API Cleanup Functionality ===")

    if add_cleanup_endpoint():
        create_cleanup_test()
        print("✅ API cleanup functionality added successfully")
        return 0
    else:
        print("❌ Failed to add API cleanup functionality")
        return 1


if __name__ == "__main__":
    sys.exit(main())
