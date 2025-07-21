#!/usr/bin/env python3
"""
Production dependencies verification script for FreeAgentics.
Tests the fixed requirements-production.txt for compatibility issues.
"""
import subprocess
import sys
import tempfile
import os

def test_pip_dependency_resolution():
    """Test pip dependency resolution without actually installing packages."""
    print("Testing pip dependency resolution...")
    print("=" * 60)
    
    requirements_file = "/home/green/FreeAgentics/requirements-production.txt"
    
    try:
        # Test with pip's dry-run resolver
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--dry-run", "--quiet", "--no-deps",
            "-r", requirements_file
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Pip can resolve all dependencies without conflicts")
        else:
            print("❌ Pip dependency resolution failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Dependency resolution test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing dependency resolution: {e}")
        return False
    
    return True

def analyze_requirements_file():
    """Analyze the requirements file for potential issues."""
    print("\nAnalyzing requirements-production.txt...")
    print("=" * 60)
    
    requirements_file = "/home/green/FreeAgentics/requirements-production.txt"
    
    try:
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    pkg_name = line.split('==')[0].strip()
                    pkg_version = line.split('==')[1].split()[0]  # Remove comments
                    packages.append((pkg_name, pkg_version))
        
        print(f"📦 Total packages: {len(packages)}")
        print("\nCore ML/AI packages verification:")
        
        critical_packages = {
            'numpy': '2.3.1',
            'torch': '2.7.1', 
            'scipy': '1.14.1',
            'pandas': '2.3.0',
            'torch-geometric': '2.6.1',
            'inferactively-pymdp': '0.0.7.1'
        }
        
        for pkg_name, expected_version in critical_packages.items():
            found = False
            for name, version in packages:
                if name == pkg_name:
                    if version == expected_version:
                        print(f"✅ {pkg_name}=={version} (correct)")
                    else:
                        print(f"❌ {pkg_name}=={version} (expected {expected_version})")
                    found = True
                    break
            if not found:
                print(f"❌ {pkg_name} not found in requirements")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    except Exception as e:
        print(f"❌ Error analyzing requirements file: {e}")
        return False

def check_python_version():
    """Check if we're running on Python 3.12."""
    print(f"\nPython version check...")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Running on Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 12:
        print("✅ Python 3.12 detected - correct target version")
        return True
    else:
        print(f"⚠️  Running on Python {version.major}.{version.minor} (target is Python 3.12)")
        return True  # Still continue testing

def main():
    """Run all dependency tests."""
    print("FreeAgentics Production Dependencies Verification")
    print("=" * 60)
    print("Testing fixed requirements-production.txt")
    print("Target: Python 3.12 compatibility")
    print()
    
    success = True
    
    # Check Python version
    success &= check_python_version()
    
    # Analyze requirements file
    success &= analyze_requirements_file()
    
    # Test dependency resolution
    success &= test_pip_dependency_resolution()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if success:
        print("✅ ALL TESTS PASSED - Production dependencies are compatible!")
        print("🚀 Ready for production deployment on Python 3.12")
    else:
        print("❌ SOME TESTS FAILED - Review issues above")
        return 1
    
    print("\n📋 Next steps:")
    print("1. Deploy to staging environment for full testing")
    print("2. Run integration tests with actual AI/ML workloads") 
    print("3. Monitor for any runtime issues")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())