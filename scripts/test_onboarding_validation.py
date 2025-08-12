#!/usr/bin/env python3
"""
Automated onboarding validation test suite.
Tests clean repository cloning and installation process.
"""

import subprocess
import tempfile
import shutil
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


class OnboardingValidator:
    """Validates the complete developer onboarding experience."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.results: Dict[str, dict] = {}
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> dict:
        """Execute command and return results with timing."""
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            duration = time.time() - start_time
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'duration': duration,
                'return_code': -1
            }
    
    def test_fresh_clone(self, test_dir: Path) -> bool:
        """Test cloning repository from local path."""
        print("📦 Testing fresh repository clone...")
        
        # Clone the repository
        result = self.run_command([
            'git', 'clone', str(self.repo_path), str(test_dir)
        ])
        
        self.results['clone'] = result
        
        if not result['success']:
            print(f"❌ Clone failed: {result['stderr']}")
            return False
            
        print(f"✅ Clone successful ({result['duration']:.1f}s)")
        return True
    
    def test_dependency_installation(self, test_dir: Path) -> bool:
        """Test make install command."""
        print("📦 Testing dependency installation...")
        
        # Test make install
        result = self.run_command(['make', 'install'], cwd=test_dir, timeout=600)
        
        self.results['install'] = result
        
        if not result['success']:
            print(f"❌ Installation failed: {result['stderr']}")
            return False
            
        print(f"✅ Installation successful ({result['duration']:.1f}s)")
        return True
    
    def test_environment_consistency(self, test_dir: Path) -> bool:
        """Test Node.js version consistency."""
        print("🔍 Testing environment consistency...")
        
        # Check root package.json node version
        root_pkg = test_dir / 'package.json'
        web_pkg = test_dir / 'web' / 'package.json'
        
        issues = []
        
        if root_pkg.exists():
            with open(root_pkg) as f:
                root_data = json.load(f)
                root_node = root_data.get('engines', {}).get('node', 'not specified')
        else:
            issues.append("Root package.json missing")
            
        if web_pkg.exists():
            with open(web_pkg) as f:
                web_data = json.load(f)
                web_node = web_data.get('engines', {}).get('node', 'not specified')
        else:
            issues.append("Web package.json missing")
            
        if not issues:
            if root_node != web_node:
                issues.append(f"Node version mismatch: root={root_node}, web={web_node}")
        
        self.results['consistency'] = {
            'success': len(issues) == 0,
            'issues': issues,
            'root_node_version': root_node if 'root_node' in locals() else 'unknown',
            'web_node_version': web_node if 'web_node' in locals() else 'unknown'
        }
        
        if issues:
            print(f"❌ Consistency issues found: {', '.join(issues)}")
            return False
            
        print("✅ Environment consistency validated")
        return True
    
    def test_build_process(self, test_dir: Path) -> bool:
        """Test build process."""
        print("🏗️ Testing build process...")
        
        # Test frontend build
        result = self.run_command(['npm', 'run', 'build'], cwd=test_dir / 'web', timeout=300)
        
        self.results['build'] = result
        
        if not result['success']:
            print(f"❌ Build failed: {result['stderr']}")
            return False
            
        print(f"✅ Build successful ({result['duration']:.1f}s)")
        return True
    
    def test_dependency_conflicts(self, test_dir: Path) -> bool:
        """Check for dependency conflicts."""
        print("🔍 Testing dependency conflicts...")
        
        issues = []
        
        # Check for duplicate dependencies
        root_pkg = test_dir / 'package.json'
        web_pkg = test_dir / 'web' / 'package.json'
        
        if root_pkg.exists() and web_pkg.exists():
            with open(root_pkg) as f:
                root_data = json.load(f)
            with open(web_pkg) as f:
                web_data = json.load(f)
                
            root_deps = set()
            root_deps.update(root_data.get('dependencies', {}).keys())
            root_deps.update(root_data.get('devDependencies', {}).keys())
            
            web_deps = set()
            web_deps.update(web_data.get('dependencies', {}).keys())
            web_deps.update(web_data.get('devDependencies', {}).keys())
            
            # Check for problematic duplicates (excluding build tools)
            allowed_duplicates = {
                '@types/jest', '@types/node', 'typescript', 
                'concurrently', 'prettier', 'eslint'
            }
            
            duplicates = (root_deps & web_deps) - allowed_duplicates
            if duplicates:
                issues.extend([f"Duplicate dependency: {dep}" for dep in duplicates])
        
        self.results['conflicts'] = {
            'success': len(issues) == 0,
            'issues': issues
        }
        
        if issues:
            print(f"❌ Dependency conflicts found: {', '.join(issues)}")
            return False
            
        print("✅ No dependency conflicts detected")
        return True
    
    def generate_report(self) -> dict:
        """Generate comprehensive validation report."""
        total_tests = len([k for k in self.results.keys() if k != 'summary'])
        passed_tests = len([v for v in self.results.values() 
                           if isinstance(v, dict) and v.get('success', False)])
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_success': passed_tests == total_tests
        }
        
        self.results['summary'] = summary
        return self.results
    
    def run_full_validation(self) -> bool:
        """Run complete onboarding validation suite."""
        print("🚀 Starting onboarding validation suite...")
        print(f"Repository: {self.repo_path}")
        
        with tempfile.TemporaryDirectory(prefix="onboarding_test_") as temp_dir:
            test_dir = Path(temp_dir) / "freeagentics_test"
            
            # Run all tests
            tests = [
                self.test_fresh_clone(test_dir),
                self.test_environment_consistency(test_dir),
                self.test_dependency_conflicts(test_dir),
                self.test_dependency_installation(test_dir),
                self.test_build_process(test_dir)
            ]
            
            success = all(tests)
            
            # Generate report
            report = self.generate_report()
            
            print("\n📊 Validation Results:")
            print(f"   Tests passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
            print(f"   Success rate: {report['summary']['success_rate']:.1%}")
            
            if success:
                print("✅ All onboarding validation tests passed!")
            else:
                print("❌ Some validation tests failed. Check details above.")
                
            return success


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = Path(__file__).parent.parent
    
    validator = OnboardingValidator(repo_path)
    success = validator.run_full_validation()
    
    # Save detailed report
    report_path = Path(__file__).parent.parent / "onboarding_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validator.results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()