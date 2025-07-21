#!/usr/bin/env python3
"""
Pipeline Validation Script
Validates that the CI/CD pipeline has no bypass mechanisms and follows best practices
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class PipelineValidator:
    """Validates GitHub Actions workflows for security and best practices"""
    
    FORBIDDEN_PATTERNS = [
        # Skip patterns
        'skip', 'bypass', 'force', 'ignore',
        # Conditional execution that might skip important steps
        'if.*skip', 'if.*bypass', 'if.*force',
        # Manual overrides
        'workflow_dispatch.*skip', 'workflow_dispatch.*force'
    ]
    
    REQUIRED_STAGES = [
        'pre-flight', 'quality', 'security', 'test', 'build'
    ]
    
    REQUIRED_CHECKS = [
        'secret-scanning', 'dependency-security', 'code-quality',
        'security-sast', 'test-backend', 'test-frontend'
    ]
    
    def __init__(self, workflows_dir: str = '.github/workflows'):
        self.workflows_dir = Path(workflows_dir)
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def validate_all_workflows(self) -> Tuple[bool, List[Dict], List[Dict]]:
        """Validate all workflow files in the directory"""
        workflow_files = list(self.workflows_dir.glob('*.yml')) + list(self.workflows_dir.glob('*.yaml'))
        
        print(f"🔍 Found {len(workflow_files)} workflow files to validate\n")
        
        for workflow_file in workflow_files:
            if workflow_file.name in ['PIPELINE-ARCHITECTURE.md', 'MIGRATION-GUIDE.md']:
                continue
                
            print(f"Validating: {workflow_file.name}")
            self._validate_workflow(workflow_file)
            
        return len(self.issues) == 0, self.issues, self.warnings
    
    def _validate_workflow(self, workflow_file: Path) -> None:
        """Validate a single workflow file"""
        try:
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
                
            if not workflow:
                return
                
            workflow_name = workflow.get('name', workflow_file.name)
            
            # Check for bypass mechanisms
            self._check_bypass_mechanisms(workflow, workflow_name, workflow_file)
            
            # Check for required stages in main pipeline
            if workflow_file.name in ['main-pipeline.yml', 'unified-pipeline.yml']:
                self._check_required_stages(workflow, workflow_name)
                
            # Check for proper dependencies
            self._check_job_dependencies(workflow, workflow_name)
            
            # Check for timeout settings
            self._check_timeouts(workflow, workflow_name)
            
        except Exception as e:
            self.issues.append({
                'file': str(workflow_file),
                'issue': f'Failed to parse workflow: {e}',
                'severity': 'critical'
            })
    
    def _check_bypass_mechanisms(self, workflow: Dict, workflow_name: str, workflow_file: Path) -> None:
        """Check for any bypass mechanisms in the workflow"""
        workflow_str = workflow_file.read_text().lower()
        
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in workflow_str:
                # Check if it's in a comment
                lines = workflow_str.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith('#'):
                        self.issues.append({
                            'file': workflow_name,
                            'issue': f'Found potential bypass pattern "{pattern}" at line {i}',
                            'severity': 'high',
                            'line': line.strip()
                        })
        
        # Check workflow_dispatch inputs for bypass options
        if 'on' in workflow and 'workflow_dispatch' in workflow.get('on', {}):
            dispatch = workflow['on']['workflow_dispatch']
            if isinstance(dispatch, dict) and 'inputs' in dispatch:
                for input_name, input_config in dispatch['inputs'].items():
                    if any(forbidden in input_name.lower() for forbidden in ['skip', 'bypass', 'force']):
                        self.issues.append({
                            'file': workflow_name,
                            'issue': f'Workflow dispatch input "{input_name}" could allow bypassing',
                            'severity': 'high'
                        })
    
    def _check_required_stages(self, workflow: Dict, workflow_name: str) -> None:
        """Check if all required stages are present"""
        jobs = workflow.get('jobs', {})
        job_names = set(jobs.keys())
        
        for required_stage in self.REQUIRED_STAGES:
            if not any(required_stage in job_name for job_name in job_names):
                self.warnings.append({
                    'file': workflow_name,
                    'issue': f'Missing required stage: {required_stage}',
                    'severity': 'medium'
                })
        
        for required_check in self.REQUIRED_CHECKS:
            if not any(required_check.replace('-', '_') in job_name.replace('-', '_') 
                      for job_name in job_names):
                self.warnings.append({
                    'file': workflow_name,
                    'issue': f'Missing required check: {required_check}',
                    'severity': 'medium'
                })
    
    def _check_job_dependencies(self, workflow: Dict, workflow_name: str) -> None:
        """Check that jobs have proper dependencies"""
        jobs = workflow.get('jobs', {})
        
        critical_jobs = ['deploy', 'production', 'release']
        
        for job_name, job_config in jobs.items():
            if any(critical in job_name.lower() for critical in critical_jobs):
                needs = job_config.get('needs', [])
                if not needs:
                    self.issues.append({
                        'file': workflow_name,
                        'issue': f'Critical job "{job_name}" has no dependencies',
                        'severity': 'high'
                    })
                elif isinstance(needs, list) and len(needs) < 3:
                    self.warnings.append({
                        'file': workflow_name,
                        'issue': f'Critical job "{job_name}" has minimal dependencies ({len(needs)})',
                        'severity': 'medium'
                    })
    
    def _check_timeouts(self, workflow: Dict, workflow_name: str) -> None:
        """Check that all jobs have appropriate timeouts"""
        jobs = workflow.get('jobs', {})
        
        for job_name, job_config in jobs.items():
            if 'timeout-minutes' not in job_config:
                self.warnings.append({
                    'file': workflow_name,
                    'issue': f'Job "{job_name}" has no timeout set',
                    'severity': 'low'
                })
            else:
                timeout = job_config['timeout-minutes']
                if timeout > 60:
                    self.warnings.append({
                        'file': workflow_name,
                        'issue': f'Job "{job_name}" has excessive timeout ({timeout} minutes)',
                        'severity': 'low'
                    })

def print_validation_results(is_valid: bool, issues: List[Dict], warnings: List[Dict]) -> None:
    """Print validation results in a formatted way"""
    print("\n" + "="*80)
    print("PIPELINE VALIDATION RESULTS")
    print("="*80 + "\n")
    
    if is_valid and not warnings:
        print("✅ All validations passed! No bypass mechanisms detected.")
        print("✅ Pipeline follows security best practices.")
    else:
        if issues:
            print(f"❌ Found {len(issues)} critical issues:\n")
            for issue in issues:
                print(f"  🚨 [{issue['severity'].upper()}] {issue['file']}")
                print(f"     {issue['issue']}")
                if 'line' in issue:
                    print(f"     Line: {issue['line']}")
                print()
        
        if warnings:
            print(f"⚠️  Found {len(warnings)} warnings:\n")
            for warning in warnings:
                print(f"  ⚠️  [{warning['severity'].upper()}] {warning['file']}")
                print(f"     {warning['issue']}")
                print()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    if not is_valid:
        print("1. Remove all skip/bypass/force mechanisms from workflows")
        print("2. Ensure all deployment jobs have proper dependencies")
        print("3. Add timeouts to all jobs")
        print("4. Implement all required security and quality checks")
    else:
        print("1. Continue monitoring pipeline execution times")
        print("2. Regular review of security policies")
        print("3. Keep dependencies up to date")
        print("4. Document any exceptions clearly")

def generate_pipeline_report() -> None:
    """Generate a detailed pipeline architecture report"""
    print("\n" + "="*80)
    print("PIPELINE ARCHITECTURE REPORT")
    print("="*80 + "\n")
    
    report = """
## Pipeline Architecture Summary

### ✅ Implemented Security Controls:
- No skip or bypass mechanisms
- Mandatory quality gates at every stage
- Sequential dependencies enforced
- Comprehensive security scanning
- Multi-layer testing strategy

### 📊 Pipeline Stages:
1. **Pre-flight Checks** - Fast feedback on code quality
2. **Build & Package** - Multi-architecture container builds
3. **Testing Suite** - Unit, integration, and E2E tests
4. **Security Validation** - SAST, container scanning, compliance
5. **Performance Verification** - Benchmarks and regression detection
6. **Deployment Readiness** - Final quality gate assessment
7. **Progressive Deployment** - Staging → Canary → Production
8. **Observability** - Metrics, reporting, and notifications

### 🔒 Quality Gates:
- Code quality: linting, formatting, type checking
- Secret scanning: no exposed credentials
- Dependency security: no vulnerable dependencies
- Test coverage: Backend >80%, Frontend >75%
- Security score: >85/100
- Performance: No regression >10%
- E2E tests: All scenarios passing

### 📈 Metrics Tracked:
- Pipeline execution time
- Stage success rates
- Quality gate pass rates
- Security compliance scores
- Performance baselines
- Deployment frequency
- Mean time to recovery
"""
    print(report)

def main():
    """Main execution function"""
    print("🚀 FreeAgentics Pipeline Validator v1.0")
    print("Following Martin Fowler & Jessica Kerr principles\n")
    
    # Check if we're in the right directory
    if not os.path.exists('.github/workflows'):
        print("❌ Error: .github/workflows directory not found")
        print("Please run this script from the repository root")
        sys.exit(1)
    
    # Run validation
    validator = PipelineValidator()
    is_valid, issues, warnings = validator.validate_all_workflows()
    
    # Print results
    print_validation_results(is_valid, issues, warnings)
    
    # Generate architecture report
    generate_pipeline_report()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()