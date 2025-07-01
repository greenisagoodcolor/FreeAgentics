"""
Comprehensive tests for Export Validator

This test suite provides complete coverage of the deployment package validation
system, including structure checks, file integrity, configuration validation,
and hardware compatibility.
"""

import pytest
import json
import hashlib
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module under test
from infrastructure.deployment.export_validator import (
    ExportValidator,
    DeploymentVerifier,
    ValidationStatus,
    HardwarePlatform,
    ValidationResult,
    PackageManifest,
    HardwareRequirements,
    validate_export,
)


class TestValidationStatus:
    """Test the ValidationStatus enum"""

    def test_validation_status_values(self):
        """Test validation status enum values"""
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.SKIPPED.value == "skipped"


class TestExportValidator:
    """Test the ExportValidator class"""

    @pytest.fixture
    def validator(self):
        """Create an ExportValidator instance"""
        return ExportValidator()

    @pytest.fixture
    def temp_package_dir(self):
        """Create a temporary package directory with basic structure"""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_package_"))
        
        # Create required files
        (temp_dir / "manifest.json").write_text(json.dumps({
            "package_name": "test_agent",
            "version": "1.0.0",
            "agent_class": "TestAgent",
            "created_at": "2024-01-01T00:00:00Z",
            "platform": "linux",
            "files": {},
            "dependencies": []
        }))
        
        (temp_dir / "agent_config.json").write_text(json.dumps({
            "agent_class": "TestAgent",
            "personality": "helpful"
        }))
        
        (temp_dir / "gnn_model.json").write_text(json.dumps({
            "metadata": {"version": "1.0"},
            "layers": []
        }))
        
        (temp_dir / "requirements.txt").write_text("numpy>=1.0\ntorch>=1.0")
        (temp_dir / "run.sh").write_text("#!/bin/bash\necho 'Starting agent'")
        (temp_dir / "README.md").write_text("# Test Agent")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_initialization(self, validator):
        """Test validator initialization"""
        assert isinstance(validator.required_files, set)
        assert isinstance(validator.optional_files, set)
        assert isinstance(validator.hardware_profiles, dict)
        
        # Check required files
        assert "manifest.json" in validator.required_files
        assert "agent_config.json" in validator.required_files
        assert "gnn_model.json" in validator.required_files
        
        # Check hardware profiles
        assert HardwarePlatform.RASPBERRY_PI in validator.hardware_profiles
        assert HardwarePlatform.MAC_MINI in validator.hardware_profiles

    def test_validate_package_directory(self, validator, temp_package_dir):
        """Test validating a package directory"""
        results = validator.validate_package(temp_package_dir)
        
        # Should have multiple validation results
        assert len(results) > 0
        assert isinstance(results[0], ValidationResult)
        
        # Check that structure validation passed
        structure_results = [r for r in results if r.check_name == "package_structure"]
        assert len(structure_results) > 0
        assert structure_results[0].status == ValidationStatus.PASSED

    def test_check_structure_all_files_present(self, validator, temp_package_dir):
        """Test structure check when all required files are present"""
        results = validator._check_structure(temp_package_dir)
        
        # Should pass structure check
        structure_result = next(r for r in results if r.check_name == "package_structure")
        assert structure_result.status == ValidationStatus.PASSED
        assert "All required files present" in structure_result.message

    def test_check_structure_missing_files(self, validator, temp_package_dir):
        """Test structure check with missing required files"""
        # Remove a required file
        (temp_package_dir / "manifest.json").unlink()
        
        results = validator._check_structure(temp_package_dir)
        
        # Should fail structure check
        structure_result = next(r for r in results if r.check_name == "package_structure")
        assert structure_result.status == ValidationStatus.FAILED
        assert "manifest.json" in structure_result.message
        assert "manifest.json" in structure_result.details["missing_files"]

    def test_check_manifest_valid(self, validator, temp_package_dir):
        """Test checking valid manifest"""
        results = validator._check_manifest(temp_package_dir)
        
        # Should pass manifest checks
        structure_results = [r for r in results if r.check_name == "manifest_structure"]
        assert len(structure_results) > 0
        assert structure_results[0].status == ValidationStatus.PASSED

    def test_check_manifest_missing_file(self, validator, temp_package_dir):
        """Test checking missing manifest file"""
        # Remove manifest
        (temp_package_dir / "manifest.json").unlink()
        
        results = validator._check_manifest(temp_package_dir)
        
        # Should fail
        assert len(results) == 1
        assert results[0].status == ValidationStatus.FAILED
        assert "not found" in results[0].message

    def test_calculate_file_hash(self, validator, temp_package_dir):
        """Test calculating file hash"""
        # Create test file
        test_file = temp_package_dir / "test.txt"
        test_content = "test content"
        test_file.write_text(test_content)
        
        # Calculate hash
        calculated_hash = validator._calculate_file_hash(test_file)
        expected_hash = hashlib.sha256(test_content.encode()).hexdigest()
        
        assert calculated_hash == expected_hash

    def test_check_files_valid_configs(self, validator, temp_package_dir):
        """Test checking valid configuration files"""
        results = validator._check_files(temp_package_dir)
        
        # Should pass config checks
        config_results = [r for r in results if r.check_name == "agent_config"]
        model_results = [r for r in results if r.check_name == "gnn_model"]
        
        assert len(config_results) > 0
        assert config_results[0].status == ValidationStatus.PASSED
        
        assert len(model_results) > 0
        assert model_results[0].status == ValidationStatus.PASSED


class TestDeploymentVerifier:
    """Test the DeploymentVerifier class"""

    @pytest.fixture
    def verifier(self):
        """Create a DeploymentVerifier instance"""
        return DeploymentVerifier()

    @pytest.fixture
    def temp_package_dir(self):
        """Create a temporary package directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_verify_"))
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, verifier):
        """Test verifier initialization"""
        assert hasattr(verifier, 'verify_deployment')

    def test_verify_deployment(self, verifier, temp_package_dir):
        """Test deployment verification"""
        results = verifier.verify_deployment(temp_package_dir, HardwarePlatform.RASPBERRY_PI)
        
        # Should return list of validation results
        assert isinstance(results, list)
        assert len(results) > 0

    @patch('os.kill')
    def test_check_process_running(self, mock_kill, verifier, temp_package_dir):
        """Test checking if process is running"""
        # Create a PID file
        pid_file = temp_package_dir / "agent.pid"
        pid_file.write_text("1234")
        
        # Mock successful process check (os.kill with signal 0 doesn't kill, just checks existence)
        mock_kill.return_value = None  # No exception means process exists
        
        results = verifier._check_process_running(temp_package_dir)
        
        # Should find running process
        assert len(results) > 0
        process_results = [r for r in results if "process" in r.check_name]
        assert len(process_results) > 0


class TestValidateExportFunction:
    """Test the validate_export function"""

    @patch('infrastructure.deployment.export_validator.ExportValidator')
    def test_validate_export_success(self, mock_validator_class):
        """Test successful export validation"""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_package.return_value = [
            ValidationResult("test", ValidationStatus.PASSED, "Passed")
        ]
        mock_validator_class.return_value = mock_validator
        
        # Test function
        result = validate_export("test_package.zip", "raspberry_pi")
        
        # Should return True for successful validation
        assert result is True
        mock_validator.validate_package.assert_called_once()
