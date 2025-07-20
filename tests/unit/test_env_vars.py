"""
Environment Variable Validation Tests
Following TDD: RED -> GREEN -> REFACTOR
"""

import os
from unittest.mock import patch

import pytest


class TestEnvironmentVariables:
    """Test suite for environment variable validation"""

    def test_database_url_required(self):
        """Test that DATABASE_URL is required and validated"""
        # RED: This test should fail initially
        from config.env_validator import Settings

        # Remove DATABASE_URL if it exists
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL is required"):
                Settings()

    def test_database_url_format_validation(self):
        """Test DATABASE_URL must be valid PostgreSQL connection string"""
        from config.env_validator import Settings

        invalid_urls = [
            "invalid_url",
            "mysql://user:pass@host/db",  # Wrong protocol
            "postgresql://",  # Missing components
            "postgresql://user@/db",  # Missing host
            "postgresql://user:pass@host",  # Missing database
        ]

        # Base environment with all required vars except DATABASE_URL
        base_env = {
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
            "REDIS_URL": "redis://localhost:6379/0",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        for invalid_url in invalid_urls:
            env_vars = {**base_env, "DATABASE_URL": invalid_url}
            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(
                    ValueError, match="Invalid PostgreSQL DATABASE_URL"
                ):
                    Settings()

    def test_valid_database_url_accepted(self):
        """Test valid PostgreSQL URLs are accepted"""
        from config.env_validator import Settings

        valid_urls = [
            "postgresql://user:pass@localhost/dbname",
            "postgresql://user:pass@localhost:5432/dbname",
            "postgres://user:pass@localhost/dbname",
            "postgresql+asyncpg://user:pass@localhost/dbname",
        ]

        required_env = {
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
            "REDIS_URL": "redis://localhost:6379/0",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        for valid_url in valid_urls:
            env_vars = {**required_env, "DATABASE_URL": valid_url}
            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings()
                assert settings.DATABASE_URL == valid_url

    def test_api_key_required(self):
        """Test that API_KEY is required"""
        from config.env_validator import Settings

        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql://user:pass@localhost/db"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="API_KEY is required"):
                Settings()

    def test_secret_key_required(self):
        """Test that SECRET_KEY is required"""
        from config.env_validator import Settings

        env_vars = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="SECRET_KEY is required"):
                Settings()

    def test_jwt_secret_key_required(self):
        """Test that JWT_SECRET_KEY is required"""
        from config.env_validator import Settings

        env_vars = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="JWT_SECRET_KEY is required"):
                Settings()

    def test_redis_url_required_and_validated(self):
        """Test that REDIS_URL is required and must be valid"""
        from config.env_validator import Settings

        base_env = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
        }

        # Test missing REDIS_URL
        with patch.dict(os.environ, base_env, clear=True):
            with pytest.raises(ValueError, match="REDIS_URL is required"):
                Settings()

        # Test invalid REDIS_URL
        invalid_redis_urls = [
            "invalid_url",
            "http://localhost:6379",  # Wrong protocol
            "redis://",  # Missing host
        ]

        # Add all required fields for validation test
        complete_env = {
            **base_env,
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        for invalid_url in invalid_redis_urls:
            env_vars = {**complete_env, "REDIS_URL": invalid_url}
            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(ValueError, match="Invalid Redis URL"):
                    Settings()

    def test_postgres_credentials_required(self):
        """Test that POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB are required"""
        from config.env_validator import Settings

        base_env = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
            "REDIS_URL": "redis://localhost:6379/0",
        }

        # Test missing POSTGRES_USER
        with patch.dict(os.environ, base_env, clear=True):
            with pytest.raises(ValueError, match="POSTGRES_USER is required"):
                Settings()

        # Test missing POSTGRES_PASSWORD
        env_with_user = {**base_env, "POSTGRES_USER": "testuser"}
        with patch.dict(os.environ, env_with_user, clear=True):
            with pytest.raises(
                ValueError, match="POSTGRES_PASSWORD is required"
            ):
                Settings()

        # Test missing POSTGRES_DB
        env_with_user_pass = {**env_with_user, "POSTGRES_PASSWORD": "testpass"}
        with patch.dict(os.environ, env_with_user_pass, clear=True):
            with pytest.raises(ValueError, match="POSTGRES_DB is required"):
                Settings()

    def test_all_required_vars_present(self):
        """Test successful validation when all required vars are present"""
        from config.env_validator import Settings

        env_vars = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/dbname",
            "API_KEY": "test-api-key-123",
            "SECRET_KEY": "test-secret-key-456",
            "JWT_SECRET_KEY": "test-jwt-secret-789",
            "REDIS_URL": "redis://localhost:6379/0",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.DATABASE_URL == env_vars["DATABASE_URL"]
            assert settings.API_KEY == env_vars["API_KEY"]
            assert settings.SECRET_KEY == env_vars["SECRET_KEY"]
            assert settings.JWT_SECRET_KEY == env_vars["JWT_SECRET_KEY"]
            assert settings.REDIS_URL == env_vars["REDIS_URL"]
            assert settings.POSTGRES_USER == env_vars["POSTGRES_USER"]
            assert settings.POSTGRES_PASSWORD == env_vars["POSTGRES_PASSWORD"]
            assert settings.POSTGRES_DB == env_vars["POSTGRES_DB"]

    def test_no_default_values_allowed(self):
        """Test that no default values are used - all vars must be explicitly set"""
        from config.env_validator import Settings

        # Even with partial env vars, should fail - no defaults
        partial_env = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-key",
        }

        with patch.dict(os.environ, partial_env, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_empty_string_values_rejected(self):
        """Test that empty string values are rejected"""
        from config.env_validator import Settings

        base_env = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
            "REDIS_URL": "redis://localhost:6379/0",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        # Test each required var with empty string
        for key in base_env.keys():
            test_env = base_env.copy()
            test_env[key] = ""

            with patch.dict(os.environ, test_env, clear=True):
                with pytest.raises(ValueError, match=f"{key} cannot be empty"):
                    Settings()

    def test_whitespace_only_values_rejected(self):
        """Test that whitespace-only values are rejected"""
        from config.env_validator import Settings

        base_env = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
            "REDIS_URL": "redis://localhost:6379/0",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        # Test each required var with whitespace only
        for key in base_env.keys():
            test_env = base_env.copy()
            test_env[key] = "   "

            with patch.dict(os.environ, test_env, clear=True):
                with pytest.raises(ValueError, match=f"{key} cannot be empty"):
                    Settings()


class TestEnvironmentValidatorImport:
    """Test that env_validator can be imported and used"""

    def test_validator_module_exists(self):
        """Test that the env_validator module can be imported"""
        try:
            from config import env_validator

            assert hasattr(env_validator, "Settings")
        except ImportError:
            pytest.fail("config.env_validator module does not exist")

    def test_settings_class_exists(self):
        """Test that Settings class exists and can be instantiated with valid env"""
        from config.env_validator import Settings

        valid_env = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "API_KEY": "test-api-key",
            "SECRET_KEY": "test-secret-key",
            "JWT_SECRET_KEY": "test-jwt-secret",
            "REDIS_URL": "redis://localhost:6379/0",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        with patch.dict(os.environ, valid_env, clear=True):
            settings = Settings()
            assert isinstance(settings, Settings)
