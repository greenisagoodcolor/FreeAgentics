#!/usr/bin/env python3
"""
Test Dependencies - TDD approach for backend requirements
Tests must fail first (RED), then pass after installation (GREEN)
No try/except blocks - hard failures only
"""

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version


def test_package_installed(package_name: str, import_name: str = None) -> None:
    """Test if a package is installed and importable - no graceful fallbacks"""
    import_name = import_name or package_name

    # Test 1: Package metadata exists
    try:
        installed_version = version(package_name)
        print(f"✓ {package_name} version {installed_version} found in metadata")
    except PackageNotFoundError:
        print(f"✗ {package_name} NOT FOUND in package metadata")
        raise AssertionError(f"{package_name} is not installed")

    # Test 2: Package is importable
    try:
        importlib.import_module(import_name)
        print(f"✓ {import_name} successfully imported")
    except ImportError as e:
        print(f"✗ Failed to import {import_name}: {e}")
        raise AssertionError(f"Cannot import {import_name}")

    return installed_version


def test_package_version(package_name: str, expected_version: str) -> None:
    """Test if package has exact version - no flexibility"""
    actual_version = version(package_name)

    if actual_version != expected_version:
        print(
            f"✗ {package_name} version mismatch: expected {expected_version}, got {actual_version}"
        )
        raise AssertionError(f"{package_name} version {actual_version} != {expected_version}")

    print(f"✓ {package_name} version {expected_version} verified")


def test_passlib() -> None:
    """Test passlib for password hashing"""
    print("\n[TEST] passlib - password hashing library")

    # Test installation
    test_package_installed("passlib")

    # Test specific functionality
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Test hashing
    test_password = "test_password_123"
    hashed = pwd_context.hash(test_password)
    assert pwd_context.verify(test_password, hashed), "Password verification failed"
    assert not pwd_context.verify("wrong_password", hashed), "Wrong password should fail"

    print("✓ passlib password hashing working correctly")

    # Test exact version
    test_package_version("passlib", "1.7.4")


def test_pyjwt() -> None:
    """Test PyJWT for JWT token handling"""
    print("\n[TEST] PyJWT - JWT token library")

    # Test installation
    test_package_installed("PyJWT", "jwt")

    # Test functionality
    import jwt

    secret = "test_secret_key"
    payload = {"user_id": "123", "exp": 1234567890}

    # Test encoding
    token = jwt.encode(payload, secret, algorithm="HS256")
    assert isinstance(token, str), "Token should be string"

    # Test decoding
    decoded = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": False})
    assert decoded["user_id"] == "123", "Decoded payload mismatch"

    print("✓ PyJWT encoding/decoding working correctly")

    # Test exact version
    test_package_version("PyJWT", "2.8.0")


def test_python_jose() -> None:
    """Test python-jose with cryptography backend"""
    print("\n[TEST] python-jose[cryptography] - JWT with crypto support")

    # Test installation
    test_package_installed("python-jose", "jose")

    # Test cryptography backend
    test_package_installed("cryptography")

    # Test functionality
    from jose import jwt
    from jose.exceptions import JWTError

    secret = "test_secret_key"
    payload = {"sub": "user123", "name": "Test User"}

    # Test encoding with RS256 (requires cryptography)
    token = jwt.encode(payload, secret, algorithm="HS256")
    decoded = jwt.decode(token, secret, algorithms=["HS256"])

    assert decoded["sub"] == "user123", "JWT decode failed"

    # Test error handling
    try:
        jwt.decode(token, "wrong_secret", algorithms=["HS256"])
        raise AssertionError("Should have raised JWTError")
    except JWTError:
        print("✓ python-jose error handling working correctly")

    # Test exact version
    test_package_version("python-jose", "3.3.0")
    test_package_version("cryptography", "41.0.7")


def test_asyncpg() -> None:
    """Test asyncpg for PostgreSQL async driver"""
    print("\n[TEST] asyncpg - PostgreSQL async driver")

    # Test installation
    test_package_installed("asyncpg")

    # Test basic import of key components
    import asyncpg

    # Verify core functions exist
    assert hasattr(asyncpg, "connect"), "asyncpg.connect missing"
    assert hasattr(asyncpg, "create_pool"), "asyncpg.create_pool missing"

    print("✓ asyncpg core components available")

    # Test exact version
    test_package_version("asyncpg", "0.29.0")


def test_sqlalchemy() -> None:
    """Test SQLAlchemy ORM"""
    print("\n[TEST] SQLAlchemy - ORM library")

    # Test installation
    test_package_installed("SQLAlchemy", "sqlalchemy")

    # Test core components
    from sqlalchemy import Column, Integer, String, create_engine
    from sqlalchemy.orm import declarative_base

    # Test basic ORM functionality
    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String)

    # Test engine creation (sqlite in-memory for testing)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    print("✓ SQLAlchemy ORM functionality working")

    # Test exact version
    test_package_version("SQLAlchemy", "2.0.23")


def test_fastapi() -> None:
    """Test FastAPI web framework"""
    print("\n[TEST] FastAPI - web framework")

    # Test installation
    test_package_installed("fastapi")

    # Test core components
    from fastapi import FastAPI
    from fastapi.security import HTTPBearer

    # Test basic app creation
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    # Test security utilities
    HTTPBearer()

    print("✓ FastAPI core components working")

    # Test exact version
    test_package_version("fastapi", "0.104.1")

    # Also test key dependencies
    test_package_installed("pydantic")
    test_package_installed("starlette")
    test_package_installed("uvicorn")


def main():
    """Run all dependency tests - fail fast, no graceful handling"""
    print("=" * 60)
    print("DEPENDENCY VALIDATION TESTS")
    print("TDD: These tests MUST fail before package installation")
    print("=" * 60)

    all_tests = [
        test_passlib,
        test_pyjwt,
        test_python_jose,
        test_asyncpg,
        test_sqlalchemy,
        test_fastapi,
    ]

    failed_tests = []

    for test_func in all_tests:
        try:
            test_func()
        except AssertionError as e:
            failed_tests.append((test_func.__name__, str(e)))
            print(f"\n✗ FAILED: {test_func.__name__}")
        except Exception as e:
            failed_tests.append((test_func.__name__, f"Unexpected error: {e}"))
            print(f"\n✗ ERROR in {test_func.__name__}: {e}")

    print("\n" + "=" * 60)

    if failed_tests:
        print(f"FAILED: {len(failed_tests)} tests failed")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        sys.exit(1)
    else:
        print("SUCCESS: All dependency tests passed!")
        print("\nVerified packages:")
        print("  - passlib==1.7.4")
        print("  - PyJWT==2.8.0")
        print("  - python-jose==3.3.0")
        print("  - cryptography==41.0.7")
        print("  - asyncpg==0.29.0")
        print("  - SQLAlchemy==2.0.23")
        print("  - fastapi==0.104.1")
        sys.exit(0)


if __name__ == "__main__":
    main()
