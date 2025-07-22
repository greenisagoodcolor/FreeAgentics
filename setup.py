#!/usr/bin/env python
"""Setup configuration for FreeAgentics.

This makes FreeAgentics a proper Python package that can be installed
in development mode, eliminating the need for sys.path manipulation.
"""

from setuptools import setup, find_packages

setup(
    name="freeagentics",
    version="1.0.0-alpha",
    description="Revolutionary Multi-Agent Active Inference Research Platform",
    author="FreeAgentics Committee",
    packages=find_packages(exclude=["tests*", "benchmarks*", "scripts*", "web*"]),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies only - full list in requirements.txt
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "sqlalchemy>=2.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pymdp>=0.0.7.1",
    ],
    entry_points={
        "console_scripts": [
            # Convert key scripts to proper CLI commands
            "freeagentics-db-indexes=scripts.apply_database_indexes:main",
            "freeagentics-db-seed=scripts.seed_database:main",
            "freeagentics-db-test=scripts.test_database_connection:main",
            "freeagentics-memory-report=scripts.generate_memory_profiling_report:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
