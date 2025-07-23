#!/bin/bash
# Production dependencies verification using virtual environment
# Tests the fixed requirements-production.txt for Python 3.12 compatibility

set -e

echo "=========================================="
echo "FreeAgentics Dependency Verification"
echo "Target: Python 3.12 production deployment"
echo "=========================================="

# Check Python version
echo "🐍 Python version check:"
python3.12 --version

# Create clean test environment
echo "🧹 Creating clean test environment..."
rm -rf dep_test_env
python3.12 -m venv dep_test_env
source dep_test_env/bin/activate

# Upgrade pip to latest version
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Test installation with verbose output
echo "🔍 Testing dependency installation..."
echo "Installing from requirements-production.txt..."

if pip install -r requirements-production.txt; then
    echo "✅ All dependencies installed successfully!"

    # Test imports
    echo "🧪 Testing critical imports..."
    python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy failed: {e}')
    sys.exit(1)

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    # Test basic tensor operation
    x = torch.tensor([1, 2, 3])
    print(f'✅ PyTorch tensor ops: {x * 2}')
except Exception as e:
    print(f'❌ PyTorch failed: {e}')
    sys.exit(1)

try:
    import scipy
    print(f'✅ SciPy: {scipy.__version__}')
except Exception as e:
    print(f'❌ SciPy failed: {e}')
    sys.exit(1)

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except Exception as e:
    print(f'❌ Pandas failed: {e}')
    sys.exit(1)

try:
    import torch_geometric
    print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
except Exception as e:
    print(f'❌ PyTorch Geometric failed: {e}')
    sys.exit(1)

try:
    import pymdp
    print(f'✅ pymdp (inferactively-pymdp): imported successfully')
except Exception as e:
    print(f'❌ pymdp failed: {e}')
    sys.exit(1)

try:
    import networkx as nx
    print(f'✅ NetworkX: {nx.__version__}')
except Exception as e:
    print(f'❌ NetworkX failed: {e}')
    sys.exit(1)

print('🎉 ALL CRITICAL IMPORTS SUCCESSFUL!')
print('🚀 Dependencies are compatible with Python 3.12!')
"

    echo ""
    echo "📋 Installed package versions:"
    pip list | grep -E "(numpy|torch|scipy|pandas|torch-geometric|inferactively-pymdp|networkx)"

    echo ""
    echo "✅ VERIFICATION COMPLETE: All dependencies compatible!"
    echo "🎯 Production deployment ready for Python 3.12"

else
    echo "❌ Dependency installation failed!"
    echo "Check error messages above for details"
    exit 1
fi

# Cleanup
deactivate
rm -rf dep_test_env

echo "=========================================="
echo "🎉 SUCCESS: FreeAgentics production dependencies verified!"
echo "📝 See DEPENDENCY_COMPATIBILITY_MATRIX.md for full details"
echo "=========================================="
