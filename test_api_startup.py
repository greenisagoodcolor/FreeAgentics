#!/usr/bin/env python3
"""
Test script to verify API can start
"""

import os

# Set a dummy database URL to bypass the check
os.environ['DATABASE_URL'] = 'postgresql://user:pass@localhost:5432/testdb'

# Try to import the main FastAPI app
try:
    from main import app

    print("✅ API imports successful!")

    # Check routes
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append(route.path)

    print(f"\n✅ Found {len(routes)} routes:")
    for route in sorted(routes):
        print(f"   - {route}")

    # Check for prompts endpoint
    if '/api/v1/prompts' in routes or any('/prompts' in r for r in routes):
        print("\n✅ /api/v1/prompts endpoint is registered!")
    else:
        print("\n⚠️  /api/v1/prompts endpoint not found")

except Exception as e:
    print(f"❌ Failed to import API: {e}")
    import traceback

    traceback.print_exc()
