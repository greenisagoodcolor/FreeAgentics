"""Check registered routes in the minimal app."""

import os

os.environ.update(
    {
        "DATABASE_URL": "sqlite:///test_routes.db",
        "API_KEY": "test-api-key",
        "SECRET_KEY": "test-secret-key-32-characters-minimum-required-for-security",
        "JWT_SECRET": "test-jwt-secret-32-characters-minimum-required-for-security",
        "REDIS_URL": "redis://localhost:6379/0",
        "ENVIRONMENT": "testing",
    }
)

from api.main_minimal import app

print("=== Registered Routes ===\n")
for route in app.routes:
    if hasattr(route, "path"):
        methods = getattr(route, "methods", ["?"])
        print(f"{', '.join(methods):10} {route.path}")

print("\n=== OpenAPI Paths ===")
schema = app.openapi()
if schema and "paths" in schema:
    for path in sorted(schema["paths"].keys()):
        print(f"  {path}")
