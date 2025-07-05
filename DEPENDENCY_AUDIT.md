# Dependency Audit Report

## Actual Package Counts

### requirements.txt (Original)

- Total lines: 261
- Actual packages (excluding comments/blanks): 260
- Includes editable install of freeagentics itself

### requirements-core.txt

- Total lines: 58
- Actual packages: 31
- Percentage of original: 31/260 = 11.9%
- Reduction: 88.1%

### requirements-dev.txt

- Total lines: 52
- References requirements-core.txt (-r requirements-core.txt)
- Additional dev packages: ~20

### requirements-production.txt

- Total lines: 52
- Standalone production packages (doesn't reference core)
- Actual packages: 35

## Key Findings

1. **The 82% reduction claim is actually conservative**
   - Actual reduction: 88.1% (260 → 31 packages)
   - Production requirements: 35 packages (86.5% reduction)

2. **PyMDP IS installed**
   - Package name: inferactively-pymdp==0.0.7.1
   - Successfully imports
   - But code doesn't actually use it properly

3. **Package Organization**
   - Core: Minimal essentials
   - Dev: Includes core + testing tools
   - Production: Standalone minimal set

## Corrections to Previous Claims

1. **Dependency count was actually UNDERSTATED**
   - Claimed: 47 core packages
   - Actual: 31 core packages
   - Better than claimed!

2. **PyMDP availability**
   - Previously stated: "Not installed"
   - Reality: Installed but not properly utilized

## Actual Core Dependencies

```
fastapi==0.115.14
uvicorn==0.35.0
pydantic==2.9.2
starlette==0.46.2
numpy==2.3.1
torch==2.7.1
scipy==1.16.0
inferactively-pymdp==0.0.7.1  # <-- IT'S HERE!
torch-geometric==2.6.1
networkx==3.5
httpx==0.28.1
aiofiles==24.1.0
python-multipart==0.0.20
sqlalchemy==2.0.41
alembic==1.16.2
psycopg2-binary==2.9.10
python-dotenv==1.1.1
pyyaml==6.0.2
toml==0.10.2
structlog==24.4.0
passlib==1.7.4
pyjwt==2.10.1
cryptography==45.0.5
pandas==2.3.0
pytest==8.4.1
pytest-asyncio==1.0.0
pytest-cov==6.2.1
mypy==1.16.1
black==25.1.0
isort==6.0.1
flake8==7.3.0
```

## Conclusion

The dependency reduction was actually MORE successful than claimed (88% vs 82%). However, the real issue isn't missing dependencies - it's that the code doesn't properly use what's installed, particularly PyMDP.
