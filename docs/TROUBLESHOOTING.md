# FreeAgentics Troubleshooting Guide

> Common issues and solutions

## Installation Issues

### Python Package Installation Problems
```bash
# Issue: Permission denied during pip install
# Solution: Use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Node.js Version Compatibility
```bash
# Issue: Node version too old
# Solution: Install Node 18+ using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

### Port Already in Use
```bash
# Issue: Port 3000 already in use
# Solution: Use different port
PORT=3001 npm run dev

# Or find and kill process using port 3000
lsof -ti:3000 | xargs kill -9
```

## Agent Creation Problems

### Agent Template Loading Fails
**Symptoms**: Template selection shows empty or errors
**Causes**: 
- Missing Python dependencies
- Incorrect virtual environment

**Solutions**:
```bash
# Verify Active Inference dependencies
pip install pymdp numpy torch

# Check if in correct virtual environment
which python  # Should point to venv/bin/python
```

### Belief State Validation Errors
**Symptoms**: "Belief state must sum to 1.0" errors
**Cause**: Numerical precision issues

**Solution**:
```python
# Ensure proper normalization
beliefs = beliefs / beliefs.sum()
```

### Agent Creation Timeout
**Symptoms**: Agent creation hangs or times out
**Causes**:
- Insufficient memory
- Heavy computational load

**Solutions**:
```bash
# Reduce agent complexity
# Use simpler personality settings
# Monitor memory usage: top or htop
```

## Coalition Formation Issues

### Agents Not Forming Coalitions
**Symptoms**: Agents remain isolated despite proximity
**Debugging**:
```python
# Check agent compatibility
agent1.get_compatibility(agent2)

# Verify communication range
world.communication_range = 5  # Increase range

# Check coalition criteria
agent.coalition_threshold = 0.5  # Lower threshold
```

### Coalition Stability Problems
**Symptoms**: Coalitions form and immediately dissolve
**Solutions**:
- Reduce environmental volatility
- Increase coalition stability parameters
- Check for conflicting agent objectives

## Performance Problems

### Slow Simulation Speed
**Diagnostic Steps**:
```bash
# Check system resources
top
# Look for high CPU/memory usage

# Profile Python code
python -m cProfile simulation.py

# Reduce simulation complexity
world.grid_size = 10  # Smaller world
world.max_agents = 5  # Fewer agents
```

### Memory Leaks
**Symptoms**: Memory usage continuously increases
**Solutions**:
```python
# Enable garbage collection
import gc
gc.collect()

# Use agent pooling
world.enable_agent_pooling(pool_size=100)

# Monitor memory
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### UI Responsiveness Issues
**Browser**: Use Chrome for best D3.js performance
**Network**: Check for failed API calls in browser dev tools
**Caching**: Clear browser cache and restart dev server

## API Integration Issues

### REST API Connection Failures
```bash
# Check if API server is running
curl http://localhost:3000/api/health

# Verify API endpoints
curl -X GET http://localhost:3000/api/v1/agents

# Check CORS issues (browser dev console)
```

### WebSocket Connection Problems
**Symptoms**: Real-time updates not working
**Solutions**:
- Check firewall settings
- Verify WebSocket support in browser
- Monitor network tab for WebSocket errors

### Authentication Issues
```bash
# Check API key configuration
echo $FREEAGENTICS_API_KEY

# Verify token format
curl -H "Authorization: Bearer $TOKEN" http://localhost:3000/api/v1/agents
```

## Database Issues

### Connection Failed
```bash
# Check database service
docker ps  # If using Docker
pg_isready  # If using PostgreSQL directly

# Reset database
./infrastructure/scripts/setup/initialize-database.sh
```

### Migration Errors
```bash
# Run migrations manually
alembic upgrade head

# Reset migrations if corrupted
alembic stamp head
```

## Edge Deployment Issues

### Hardware Compatibility
**ARM Devices**: Ensure ARM-compatible Python packages
```bash
# Install ARM-specific packages
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Memory Constraints**: 
- Reduce agent complexity
- Use smaller models
- Enable memory optimization flags

### Network Connectivity
- Check for internet access if using cloud APIs
- Configure local mode for offline operation
- Verify firewall rules for required ports

## Development Environment Issues

### Pre-commit Hooks Failing
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run hooks manually to debug
pre-commit run --all-files
```

### Test Failures
```bash
# Run tests with maximum verbosity
pytest -vvv --tb=long

# Run specific test
pytest tests/test_agent_creation.py::test_explorer_creation -v

# Clear pytest cache
rm -rf .pytest_cache
```

### Linting Errors
```bash
# Fix Python formatting
black .
isort .

# Fix TypeScript formatting  
cd web && npm run lint:fix
```

## Getting Help

### Log Collection
```bash
# Collect system info
python --version
node --version
npm --version

# Save logs
npm run dev > app.log 2>&1

# Python error logs
python simulation.py 2> error.log
```

### Issue Reporting
When reporting issues, include:
- [ ] Operating system and version
- [ ] Python and Node.js versions
- [ ] Full error message and stack trace
- [ ] Steps to reproduce
- [ ] Expected vs actual behavior

### Community Support
- **GitHub Issues**: [Report bugs](https://github.com/your-org/freeagentics/issues)
- **Discussions**: [Ask questions](https://github.com/your-org/freeagentics/discussions)
- **Documentation**: [Check docs](README.md)

---

*Still having issues? [Create an issue](https://github.com/your-org/freeagentics/issues/new) with full details.*
