# FreeAgentics

> **Multi-agent AI platform implementing Active Inference for autonomous, mathematically-principled intelligent systems**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

## 🎯 What is FreeAgentics?

FreeAgentics creates **truly autonomous AI agents** using **Active Inference** - a mathematical framework from cognitive science. Unlike chatbots or scripted AI, our agents make decisions by minimizing free energy, leading to emergent, intelligent behavior.

### ⚡ Key Features
- 🧠 **Mathematical Rigor**: Based on peer-reviewed Active Inference theory (pymdp)
- 🤖 **True Autonomy**: No hardcoded behaviors - all actions emerge from principles  
- 👥 **Multi-Agent Coordination**: Agents form coalitions and collaborate dynamically
- 🎮 **Real-time Visualization**: Live dashboard showing belief states and decisions
- 🚀 **Production Ready**: Enterprise-grade performance and edge deployment
- 📝 **Natural Language**: Create agents using human-readable specifications

## 🚀 Quick Start (5 minutes)

```bash
# Install and run
git clone https://github.com/your-org/freeagentics.git
cd freeagentics
npm install && pip install -e .

# Start the platform  
npm run dev
# Visit http://localhost:3000
```

**Create your first agent**:
```python
from freeagentics import Agent, World

# Create autonomous explorer
agent = Agent.create("Explorer", personality={'curiosity': 0.8})

# Add to world and watch it act intelligently
world = World(grid_size=20)
world.add_agent(agent)
world.simulate(steps=100)
```

## 📚 Documentation

### **For Different Users**
- 👤 **New Users**: [Quick Start Guide](docs/QUICKSTART.md) 
- 👩‍💻 **Developers**: [Developer Guide](docs/DEVELOPER-GUIDE.md)
- 🏗️ **Architects**: [Architecture Docs](docs/ARCHITECTURE.md) 
- 🚀 **DevOps**: [Deployment Guide](docs/DEPLOYMENT.md)

### **Complete Documentation**: [docs/](docs/)

## 🎮 Live Demo

**[Try FreeAgentics Demo](http://localhost:3000)** - Create agents and watch them interact in real-time

## 🤝 Community & Contributing

- 📖 **Contributing**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/freeagentics/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/freeagentics/issues)
- 💼 **Enterprise**: [Contact Us](mailto:enterprise@freeagentics.ai)

## 📊 Project Status

- **Version**: 0.9.0 (Seed-stage MVP)
- **Test Coverage**: 88% (375 passing tests)
- **Expert Review**: Committee approved
- **Production Ready**: ✅ Edge deployment ready

## 🔬 Research & Academic Use

FreeAgentics is designed for:
- 🎓 **Cognitive Science Research**: Test Active Inference theories
- 🤖 **AI Research**: Multi-agent systems and emergent behavior
- 📚 **Education**: Interactive Active Inference demonstrations
- 🏢 **Industry**: Production multi-agent applications

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Expert Committee Validated** | **Production Ready** | **Open Source**

*Making Active Inference accessible, visual, and deployable for everyone.*
