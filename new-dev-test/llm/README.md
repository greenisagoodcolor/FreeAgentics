# LLM Integration Module

This module provides production-ready LLM integration for FreeAgentics, enabling GMN (Generative Model Notation) generation from natural language prompts.

## Features

- **Multiple Provider Support**: OpenAI, Anthropic, Ollama (local), and Mock providers
- **Automatic Failover**: Seamless fallback when primary provider fails
- **Rate Limiting**: Built-in rate limit handling for API providers
- **Health Monitoring**: Track provider performance and availability
- **GMN Specialization**: Optimized prompts for GMN generation
- **Async/Await**: Full async support for high performance

## Quick Start

```python
from llm import create_llm_factory

# Create factory with auto-detection
factory = create_llm_factory()

# Generate GMN from natural language
gmn = await factory.generate_gmn(
    prompt="Create an agent that explores a grid world",
    agent_type="explorer"
)
```

## Providers

### OpenAI

- Models: GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo
- Requires: `OPENAI_API_KEY` environment variable
- Best for: High-quality GMN generation, complex reasoning

### Anthropic

- Models: Claude 3 Opus, Sonnet, Haiku, Claude 3.5 Sonnet
- Requires: `ANTHROPIC_API_KEY` environment variable
- Best for: Detailed GMN with safety considerations

### Ollama (Local)

- Models: Llama 3, Mistral, Mixtral, CodeLlama, etc.
- Requires: Ollama service running locally
- Best for: Privacy, no API costs, offline usage

### Mock

- For testing and development
- No external dependencies
- Generates realistic GMN structures

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export LLM_PROVIDER="auto"  # auto, openai, anthropic, ollama, mock
```

### Configuration File

```python
config = {
    "provider": "auto",
    "openai": {
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_retries": 3
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.2
    },
    "ollama": {
        "model": "llama3.2",
        "base_url": "http://localhost:11434"
    }
}

factory = LLMProviderFactory(config)
```

## GMN Generation

### Basic Generation

```python
# Simple agent
gmn = await factory.generate_gmn(
    prompt="Create a cautious explorer agent"
)

# With agent type hints
gmn = await factory.generate_gmn(
    prompt="Create a market trading agent",
    agent_type="trader"
)

# With constraints
gmn = await factory.generate_gmn(
    prompt="Create a team coordinator",
    agent_type="coordinator",
    constraints={
        "max_states": 10,
        "max_actions": 5
    }
)
```

### GMN Validation

```python
provider = await factory.get_provider()
is_valid, errors = await provider.validate_gmn(gmn)

if not is_valid:
    print(f"Validation errors: {errors}")
```

### GMN Refinement

```python
# Iteratively improve GMN
refined_gmn = await provider.refine_gmn(
    gmn_spec=current_gmn,
    feedback="Add memory capabilities"
)
```

## Advanced Usage

### Custom Messages

```python
from llm import LLMMessage, LLMRole

messages = [
    LLMMessage(
        role=LLMRole.SYSTEM,
        content="You are an expert in active inference"
    ),
    LLMMessage(
        role=LLMRole.USER,
        content="Design a multi-agent system"
    )
]

response = await factory.generate(messages)
```

### Provider-Specific Features

```python
# Force specific provider
response = await factory.generate(
    messages,
    provider_type=ProviderType.ANTHROPIC
)

# Ollama with GPU acceleration
config = {
    "ollama": {
        "num_gpu": -1,  # Use all GPU layers
        "num_ctx": 8192  # Larger context window
    }
}
```

### Health Monitoring

```python
health = await factory.health_check()

for provider, status in health["providers"].items():
    print(f"{provider}: {status['success_rate']:.1f}% success rate")
```

## Error Handling

```python
try:
    gmn = await factory.generate_gmn(prompt)
except LLMError as e:
    # Handle LLM-specific errors
    print(f"LLM Error: {e}")
```

## Performance Tips

1. **Use Caching**: Providers cache sessions for connection reuse
2. **Batch Requests**: Generate multiple GMNs in parallel
3. **Choose Right Model**: Use smaller models for simple tasks
4. **Local Models**: Use Ollama for high-volume, low-latency needs
5. **Rate Limits**: Built-in handling, but be mindful of quotas

## Testing

Run provider tests:

```bash
# Test all providers
pytest tests/test_llm_providers.py

# Test specific provider
pytest tests/test_llm_providers.py::TestLLMProviders::test_openai_provider

# Skip slow tests
pytest tests/test_llm_providers.py -m "not slow"
```

## Troubleshooting

### OpenAI/Anthropic Issues

- Check API key is set correctly
- Verify network connectivity
- Monitor rate limits in logs

### Ollama Issues

- Ensure Ollama service is running: `ollama serve`
- Pull models first: `ollama pull llama3.2`
- Check available models: `ollama list`

### Fallback Not Working

- Check health status of providers
- Verify fallback chain configuration
- Look for errors in logs

## Examples

See `examples/llm_integration_example.py` for comprehensive examples including:

- Basic usage
- GMN generation for different agent types
- Provider comparison
- Error handling
- Interactive refinement

## Future Enhancements

- [ ] Streaming response support
- [ ] Token usage optimization
- [ ] Caching for repeated prompts
- [ ] Fine-tuned models for GMN
- [ ] Multi-modal support (diagrams to GMN)
