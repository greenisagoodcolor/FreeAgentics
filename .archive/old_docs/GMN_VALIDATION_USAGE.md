# GMN Validation Framework Usage Guide

## Overview

The GMN Validation Framework provides comprehensive validation for Generative Model Network specifications with hard failures on any violation. This ensures that only well-formed, mathematically valid, and practically sound specifications reach the AI model implementation.

## Quick Start

```python
from inference.active.gmn_validation import GMNValidationFramework
from inference.active.gmn_parser import GMNParser

# Initialize framework
framework = GMNValidationFramework()
parser = GMNParser()

# Validate a GMN specification
spec = {
    "nodes": [
        {"name": "location", "type": "state", "num_states": 4},
        {"name": "obs_location", "type": "observation", "num_observations": 4},
        {"name": "move", "type": "action", "num_actions": 4}
    ],
    "edges": [
        {"from": "location", "to": "obs_location", "type": "generates"}
    ]
}

# Validate
result = framework.validate(spec)
if result.is_valid:
    print("✓ Specification is valid!")
else:
    print("✗ Validation failed:")
    for error in result.errors:
        print(f"  - {error.validator}: {error.message}")
```

## Validation Types

### 1. Syntax Validation
Validates basic GMN format structure and required fields.

```python
from inference.active.gmn_validation import GMNSyntaxValidator

validator = GMNSyntaxValidator()

# This will fail - empty specification
try:
    validator.validate({})
except GMNValidationError as e:
    print(f"Syntax error: {e}")

# This will fail - missing nodes
try:
    validator.validate({"edges": []})
except GMNValidationError as e:
    print(f"Syntax error: {e}")
```

### 2. Semantic Validation
Validates logical consistency and graph structure.

```python
from inference.active.gmn_validation import GMNSemanticValidator

validator = GMNSemanticValidator()

# This will fail - circular dependency
spec = {
    "nodes": [
        {"name": "node1", "type": "state", "num_states": 4},
        {"name": "node2", "type": "belief", "about": "node1"},
        {"name": "node3", "type": "transition"}
    ],
    "edges": [
        {"from": "node1", "to": "node2", "type": "depends_on"},
        {"from": "node2", "to": "node3", "type": "depends_on"},
        {"from": "node3", "to": "node1", "type": "depends_on"}  # Creates cycle
    ]
}

try:
    validator.validate(spec)
except GMNValidationError as e:
    print(f"Semantic error: {e}")
```

### 3. Mathematical Validation
Validates probability distributions and numerical constraints.

```python
from inference.active.gmn_validation import GMNMathematicalValidator

validator = GMNMathematicalValidator()

# This will fail - probabilities don't sum to 1
spec = {
    "nodes": [
        {
            "name": "belief1",
            "type": "belief",
            "about": "state1",
            "initial_distribution": [0.3, 0.3, 0.3]  # Sum = 0.9
        }
    ]
}

try:
    validator.validate(spec)
except GMNValidationError as e:
    print(f"Mathematical error: {e}")
```

### 4. Type Validation
Validates data types and required attributes.

```python
from inference.active.gmn_validation import GMNTypeValidator

validator = GMNTypeValidator()

# This will fail - invalid node type
spec = {
    "nodes": [
        {"name": "invalid", "type": "unknown_type"}
    ]
}

try:
    validator.validate(spec)
except GMNValidationError as e:
    print(f"Type error: {e}")
```

### 5. Constraint Validation
Validates business rules and practical constraints.

```python
from inference.active.gmn_validation import GMNConstraintValidator

validator = GMNConstraintValidator()

# This will fail - action space too large
spec = {
    "nodes": [
        {"name": "action1", "type": "action", "num_actions": 1000000}
    ]
}

try:
    validator.validate(spec)
except GMNValidationError as e:
    print(f"Constraint error: {e}")
```

## Reality Checkpoints

Use reality checkpoints to catch suspicious but technically valid patterns:

```python
# Test with reality checks
problematic_spec = {
    "nodes": [
        {"name": "state1", "type": "state", "num_states": 1},
        {"name": "obs1", "type": "observation", "num_observations": 1000}
    ],
    "edges": [
        {"from": "state1", "to": "obs1", "type": "generates"}
    ]
}

result = framework.validate_with_reality_checks(problematic_spec)
print(f"Valid: {result.is_valid}")
print(f"Errors: {len(result.errors)}")
print(f"Warnings: {len(result.warnings)}")

for warning in result.warnings:
    print(f"Warning: {warning.message}")
```

## Text Format Validation

Validate GMN text format files:

```python
# Load and validate GMN text file
with open('examples/gmn_specifications/minimal_valid.gmn', 'r') as f:
    gmn_text = f.read()

# Parse text format
spec = parser.parse_text(gmn_text)

# Validate parsed specification
result = framework.validate(spec)
print(f"Text format validation: {'✓ PASS' if result.is_valid else '✗ FAIL'}")
```

## Comprehensive Validation

Get detailed validation results with all error types:

```python
# Invalid specification with multiple errors
invalid_spec = {
    "nodes": [
        {"name": "invalid1"},  # Missing type
        {"name": "invalid2", "type": "invalid_type"},  # Invalid type
        {"name": "invalid3", "type": "state", "num_states": -1}  # Negative dimension
    ]
}

result = framework.validate(invalid_spec)

print(f"Validation result: {result.is_valid}")
print(f"Total errors: {len(result.errors)}")
print(f"Total warnings: {len(result.warnings)}")

print("\nErrors by validator:")
for error in result.errors:
    print(f"  {error.validator}: {error.message}")
    if error.node_name:
        print(f"    Node: {error.node_name}")
    if error.context:
        print(f"    Context: {error.context}")
```

## Integration with Existing Code

### With GMN Parser

```python
from inference.active.gmn_parser import GMNParser
from inference.active.gmn_validation import GMNValidationFramework

def parse_and_validate_gmn(gmn_text: str):
    """Parse and validate GMN specification."""
    parser = GMNParser()
    framework = GMNValidationFramework()
    
    # Parse text to specification
    spec = parser.parse_text(gmn_text)
    
    # Validate specification
    result = framework.validate(spec)
    
    if not result.is_valid:
        error_details = [f"{e.validator}: {e.message}" for e in result.errors]
        raise GMNValidationError(f"Validation failed: {'; '.join(error_details)}")
    
    return spec
```

### With API Endpoints

```python
from fastapi import HTTPException

async def create_agent_with_gmn(gmn_spec: str):
    """Create agent with validated GMN specification."""
    try:
        # Parse and validate GMN
        spec = parse_and_validate_gmn(gmn_spec)
        
        # Create agent with validated specification
        agent = create_agent_from_gmn(spec)
        return agent
        
    except GMNValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Valid GMN Example

```python
# Complete valid GMN specification
valid_gmn = {
    "nodes": [
        {
            "name": "agent_state",
            "type": "state",
            "num_states": 9
        },
        {
            "name": "agent_observation",
            "type": "observation", 
            "num_observations": 9
        },
        {
            "name": "agent_action",
            "type": "action",
            "num_actions": 5
        },
        {
            "name": "state_belief",
            "type": "belief",
            "about": "agent_state",
            "initial_distribution": [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112]
        },
        {
            "name": "goal_preference",
            "type": "preference",
            "preferred_observation": 4,
            "preference_strength": 2.0
        },
        {
            "name": "state_transition",
            "type": "transition"
        },
        {
            "name": "observation_likelihood", 
            "type": "likelihood"
        }
    ],
    "edges": [
        {"from": "agent_state", "to": "observation_likelihood", "type": "depends_on"},
        {"from": "observation_likelihood", "to": "agent_observation", "type": "generates"},
        {"from": "agent_state", "to": "state_transition", "type": "depends_on"},
        {"from": "agent_action", "to": "state_transition", "type": "depends_on"},
        {"from": "state_belief", "to": "agent_state", "type": "depends_on"}
    ]
}

# This should pass all validation
result = framework.validate(valid_gmn)
assert result.is_valid
print("✓ Complete GMN specification is valid!")
```

## Error Handling Best Practices

```python
def robust_gmn_validation(spec):
    """Robust GMN validation with comprehensive error handling."""
    framework = GMNValidationFramework()
    
    try:
        # Attempt validation
        result = framework.validate(spec)
        
        if result.is_valid:
            print("✓ Specification is valid")
            
            # Show warnings if any
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning.message}")
            
            return True
            
        else:
            print("✗ Validation failed")
            
            # Group errors by validator
            errors_by_validator = {}
            for error in result.errors:
                if error.validator not in errors_by_validator:
                    errors_by_validator[error.validator] = []
                errors_by_validator[error.validator].append(error.message)
            
            # Display grouped errors
            for validator, messages in errors_by_validator.items():
                print(f"\n{validator} errors:")
                for message in messages:
                    print(f"  - {message}")
            
            return False
            
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False
```

## Performance Testing

```python
import time

def test_validation_performance():
    """Test validation performance with large specifications."""
    
    # Create large specification
    large_spec = {
        "nodes": [
            {"name": f"state_{i}", "type": "state", "num_states": 10}
            for i in range(100)
        ] + [
            {"name": f"obs_{i}", "type": "observation", "num_observations": 10}
            for i in range(100)
        ],
        "edges": [
            {"from": f"state_{i}", "to": f"obs_{i}", "type": "generates"}
            for i in range(100)
        ]
    }
    
    framework = GMNValidationFramework()
    
    # Time validation
    start_time = time.time()
    result = framework.validate(large_spec)
    end_time = time.time()
    
    print(f"Large spec validation: {'✓ PASS' if result.is_valid else '✗ FAIL'}")
    print(f"Validation time: {end_time - start_time:.3f} seconds")
    print(f"Nodes: {len(large_spec['nodes'])}, Edges: {len(large_spec['edges'])}")
```

## Common Validation Errors and Solutions

### 1. Missing Required Attributes

**Error**: `TypeValidator: Belief node 'belief1' missing required attribute: about`

**Solution**:
```python
# Wrong
{"name": "belief1", "type": "belief"}

# Correct  
{"name": "belief1", "type": "belief", "about": "state1"}
```

### 2. Probability Distribution Issues

**Error**: `MathematicalValidator: Probability distribution does not sum to 1`

**Solution**:
```python
# Wrong
"initial_distribution": [0.3, 0.3, 0.3]  # Sum = 0.9

# Correct
"initial_distribution": [0.333, 0.333, 0.334]  # Sum = 1.0
```

### 3. Dimension Mismatches

**Error**: `MathematicalValidator: Dimension mismatch: state has 4 dimensions but observation has 3`

**Solution**:
```python
# Wrong
{"name": "state1", "type": "state", "num_states": 4}
{"name": "obs1", "type": "observation", "num_observations": 3}

# Correct - matching dimensions
{"name": "state1", "type": "state", "num_states": 4}
{"name": "obs1", "type": "observation", "num_observations": 4}
```

### 4. Circular Dependencies

**Error**: `SemanticValidator: Circular dependency detected`

**Solution**: Review edge relationships to ensure no cycles in dependency graph.

This validation framework ensures robust, reliable GMN specifications for production AI systems.