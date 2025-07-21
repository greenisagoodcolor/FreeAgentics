#!/usr/bin/env python3
"""Script to add authentication headers to all API test calls."""

import re

# Read the file
with open("/home/green/FreeAgentics/tests/unit/test_api_agents.py", "r") as f:
    content = f.read()

# Pattern to match client API calls without headers
patterns = [
    # client.post without headers
    (r"(client\.post\([^)]+\))((?!headers=).)*$", r"\1, headers=get_auth_headers()"),
    # client.get without headers
    (r"(client\.get\([^)]+\))((?!headers=).)*$", r"\1, headers=get_auth_headers()"),
    # client.put without headers
    (r"(client\.put\([^)]+\))((?!headers=).)*$", r"\1, headers=get_auth_headers()"),
    # client.delete without headers
    (r"(client\.delete\([^)]+\))((?!headers=).)*$", r"\1, headers=get_auth_headers()"),
]

# Process line by line to handle multi-line calls
lines = content.split("\n")
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]

    # Check if line contains client.method call
    if re.search(r"client\.(post|get|put|delete)\(", line):
        # Check if headers are already present
        if "headers=" not in line:
            # Simple single-line calls
            if line.strip().endswith(")"):
                # Add headers before closing paren
                line = line.rstrip(")")
                line += ", headers=get_auth_headers())"
            else:
                # Multi-line call - look for closing paren
                j = i + 1
                while j < len(lines) and ")" not in lines[j]:
                    j += 1
                if j < len(lines):
                    # Add headers before closing paren
                    lines[j] = lines[j].replace(")", ", headers=get_auth_headers())")

    new_lines.append(line)
    i += 1

# Write back
with open("/home/green/FreeAgentics/tests/unit/test_api_agents.py", "w") as f:
    f.write("\n".join(new_lines))

print("Updated API test file with authentication headers")
