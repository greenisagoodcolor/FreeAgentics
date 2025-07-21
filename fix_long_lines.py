#!/usr/bin/env python3
"""Fix long lines (E501) in Python files."""

import os
import re
from pathlib import Path

def fix_long_lines_in_file(filepath):
    """Fix long lines in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        # Skip if line is already ok
        if len(line.rstrip()) <= 88:
            new_lines.append(line)
            continue
            
        # Handle long string literals in logging/print statements
        if 'logger.' in line or 'print(' in line:
            # Check if it's a simple f-string that can be split
            match = re.match(r'^(\s*)(logger\.\w+|print)\(f"(.+)"\)(\s*)$', line.rstrip())
            if match:
                indent, func, msg, trailing = match.groups()
                if len(msg) > 60:
                    # Split at a reasonable point
                    split_point = msg.rfind(' ', 0, 60)
                    if split_point > 0:
                        part1 = msg[:split_point]
                        part2 = msg[split_point+1:]
                        new_line = f'{indent}{func}(\n{indent}    f"{part1}"\n{indent}    f" {part2}"\n{indent}){trailing}\n'
                        new_lines.append(new_line)
                        modified = True
                        continue
        
        # Handle long dictionary/list literals
        if '{' in line or '[' in line:
            # Try to add line breaks after commas
            if line.count(',') > 1 and not line.strip().endswith((',', '{', '[')):
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ''
                # Find opening bracket
                open_pos = max(line.find('{'), line.find('['))
                if open_pos > 0:
                    prefix = line[:open_pos+1]
                    content = line[open_pos+1:].rstrip()
                    if content.endswith(('}', ']')):
                        closer = content[-1]
                        content = content[:-1]
                        # Split on commas
                        parts = content.split(', ')
                        if len(parts) > 1:
                            new_line = prefix + '\n'
                            for j, part in enumerate(parts):
                                new_line += f'{indent}    {part.strip()}'
                                if j < len(parts) - 1:
                                    new_line += ','
                                new_line += '\n'
                            new_line += indent + closer
                            if line.endswith('\n'):
                                new_line += '\n'
                            new_lines.append(new_line)
                            modified = True
                            continue
        
        # For other long lines, try to break at operators or parentheses
        if len(line.rstrip()) > 88:
            # Find good breaking points
            break_chars = [' and ', ' or ', ' + ', ', ', ' if ', ' else ']
            best_break = -1
            best_char = ''
            
            for bc in break_chars:
                pos = line.rfind(bc, 0, 88)
                if pos > best_break:
                    best_break = pos
                    best_char = bc
            
            if best_break > 0:
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ''
                part1 = line[:best_break].rstrip()
                part2 = line[best_break + len(best_char):].lstrip()
                
                # Add extra indent for continuation
                new_line = part1 + best_char.rstrip() + '\n'
                new_line += indent + '    ' + part2
                if not part2.endswith('\n'):
                    new_line += '\n'
                new_lines.append(new_line)
                modified = True
                continue
        
        # If we couldn't fix it, keep the original
        new_lines.append(line)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    """Fix long lines in Python files."""
    dirs_to_fix = ['agents', 'inference']
    fixed_count = 0
    
    for dir_name in dirs_to_fix:
        for filepath in Path(dir_name).rglob('*.py'):
            if fix_long_lines_in_file(filepath):
                print(f"Fixed: {filepath}")
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()