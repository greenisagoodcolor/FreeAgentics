#!/usr/bin/env python3
"""Validate documentation links in markdown files."""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def find_markdown_files(root_dir: str = ".") -> List[Path]:
    """Find all markdown files in the project."""
    markdown_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common non-doc directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in ["node_modules", "venv", "__pycache__"]
        ]

        for file in files:
            if file.endswith(".md"):
                markdown_files.append(Path(root) / file)

    return markdown_files


def extract_links(file_path: Path) -> List[Tuple[str, str]]:
    """Extract markdown links from a file."""
    links = []

    # Regex pattern for markdown links: [text](url)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for match in link_pattern.finditer(content):
            text, url = match.groups()
            # Skip external links
            if not url.startswith(("http://", "https://", "mailto:", "#")):
                links.append((text, url))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return links


def validate_links(root_dir: str = ".") -> Dict[str, List[str]]:
    """Validate all internal links in markdown files."""
    root_path = Path(root_dir).resolve()
    markdown_files = find_markdown_files(root_dir)
    broken_links = {}

    for file_path in markdown_files:
        file_path = file_path.resolve()
        file_broken_links = []
        links = extract_links(file_path)

        for text, url in links:
            # Resolve the link relative to the file's directory
            link_path = (file_path.parent / url).resolve()

            # Check if the file exists
            if not link_path.exists():
                # Try without anchor
                if "#" in url:
                    base_url = url.split("#")[0]
                    if base_url:  # Only check if there's a file part
                        link_path = (file_path.parent / base_url).resolve()
                        if not link_path.exists():
                            file_broken_links.append(f"[{text}]({url})")
                else:
                    file_broken_links.append(f"[{text}]({url})")

        if file_broken_links:
            try:
                relative_path = file_path.relative_to(root_path)
                broken_links[str(relative_path)] = file_broken_links
            except ValueError:
                # If path is not relative to root, use absolute path
                broken_links[str(file_path)] = file_broken_links

    return broken_links


def print_report(broken_links: Dict[str, List[str]]):
    """Print a report of broken links."""
    if not broken_links:
        print("✅ All documentation links are valid!")
        return

    print("❌ Found broken links in the following files:\n")

    total_broken = 0
    for file_path, links in broken_links.items():
        print(f"📄 {file_path}")
        for link in links:
            print(f"   └─ {link}")
            total_broken += 1
        print()

    print(f"Total broken links: {total_broken}")


if __name__ == "__main__":
    print("🔍 Validating documentation links...\n")
    broken_links = validate_links()
    print_report(broken_links)
