import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

"""
GMN Parser Module
This module implements the parser for .gmn.md (Generative Model Notation Markdown) files.
It extracts model definitions, validates syntax, and produces an Abstract Syntax Tree (AST)
for further processing.
"""
# Configure logging
logger = logging.getLogger(__name__)


class GMNSyntaxError(Exception):
    """Custom exception for GMN syntax errors"""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, Column {column}: {message}")


class SectionType(Enum):
    """Enumeration of GMN file sections"""

    METADATA = "metadata"
    DESCRIPTION = "description"
    ARCHITECTURE = "architecture"
    PARAMETERS = "parameters"
    ACTIVE_INFERENCE = "active_inference"
    NODE_FEATURES = "node_features"
    EDGE_FEATURES = "edge_features"
    CONSTRAINTS = "constraints"
    VALIDATION = "validation"


@dataclass
class Token:
    """Represents a token in the GMN syntax"""

    type: str
    value: Any
    line: int
    column: int


@dataclass
class ASTNode:
    """Base class for AST nodes"""

    node_type: str
    line: int
    column: int
    children: List["ASTNode"] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a GMN file"""

    ast: ASTNode
    metadata: Dict[str, Any]
    sections: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class GMNLexer:
    """Lexical analyzer for GMN notation blocks"""

    TOKEN_PATTERNS = [
        ("NUMBER", r"-?\d+\.?\d*"),
        ("STRING", r'"[^"]*"'),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COLON", r":"),
        ("COMMA", r", "),
        ("COMMENT", r"//.*$"),
        ("MULTILINE_COMMENT", r"/\*.*?\*/"),
        ("WHITESPACE", r"[ \t]+"),
        ("NEWLINE", r"\n"),
        ("DIRECTIVE", r"@[a-zA-Z_][a-zA-Z0-9_]*"),
    ]

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """Tokenize the input text"""
        while self.position < len(self.text):
            match_found = False
            for token_type, pattern in self.TOKEN_PATTERNS:
                regex = re.compile(pattern, re.MULTILINE if token_type == "COMMENT" else 0)
                match = regex.match(self.text, self.position)
                if match:
                    value = match.group(0)
                    # Skip whitespace and comments
                    if token_type not in ["WHITESPACE", "COMMENT", "MULTILINE_COMMENT"]:
                        token = Token(token_type, value, self.line, self.column)
                        self.tokens.append(token)
                    # Update position
                    self.position = match.end()
                    # Update line and column
                    if token_type == "NEWLINE":
                        self.line += 1
                        self.column = 1
                    else:
                        self.column += len(value)
                    match_found = True
                    break
            if not match_found:
                raise GMNSyntaxError(
                    f"Unexpected character: {self.text[self.position]}",
                    self.line,
                    self.column,
                )
        return self.tokens


class GMNParser:
    """Parser for GMN .gmn.md files"""

    def __init__(self) -> None:
        self.content = ""
        self.lines: List[str] = []
        self.current_line = 0
        self.sections: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def parse(self, content: str) -> ParseResult:
        """Parse GMN content from string"""
        self.content = content
        self.lines = content.split("\n")
        self.current_line = 0
        # Create root AST node
        root = ASTNode("root", 0, 0)
        # Parse sections
        self._parse_sections()
        # Validate required sections
        self._validate_sections()
        # Build AST from sections
        self._build_ast(root)
        return ParseResult(
            ast=root,
            metadata=self.metadata,
            sections=self.sections,
            errors=self.errors,
            warnings=self.warnings,
        )

    def parse_file(self, filepath: Path) -> ParseResult:
        """Parse GMN notation from file"""
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
            return self.parse(content)
        except FileNotFoundError:
            raise GMNSyntaxError(f"File not found: {filepath}")
        except Exception as e:
            raise GMNSyntaxError(f"Error reading file: {e}")

    def validate_syntax(self, content: str) -> List[str]:
        """Validate syntax without full parsing"""
        errors = []
        lines = content.split("\n")
        # Check for required sections
        required_sections = ["# ", "## Metadata", "## Architecture"]
        for section in required_sections:
            if not any(line.strip().startswith(section) for line in lines):
                errors.append(f"Missing required section: {section}")
        # Check GMN blocks
        in_gmn_block = False
        block_start_line = 0
        for i, line in enumerate(lines):
            if line.strip() == "```gmn":
                if in_gmn_block:
                    errors.append(f"Line {i+1}: Nested GMN blocks not allowed")
                in_gmn_block = True
                block_start_line = i + 1
            elif line.strip() == "```" and in_gmn_block:
                in_gmn_block = False
        if in_gmn_block:
            errors.append(f"Line {block_start_line}: Unclosed GMN block")
        return errors

    def _parse_sections(self) -> None:
        """Parse all sections in the document"""
        current_section = None
        section_content: List[str] = []
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            # Check for section headers
            if line.startswith("# "):
                # Model name
                self.metadata["name"] = line[2:].strip()
                self.current_line += 1
            elif line.startswith("## "):
                # Save previous section
                if current_section:
                    # Process the previous section
                    self._process_section(current_section, section_content)
                # Start new section
                current_section = line[3:].strip().lower().replace(" ", "_")
                section_content = []
                self.current_line += 1
            else:
                # Add to current section
                if current_section:
                    section_content.append(line)
                self.current_line += 1
        # Process the last section
        if current_section:
            self._process_section(current_section, section_content)

    def _process_section(self, section_name: str, content: List[str]) -> int:
        """
        Process a specific section based on its type.
        Returns the number of lines consumed by the section block.
        """
        if section_name == "metadata":
            self._parse_metadata("\n".join(content))
            return len(content)
        gmn_content, lines_consumed = self._extract_gmn_block(content)
        if gmn_content:
            parsed = self._parse_gmn_block(gmn_content, section_name)
            if parsed:
                self.sections[section_name] = parsed
        elif section_name == "description":
            self.sections["description"] = "\n".join(content).strip()
        return lines_consumed if lines_consumed > 0 else len(content)

    def _parse_metadata(self, content: str) -> None:
        """Parse metadata section"""
        for line in content.split("\n"):
            if line.strip().startswith("- "):
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    # Parse specific metadata fields
                    if key == "tags":
                        self.metadata[key] = [tag.strip() for tag in value.strip("[]").split(", ")]
                    elif key in ["created", "modified"]:
                        try:
                            self.metadata[key] = datetime.fromisoformat(
                                value.replace("Z", "+00:00")
                            )
                        except Exception:
                            self.metadata[key] = value
                    else:
                        self.metadata[key] = value

    def _extract_gmn_block(self, content_lines: List[str]) -> tuple[Optional[str], int]:
        """Extracts content from a GMN code block and returns it and the number of lines consumed."""
        try:
            start_index = content_lines.index("```gmn")
            # Look for the end of the block *after* the start
            end_index = content_lines.index("```", start_index + 1)
            block_content = "\n".join(content_lines[start_index + 1 : end_index])
            lines_consumed = end_index + 1
            return block_content, lines_consumed
        except ValueError:
            # Block delimiters not found
            return None, 0

    def _parse_gmn_block(self, content: str, section_name: str) -> Dict[str, Any]:
        """Parse a GMN notation block"""
        lexer = GMNLexer(content)
        tokens = lexer.tokenize()
        # Simple recursive descent parser for GMN blocks
        parser = GMNBlockParser(tokens)
        return parser.parse()

    def _validate_sections(self) -> None:
        """Validate that required sections are present"""
        required = ["metadata", "architecture"]
        for section in required:
            if section not in self.sections and section != "metadata":
                self.errors.append(f"Missing required section: {section}")

    def _build_ast(self, root: ASTNode) -> None:
        """Build AST from parsed sections"""
        # Add metadata node
        if self.metadata:
            metadata_node = ASTNode("metadata", 0, 0)
            metadata_node.attributes = self.metadata
            root.children.append(metadata_node)
        # Add section nodes
        for section_name, section_data in self.sections.items():
            section_node = ASTNode(section_name, 0, 0)
            section_node.attributes = (
                section_data if isinstance(section_data, dict) else {"content": section_data}
            )
            root.children.append(section_node)


class GMNBlockParser:
    """Parser for GMN notation blocks"""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None

    def parse(self) -> Dict[str, Any]:
        """Parse the token stream into a dictionary"""
        return self._parse_object()

    def _advance(self) -> None:
        """Move to the next token"""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None

    def _expect(self, token_type: str) -> Token:
        """Expect a specific token type"""
        if not self.current_token or self.current_token.type != token_type:
            raise GMNSyntaxError(
                f"Expected {token_type}, got {self.current_token.type if self.current_token else 'EOF'}"
            )
        token = self.current_token
        self._advance()
        return token

    def _parse_object(self) -> Dict[str, Any]:
        """Parse an object {...}"""
        result = {}
        # Handle both with and without braces
        if self.current_token and self.current_token.type == "LBRACE":
            self._advance()  # Skip {
            while self.current_token and self.current_token.type != "RBRACE":
                # Parse key
                if self.current_token.type == "IDENTIFIER":
                    key = self.current_token.value
                    self._advance()
                    self._expect("COLON")
                    # Parse value
                    value = self._parse_value()
                    result[key] = value
                    # Optional comma
                    if self.current_token and self.current_token.type == "COMMA":
                        self._advance()
                else:
                    break
            if self.current_token and self.current_token.type == "RBRACE":
                self._advance()  # Skip }
        else:
            # Parse without braces (top-level)
            while self.current_token:
                if self.current_token.type == "IDENTIFIER":
                    key = self.current_token.value
                    self._advance()
                    # Handle both: and { after identifier
                    if self.current_token and self.current_token.type == "COLON":
                        self._advance()
                        value = self._parse_value()
                    elif self.current_token and self.current_token.type == "LBRACE":
                        value = self._parse_object()
                    else:
                        value = True  # Boolean flag
                    result[key] = value
                    # Skip newlines
                    while self.current_token and self.current_token.type == "NEWLINE":
                        self._advance()
                else:
                    self._advance()
        return result

    def _parse_value(self) -> Any:
        """Parse a value (string, number, object, array, etc.)"""
        if not self.current_token:
            return None
        if self.current_token.type == "STRING":
            value = self.current_token.value[1:-1]  # Remove quotes
            self._advance()
            return value
        elif self.current_token.type == "NUMBER":
            value = float(self.current_token.value)
            if value.is_integer():
                value = int(value)
            self._advance()
            return value
        elif self.current_token.type == "IDENTIFIER":
            value = self.current_token.value
            self._advance()
            # Check for boolean values
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            return value
        elif self.current_token.type == "LBRACE":
            return self._parse_object()
        elif self.current_token.type == "LBRACKET":
            return self._parse_array()
        else:
            raise GMNSyntaxError(f"Unexpected token: {self.current_token.type}")

    def _parse_array(self) -> List[Any]:
        """Parse an array [...]"""
        result = []
        self._expect("LBRACKET")
        while self.current_token and self.current_token.type != "RBRACKET":
            # Skip newlines and whitespace within the array
            while self.current_token and self.current_token.type in [
                "NEWLINE",
                "WHITESPACE",
            ]:
                self._advance()
            if self.current_token and self.current_token.type != "RBRACKET":
                value = self._parse_value()
                result.append(value)
            # Skip newlines and whitespace after a value
            while self.current_token and self.current_token.type in [
                "NEWLINE",
                "WHITESPACE",
            ]:
                self._advance()
            if self.current_token and self.current_token.type == "COMMA":
                self._advance()
        self._expect("RBRACKET")
        return result


# Example usage and testing
if __name__ == "__main__":
    # Test with sample GMN content
    sample_gmn = """
# Explorer Cautious Model
## Metadata
- Version: 1.0.0
- Author: FreeAgentics Team
- Created: 2024-01-15T10:30:00Z
- Tags: [explorer, cautious]
## Description
A cautious explorer agent.
## Architecture
```gmn
architecture {
  type: "GraphSAGE"
  layers: 3
  hidden_dim: 128
  activation: "relu"
  dropout: 0.2
}
```
"""
    parser = GMNParser()
    result = parser.parse(sample_gmn)
    print("Metadata:", result.metadata)
    print("Sections:", list(result.sections.keys()))
    print("Architecture:", result.sections.get("architecture"))
    print("Errors:", result.errors)
