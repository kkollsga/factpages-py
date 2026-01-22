"""
Centralized Display Templates

Declarative markdown-like templates for entity printouts.
All display formats are defined here for easy modification.

Template Syntax:
    {property}              - Entity property (e.g., {name}, {status})
    {table.column}          - Related table value (e.g., {field_reserves.fldRecoverableOil})
    {value:format}          - With format spec (e.g., {value:>10,.1f})
    {value:<20}             - Left-align with width 20

    Arithmetic expressions (columns without table inherit from first):
        {table.col1 + col2 + col3}      - Addition
        {table.col1 - col2}             - Subtraction
        {table.col1 * col2}             - Multiplication
        {table.col1 / col2}             - Division
        {table.col1 ^ 2}                - Power

    Functions:
        {pow(table.col, 2)}             - Power
        {sqrt(table.col)}               - Square root
        {exp(table.col)}                - Exponential
        {log(table.col)}                - Natural logarithm
        {abs(table.col)}                - Absolute value
        {min(table.col1, col2)}         - Minimum
        {max(table.col1, col2)}         - Maximum

    Conditionals:
        {if(table.col > 100, table.col, 0)}     - If-then-else
        Comparisons: <, >, <=, >=, ==, !=

    # Title                 - Section header
    ===                     - Major divider (full width)
    ---                     - Minor divider

    ?{condition} text       - Conditional line (only show if condition is truthy)
    @partners               - Special block (partners list, etc.)

Tables:
    Tables with header (3 rows: header, separator, data):
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | {prop1}  | {prop2}  | {prop3}  |

    Tables without header (just data rows):
        | Label    | {value}  |
        | Label 2  | {value2} |

    Column alignment (in separator for headers, first row for headerless):
        |: cell |               - Left-align column
        | cell :|               - Right-align column
        |: cell :|              - Center-align column
        | cell |                - Default (first col left, rest right)

        Example with header:
            | Name     | Value    |
            |:---------|----------|   <- left-align col 1, default col 2
            | {name}   | {value}  |

        Example headerless:
            |: Label :|: {value} :|   <- center both columns

    Cell merging (N* prefix merges this cell with N-1 cells to the right):
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | {prop1}  |2* {spans_2_cols}   |   <- merges cols 2-3
        |3* {spans_all_3_columns}       |   <- merges all 3 cols

    Cell-specific alignment (overrides column default):
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | {prop1}  |2*: {centered} :|   |   <- centered spanning cell

    Auto-merge: Missing cells at row end are merged into last cell:
        | Col 1 | Col 2 | Col 3 |
        |-------|-------|-------|
        | {a}   | {b}             |   <- {b} auto-spans cols 2-3

    Table dividers (thick separator between sections):
        | Col 1 | Col 2 |
        |-------|-------|
        | {a}   | {b}   |
        |=======|=======|    <- thick divider (===)
        | {c}   | {d}   |

    Optional top separator for headerless tables:
        |:------|------:|     <- defines alignments for headerless
        | {a}   | {b}   |

    Grid locking: Column count is locked to first row's definition.
    All subsequent rows conform to this grid.

Usage:
    from .display import render_entity

    class Field:
        def __str__(self):
            return render_entity(self, "field")
"""

import re
import math
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd


# =============================================================================
# ExpressionParser - Arithmetic Expression Evaluator
# =============================================================================

class ExpressionParser:
    """
    Parse and evaluate arithmetic expressions with variable resolution.

    Supports:
        - Arithmetic: +, -, *, /, ^ (power)
        - Functions: pow(x,y), sqrt(x), exp(x), log(x), abs(x), min(a,b), max(a,b)
        - Conditionals: if(condition, true_val, false_val)
        - Comparisons: <, >, <=, >=, ==, !=
        - Variables: table.column syntax with inherited table names

    Example:
        parser = ExpressionParser(resolver_fn)
        result = parser.evaluate("table.col1 + col2 * 2")
        result = parser.evaluate("if(table.value > 100, table.value, 0)")
        result = parser.evaluate("sqrt(pow(table.x, 2) + pow(table.y, 2))")
    """

    # Token types
    NUMBER = 'NUMBER'
    IDENTIFIER = 'IDENTIFIER'
    OPERATOR = 'OPERATOR'
    COMPARISON = 'COMPARISON'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    COMMA = 'COMMA'
    FUNCTION = 'FUNCTION'
    EOF = 'EOF'

    FUNCTIONS = {'pow', 'sqrt', 'exp', 'log', 'abs', 'min', 'max', 'if'}

    def __init__(self, resolver: Callable[[str], Any]):
        """
        Args:
            resolver: Function that resolves variable names to values.
                      Takes a string like "table.column" and returns the value.
        """
        self.resolver = resolver
        self.text = ""
        self.pos = 0
        self.current_token = None
        self.default_table = None

    def evaluate(self, expression: str) -> Any:
        """Evaluate an expression string and return the result."""
        self.text = expression.strip()
        self.pos = 0
        self.default_table = None
        self.current_token = self._next_token()

        if self.current_token[0] == self.EOF:
            return None

        result = self._parse_expression()

        if self.current_token[0] != self.EOF:
            raise ValueError(f"Unexpected token: {self.current_token[1]}")

        return result

    def _next_token(self):
        """Get the next token from the input."""
        # Skip whitespace
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

        if self.pos >= len(self.text):
            return (self.EOF, None)

        char = self.text[self.pos]

        # Number (including decimals)
        if char.isdigit() or (char == '.' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
            return self._read_number()

        # Comparison operators (must check before single-char operators)
        if char in '<>=!' and self.pos + 1 < len(self.text):
            two_char = self.text[self.pos:self.pos + 2]
            if two_char in ('<=', '>=', '==', '!='):
                self.pos += 2
                return (self.COMPARISON, two_char)
            if char in '<>':
                self.pos += 1
                return (self.COMPARISON, char)

        # Single character tokens
        if char == '(':
            self.pos += 1
            return (self.LPAREN, '(')
        if char == ')':
            self.pos += 1
            return (self.RPAREN, ')')
        if char == ',':
            self.pos += 1
            return (self.COMMA, ',')
        if char in '+-*/^':
            self.pos += 1
            return (self.OPERATOR, char)

        # Identifier (variable name or function)
        if char.isalpha() or char == '_':
            return self._read_identifier()

        raise ValueError(f"Unexpected character: {char}")

    def _read_number(self):
        """Read a number token."""
        start = self.pos
        has_dot = False

        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break

        return (self.NUMBER, float(self.text[start:self.pos]))

    def _read_identifier(self):
        """Read an identifier (variable or function name)."""
        start = self.pos

        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isalnum() or char in '_.':
                self.pos += 1
            else:
                break

        name = self.text[start:self.pos]

        # Check if it's a function (followed by parenthesis)
        # Look ahead for '('
        save_pos = self.pos
        while save_pos < len(self.text) and self.text[save_pos].isspace():
            save_pos += 1

        if save_pos < len(self.text) and self.text[save_pos] == '(' and name.lower() in self.FUNCTIONS:
            return (self.FUNCTION, name.lower())

        return (self.IDENTIFIER, name)

    def _parse_expression(self):
        """Parse an expression (lowest precedence: comparison)."""
        left = self._parse_additive()

        while self.current_token[0] == self.COMPARISON:
            op = self.current_token[1]
            self.current_token = self._next_token()
            right = self._parse_additive()
            left = self._apply_comparison(op, left, right)

        return left

    def _parse_additive(self):
        """Parse addition and subtraction."""
        left = self._parse_multiplicative()

        while self.current_token[0] == self.OPERATOR and self.current_token[1] in '+-':
            op = self.current_token[1]
            self.current_token = self._next_token()
            right = self._parse_multiplicative()

            if left is None:
                left = 0
            if right is None:
                right = 0

            if op == '+':
                left = float(left) + float(right)
            else:
                left = float(left) - float(right)

        return left

    def _parse_multiplicative(self):
        """Parse multiplication and division."""
        left = self._parse_power()

        while self.current_token[0] == self.OPERATOR and self.current_token[1] in '*/':
            op = self.current_token[1]
            self.current_token = self._next_token()
            right = self._parse_power()

            if left is None or right is None:
                return None

            if op == '*':
                left = float(left) * float(right)
            else:
                if float(right) == 0:
                    return None
                left = float(left) / float(right)

        return left

    def _parse_power(self):
        """Parse exponentiation (right associative)."""
        base = self._parse_unary()

        if self.current_token[0] == self.OPERATOR and self.current_token[1] == '^':
            self.current_token = self._next_token()
            exp = self._parse_power()  # Right associative

            if base is None or exp is None:
                return None

            return math.pow(float(base), float(exp))

        return base

    def _parse_unary(self):
        """Parse unary minus."""
        if self.current_token[0] == self.OPERATOR and self.current_token[1] == '-':
            self.current_token = self._next_token()
            val = self._parse_unary()
            return -float(val) if val is not None else None

        return self._parse_primary()

    def _parse_primary(self):
        """Parse primary expressions: numbers, variables, functions, parentheses."""
        token_type, token_value = self.current_token

        # Number literal
        if token_type == self.NUMBER:
            self.current_token = self._next_token()
            return token_value

        # Function call
        if token_type == self.FUNCTION:
            return self._parse_function_call(token_value)

        # Parenthesized expression
        if token_type == self.LPAREN:
            self.current_token = self._next_token()
            result = self._parse_expression()
            if self.current_token[0] != self.RPAREN:
                raise ValueError("Expected closing parenthesis")
            self.current_token = self._next_token()
            return result

        # Variable/identifier
        if token_type == self.IDENTIFIER:
            return self._resolve_variable(token_value)

        if token_type == self.EOF:
            return None

        raise ValueError(f"Unexpected token: {token_value}")

    def _parse_function_call(self, func_name: str):
        """Parse a function call like pow(x, y) or if(cond, a, b)."""
        self.current_token = self._next_token()  # consume function name

        if self.current_token[0] != self.LPAREN:
            raise ValueError(f"Expected '(' after function {func_name}")
        self.current_token = self._next_token()  # consume '('

        # Parse arguments
        args = []
        if self.current_token[0] != self.RPAREN:
            args.append(self._parse_expression())

            while self.current_token[0] == self.COMMA:
                self.current_token = self._next_token()  # consume ','
                args.append(self._parse_expression())

        if self.current_token[0] != self.RPAREN:
            raise ValueError(f"Expected ')' after function arguments")
        self.current_token = self._next_token()  # consume ')'

        return self._apply_function(func_name, args)

    def _apply_function(self, func_name: str, args: list):
        """Apply a function to its arguments."""
        if func_name == 'if':
            if len(args) != 3:
                raise ValueError("if() requires 3 arguments: if(condition, true_val, false_val)")
            condition, true_val, false_val = args
            return true_val if condition else false_val

        if func_name == 'pow':
            if len(args) != 2:
                raise ValueError("pow() requires 2 arguments")
            if args[0] is None or args[1] is None:
                return None
            return math.pow(float(args[0]), float(args[1]))

        if func_name == 'sqrt':
            if len(args) != 1:
                raise ValueError("sqrt() requires 1 argument")
            if args[0] is None or args[0] < 0:
                return None
            return math.sqrt(float(args[0]))

        if func_name == 'exp':
            if len(args) != 1:
                raise ValueError("exp() requires 1 argument")
            if args[0] is None:
                return None
            return math.exp(float(args[0]))

        if func_name == 'log':
            if len(args) != 1:
                raise ValueError("log() requires 1 argument")
            if args[0] is None or args[0] <= 0:
                return None
            return math.log(float(args[0]))

        if func_name == 'abs':
            if len(args) != 1:
                raise ValueError("abs() requires 1 argument")
            if args[0] is None:
                return None
            return abs(float(args[0]))

        if func_name == 'min':
            if len(args) < 2:
                raise ValueError("min() requires at least 2 arguments")
            valid_args = [a for a in args if a is not None]
            return min(valid_args) if valid_args else None

        if func_name == 'max':
            if len(args) < 2:
                raise ValueError("max() requires at least 2 arguments")
            valid_args = [a for a in args if a is not None]
            return max(valid_args) if valid_args else None

        raise ValueError(f"Unknown function: {func_name}")

    def _apply_comparison(self, op: str, left, right):
        """Apply a comparison operator."""
        if left is None or right is None:
            return False

        left, right = float(left), float(right)

        if op == '<':
            return left < right
        if op == '>':
            return left > right
        if op == '<=':
            return left <= right
        if op == '>=':
            return left >= right
        if op == '==':
            return left == right
        if op == '!=':
            return left != right

        raise ValueError(f"Unknown comparison operator: {op}")

    def _resolve_variable(self, name: str):
        """Resolve a variable name to its value."""
        self.current_token = self._next_token()

        # Handle table.column syntax
        if '.' in name:
            table, column = name.split('.', 1)
            self.default_table = table  # Set as default for subsequent variables
            return self.resolver(name)
        elif self.default_table:
            # Use inherited table
            return self.resolver(f"{self.default_table}.{name}")
        else:
            # Direct property
            return self.resolver(name)


# =============================================================================
# MarkdownTable - Advanced Table Rendering
# =============================================================================

@dataclass
class Cell:
    """Represents a single cell in the table."""
    content: str = ""
    span: int = 1  # Number of columns this cell spans
    alignment: Optional[str] = None  # 'left', 'right', 'center', or None (use column default)


@dataclass
class TableRow:
    """Represents a row in the table."""
    cells: List[Cell] = field(default_factory=list)
    is_divider: bool = False  # True for |====| divider rows
    is_separator: bool = False  # True for |---| separator rows


class MarkdownTable:
    """
    Advanced Markdown table renderer with support for:

    - Cell merging: `2*` prefix merges cell with next column, `3*` merges 2 more, etc.
    - Auto-merge: Missing cells at row end are automatically merged into last cell
    - Dividers: `|====|` creates a thick divider row
    - Separators: `|---|---|` creates column separator (header separator or top border)
    - Cell alignments: `|: cell |` left, `| cell :|` right, `|: cell :|` center
    - Column alignments: Set in first row (headerless) or separator row (with header)
    - Grid locking: Column count locked to first row's definition

    Example:
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | {prop1}  |2*: {prop2} spans 2 :|
        |====|
        |3*: This cell spans all 3 columns :|
    """

    def __init__(self, interpolate_fn: Callable[[str], str]):
        """
        Initialize with an interpolation function.

        Args:
            interpolate_fn: Function that interpolates {variables} in text
        """
        self.interpolate = interpolate_fn
        self.num_cols: int = 0
        self.col_alignments: List[str] = []  # Column default alignments
        self.col_widths: List[int] = []  # Column widths (auto-calculated)
        self.rows: List[TableRow] = []
        self.has_header: bool = False
        self.header_row: Optional[TableRow] = None

    def parse(self, lines: List[str]) -> "MarkdownTable":
        """Parse table lines and populate internal structures."""
        if not lines:
            return self

        # Detect if table has header (look for |---| separator, not |====|)
        self.has_header = any(
            self._is_separator_line(line) and not self._is_divider_line(line)
            for line in lines
        )

        # Parse first row to establish grid (column count)
        first_line = lines[0]
        if self._is_separator_line(first_line):
            # Headerless table with top separator
            self.num_cols = self._count_columns(first_line)
            self.col_alignments = self._parse_alignments_from_separator(first_line)
            start_idx = 1
        else:
            # Parse first data row for column count
            first_row = self._parse_row(first_line, establish_grid=True)
            self.num_cols = self._count_logical_columns(first_row)

            if self.has_header:
                # Header row
                self.header_row = first_row
                start_idx = 1
            else:
                # First row is data, parse alignments from it
                self.col_alignments = self._extract_alignments_from_row(first_line)
                self.rows.append(first_row)
                start_idx = 1

        # Parse remaining lines
        header_sep_found = False
        for line in lines[start_idx:]:
            if self._is_divider_line(line):
                # |====| thick divider
                self.rows.append(TableRow(is_divider=True))
            elif self._is_separator_line(line):
                # |---| separator
                if self.has_header and not header_sep_found:
                    # This is the header separator - parse alignments, don't add to rows
                    self.col_alignments = self._parse_alignments_from_separator(line)
                    header_sep_found = True
                else:
                    # Additional separator row (should be rare)
                    self.rows.append(TableRow(is_separator=True))
            else:
                # Data row
                row = self._parse_row(line)
                self.rows.append(row)

        # Ensure we have default alignments
        if not self.col_alignments or len(self.col_alignments) < self.num_cols:
            self.col_alignments = self._default_alignments(self.num_cols)

        return self

    def render(self) -> List[str]:
        """Render the parsed table to output lines."""
        if self.num_cols == 0:
            return []

        # First pass: interpolate all cell content
        self._interpolate_cells()

        # Check if table has any data
        if not self._has_data():
            return []

        # Calculate column widths
        self._calculate_widths()

        # Render output
        result = [""]  # Leading empty line

        # Header row (if present)
        if self.header_row:
            result.append(self._render_row(self.header_row))
            # Header separator
            result.append(self._render_separator())

        # Data rows
        for row in self.rows:
            if row.is_divider:
                result.append(self._render_divider())
            elif row.is_separator:
                result.append(self._render_separator())
            else:
                rendered = self._render_row(row)
                if rendered:  # Skip empty rows
                    result.append(rendered)

        return result

    def _is_separator_line(self, line: str) -> bool:
        """Check if line is a separator (contains ---)."""
        return "---" in line

    def _is_divider_line(self, line: str) -> bool:
        """Check if line is a thick divider (contains ===)."""
        return "===" in line

    def _count_columns(self, line: str) -> int:
        """Count columns from a separator/divider line."""
        parts = line.split("|")
        return len([p for p in parts[1:-1] if p.strip()])

    def _count_logical_columns(self, row: TableRow) -> int:
        """Count logical columns accounting for spans."""
        total = 0
        for cell in row.cells:
            total += cell.span
        return total

    def _parse_row(self, line: str, establish_grid: bool = False) -> TableRow:
        """Parse a data row into cells."""
        row = TableRow()
        parts = line.split("|")[1:-1]  # Remove leading/trailing empty parts

        col_idx = 0
        for part in parts:
            cell = self._parse_cell(part)

            # Grid locking: if we've reached column limit, auto-merge
            if not establish_grid and self.num_cols > 0:
                remaining_cols = self.num_cols - col_idx
                if remaining_cols <= 0:
                    break
                # Adjust span if it would exceed grid
                if col_idx + cell.span > self.num_cols:
                    cell.span = remaining_cols

            row.cells.append(cell)
            col_idx += cell.span

        # Auto-merge: if row has fewer columns than grid, extend last cell
        if not establish_grid and self.num_cols > 0 and row.cells:
            current_span = sum(c.span for c in row.cells)
            if current_span < self.num_cols:
                row.cells[-1].span += (self.num_cols - current_span)

        return row

    def _parse_cell(self, text: str) -> Cell:
        """Parse a single cell, extracting span and alignment."""
        # Check for alignment markers BEFORE stripping
        # Colon must be directly adjacent to the pipe (at start/end of raw text)
        left_colon = text.startswith(":")
        right_colon = text.endswith(":")

        text = text.strip()
        cell = Cell()

        # Check for span prefix: 2*, 3*, etc.
        span_match = re.match(r'^(\d+)\*(.*)$', text)
        if span_match:
            cell.span = int(span_match.group(1))
            remainder = span_match.group(2)
            # Re-check alignment on remainder (for cases like |2*:centered:|)
            left_colon = remainder.startswith(":")
            right_colon = remainder.endswith(":")
            text = remainder.strip()

        if left_colon and right_colon:
            cell.alignment = "center"
            text = text[1:-1].strip()
        elif left_colon:
            cell.alignment = "left"
            text = text[1:].strip()
        elif right_colon:
            cell.alignment = "right"
            text = text[:-1].strip()

        cell.content = text
        return cell

    def _extract_alignments_from_row(self, line: str) -> List[str]:
        """Extract column alignments from a data row."""
        alignments = []
        parts = line.split("|")[1:-1]

        for part in parts:
            # Check for alignment markers BEFORE stripping
            # Colon must be directly adjacent to the pipe
            left_colon = part.startswith(":")
            right_colon = part.endswith(":")

            part = part.strip()
            # Check for span prefix
            span_match = re.match(r'^(\d+)\*(.*)$', part)
            if span_match:
                span = int(span_match.group(1))
                remainder = span_match.group(2)
                # Re-check alignment on remainder
                left_colon = remainder.startswith(":")
                right_colon = remainder.endswith(":")
            else:
                span = 1

            # Detect alignment
            if left_colon and right_colon:
                align = "center"
            elif left_colon:
                align = "left"
            elif right_colon:
                align = "right"
            else:
                align = "default"

            # Add alignment for each spanned column
            for _ in range(span):
                alignments.append(align)

        # Pad to num_cols
        while len(alignments) < self.num_cols:
            alignments.append("default")

        return alignments

    def _parse_alignments_from_separator(self, line: str) -> List[str]:
        """Parse alignments from a separator line like |:---|---:|."""
        alignments = []
        parts = line.split("|")[1:-1]

        for part in parts:
            part = part.strip()
            left_colon = part.startswith(":")
            right_colon = part.endswith(":")

            if left_colon and right_colon:
                alignments.append("center")
            elif left_colon:
                alignments.append("left")
            elif right_colon:
                alignments.append("right")
            else:
                alignments.append("default")

        return alignments

    def _default_alignments(self, num_cols: int) -> List[str]:
        """Return default alignments (first col left, rest right)."""
        return ["left"] + ["right"] * (num_cols - 1) if num_cols > 0 else []

    def _interpolate_cells(self):
        """Interpolate all cell content."""
        if self.header_row:
            for cell in self.header_row.cells:
                cell.content = self.interpolate(cell.content).strip()

        for row in self.rows:
            if not row.is_divider and not row.is_separator:
                for cell in row.cells:
                    cell.content = self.interpolate(cell.content).strip()

    def _has_data(self) -> bool:
        """Check if table has any meaningful data."""
        def is_empty(val: str) -> bool:
            return not val or val in ("", "0.0", "0")

        # Check header
        if self.header_row:
            if any(not is_empty(c.content) for c in self.header_row.cells):
                return True

        # Check data rows
        for row in self.rows:
            if not row.is_divider and not row.is_separator:
                if any(not is_empty(c.content) for c in row.cells):
                    return True

        return False

    def _calculate_widths(self):
        """Calculate column widths based on content."""
        self.col_widths = [0] * self.num_cols

        # From header
        if self.header_row:
            self._update_widths_from_row(self.header_row)

        # From data rows
        for row in self.rows:
            if not row.is_divider and not row.is_separator:
                self._update_widths_from_row(row)

    def _update_widths_from_row(self, row: TableRow):
        """Update column widths based on row content."""
        col_idx = 0
        for cell in row.cells:
            if cell.span == 1:
                # Simple case: single column
                if col_idx < self.num_cols:
                    self.col_widths[col_idx] = max(
                        self.col_widths[col_idx],
                        len(cell.content)
                    )
            else:
                # Spanning cell: distribute width across columns
                # For width calculation, we don't expand columns for spanning cells
                # The cell will use the combined width of spanned columns
                pass
            col_idx += cell.span

    def _get_effective_alignment(self, cell: Cell, col_idx: int) -> str:
        """Get effective alignment for a cell (cell-specific or column default)."""
        if cell.alignment:
            return cell.alignment
        if col_idx < len(self.col_alignments):
            align = self.col_alignments[col_idx]
            if align == "default":
                return "left" if col_idx == 0 else "right"
            return align
        return "left" if col_idx == 0 else "right"

    def _format_cell(self, content: str, width: int, alignment: str) -> str:
        """Format cell content with alignment and padding."""
        if alignment == "left":
            return f" {content:<{width}} "
        elif alignment == "right":
            return f" {content:>{width}} "
        elif alignment == "center":
            return f" {content:^{width}} "
        return f" {content:<{width}} "

    def _render_row(self, row: TableRow) -> str:
        """Render a single row."""
        parts = []
        col_idx = 0

        for cell in row.cells:
            # Calculate width for this cell (sum of spanned columns + padding)
            if cell.span == 1:
                width = self.col_widths[col_idx] if col_idx < len(self.col_widths) else 0
            else:
                # Spanning cell: sum widths of spanned columns + internal padding
                end_col = min(col_idx + cell.span, len(self.col_widths))
                width = sum(self.col_widths[col_idx:end_col])
                # Add padding for internal column boundaries (2 spaces per boundary)
                width += 2 * (cell.span - 1)

            alignment = self._get_effective_alignment(cell, col_idx)
            parts.append(self._format_cell(cell.content, width, alignment))
            col_idx += cell.span

        # Check if row has meaningful content
        if all(not p.strip() or p.strip() in ("0.0", "0") for p in parts):
            # Skip rows with only empty/zero values (except first column)
            if len(parts) > 1:
                has_label = parts[0].strip() and parts[0].strip() not in ("0.0", "0")
                has_values = any(
                    p.strip() and p.strip() not in ("0.0", "0")
                    for p in parts[1:]
                )
                if has_label and not has_values:
                    return ""

        return "".join(parts)

    def _render_separator(self) -> str:
        """Render a separator line (---)."""
        total_width = sum(self.col_widths) + 2 * self.num_cols
        return "-" * total_width

    def _render_divider(self) -> str:
        """Render a thick divider line (===)."""
        total_width = sum(self.col_widths) + 2 * self.num_cols
        return "=" * total_width


# =============================================================================
# Template Definitions
# =============================================================================

FIELD_TEMPLATE = """
# FIELD: {name}
===
Status:     {status:<20}  Operator:  {operator}
HC Type:    {hc_type:<20}  Main Area: {main_area}
?{discovery_year} Discovered: {discovery_year}

| Volumes      | In-place   | Recoverable | Remaining  |
|--------------|------------|-------------|------------|
| Oil (mSm3)   | {field_reserves.fldInplaceOil} | {field_reserves.fldRecoverableOil} | {field_reserves.fldRemainingOil} |
| Gas (bSm3)   | {field_reserves.fldInplaceFreeGas+fldInplaceAssGas} | {field_reserves.fldRecoverableGas} | {field_reserves.fldRemainingGas} |
| NGL (mtoe)   | | {field_reserves.fldRecoverableNGL} | {field_reserves.fldRemainingNGL} |
| Cond (mSm3)  | | {field_reserves.fldRecoverableCondensate} | {field_reserves.fldRemainingCondensate} |

@partners:field_licensee_hst.fldNpdidField=id|cmpLongName|fldCompanyShare|fldLicenseeTo
"""


DISCOVERY_TEMPLATE = """
# Discovery: {name}
===
Discovered: {discovery_year:<12}  HC Type: {hc_type}
Status:     {status}
Area:       {main_area}
?{discovery_well} Discovery Well: {discovery_well}
Operator:   {operator}

?{field_name} -> Developed as field: {field_name}

| Recoverable       | Value      |
|-------------------|------------|
| Oil (mSm3)        | {discovery_reserves.dscRecoverableOil} |
| Gas (bSm3)        | {discovery_reserves.dscRecoverableGas} |
| Cond (mSm3)       | {discovery_reserves.dscRecoverableCondensate} |
| NGL (mtoe)        | {discovery_reserves.dscRecoverableNGL} |

Explore: .wells  .resources_history()  .discovery_well
"""


WELLBORE_TEMPLATE = """
# Wellbore: {name}
===
Purpose:    {purpose:<12}  Content: {content}
Status:     {status:<12}  Area:    {main_area}
?{total_depth} Depth: TD: {total_depth:.0f}m, WD: {water_depth:.0f}m
?{completion_date} Completed: {completion_date}
Operator:   {operator}

?{hc_formations_str} HC Formations: {hc_formations_str}
?{hc_ages_str} HC Ages: {hc_ages_str}

?{field_name} Field:     {field_name}
?{discovery_name} Discovery: {discovery_name}

Explore: .formation_tops  .dst_results  .cores  .drilling_history  .casing
"""


COMPANY_TEMPLATE = """
# Company: {name}
===
Nation:     {nation:<12}  Org#: {org_number}

@field_interests
"""


LICENSE_TEMPLATE = """
# License: {name}
===
Status:     {status:<12}  Phase: {current_phase}
Area:       {main_area}
Operator:   {operator}
?{date_granted} Granted:    {date_granted}
?{date_valid_to} Expires:    {date_valid_to}

@partners:licence_licensee_hst.prlNpdidLicence=id|cmpLongName|prlLicenseeInterest|prlLicenseeDateValidTo|Licensees

Explore: .licensees  .fields  .discoveries  .wells
         .ownership_history  .phase_history  .work_obligations
"""


FACILITY_TEMPLATE = """
# Facility: {name}
===
Kind:       {kind:<15}  Functions: {functions}
Phase:      {phase:<15}  Status:    {status}
?{water_depth} Water Depth: {water_depth:.0f}m
?{startup_date} Startup:    {startup_date}
?{field_name} Field:      {field_name}

Explore: .facility_function  .related('field')
"""


PIPELINE_TEMPLATE = """
# Pipeline: {name}
===
Medium:     {medium:<15}  Dimension: {dimension}"
Status:     {status}
Area:       {main_area}
From:       {from_facility}
To:         {to_facility}
?{operator} Operator:   {operator}

Explore: .related('facility')
"""


PLAY_TEMPLATE = """
# Play: {name}
===
Status:     {status}
Area:       {main_area}

Explore: .related('discovery')  .related('wellbore')
"""


BLOCK_TEMPLATE = """
# Block: {name}
===
Quadrant:   {quadrant}
Area:       {main_area}
Status:     {status}

Explore: .related('licence')  .related('wellbore')
"""


QUADRANT_TEMPLATE = """
# Quadrant: {name}
===
Area:       {main_area}

Explore: .related('block')  .related('licence')
"""


TUF_TEMPLATE = """
# TUF: {name}
===
Kind:       {kind}
Status:     {status}
?{startup_date} Startup:    {startup_date}

Explore: .operators  .owners
"""


SEISMIC_TEMPLATE = """
# Seismic Survey: {name}
===
Type:       {survey_type:<12}  Status: {status}
Area:       {main_area}
?{company} Company:    {company}
?{start_date} Started:    {start_date}
?{end_date} Completed:  {end_date}
?{planned_total_km} Planned km: {planned_total_km:.0f}

Explore: .related('seismic_acquisition_progress')
"""


STRATIGRAPHY_TEMPLATE = """
# Stratigraphy: {name}
===
Type:       {strat_type}
Level:      {level}
?{parent} Parent:     {parent}

Explore: .related('strat_litho_wellbore')
"""


BUSINESS_ARRANGEMENT_TEMPLATE = """
# Business Arrangement: {name}
===
Kind:       {kind}
Status:     {status}
?{date_approved} Approved:   {date_approved}
?{operator} Operator:   {operator}

Explore: .licensees  .related('business_arrangement_operator')
"""


# Template registry
TEMPLATES = {
    "field": FIELD_TEMPLATE,
    "discovery": DISCOVERY_TEMPLATE,
    "wellbore": WELLBORE_TEMPLATE,
    "company": COMPANY_TEMPLATE,
    "license": LICENSE_TEMPLATE,
    "facility": FACILITY_TEMPLATE,
    "pipeline": PIPELINE_TEMPLATE,
    "play": PLAY_TEMPLATE,
    "block": BLOCK_TEMPLATE,
    "quadrant": QUADRANT_TEMPLATE,
    "tuf": TUF_TEMPLATE,
    "seismic": SEISMIC_TEMPLATE,
    "stratigraphy": STRATIGRAPHY_TEMPLATE,
    "business_arrangement": BUSINESS_ARRANGEMENT_TEMPLATE,
}


# =============================================================================
# Special Blocks Configuration
# =============================================================================

# Partners block config: table.match_col=entity_key|company_col|share_col|date_col|title
PARTNERS_DEFAULTS = {
    "field": ("field_licensee_hst", "fldNpdidField", "id", "cmpLongName", "fldCompanyShare", "fldLicenseeTo", "Partners"),
    "license": ("licence_licensee_hst", "prlNpdidLicence", "id", "cmpLongName", "prlLicenseeInterest", "prlLicenseeDateValidTo", "Licensees"),
}


# =============================================================================
# Template Renderer
# =============================================================================

class TemplateRenderer:
    """Renders markdown-like templates with entity data."""

    # Match keys for different entity types
    MATCH_KEYS = {
        "field_reserves": ("fldNpdidField", "id"),
        "field_licensee_hst": ("fldNpdidField", "id"),
        "discovery_reserves": ("dscNpdidDiscovery", "id"),
        "licence_licensee_hst": ("prlNpdidLicence", "id"),
    }

    def __init__(self, entity: Any, db: Any, entity_type: str):
        self.entity = entity
        self.db = db
        self.entity_type = entity_type
        self._related_cache: Dict[str, pd.DataFrame] = {}

    def render(self, template: str) -> str:
        """Render a template to string."""
        lines = []
        in_table = False
        table_lines = []

        for line in template.strip().split("\n"):
            line = line.rstrip()

            # Skip empty lines at start
            if not lines and not line:
                continue

            # Handle table accumulation
            if line.startswith("|"):
                in_table = True
                table_lines.append(line)
                continue
            elif in_table:
                # End of table - render it
                rendered_table = self._render_table(table_lines)
                if rendered_table:
                    lines.extend(rendered_table)
                table_lines = []
                in_table = False

            # Process non-table lines
            rendered = self._render_line(line)
            if rendered is not None:
                lines.append(rendered)

        # Handle trailing table
        if table_lines:
            rendered_table = self._render_table(table_lines)
            if rendered_table:
                lines.extend(rendered_table)

        return "\n".join(lines)

    def _render_line(self, line: str) -> Optional[str]:
        """Render a single line."""
        # Header: # Title
        if line.startswith("# "):
            title = self._interpolate(line[2:])
            return title

        # Major divider: ===
        if line.strip() == "===":
            return "=" * 60

        # Minor divider: ---
        if line.strip() == "---":
            return "-" * 60

        # Conditional line: ?{condition} text
        if line.startswith("?{"):
            match = re.match(r'\?\{([^}]+)\}\s*(.*)', line)
            if match:
                condition = match.group(1)
                rest = match.group(2)
                if not self._get_value(condition):
                    return None
                return self._interpolate(rest)
            return None

        # Special block: @partners or @partners:config
        if line.startswith("@"):
            return self._render_special_block(line[1:])

        # Regular line with interpolation
        return self._interpolate(line)

    def _render_table(self, lines: List[str]) -> List[str]:
        """Render a markdown table using the MarkdownTable class.

        Supports:
        - Tables with headers: | Header | / |---| / | data |
        - Headerless tables: | data | / | data |
        - Cell merging: 2* prefix merges cell with next column
        - Dividers: |====| creates thick divider
        - Cell alignments: |: left |, | right :|, |: center :|
        - Grid locking: Column count locked to first row
        - Auto-merge: Missing cells merged into last cell
        """
        if not lines:
            return []

        table = MarkdownTable(self._interpolate)
        table.parse(lines)
        return table.render()

    def _render_special_block(self, block_spec: str) -> Optional[str]:
        """Render a special block like @partners."""
        # Parse block spec: name:config or just name
        if ":" in block_spec:
            block_name, config = block_spec.split(":", 1)
        else:
            block_name = block_spec
            config = None

        if block_name == "partners":
            return self._render_partners_block(config)
        elif block_name == "field_interests":
            return self._render_field_interests()

        return None

    def _render_partners_block(self, config: Optional[str]) -> str:
        """Render a partners list block."""
        # Parse config: table.match_col=entity_key|company_col|share_col|date_col|title
        if config:
            parts = config.split("|")
            table_match = parts[0]  # e.g., "field_licensee_hst.fldNpdidField=id"
            table_part, match_part = table_match.split(".")
            match_col, entity_key = match_part.split("=")
            company_col = parts[1] if len(parts) > 1 else "cmpLongName"
            share_col = parts[2] if len(parts) > 2 else "share"
            date_col = parts[3] if len(parts) > 3 else None
            title = parts[4] if len(parts) > 4 else "Partners"
        else:
            # Use defaults based on entity type
            if self.entity_type in PARTNERS_DEFAULTS:
                table_part, match_col, entity_key, company_col, share_col, date_col, title = PARTNERS_DEFAULTS[self.entity_type]
            else:
                return ""

        # Get related data
        df = self._get_related_table(table_part, match_col, entity_key)
        if df is None or df.empty:
            return ""

        # Filter to current partners (handle both Unix timestamps and date strings)
        if date_col and date_col in df.columns:
            col_dtype = df[date_col].dtype
            if col_dtype in ['float64', 'int64']:
                # Unix timestamp in milliseconds
                today_ms = datetime.now().timestamp() * 1000
                df = df[df[date_col].isna() | (df[date_col] > today_ms)]
            else:
                # String date format
                today = datetime.now().strftime('%Y-%m-%d')
                df = df[df[date_col].isna() | (df[date_col] >= today)]

        if df.empty:
            return ""

        # Get operator name
        operator = self._get_value("operator")

        # Build partner list
        partner_list = []
        for _, row in df.iterrows():
            company = row.get(company_col, "")
            share = float(row.get(share_col, 0) or 0)
            is_op = company == operator
            partner_list.append({"company": company, "share": share, "is_operator": is_op})

        # Sort by share
        partner_list = sorted(partner_list, key=lambda x: x["share"], reverse=True)

        lines = [f"\n{title} ({len(partner_list)}):"]
        for p in partner_list[:5]:
            op_mark = " *" if p["is_operator"] else ""
            lines.append(f"  {p['company']:<40} {p['share']:>6.2f}%{op_mark}")

        if len(partner_list) > 5:
            lines.append(f"  ... and {len(partner_list) - 5} more")

        return "\n".join(lines)

    def _render_field_interests(self) -> str:
        """Render field interests for a company."""
        # This needs special handling - get from entity method
        if not hasattr(self.entity, 'field_interests'):
            return ""

        interests = self.entity.field_interests
        if not interests:
            return ""

        lines = ["Top equity positions (* = operator):"]
        for i in interests[:5]:
            op_mark = "*" if i.get('is_operator') else " "
            lines.append(f"  {op_mark} {i['field']:<28} {i['share']:>6.2f}%")

        if len(interests) > 5:
            lines.append(f"  ... and {len(interests) - 5} more")

        lines.append("")
        lines.append("Explore: .field_interests  .operated_fields  .wells_drilled")

        return "\n".join(lines)

    def _interpolate(self, text: str) -> str:
        """Interpolate {variables} in text."""
        def replacer(match):
            expr = match.group(1)

            # Parse format spec
            if ":" in expr and not expr.startswith(":"):
                var_part, fmt_part = expr.rsplit(":", 1)
            else:
                var_part = expr
                fmt_part = None

            # Get value
            value = self._resolve_value(var_part)

            # Handle None/empty
            if value is None:
                return ""

            # Apply format
            if fmt_part:
                try:
                    # Handle alignment specs like <20, >10
                    if fmt_part[0] in "<>^" and fmt_part[1:].isdigit():
                        return f"{value:{fmt_part}}"
                    else:
                        return f"{value:{fmt_part}}"
                except (ValueError, TypeError):
                    return str(value) if value else ""
            else:
                if isinstance(value, float):
                    if value == 0:
                        return ""
                    return f"{value:,.1f}"
                return str(value) if value else ""

        return re.sub(r'\{([^}]+)\}', replacer, text)

    def _resolve_value(self, var_expr: str) -> Any:
        """Resolve a variable expression with full arithmetic support.

        Supports:
            {property}                          - Entity property
            {table.column}                      - Related table value
            {table.col1 + col2 + col3}          - Arithmetic with inherited table
            {table.col1 * col2 / col3}          - Full arithmetic: +, -, *, /, ^
            {pow(table.col1, 2)}                - Functions: pow, sqrt, exp, log, abs, min, max
            {if(table.val > 100, table.val, 0)} - Conditionals with comparisons
        """
        # Check if expression needs the full parser (has operators or functions)
        has_operators = any(op in var_expr for op in ['+', '-', '*', '/', '^', '<', '>', '=', '!'])
        has_functions = any(f + '(' in var_expr for f in ExpressionParser.FUNCTIONS)

        if has_operators or has_functions:
            # Use full expression parser
            parser = ExpressionParser(self._resolve_simple_value)
            try:
                return parser.evaluate(var_expr)
            except (ValueError, TypeError):
                return None

        # Simple case: just a variable reference
        return self._resolve_simple_value(var_expr)

    def _resolve_simple_value(self, var_expr: str) -> Any:
        """Resolve a simple variable (no operators)."""
        # Check for table.column syntax
        if "." in var_expr:
            table_name, col_name = var_expr.split(".", 1)
            return self._get_table_column_value(table_name, col_name)

        # Direct entity property
        return self._get_value(var_expr)

    def _get_table_column_value(self, table_name: str, col_name: str) -> Any:
        """Get a single column value from a related table."""
        # Get related table data
        if table_name in self.MATCH_KEYS:
            match_col, entity_key = self.MATCH_KEYS[table_name]
        else:
            match_col = None
            entity_key = "id"

        df = self._get_related_table(table_name, match_col, entity_key)
        if df is None or df.empty:
            return None

        row = df.iloc[0]
        val = row.get(col_name)
        if pd.notna(val):
            return float(val) if isinstance(val, (int, float)) else val
        return None

    def _get_value(self, property_name: str) -> Any:
        """Get a value from the entity."""
        # Handle computed properties
        if property_name == "hc_formations_str":
            formations = getattr(self.entity, "hc_formations", [])
            return ", ".join(formations) if formations else None
        if property_name == "hc_ages_str":
            ages = getattr(self.entity, "hc_ages", [])
            return ", ".join(ages) if ages else None

        # Direct property access
        if hasattr(self.entity, property_name):
            val = getattr(self.entity, property_name)
            return val

        # Try _data dict
        if hasattr(self.entity, "_data"):
            return self.entity._data.get(property_name)

        return None

    def _get_related_table(self, table_name: str, match_col: Optional[str], entity_key: str) -> Optional[pd.DataFrame]:
        """Get filtered related table data."""
        cache_key = f"{table_name}_{entity_key}"

        if cache_key in self._related_cache:
            return self._related_cache[cache_key]

        df = self.db.get_or_none(table_name)
        if df is None:
            return None

        entity_value = self._get_value(entity_key)
        if entity_value is None:
            return None

        if match_col and match_col in df.columns:
            filtered = df[df[match_col] == entity_value]
        else:
            filtered = df

        self._related_cache[cache_key] = filtered
        return filtered


def render_entity(entity: Any, entity_type: str) -> str:
    """
    Render an entity using its template.

    Uses custom template if one exists in the database, otherwise uses default.

    Args:
        entity: The entity object (Field, Discovery, etc.)
        entity_type: Type of entity ('field', 'discovery', etc.)

    Returns:
        Formatted string representation
    """
    if entity_type not in TEMPLATES:
        return f"<{entity.__class__.__name__}: {getattr(entity, 'name', 'unknown')}>"

    # Check for custom template in database
    custom_template = entity._db.get_template(entity_type)
    template = custom_template if custom_template is not None else TEMPLATES[entity_type]

    renderer = TemplateRenderer(entity, entity._db, entity_type)
    return renderer.render(template)
