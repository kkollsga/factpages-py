"""
Centralized Display Templates

Declarative markdown-like templates for entity printouts.
All display formats are defined here for easy modification.

Template Syntax:
    {property}              - Entity property (e.g., {name}, {status})
    {table.column}          - Related table value (e.g., {field_reserves.fldRecoverableOil})
    {table.col1+col2}       - Sum of columns
    {value:format}          - With format spec (e.g., {value:>10,.1f})
    {value:<20}             - Left-align with width 20

    # Title                 - Section header
    ===                     - Major divider (full width)
    ---                     - Minor divider

    | Col1 | Col2 |         - Table header
    |------|------|         - Table separator
    | {a}  | {b}  |         - Table row

    ?{condition} text       - Conditional line (only show if condition is truthy)
    @partners               - Special block (partners list, etc.)

Usage:
    from .display import render_entity

    class Field:
        def __str__(self):
            return render_entity(self, "field")
"""

import re
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd


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

@partners:field_licensee_hst.fldNpdidField=id|cmpLongName|fldLicenseeInterest|fldLicenseeDateValidTo
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
    "field": ("field_licensee_hst", "fldNpdidField", "id", "cmpLongName", "fldLicenseeInterest", "fldLicenseeDateValidTo", "Partners"),
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
        """Render a markdown table with proper alignment."""
        if len(lines) < 2:
            return []

        # Parse header - get column widths from header definition
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split("|")[1:-1]]

        # Calculate column widths from header cells (includes padding)
        raw_headers = header_line.split("|")[1:-1]
        col_widths = [len(h) for h in raw_headers]

        # Find separator line (contains ---)
        sep_idx = 1
        for i, line in enumerate(lines[1:], 1):
            if "---" in line:
                sep_idx = i
                break

        # Parse and render data rows
        data_rows = []
        has_data = False

        for line in lines[sep_idx + 1:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            rendered_cells = []

            for cell in cells:
                rendered = self._interpolate(cell)
                # Check if cell has actual value (not empty, not zero)
                if rendered and rendered.strip() and rendered.strip() not in ("0.0", "0"):
                    has_data = True
                rendered_cells.append(rendered.strip() if rendered else "")

            data_rows.append(rendered_cells)

        # Only render if there's data
        if not has_data:
            return []

        # Render table
        result = [""]

        # Header row - first col left-aligned, rest right-aligned
        header_parts = []
        for i, h in enumerate(headers):
            if i == 0:
                header_parts.append(f"{h:<{col_widths[i]}}")
            else:
                header_parts.append(f"{h:>{col_widths[i]}}")
        result.append("".join(header_parts))

        # Separator
        result.append("-" * sum(col_widths))

        # Data rows - only include rows with some data
        for row in data_rows:
            row_has_data = any(
                c and c not in ("", "0.0", "0")
                for c in row[1:]  # Skip label column
            )
            if row_has_data:
                parts = []
                for i in range(len(headers)):
                    val = row[i] if i < len(row) else ""
                    if i == 0:
                        parts.append(f"{val:<{col_widths[i]}}")
                    else:
                        parts.append(f"{val:>{col_widths[i]}}")
                result.append("".join(parts))

        return result

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

        # Filter to current partners
        today = datetime.now().strftime('%Y-%m-%d')
        if date_col and date_col in df.columns:
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
        """Resolve a variable expression like 'name' or 'field_reserves.fldRecoverableOil'."""
        # Check for table.column or table.col1+col2 syntax
        if "." in var_expr:
            parts = var_expr.split(".", 1)
            table_name = parts[0]
            col_expr = parts[1]

            # Get related table data
            if table_name in self.MATCH_KEYS:
                match_col, entity_key = self.MATCH_KEYS[table_name]
            else:
                # Try to infer from table name
                match_col = None
                entity_key = "id"

            df = self._get_related_table(table_name, match_col, entity_key)
            if df is None or df.empty:
                return None

            row = df.iloc[0]

            # Handle column expressions with +
            if "+" in col_expr:
                total = 0
                for col in col_expr.split("+"):
                    val = row.get(col.strip(), 0)
                    if pd.notna(val):
                        total += float(val)
                return total if total > 0 else None
            else:
                val = row.get(col_expr)
                if pd.notna(val):
                    return float(val) if isinstance(val, (int, float)) else val
                return None

        # Direct entity property
        return self._get_value(var_expr)

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
