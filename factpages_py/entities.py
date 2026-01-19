"""
Entity Classes

Object-oriented wrappers for petroleum data entities.
Provides a clean, intuitive API for accessing field, well, discovery data.

Example:
    >>> fp = Factpages()
    >>> troll = fp.field("troll")
    >>> print(troll.partners)
    >>> print(troll.production(2025, 8))
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict, Any

import pandas as pd

if TYPE_CHECKING:
    from .database import Database

# Module-level alias cache (loaded from columns.json)
_ALIAS_CACHE: Optional[Dict[str, str]] = None
_COLUMNS_FILE = Path(__file__).parent / "columns.json"


def _load_aliases() -> Dict[str, str]:
    """Load and merge all column aliases from the schema cache file."""
    global _ALIAS_CACHE

    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE

    _ALIAS_CACHE = {}
    if _COLUMNS_FILE.exists():
        try:
            with open(_COLUMNS_FILE) as f:
                data = json.load(f)
            # Merge all datasets' column aliases into one dict
            for dataset_columns in data.values():
                for col_name, col_info in dataset_columns.items():
                    _ALIAS_CACHE[col_name] = col_info.get('alias', col_name)
        except (json.JSONDecodeError, IOError):
            pass

    return _ALIAS_CACHE


def clear_alias_cache() -> None:
    """Clear the alias cache (useful after updating schema)."""
    global _ALIAS_CACHE
    _ALIAS_CACHE = None


# =============================================================================
# Display Wrapper Classes
# =============================================================================

class EntityData:
    """
    Wrapper for entity data that provides both formatted display and attribute access.

    When printed, shows all columns with truncated long values.
    Attribute access returns full values.

    Example:
        >>> print(troll.data)
        fldNpdidField: 46437
        fldName: TROLL
        _geometry: {"type": "Polygon", "coo...40640798473664]]]}

        >>> troll.data._geometry  # Full value
        '{"type": "Polygon", "coordinates": [[[4.172...]]}'
    """

    MAX_VALUE_LENGTH = 60  # Max display length for values

    def __init__(self, data: pd.Series):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to get full values."""
        # _data is handled by normal attribute lookup, not __getattr__
        if name in self._data.index:
            return self._data[name]
        raise AttributeError(f"No column '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: entity.data['column_name']"""
        return self._data[key]

    def _truncate(self, value: Any) -> str:
        """Truncate a value for display, keeping start and end visible."""
        s = str(value)
        if len(s) <= self.MAX_VALUE_LENGTH:
            return s
        # Keep some from start and end, with ... in middle
        keep = (self.MAX_VALUE_LENGTH - 3) // 2
        return s[:keep] + "..." + s[-keep:]

    def __str__(self) -> str:
        """Formatted string with truncated values."""
        lines = []
        for col in self._data.index:
            value = self._data[col]
            display_value = self._truncate(value)
            lines.append(f"{col}: {display_value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<EntityData: {len(self._data)} columns>"

    def keys(self) -> list:
        """Return column names."""
        return list(self._data.index)

    def values(self) -> list:
        """Return all values."""
        return list(self._data.values)

    def items(self):
        """Return (column, value) pairs."""
        return zip(self._data.index, self._data.values)


class RelatedData:
    """
    Wrapper for related DataFrame that provides .data and .df API.

    When printed, shows a nice overview of the related data.
    Provides attribute access to columns and DataFrame access.

    Example:
        >>> reserves = discovery.discovery_reserves
        >>> print(reserves.data)  # Nice formatted output
        >>> reserves.df           # Get as DataFrame
        >>> reserves.data.dscRecoverableOil  # Access column
    """

    MAX_VALUE_LENGTH = 60

    def __init__(self, df: pd.DataFrame, table_name: str = ""):
        self._df = df
        self._table_name = table_name

    @property
    def df(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self._df

    @property
    def data(self) -> "RelatedDataView":
        """Return a data view for nice printing and column access."""
        return RelatedDataView(self._df, self._table_name)

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        """Iterate over rows as Series."""
        for _, row in self._df.iterrows():
            yield row

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the DataFrame."""
        if name in self._df.columns:
            return self._df[name]
        return getattr(self._df, name)

    def __getitem__(self, key):
        """Allow indexing like a DataFrame."""
        return self._df[key]

    def __repr__(self) -> str:
        return f"<RelatedData: {self._table_name} ({len(self._df)} rows)>"

    def __str__(self) -> str:
        """Short summary when printed directly."""
        return f"{self._table_name}: {len(self._df)} rows"


class RelatedDataView:
    """
    View for RelatedData that provides nice printing and column access.

    Example:
        >>> print(reserves.data)  # Formatted output
        >>> reserves.data.dscRecoverableOil  # Column values
    """

    MAX_VALUE_LENGTH = 60
    MAX_ROWS_DISPLAY = 5

    def __init__(self, df: pd.DataFrame, table_name: str = ""):
        self._df = df
        self._table_name = table_name

    def _truncate(self, value: Any) -> str:
        """Truncate a value for display."""
        s = str(value)
        if len(s) <= self.MAX_VALUE_LENGTH:
            return s
        keep = (self.MAX_VALUE_LENGTH - 3) // 2
        return s[:keep] + "..." + s[-keep:]

    def __getattr__(self, name: str) -> Any:
        """Allow column access."""
        if name in self._df.columns:
            return self._df[name].tolist()
        raise AttributeError(f"No column '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Dict-style column access."""
        return self._df[key].tolist()

    def keys(self) -> list:
        """Return column names."""
        return list(self._df.columns)

    def __str__(self) -> str:
        """Formatted string showing all rows with truncated values."""
        if self._df.empty:
            return f"{self._table_name}: (empty)"

        lines = []
        n_rows = len(self._df)
        show_rows = min(n_rows, self.MAX_ROWS_DISPLAY)

        if self._table_name:
            lines.append(f"{self._table_name}: {n_rows} row{'s' if n_rows != 1 else ''}")
            lines.append("-" * 60)

        for i in range(show_rows):
            row = self._df.iloc[i]
            if i > 0:
                lines.append("")
            lines.append(f"[Row {i}]")
            for col in self._df.columns:
                value = row[col]
                display_value = self._truncate(value)
                lines.append(f"  {col}: {display_value}")

        if n_rows > self.MAX_ROWS_DISPLAY:
            lines.append(f"\n... and {n_rows - self.MAX_ROWS_DISPLAY} more rows")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<RelatedDataView: {len(self._df)} rows, {len(self._df.columns)} columns>"


class RelatedTableMixin:
    """
    Mixin that provides dynamic related table access for entity classes.

    Requires the class to have:
    - self._data: pd.Series with entity data
    - self._db: Database instance
    - self.id: Entity's primary ID

    Example:
        >>> troll = fp.field("troll")
        >>> troll.field_reserves  # Returns DataFrame of related reserves
        >>> troll.field_licensee_hst  # Returns DataFrame of licensees
    """

    # Common ID column patterns in Sodir data
    ID_COLUMN_PATTERNS = [
        'Npdid',      # Standard pattern: fldNpdidField, dscNpdidDiscovery
        'NpdidField', # Specific patterns
        'NpdidDiscovery',
        'NpdidWellbore',
        'NpdidLicence',
        'NpdidCompany',
        'NpdidFacility',
    ]

    @classmethod
    def _find_id_columns(cls, df: pd.DataFrame) -> List[str]:
        """Find potential ID columns in a DataFrame."""
        id_cols = []
        for col in df.columns:
            for pattern in cls.ID_COLUMN_PATTERNS:
                if pattern in col:
                    id_cols.append(col)
                    break
        return id_cols

    @classmethod
    def _find_common_id_columns(cls, cols1: List[str], cols2: List[str]) -> List[tuple]:
        """
        Find common ID columns between two column lists.

        Matches on specific ID type patterns (NpdidField, NpdidDiscovery, etc.)
        not just the generic 'Npdid' pattern.
        """
        # Specific ID type patterns to match on
        specific_patterns = [
            'NpdidField', 'NpdidDiscovery', 'NpdidWellbore',
            'NpdidLicence', 'NpdidCompany', 'NpdidFacility',
        ]

        common = []
        for c1 in cols1:
            for c2 in cols2:
                # Match on specific patterns only
                for pattern in specific_patterns:
                    if pattern in c1 and pattern in c2:
                        common.append((c1, c2))
                        break
        return common

    def related(self, table_name: str) -> "RelatedData":
        """
        Get related rows from another table.

        Finds the best matching ID column between this entity's data and the
        target table, then filters the target table to matching rows.

        Uses intelligent ID column matching:
        - Prioritizes the ID column that best matches the target table name
        - e.g., for 'field' table, prefers 'fldNpdidField' over 'cmpNpdidCompany'

        Args:
            table_name: Name of the table to query

        Returns:
            RelatedData with matching rows (use .df for DataFrame, .data for display)

        Example:
            >>> troll = fp.field("troll")
            >>> reserves = troll.related('field_reserves')
            >>> print(reserves.data)  # Nice formatted output
            >>> reserves.df           # Get as DataFrame
        """
        target_df = self._db.get_or_none(table_name)
        if target_df is None:
            return RelatedData(pd.DataFrame(), table_name)

        # Find ID columns in both
        my_id_cols = self._find_id_columns(pd.DataFrame([self._data]))
        target_id_cols = self._find_id_columns(target_df)

        # Find common ID columns
        common = self._find_common_id_columns(my_id_cols, target_id_cols)

        # Check for information carrier pattern (generic ID type)
        # Tables like 'profiles' use NpdidInformationCarrier + Kind column
        info_carrier_col = None
        for col in target_id_cols:
            if 'NpdidInformationCarrier' in col:
                info_carrier_col = col
                break

        if info_carrier_col and not common:
            # Try to use information carrier as the join
            # Map entity types to their InformationCarrierKind values
            base_table = getattr(self, '_table_name', None) or type(self).__name__.lower()
            entity_to_carrier = {
                'field': ('fldNpdidField', 'FIELD'),
                'discovery': ('dscNpdidDiscovery', 'DISCOVERY'),
            }
            if base_table in entity_to_carrier:
                my_id_col, kind_value = entity_to_carrier[base_table]
                kind_col = info_carrier_col.replace('NpdidInformationCarrier', 'InformationCarrierKind')
                if my_id_col in self._data.index and kind_col in target_df.columns:
                    my_value = self._data.get(my_id_col)
                    if pd.notna(my_value):
                        result = target_df[
                            (target_df[info_carrier_col] == my_value) &
                            (target_df[kind_col] == kind_value)
                        ]
                        return RelatedData(result, table_name)

        if not common:
            return RelatedData(pd.DataFrame(), table_name)

        # Prioritize the ID column that best matches the target table name
        # e.g., for 'field' table, prefer 'NpdidField' pattern
        best_match = None
        table_lower = table_name.lower()

        # Map table names to their primary ID patterns
        table_patterns = {
            'field': 'NpdidField',
            'discovery': 'NpdidDiscovery',
            'wellbore': 'NpdidWellbore',
            'licence': 'NpdidLicence',
            'company': 'NpdidCompany',
            'facility': 'NpdidFacility',
        }

        # Find the primary pattern for this table
        primary_pattern = None
        for prefix, pattern in table_patterns.items():
            if table_lower.startswith(prefix):
                primary_pattern = pattern
                break

        # Look for a match using the primary pattern first
        if primary_pattern:
            for my_col, target_col in common:
                if primary_pattern in my_col and primary_pattern in target_col:
                    best_match = (my_col, target_col)
                    break

        # If no primary match found, use the first available
        if best_match is None:
            best_match = common[0]

        # Filter using only the best matching ID column
        my_col, target_col = best_match
        my_value = self._data.get(my_col)
        if pd.notna(my_value):
            result = target_df[target_df[target_col] == my_value]
            return RelatedData(result, table_name)

        return RelatedData(pd.DataFrame(), table_name)

    def _lookup_related_table(self, name: str) -> Optional["RelatedData"]:
        """
        Try to find a related table by name.

        Returns RelatedData if table exists (may be empty), None if table doesn't exist.
        """
        if self._db.has_dataset(name):
            return self.related(name)
        return None

    @property
    def data(self) -> "EntityData":
        """
        Display all column names and values for this entity.

        Returns an EntityData object that:
        - Prints as a formatted string with truncated long values
        - Allows attribute access for full values: entity.data._geometry

        Example:
            >>> print(troll.data)
            fldNpdidField: 46437
            fldName: TROLL
            _geometry: {"type": "Polygon", "coo...40640798473664]]]}
            ...

            >>> troll.data._geometry  # Full value
            '{"type": "Polygon", "coordinates": [[[4.172...]]}'
        """
        return EntityData(self._data)

    @property
    def df(self) -> pd.DataFrame:
        """
        Get this entity's data as a single-row DataFrame.

        For most entities this returns a DataFrame with one row.
        For entities that may have multiple rows (like reserves history),
        this returns all matching rows.

        Example:
            >>> troll_df = troll.df
            >>> troll_df.columns
        """
        return pd.DataFrame([self._data])

    # =========================================================================
    # Template Customization
    # =========================================================================

    def _get_entity_type(self) -> str:
        """Get the entity type string for this entity."""
        # Subclasses should define _entity_type, otherwise infer from class name
        if hasattr(self, '_entity_type'):
            return self._entity_type
        return type(self).__name__.lower()

    def _get_default_template(self) -> str:
        """Get the default template for this entity type."""
        from .display import TEMPLATES
        entity_type = self._get_entity_type()
        return TEMPLATES.get(entity_type, "")

    def _get_current_template(self) -> str:
        """Get the current template (custom if exists, else default)."""
        entity_type = self._get_entity_type()
        custom = self._db.get_template(entity_type)
        if custom is not None:
            return custom
        return self._get_default_template()

    def str_template(self, index: bool = True) -> str:
        """
        Show the current template for this entity type.

        Args:
            index: If True, show line numbers. If False, show raw template.

        Returns:
            Template string with or without line numbers.

        Example:
            >>> print(discovery.str_template())
            DISCOVERY_TEMPLATE:
            1: "# Discovery: {name}"
            2: "==="
            3: "Discovered: {discovery_year:<12}  HC Type: {hc_type}"
            ...

            >>> print(discovery.str_template(index=False))
            DISCOVERY_TEMPLATE = \"\"\"
            # Discovery: {name}
            ===
            ...
            \"\"\"
        """
        entity_type = self._get_entity_type()
        template = self._get_current_template()
        template_name = f"{entity_type.upper()}_TEMPLATE"

        lines = template.split('\n')

        if index:
            # Show with line numbers
            result = [f"{template_name}:"]
            for i, line in enumerate(lines, 1):
                result.append(f'{i}: "{line}"')
            return '\n'.join(result)
        else:
            # Show raw format (for copying to code)
            return f'{template_name} = """\n{template}\n"""'

    def update_template(self, line_num: int, new_text: str) -> None:
        """
        Update a single line in the template.

        Args:
            line_num: Line number to update (1-indexed)
            new_text: New text for that line

        Example:
            >>> discovery.update_template(3, "Discovered: {discovery_year}  Type: {hc_type}")
            >>> discovery.update_template(20, "Custom footer")  # Adds padding if needed
        """
        if line_num < 1:
            raise ValueError("Line number must be >= 1")

        template = self._get_current_template()
        lines = template.split('\n')

        # Pad with empty lines if needed
        while len(lines) < line_num:
            lines.append("")

        # Update the line (convert to 0-indexed)
        lines[line_num - 1] = new_text

        # Save custom template
        entity_type = self._get_entity_type()
        self._db.save_template(entity_type, '\n'.join(lines))

    def reset_template(self) -> bool:
        """
        Reset template to default, removing any customizations.

        Returns:
            True if a custom template was removed, False if no custom template existed.

        Example:
            >>> discovery.reset_template()
            True
        """
        entity_type = self._get_entity_type()
        return self._db.delete_template(entity_type)

    def _get_primary_id_pattern(self) -> Optional[str]:
        """
        Determine the primary ID pattern for this entity.

        Returns the ID pattern that represents this entity's own identity
        (e.g., 'NpdidField' for Field entities).
        """
        my_id_cols = self._find_id_columns(pd.DataFrame([self._data]))
        if not my_id_cols:
            return None

        # The primary ID is typically the first one, or we can infer from table name
        # For entities with _table_name, use that
        table_name = getattr(self, '_table_name', None) or type(self).__name__.lower()

        table_patterns = {
            'field': 'NpdidField',
            'discovery': 'NpdidDiscovery',
            'wellbore': 'NpdidWellbore',
            'licence': 'NpdidLicence',
            'license': 'NpdidLicence',
            'company': 'NpdidCompany',
            'facility': 'NpdidFacility',
        }

        # Check if table name starts with known prefix
        for prefix, pattern in table_patterns.items():
            if table_name.lower().startswith(prefix):
                # Verify this entity actually has this pattern
                for col in my_id_cols:
                    if pattern in col:
                        return pattern
                break

        # Fallback: use the pattern from the first ID column
        if my_id_cols:
            for pattern in ['NpdidField', 'NpdidDiscovery', 'NpdidWellbore',
                            'NpdidLicence', 'NpdidCompany', 'NpdidFacility']:
                if pattern in my_id_cols[0]:
                    return pattern

        return None

    @property
    def connections(self) -> Dict[str, List[str]]:
        """
        Get lists of tables that reference this entity and tables this entity references.

        Returns a dict with:
        - 'incoming': Tables that have my primary ID as a foreign key
          (e.g., field_reserves, field_licensee_hst reference field)
        - 'outgoing': Tables whose primary entities I have foreign keys to
          (e.g., field references company via cmpNpdidCompany)

        Example:
            >>> troll = fp.field("troll")
            >>> troll.connections
            {'incoming': ['field_reserves', 'field_licensee_hst', ...],
             'outgoing': ['company', 'licence']}
        """
        incoming = []

        primary_pattern = self._get_primary_id_pattern()
        if not primary_pattern:
            return {'incoming': [], 'outgoing': []}

        # Get my foreign keys (ID columns that are NOT my primary)
        my_id_cols = self._find_id_columns(pd.DataFrame([self._data]))
        my_foreign_keys = [col for col in my_id_cols if primary_pattern not in col]

        # Get the base table name for this entity
        base_table = getattr(self, '_table_name', None) or type(self).__name__.lower()

        # Get all available tables
        all_tables = self._db.list_datasets()

        # Map entity types to their InformationCarrierKind values
        entity_to_carrier_kind = {
            'field': 'FIELD',
            'discovery': 'DISCOVERY',
        }

        # Find tables that reference me (have my primary ID)
        for table_name in all_tables:
            # Skip internal tables and own base table
            if table_name.startswith('_'):
                continue
            if table_name == base_table:
                continue

            target_df = self._db.get_or_none(table_name)
            if target_df is None or target_df.empty:
                continue

            target_id_cols = self._find_id_columns(target_df)

            # Check if target table has MY primary ID → it references me
            for col in target_id_cols:
                if primary_pattern in col:
                    if table_name not in incoming:
                        incoming.append(table_name)
                    break

            # Check for information carrier pattern (generic ID type)
            # Tables like 'profiles' use NpdidInformationCarrier + Kind column
            if table_name not in incoming:
                for col in target_id_cols:
                    if 'NpdidInformationCarrier' in col:
                        # Check if there's a Kind column indicating which entity types
                        kind_col = col.replace('NpdidInformationCarrier', 'InformationCarrierKind')
                        if kind_col in target_df.columns:
                            # Check if this entity's type is in the kind values
                            my_kind = entity_to_carrier_kind.get(base_table)
                            if my_kind and my_kind in target_df[kind_col].unique():
                                incoming.append(table_name)
                        break

        # For outgoing, only include the BASE entity tables (not all tables with that ID)
        # Map patterns to their base tables
        pattern_to_base_table = {
            'NpdidField': 'field',
            'NpdidDiscovery': 'discovery',
            'NpdidWellbore': 'wellbore',
            'NpdidCompany': 'company',
            'NpdidLicence': 'licence',
            'NpdidFacility': 'facility',
            'NpdidPipeline': 'pipeline',
            'NpdidPlay': 'play',
            'NpdidBlock': 'block',
            'NpdidQuadrant': 'quadrant',
            'NpdidTuf': 'tuf',
            'NpdidSurvey': 'seismic_acquisition',
            'NpdidLithoStrat': 'strat_litho',
            'NpdidBsnsArrArea': 'business_arrangement_area',
        }

        # Find which base tables I reference via my foreign keys
        outgoing_base = set()
        for my_fk in my_foreign_keys:
            for pattern, base_table in pattern_to_base_table.items():
                if pattern in my_fk and base_table in all_tables:
                    outgoing_base.add(base_table)
                    break

        return {
            'incoming': sorted(incoming),
            'outgoing': sorted(outgoing_base)
        }

    @property
    def full_connections(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get filtered DataFrames for all connections.

        Returns a dict with:
        - 'incoming': Dict of {table_name: filtered_DataFrame} for tables that reference this entity
        - 'outgoing': Dict of {table_name: filtered_DataFrame} for tables this entity references

        Example:
            >>> troll = fp.field("troll")
            >>> conns = troll.full_connections
            >>> conns['incoming']['field_reserves']  # DataFrame of Troll's reserves
            >>> conns['outgoing']['company']  # DataFrame of Troll's operator
        """
        # Get connection lists first
        conn_lists = self.connections

        incoming = {}
        outgoing = {}

        # Get data for tables that reference me
        for table_name in conn_lists['incoming']:
            related_df = self.related(table_name)
            if not related_df.empty:
                incoming[table_name] = related_df

        # Get data for tables I reference
        for table_name in conn_lists['outgoing']:
            related_df = self.related(table_name)
            if not related_df.empty:
                outgoing[table_name] = related_df

        return {
            'incoming': incoming,
            'outgoing': outgoing
        }


class PartnersList(list):
    """
    A list of partners with nice formatted printing.

    Behaves exactly like a regular list, but prints nicely.

    Example:
        >>> print(troll.partners)

        Partners (5):
        ============================================================
        Company                                   Share %  Operator
        ------------------------------------------------------------
        Equinor Energy AS                           30.58  *
        Petoro AS                                   30.00
        TotalEnergies EP Norge AS                    8.44
        Shell Norge AS                               8.10
        ConocoPhillips Skandinavia AS               22.88
        ------------------------------------------------------------
        Total: 100.00%
    """

    def __init__(self, partners: list, field_name: str = "", as_of: Optional[str] = None):
        super().__init__(partners)
        self.field_name = field_name
        self.as_of = as_of

    def __str__(self) -> str:
        if not self:
            if self.as_of:
                return f"No partners found as of {self.as_of}"
            return "No partners found"

        # Calculate column widths
        company_width = max(
            len("Company"),
            max((len(p['company'][:40]) for p in self), default=8)
        )
        share_width = 8  # "Share %" header
        op_width = 8  # "Operator" header

        # Build header row
        header = f"{'Company':<{company_width}}  {'Share %':>{share_width}}  {'Operator':<{op_width}}"
        table_width = len(header)

        # Title with optional date
        if self.as_of:
            lines = [f"\nPartners as of {self.as_of} ({len(self)}):"]
        else:
            lines = [f"\nPartners ({len(self)}):"]
        lines.append("=" * table_width)
        lines.append(header)
        lines.append("-" * table_width)

        # Table rows
        for p in self:
            company = p['company'][:40]
            share = f"{p['share']:>.2f}"
            op_mark = "*" if p.get('is_operator') else ""
            lines.append(f"{company:<{company_width}}  {share:>{share_width}}  {op_mark:<{op_width}}")

        lines.append("-" * table_width)

        # Total
        total = sum(p['share'] for p in self)
        lines.append(f"Total: {total:.2f}%")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"PartnersList({len(self)} partners)"


class EntityDataFrame(pd.DataFrame):
    """
    A DataFrame with nice formatted printing for entity data.

    Behaves exactly like a pandas DataFrame, but has a nicer default print.
    """

    _metadata = ['_entity_type', '_field_name', '_display_columns']

    def __init__(self, data=None, entity_type: str = "Entity", field_name: str = "",
                 display_columns: Optional[List[str]] = None, **kwargs):
        super().__init__(data, **kwargs)
        self._entity_type = entity_type
        self._field_name = field_name
        self._display_columns = display_columns

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            df = EntityDataFrame(*args, **kwargs)
            df._entity_type = getattr(self, '_entity_type', 'Entity')
            df._field_name = getattr(self, '_field_name', '')
            df._display_columns = getattr(self, '_display_columns', None)
            return df
        return _c

    def __str__(self) -> str:
        if self.empty:
            return f"No {self._entity_type.lower()} found"

        lines = []

        # Determine which columns to display
        display_cols = self._display_columns
        if display_cols is None:
            display_cols = self._get_default_display_columns()

        # Filter to existing columns
        display_cols = [c for c in display_cols if c in self.columns]

        if not display_cols:
            display_cols = list(self.columns[:5])  # Fallback to first 5 columns

        # Create readable column names
        col_names = self._get_readable_column_names(display_cols)

        # Calculate column widths
        col_widths = {}
        for col, name in zip(display_cols, col_names):
            values = self[col].astype(str).str[:30]  # Truncate long values
            max_val_len = values.str.len().max() if len(values) > 0 else 0
            col_widths[col] = max(len(name), max_val_len, 8)

        # Build header row and calculate table width
        header_parts = [f"{name:<{col_widths[col]}}" for col, name in zip(display_cols, col_names)]
        header_row = "  ".join(header_parts)
        table_width = len(header_row)

        # Title header
        title = f"\n{self._entity_type}"
        if self._field_name:
            title += f" on {self._field_name}"
        title += f" ({len(self)} records):"
        lines.append(title)
        lines.append("=" * table_width)
        lines.append(header_row)
        lines.append("-" * table_width)

        # Data rows (limit to 15)
        display_df = self.head(15)
        for _, row in display_df.iterrows():
            row_parts = []
            for col in display_cols:
                val = row[col]
                if pd.isna(val):
                    val_str = ""
                elif isinstance(val, float):
                    val_str = f"{val:.1f}" if val != int(val) else f"{int(val)}"
                else:
                    val_str = str(val)[:30]
                row_parts.append(f"{val_str:<{col_widths[col]}}")
            lines.append("  ".join(row_parts))

        if len(self) > 15:
            lines.append(f"... and {len(self) - 15} more records")

        return '\n'.join(lines)

    def _get_default_display_columns(self) -> List[str]:
        """Get default columns based on entity type."""
        defaults = {
            'Wells': ['wlbWellboreName', 'wlbPurpose', 'wlbStatus', 'wlbTotalDepth', 'wlbContent'],
            'Wells Drilled': ['wlbWellboreName', 'wlbPurpose', 'wlbStatus', 'wlbTotalDepth', 'wlbCompletionDate'],
            'Facilities': ['fclName', 'fclKind', 'fclPhase', 'fclStatus'],
            'Discoveries': ['dscName', 'dscDiscoveryYear', 'dscHcType', 'dscCurrentActivityStatus'],
            'Operated Fields': ['fldName', 'fldCurrentActivitySatus', 'fldHcType', 'fldMainArea'],
            'Fields': ['fldName', 'fldCurrentActivitySatus', 'fldHcType', 'fldMainArea'],
            'Formation Tops': ['lsuName', 'lsuTopDepth', 'lsuBottomDepth', 'lsuLevel'],
            'DST Results': ['dstTestNumber', 'dstFromDepth', 'dstToDepth', 'dstChokeSize', 'dstOilRate'],
            'Cores': ['wlbCoreNumber', 'wlbCoreIntervalTop', 'wlbCoreIntervalBottom'],
        }
        return defaults.get(self._entity_type, list(self.columns[:5]))

    def _get_readable_column_names(self, columns: List[str]) -> List[str]:
        """
        Convert API column names to readable names.

        Uses aliases from field_aliases.json (populated from API metadata).
        Falls back to auto-generated names if no alias is found.
        """
        # Load aliases from cache file
        aliases = _load_aliases()

        result = []
        for col in columns:
            if col in aliases:
                result.append(aliases[col])
            else:
                result.append(self._auto_readable_name(col))
        return result

    def _auto_readable_name(self, column: str) -> str:
        """Auto-generate a readable name from column name."""
        # Remove common prefixes
        prefixes = ['wlb', 'fld', 'dsc', 'fcl', 'prl', 'cmp', 'pip', 'ply', 'dst', 'lsu', 'prf']
        name = column
        for prefix in prefixes:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]
                break

        # Convert CamelCase to Title Case with spaces
        import re
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)

        # Clean up and title case
        name = name.replace('_', ' ').strip()
        return name.title() if name else column

    def __repr__(self) -> str:
        return f"EntityDataFrame({self._entity_type}: {len(self)} records)"


class Field(RelatedTableMixin):
    """
    Represents a petroleum field on the Norwegian Continental Shelf.

    Supports dynamic related table access:
        >>> troll.field_reserves  # Returns DataFrame
        >>> troll.field_licensee_hst  # Returns DataFrame

    Example:
        >>> fp = Factpages()
        >>> troll = fp.field("troll")
        >>> print(troll.name)
        'TROLL'
        >>> print(troll.operator)
        'Equinor Energy AS'
        >>> print(troll.partners)
        [{'company': 'Equinor', 'share': 30.58}, ...]
        >>> print(troll.production(2025, 8))
        {'oil_sm3': 12450, 'gas_msm3': 119.2, ...}
    """

    _entity_type = "field"

    def __init__(self, data: pd.Series, db: "Database"):
        """
        Initialize a Field entity.

        Args:
            data: Series with field data from the field dataset
            db: Database instance for fetching related data
        """
        self._data = data
        self._db = db

        # Cache for related data
        self._partners_cache: Optional[pd.DataFrame] = None
        self._production_cache: Optional[pd.DataFrame] = None
        self._reserves_cache: Optional[pd.DataFrame] = None

    def _get_column(self, column: str, default: Any = '') -> Any:
        """
        Get a column value with warning if column doesn't exist.

        Args:
            column: Column name to retrieve
            default: Default value if column doesn't exist

        Returns:
            Column value or default
        """
        if column in self._data.index:
            return self._data[column]

        # Column doesn't exist - show warning with valid options
        valid_cols = sorted([c for c in self._data.index if not c.startswith('_')])
        # Find similar column names to suggest
        similar = [c for c in valid_cols if column.lower() in c.lower() or c.lower() in column.lower()]

        msg = f"Column '{column}' not found in field data."
        if similar:
            msg += f" Similar columns: {similar[:5]}"
        else:
            msg += f" Valid columns: {valid_cols[:10]}..."
        warnings.warn(msg, UserWarning, stacklevel=3)
        return default

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def id(self) -> int:
        """Field's unique NPD ID."""
        return int(self._get_column('fldNpdidField', 0))

    @property
    def name(self) -> str:
        """Field name."""
        return self._get_column('fldName', '')

    @property
    def status(self) -> str:
        """Current field status (e.g., 'Producing', 'Shut down')."""
        return self._get_column('fldCurrentActivitySatus', '')

    @property
    def operator(self) -> str:
        """Current operator name."""
        return self._get_column('cmpLongName', '')

    @property
    def hc_type(self) -> str:
        """Hydrocarbon type (OIL, GAS, OIL/GAS, etc.)."""
        return self._get_column('fldHcType', '')

    @property
    def main_area(self) -> str:
        """Main area (North Sea, Norwegian Sea, Barents Sea)."""
        return self._get_column('fldMainArea', '')

    @property
    def discovery_year(self) -> Optional[int]:
        """Year the field was discovered."""
        year = self._get_column('fldDiscoveryYear', None)
        return int(year) if pd.notna(year) else None

    @property
    def production_start(self) -> Optional[str]:
        """Production start date (if available)."""
        # Use silent get - this column may not exist in all field datasets
        return self._data.get('fldProdStartDate')

    @property
    def geometry(self) -> Optional[dict]:
        """Field geometry as GeoJSON dict."""
        import json
        geom_str = self._get_column('_geometry', None)
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    # =========================================================================
    # Related Data Properties
    # =========================================================================

    def partners(self, as_of: Optional[str] = None) -> PartnersList:
        """
        Get field partners (licensees) with their equity shares.

        Args:
            as_of: Optional date string (YYYY-MM-DD) to get historic partners.
                   If None, returns current partners.

        Returns:
            PartnersList with 'company', 'share', and 'is_operator' keys.
            Print it for a nice formatted table.

        Example:
            >>> # Current partners
            >>> print(troll.partners())

            Partners (5):
            ============================================================
            Company                              Share %  Operator
            ------------------------------------------------------------
            Petoro AS                              55.93
            Equinor Energy AS                      30.55  *
            ...

            >>> # Historic partners as of a specific date
            >>> print(troll.partners("2010-01-01"))
        """
        if self._partners_cache is None:
            licensees = self._db.get_or_none('field_licensee_hst')
            if licensees is not None:
                self._partners_cache = licensees[
                    licensees['fldNpdidField'] == self.id
                ]
            else:
                self._partners_cache = pd.DataFrame()

        if self._partners_cache.empty:
            return PartnersList([], field_name=self.name, as_of=as_of)

        # Filter by date
        filtered = self._partners_cache.copy()

        # Convert as_of date to epoch milliseconds
        if as_of:
            target_dt = datetime.strptime(as_of, '%Y-%m-%d')
        else:
            target_dt = datetime.now()
        target_ms = target_dt.timestamp() * 1000

        # Filter: started before target date AND (no end date OR ended after target date)
        if 'fldLicenseeFrom' in filtered.columns and 'fldLicenseeTo' in filtered.columns:
            filtered = filtered[
                (filtered['fldLicenseeFrom'] <= target_ms) &
                (filtered['fldLicenseeTo'].isna() | (filtered['fldLicenseeTo'] > target_ms))
            ]
        elif 'fldLicenseeTo' in filtered.columns:
            # Only filter by end date if start date not available
            filtered = filtered[
                filtered['fldLicenseeTo'].isna() | (filtered['fldLicenseeTo'] > target_ms)
            ]

        partners = []
        for _, row in filtered.iterrows():
            partners.append({
                'company': row.get('cmpLongName', ''),
                'share': float(row.get('fldCompanyShare', 0) or 0),
                'is_operator': row.get('cmpLongName', '') == self.operator,
            })

        partners = sorted(partners, key=lambda x: x['share'], reverse=True)
        return PartnersList(partners, field_name=self.name, as_of=as_of)

    @property
    def reserves(self) -> dict:
        """
        Current reserves estimates.

        Returns:
            Dict with 'oil_msm3', 'gas_bsm3', 'ngl_mtoe', etc.
        """
        if self._reserves_cache is None:
            reserves = self._db.get_or_none('field_reserves')
            if reserves is not None:
                self._reserves_cache = reserves[
                    reserves['fldNpdidField'] == self.id
                ]
            else:
                self._reserves_cache = pd.DataFrame()

        if self._reserves_cache.empty:
            return {}

        # Get most recent reserves
        latest = self._reserves_cache.sort_values('fldRecoverableOil', ascending=False)
        if latest.empty:
            return {}

        row = latest.iloc[0]
        return {
            'oil_msm3': float(row.get('fldRecoverableOil', 0) or 0),
            'gas_bsm3': float(row.get('fldRecoverableGas', 0) or 0),
            'ngl_mtoe': float(row.get('fldRecoverableNGL', 0) or 0),
            'condensate_msm3': float(row.get('fldRecoverableCondensate', 0) or 0),
        }

    @property
    def remaining(self) -> dict:
        """
        Remaining recoverable reserves (future production forecast).

        Returns:
            Dict with 'oil_msm3', 'gas_bsm3', 'ngl_mtoe', 'condensate_msm3', 'oe_msm3'.
            These are the remaining resources yet to be produced.

        Example:
            >>> troll = fp.field("troll")
            >>> troll.remaining
            {'oil_msm3': 12.5, 'gas_bsm3': 564.5, ...}
        """
        if self._reserves_cache is None:
            reserves = self._db.get_or_none('field_reserves')
            if reserves is not None:
                self._reserves_cache = reserves[
                    reserves['fldNpdidField'] == self.id
                ]
            else:
                self._reserves_cache = pd.DataFrame()

        if self._reserves_cache.empty:
            return {}

        # Get most recent estimate (by date if available)
        if 'fldDateOffResEstDisplay' in self._reserves_cache.columns:
            latest = self._reserves_cache.sort_values('fldDateOffResEstDisplay', ascending=False)
        else:
            latest = self._reserves_cache

        if latest.empty:
            return {}

        row = latest.iloc[0]
        return {
            'oil_msm3': float(row.get('fldRemainingOil', 0) or 0),
            'gas_bsm3': float(row.get('fldRemainingGas', 0) or 0),
            'ngl_mtoe': float(row.get('fldRemainingNGL', 0) or 0),
            'condensate_msm3': float(row.get('fldRemainingCondensate', 0) or 0),
            'oe_msm3': float(row.get('fldRemainingOE', 0) or 0),
        }

    # =========================================================================
    # Production Methods
    # =========================================================================

    def _load_production(self) -> None:
        """Load production data from profiles table (information carrier)."""
        if self._production_cache is None:
            profiles = self._db.get_or_none('profiles')
            if profiles is not None:
                # Filter by information carrier type and ID
                self._production_cache = profiles[
                    (profiles['prfInformationCarrierKind'] == 'FIELD') &
                    (profiles['prfNpdidInformationCarrier'] == self.id)
                ]
            else:
                self._production_cache = pd.DataFrame()

    def production(self, year: Optional[int] = None, month: Optional[int] = None) -> dict:
        """
        Get production figures for a specific month, or latest available.

        Args:
            year: Year (e.g., 2025). If None, returns latest month.
            month: Month (1-12). If None, returns latest month.

        Returns:
            Dict with 'oil_sm3', 'gas_msm3', 'water_sm3', etc.
            Empty dict if no data available.

        Example:
            >>> troll.production()           # Latest monthly production
            >>> troll.production(2023, 6)    # June 2023 production
        """
        self._load_production()

        if self._production_cache.empty:
            return {}

        # Get only monthly data (month > 0, as month=0 is yearly aggregates)
        monthly = self._production_cache[self._production_cache['prfMonth'] > 0]

        if monthly.empty:
            return {}

        # If no year/month specified, get the latest entry
        if year is None or month is None:
            # Sort by year and month descending to get latest
            latest = monthly.sort_values(['prfYear', 'prfMonth'], ascending=False)
            row = latest.iloc[0]
            year = int(row['prfYear'])
            month = int(row['prfMonth'])
        else:
            # Filter by specified year and month
            filtered = monthly[
                (monthly['prfYear'] == year) &
                (monthly['prfMonth'] == month)
            ]
            if filtered.empty:
                return {}
            row = filtered.iloc[0]

        return {
            'year': year,
            'month': month,
            'oil_sm3': float(row.get('prfPrdOilNetMillSm3', 0) or 0) * 1_000_000,
            'gas_msm3': float(row.get('prfPrdGasNetBillSm3', 0) or 0) * 1_000,
            'ngl_sm3': float(row.get('prfPrdNGLNetMillSm3', 0) or 0) * 1_000_000,
            'condensate_sm3': float(row.get('prfPrdCondensateNetMillSm3', 0) or 0) * 1_000_000,
            'water_sm3': float(row.get('prfPrdProducedWaterInFieldMillS', 0) or 0) * 1_000_000,
        }

    def production_yearly(self, year: Optional[int] = None) -> dict:
        """
        Get total production for a year, or latest complete year.

        Args:
            year: Year (e.g., 2024). If None, returns latest complete year.

        Returns:
            Dict with yearly totals

        Example:
            >>> troll.production_yearly()      # Latest complete year
            >>> troll.production_yearly(2023)  # 2023 totals
        """
        self._load_production()

        if self._production_cache.empty:
            return {}

        if year is None:
            # Get the latest year that has actual monthly data
            monthly = self._production_cache[self._production_cache['prfMonth'] > 0]
            if monthly.empty:
                return {}
            year = int(monthly['prfYear'].max())

        # Filter by year and sum monthly data
        filtered = self._production_cache[
            (self._production_cache['prfYear'] == year) &
            (self._production_cache['prfMonth'] > 0)
        ]

        if filtered.empty:
            return {}

        return {
            'year': year,
            'oil_sm3': float(filtered['prfPrdOilNetMillSm3'].sum() or 0) * 1_000_000,
            'gas_msm3': float(filtered['prfPrdGasNetBillSm3'].sum() or 0) * 1_000,
            'ngl_sm3': float(filtered['prfPrdNGLNetMillSm3'].sum() or 0) * 1_000_000,
            'condensate_sm3': float(filtered['prfPrdCondensateNetMillSm3'].sum() or 0) * 1_000_000,
        }

    def production_history(self, clean: bool = True) -> pd.DataFrame:
        """
        Get full production history as DataFrame.

        Args:
            clean: If True, return cleaned DataFrame with readable column names.
                   If False, return raw data from API.

        Returns:
            DataFrame with monthly production data, sorted by date.

        Example:
            >>> troll = fp.field("troll")
            >>> history = troll.production_history()
            >>> print(history.columns)
            ['year', 'month', 'oil_sm3', 'gas_sm3', 'ngl_tonnes', ...]
            >>> print(history.tail())  # Recent production
        """
        self._load_production()

        if self._production_cache is None or self._production_cache.empty:
            return pd.DataFrame()

        if not clean:
            return self._production_cache.copy()

        # Create cleaned DataFrame with readable column names
        df = self._production_cache.copy()

        # Build clean DataFrame with columns from profiles table
        clean_df = pd.DataFrame({
            'year': df['prfYear'].astype(int),
            'month': df['prfMonth'].astype(int),
            'oil_sm3': (df['prfPrdOilNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'gas_sm3': (df['prfPrdGasNetBillSm3'].fillna(0) * 1_000_000_000).astype(float),
            'ngl_sm3': (df['prfPrdNGLNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'condensate_sm3': (df['prfPrdCondensateNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'oe_sm3': (df['prfPrdOeNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'water_sm3': (df['prfPrdProducedWaterInFieldMillS'].fillna(0) * 1_000_000).astype(float),
        })

        # Add injection columns if present
        if 'prfInjectedGasMillSm3' in df.columns:
            clean_df['gas_injected_sm3'] = (df['prfInjectedGasMillSm3'].fillna(0) * 1_000_000).astype(float)
        if 'prfInjectedWaterMillSm3' in df.columns:
            clean_df['water_injected_sm3'] = (df['prfInjectedWaterMillSm3'].fillna(0) * 1_000_000).astype(float)

        # Sort by date
        clean_df = clean_df.sort_values(['year', 'month']).reset_index(drop=True)

        return clean_df

    def production_by_year(self) -> pd.DataFrame:
        """
        Get production aggregated by year.

        Returns:
            DataFrame with yearly production totals.

        Example:
            >>> troll = fp.field("troll")
            >>> yearly = troll.production_by_year()
            >>> print(yearly[yearly['year'] >= 2020])
        """
        history = self.production_history(clean=True)

        if history.empty:
            return pd.DataFrame()

        # Columns to sum
        sum_cols = ['oil_sm3', 'gas_sm3', 'ngl_tonnes', 'condensate_sm3', 'oe_sm3', 'water_sm3']
        sum_cols = [c for c in sum_cols if c in history.columns]

        # Group by year and sum
        yearly = history.groupby('year')[sum_cols].sum().reset_index()

        return yearly

    # =========================================================================
    # Reserves Methods
    # =========================================================================

    def reserves_history(self, clean: bool = True) -> pd.DataFrame:
        """
        Get historical reserves estimates as DataFrame.

        Args:
            clean: If True, return cleaned DataFrame with readable column names.

        Returns:
            DataFrame with reserves estimates over time.

        Example:
            >>> troll = fp.field("troll")
            >>> reserves = troll.reserves_history()
            >>> print(reserves[['year', 'oil_msm3', 'gas_bsm3']])
        """
        if self._reserves_cache is None:
            reserves = self._db.get_or_none('field_reserves')
            if reserves is not None:
                self._reserves_cache = reserves[
                    reserves['fldNpdidField'] == self.id
                ]
            else:
                self._reserves_cache = pd.DataFrame()

        if self._reserves_cache.empty:
            return pd.DataFrame()

        if not clean:
            return self._reserves_cache.copy()

        df = self._reserves_cache.copy()

        # Build clean DataFrame
        clean_df = pd.DataFrame({
            'year': df.get('fldYear', df.index).astype(int) if 'fldYear' in df.columns else range(len(df)),
            'oil_msm3': df['fldRecoverableOil'].fillna(0).astype(float),
            'gas_bsm3': df['fldRecoverableGas'].fillna(0).astype(float),
            'ngl_mtoe': df['fldRecoverableNGL'].fillna(0).astype(float),
            'condensate_msm3': df['fldRecoverableCondensate'].fillna(0).astype(float),
        })

        # Add remaining/original if available
        if 'fldRemainingOil' in df.columns:
            clean_df['remaining_oil_msm3'] = df['fldRemainingOil'].fillna(0).astype(float)
        if 'fldRemainingGas' in df.columns:
            clean_df['remaining_gas_bsm3'] = df['fldRemainingGas'].fillna(0).astype(float)

        return clean_df.sort_values('year').reset_index(drop=True)

    # =========================================================================
    # Related Entities
    # =========================================================================

    @property
    def wells(self) -> EntityDataFrame:
        """
        Get all wellbores associated with this field.

        Returns:
            EntityDataFrame of wellbores on this field.
            Print it for a nice formatted table.

        Example:
            >>> print(troll.wells)

            Wells on TROLL (127 records):
            ======================================================================
            Name              Purpose       Status      Depth (m)  Content
            ----------------------------------------------------------------------
            31/2-1            WILDCAT       P&A         3150       OIL/GAS
            31/2-2            APPRAISAL     P&A         3245       GAS
            ...
        """
        wellbores = self._db.get_or_none('wellbore')
        if wellbores is None:
            return EntityDataFrame(entity_type="Wells", field_name=self.name)

        # Filter by field name
        field_wells = wellbores[wellbores['fldName'] == self.name]
        return EntityDataFrame(field_wells, entity_type="Wells", field_name=self.name)

    @property
    def facilities(self) -> EntityDataFrame:
        """
        Get all facilities associated with this field.

        Returns:
            EntityDataFrame of facilities on this field.
            Print it for a nice formatted table.

        Example:
            >>> print(troll.facilities)

            Facilities on TROLL (8 records):
            ======================================================================
            Name              Type          Phase       Status
            ----------------------------------------------------------------------
            TROLL A           FIXED         IN SERVICE  IN SERVICE
            TROLL B           FIXED         IN SERVICE  IN SERVICE
            ...
        """
        facilities = self._db.get_or_none('facility')
        if facilities is None:
            return EntityDataFrame(entity_type="Facilities", field_name=self.name)

        # Filter by field name (check multiple possible column names)
        if 'fldName' in facilities.columns:
            field_facilities = facilities[facilities['fldName'] == self.name]
        elif 'fclBelongsToName' in facilities.columns:
            field_facilities = facilities[facilities['fclBelongsToName'] == self.name]
        else:
            return EntityDataFrame(entity_type="Facilities", field_name=self.name)

        return EntityDataFrame(field_facilities, entity_type="Facilities", field_name=self.name)

    @property
    def discoveries(self) -> EntityDataFrame:
        """
        Get all discoveries that became part of this field.

        Returns:
            EntityDataFrame of discoveries developed into this field.
            Print it for a nice formatted table.

        Example:
            >>> print(troll.discoveries)

            Discoveries on TROLL (3 records):
            ======================================================================
            Name              Year    HC Type     Status
            ----------------------------------------------------------------------
            TROLL             1979    GAS         PRODUCING
            TROLL VEST        1983    OIL         PRODUCING
            ...
        """
        discoveries = self._db.get_or_none('discovery')
        if discoveries is None:
            return EntityDataFrame(entity_type="Discoveries", field_name=self.name)

        # Filter by field name
        field_discoveries = discoveries[discoveries['fldName'] == self.name]
        return EntityDataFrame(field_discoveries, entity_type="Discoveries", field_name=self.name)

    # =========================================================================
    # Dynamic Related Table Access
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Enable dynamic access to related tables."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Try related table lookup
        result = self._lookup_related_table(name)
        if result is not None:
            return result

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Field('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "field")


class Discovery(RelatedTableMixin):
    """
    Represents a petroleum discovery on the Norwegian Continental Shelf.

    Example:
        >>> fp = Factpages()
        >>> johan = fp.discovery("JOHAN SVERDRUP")
        >>> print(johan)  # Shows geology-focused summary
        >>> print(johan.wells)  # All wells on discovery
        >>> print(johan.resources)  # Resource estimates
    """

    _entity_type = "discovery"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db
        self._reserves_cache: Optional[pd.DataFrame] = None

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def id(self) -> int:
        return int(self._data.get('dscNpdidDiscovery', 0))

    @property
    def name(self) -> str:
        return self._data.get('dscName', '')

    @property
    def status(self) -> str:
        """Current activity status (PRODUCING, PDO APPROVED, INCLUDED IN OTHER DISCOVERY, etc.)"""
        return self._data.get('dscCurrentActivityStatus', '')

    @property
    def hc_type(self) -> str:
        """Hydrocarbon type (OIL, GAS, OIL/GAS, CONDENSATE)."""
        return self._data.get('dscHcType', '')

    @property
    def main_area(self) -> str:
        """Main area (NORTH SEA, NORWEGIAN SEA, BARENTS SEA)."""
        return self._data.get('dscMainArea', '')

    # =========================================================================
    # Geology
    # =========================================================================

    @property
    def discovery_year(self) -> Optional[int]:
        """Year the discovery was made."""
        year = self._data.get('dscDiscoveryYear')
        return int(year) if pd.notna(year) else None

    @property
    def discovery_well(self) -> str:
        """Name of the discovery wellbore."""
        return self._data.get('dscDiscoveryWellbore', '')

    @property
    def main_ncs_area(self) -> str:
        """NCS area designation."""
        return self._data.get('nmaName', '') or self._data.get('dscMainArea', '')

    @property
    def owner_kind(self) -> str:
        """Ownership type (BUSINESS ARRANGEMENT, LICENSE)."""
        return self._data.get('dscOwnerKind', '')

    @property
    def geometry(self) -> Optional[dict]:
        """Discovery geometry as GeoJSON dict."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    # =========================================================================
    # Operator & Development
    # =========================================================================

    @property
    def operator(self) -> str:
        """Current operator."""
        return self._data.get('dscOperatorCompanyName', '')

    @property
    def field_name(self) -> Optional[str]:
        """Name of field if discovery was developed into a producing field."""
        fld = self._data.get('fldName')
        return str(fld) if fld and pd.notna(fld) else None

    @property
    def is_producing(self) -> bool:
        """Whether this discovery is currently producing."""
        return 'PRODUCING' in self.status.upper() if self.status else False

    @property
    def is_developed(self) -> bool:
        """Whether this discovery has been developed into a field."""
        return self.field_name is not None

    # =========================================================================
    # Resource Estimates
    # =========================================================================

    def _load_reserves(self) -> pd.DataFrame:
        """Load reserve estimates for this discovery."""
        if self._reserves_cache is None:
            reserves = self._db.get_or_none('discovery_reserves')
            if reserves is not None:
                self._reserves_cache = reserves[
                    reserves['dscNpdidDiscovery'] == self.id
                ]
            else:
                self._reserves_cache = pd.DataFrame()
        return self._reserves_cache

    @property
    def resources(self) -> dict:
        """
        Current resource estimates for this discovery.

        Returns:
            Dict with 'oil_msm3', 'gas_bsm3', etc.

        Example:
            >>> print(hamlet.resources)
            {'oil_msm3': 5.2, 'gas_bsm3': 12.5, ...}
        """
        df = self._load_reserves()

        if df.empty:
            return {}

        # Get most recent estimate
        if 'dscReservesUpdatedDate' in df.columns:
            latest = df.sort_values('dscReservesUpdatedDate', ascending=False)
        else:
            latest = df

        if latest.empty:
            return {}

        row = latest.iloc[0]
        return {
            'oil_msm3': float(row.get('dscRecoverableOil', 0) or 0),
            'gas_bsm3': float(row.get('dscRecoverableGas', 0) or 0),
            'ngl_mtoe': float(row.get('dscRecoverableNGL', 0) or 0),
            'condensate_msm3': float(row.get('dscRecoverableCondensate', 0) or 0),
        }

    def resources_history(self, clean: bool = True) -> pd.DataFrame:
        """
        Historical resource estimates as DataFrame.

        Args:
            clean: If True, return cleaned DataFrame with readable column names.

        Returns:
            DataFrame with resource estimates over time.
        """
        df = self._load_reserves()

        if df.empty:
            return pd.DataFrame()

        if not clean:
            return df.copy()

        # Build clean DataFrame
        clean_df = pd.DataFrame()

        if 'dscReservesUpdatedDate' in df.columns:
            clean_df['date'] = df['dscReservesUpdatedDate']

        clean_df['oil_msm3'] = df.get('dscRecoverableOil', pd.Series([0])).fillna(0).astype(float)
        clean_df['gas_bsm3'] = df.get('dscRecoverableGas', pd.Series([0])).fillna(0).astype(float)
        clean_df['ngl_mtoe'] = df.get('dscRecoverableNGL', pd.Series([0])).fillna(0).astype(float)
        clean_df['condensate_msm3'] = df.get('dscRecoverableCondensate', pd.Series([0])).fillna(0).astype(float)

        return clean_df

    # =========================================================================
    # Related Entities
    # =========================================================================

    @property
    def wells(self) -> EntityDataFrame:
        """
        Wellbores drilled on this discovery.

        Returns:
            EntityDataFrame of wellbores associated with this discovery.
        """
        wellbores = self._db.get_or_none('wellbore')
        if wellbores is None:
            return EntityDataFrame(entity_type="Wells", field_name=self.name)

        discovery_wells = wellbores[wellbores['dscName'] == self.name]
        return EntityDataFrame(discovery_wells, entity_type="Wells", field_name=self.name)

    # =========================================================================
    # Dynamic Related Table Access
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Enable dynamic access to related tables."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        result = self._lookup_related_table(name)
        if result is not None:
            return result

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Discovery('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "discovery")


class Wellbore(RelatedTableMixin):
    """
    Represents a wellbore drilled on the Norwegian Continental Shelf.

    Supports dynamic related table access:
        >>> well.wellbore_dst  # Returns DataFrame
        >>> well.strat_litho_wellbore  # Returns DataFrame

    Example:
        >>> fp = Factpages()
        >>> well = fp.well("31/2-1")
        >>> print(well)  # Shows key geology info
        >>> print(well.formation_tops)  # Stratigraphy
        >>> print(well.dst_results)  # Flow tests
    """

    _entity_type = "wellbore"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def id(self) -> int:
        return int(self._data.get('wlbNpdidWellbore', 0))

    @property
    def name(self) -> str:
        return self._data.get('wlbWellboreName', '')

    @property
    def status(self) -> str:
        """Well status (P&A, PRODUCING, JUNKED, etc.)"""
        return self._data.get('wlbStatus', '')

    @property
    def purpose(self) -> str:
        """Well purpose (WILDCAT, APPRAISAL, PRODUCTION, INJECTION)."""
        return self._data.get('wlbPurpose', '')

    @property
    def content(self) -> str:
        """What was found (OIL, GAS, OIL/GAS, SHOWS, DRY)."""
        return self._data.get('wlbContent', '')

    @property
    def operator(self) -> str:
        """Drilling operator."""
        return self._data.get('wlbDrillingOperator', '')

    # =========================================================================
    # Depth & Location
    # =========================================================================

    @property
    def total_depth(self) -> Optional[float]:
        """Total measured depth in meters."""
        td = self._data.get('wlbTotalDepth')
        return float(td) if pd.notna(td) else None

    @property
    def kelly_bushing(self) -> Optional[float]:
        """Kelly bushing elevation in meters."""
        kb = self._data.get('wlbKellyBushingElevation')
        return float(kb) if pd.notna(kb) else None

    @property
    def water_depth(self) -> Optional[float]:
        """Water depth in meters."""
        wd = self._data.get('wlbWaterDepth')
        return float(wd) if pd.notna(wd) else None

    @property
    def coordinates(self) -> dict:
        """Surface coordinates (lat/lon in decimal degrees)."""
        return {
            'lat': self._data.get('wlbNsDecDeg'),
            'lon': self._data.get('wlbEwDecDeg'),
        }

    @property
    def geometry(self) -> Optional[dict]:
        """Wellbore geometry as GeoJSON dict (surface location point)."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    # =========================================================================
    # Geology - HC Bearing Formations
    # =========================================================================

    @property
    def hc_formations(self) -> List[str]:
        """
        Formations with hydrocarbon shows/content.

        Returns:
            List of formation names where HC was encountered.
        """
        formations = []
        for i in range(1, 4):
            fm = self._data.get(f'wlbFormationWithHc{i}')
            if fm and pd.notna(fm):
                formations.append(str(fm))
        return formations

    @property
    def hc_ages(self) -> List[str]:
        """
        Geological ages with hydrocarbon shows/content.

        Returns:
            List of ages (e.g., 'JURASSIC', 'TRIASSIC') where HC was found.
        """
        ages = []
        for i in range(1, 4):
            age = self._data.get(f'wlbAgeWithHc{i}')
            if age and pd.notna(age):
                ages.append(str(age))
        return ages

    @property
    def main_area(self) -> str:
        """Main area (NORTH SEA, NORWEGIAN SEA, BARENTS SEA)."""
        return self._data.get('wlbMainArea', '')

    # =========================================================================
    # Dates
    # =========================================================================

    @property
    def completion_date(self) -> Optional[str]:
        """Date drilling was completed."""
        date = self._data.get('wlbCompletionDate')
        if date and pd.notna(date):
            return str(date)[:10]  # YYYY-MM-DD
        return None

    @property
    def entry_date(self) -> Optional[str]:
        """Date drilling started (spud date)."""
        date = self._data.get('wlbEntryDate')
        if date and pd.notna(date):
            return str(date)[:10]
        return None

    # =========================================================================
    # Associations
    # =========================================================================

    @property
    def field_name(self) -> Optional[str]:
        """Associated field name."""
        return self._data.get('fldName')

    @property
    def discovery_name(self) -> Optional[str]:
        """Associated discovery name."""
        return self._data.get('dscName')

    # =========================================================================
    # Related Data
    # =========================================================================

    @property
    def formation_tops(self) -> EntityDataFrame:
        """
        Formation tops (stratigraphy) for this wellbore.

        Returns:
            EntityDataFrame with formation names and depths.
        """
        strat = self._db.get_or_none('strat_litho_wellbore')
        if strat is None:
            return EntityDataFrame(entity_type="Formation Tops", field_name=self.name)

        well_strat = strat[strat['wlbNpdidWellbore'] == self.id]
        return EntityDataFrame(
            well_strat,
            entity_type="Formation Tops",
            field_name=self.name,
            display_columns=['lsuName', 'lsuTopDepth', 'lsuBottomDepth', 'lsuLevel']
        )

    @property
    def dst_results(self) -> EntityDataFrame:
        """
        Drill stem test (DST) results for this wellbore.

        Returns:
            EntityDataFrame with DST data including flow rates.
        """
        dst = self._db.get_or_none('wellbore_dst')
        if dst is None:
            return EntityDataFrame(entity_type="DST Results", field_name=self.name)

        well_dst = dst[dst['wlbNpdidWellbore'] == self.id]
        return EntityDataFrame(
            well_dst,
            entity_type="DST Results",
            field_name=self.name,
            display_columns=['dstTestNumber', 'dstFromDepth', 'dstToDepth', 'dstBottomHolePress']
        )

    @property
    def cores(self) -> EntityDataFrame:
        """
        Core samples from this wellbore.

        Returns:
            EntityDataFrame with core sample data.
        """
        cores = self._db.get_or_none('wellbore_core_photo')
        if cores is None:
            return EntityDataFrame(entity_type="Cores", field_name=self.name)

        well_cores = cores[cores['wlbNpdidWellbore'] == self.id]
        return EntityDataFrame(well_cores, entity_type="Cores", field_name=self.name)

    @property
    def drilling_history(self) -> EntityDataFrame:
        """
        Drilling history events for this wellbore.

        Returns:
            EntityDataFrame with drilling events and dates.

        Example:
            >>> well = fp.well("31/2-1")
            >>> print(well.drilling_history)
        """
        history = self._db.get_or_none('wellbore_history')
        if history is None:
            return EntityDataFrame(entity_type="Drilling History", field_name=self.name)

        well_history = history[history['wlbNpdidWellbore'] == self.id]
        return EntityDataFrame(
            well_history,
            entity_type="Drilling History",
            field_name=self.name,
            display_columns=['wlbHistory', 'wlbHistoryDateFrom', 'wlbHistoryDateTo']
        )

    @property
    def casing(self) -> EntityDataFrame:
        """
        Casing program for this wellbore.

        Returns:
            EntityDataFrame with casing strings, sizes and depths.

        Example:
            >>> well = fp.well("31/2-1")
            >>> print(well.casing)
        """
        casing = self._db.get_or_none('wellbore_casing')
        if casing is None:
            return EntityDataFrame(entity_type="Casing", field_name=self.name)

        well_casing = casing[casing['wlbNpdidWellbore'] == self.id]
        return EntityDataFrame(
            well_casing,
            entity_type="Casing",
            field_name=self.name,
            display_columns=['wlbCasingType', 'wlbCasingDiameter', 'wlbCasingDepth', 'wlbCasingMD']
        )

    # =========================================================================
    # Dynamic Related Table Access
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Enable dynamic access to related tables."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        result = self._lookup_related_table(name)
        if result is not None:
            return result

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Wellbore('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "wellbore")


class FieldInterestsList(list):
    """
    A list of field interests with nice formatted printing.

    Example:
        >>> print(equinor.field_interests)

        Field Interests (45):
        ============================================================
        Field                   Share %  Operator
        ------------------------------------------------------------
        TROLL                     30.58  *
        JOHAN SVERDRUP            22.60  *
        SNORRE                    33.30  *
        ...
    """

    def __init__(self, interests: list, company_name: str = ""):
        super().__init__(interests)
        self.company_name = company_name

    def __str__(self) -> str:
        if not self:
            return "No field interests found"

        # Calculate column widths
        field_width = max(
            len("Field"),
            max((len(i['field'][:30]) for i in self), default=8)
        )
        share_width = 8
        op_width = 8

        header = f"{'Field':<{field_width}}  {'Share %':>{share_width}}  {'Operator':<{op_width}}"
        table_width = len(header)

        lines = [f"\nField Interests ({len(self)}):"]
        lines.append("=" * table_width)
        lines.append(header)
        lines.append("-" * table_width)

        # Show up to 15 rows
        for i in self[:15]:
            field = i['field'][:30]
            share = f"{i['share']:>.2f}"
            op_mark = "*" if i.get('is_operator') else ""
            lines.append(f"{field:<{field_width}}  {share:>{share_width}}  {op_mark:<{op_width}}")

        if len(self) > 15:
            lines.append(f"... and {len(self) - 15} more fields")

        lines.append("-" * table_width)
        total_share = sum(i['share'] for i in self)
        lines.append(f"Total equity: {total_share:.2f}%")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"FieldInterestsList({len(self)} fields)"


class Company(RelatedTableMixin):
    """
    Represents a company operating on the NCS.

    Supports dynamic related table access:
        >>> equinor.field_licensee_hst  # Returns DataFrame

    Example:
        >>> fp = Factpages()
        >>> equinor = fp.company("equinor")
        >>> print(equinor.name)
        >>> print(equinor.field_interests)
        >>> print(equinor.operated_fields)
    """

    _entity_type = "company"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db
        self._field_interests_cache: Optional[pd.DataFrame] = None

    @property
    def id(self) -> int:
        return int(self._data.get('cmpNpdidCompany', 0))

    @property
    def name(self) -> str:
        return self._data.get('cmpLongName', '')

    @property
    def short_name(self) -> str:
        return self._data.get('cmpShortName', '')

    @property
    def org_number(self) -> str:
        return self._data.get('cmpOrgNumberBrReg', '')

    @property
    def nation(self) -> str:
        return self._data.get('cmpNationCode', '')

    # =========================================================================
    # Portfolio Properties
    # =========================================================================

    def _load_field_interests(self) -> pd.DataFrame:
        """Load field licensee data for this company."""
        if self._field_interests_cache is None:
            licensees = self._db.get_or_none('field_licensee_hst')
            if licensees is not None:
                # Filter by company name
                self._field_interests_cache = licensees[
                    licensees['cmpLongName'] == self.name
                ]
            else:
                self._field_interests_cache = pd.DataFrame()
        return self._field_interests_cache

    @property
    def field_interests(self) -> FieldInterestsList:
        """
        All field interests (equity positions) for this company.

        Returns:
            FieldInterestsList with field name, share %, and operator status.

        Example:
            >>> print(equinor.field_interests)

            Field Interests (45):
            ============================================
            Field                   Share %  Operator
            --------------------------------------------
            TROLL                     30.58  *
            JOHAN SVERDRUP            22.60  *
            ...
        """
        df = self._load_field_interests()

        if df.empty:
            return FieldInterestsList([], company_name=self.name)

        # Get current interests (no end date or future end date)
        current = df.copy()
        today = datetime.now().strftime('%Y-%m-%d')

        if 'fldLicenseeDateValidTo' in current.columns:
            current = current[
                current['fldLicenseeDateValidTo'].isna() |
                (current['fldLicenseeDateValidTo'] >= today)
            ]

        # Get field operator info to mark operated fields
        fields = self._db.get_or_none('field')
        operated_fields = set()
        if fields is not None:
            operated = fields[fields['cmpLongName'] == self.name]
            operated_fields = set(operated['fldName'].tolist())

        interests = []
        for _, row in current.iterrows():
            field_name = row.get('fldName', '')
            interests.append({
                'field': field_name,
                'share': float(row.get('fldLicenseeInterest', 0)),
                'is_operator': field_name in operated_fields,
            })

        # Sort by share descending
        interests = sorted(interests, key=lambda x: x['share'], reverse=True)
        return FieldInterestsList(interests, company_name=self.name)

    @property
    def operated_fields(self) -> EntityDataFrame:
        """
        Fields where this company is the operator.

        Returns:
            EntityDataFrame of fields operated by this company.

        Example:
            >>> print(equinor.operated_fields)

            Operated Fields (25 records):
            =============================================
            Name              Status      HC Type  Area
            ---------------------------------------------
            TROLL             PRODUCING   GAS      NORTH SEA
            ...
        """
        fields = self._db.get_or_none('field')
        if fields is None:
            return EntityDataFrame(entity_type="Operated Fields")

        operated = fields[fields['cmpLongName'] == self.name]
        return EntityDataFrame(
            operated,
            entity_type="Operated Fields",
            display_columns=['fldName', 'fldCurrentActivitySatus', 'fldHcType', 'fldMainArea']
        )

    @property
    def wells_drilled(self) -> EntityDataFrame:
        """
        Wellbores drilled by this company.

        Returns:
            EntityDataFrame of wellbores where this company was drilling operator.
        """
        wellbores = self._db.get_or_none('wellbore')
        if wellbores is None:
            return EntityDataFrame(entity_type="Wells Drilled")

        drilled = wellbores[wellbores['wlbDrillingOperator'] == self.name]
        return EntityDataFrame(drilled, entity_type="Wells Drilled")

    # =========================================================================
    # Dynamic Related Table Access
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Enable dynamic access to related tables."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        result = self._lookup_related_table(name)
        if result is not None:
            return result

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Company('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "company")


class License(RelatedTableMixin):
    """
    Represents a production license on the Norwegian Continental Shelf.

    Supports dynamic related table access:
        >>> lic.licence_licensee_hst  # Returns DataFrame
        >>> lic.licence_task  # Returns DataFrame

    Example:
        >>> fp = Factpages()
        >>> pl001 = fp.license("PL001")
        >>> print(pl001)  # Shows key license info
        >>> print(pl001.licensees)  # Current licensees
        >>> print(pl001.fields)  # Related fields
    """

    _entity_type = "license"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db
        self._licensees_cache: Optional[pd.DataFrame] = None

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def id(self) -> int:
        return int(self._data.get('prlNpdidLicence', 0))

    @property
    def name(self) -> str:
        return self._data.get('prlName', '')

    @property
    def status(self) -> str:
        """License status (ACTIVE, INACTIVE, etc.)"""
        return self._data.get('prlStatus', '')

    @property
    def operator(self) -> str:
        """Current operator company."""
        return self._data.get('prlOperatorCompanyName', '')

    @property
    def main_area(self) -> str:
        """Main area (NORTH SEA, NORWEGIAN SEA, BARENTS SEA)."""
        return self._data.get('prlMainArea', '')

    @property
    def licensing_activity(self) -> str:
        """Licensing round (APA 2020, TFO 2022, etc.)"""
        return self._data.get('prlLicensingActivityName', '')

    # =========================================================================
    # Dates
    # =========================================================================

    @property
    def date_granted(self) -> Optional[str]:
        """Date the license was granted."""
        date = self._data.get('prlDateGranted')
        if date and pd.notna(date):
            return str(date)[:10]
        return None

    @property
    def date_valid_to(self) -> Optional[str]:
        """License expiry date."""
        date = self._data.get('prlDateValidTo')
        if date and pd.notna(date):
            return str(date)[:10]
        return None

    @property
    def current_phase(self) -> str:
        """Current phase (EXPLORATION, PRODUCTION, etc.)"""
        return self._data.get('prlCurrentPhase', '')

    @property
    def geometry(self) -> Optional[dict]:
        """License geometry as GeoJSON dict (license area polygon)."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    # =========================================================================
    # Related Data
    # =========================================================================

    def _load_licensees(self) -> pd.DataFrame:
        """Load licensee history for this license."""
        if self._licensees_cache is None:
            licensees = self._db.get_or_none('licence_licensee_hst')
            if licensees is not None:
                self._licensees_cache = licensees[
                    licensees['prlNpdidLicence'] == self.id
                ]
            else:
                self._licensees_cache = pd.DataFrame()
        return self._licensees_cache

    @property
    def licensees(self) -> PartnersList:
        """
        Current licensees with equity shares.

        Returns:
            PartnersList with company, share, and operator status.
        """
        df = self._load_licensees()

        if df.empty:
            return PartnersList([], field_name=self.name)

        # Get current licensees (no end date or future end date)
        current = df.copy()
        today = datetime.now().strftime('%Y-%m-%d')

        if 'prlLicenseeDateValidTo' in current.columns:
            current = current[
                current['prlLicenseeDateValidTo'].isna() |
                (current['prlLicenseeDateValidTo'] >= today)
            ]

        partners = []
        for _, row in current.iterrows():
            company = row.get('cmpLongName', '')
            partners.append({
                'company': company,
                'share': float(row.get('prlLicenseeInterest', 0) or 0),
                'is_operator': company == self.operator,
            })

        partners = sorted(partners, key=lambda x: x['share'], reverse=True)
        return PartnersList(partners, field_name=self.name)

    @property
    def fields(self) -> EntityDataFrame:
        """
        Fields associated with this license.

        Returns:
            EntityDataFrame of fields on this license.
        """
        fields = self._db.get_or_none('field')
        if fields is None:
            return EntityDataFrame(entity_type="Fields", field_name=self.name)

        # Filter by license name
        if 'prlName' in fields.columns:
            license_fields = fields[fields['prlName'] == self.name]
        else:
            return EntityDataFrame(entity_type="Fields", field_name=self.name)

        return EntityDataFrame(license_fields, entity_type="Fields", field_name=self.name)

    @property
    def discoveries(self) -> EntityDataFrame:
        """
        Discoveries on this license.

        Returns:
            EntityDataFrame of discoveries on this license.
        """
        discoveries = self._db.get_or_none('discovery')
        if discoveries is None:
            return EntityDataFrame(entity_type="Discoveries", field_name=self.name)

        # Filter by license name
        if 'prlName' in discoveries.columns:
            license_discoveries = discoveries[discoveries['prlName'] == self.name]
        else:
            return EntityDataFrame(entity_type="Discoveries", field_name=self.name)

        return EntityDataFrame(license_discoveries, entity_type="Discoveries", field_name=self.name)

    @property
    def wells(self) -> EntityDataFrame:
        """
        Wellbores drilled on this license.

        Returns:
            EntityDataFrame of wellbores on this license.
        """
        wellbores = self._db.get_or_none('wellbore')
        if wellbores is None:
            return EntityDataFrame(entity_type="Wells", field_name=self.name)

        # Filter by license name
        if 'prlName' in wellbores.columns:
            license_wells = wellbores[wellbores['prlName'] == self.name]
        else:
            return EntityDataFrame(entity_type="Wells", field_name=self.name)

        return EntityDataFrame(license_wells, entity_type="Wells", field_name=self.name)

    @property
    def ownership_history(self) -> EntityDataFrame:
        """
        Historical ownership changes for this license.

        Shows all licensees over time, including entry and exit dates.

        Returns:
            EntityDataFrame with ownership history.

        Example:
            >>> lic = fp.license("PL001")
            >>> print(lic.ownership_history)
        """
        df = self._load_licensees()

        if df.empty:
            return EntityDataFrame(entity_type="Ownership History", field_name=self.name)

        return EntityDataFrame(
            df,
            entity_type="Ownership History",
            field_name=self.name,
            display_columns=['cmpLongName', 'prlLicenseeInterest', 'prlLicenseeDateFrom', 'prlLicenseeDateValidTo']
        )

    @property
    def phase_history(self) -> EntityDataFrame:
        """
        License phase history (work program obligations).

        Shows the exploration and production phases with deadlines.

        Returns:
            EntityDataFrame with phase information.

        Example:
            >>> lic = fp.license("PL001")
            >>> print(lic.phase_history)
        """
        phases = self._db.get_or_none('licence_phase_hst')
        if phases is None:
            return EntityDataFrame(entity_type="Phase History", field_name=self.name)

        license_phases = phases[phases['prlNpdidLicence'] == self.id]
        return EntityDataFrame(
            license_phases,
            entity_type="Phase History",
            field_name=self.name,
            display_columns=['prlPhaseName', 'prlPhaseStatus', 'prlPhaseDateFrom', 'prlPhaseDateTo']
        )

    @property
    def work_obligations(self) -> EntityDataFrame:
        """
        Work program obligations for this license.

        Shows specific commitments like wells to drill, seismic to acquire.

        Returns:
            EntityDataFrame with work obligations.

        Example:
            >>> lic = fp.license("PL001")
            >>> print(lic.work_obligations)
        """
        tasks = self._db.get_or_none('licence_task')
        if tasks is None:
            return EntityDataFrame(entity_type="Work Obligations", field_name=self.name)

        license_tasks = tasks[tasks['prlNpdidLicence'] == self.id]
        return EntityDataFrame(
            license_tasks,
            entity_type="Work Obligations",
            field_name=self.name,
            display_columns=['prlTaskName', 'prlTaskTargetDate', 'prlTaskStatus', 'prlTaskDescription']
        )

    # =========================================================================
    # Dynamic Related Table Access
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Enable dynamic access to related tables."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        result = self._lookup_related_table(name)
        if result is not None:
            return result

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"License('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "license")


class Entity:
    """
    Generic entity representing a row from any table.

    Used for accessing data from tables that don't have a dedicated
    entity class (like Field, Discovery, etc.).

    Supports dynamic related table access via attribute syntax:
        >>> reserves = fp.field_reserves(43506)
        >>> reserves.field  # Returns matching rows from field table

    Example:
        >>> reserves = fp.field_reserves(43506)  # Get by npdid
        >>> print(reserves)
        >>> print(reserves.fldRecoverableOil)
        >>> print(reserves.field)  # Get related field data
    """

    # Common ID column patterns in Sodir data
    ID_COLUMN_PATTERNS = [
        'Npdid',      # Standard pattern: fldNpdidField, dscNpdidDiscovery
        'NpdidField', # Specific patterns
        'NpdidDiscovery',
        'NpdidWellbore',
        'NpdidLicence',
        'NpdidCompany',
        'NpdidFacility',
    ]

    def __init__(self, data: pd.Series, db: "Database", table_name: str):
        self._data = data
        self._db = db
        self._table_name = table_name

    @classmethod
    def find_id_columns(cls, df: pd.DataFrame) -> List[str]:
        """
        Find potential ID columns in a DataFrame.

        Returns columns that match common ID patterns (Npdid).
        """
        id_cols = []
        for col in df.columns:
            for pattern in cls.ID_COLUMN_PATTERNS:
                if pattern in col:
                    id_cols.append(col)
                    break
        return id_cols

    @classmethod
    def find_by_id(cls, df: pd.DataFrame, id_value: int) -> Optional[pd.Series]:
        """
        Find a row by searching ID columns.

        Args:
            df: DataFrame to search
            id_value: ID value to find

        Returns:
            Matching row or None
        """
        id_cols = cls.find_id_columns(df)

        for col in id_cols:
            match = df[df[col] == id_value]
            if not match.empty:
                return match.iloc[0]

        return None

    @classmethod
    def find_common_id_columns(cls, cols1: List[str], cols2: List[str]) -> List[tuple]:
        """
        Find common ID columns between two column lists.

        Matches on specific ID type patterns (NpdidField, NpdidDiscovery, etc.)
        not just the generic 'Npdid' pattern.
        """
        # Specific ID type patterns to match on
        specific_patterns = [
            'NpdidField', 'NpdidDiscovery', 'NpdidWellbore',
            'NpdidLicence', 'NpdidCompany', 'NpdidFacility',
        ]

        common = []
        for c1 in cols1:
            for c2 in cols2:
                # Match on specific patterns only
                for pattern in specific_patterns:
                    if pattern in c1 and pattern in c2:
                        common.append((c1, c2))
                        break
        return common

    @property
    def id(self) -> Optional[int]:
        """Get the primary ID of this entity."""
        id_cols = self.find_id_columns(pd.DataFrame([self._data]))
        if id_cols:
            return int(self._data.get(id_cols[0], 0))
        return None

    @property
    def table_name(self) -> str:
        """The source table name."""
        return self._table_name

    def related(self, table_name: str) -> "RelatedData":
        """
        Get related rows from another table.

        Finds the best matching ID column between this entity's data and the
        target table, then filters the target table to matching rows.

        Uses intelligent ID column matching:
        - Prioritizes the ID column that best matches the target table name
        - e.g., for 'field' table, prefers 'fldNpdidField' over 'cmpNpdidCompany'

        Args:
            table_name: Name of the table to query

        Returns:
            RelatedData with matching rows (use .df for DataFrame, .data for display)

        Example:
            >>> reserves = fp.field_reserves(43506)
            >>> field_df = reserves.related('field')
            >>> print(field_df.data)  # Nice output
        """
        target_df = self._db.get_or_none(table_name)
        if target_df is None:
            return RelatedData(pd.DataFrame(), table_name)

        # Find ID columns in both
        my_id_cols = self.find_id_columns(pd.DataFrame([self._data]))
        target_id_cols = self.find_id_columns(target_df)

        # Find common ID columns
        common = self.find_common_id_columns(my_id_cols, target_id_cols)

        if not common:
            return RelatedData(pd.DataFrame(), table_name)

        # Prioritize the ID column that best matches the target table name
        best_match = None
        table_lower = table_name.lower()

        # Map table names to their primary ID patterns
        table_patterns = {
            'field': 'NpdidField',
            'discovery': 'NpdidDiscovery',
            'wellbore': 'NpdidWellbore',
            'licence': 'NpdidLicence',
            'company': 'NpdidCompany',
            'facility': 'NpdidFacility',
        }

        # Find the primary pattern for this table
        primary_pattern = None
        for prefix, pattern in table_patterns.items():
            if table_lower.startswith(prefix):
                primary_pattern = pattern
                break

        # Look for a match using the primary pattern first
        if primary_pattern:
            for my_col, target_col in common:
                if primary_pattern in my_col and primary_pattern in target_col:
                    best_match = (my_col, target_col)
                    break

        # If no primary match found, use the first available
        if best_match is None:
            best_match = common[0]

        # Filter using only the best matching ID column
        my_col, target_col = best_match
        my_value = self._data.get(my_col)
        if pd.notna(my_value):
            result = target_df[target_df[target_col] == my_value]
            return RelatedData(result, table_name)

        return RelatedData(pd.DataFrame(), table_name)

    def _get_primary_id_pattern(self) -> Optional[str]:
        """
        Determine the primary ID pattern for this entity.

        Returns the ID pattern that represents this entity's own identity.
        """
        my_id_cols = self.find_id_columns(pd.DataFrame([self._data]))
        if not my_id_cols:
            return None

        table_patterns = {
            'field': 'NpdidField',
            'discovery': 'NpdidDiscovery',
            'wellbore': 'NpdidWellbore',
            'licence': 'NpdidLicence',
            'company': 'NpdidCompany',
            'facility': 'NpdidFacility',
        }

        # Check if table name starts with known prefix
        for prefix, pattern in table_patterns.items():
            if self._table_name.lower().startswith(prefix):
                for col in my_id_cols:
                    if pattern in col:
                        return pattern
                break

        # Fallback: use the pattern from the first ID column
        if my_id_cols:
            for pattern in ['NpdidField', 'NpdidDiscovery', 'NpdidWellbore',
                            'NpdidLicence', 'NpdidCompany', 'NpdidFacility']:
                if pattern in my_id_cols[0]:
                    return pattern

        return None

    @property
    def connections(self) -> Dict[str, List[str]]:
        """
        Get lists of tables that reference this entity and tables this entity references.

        Returns a dict with:
        - 'incoming': Tables that have my primary ID as a foreign key
        - 'outgoing': Base entity tables whose primary ID I have as a foreign key

        Example:
            >>> reserves = fp.field_reserves(43506)
            >>> reserves.connections
            {'incoming': [...], 'outgoing': ['field', 'company']}
        """
        incoming = []

        my_id_cols = self.find_id_columns(pd.DataFrame([self._data]))
        primary_pattern = self._get_primary_id_pattern()

        if not primary_pattern:
            return {'incoming': [], 'outgoing': []}

        # Get my foreign keys (ID columns that are NOT my primary)
        my_foreign_keys = [col for col in my_id_cols if primary_pattern not in col]

        all_tables = self._db.list_datasets()

        # Map entity types to their InformationCarrierKind values
        entity_to_carrier_kind = {
            'field': 'FIELD',
            'discovery': 'DISCOVERY',
        }

        # Find tables that reference me
        for table_name in all_tables:
            if table_name.startswith('_'):
                continue
            if table_name == self._table_name:
                continue

            target_df = self._db.get_or_none(table_name)
            if target_df is None or target_df.empty:
                continue

            target_id_cols = self.find_id_columns(target_df)

            # Check if target table has MY primary ID
            for col in target_id_cols:
                if primary_pattern in col:
                    if table_name not in incoming:
                        incoming.append(table_name)
                    break

            # Check for information carrier pattern (generic ID type)
            if table_name not in incoming:
                for col in target_id_cols:
                    if 'NpdidInformationCarrier' in col:
                        kind_col = col.replace('NpdidInformationCarrier', 'InformationCarrierKind')
                        if kind_col in target_df.columns:
                            my_kind = entity_to_carrier_kind.get(self._table_name)
                            if my_kind and my_kind in target_df[kind_col].unique():
                                incoming.append(table_name)
                        break

        # Map patterns to their base tables for outgoing
        pattern_to_base_table = {
            'NpdidField': 'field',
            'NpdidDiscovery': 'discovery',
            'NpdidWellbore': 'wellbore',
            'NpdidCompany': 'company',
            'NpdidLicence': 'licence',
            'NpdidFacility': 'facility',
            'NpdidPipeline': 'pipeline',
            'NpdidPlay': 'play',
            'NpdidBlock': 'block',
            'NpdidQuadrant': 'quadrant',
            'NpdidTuf': 'tuf',
            'NpdidSurvey': 'seismic_acquisition',
            'NpdidLithoStrat': 'strat_litho',
            'NpdidBsnsArrArea': 'business_arrangement_area',
        }

        # Find which base tables I reference via my foreign keys
        outgoing_base = set()
        for my_fk in my_foreign_keys:
            for pattern, base_table in pattern_to_base_table.items():
                if pattern in my_fk and base_table in all_tables:
                    outgoing_base.add(base_table)
                    break

        return {
            'incoming': sorted(incoming),
            'outgoing': sorted(outgoing_base)
        }

    @property
    def full_connections(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get filtered DataFrames for all connections.

        Returns a dict with:
        - 'incoming': Dict of {table_name: filtered_DataFrame}
        - 'outgoing': Dict of {table_name: filtered_DataFrame}

        Example:
            >>> reserves = fp.field_reserves(43506)
            >>> conns = reserves.full_connections
            >>> conns['outgoing']['field']  # DataFrame with the field
        """
        # Get connection lists first
        conn_lists = self.connections

        incoming = {}
        outgoing = {}

        # Get data for tables that reference me
        for table_name in conn_lists['incoming']:
            related_df = self.related(table_name)
            if not related_df.empty:
                incoming[table_name] = related_df

        # Get data for tables I reference
        for table_name in conn_lists['outgoing']:
            related_df = self.related(table_name)
            if not related_df.empty:
                outgoing[table_name] = related_df

        return {
            'incoming': incoming,
            'outgoing': outgoing
        }

    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute access to columns and related tables.

        First checks for column names in the entity data.
        If not found, checks if 'name' is a table in the database
        and returns related rows as a DataFrame.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # First try column lookup
        if name in self._data.index:
            return self._data[name]

        # Try case-insensitive column match
        for col in self._data.index:
            if col.lower() == name.lower():
                return self._data[col]

        # Try related table lookup
        if self._db.has_dataset(name):
            return self.related(name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self._data.to_dict()

    def __repr__(self) -> str:
        return f"Entity(table='{self._table_name}', id={self.id})"

    def __str__(self) -> str:
        """Display the entity data."""
        lines = [
            f"\n{self._table_name.upper().replace('_', ' ')}",
            "=" * 50,
        ]

        # Show all columns
        for col, value in self._data.items():
            if pd.notna(value):
                # Format the column name nicely
                display_name = col
                if len(display_name) > 30:
                    display_name = display_name[:27] + "..."
                lines.append(f"{display_name:<32} {value}")

        return "\n".join(lines)


# =============================================================================
# Additional Entity Classes
# =============================================================================

class Facility(RelatedTableMixin):
    """
    Represents a facility (platform, FPSO, subsea installation) on the NCS.

    Example:
        >>> fp = Factpages()
        >>> troll_a = fp.facility("TROLL A")
        >>> print(troll_a.kind)  # Platform type
        >>> print(troll_a.field_name)  # Associated field
    """

    _entity_type = "facility"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('fclNpdidFacility', 0))

    @property
    def name(self) -> str:
        return self._data.get('fclName', '')

    @property
    def kind(self) -> str:
        """Facility kind (FIXED, FLOATING, SUBSEA, etc.)."""
        return self._data.get('fclKind', '')

    @property
    def functions(self) -> str:
        """Facility functions (WELLHEAD, PROCESSING, etc.)."""
        return self._data.get('fclFunctions', '')

    @property
    def phase(self) -> str:
        """Current phase (IN SERVICE, REMOVED, etc.)."""
        return self._data.get('fclPhase', '')

    @property
    def status(self) -> str:
        """Current status."""
        return self._data.get('fclStatus', '')

    @property
    def water_depth(self) -> Optional[float]:
        """Water depth in meters."""
        depth = self._data.get('fclWaterDepth')
        return float(depth) if pd.notna(depth) else None

    @property
    def field_name(self) -> str:
        """Associated field name."""
        return self._data.get('fldName', '')

    @property
    def startup_date(self) -> Optional[str]:
        """Startup date."""
        date = self._data.get('fclStartupDate')
        return str(date)[:10] if pd.notna(date) else None

    @property
    def geometry(self) -> Optional[dict]:
        """Facility geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Facility('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "facility")


class Pipeline(RelatedTableMixin):
    """
    Represents a pipeline on the NCS.

    Example:
        >>> fp = Factpages()
        >>> pipe = fp.pipeline("STATPIPE")
        >>> print(pipe.medium)
        >>> print(pipe.length)
    """

    _entity_type = "pipeline"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('pipNpdidPipeline', 0))

    @property
    def name(self) -> str:
        return self._data.get('pipName', '')

    @property
    def medium(self) -> str:
        """What the pipeline carries (GAS, OIL, WATER, etc.)."""
        return self._data.get('pipMedium', '')

    @property
    def dimension(self) -> Optional[float]:
        """Pipeline diameter in inches."""
        dim = self._data.get('pipDimension')
        return float(dim) if pd.notna(dim) else None

    @property
    def status(self) -> str:
        """Current status."""
        return self._data.get('pipCurrentPhase', '')

    @property
    def from_facility(self) -> str:
        """Starting facility name."""
        return self._data.get('pipFromFacility', '')

    @property
    def to_facility(self) -> str:
        """Ending facility name."""
        return self._data.get('pipToFacility', '')

    @property
    def operator(self) -> str:
        """Operator company."""
        return self._data.get('cmpLongName', '')

    @property
    def main_area(self) -> str:
        """Main area."""
        return self._data.get('pipMainArea', '')

    @property
    def geometry(self) -> Optional[dict]:
        """Pipeline geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Pipeline('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "pipeline")


class Play(RelatedTableMixin):
    """
    Represents a geological play (prospective hydrocarbon fairway).

    Example:
        >>> fp = Factpages()
        >>> play = fp.play("UPPER JURASSIC TROLL/DRAUPNE PLAY")
        >>> print(play.main_area)
        >>> print(play.status)
    """

    _entity_type = "play"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('plyNpdidPlay', 0))

    @property
    def name(self) -> str:
        return self._data.get('plyName', '')

    @property
    def status(self) -> str:
        """Play status (PUBLIC, CONFIDENTIAL, etc.)."""
        return self._data.get('plyStatus', '')

    @property
    def main_area(self) -> str:
        """Main area."""
        return self._data.get('plyMainArea', '')

    @property
    def geometry(self) -> Optional[dict]:
        """Play geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Play('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "play")


class Block(RelatedTableMixin):
    """
    Represents an administrative block on the NCS.

    Blocks are the basic unit of licensing on the continental shelf.

    Example:
        >>> fp = Factpages()
        >>> block = fp.block("34/10")
        >>> print(block.quadrant)
        >>> print(block.main_area)
    """

    _entity_type = "block"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('blkNpdidBlock', 0))

    @property
    def name(self) -> str:
        return self._data.get('blkName', '')

    @property
    def quadrant(self) -> str:
        """Parent quadrant."""
        return self._data.get('quaName', '')

    @property
    def main_area(self) -> str:
        """Main area (NORTH SEA, NORWEGIAN SEA, BARENTS SEA)."""
        return self._data.get('blkMainArea', '')

    @property
    def status(self) -> str:
        """Block status."""
        return self._data.get('blkStatus', '')

    @property
    def geometry(self) -> Optional[dict]:
        """Block geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Block('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "block")


class Quadrant(RelatedTableMixin):
    """
    Represents an administrative quadrant on the NCS.

    Quadrants are large areas that contain multiple blocks.

    Example:
        >>> fp = Factpages()
        >>> quad = fp.quadrant("34")
        >>> print(quad.main_area)
    """

    _entity_type = "quadrant"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('quaNpdidQuadrant', 0))

    @property
    def name(self) -> str:
        return self._data.get('quaName', '')

    @property
    def main_area(self) -> str:
        """Main area."""
        return self._data.get('quaMainArea', '')

    @property
    def geometry(self) -> Optional[dict]:
        """Quadrant geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Quadrant('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "quadrant")


class TUF(RelatedTableMixin):
    """
    Represents a TUF (Transport og Utnyttelsesanlegg på Fastlandet).

    TUFs are onshore facilities for transport and utilization of petroleum.

    Example:
        >>> fp = Factpages()
        >>> tuf = fp.tuf("KOLLSNES")
        >>> print(tuf.kind)
    """

    _entity_type = "tuf"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('tufNpdidTuf', 0))

    @property
    def name(self) -> str:
        return self._data.get('tufName', '')

    @property
    def kind(self) -> str:
        """TUF kind/type."""
        return self._data.get('tufKind', '')

    @property
    def status(self) -> str:
        """Current status."""
        return self._data.get('tufStatus', '')

    @property
    def startup_date(self) -> Optional[str]:
        """Startup date."""
        date = self._data.get('tufStartupDate')
        return str(date)[:10] if pd.notna(date) else None

    @property
    def operators(self) -> EntityDataFrame:
        """TUF operators history."""
        operators = self._db.get_or_none('tuf_operator_hst')
        if operators is None:
            return EntityDataFrame(entity_type="TUF Operators", field_name=self.name)
        filtered = operators[operators['tufNpdidTuf'] == self.id]
        return EntityDataFrame(filtered, entity_type="TUF Operators", field_name=self.name)

    @property
    def owners(self) -> EntityDataFrame:
        """TUF owners history."""
        owners = self._db.get_or_none('tuf_owner_hst')
        if owners is None:
            return EntityDataFrame(entity_type="TUF Owners", field_name=self.name)
        filtered = owners[owners['tufNpdidTuf'] == self.id]
        return EntityDataFrame(filtered, entity_type="TUF Owners", field_name=self.name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"TUF('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "tuf")


class Seismic(RelatedTableMixin):
    """
    Represents a seismic acquisition survey.

    Example:
        >>> fp = Factpages()
        >>> survey = fp.seismic("NPD-1901")
        >>> print(survey.status)
        >>> print(survey.area)
    """

    _entity_type = "seismic"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('seisNpdidSurvey', 0))

    @property
    def name(self) -> str:
        return self._data.get('seisSurveyName', '')

    @property
    def status(self) -> str:
        """Survey status."""
        return self._data.get('seisStatus', '')

    @property
    def survey_type(self) -> str:
        """Type of survey (2D, 3D, 4D, etc.)."""
        return self._data.get('seisSurveyTypeMain', '')

    @property
    def main_area(self) -> str:
        """Main area."""
        return self._data.get('seisMainArea', '')

    @property
    def company(self) -> str:
        """Company conducting the survey."""
        return self._data.get('cmpLongName', '')

    @property
    def start_date(self) -> Optional[str]:
        """Survey start date."""
        date = self._data.get('seisDateStarting')
        return str(date)[:10] if pd.notna(date) else None

    @property
    def end_date(self) -> Optional[str]:
        """Survey end date."""
        date = self._data.get('seisDateFinalized')
        return str(date)[:10] if pd.notna(date) else None

    @property
    def planned_total_km(self) -> Optional[float]:
        """Planned total kilometers."""
        km = self._data.get('seisPlannedTotalLengthKm')
        return float(km) if pd.notna(km) else None

    @property
    def geometry(self) -> Optional[dict]:
        """Survey geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Seismic('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "seismic")


class Stratigraphy(RelatedTableMixin):
    """
    Represents a stratigraphic unit (formation).

    Example:
        >>> fp = Factpages()
        >>> strat = fp.stratigraphy("DRAUPNE")
        >>> print(strat.level)
        >>> print(strat.type)
    """

    _entity_type = "stratigraphy"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        # Try lithostratigraphy ID first
        val = self._data.get('lsuNpdidLithoStrat')
        if pd.notna(val):
            return int(val)
        # Fall back to chronostratigraphy
        val = self._data.get('strNpdidChronoStrat')
        return int(val) if pd.notna(val) else 0

    @property
    def name(self) -> str:
        return self._data.get('lsuName', '') or self._data.get('strName', '')

    @property
    def level(self) -> str:
        """Stratigraphic level (GROUP, FORMATION, MEMBER, etc.)."""
        return self._data.get('lsuLevel', '') or self._data.get('strLevel', '')

    @property
    def strat_type(self) -> str:
        """Stratigraphy type (LITHO or CHRONO)."""
        if pd.notna(self._data.get('lsuNpdidLithoStrat')):
            return 'LITHO'
        return 'CHRONO'

    @property
    def parent(self) -> str:
        """Parent unit name."""
        return self._data.get('lsuParentName', '') or self._data.get('strParentName', '')

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Stratigraphy('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "stratigraphy")


class BusinessArrangement(RelatedTableMixin):
    """
    Represents a Business Arrangement Area (unitization agreement).

    Business arrangements govern shared resources across license boundaries.

    Example:
        >>> fp = Factpages()
        >>> ba = fp.business_arrangement("TROLL UNIT")
        >>> print(ba.status)
        >>> print(ba.licensees)
    """

    _entity_type = "business_arrangement"

    def __init__(self, data: pd.Series, db: "Database"):
        self._data = data
        self._db = db

    @property
    def id(self) -> int:
        return int(self._data.get('baaNpdidBsnsArrArea', 0))

    @property
    def name(self) -> str:
        return self._data.get('baaName', '')

    @property
    def status(self) -> str:
        """Current status."""
        return self._data.get('baaStatus', '')

    @property
    def kind(self) -> str:
        """Arrangement kind/type."""
        return self._data.get('baaKind', '')

    @property
    def date_approved(self) -> Optional[str]:
        """Date approved."""
        date = self._data.get('baaDateApproved')
        return str(date)[:10] if pd.notna(date) else None

    @property
    def operator(self) -> str:
        """Current operator."""
        return self._data.get('cmpLongName', '')

    @property
    def licensees(self) -> EntityDataFrame:
        """Business arrangement licensees."""
        licensees = self._db.get_or_none('business_arrangement_licensee_hst')
        if licensees is None:
            return EntityDataFrame(entity_type="BA Licensees", field_name=self.name)
        filtered = licensees[licensees['baaNpdidBsnsArrArea'] == self.id]
        return EntityDataFrame(filtered, entity_type="BA Licensees", field_name=self.name)

    @property
    def geometry(self) -> Optional[dict]:
        """Business arrangement geometry as GeoJSON."""
        import json
        geom_str = self._data.get('_geometry')
        if geom_str and isinstance(geom_str, str):
            return json.loads(geom_str)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._lookup_related_table(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"BusinessArrangement('{self.name}')"

    def __str__(self) -> str:
        from .display import render_entity
        return render_entity(self, "business_arrangement")
