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

    def __init__(self, partners: list, field_name: str = ""):
        super().__init__(partners)
        self.field_name = field_name

    def __str__(self) -> str:
        if not self:
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


class Field:
    """
    Represents a petroleum field on the Norwegian Continental Shelf.

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
        """Production start date."""
        return self._get_column('fldProdStartDate', None)

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

    @property
    def partners(self) -> PartnersList:
        """
        Current field partners (licensees) with their equity shares.

        Returns:
            PartnersList with 'company', 'share', and 'is_operator' keys.
            Print it for a nice formatted table.

        Example:
            >>> print(troll.partners)

            Partners (5):
            ┌──────────────────────────────────────────┬─────────┬──────────┐
            │ Company                                  │ Share % │ Operator │
            ├──────────────────────────────────────────┼─────────┼──────────┤
            │ Equinor Energy AS                        │   30.58 │    ✓     │
            ...
        """
        if self._partners_cache is None:
            licensees = self._db.get_or_none('field_licensee')
            if licensees is not None:
                self._partners_cache = licensees[
                    licensees['fldNpdidField'] == self.id
                ]
            else:
                self._partners_cache = pd.DataFrame()

        if self._partners_cache.empty:
            return PartnersList([], field_name=self.name)

        # Get current partners (those without end date or future end date)
        current = self._partners_cache.copy()
        today = datetime.now().strftime('%Y-%m-%d')

        if 'fldLicenseeDateValidTo' in current.columns:
            current = current[
                current['fldLicenseeDateValidTo'].isna() |
                (current['fldLicenseeDateValidTo'] >= today)
            ]

        partners = []
        for _, row in current.iterrows():
            partners.append({
                'company': row.get('cmpLongName', ''),
                'share': float(row.get('fldLicenseeInterest', 0)),
                'is_operator': row.get('cmpLongName', '') == self.operator,
            })

        partners = sorted(partners, key=lambda x: x['share'], reverse=True)
        return PartnersList(partners, field_name=self.name)

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

    # =========================================================================
    # Production Methods
    # =========================================================================

    def _load_production(self) -> None:
        """Load production data into cache."""
        if self._production_cache is None:
            production = self._db.get_or_none('field_production_monthly')
            if production is not None:
                self._production_cache = production[
                    production['fldNpdidField'] == self.id
                ]
            else:
                self._production_cache = pd.DataFrame()

    def production(self, year: int, month: int) -> dict:
        """
        Get production figures for a specific month.

        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)

        Returns:
            Dict with 'oil_sm3', 'gas_msm3', 'water_sm3', etc.
            Empty dict if no data available.
        """
        self._load_production()

        if self._production_cache.empty:
            return {}

        # Filter by year and month
        filtered = self._production_cache[
            (self._production_cache['prfYear'] == year) &
            (self._production_cache['prfMonth'] == month)
        ]

        if filtered.empty:
            return {}

        row = filtered.iloc[0]
        return {
            'year': year,
            'month': month,
            'oil_sm3': float(row.get('prfPrdOilNetMillSm3', 0) or 0) * 1_000_000,
            'gas_msm3': float(row.get('prfPrdGasNetBillSm3', 0) or 0) * 1_000,
            'ngl_tonnes': float(row.get('prfPrdNGLNetMillTonnes', 0) or 0) * 1_000_000,
            'condensate_sm3': float(row.get('prfPrdCondensateNetMillSm3', 0) or 0) * 1_000_000,
            'water_sm3': float(row.get('prfPrdProducedWaterInFieldMillSm3', 0) or 0) * 1_000_000,
        }

    def production_yearly(self, year: int) -> dict:
        """
        Get total production for a year.

        Args:
            year: Year (e.g., 2024)

        Returns:
            Dict with yearly totals
        """
        self._load_production()

        if self._production_cache.empty:
            return {}

        # Filter by year and sum
        filtered = self._production_cache[
            self._production_cache['prfYear'] == year
        ]

        if filtered.empty:
            return {}

        return {
            'year': year,
            'oil_sm3': float(filtered['prfPrdOilNetMillSm3'].sum() or 0) * 1_000_000,
            'gas_msm3': float(filtered['prfPrdGasNetBillSm3'].sum() or 0) * 1_000,
            'ngl_tonnes': float(filtered['prfPrdNGLNetMillTonnes'].sum() or 0) * 1_000_000,
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

        # Build clean DataFrame
        clean_df = pd.DataFrame({
            'year': df['prfYear'].astype(int),
            'month': df['prfMonth'].astype(int),
            'oil_sm3': (df['prfPrdOilNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'gas_sm3': (df['prfPrdGasNetBillSm3'].fillna(0) * 1_000_000_000).astype(float),
            'ngl_tonnes': (df['prfPrdNGLNetMillTonnes'].fillna(0) * 1_000_000).astype(float),
            'condensate_sm3': (df['prfPrdCondensateNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'oe_sm3': (df['prfPrdOeNetMillSm3'].fillna(0) * 1_000_000).astype(float),
            'water_sm3': (df['prfPrdProducedWaterInFieldMillSm3'].fillna(0) * 1_000_000).astype(float),
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
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Field('{self.name}')"

    def __str__(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"FIELD: {self.name}",
            f"{'=' * 60}",
            f"Status:     {self.status}",
            f"Operator:   {self.operator}",
            f"HC Type:    {self.hc_type}",
            f"Main Area:  {self.main_area}",
        ]

        if self.discovery_year:
            lines.append(f"Discovered: {self.discovery_year}")
        if self.production_start:
            lines.append(f"Prod Start: {self.production_start}")

        # Add reserves
        reserves = self.reserves
        if reserves:
            lines.append(f"\nReserves:")
            if reserves.get('oil_msm3', 0) > 0:
                lines.append(f"  Oil:         {reserves['oil_msm3']:,.1f} mill Sm3")
            if reserves.get('gas_bsm3', 0) > 0:
                lines.append(f"  Gas:         {reserves['gas_bsm3']:,.1f} bill Sm3")
            if reserves.get('ngl_mtoe', 0) > 0:
                lines.append(f"  NGL:         {reserves['ngl_mtoe']:,.1f} mill toe")
            if reserves.get('condensate_msm3', 0) > 0:
                lines.append(f"  Condensate:  {reserves['condensate_msm3']:,.1f} mill Sm3")

        # Add cumulative production
        try:
            history = self.production_history(clean=True)
            if not history.empty:
                total_oil = history['oil_msm3'].sum() if 'oil_msm3' in history.columns else 0
                total_gas = history['gas_bsm3'].sum() if 'gas_bsm3' in history.columns else 0
                if total_oil > 0 or total_gas > 0:
                    lines.append(f"\nProduced (cumulative):")
                    if total_oil > 0:
                        lines.append(f"  Oil:         {total_oil:,.1f} mill Sm3")
                    if total_gas > 0:
                        lines.append(f"  Gas:         {total_gas:,.1f} bill Sm3")
        except Exception:
            pass  # Skip if production data unavailable

        # Add partners summary
        partners = self.partners
        if partners:
            lines.append(f"\nPartners ({len(partners)}):")
            for p in partners[:5]:  # Show top 5
                op_marker = " *" if p['is_operator'] else ""
                lines.append(f"  {p['company']:<40} {p['share']:>6.2f}%{op_marker}")
            if len(partners) > 5:
                lines.append(f"  ... and {len(partners) - 5} more")

        return '\n'.join(lines)


class Discovery:
    """
    Represents a petroleum discovery on the Norwegian Continental Shelf.

    Example:
        >>> fp = Factpages()
        >>> johan = fp.discovery("JOHAN SVERDRUP")
        >>> print(johan)  # Shows geology-focused summary
        >>> print(johan.wells)  # All wells on discovery
        >>> print(johan.resources)  # Resource estimates
    """

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
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Discovery('{self.name}')"

    def __str__(self) -> str:
        resources = self.resources

        lines = [
            f"\nDiscovery: {self.name}",
            f"{'=' * 55}",
        ]

        # Key info: year, type, status
        year_str = str(self.discovery_year) if self.discovery_year else "?"
        lines.append(f"Discovered: {year_str:<8}  HC Type: {self.hc_type}")
        lines.append(f"Status:     {self.status}")
        lines.append(f"Area:       {self.main_area}")

        # Discovery well - critical for geologists!
        if self.discovery_well:
            lines.append(f"Discovery Well: {self.discovery_well}")

        lines.append(f"Operator:   {self.operator}")

        # Development status
        if self.field_name:
            lines.append("")
            lines.append(f"→ Developed as field: {self.field_name}")

        # Resources (the key numbers!)
        if resources:
            lines.append("")
            lines.append("Recoverable Resources:")
            res_parts = []
            if resources.get('oil_msm3'):
                res_parts.append(f"Oil: {resources['oil_msm3']:.1f} MSm³")
            if resources.get('gas_bsm3'):
                res_parts.append(f"Gas: {resources['gas_bsm3']:.1f} BSm³")
            if resources.get('condensate_msm3'):
                res_parts.append(f"Cond: {resources['condensate_msm3']:.1f} MSm³")
            if res_parts:
                lines.append(f"  {', '.join(res_parts)}")

        # Exploration hints
        lines.append("")
        lines.append("Explore: .wells  .resources_history()  .discovery_well")

        return '\n'.join(lines)


class Wellbore:
    """
    Represents a wellbore drilled on the Norwegian Continental Shelf.

    Example:
        >>> fp = Factpages()
        >>> well = fp.well("31/2-1")
        >>> print(well)  # Shows key geology info
        >>> print(well.formation_tops)  # Stratigraphy
        >>> print(well.dst_results)  # Flow tests
    """

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
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Wellbore('{self.name}')"

    def __str__(self) -> str:
        lines = [
            f"\nWellbore: {self.name}",
            f"{'=' * 55}",
        ]

        # Classification
        lines.append(f"Purpose:  {self.purpose:<12}  Content: {self.content}")
        lines.append(f"Status:   {self.status:<12}  Area:    {self.main_area}")

        # Depths
        depth_info = []
        if self.total_depth:
            depth_info.append(f"TD: {self.total_depth:.0f}m")
        if self.water_depth:
            depth_info.append(f"WD: {self.water_depth:.0f}m")
        if depth_info:
            lines.append(f"Depth:    {', '.join(depth_info)}")

        # Dates and operator
        if self.completion_date:
            lines.append(f"Completed: {self.completion_date}")
        lines.append(f"Operator:  {self.operator}")

        # HC bearing formations (key geological info!)
        if self.hc_formations:
            lines.append("")
            lines.append(f"HC Formations: {', '.join(self.hc_formations)}")
        if self.hc_ages:
            lines.append(f"HC Ages:       {', '.join(self.hc_ages)}")

        # Associations
        if self.field_name or self.discovery_name:
            lines.append("")
            if self.field_name:
                lines.append(f"Field:     {self.field_name}")
            if self.discovery_name:
                lines.append(f"Discovery: {self.discovery_name}")

        # Exploration hints
        lines.append("")
        lines.append("Explore: .formation_tops  .dst_results  .cores  .drilling_history  .casing")

        return '\n'.join(lines)


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


class Company:
    """
    Represents a company operating on the NCS.

    Example:
        >>> fp = Factpages()
        >>> equinor = fp.company("equinor")
        >>> print(equinor.name)
        >>> print(equinor.field_interests)
        >>> print(equinor.operated_fields)
    """

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
            licensees = self._db.get_or_none('field_licensee')
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
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"Company('{self.name}')"

    def __str__(self) -> str:
        interests = self.field_interests
        operated = len([i for i in interests if i.get('is_operator')])

        lines = [
            f"\nCompany: {self.name}",
            f"{'=' * 55}",
            f"Nation: {self.nation:<10}  Org#: {self.org_number}",
            f"Fields: {len(interests):<10}  Operated: {operated}",
        ]

        if interests:
            lines.append(f"\nTop equity positions (* = operator):")
            for i in interests[:5]:
                op_mark = "*" if i.get('is_operator') else " "
                lines.append(f"  {op_mark} {i['field']:<28} {i['share']:>6.2f}%")

        lines.append("")
        lines.append("Explore: .field_interests  .operated_fields  .wells_drilled")
        return '\n'.join(lines)


class License:
    """
    Represents a production license on the Norwegian Continental Shelf.

    Example:
        >>> fp = Factpages()
        >>> pl001 = fp.license("PL001")
        >>> print(pl001)  # Shows key license info
        >>> print(pl001.licensees)  # Current licensees
        >>> print(pl001.fields)  # Related fields
    """

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
            licensees = self._db.get_or_none('licence_licensee_history')
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
        phases = self._db.get_or_none('licence_phase_history')
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
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        return f"License('{self.name}')"

    def __str__(self) -> str:
        licensees = self.licensees

        lines = [
            f"\nLicense: {self.name}",
            f"{'=' * 55}",
            f"Status:   {self.status:<12}  Phase: {self.current_phase}",
            f"Area:     {self.main_area}",
            f"Operator: {self.operator}",
        ]

        # Dates
        if self.date_granted:
            lines.append(f"Granted:  {self.date_granted}")
        if self.date_valid_to:
            lines.append(f"Expires:  {self.date_valid_to}")

        # Licensees summary
        if licensees:
            lines.append(f"\nLicensees ({len(licensees)}):")
            for p in licensees[:5]:
                op_mark = "*" if p.get('is_operator') else " "
                lines.append(f"  {op_mark} {p['company'][:28]:<28} {p['share']:>6.2f}%")
            if len(licensees) > 5:
                lines.append(f"  ... and {len(licensees) - 5} more")

        lines.append("")
        lines.append("Explore: .licensees  .fields  .discoveries  .wells")
        lines.append("         .ownership_history  .phase_history  .work_obligations")
        return '\n'.join(lines)
