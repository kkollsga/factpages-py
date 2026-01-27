"""
Analysis Module

High-level analysis methods for petroleum data.
These methods provide user-friendly interfaces for common queries
and return formatted text output.
"""

from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from .database import Database


class AnalysisMixin:
    """
    Mixin class providing analysis methods.

    This is mixed into the main Factpages class to provide
    high-level analysis capabilities while keeping the code
    compartmentalized.
    """

    # These will be set by the main class
    db: "Database"

    # =========================================================================
    # Field Analysis
    # =========================================================================

    def field_summary(self, name: str) -> str:
        """
        Get a comprehensive summary of a field.

        Args:
            name: Field name (case-insensitive)

        Returns:
            Formatted text summary

        Example:
            >>> api = Factpages()
            >>> print(api.field_summary('TROLL'))
        """
        from .formatters import format_field_summary

        # Get field data
        fields = self.db.get_or_none('field')
        if fields is None:
            return f"Field data not available. Run sync first."

        # Find field (case-insensitive)
        name_upper = name.upper()
        field = fields[fields['fldName'].str.upper() == name_upper]

        if field.empty:
            return f"Field '{name}' not found."

        field_row = field.iloc[0]
        field_id = field_row.get('fldNpdidField')
        field_name = field_row.get('fldName')

        # Get related data
        production = self._get_field_production(field_id, field_name)
        reserves = self._get_field_reserves(field_id)
        licensees = self._get_field_licensees(field_id)
        operators = self._get_field_operators(field_id)

        return format_field_summary(
            field=field_row,
            production=production,
            reserves=reserves,
            licensees=licensees,
            operators=operators,
        )

    def _get_field_production(self, field_id: int, field_name: str = None) -> Optional[pd.DataFrame]:
        """Get production data for a field from profiles table."""
        prod = self.db.get_or_none('profiles')
        if prod is None:
            return None
        # Profiles uses prfNpdidInformationCarrier or prfInformationCarrier (name)
        if field_name:
            return prod[prod['prfInformationCarrier'] == field_name.upper()]
        return prod[prod['prfNpdidInformationCarrier'] == field_id]

    def _get_field_reserves(self, field_id: int) -> Optional[pd.DataFrame]:
        """Get reserves data for a field."""
        reserves = self.db.get_or_none('field_reserves')
        if reserves is None:
            return None
        return reserves[reserves['fldNpdidField'] == field_id]

    def _get_field_licensees(self, field_id: int) -> Optional[pd.DataFrame]:
        """Get licensee data for a field."""
        licensees = self.db.get_or_none('field_licensee_hst')
        if licensees is None:
            return None
        return licensees[licensees['fldNpdidField'] == field_id]

    def _get_field_operators(self, field_id: int) -> Optional[pd.DataFrame]:
        """Get operator history for a field."""
        operators = self.db.get_or_none('field_operator_hst')
        if operators is None:
            return None
        return operators[operators['fldNpdidField'] == field_id]

    # =========================================================================
    # Well Analysis
    # =========================================================================

    def well_info(self, name: str) -> str:
        """
        Get information about a wellbore.

        Args:
            name: Wellbore name (e.g., '35/11-25')

        Returns:
            Formatted text summary
        """
        from .formatters import format_well_info

        wellbores = self.db.get_or_none('wellbore')
        if wellbores is None:
            return "Wellbore data not available. Run sync first."

        # Find wellbore (exact or partial match)
        well = wellbores[wellbores['wlbWellboreName'] == name]
        if well.empty:
            # Try partial match
            well = wellbores[wellbores['wlbWellboreName'].str.contains(name, case=False, na=False)]

        if well.empty:
            return f"Wellbore '{name}' not found."

        return format_well_info(well.iloc[0])

    def active_wells(self) -> str:
        """
        Get currently drilling wells.

        Returns:
            Formatted text listing active wells
        """
        from .formatters import format_active_wells

        wellbores = self.db.get_or_none('wellbore')
        if wellbores is None:
            return "Wellbore data not available. Run sync first."

        # Filter for drilling status
        active = wellbores[wellbores['wlbStatus'] == 'DRILLING']

        if active.empty:
            return "No wells currently drilling."

        return format_active_wells(active)

    # =========================================================================
    # Discovery Analysis
    # =========================================================================

    def discovery_info(self, name: str) -> str:
        """
        Get information about a discovery.

        Args:
            name: Discovery name

        Returns:
            Formatted text summary
        """
        from .formatters import format_discovery_info

        discoveries = self.db.get_or_none('discovery')
        if discoveries is None:
            return "Discovery data not available. Run sync first."

        # Find discovery (case-insensitive)
        name_upper = name.upper()
        disc = discoveries[discoveries['dscName'].str.upper() == name_upper]

        if disc.empty:
            return f"Discovery '{name}' not found."

        return format_discovery_info(disc.iloc[0])

    def recent_discoveries(self, years: int = 3) -> str:
        """
        List recent discoveries.

        Args:
            years: Number of years to look back

        Returns:
            Formatted text listing
        """
        from datetime import datetime
        from .formatters import format_discovery_list

        discoveries = self.db.get_or_none('discovery')
        if discoveries is None:
            return "Discovery data not available. Run sync first."

        # Filter by discovery year
        current_year = datetime.now().year
        min_year = current_year - years

        # Convert discovery year if present
        if 'dscDiscoveryYear' in discoveries.columns:
            recent = discoveries[discoveries['dscDiscoveryYear'] >= min_year]
        else:
            recent = discoveries.tail(20)  # Fallback to recent records

        if recent.empty:
            return f"No discoveries found in the last {years} years."

        return format_discovery_list(recent, years)

    # =========================================================================
    # License Analysis
    # =========================================================================

    def license_info(self, name: str) -> str:
        """
        Get information about a license.

        Args:
            name: License name (e.g., 'PL001')

        Returns:
            Formatted text summary
        """
        from .formatters import format_license_info

        licences = self.db.get_or_none('licence')
        if licences is None:
            return "License data not available. Run sync first."

        # Find license
        lic = licences[licences['prlName'] == name]
        if lic.empty:
            # Try partial match
            lic = licences[licences['prlName'].str.contains(name, case=False, na=False)]

        if lic.empty:
            return f"License '{name}' not found."

        return format_license_info(lic.iloc[0])

    # =========================================================================
    # Company Analysis
    # =========================================================================

    def company_portfolio(self, name: str) -> str:
        """
        Get a company's license portfolio.

        Args:
            name: Company name (partial match supported)

        Returns:
            Formatted text summary
        """
        from .formatters import format_company_portfolio

        companies = self.db.get_or_none('company')
        if companies is None:
            return "Company data not available. Run sync first."

        # Find company (partial match)
        company = companies[companies['cmpLongName'].str.contains(name, case=False, na=False)]

        if company.empty:
            return f"Company matching '{name}' not found."

        # Get licensee history to find portfolio
        licensee_history = self.db.get_or_none('licence_licensee_history')

        return format_company_portfolio(company.iloc[0], licensee_history)

    # =========================================================================
    # Production Analysis
    # =========================================================================

    def production_ranking(self, n: int = 10, hc_type: str = "all") -> str:
        """
        Get top producing fields.

        Args:
            n: Number of fields to show
            hc_type: 'oil', 'gas', or 'all'

        Returns:
            Formatted text ranking
        """
        from .formatters import format_production_ranking

        production = self.db.get_or_none('field_production_monthly')
        fields = self.db.get_or_none('field')

        if production is None or fields is None:
            return "Production data not available. Run sync first."

        return format_production_ranking(production, fields, n, hc_type)

    # =========================================================================
    # Data Endpoints (for visualization tools)
    # =========================================================================

    def get_production_timeseries(self, field_name: str) -> pd.DataFrame:
        """
        Get production time series for a field.

        Args:
            field_name: Field name

        Returns:
            DataFrame with monthly production data
        """
        fields = self.db.get_or_none('field')
        production = self.db.get_or_none('field_production_monthly')

        if fields is None or production is None:
            return pd.DataFrame()

        # Find field ID
        name_upper = field_name.upper()
        field = fields[fields['fldName'].str.upper() == name_upper]

        if field.empty:
            return pd.DataFrame()

        field_id = field.iloc[0]['fldNpdidField']

        # Get production for this field
        return production[production['fldNpdidField'] == field_id].copy()

    def get_field_geometry(self, field_name: str) -> Optional[dict]:
        """
        Get field geometry as GeoJSON.

        Args:
            field_name: Field name

        Returns:
            GeoJSON dict or None if not found
        """
        import json

        fields = self.db.get_or_none('field')
        if fields is None:
            return None

        name_upper = field_name.upper()
        field = fields[fields['fldName'].str.upper() == name_upper]

        if field.empty:
            return None

        geometry_str = field.iloc[0].get('_geometry')
        if geometry_str:
            return json.loads(geometry_str)
        return None

    def get_well_coordinates(self) -> pd.DataFrame:
        """
        Get all well coordinates.

        Returns:
            DataFrame with well names and coordinates
        """
        wellbores = self.db.get_or_none('wellbore')
        if wellbores is None:
            return pd.DataFrame()

        cols = ['wlbWellboreName', 'wlbNsDecDeg', 'wlbEwDecDeg',
                'wlbStatus', 'wlbPurpose', 'wlbContent']
        available_cols = [c for c in cols if c in wellbores.columns]

        return wellbores[available_cols].copy()
