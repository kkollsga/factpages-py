"""
Text Formatters

Functions for formatting petroleum data as readable text output.
Designed for terminal/REPL usage where quick answers are needed.
"""

from typing import Optional

import pandas as pd


def format_field_summary(
    field: pd.Series,
    production: Optional[pd.DataFrame] = None,
    reserves: Optional[pd.DataFrame] = None,
    licensees: Optional[pd.DataFrame] = None,
    operators: Optional[pd.DataFrame] = None,
) -> str:
    """Format a complete field summary."""
    lines = []

    name = field.get('fldName', 'Unknown')

    lines.append("")
    lines.append(name)
    lines.append("=" * 60)

    # Basic info
    lines.append(f"Status:     {field.get('fldStatus', 'N/A')}")
    lines.append(f"Operator:   {field.get('fldOperatorCompanyName', 'N/A')}")
    lines.append(f"HC Type:    {field.get('fldHcType', 'N/A')}")
    lines.append(f"Main Area:  {field.get('fldMainArea', 'N/A')}")

    # Production info
    if production is not None and not production.empty:
        latest = production.sort_values(['prfYear', 'prfMonth'], ascending=False).iloc[0]
        lines.append("")
        lines.append(f"CURRENT PRODUCTION ({int(latest.get('prfMonth', 0))}/{int(latest.get('prfYear', 0))})")

        oil = latest.get('prfPrdOilNetMillSm3', 0) or 0
        gas = latest.get('prfPrdGasNetBillSm3', 0) or 0

        if oil > 0:
            lines.append(f"  Oil:  {oil * 1000:.1f} kSm3/month")
        if gas > 0:
            lines.append(f"  Gas:  {gas * 1000:.1f} MSm3/month")

    # Reserves info
    if reserves is not None and not reserves.empty:
        latest = reserves.iloc[-1]
        lines.append("")
        lines.append("RESERVES (Remaining)")

        oil = latest.get('fldRecoverableOil', 0) or 0
        gas = latest.get('fldRecoverableGas', 0) or 0

        if oil > 0:
            lines.append(f"  Oil:  {oil:.1f} mill Sm3")
        if gas > 0:
            lines.append(f"  Gas:  {gas:.1f} bill Sm3")

    # Ownership
    if licensees is not None and not licensees.empty:
        lines.append("")
        lines.append("OWNERSHIP")

        for _, row in licensees.head(5).iterrows():
            company = row.get('cmpLongName', 'Unknown')
            share = row.get('fldLicenseeInterest', 0) or 0
            lines.append(f"  {company:<40} {share:>6.2f}%")

        if len(licensees) > 5:
            lines.append(f"  ... and {len(licensees) - 5} more")

    return '\n'.join(lines)


def format_well_info(well: pd.Series) -> str:
    """Format wellbore information."""
    lines = []

    name = well.get('wlbWellboreName', 'Unknown')

    lines.append("")
    lines.append(f"{name} - {well.get('wlbPurpose', '')} Well")
    lines.append("=" * 60)

    lines.append(f"Status:       {well.get('wlbStatus', 'N/A')}")
    lines.append(f"Operator:     {well.get('wlbDrillingOperator', 'N/A')}")
    lines.append(f"Content:      {well.get('wlbContent', 'N/A')}")

    # Location
    lines.append("")
    lines.append("LOCATION")
    lat = well.get('wlbNsDecDeg')
    lon = well.get('wlbEwDecDeg')
    if lat and lon:
        lines.append(f"  Latitude:   {lat:.4f} N")
        lines.append(f"  Longitude:  {lon:.4f} E")

    wd = well.get('wlbWaterDepth')
    if wd:
        lines.append(f"  Water depth: {wd:.0f} m")

    # Drilling info
    td = well.get('wlbTotalDepth')
    if td:
        lines.append("")
        lines.append("DRILLING")
        lines.append(f"  Total depth: {td:.0f} m")

    # Field/Discovery
    field = well.get('fldName')
    disc = well.get('dscName')
    if field or disc:
        lines.append("")
        if field:
            lines.append(f"Field:      {field}")
        if disc:
            lines.append(f"Discovery:  {disc}")

    return '\n'.join(lines)


def format_active_wells(wells: pd.DataFrame) -> str:
    """Format list of active/drilling wells."""
    lines = []

    lines.append("")
    lines.append(f"{len(wells)} wells currently drilling:")
    lines.append("")

    for _, well in wells.iterrows():
        name = well.get('wlbWellboreName', 'Unknown')
        operator = well.get('wlbDrillingOperator', 'N/A')
        field = well.get('fldName', '')

        lines.append(f"  {name:<15} {operator:<20} {field}")

    return '\n'.join(lines)


def format_discovery_info(discovery: pd.Series) -> str:
    """Format discovery information."""
    lines = []

    name = discovery.get('dscName', 'Unknown')

    lines.append("")
    lines.append(f"{name}")
    lines.append("=" * 60)

    lines.append(f"Status:     {discovery.get('dscCurrentActivityStatus', 'N/A')}")
    lines.append(f"HC Type:    {discovery.get('dscHcType', 'N/A')}")
    lines.append(f"Main Area:  {discovery.get('dscMainArea', 'N/A')}")
    lines.append(f"Operator:   {discovery.get('dscOperatorCompanyName', 'N/A')}")

    year = discovery.get('dscDiscoveryYear')
    if year:
        lines.append(f"Discovered: {int(year)}")

    well = discovery.get('dscDiscoveryWellbore')
    if well:
        lines.append(f"Disc. well: {well}")

    field = discovery.get('fldName')
    if field:
        lines.append(f"Developed:  {field}")

    return '\n'.join(lines)


def format_discovery_list(discoveries: pd.DataFrame, years: int) -> str:
    """Format list of recent discoveries."""
    lines = []

    lines.append("")
    lines.append(f"Discoveries (last {years} years):")
    lines.append("")

    # Sort by year descending
    if 'dscDiscoveryYear' in discoveries.columns:
        discoveries = discoveries.sort_values('dscDiscoveryYear', ascending=False)

    for _, disc in discoveries.head(20).iterrows():
        name = disc.get('dscName', 'Unknown')
        year = disc.get('dscDiscoveryYear', '')
        hc = disc.get('dscHcType', '')
        operator = disc.get('dscOperatorCompanyName', 'N/A')
        status = disc.get('dscCurrentActivityStatus', '')

        year_str = f"({int(year)})" if year else ""
        lines.append(f"  {name:<20} {year_str:<8} {hc:<10} {operator:<20} {status}")

    if len(discoveries) > 20:
        lines.append(f"  ... and {len(discoveries) - 20} more")

    return '\n'.join(lines)


def format_license_info(license_data: pd.Series) -> str:
    """Format license information."""
    lines = []

    name = license_data.get('prlName', 'Unknown')

    lines.append("")
    lines.append(f"{name}")
    lines.append("=" * 60)

    lines.append(f"Status:    {license_data.get('prlStatus', 'N/A')}")
    lines.append(f"Operator:  {license_data.get('prlOperatorCompanyName', 'N/A')}")
    lines.append(f"Main Area: {license_data.get('prlMainArea', 'N/A')}")

    date = license_data.get('prlDateGranted')
    if date:
        lines.append(f"Granted:   {date}")

    return '\n'.join(lines)


def format_company_portfolio(
    company: pd.Series,
    licensee_history: Optional[pd.DataFrame] = None
) -> str:
    """Format company portfolio summary."""
    lines = []

    name = company.get('cmpLongName', 'Unknown')

    lines.append("")
    lines.append(f"{name}")
    lines.append("=" * 60)

    short = company.get('cmpShortName')
    if short:
        lines.append(f"Short name: {short}")

    org = company.get('cmpOrgNumberBrReg')
    if org:
        lines.append(f"Org number: {org}")

    nation = company.get('cmpNationCode')
    if nation:
        lines.append(f"Nation:     {nation}")

    # License count if available
    if licensee_history is not None and not licensee_history.empty:
        company_id = company.get('cmpNpdidCompany')
        company_licenses = licensee_history[
            licensee_history['cmpNpdidCompany'] == company_id
        ]
        if not company_licenses.empty:
            lines.append(f"\nLicenses:   {len(company_licenses.groupby('prlNpdidLicence'))}")

    return '\n'.join(lines)


def format_production_ranking(
    production: pd.DataFrame,
    fields: pd.DataFrame,
    n: int = 10,
    hc_type: str = "all"
) -> str:
    """Format top producing fields."""
    lines = []

    lines.append("")
    lines.append(f"Top {n} Producing Fields")
    lines.append("=" * 60)

    # Get latest month for each field
    latest = production.sort_values(['prfYear', 'prfMonth'], ascending=False)
    latest = latest.groupby('fldNpdidField').first().reset_index()

    # Merge with field names
    merged = latest.merge(
        fields[['fldNpdidField', 'fldName', 'fldHcType']],
        on='fldNpdidField',
        how='left'
    )

    # Filter by HC type if specified
    if hc_type.lower() == 'oil':
        merged = merged.sort_values('prfPrdOilNetMillSm3', ascending=False)
        value_col = 'prfPrdOilNetMillSm3'
        unit = 'kSm3/mo'
    elif hc_type.lower() == 'gas':
        merged = merged.sort_values('prfPrdGasNetBillSm3', ascending=False)
        value_col = 'prfPrdGasNetBillSm3'
        unit = 'MSm3/mo'
    else:
        # Combined ranking by oil equivalent
        merged['total'] = (
            merged.get('prfPrdOilNetMillSm3', 0).fillna(0) +
            merged.get('prfPrdGasNetBillSm3', 0).fillna(0) * 0.001  # rough conversion
        )
        merged = merged.sort_values('total', ascending=False)
        value_col = 'total'
        unit = 'oil eq'

    for i, (_, row) in enumerate(merged.head(n).iterrows(), 1):
        name = row.get('fldName', 'Unknown')
        value = row.get(value_col, 0) or 0
        hc = row.get('fldHcType', '')

        lines.append(f"  {i:>2}. {name:<25} {value * 1000:>10.1f} {unit}  ({hc})")

    return '\n'.join(lines)
