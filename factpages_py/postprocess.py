"""
Post-processing transforms applied after API download.

Normalizes data for consistent downstream use:
- Adds integer ocean_id to tables with mainArea columns
- Adds integer qadId from quadrant name (e.g. "630W" → 14)
"""

import pandas as pd


# =============================================================================
# Ocean ID mapping
# =============================================================================

OCEAN_MAP = {"NORTH SEA": 0, "NORWEGIAN SEA": 1, "BARENTS SEA": 2}

# Tables that have a mainArea column → get ocean_id added
OCEAN_COLUMNS = {
    "field": "fldMainArea",
    "discovery": "nmaName",
    "wellbore": "wlbMainArea",
    "licence": "prlMainArea",
    "block": "blcMainArea",
    "play": "plyRegion",
}


# =============================================================================
# Main dispatcher
# =============================================================================

def postprocess(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dataset-specific post-processing after download.

    Called automatically by Factpages.download() and Factpages.reprocess().

    Args:
        dataset: Dataset name (e.g., 'field', 'quadrant')
        df: Downloaded DataFrame

    Returns:
        Transformed DataFrame
    """
    if df.empty:
        return df

    df = df.copy()

    # Add ocean_id to tables with mainArea columns
    if dataset in OCEAN_COLUMNS:
        col = OCEAN_COLUMNS[dataset]
        if col in df.columns:
            df["ocean_id"] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(OCEAN_MAP)
            )

    # Quadrant & Block: add integer qadId from qadName (keep qadName as string)
    if dataset in ("quadrant", "block"):
        df = _add_qad_id(df)

    return df


def _add_qad_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add qadId (int) derived from qadName (string). Maps '630W' → 14."""
    if "qadName" not in df.columns:
        return df
    df["qadName"] = df["qadName"].astype(str).str.strip()
    qad_int = df["qadName"].replace("630W", "14")
    df["qadId"] = pd.to_numeric(qad_int, errors="coerce").astype("Int64")
    return df
