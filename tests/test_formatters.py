"""
Tests for formatters.py - pure formatting functions.

These tests use mock pandas data and run very fast.
"""

import pandas as pd
import pytest

from factpages_py.formatters import (
    format_field_summary,
    format_well_info,
    format_active_wells,
    format_discovery_info,
    format_discovery_list,
    format_license_info,
    format_company_portfolio,
    format_production_ranking,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_field():
    """Mock field data as pandas Series."""
    return pd.Series({
        "fldName": "TROLL",
        "fldCurrentActivitySatus": "Producing",
        "cmpLongName": "Equinor Energy AS",
        "fldHcType": "OIL/GAS",
        "fldMainArea": "North Sea",
        "wlbName": "31/2-1",
        "fldDiscoveryYear": 1979,
        "fldNpdidField": 46437,
    })


@pytest.fixture
def mock_wellbore():
    """Mock wellbore data as pandas Series."""
    return pd.Series({
        "wlbWellboreName": "33/9-1",
        "wlbPurpose": "EXPLORATION",
        "wlbStatus": "P&A",
        "wlbDrillingOperator": "Equinor",
        "wlbContent": "OIL",
        "wlbNsDecDeg": 61.2345,
        "wlbEwDecDeg": 2.3456,
        "wlbWaterDepth": 350,
        "wlbTotalDepth": 4500,
        "fldName": "TROLL",
        "dscName": None,
    })


@pytest.fixture
def mock_discovery():
    """Mock discovery data as pandas Series."""
    return pd.Series({
        "dscName": "JOHAN SVERDRUP",
        "dscCurrentActivityStatus": "Producing",
        "dscHcType": "OIL",
        "dscMainArea": "North Sea",
        "dscOperatorCompanyName": "Equinor Energy AS",
        "dscDiscoveryYear": 2010,
        "dscDiscoveryWellbore": "16/2-6",
        "fldName": "JOHAN SVERDRUP",
    })


@pytest.fixture
def mock_license():
    """Mock license data as pandas Series."""
    return pd.Series({
        "prlName": "PL001",
        "prlStatus": "ACTIVE",
        "prlOperatorCompanyName": "Equinor Energy AS",
        "prlMainArea": "North Sea",
        "prlDateGranted": "1965-04-01",
    })


@pytest.fixture
def mock_company():
    """Mock company data as pandas Series."""
    return pd.Series({
        "cmpLongName": "Equinor Energy AS",
        "cmpShortName": "EQUINOR",
        "cmpOrgNumberBrReg": "123456789",
        "cmpNationCode": "NO",
        "cmpNpdidCompany": 12345,
    })


# =============================================================================
# format_field_summary Tests
# =============================================================================

class TestFormatFieldSummary:
    def test_basic_field_info(self, mock_field):
        result = format_field_summary(mock_field)

        assert "TROLL" in result
        assert "Producing" in result
        assert "Equinor Energy AS" in result
        assert "OIL/GAS" in result
        assert "North Sea" in result
        assert "1979" in result

    def test_with_reserves(self, mock_field):
        reserves = pd.DataFrame({
            "fldVersion": [2023],
            "fldRemainingOil": [10.5],
            "fldRemainingGas": [25.3],
            "fldRemainingNGL": [2.1],
            "fldRemainingCondensate": [0.5],
        })
        result = format_field_summary(mock_field, reserves=reserves)

        assert "RESERVES" in result
        assert "10.5" in result  # Oil
        assert "25.3" in result  # Gas

    def test_with_production(self, mock_field):
        production = pd.DataFrame({
            "prfYear": [2024],
            "prfMonth": [6],
            "prfPrdOilNetMillSm3": [0.5],
            "prfPrdGasNetBillSm3": [1.2],
            "prfPrdNGLNetMillSm3": [0.1],
            "prfPrdCondensateNetMillSm3": [0.05],
        })
        # Production is shown alongside reserves, so we need both
        reserves = pd.DataFrame({
            "fldVersion": [2023],
            "fldRemainingOil": [10.5],
            "fldRemainingGas": [25.3],
            "fldRemainingNGL": [0],
            "fldRemainingCondensate": [0],
        })
        result = format_field_summary(mock_field, production=production, reserves=reserves)

        assert "PRODUCTION" in result
        assert "Jun" in result  # Month name

    def test_with_licensees(self, mock_field):
        licensees = pd.DataFrame({
            "cmpLongName": ["Equinor Energy AS", "Petoro AS"],
            "fldCompanyShare": [60.0, 30.0],
            "fldLicenseeTo": [None, None],  # Current (not expired)
        })
        result = format_field_summary(mock_field, licensees=licensees)

        assert "EQUITY" in result
        assert "60.00%" in result
        assert "30.00%" in result

    def test_empty_reserves(self, mock_field):
        empty_reserves = pd.DataFrame()
        result = format_field_summary(mock_field, reserves=empty_reserves)
        assert "TROLL" in result  # Should still work

    def test_none_values(self, mock_field):
        result = format_field_summary(
            mock_field,
            production=None,
            reserves=None,
            licensees=None,
            operators=None
        )
        assert "TROLL" in result


# =============================================================================
# format_well_info Tests
# =============================================================================

class TestFormatWellInfo:
    def test_basic_well_info(self, mock_wellbore):
        result = format_well_info(mock_wellbore)

        assert "33/9-1" in result
        assert "EXPLORATION" in result
        assert "P&A" in result
        assert "Equinor" in result
        assert "OIL" in result

    def test_location_info(self, mock_wellbore):
        result = format_well_info(mock_wellbore)

        assert "LOCATION" in result
        assert "61.2345" in result  # Latitude
        assert "2.3456" in result   # Longitude
        assert "350" in result      # Water depth

    def test_drilling_info(self, mock_wellbore):
        result = format_well_info(mock_wellbore)

        assert "DRILLING" in result
        assert "4500" in result  # Total depth

    def test_field_association(self, mock_wellbore):
        result = format_well_info(mock_wellbore)
        assert "TROLL" in result

    def test_minimal_well(self):
        minimal = pd.Series({
            "wlbWellboreName": "TEST-1",
            "wlbPurpose": "TEST",
            "wlbStatus": "ACTIVE",
        })
        result = format_well_info(minimal)
        assert "TEST-1" in result
        assert "N/A" in result  # Missing values show N/A


# =============================================================================
# format_active_wells Tests
# =============================================================================

class TestFormatActiveWells:
    def test_multiple_wells(self):
        wells = pd.DataFrame({
            "wlbWellboreName": ["35/9-1", "35/9-2", "31/2-1"],
            "wlbDrillingOperator": ["Equinor", "Aker BP", "ConocoPhillips"],
            "fldName": ["TROLL", "VALHALL", "EKOFISK"],
        })
        result = format_active_wells(wells)

        assert "3 wells currently drilling" in result
        assert "35/9-1" in result
        assert "35/9-2" in result
        assert "31/2-1" in result
        assert "Equinor" in result
        assert "TROLL" in result

    def test_single_well(self):
        wells = pd.DataFrame({
            "wlbWellboreName": ["35/9-1"],
            "wlbDrillingOperator": ["Equinor"],
            "fldName": ["TROLL"],
        })
        result = format_active_wells(wells)
        assert "1 wells currently drilling" in result


# =============================================================================
# format_discovery_info Tests
# =============================================================================

class TestFormatDiscoveryInfo:
    def test_basic_discovery(self, mock_discovery):
        result = format_discovery_info(mock_discovery)

        assert "JOHAN SVERDRUP" in result
        assert "Producing" in result
        assert "OIL" in result
        assert "North Sea" in result
        assert "Equinor" in result
        assert "2010" in result
        assert "16/2-6" in result

    def test_developed_into_field(self, mock_discovery):
        result = format_discovery_info(mock_discovery)
        assert "Developed:" in result
        assert "JOHAN SVERDRUP" in result


# =============================================================================
# format_discovery_list Tests
# =============================================================================

class TestFormatDiscoveryList:
    def test_discovery_list(self):
        discoveries = pd.DataFrame({
            "dscName": ["DISC A", "DISC B", "DISC C"],
            "dscDiscoveryYear": [2023, 2022, 2021],
            "dscHcType": ["OIL", "GAS", "OIL/GAS"],
            "dscOperatorCompanyName": ["Equinor", "Aker BP", "Shell"],
            "dscCurrentActivityStatus": ["Active", "Active", "PDO approved"],
        })
        result = format_discovery_list(discoveries, years=3)

        assert "last 3 years" in result
        assert "DISC A" in result
        assert "DISC B" in result
        assert "(2023)" in result
        assert "(2022)" in result

    def test_more_than_20(self):
        # Create 25 discoveries
        discoveries = pd.DataFrame({
            "dscName": [f"DISC_{i}" for i in range(25)],
            "dscDiscoveryYear": [2023] * 25,
            "dscHcType": ["OIL"] * 25,
            "dscOperatorCompanyName": ["Equinor"] * 25,
            "dscCurrentActivityStatus": ["Active"] * 25,
        })
        result = format_discovery_list(discoveries, years=3)

        assert "... and 5 more" in result


# =============================================================================
# format_license_info Tests
# =============================================================================

class TestFormatLicenseInfo:
    def test_basic_license(self, mock_license):
        result = format_license_info(mock_license)

        assert "PL001" in result
        assert "ACTIVE" in result
        assert "Equinor" in result
        assert "North Sea" in result
        assert "1965-04-01" in result


# =============================================================================
# format_company_portfolio Tests
# =============================================================================

class TestFormatCompanyPortfolio:
    def test_basic_company(self, mock_company):
        result = format_company_portfolio(mock_company)

        assert "Equinor Energy AS" in result
        assert "EQUINOR" in result
        assert "123456789" in result
        assert "NO" in result

    def test_with_licensee_history(self, mock_company):
        licensee_history = pd.DataFrame({
            "cmpNpdidCompany": [12345, 12345, 12345],
            "prlNpdidLicence": [100, 200, 300],
        })
        result = format_company_portfolio(mock_company, licensee_history)

        assert "Licenses:" in result
        assert "3" in result


# =============================================================================
# format_production_ranking Tests
# =============================================================================

class TestFormatProductionRanking:
    @pytest.fixture
    def mock_production(self):
        return pd.DataFrame({
            "fldNpdidField": [1, 1, 2, 2, 3, 3],
            "prfYear": [2024, 2024, 2024, 2024, 2024, 2024],
            "prfMonth": [5, 6, 5, 6, 5, 6],
            "prfPrdOilNetMillSm3": [0.5, 0.6, 0.3, 0.35, 0.8, 0.9],
            "prfPrdGasNetBillSm3": [1.0, 1.1, 0.5, 0.55, 2.0, 2.2],
        })

    @pytest.fixture
    def mock_fields(self):
        return pd.DataFrame({
            "fldNpdidField": [1, 2, 3],
            "fldName": ["TROLL", "OSEBERG", "EKOFISK"],
            "fldHcType": ["OIL/GAS", "OIL", "OIL/GAS"],
        })

    def test_top_n(self, mock_production, mock_fields):
        result = format_production_ranking(mock_production, mock_fields, n=3)

        assert "Top 3 Producing Fields" in result
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_oil_ranking(self, mock_production, mock_fields):
        result = format_production_ranking(
            mock_production, mock_fields, n=3, hc_type="oil"
        )
        assert "kSm3/mo" in result

    def test_gas_ranking(self, mock_production, mock_fields):
        result = format_production_ranking(
            mock_production, mock_fields, n=3, hc_type="gas"
        )
        assert "MSm3/mo" in result

    def test_all_ranking(self, mock_production, mock_fields):
        result = format_production_ranking(
            mock_production, mock_fields, n=3, hc_type="all"
        )
        assert "oil eq" in result
