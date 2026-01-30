"""
Tests for analysis.py - AnalysisMixin methods.

These tests cover both edge cases (no data available) and basic functionality
using the session-scoped fixture from conftest.py.
"""

import pandas as pd
import pytest


# =============================================================================
# Edge Case Tests (No Database Required)
# =============================================================================

class TestAnalysisEdgeCases:
    """Test edge cases that don't require database access."""

    def test_field_summary_not_found(self, fp):
        """Test field_summary with non-existent field."""
        result = fp.field_summary("NONEXISTENT_FIELD_12345")
        assert "not found" in result.lower()

    def test_well_info_not_found(self, fp):
        """Test well_info with non-existent wellbore."""
        result = fp.well_info("99/99-999")
        assert "not found" in result.lower()

    def test_discovery_info_not_found(self, fp):
        """Test discovery_info with non-existent discovery."""
        result = fp.discovery_info("NONEXISTENT_DISCOVERY")
        assert "not found" in result.lower()

    def test_license_info_not_found(self, fp):
        """Test license_info with non-existent license."""
        result = fp.license_info("PL99999")
        assert "not found" in result.lower()

    def test_company_portfolio_not_found(self, fp):
        """Test company_portfolio with non-existent company."""
        result = fp.company_portfolio("NONEXISTENT_COMPANY_XYZ")
        assert "not found" in result.lower()


# =============================================================================
# Basic Functionality Tests (Using session fixture)
# =============================================================================

class TestAnalysisBasic:
    """Test basic analysis functionality with actual data."""

    def test_field_summary_troll(self, fp):
        """Test field_summary with known field."""
        result = fp.field_summary("TROLL")

        assert "TROLL" in result
        assert "=" in result  # Header separator
        # Should contain basic field info
        assert "Status:" in result or "Operator:" in result

    def test_well_info_basic(self, fp):
        """Test well_info with known wellbore."""
        result = fp.well_info("33/9-1")

        assert "33/9-1" in result
        assert "LOCATION" in result or "Status:" in result

    def test_discovery_info_basic(self, fp):
        """Test discovery_info with known discovery."""
        result = fp.discovery_info("JOHAN SVERDRUP")

        # Should contain discovery name
        assert "JOHAN" in result.upper() or "SVERDRUP" in result.upper()

    def test_recent_discoveries(self, fp):
        """Test recent_discoveries returns formatted output."""
        result = fp.recent_discoveries(years=5)

        # Should have header
        assert "Discoveries" in result or "last" in result.lower()

    def test_active_wells(self, fp):
        """Test active_wells returns formatted output."""
        result = fp.active_wells()

        # Either has drilling wells or says none
        assert "drilling" in result.lower() or "wells" in result.lower()


# =============================================================================
# Data Endpoint Tests
# =============================================================================

class TestDataEndpoints:
    """Test data endpoint methods that return DataFrames."""

    def test_get_production_timeseries_nonexistent(self, fp):
        """Test get_production_timeseries with non-existent field."""
        result = fp.get_production_timeseries("NONEXISTENT_FIELD")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_field_geometry_nonexistent(self, fp):
        """Test get_field_geometry with non-existent field."""
        result = fp.get_field_geometry("NONEXISTENT_FIELD")
        assert result is None

    def test_get_well_coordinates(self, fp):
        """Test get_well_coordinates returns DataFrame."""
        result = fp.get_well_coordinates()
        assert isinstance(result, pd.DataFrame)
        # Should have coordinate columns if data exists
        if not result.empty:
            assert "wlbWellboreName" in result.columns


# =============================================================================
# Case Insensitivity Tests
# =============================================================================

class TestCaseInsensitivity:
    """Test that lookups are case-insensitive where appropriate."""

    def test_field_summary_lowercase(self, fp):
        """Test field_summary accepts lowercase."""
        result = fp.field_summary("troll")
        assert "TROLL" in result

    def test_field_summary_mixed_case(self, fp):
        """Test field_summary accepts mixed case."""
        result = fp.field_summary("Troll")
        assert "TROLL" in result

    def test_discovery_info_lowercase(self, fp):
        """Test discovery_info accepts lowercase."""
        result = fp.discovery_info("johan sverdrup")
        assert "not found" not in result.lower() or "JOHAN" in result.upper()
