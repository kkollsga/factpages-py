"""
Tests for database.py utility functions.

These tests cover the pure functions that don't require database connections.
"""

import pytest

from factpages_py.database import get_category_for_dataset, FILE_MAPPING


# =============================================================================
# FILE_MAPPING Tests
# =============================================================================

class TestFileMapping:
    def test_entities_category_exists(self):
        assert "entities" in FILE_MAPPING
        entities = FILE_MAPPING["entities"]
        assert "field" in entities
        assert "wellbore" in entities
        assert "discovery" in entities
        assert "facility" in entities
        assert "pipeline" in entities
        assert "licence" in entities
        assert "company" in entities

    def test_geometries_category_exists(self):
        assert "geometries" in FILE_MAPPING
        geometries = FILE_MAPPING["geometries"]
        assert "field" in geometries
        assert "wellbore" in geometries

    def test_production_category_exists(self):
        assert "production" in FILE_MAPPING
        production = FILE_MAPPING["production"]
        assert "field_production_monthly" in production
        assert "field_production_yearly" in production

    def test_supporting_category_exists(self):
        assert "supporting" in FILE_MAPPING
        supporting = FILE_MAPPING["supporting"]
        assert "strat_chrono" in supporting
        assert "discovery_reserves" in supporting
        assert "field_reserves" in supporting


# =============================================================================
# get_category_for_dataset Tests
# =============================================================================

class TestGetCategoryForDataset:
    # Entity datasets
    @pytest.mark.parametrize("dataset", [
        "discovery", "field", "wellbore", "facility",
        "pipeline", "licence", "play", "block", "quadrant", "company"
    ])
    def test_entity_datasets(self, dataset):
        assert get_category_for_dataset(dataset) == "entities"

    # Geometry datasets
    @pytest.mark.parametrize("dataset", [
        "discovery", "field", "wellbore", "facility",
        "pipeline", "licence", "block", "quadrant", "play"
    ])
    def test_geometry_datasets(self, dataset):
        # Note: geometries overlap with entities - entities is checked first
        category = get_category_for_dataset(dataset)
        assert category in ["entities", "geometries"]

    # Production datasets
    @pytest.mark.parametrize("dataset", [
        "field_production_monthly", "field_production_yearly",
        "field_investment_yearly", "field_pipeline_transport",
        "field_facility_transport", "csd_injection"
    ])
    def test_production_datasets(self, dataset):
        assert get_category_for_dataset(dataset) == "production"

    # Supporting datasets
    @pytest.mark.parametrize("dataset", [
        "strat_chrono", "strat_litho",
        "discovery_reserves", "discovery_resource",
        "field_reserves", "field_description",
        "wellbore_casing", "wellbore_core_photo",
        "licence_licensee_history", "licence_operator_history",
        "seismic_acquisition"
    ])
    def test_supporting_datasets(self, dataset):
        assert get_category_for_dataset(dataset) == "supporting"

    # Unknown datasets default to "supporting"
    @pytest.mark.parametrize("dataset", [
        "unknown_dataset",
        "random_table",
        "not_in_mapping",
        ""
    ])
    def test_unknown_datasets_default_to_supporting(self, dataset):
        assert get_category_for_dataset(dataset) == "supporting"

    def test_case_sensitivity(self):
        # Function is case-sensitive
        assert get_category_for_dataset("field") == "entities"
        assert get_category_for_dataset("FIELD") == "supporting"  # Not found
        assert get_category_for_dataset("Field") == "supporting"  # Not found
