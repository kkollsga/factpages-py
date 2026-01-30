"""
Tests for schema.py - dataclasses and pure methods.

These tests don't require database access and run very fast.
"""

import pandas as pd
import pytest

from factpages_py.schema import (
    ColumnInfo,
    TableConfig,
    ConnectionConfig,
    DirectConnection,
    ConnectionResult,
    SchemaRegistry,
)


# =============================================================================
# ColumnInfo Tests
# =============================================================================

class TestColumnInfo:
    def test_defaults(self):
        col = ColumnInfo(name="test_col", alias="Test Column")
        assert col.name == "test_col"
        assert col.alias == "Test Column"
        assert col.type == "esriFieldTypeString"
        assert col.nullable is True
        assert col.length is None
        assert col.references_table is None
        assert col.is_primary_id is False

    def test_with_reference(self):
        col = ColumnInfo(
            name="fldNpdidField",
            alias="NPDID field",
            references_table="field",
            is_primary_id=True
        )
        assert col.references_table == "field"
        assert col.is_primary_id is True


# =============================================================================
# TableConfig Tests
# =============================================================================

class TestTableConfig:
    @pytest.fixture
    def field_config(self):
        return TableConfig(
            name="field",
            table_id=7100,
            node_type="Field",
            id_field="fldNpdidField",
            name_field="fldName",
            has_geometry=True,
            rename={"fldNpdidField": "field_id", "fldName": "name"},
            foreign_keys={
                "cmpNpdidCompany": {
                    "target_table": "company",
                    "target_field": "cmpNpdidCompany",
                    "connection_name": "OPERATED_BY"
                }
            }
        )

    def test_basic_properties(self, field_config):
        assert field_config.name == "field"
        assert field_config.table_id == 7100
        assert field_config.node_type == "Field"
        assert field_config.has_geometry is True

    def test_id_field_renamed(self, field_config):
        assert field_config.id_field_renamed == "field_id"

    def test_name_field_renamed(self, field_config):
        assert field_config.name_field_renamed == "name"

    def test_get_original_column(self, field_config):
        assert field_config.get_original_column("field_id") == "fldNpdidField"
        assert field_config.get_original_column("name") == "fldName"
        assert field_config.get_original_column("unknown") is None

    def test_get_renamed_column(self, field_config):
        assert field_config.get_renamed_column("fldNpdidField") == "field_id"
        assert field_config.get_renamed_column("fldName") == "name"
        # Non-renamed column returns itself
        assert field_config.get_renamed_column("other_col") == "other_col"

    def test_empty_rename(self):
        config = TableConfig(
            name="test",
            table_id=1,
            node_type="Test",
            id_field="id",
            name_field="name"
        )
        assert config.id_field_renamed == "id"
        assert config.name_field_renamed == "name"


# =============================================================================
# ConnectionConfig Tests
# =============================================================================

class TestConnectionConfig:
    def test_basic(self):
        conn = ConnectionConfig(
            name="field_licensee",
            table_id=7108,
            description="Field licensee history",
            source_table="field",
            source_field="fldNpdidField",
            target_table="company",
            target_field="cmpNpdidCompany",
            connection_name="LICENSED_TO",
            properties={"share": "fldCompanyShare", "from": "fldLicenseeFrom"}
        )
        assert conn.name == "field_licensee"
        assert conn.source_table == "field"
        assert conn.target_table == "company"
        assert conn.connection_name == "LICENSED_TO"

    def test_property_columns(self):
        conn = ConnectionConfig(
            name="test",
            table_id=1,
            description="Test",
            source_table="a",
            source_field="a_id",
            target_table="b",
            target_field="b_id",
            connection_name="RELATES",
            properties={"prop1": "col1", "prop2": "col2"}
        )
        assert conn.property_columns == ["col1", "col2"]

    def test_empty_properties(self):
        conn = ConnectionConfig(
            name="test",
            table_id=1,
            description="Test",
            source_table="a",
            source_field="a_id",
            target_table="b",
            target_field="b_id",
            connection_name="RELATES"
        )
        assert conn.property_columns == []


# =============================================================================
# DirectConnection Tests
# =============================================================================

class TestDirectConnection:
    def test_basic(self):
        conn = DirectConnection(
            source_table="wellbore",
            source_field="wlbNpdidWellbore",
            target_table="field",
            target_field="fldNpdidField",
            connection_name="DRILLED_ON",
            description="Wellbore drilled on field"
        )
        assert conn.source_table == "wellbore"
        assert conn.target_table == "field"
        assert conn.connection_name == "DRILLED_ON"


# =============================================================================
# ConnectionResult Tests
# =============================================================================

class TestConnectionResult:
    @pytest.fixture
    def sample_result(self):
        return ConnectionResult(
            source_type="Wellbore",
            target_type="Field",
            connection_name="DRILLED_ON",
            source_id_field="wlbNpdidWellbore",
            target_id_field="fldNpdidField",
            data=pd.DataFrame({
                "source_id": [1, 2, 3],
                "target_id": [100, 100, 200]
            }),
            properties={"depth": "wlbTotalDepth"}
        )

    def test_connection_type_alias(self, sample_result):
        # connection_type is an alias for connection_name (rusty-graph compat)
        assert sample_result.connection_type == "DRILLED_ON"
        assert sample_result.connection_type == sample_result.connection_name

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["source_type"] == "Wellbore"
        assert d["target_type"] == "Field"
        assert d["connection_type"] == "DRILLED_ON"
        assert d["source_id_field"] == "wlbNpdidWellbore"
        assert d["target_id_field"] == "fldNpdidField"
        assert len(d["data"]) == 3
        assert d["properties"] == {"depth": "wlbTotalDepth"}


# =============================================================================
# SchemaRegistry NPDID Parsing Tests (no DB needed)
# =============================================================================

class TestSchemaNpdidParsing:
    def test_parse_npdid_column_wellbore(self):
        # Test the NPDID_PREFIX_MAP lookup
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("wlb") == "wellbore"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("fld") == "field"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("dsc") == "discovery"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("fcl") == "facility"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("cmp") == "company"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("prl") == "licence"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("pip") == "pipeline"
        assert SchemaRegistry.NPDID_PREFIX_MAP.get("ply") == "play"

    def test_npdid_alias_patterns(self):
        patterns = SchemaRegistry.NPDID_ALIAS_PATTERNS
        assert patterns["NPDID wellbore"] == "wellbore"
        assert patterns["NPDID field"] == "field"
        assert patterns["NPDID discovery"] == "discovery"
        assert patterns["NPDID company"] == "company"
        assert patterns["NPDID drilling operator"] == "company"
        assert patterns["NPDID operating company"] == "company"
