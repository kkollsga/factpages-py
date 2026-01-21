from factpages_py import Factpages

import pytest

@pytest.fixture()
def fp():
    # Initialize client with data directory
    fp_obj = Factpages(data_dir="./factpages_data")

    # Download core datasets
    fp_obj.refresh()
    yield fp_obj


# Field tests
def test_field_list(fp):
    assert len(fp.field.list()) == fp.field.count()
    assert len(fp.field.list()) == 141


def test_field_ids(fp):
    ids = fp.field.ids()
    assert len(ids) == fp.field.count()
    assert all(isinstance(id, int) for id in ids)


@pytest.mark.parametrize("name", ["troll", 46437])
def test_troll(fp, name):
    troll = fp.field(name)

    assert troll.name           == "TROLL"
    assert troll.operator       == "Equinor Energy AS"
    assert troll.status         == "Producing"
    assert troll.hc_type        == "OIL/GAS"
    assert troll.discovery_year == 1979
    assert troll.id             == 46437


# Discovery tests
def test_discovery_count(fp):
    assert fp.discovery.count() > 0
    assert len(fp.discovery.list()) == fp.discovery.count()


def test_johan_sverdrup(fp):
    sverdrup = fp.discovery("johan sverdrup")
    assert "JOHAN SVERDRUP" in sverdrup.name.upper()


# Wellbore tests
def test_wellbore_count(fp):
    assert fp.wellbore.count() > 0
    assert len(fp.wellbore.list()) == fp.wellbore.count()


# Company tests
def test_company_count(fp):
    assert fp.company.count() > 0


def test_equinor(fp):
    equinor = fp.company("equinor")
    assert "EQUINOR" in equinor.name.upper()

