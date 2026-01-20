from factpages_py import Factpages

import pytest

@pytest.fixture()
def fp():
    # Initialize client with data directory
    fp_obj = Factpages(data_dir="./factpages_data")

    # Download core datasets
    fp_obj.refresh()
    yield fp_obj


def test_field_list(fp):
    assert len(fp.field.list()) == fp.field.count()
    assert len(fp.field.list()) == 141


@pytest.mark.parametrize("name", ["troll", 46437])
def test_troll(fp, name):

    troll = fp.field(name)

    assert troll.name           == "TROLL"
    assert troll.operator       == "Equinor Energy AS"
    assert troll.status         == "Producing"
    assert troll.hc_type        == "OIL/GAS"
    assert troll.discovery_year == 1979
    assert troll.id             == 46437

