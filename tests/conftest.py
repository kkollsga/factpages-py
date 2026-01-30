"""
Shared pytest fixtures for factpages_py tests.
"""

import pytest

from factpages_py import Factpages


@pytest.fixture(scope="session")
def fp():
    """
    Session-scoped Factpages client with data loaded.

    This fixture is expensive (downloads data) so it's session-scoped
    to run only once per test session.
    """
    fp_obj = Factpages(data_dir="./factpages_data")
    fp_obj.refresh()
    yield fp_obj
