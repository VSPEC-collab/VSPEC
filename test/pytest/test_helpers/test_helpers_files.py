#!/usr/bin/env python

"""
Tests for `VSPEC.files` module
"""

import pytest
from VSPEC import helpers
from pathlib import Path
import shutil


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for testing"""
    tmpdir = Path("test_dir")
    tmpdir.mkdir()
    yield tmpdir
    # clean up the temporary directory
    shutil.rmtree(tmpdir)


def test_path_exists(tmp_dir: Path):
    """
    Test path exists.

    Test `VSPEC.files.check_and_build_dir()` function
    with a path that already exists.
    """
    helpers.check_and_build_dir(tmp_dir)
    # check that the directory still exists
    assert tmp_dir.exists()


def test_path_does_not_exist(tmp_dir):
    """
    Test path exists after creation.

    Test `VSPEC.files.check_and_build_dir()` function
    with a path that does not yet exist.
    """
    tmp_dir_name = "test_dir"
    tmp_dir = Path(tmp_dir_name)
    # call the function with a non-existing path
    helpers.check_and_build_dir(tmp_dir)
    # check that the directory was created
    assert tmp_dir.exists()


def test_get_filename():
    """
    Run tests for `VSPEC.files.get_filename()`
    """
    assert helpers.get_filename(0, 5, 'rad') == 'phase00000.rad'

    with pytest.warns(RuntimeWarning):
        helpers.get_filename(999, 2, 'rad')
