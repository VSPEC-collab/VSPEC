#!/usr/bin/env python

"""
Tests for `VSPEC.files` module
"""

import pytest
from VSPEC import files
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


def test_path_exists(tmp_dir:Path):
    """
    Test path exists.

    Test `VSPEC.files.check_and_build_dir()` function
    with a path that already exists.
    """
    files.check_and_build_dir(tmp_dir)
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
    files.check_and_build_dir(tmp_dir)
    # check that the directory was created
    assert tmp_dir.exists()


def test_build_directories(tmp_dir):
    """
    Test `build_directories()`

    Run tests for `VSPEC.files.build_directories()`
    """
    # call the function with a test run name and temporary directory
    directories_dict = files.build_directories(tmp_dir)

    # check that all subdirectories exist
    assert directories_dict['parent'].exists()
    assert directories_dict['data'].exists()
    assert directories_dict['binned'].exists()
    assert directories_dict['all_model'].exists()
    assert directories_dict['psg_combined'].exists()
    assert directories_dict['psg_thermal'].exists()
    assert directories_dict['psg_noise'].exists()
    assert directories_dict['psg_layers'].exists()
    assert directories_dict['psg_configs'].exists()

def test_get_filename():
    """
    Run tests for `VSPEC.files.get_filename()`
    """
    assert files.get_filename(0,5,'rad') == 'phase00000.rad'

    with pytest.warns(RuntimeWarning):
        files.get_filename(999,2,'rad')
    