#!/usr/bin/env python

"""
Tests for `VSPEC.read_info` module
"""

from VSPEC.read_info import ParamModel
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'default.cfg'


def test_default_init():
    """
    Test `ParamModel` with `default.cfg`
    """
    ParamModel(DEFAULT_CONFIG_PATH)

if __name__ in '__main__':
    test_default_init()