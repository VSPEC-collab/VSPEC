"""
Some general quality-of-life tests
"""
import sys
from pathlib import Path
from subprocess import Popen, PIPE
import tomllib
import json
import pytest
from VSPEC import __version__



DOCS_SOURCE_PATH = Path(__file__).parent.parent.parent / 'docs' / 'source'
sys.path.append(DOCS_SOURCE_PATH.as_posix())
# pylint: disable-next=import-error
import conf as sphinx_conf

def get_current_branch_name() -> str:
    """
    Current branch name
    """
    return Popen(['git', 'branch', '--show-current'], stdout=PIPE).communicate()[0].decode('utf-8').strip()


def test_version():
    """
    Check that the version number is the same as in the pyproject.toml
    """
    
    with open(Path(__file__).parent.parent.parent / 'pyproject.toml', 'rb') as f:
        version: str = tomllib.load(f)['project']['version']
    
    assert version == __version__, f'Version in `__init__.py` is {__version__}, but in `pyproject.toml` is {version}'
    
    assert sphinx_conf.release == version, f'Version in `pyproject.toml` is {version}, but in `conf.py` is {sphinx_conf.release}'
    
    try:
        branch_name = get_current_branch_name()
        if 'release' in branch_name:
            git_version = branch_name.split('/')[1]
            assert git_version == version, f'Version in `pyproject.toml` is {version}, but the branch name is {branch_name}'
    except Exception as e:
        if isinstance(e, AssertionError):
            raise
    
    with open(DOCS_SOURCE_PATH / 'versions.json', 'rt', encoding='UTF-8') as f:
        versions = json.load(f)
    assert {'version': version, 'url': f'https://VSPEC-collab.github.io/VSPEC/{version}/index.html'} in versions, f'Version {version} is not in `versions.json`'
    
    
if __name__ == '__main__':
    pytest.main(args=[__file__])