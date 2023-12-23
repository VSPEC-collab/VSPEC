"""
VSPEC Builtin data.
"""

from pathlib import Path
import requests

BASE_PATH = Path.home() / '.vspec'
DATA_PATH = BASE_PATH / 'data'

WACCM_URL = 'https://zenodo.org/records/10426886/files/vspec_waccm_test.nc?#mode=bytes'

WACCM_PATH = DATA_PATH / 'vspec_waccm_test.nc'
"""
The local path for the WACCM test dataset.

:type: pathlib.Path
"""

def download_waccm_test_data(rewrite=False):
    """
    Download the WACCM test dataset.
    
    Taken from this SO post:
        https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    
    Parameters
    ----------
    rewrite : bool, optional
        If True, re-download the data. The default is False.
    
    Returns
    -------
    pathlib.Path
        The path to the downloaded data.
    """
    DATA_PATH.mkdir(exist_ok=True)
    WACCM_PATH.parent.mkdir(exist_ok=True)
    if WACCM_PATH.exists() and not rewrite:
        return WACCM_PATH
    else:
        WACCM_PATH.unlink(missing_ok=True)
        with requests.get(WACCM_URL,stream=True,timeout=20) as req:
            req.raise_for_status()
            with WACCM_PATH.open('wb') as f:
                for chunk in req.iter_content(chunk_size=8192):
                    f.write(chunk)
        return WACCM_PATH