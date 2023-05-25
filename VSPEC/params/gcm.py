"""
GCM parameters module
"""
from pathlib import Path
from astropy import units as u
from netCDF4 import Dataset

from VSPEC.config import psg_encoding
from VSPEC.params.base import BaseParameters
from VSPEC.waccm.read_nc import get_time_index
from VSPEC.waccm.write_psg import get_cfg_contents


class binaryGCM(BaseParameters):
    """
    Class to store a GCM given in the PSG format.

    Parameters
    ----------
    path : pathlib.Path, default=None
        The path to the data file.
    data : bytes, default=None
        The data as a Python bytes object.

    Raises
    ------
    ValueError
        If neither path nor data is provided.

    Attributes
    ----------
    path : pathlib.Path or None
        The path to the config file.
    data : bytes or None
        The data as a Python bytes object.

    Methods
    -------
    content()
        Get the content of the GCM.

    """

    def __init__(self, path: Path = None, data: bytes = None):
        if path is None and data is None:
            raise ValueError('Must provide some way to access the data!')
        self.path = path
        self.data = data

    def content(self) -> bytes:
        """
        Get the GCM as a `bytes` object.

        Returns
        -------
        bytes
            The content of the GCM.

        """
        if self.path is None:
            return self.data
        else:
            with open(self.path, 'rb') as file:
                return file.read()
    @classmethod
    def _from_dict(cls,d:dict):
        return cls(
            path = None if d.get('path',None) is None else Path(d.get('path',None)),
            data = None if d.get('data',None) is None else bytes(d.get('data',None),encoding=psg_encoding)
        )


class waccmGCM(BaseParameters):
    """
    Class to store and provide access to Whole Atmosphere
    Community Climate Model (WACCM) data.

    Parameters
    ----------
    path : pathlib.Path
        The path to the netCDF file.
    tstart : astropy.units.Quantity
        The start time of the data.
    molecules : list
        A list of molecules to extract from the data.
    aerosols : list
        A list of aerosols to extract from the data.
    background : str, default=None
        The background molecule to include in the GCM.

    Attributes
    ----------
    path : pathlib.Path
        The path to the netCDF file.
    tstart : astropy.units.Quantity
        The start time of the data.
    molecules : list
        A list of molecules to extract from the data.
    aerosols : list
        A list of aerosols to extract from the data.
    background : str or None
        The background background molecule to include in the GCM.

    Methods
    -------
    content(obs_time: astropy.units.Quantity) -> bytes
        Get the content of the GCM for the specified observation time.

    """

    def __init__(self, path: Path, tstart: u.Quantity, molecules: list, aerosols: list, background: str = None):
        self.path = path
        self.tstart = tstart
        self.molecules = molecules
        self.aerosols = aerosols
        self.background = background

    def content(self, obs_time: u.Quantity) -> bytes:
        """
        Get the content of the GCM for the specified observation time.

        Parameters
        ----------
        obs_time : astropy.units.Quantity
            The observation time.

        Returns
        -------
        bytes
            The content of the data.

        """

        with Dataset(self.path, 'r', format='NETCDF4') as data:
            itime = get_time_index(data, obs_time + self.tstart)
            return get_cfg_contents(
                data=data,
                itime=itime,
                molecules=self.molecules,
                aerosols=self.aerosols,
                background=self.background
            )
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            path = Path(d['path']),
            tstart = u.Quantity(d['tstart']),
            molecules = list(d['molecules']),
            aerosols = list(d['aerosols']),
            background = None if d.get('background',None) is None else str(d.get('background',None))
        )


class APIkey(BaseParameters):
    """
    Class to store a PSG API key.
    Do not commit your API key to a git repository!

    Parameters
    ----------
    path : pathlib.Path, default = None
        The path to the file containing the API key.
    value : str, default = None
        The API key value.

    Attributes
    ----------
    path : pathlib.Path or None
        The path to the file containing the API key.
    _value : str or None
        The API key value.

    Properties
    ----------
    value : str
        The API key value. If `path` is provided, the value is read from the file.

    """

    def __init__(
        self,
        path:Path=None,
        value:str=None
    ):
        self.path = path
        self._value = value
    @property
    def value(self):
        """
        Get the API key value.

        Returns
        -------
        str
            The API key value.

        """

        if self.path is None:
            return self._value
        else:
            with open(self.path,'rt',encoding='UTF-8') as file:
                return file.read()
    @classmethod
    def _from_dict(cls,d:dict):
        return cls(
            path = None if d.get('path',None) is None else Path(d.get('path',None)),
            value = None if d.get('value',None) is None else str(d.get('value',None))
        )
    @classmethod
    def none(cls):
        return cls(None,None)

class psgParameters(BaseParameters):
    """
    Class to store parameters for the Planetary Spectrum Generator (PSG).

    Parameters
    ----------
    gcm_binning : int
        Number of spatial points to bin together in the GCM data. Use 3 for science.
    phase_binning : int
        Number of phase epochs to bin together when simulating the planet. These are later
        interpolated to match the cadence of the variable star simulation.
    use_molecular_signatures : bool
        Whether to use molecular signatures (PSG atmosphere) or not.
    url : str
        URL of the Planetary Spectrum Generator.
    api_key : APIkey
        An instance of the APIkey class representing the PSG API key. Provide either the
        path to the API key file or the API key value.

    Attributes
    ----------
    gcm_binning : int
        Number of spatial points to bin together in the GCM data.
    phase_binning : int
        Number of phase epochs to bin together when simulating the planet.
    use_molecular_signatures : bool
        Whether to use molecular signatures (PSG atmosphere) or not.
    url : str
        URL of the Planetary Spectrum Generator.
    api_key : APIkey
        An instance of the APIkey class representing the PSG API key.

    """

    def __init__(
        self,
        gcm_binning:int,
        phase_binning:int,
        use_molecular_signatures:bool,
        url:str,
        api_key:APIkey
    ):
        self.gcm_binning = gcm_binning
        self.phase_binning = phase_binning
        self.use_molecular_signatures = use_molecular_signatures
        self.url = url
        self.api_key = api_key
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            gcm_binning = int(d['gcm_binning']),
            phase_binning = int(d['phase_binning']),
            use_molecular_signatures = bool(d['use_molecular_signatures']),
            url = str(d['url']),
            api_key = APIkey.none() if d.get('api_key',None) is None else APIkey.from_dict(d['api_key'])
        )
    def to_psg(self):
        """
        Convert the PSG parameters to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the PSG parameters in the PSG input format.

        """
        return {
            'GENERATOR-GCM-BINNING': f'{self.gcm_binning}',
            'GENERATOR-GAS-MODEL': 'Y' if self.use_molecular_signatures else 'N',
        }