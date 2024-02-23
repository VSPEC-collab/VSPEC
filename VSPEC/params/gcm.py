"""
GCM parameters module
"""
from typing import Union
from pathlib import Path
from astropy import units as u
from netCDF4 import Dataset
import yaml

from pypsg import PyConfig
from pypsg.globes.waccm.waccm import get_time_index
from pypsg.globes import waccm_to_pygcm, PyGCM

from VSPEC.config import psg_encoding, PRESET_PATH
from VSPEC.params.base import BaseParameters
from VSPEC.gcm.heat_transfer import to_pygcm as vspec_to_pygcm


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
    static : bool
        If true, the GCM does not change with time.
        (Set to True for `binaryGCM`)

    Methods
    -------
    content()
        Get the content of the GCM.

    """
    static = True

    def __init__(self, path: Path = None, data: bytes = None):
        if path is None and data is None:
            raise ValueError('Must provide some way to access the data!')
        self.path = path if path is None else path.expanduser()
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

    def to_pycfg(self) -> PyConfig:
        """
        Get a `PyConfig` object from the GCM.
        """
        return PyConfig.from_bytes(self.content())

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            path=None if d.get('path', None) is None else Path(
                d.get('path', None)),
            data=None if d.get('data', None) is None else bytes(
                d.get('data', None), encoding=psg_encoding)
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
    static : bool, default=False
        If true, the GCM does not change with time.

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
    static : bool
        If true, the GCM does not change with time.

    Methods
    -------
    content(obs_time: astropy.units.Quantity) -> bytes
        Get the content of the GCM for the specified observation time.

    """

    def __init__(self, path: Path, tstart: u.Quantity, molecules: list, aerosols: list, background: str = None, static: bool = False):
        self.path = path
        self.tstart = tstart
        self.molecules = molecules
        self.aerosols = aerosols
        self.background = background
        self.static = static

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

            gcm = waccm_to_pygcm(
                data=data,
                itime=itime,
                molecules=self.molecules,
                aerosols=self.aerosols,
                background=self.background,
            )
            return PyConfig(
                atmosphere=gcm.update_params(),
                gcm=gcm
            ).content

    def to_pycfg(
        self,
        obs_time: u.Quantity
    ):
        """
        Get a `pypsg.PyConfig` representation of the GCM.

        Parameters
        ----------
        obs_time : astropy.units.Quantity
            The observation time.

        Returns
        -------
        pypsg.PyConfig
            The PyConfig representation of the GCM.
        """
        with Dataset(self.path, 'r', format='NETCDF4') as data:
            itime = get_time_index(data, obs_time + self.tstart)

            pygcm = waccm_to_pygcm(
                data=data,
                itime=itime,
                molecules=self.molecules,
                aerosols=self.aerosols,
                background=self.background,
            )

            atmosphere = pygcm.update_params()
            return PyConfig(
                atmosphere=atmosphere,
                gcm=pygcm
            )

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            path=Path(d['path']).expanduser(),
            tstart=u.Quantity(d['tstart']),
            molecules=list(d['molecules'].replace(
                ' ', '').split(',')) if 'molecules' in d else [],
            aerosols=list(d['aerosols'].replace(' ', '').split(
                ',')) if 'aerosols' in d else [],
            background=None if d.get('background', None) is None else str(
                d.get('background', None))
        )


class vspecGCM(BaseParameters):
    static = True

    def __init__(
        self,
        gcm: PyGCM
    ):
        self.gcm = gcm

    @classmethod
    def _from_dict(cls, gcm_dict: dict, star_dict: dict, planet_dict: dict):

        return cls(
            vspec_to_pygcm(
                shape=(
                    int(gcm_dict['nlayer']),
                    int(gcm_dict['nlon']),
                    int(gcm_dict['nlat'])
                ),
                epsilon=float(gcm_dict['epsilon']),
                star_teff=u.Quantity(star_dict['teff']),
                r_star=u.Quantity(star_dict['radius']),
                r_orbit=u.Quantity(planet_dict['semimajor_axis']),
                lat_redistribution=float(gcm_dict['lat_redistribution']),
                p_surf=u.Quantity(gcm_dict['psurf']),
                p_stop=u.Quantity(gcm_dict['ptop']),
                wind_u=u.Quantity(gcm_dict['wind']['U']),
                wind_v=u.Quantity(gcm_dict['wind']['V']),
                gamma=float(gcm_dict['gamma']),
                albedo=u.Quantity(gcm_dict['albedo']),
                emissivity=u.Quantity(gcm_dict['emissivity']),
                molecules=gcm_dict['molecules'],
            )
        )

    def content(self) -> bytes:
        """
        Get bytes representation.
        """
        return self.gcm.content

    def to_pycfg(self):
        """
        Get `pypsg.PyConfig` representation.
        """
        atmosphere = self.gcm.update_params()
        return PyConfig(
            atmosphere=atmosphere,
            gcm=self.gcm
        )

    @classmethod
    def earth(cls, **kwargs):
        path = PRESET_PATH / 'earth.yaml'
        with open(path, 'r', encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            gcm_dict: dict = data['gcm']
            star_dict = data['star']
            planet_dict = data['planet']
            gcm_dict.update(**kwargs)

            return cls._from_dict(
                gcm_dict=gcm_dict,
                star_dict=star_dict,
                planet_dict=planet_dict
            )


class gcmParameters(BaseParameters):
    """
    Class to store GCM parameters.

    Parameters
    ----------
    gcm : binaryGCM or waccmGCM
        The GCM instance containing the GCM data.
    mean_molec_weight : float
        The mean molecular weight of the atmosphere
        in g/mol.

    Attributes
    ----------
    is_static
    gcmtype
    gcm : binaryGCM or waccmGCM
        The GCM instance containing the GCM data.
    mean_molec_weight : float
        The mean molecular weight of the atmosphere
        in g/mol.

    """

    def __init__(
        self,
        gcm: Union[binaryGCM, Union[vspecGCM, waccmGCM]],
        mean_molec_weight: float
    ):
        self.gcm = gcm
        self.mean_molec_weight = mean_molec_weight

    def content(self, **kwargs):
        """
        Get a bytes representation of the GCM.
        """
        return self.gcm.content(**kwargs)

    def to_pycfg(self, **kwargs) -> PyConfig:
        """
        Get `pypsg.PyConfig` representation of the GCM.

        Parameters
        ----------
        obs_time : astropy.time.Time, optional
            The time of the observation. Necessary for a waccm GCM.
        """
        return self.gcm.to_pycfg(**kwargs)

    @property
    def is_staic(self) -> bool:
        return self.gcm.static

    @property
    def gcmtype(self) -> str:
        if isinstance(self.gcm, binaryGCM):
            return 'binary'
        elif isinstance(self.gcm, waccmGCM):
            return 'waccm'
        elif isinstance(self.gcm, vspecGCM):
            return 'vspec'
        else:
            raise TypeError('Unknown GCM type')

    @classmethod
    def _from_dict(cls, d: dict):
        gcm_dict = d['gcm']
        star_dict = d['star']
        planet_dict = d['planet']
        mean_molec_weight = float(gcm_dict['mean_molec_weight'])
        if 'binary' in gcm_dict:
            return cls(
                gcm=binaryGCM.from_dict(gcm_dict['binary']),
                mean_molec_weight=mean_molec_weight
            )
        elif 'waccm' in gcm_dict:
            return cls(
                gcm=waccmGCM.from_dict(gcm_dict['waccm']),
                mean_molec_weight=mean_molec_weight
            )
        elif 'vspec' in gcm_dict:
            return cls(
                gcm=vspecGCM.from_dict(
                    gcm_dict['vspec'], star_dict, planet_dict),
                mean_molec_weight=mean_molec_weight
            )
        else:
            raise KeyError(
                f'`binary`, `waccm`, or `vspec` not in {list(d.keys())}')

    def to_psg(self) -> dict:
        """
        Write parameters to the PSG format.

        Returns
        -------
        dict
            The PSG parameters in a dictionary
        """
        return {
            'ATMOSPHERE-WEIGHT': f'{self.mean_molec_weight:.1f}'
        }


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
    nmax : int
        PSG handbook: 'When performing scattering aerosols calculations, this
        parameter indicates the number of n-stream pairs - Use 0 for extinction
        calculations only (e.g. transit, occultation)' :cite:p:`2022fpsg.book.....V`
    lmax : int
        PSG handbook: 'When performing scattering aerosols calculations, this
        parameter indicates the number of scattering Legendre polynomials used
        for describing the phase function - Use 0 for extinction calculations
        only (e.g. transit, occultation)' :cite:p:`2022fpsg.book.....V`
    continuum : list of str
        The continuum opacities to include in the radiative transfer calculation, such as
        'Rayleigh', 'Refraction', 'CIA_all'.
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
    nmax : int
        PSG handbook: 'When performing scattering aerosols calculations, this
        parameter indicates the number of n-stream pairs - Use 0 for extinction
        calculations only (e.g. transit, occultation)' :cite:p:`2022fpsg.book.....V`
    lmax : int
        PSG handbook: 'When performing scattering aerosols calculations, this
        parameter indicates the number of scattering Legendre polynomials used
        for describing the phase function - Use 0 for extinction calculations
        only (e.g. transit, occultation)' :cite:p:`2022fpsg.book.....V`
    url : str
        URL of the Planetary Spectrum Generator.
    api_key : APIkey
        An instance of the APIkey class representing the PSG API key.

    """

    def __init__(
        self,
        gcm_binning: int,
        phase_binning: int,
        use_molecular_signatures: bool,
        nmax: int,
        lmax: int,
        continuum: list
    ):
        self.gcm_binning = gcm_binning
        self.phase_binning = phase_binning
        self.use_molecular_signatures = use_molecular_signatures
        self.nmax = nmax
        self.lmax = lmax
        self.continuum = continuum

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            gcm_binning=int(d['gcm_binning']),
            phase_binning=int(d['phase_binning']),
            use_molecular_signatures=bool(d['use_molecular_signatures']),
            nmax=int(d['nmax']),
            lmax=int(d['lmax']),
            continuum=list(d['continuum']),
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
            'ATMOSPHERE-NMAX': f'{self.nmax}',
            'ATMOSPHERE-LMAX': f'{self.lmax}',
            'ATMOSPHERE-CONTINUUM': ','.join(self.continuum)
        }
