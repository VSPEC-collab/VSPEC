"""
GCM parameters module
"""
from typing import Callable
from pathlib import Path
from astropy import units as u
from netCDF4 import Dataset

from libpypsg import PyConfig
from libpypsg.globes import waccm_to_pygcm, PyGCM, exocam_to_pygcm, GCMdecoder, exoplasim_to_pygcm
from libpypsg.globes.exocam import exocam
from libpypsg.globes.waccm import waccm

from VSPEC.params.base import BaseParameters
from VSPEC.gcm.heat_transfer import to_pygcm as vspec_to_pygcm


def parse_molec_list(molec_list: list):
    """
    Annoyingly YAML automatically converts NO (nitrous oxide) to the boolean False.
    """
    return [mol if mol is not False else 'NO' for mol in molec_list]


class gcmParameters(BaseParameters):
    """
    Class to store GCM parameters.

    Parameters
    ----------
    gcm_getter : Callable
        A function that returns a PyGCM instance.
    mean_molec_weight : float
        The mean molecular weight of the atmosphere in g/mol.
    is_staic : bool
        If true, the GCM does not change over time. In the case that this is false a time
        parameter will be passed to the `gcm_getter`.

    """
    _defaults = {
        'lat_redistribution': 0.0
    }

    def __init__(
        self,
        gcm_getter: Callable[..., PyGCM],
        mean_molec_weight: float,
        is_static: bool
    ):
        self.get_gcm = gcm_getter
        self.mean_molec_weight = mean_molec_weight
        self.is_staic = is_static

    def content(self, **kwargs):
        """
        Get a bytes representation of the GCM.
        """
        return self.get_gcm(**kwargs).content

    def to_pycfg(self, **kwargs) -> PyConfig:
        """
        Get `libpypsg.PyConfig` representation of the GCM.

        Parameters
        ----------
        obs_time : astropy.time.Time, optional
            The time of the observation. Necessary for a waccm GCM.
        """
        _gcm = self.get_gcm(**kwargs)
        return PyConfig(
            atmosphere=_gcm.update_params(),
            gcm=_gcm
        )

    @classmethod
    def _from_dict(cls, d: dict):
        gcm_dict = d['gcm']
        star_dict = d['star']
        planet_dict = d['planet']
        mean_molec_weight = float(gcm_dict['mean_molec_weight'])
        if 'binary' in gcm_dict:
            args_dict: dict = gcm_dict['binary']
            path = Path(args_dict['path']).expanduser()

            def fun():
                return PyGCM.from_decoder(GCMdecoder.from_psg(path))
            return cls(
                gcm_getter=fun,
                mean_molec_weight=mean_molec_weight,
                is_static=True
            )
        elif 'vspec' in gcm_dict:
            args_dict: dict = gcm_dict['vspec']

            def fun():
                return vspec_to_pygcm(
                    shape=(
                        int(args_dict['nlayer']),
                        int(args_dict['nlon']),
                        int(args_dict['nlat'])
                    ),
                    epsilon=float(args_dict['epsilon']),
                    star_teff=u.Quantity(star_dict['teff']),
                    r_star=u.Quantity(star_dict['radius']),
                    r_orbit=u.Quantity(planet_dict['semimajor_axis']),
                    lat_redistribution=float(args_dict.get('lat_redistribution', cls._defaults['lat_redistribution'])),
                    p_surf=u.Quantity(args_dict['psurf']),
                    p_stop=u.Quantity(args_dict['ptop']),
                    wind_u=u.Quantity(args_dict['wind']['U']),
                    wind_v=u.Quantity(args_dict['wind']['V']),
                    gamma=float(args_dict['gamma']),
                    albedo=u.Quantity(args_dict['albedo']),
                    emissivity=u.Quantity(args_dict['emissivity']),
                    molecules=args_dict['molecules'],
                )
            return cls(
                gcm_getter=fun,
                mean_molec_weight=mean_molec_weight,
                is_static=True
            )
        elif 'waccm' in gcm_dict:
            args_dict: dict = gcm_dict['waccm']
            path = Path(args_dict['path']).expanduser()
            itime = int(args_dict['itime']) if 'itime' in args_dict else None
            is_static = itime is not None
            if is_static:
                def fun():
                    with Dataset(path) as data:
                        return waccm_to_pygcm(
                            data=data,
                            itime=itime,
                            molecules=parse_molec_list(args_dict['molecules']),
                            aerosols=args_dict['aerosols'],
                            background=args_dict.get('background', None),
                            lon_start=args_dict.get('lon_start', -180),
                            lat_start=args_dict.get('lat_start', -90)
                        )
            else:
                def fun(obs_time: u.Quantity):
                    with Dataset(path) as data:
                        return waccm_to_pygcm(
                            data=data,
                            itime=waccm.get_time_index(
                                data, obs_time + u.Quantity(args_dict['tstart'])),
                            molecules=parse_molec_list(args_dict['molecules']),
                            aerosols=args_dict.get('aerosols',None),
                            background=args_dict.get('background', None),
                            lon_start=args_dict.get('lon_start', -180),
                            lat_start=args_dict.get('lat_start', -90)
                        )
            return cls(
                gcm_getter=fun,
                mean_molec_weight=mean_molec_weight,
                is_static=is_static
            )
        elif 'exocam' in gcm_dict:
            args_dict: dict = gcm_dict['exocam']
            path = Path(args_dict['path']).expanduser()
            itime = int(args_dict['itime']) if 'itime' in args_dict else None
            is_static = itime is not None
            if is_static:
                def fun():
                    with Dataset(path) as data:
                        return exocam_to_pygcm(
                            data=data,
                            itime=itime,
                            molecules=parse_molec_list(args_dict['molecules']),
                            aerosols=args_dict['aerosols'],
                            background=args_dict.get('background', None),
                            lon_start=args_dict.get('lon_start', -180),
                            lat_start=args_dict.get('lat_start', -90),
                            mean_molecular_mass=args_dict.get('mean_molecular_mass',None)
                        )
            else:
                def fun(obs_time: u.Quantity):
                    with Dataset(path) as data:
                        return exocam_to_pygcm(
                            data=data,
                            itime=exocam.get_time_index(data, obs_time),
                            molecules=parse_molec_list(args_dict['molecules']),
                            aerosols=args_dict['aerosols'],
                            background=args_dict.get('background', None),
                            lon_start=args_dict.get('lon_start', -180),
                            lat_start=args_dict.get('lat_start', -90),
                            mean_molecular_mass=args_dict.get('mean_molecular_mass',None)
                        )
            return cls(
                gcm_getter=fun,
                mean_molec_weight=mean_molec_weight,
                is_static=is_static
            )
        elif 'exoplasim' in gcm_dict:
            args_dict: dict = gcm_dict['exoplasim']
            path = Path(args_dict['path']).expanduser()

            def fun():
                with Dataset(path) as data:
                    return exoplasim_to_pygcm(
                        data=data,
                        itime=int(args_dict['itime']),
                        molecules=parse_molec_list(args_dict['molecules']),
                        aerosols=args_dict['aerosols'],
                        background=args_dict.get('background', None),
                        lon_start=args_dict.get('lon_start', -180),
                        lat_start=args_dict.get('lat_start', -90),
                        mean_molecular_mass=args_dict.get('mean_molecular_mass',None)
                    )
            return cls(
                gcm_getter=fun,
                mean_molec_weight=mean_molec_weight,
                is_static=True
            )

        else:
            raise KeyError(
                f'`binary`, `waccm`, or `vspec` not in {list(d.keys())}')


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
    use_continuum_stellar : bool
        Whether to include the stellar contiuum or not.
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

    Attributes
    ----------
    gcm_binning : int
        Number of spatial points to bin together in the GCM data.
    phase_binning : int
        Number of phase epochs to bin together when simulating the planet.
    use_molecular_signatures : bool
        Whether to use molecular signatures (PSG atmosphere) or not.
    use_continuum_stellar : bool
        Whether to include the stellar contiuum or not.
    nmax : int
        PSG handbook: 'When performing scattering aerosols calculations, this
        parameter indicates the number of n-stream pairs - Use 0 for extinction
        calculations only (e.g. transit, occultation)' :cite:p:`2022fpsg.book.....V`
    lmax : int
        PSG handbook: 'When performing scattering aerosols calculations, this
        parameter indicates the number of scattering Legendre polynomials used
        for describing the phase function - Use 0 for extinction calculations
        only (e.g. transit, occultation)' :cite:p:`2022fpsg.book.....V`

    """
    _defaults = {
        'use_stellar_continuum': True
    }

    def __init__(
        self,
        gcm_binning: int,
        phase_binning: int,
        use_molecular_signatures: bool,
        use_continuum_stellar: bool,
        nmax: int,
        lmax: int,
        continuum: list
    ):
        self.gcm_binning = gcm_binning
        self.phase_binning = phase_binning
        self.use_molecular_signatures = use_molecular_signatures
        self.use_continuum_stellar = use_continuum_stellar
        self.nmax = nmax
        self.lmax = lmax
        self.continuum = continuum

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            gcm_binning=int(d['gcm_binning']),
            phase_binning=int(d['phase_binning']),
            use_molecular_signatures=bool(d['use_molecular_signatures']),
            use_continuum_stellar=bool(d.get('use_continuum_stellar',cls._defaults['use_stellar_continuum'])),
            nmax=int(d['nmax']),
            lmax=int(d['lmax']),
            continuum=list(d['continuum']),
        )
