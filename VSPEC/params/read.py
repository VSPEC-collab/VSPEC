"""
Module to read parameters
"""
from typing import Callable, List, Union
from pathlib import Path
import yaml
from astropy import units as u
from libpypsg import PyConfig
from libpypsg.cfg import models
from libpypsg.units import resolving_power as u_rp
from GridPolator import GridSpectra

from .. import config
from ..spectra import ForwardSpectra
from ..helpers import arrange_teff
from .base import BaseParameters
from .stellar import StarParameters
from .planet import PlanetParameters, SystemParameters
from .gcm import gcmParameters, psgParameters
from .observation import InstrumentParameters, ObservationParameters, SingleDishParameters, CoronagraphParameters


class AbstractGridParameters(BaseParameters):
    """
    The particulars of the grid of stellar spectral models.
    
    """
    
    def __init__(
        self,
        builder: Callable[..., GridSpectra],
        **kwargs
    ):
        self._builder = builder
        self._kwargs = kwargs
    def build(self,**kwargs)->GridSpectra:
        """
        Create a ``GridSpectra`` instance using additional parameters
        """
        return self._builder(**self._kwargs,**kwargs)
    @staticmethod
    def _get_subtype(name: str)->Union['VSPECGridParameters']:
        match name:
            case 'vspec':
                return VSPECGridParameters
            case 'bb':
                return BlackbodyGridParameters
            case _:
                raise NotImplementedError(f'Grid type {name} not implemented.')
    @classmethod
    def from_dict(cls, d: dict):
        return cls._get_subtype(d['name']).from_dict(d)
class VSPECGridParameters(AbstractGridParameters):
    """
    Parameter container for the default VSPEC grid.
    
    Parameters
    ----------
    max_teff : astropy.units.Quantity
        The maximum effective temperature.
    min_teff : astropy.units.Quantity
        The minimum effective temperature.
    impl_bin : str, optional
        The implementation of the binning algorithm. Default is 'rust'.
    impl_interp : str, optional
        The implementation of the interpolation algorithm. Default is 'scipy'.
    fail_on_missing : bool, optional
        If ``True``, raise an error if a spectrum from the grid is not found.
        If ``False``, download the needed spectra. Default is ``False``.
    
    """
    
    _defaults = {
        'impl_bin': 'rust',
        'impl_interp': 'scipy',
        'fail_on_missing': False
    }
    def __init__(
        self,
        max_teff: u.Quantity,
        min_teff: u.Quantity,
        impl_bin: str = 'rust',
        impl_interp: str = 'scipy',
        fail_on_missing: bool = False
    ):
        teffs = arrange_teff(minteff=min_teff, maxteff=max_teff)
        super().__init__(GridSpectra.from_vspec, teffs=teffs, impl_bin=impl_bin, impl_interp=impl_interp, fail_on_missing=fail_on_missing)

    def build(
        self,
        w1: u.Quantity,
        w2: u.Quantity,
        resolving_power: float,
    ):
        """
        Initialize a ``GridSpectra`` instance using additional parameters.
        
        Parameters
        ----------
        w1 : astropy.units.Quantity
            The w1 wavelength.
        w2 : astropy.units.Quantity
            The w2 wavelength.
        resolving_power : float
            The resolving power.
        
        Returns
        -------
        GridSpectra
            The grid of stellar spectral models.
        """
        return super().build(
            w1=w1,
            w2=w2,
            resolving_power=resolving_power,
        )
    @classmethod
    def from_dict(cls, d: dict):
        """
        Initialize a VSPEC grid parameter object from a dictionary.

        Parameters
        ----------
        d : dict
            The parameter dictionary.

        Returns
        -------
        VSPECGridParameters
            The VSPEC grid parameters.
        """
        return cls(
            max_teff=u.Quantity(d['max_teff']),
            min_teff=u.Quantity(d['min_teff']),
            impl_bin=str(d.get('impl_bin', cls._defaults['impl_bin'])),
            impl_interp=str(d.get('impl_interp', cls._defaults['impl_interp'])),
            fail_on_missing=bool(d.get('fail_on_missing', cls._defaults['fail_on_missing']))
        )

class BlackbodyGridParameters(AbstractGridParameters):
    """
    Parameter container for a forward-model blackbody stellar spectrum.
    
    Note
    ----
    The ``ForwardSpectra`` object API is identical to the ``GridSpectra`` object API.
    
    """
    
    def __init__(self):
        super().__init__(ForwardSpectra.blackbody)
    
    def build(self):
        """
        Initialize the ``ForwardSpectra`` object.
        """
        return super().build()

    @classmethod
    def from_dict(cls, d: dict):
        """
        Initialize from a dictionary.
        """
        return cls()


class Header(BaseParameters):
    """
    Header for VSPEC simulation

    Parameters
    ----------
    data_path : pathlib.Path
        The path to store run data.
    max_teff : astropy.units.Quantity
        The maximum Teff to bin.
    min_teff : astropy.units.Quantity
        The minimum Teff to bin.
    seed : int, default=None
        The seed for the random number generator.
    verbose : int, default=1,
        The level of verbosity for the simulation.
    desc : str, default=None
        A description of the run.

    Attributes
    ----------
    data_path : pathlib.Path
        The path to store run data.
    max_teff : astropy.units.Quantity
        The maximum Teff to bin.
    min_teff : astropy.units.Quantity
        The minimum Teff to bin.
    seed : int or None
        The seed for the random number generator.
    verbose : int
        The level of verbosity for the simulation.
    desc : str or None
        A description of the run.

    """

    def __init__(
        self,
        data_path: Path,
        spec_grid: AbstractGridParameters,
        seed: int,
        verbose: int = 1,
        desc: str = None
    ):
        self.data_path = data_path
        self.spec_grid = spec_grid
        self.seed = seed
        self.verbose = verbose
        self.desc = desc

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            data_path=config.VSPEC_PARENT_PATH / d['data_path'],
            spec_grid=AbstractGridParameters.from_dict(d['spec_grid']),
            seed=None if d.get('seed', None) is None else int(
                d.get('seed', None)),
            desc=None if d.get('desc', None) is None else str(
                d.get('desc', None))
        )


class InternalParameters(BaseParameters):
    """
    Class to store parameters for a VSPEC simulation.

    Parameters
    ----------
    header : Header
        The VSPEC simulation header.
    star : StarParameters
        The parameters related to the star.
    planet : PlanetParameters
        The parameters related to the planet.
    system : SystemParameters
        The parameters related to the system.
    obs : ObservationParameters
        The parameters related to the observation.
    psg : psgParameters
        The PSG parameters.
    inst : InstrumentParameters
        The instrument parameters.
    gcm : gcmParameters
        The GCM parameters.

    Attributes
    ----------
    header : Header
        The VSPEC simulation header.
    star : StarParameters
        The parameters related to the star.
    planet : PlanetParameters
        The parameters related to the planet.
    system : SystemParameters
        The parameters related to the system.
    obs : ObservationParameters
        The parameters related to the observation.
    psg : psgParameters
        The PSG parameters.
    inst : InstrumentParameters
        The instrument parameters.
    gcm : gcmParameters
        The GCM parameters.


    """

    def __init__(
        self,
        header: Header,
        star: StarParameters,
        planet: PlanetParameters,
        system: SystemParameters,
        obs: ObservationParameters,
        psg: psgParameters,
        inst: InstrumentParameters,
        gcm: gcmParameters
    ):
        self.header = header
        self.star = star
        self.planet = planet
        self.system = system
        self.obs = obs
        self.psg = psg
        self.inst = inst
        self.gcm = gcm

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            header=Header.from_dict(d['header']),
            star=StarParameters.from_dict(d['star']),
            planet=PlanetParameters.from_dict(d['planet']),
            system=SystemParameters.from_dict(d['system']),
            obs=ObservationParameters.from_dict(d['obs']),
            psg=psgParameters.from_dict(d['psg']),
            inst=InstrumentParameters.from_dict(d['inst']),
            gcm=gcmParameters.from_dict(d)
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'InternalParameters':
        """
        Create and `InternalParameters` instance from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary to construct the class from.

        Returns
        -------
        InternalParameters
            An instance of `InternalParameters`.
        """
        return super().from_dict(d)

    @classmethod
    def from_yaml(cls, path: Path) -> 'InternalParameters':
        """
        Create an `InternalParameters` instance from a YAML file.

        Parameters
        ----------
        path : Path
            The path to the YAML file.

        Returns
        -------
        InternalParameters
            An instance of the `InternalParameters` class.

        """
        with open(path, 'r', encoding='UTF-8') as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    @property
    def target(self):
        """
        Get the target model.
        """
        target = models.Target(
            object='Exoplanet',
            name=self.planet.name,
            date=None,
            diameter=2*self.planet.radius,
            gravity=self.planet.gravity.value,
            star_distance=self.planet.semimajor_axis,
            star_velocity=None,
            solar_longitude=None,
            solar_latitude=None,
            season=None,
            inclination=self.system.inclination,
            position_angle=None,
            star_type=self.star.psg_star_template,
            star_temperature=self.star.teff,
            star_radius=self.star.radius,
            star_metallicity=None,
            obs_longitude=None,
            obs_latitude=None,
            obs_velocity=None,
            period=self.planet.orbit_period,
            orbit=None
        )
        return target

    @property
    def geometry(self):
        """
        Get the geometry model.
        """
        return models.Observatory(
            ref=None,
            offset=None,
            observer_altitude=self.system.distance,
            azimuth=None,
            stellar_type=self.star.psg_star_template,
            stellar_temperature=self.star.teff,
            stellar_magnitude=None,
            disk_angles=None,
        )

    @property
    def atmosphere(self):
        """
        Get the atmosphere model.
        """
        return models.EquilibriumAtmosphere(
            weight=self.gcm.mean_molec_weight*u.Unit('g/mol'),
            continuum=','.join(self.psg.continuum),
            nmax=self.psg.nmax,
            lmax=self.psg.lmax
        )

    @property
    def surface(self):
        """
        Get the surface model.
        """
        return models.Surface(
            albedo=None,
            temperature=None,
            emissivity=None
        )

    @property
    def generator(self):
        """
        Get the generator model.
        """
        return models.Generator(
            resolution_kernel=None,
            gas_model=self.psg.use_molecular_signatures,
            continuum_stellar=self.psg.use_continuum_stellar,
            apply_telluric_noise=None,
            apply_telluric_obs=None,
            telluric_params=None,
            rad_units=config.flux_unit,
            log_rad=None,
            gcm_binning=self.psg.gcm_binning
        )

    @property
    def telescope(self):
        """
        Get the telescope model.
        """
        if isinstance(self.inst.telescope, SingleDishParameters):
            return models.SingleTelescope(
                apperture=self.inst.telescope.aperture,
                zodi=self.inst.telescope.zodi,
                fov=self.inst.detector.beam_width,
                range1=self.inst.bandpass.wl_blue,
                range2=self.inst.bandpass.wl_red,
                resolution=self.inst.bandpass.resolving_power*u_rp
            )
        elif isinstance(self.inst.telescope, CoronagraphParameters):
            return models.Coronagraph(
                apperture=self.inst.telescope.aperture,
                zodi=self.inst.telescope.zodi,
                fov=self.inst.detector.beam_width,
                range1=self.inst.bandpass.wl_blue,
                range2=self.inst.bandpass.wl_red,
                resolution=self.inst.bandpass.resolving_power*u_rp,
                contrast=self.inst.telescope.contrast,
                iwa=self.inst.telescope.iwa
            )
        else:
            raise ValueError('Unknown telescope type')

    @property
    def noise(self):
        """
        Get the noise model.
        """
        return models.CCD(
            exp_time=self.inst.detector.integration_time,
            n_frames=int(round(
                (self.obs.integration_time/self.inst.detector.integration_time).to_value(u.dimensionless_unscaled))),
            n_pixels=self.inst.detector.ccd.pixel_sampling,
            read_noise=self.inst.detector.ccd.read_noise,
            dark_current=self.inst.detector.ccd.dark_current,
            thoughput=self.inst.detector.ccd.throughput,
            emissivity=self.inst.detector.ccd.emissivity,
            temperature=self.inst.detector.ccd.temperature,
            pixel_depth=None
        )

    def to_pyconfig(self):
        """
        Write VSPEC parameters into a `libpypsg` `PyConfig` object.
        """
        return PyConfig(
            target=self.target,
            geometry=self.geometry,
            atmosphere=self.atmosphere,
            surface=self.surface,
            generator=self.generator,
            telescope=self.telescope,
            noise=self.noise,
            gcm=None
        )

    @property
    def flux_correction(self) -> float:
        """
        The flux correction for the stellar radius and distance.

        Returns
        -------
        float
            Correction for the solid angle of the star.
        """
        return (self.star.radius/self.system.distance).to_value(u.dimensionless_unscaled)**2

    @property
    def star_total_images(self) -> int:
        """
        The number of epochs to simulate for the stellar model.

        Returns
        -------
        int
            The total number of epochs to simulate the star.
        """
        return self.obs.total_images

    @property
    def planet_total_images(self) -> int:
        """
        The number of epochs to simulate for the planet model.

        Returns
        -------
        int
            The total number of epochs to simulate the planet.
        """
        return self.obs.total_images // self.psg.phase_binning
