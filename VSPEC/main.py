"""VSPEC main module

This module performs all of VSPEC's interaction with the user.
It contains the `ObservationModel` class, which has methods that
perform all of the model aggregation from the rest of the package
and PSG.
"""

from pathlib import Path
import warnings
import logging
from functools import partial
from typing import Dict, Tuple

import numpy as np
from astropy import units as u
from astropy.table import QTable
from tqdm.auto import tqdm

from GridPolator import GridSpectra
from GridPolator.binning import get_wavelengths

import libpypsg
from libpypsg import docker as psg_docker

import vspec_vsm as vsm

# from VSPEC import variable_star_model as vsm
from VSPEC.config import PSG_CFG_MAX_LINES, N_ZFILL
from VSPEC import config
from VSPEC.geometry import SystemGeometry
from VSPEC.helpers import check_and_build_dir, get_filename
from VSPEC.helpers import get_planet_indicies
from VSPEC.psg_api import change_psg_parameters
from VSPEC.params.read import InternalParameters, VSPECGridParameters, BlackbodyGridParameters
from VSPEC.spectra import ForwardSpectra


class ObservationModel:
    """
    Main class that stores the information of this simulation.

    Parameters
    ----------
    params : VSPEC.params.InternalParameters
        The global parameters describing the VSPEC simulation.

    Examples
    --------
    >>> model = ObservationModel.from_yaml('config.yaml')
    >>> model.build_planet()
    Build Planet: 100%|██████████| 26/26 [00:02<00:00,  8.68it/s]
    >>> model.build_spectra()
    Build Spectra: 100%|██████████| 26/26 [00:15<00:00,  1.66it/s]

    >>> model = ObservationModel(params=my_params)
    >>> model.build_planet()
    Build Planet: 100%|██████████| 18/18 [00:07<00:00,  2.62it/s]
    >>> model.build_spectra()
    Build Spectra: 100%|██████████| 18/18 [02:35<00:00,  8.65s/it]

    """

    def __init__(
        self,
        params: InternalParameters
    ):
        if isinstance(params, (Path, str)):
            msg = 'Please use the `from_yaml` classmethod'
            raise TypeError(msg)
        self.params = params
        self.verbose = params.header.verbose
        self._build_directories()
        self.rng = np.random.default_rng(self.params.header.seed)
        # Load later when needed.
        self.spec: GridSpectra | ForwardSpectra = None
        self.star: vsm.Star = None
        self.bb = ForwardSpectra.blackbody()
        self.logger = logging.Logger('VSPEC')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.directories['parent'] / 'psg.log',mode='w')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        # warn if the planet sampling is too low.
        planet_dt = self.params.obs.observation_time / self.params.planet_total_images
        max_dt = self.params.planet.orbit_period / 8.0
        if planet_dt > max_dt:
            msg = f'Planet sampling is too low ({planet_dt:.2f} > {max_dt:.2f})'
            warnings.warn(msg, RuntimeWarning)

    @classmethod
    def from_yaml(cls, config_path: Path):
        """
        Initialize a VSPEC run from a YAML file.

        Parameters
        ----------
        config_path : pathlib.Path
            The path to the YAML file.
        """
        params = InternalParameters.from_yaml(config_path)
        return cls(params)

    class __flags__:
        """
        Flags attribute

        Attributes
        ----------
        psg_needs_set : bool
            For security reasons, PSG config files can contain a
            maximum of ~2000 lines. After that number, the server
            will no longer update after our API call. In this case,
            we issue the `set` command. Unfortuantely, we must re-
            upload the GCM.
        """
        psg_needs_set = True
    _directories = {
        'parent': '',
        'all_model': 'AllModelSpectraValues',
        'psg_combined': 'PSGCombinedSpectra',
        'psg_thermal': 'PSGThermalSpectra',
        'psg_noise': 'PSGNoise',
        'psg_layers': 'PSGLayers',
        'psg_configs': 'PSGConfig'
    }

    @property
    def directories(self) -> dict:
        """
        The directory structure for the VSPEC run.

        Returns
        -------
        dict
            Keys represent the identifiers of directories, and the values are
            `pathlib.Path` objects.

        """
        parent_dir = self.params.header.data_path
        dir_dict = {key: parent_dir/value for key,
                    value in self._directories.items()}
        return dir_dict

    def _wrap_iterator(self, iterator, **kwargs):
        """
        Wrapper for iterators so that `tqdm` can be used
        only if `self.verbose` > 0

        Parameters
        ----------
        iterator : iterable
            Iterator to be passed to `tqdm`
        **kwargs : dict
            The keywords to pass to `tqdm`

        Returns
        -------
        iterable
            The iterator wrapped appropriately.
        """
        if self.verbose > 0:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def _build_directories(self):
        """
        Build the file system for this run.
        """
        for _, path in self.directories.items():
            check_and_build_dir(path)

    def _load_spectra(self):
        """
        Load a GridSpec instance.

        Returns
        -------
        VSPEC.spectra.GridSpec
            The spectal grid object to draw stellar spectra from.
        """
        p = self.params.header.spec_grid
        if isinstance(p, VSPECGridParameters):
            return p.build(
                w1=self.params.inst.bandpass.wl_blue,
                w2=self.params.inst.bandpass.wl_red,
                resolving_power=self.params.inst.bandpass.resolving_power,
            )
        elif isinstance(p, BlackbodyGridParameters):
            return p.build()
        else:
            raise NotImplementedError(f'Unsure how to build ``GridSpectra`` object from {p}')

    @property
    def _wl(self)->u.Quantity:
        """
        The wavelength axis of the observation.

        Returns
        -------
        wl : astropy.units.Quantity
            The wavelength axis.
        """
        return get_wavelengths(
            resolving_power=self.params.inst.bandpass.resolving_power,
            lam1=self.params.inst.bandpass.wl_blue.to_value(config.wl_unit),
            lam2=self.params.inst.bandpass.wl_red.to_value(config.wl_unit)
        )[:-1]*config.wl_unit

    def _get_model_spectrum(self, teff: u.Quantity) -> u.Quantity:
        """
        Get the interpolated spectrum given an effective temperature.

        Parameters
        ----------
        teff : astropy.units.Quantity
            The effective temperature

        Returns
        -------
        astropy.units.Quantity
            The flux of the spectrum.

        Notes
        -----
        This function applies the solid angle correction.
        """
        if self.spec is None:
            self.spec = self._load_spectra()

        
        if isinstance(self.params.header.spec_grid, VSPECGridParameters):
            teffs = np.atleast_1d(teff.to_value(config.teff_unit))
            return self.spec.evaluate(
                params=(teffs,),
                wl=np.array(self._wl.to_value(config.wl_unit))
            )[0, :] * config.flux_unit * self.params.flux_correction
        elif isinstance(self.params.header.spec_grid, BlackbodyGridParameters):
            return self.spec.evaluate(self._wl, teff) * self.params.flux_correction
        else:
            raise TypeError(f'Unsure how to handle grid parameters of type {type(self.params.header.spec_grid)}')

    def get_observation_parameters(self) -> SystemGeometry:
        """

        Get an object to store and compute the geometric observational
        parameters for this simulation.


        Returns
        -------
        VSPEC.geometry.SystemGeometry
            An bbject storing the geometric observation parameters
            of this simulation.
        """
        return SystemGeometry(inclination=self.params.system.inclination,
                              init_stellar_lon=0*u.deg,
                              init_planet_phase=self.params.planet.init_phase,
                              stellar_period=self.params.star.period,
                              orbital_period=self.params.planet.orbit_period,
                              semimajor_axis=self.params.planet.semimajor_axis,
                              planetary_rot_period=self.params.planet.rotation_period,
                              planetary_init_substellar_lon=self.params.planet.init_substellar_lon,
                              stellar_offset_amp=self.params.star.misalignment,
                              stellar_offset_phase=self.params.star.misalignment_dir,
                              eccentricity=self.params.planet.eccentricity,
                              phase_of_periastron=self.params.system.phase_of_periastron,
                              system_distance=self.params.system.distance,
                              obliquity=self.params.planet.obliquity,
                              obliquity_direction=self.params.planet.obliquity_direction)

    def _get_observation_plan(self, observation_parameters: SystemGeometry, planet=False) -> QTable:
        """
        Compute the locations and geometries of each object in this simulation.

        Parameters
        ----------
        observation_parameters : VSPEC.geometry.SystemGeometry
            An object containting the system geometry.
        planet : bool
            If true, use the planet phase binning parameter to
            compute the number of steps.

        Returns
        -------
        QTable
            A table containing the observation plan.
        """
        int_time = self.params.obs.integration_time
        obs_time = self.params.obs.observation_time

        if not planet:
            n_obs = int(
                round((obs_time / int_time).to_value(u.dimensionless_unscaled)))
            start_times = np.arange(0, n_obs) * int_time
        else:
            n_obs = int(round(
                (obs_time / int_time/self.params.psg.phase_binning).to_value(u.dimensionless_unscaled)))
            start_times = np.arange(0, n_obs+1) * \
                int_time * self.params.psg.phase_binning

        return observation_parameters.get_observation_plan(start_times)

    def _check_psg(self):
        """
        Check that PSG is configured correctly.

        Raises
        ------
        RuntimeError
            If a local PSG container is specified but the port is not in use
            (i.e. PSG is not running).

        Warns
        -----
        RuntimeWarning
            If calling the online PSG API, but no API key is specified.
        """
        if libpypsg.settings.get_setting('url') == libpypsg.settings.PSG_URL:
            if libpypsg.settings.get_setting('api_key') is None:
                warnings.warn('PSG is being called without an API key. ' +
                              'After 100 API calls in a 24hr period you will need to get a key. ' +
                              'We suggest installing PSG locally using docker. (see https://psg.gsfc.nasa.gov/help.php#handbook)',
                              RuntimeWarning)
        else:
            if not psg_docker.is_psg_installed():
                msg = 'PSG is not installed. '
                msg += 'We suggest installing PSG locally using docker. (see https://psg.gsfc.nasa.gov/help.php#handbook) '
                msg += 'otherwise set the `libpypsg` url setting to the online PSG url:\n'
                msg += f'libpypsg.settings.save_settings(url=\'{libpypsg.settings.PSG_URL}\')'
                raise RuntimeError(msg)
            elif not psg_docker.is_psg_running():
                raise RuntimeError('PSG is not running.\n' +
                                   'Type `docker start psg` in the command line ' +
                                   'or run `libpypsg.docker.start_psg()`')

    def _upload_gcm(self, obstime: u.Quantity = 0*u.s, update=False):
        """
        Upload GCM file to PSG

        Parameters
        ----------
        is_ncdf : bool
            Whether or not to use a netCDF file for the GCM
        obstime : astropy.units.Quantity, default=0*u.s
            The time since the start of the observation.
        update : bool
            Whether to use the `'upd'` keyword rather than `'set'`
        """
        if not self.params.gcm.is_staic:
            kwargs = {'obs_time': obstime}
        else:
            kwargs = {}
        cfg = self.params.gcm.to_pycfg(**kwargs)
        caller = libpypsg.APICall(
            cfg=cfg,
            output_type='upd' if update else 'set',
            app='globes',
            logger=self.logger
        )
        _ = caller()
        if not update:
            self.__flags__.psg_needs_set = False

    def _set_static_config(self):
        """
        Upload the non-changing parts of the PSG config.
        """
        cfg = self.params.to_pyconfig()
        caller = libpypsg.APICall(
            cfg=cfg,
            output_type='upd',
            app='globes',
            logger=self.logger
        )
        _ = caller()

    def _psg_all_call(
        self,
        phase: u.Quantity,
        orbit_radius_coeff: float,
        sub_stellar_lon: u.Quantity,
        sub_stellar_lat: u.Quantity,
        pl_sub_obs_lon: u.Quantity,
        pl_sub_obs_lat: u.Quantity,
        include_star: bool,
        path_dict: dict,
        i: int
    ):
        """
        Update the PSG config with time-dependent values.

        Parameters
        ----------
        phase : astropy.units.Quantity
            The phase of the planet
        orbit_radius_coeff : float
            The ratio between the current planet-host difference
            and the semimajor axis
        sub_stellar_lon : astropy.units.Quantity
            The sub-stellar longitude of the planet.
        sub_stellar_lat : astropy.units.Quantity
            The sub-stellar latitude of the planet.
        pl_sub_obs_lon : astropy.units.Quantity
            The sub-observer longitude of the planet.
        pl_sub_obs_lat : astropy.units.Quantity
            The sub-observer latitude of the planet.
        include_star : bool
            Whether to include the star in the simulation.
        path_dict : dict
            A dictionary that determines where each downloaded file
            gets written to.
        i : int
            The index of the current observation
        """
        cfg = change_psg_parameters(
            params=self.params,
            phase=phase,
            orbit_radius_coeff=orbit_radius_coeff,
            sub_stellar_lon=sub_stellar_lon,
            sub_stellar_lat=sub_stellar_lat,
            pl_sub_obs_lon=pl_sub_obs_lon,
            pl_sub_obs_lat=pl_sub_obs_lat,
            include_star=include_star
        )
        caller = libpypsg.APICall(
            cfg=cfg,
            output_type='all',
            app='globes',
            logger=self.logger
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            response = caller()
            for _w in w:
                if _w.category is libpypsg.cfg.ConfigTooLongWarning:
                    self.__flags__.psg_needs_set = True
        rad = response.rad
        noi = response.noi
        lyr = response.lyr
        cfg = response.cfg
        
        if rad is None:
            raise RuntimeError('PSG returned no .rad file')
        else:
            key = 'rad'
            filename = get_filename(i, N_ZFILL, 'fits')
            rad.write(
                path_dict[key]/filename,
                format='fits',
                overwrite=True
            )
            if include_star and 'Stellar' not in rad.colnames:
                raise ValueError('No Column named "Stellar" in .rad file')

        if noi is None or 'noi' not in path_dict:
            pass
        else:
            key = 'noi'
            filename = get_filename(i, N_ZFILL, 'fits')
            noi.write(
                path_dict[key]/filename,
                format='fits',
                overwrite=True
            )

        if cfg is None:
            raise RuntimeError('PSG returned no .cfg file')
        elif 'cfg' not in path_dict:
            pass
        else:
            key = 'cfg'
            filename = get_filename(i, N_ZFILL, 'cfg')
            cfg.to_file(path_dict[key]/filename)

            # TODO: Check config against expected

        if lyr is None or 'lyr' not in path_dict:
            pass
        else:
            key = 'lyr'
            filename = get_filename(i, N_ZFILL, 'fits')
            lyr.to_fits(path_dict[key]/filename)

    def _check_config(self, cfg_from_psg: libpypsg.PyConfig):
        """
        Validate the config file recieved from PSG to ensure that
        the parameters sent are the parameters used in the simulation.

        Parameters
        ----------
        cfg_from_psg : libpypsg.PyConfig
            The config file recieved from PSG.

        Raises
        ------
        RuntimeError
            If the config recieved does not match the config sent.
        """
        n_lines = len(cfg_from_psg.content.split('\n'))
        if n_lines > PSG_CFG_MAX_LINES:
            self.__flags__.psg_needs_set = True

        expected_cfg = self.params.to_pyconfig()

        assert expected_cfg.target == cfg_from_psg.target, 'PSG target does not match'
        assert expected_cfg.geometry == cfg_from_psg.geometry, 'PSG geometry does not match'
        assert expected_cfg.atmosphere == cfg_from_psg.atmosphere, 'PSG atmosphere does not match'
        assert expected_cfg.surface == cfg_from_psg.surface, 'PSG surface does not match'
        assert expected_cfg.generator == cfg_from_psg.generator, 'PSG generator does not match'
        assert expected_cfg.telescope == cfg_from_psg.telescope, 'PSG telescope does not match'
        assert expected_cfg.noise == cfg_from_psg.noise, 'PSG noise does not match'
        # do not check the GCM


    def build_planet(self):
        """
        Use the PSG GlobES API to construct a planetary phase curve.
        Follow steps in original PlanetBuilder.py file
        
        Examples
        --------
        >>> model = ObservationModel.from_yaml(CFG_PATH)
        >>> model.build_planet()
        Build Planet: 100%|██████████| 36/36 [00:07<00:00,  4.60it/s]

        """
        # check that psg is running
        self._check_psg()
        # for not using globes, append all configurations instead of rewritting

        ####################################
        # Initial upload of GCM

        self._upload_gcm(
            obstime=0*u.s,
            update=False
        )
        ####################################
        # Set observation parameters that do not change
        self._set_static_config()

        ####################################
        # Calculate observation parameters
        observation_parameters = self.get_observation_parameters()
        obs_plan = self._get_observation_plan(
            observation_parameters, planet=True)

        obs_info_filename = Path(
            self.directories['parent']) / 'observation_info.fits'
        obs_plan.write(obs_info_filename, overwrite=True)

        if self.verbose > 0:
            print(
                f'Starting at phase {self.params.planet.init_phase}, observe for {self.params.obs.observation_time} in {self.params.planet_total_images} steps')
            print('Phases = ' +
                  str(np.round(np.asarray((obs_plan['phase']/u.deg).to(u.Unit(''))), 2)) + ' deg')
        ####################################
        # iterate through phases
        for i in self._wrap_iterator(range(self.params.planet_total_images+1), desc='Build Planet', total=self.params.planet_total_images+1):
            phase = obs_plan['phase'][i]
            sub_stellar_lon = obs_plan['sub_stellar_lon'][i]
            sub_stellar_lat = obs_plan['sub_stellar_lat'][i]
            pl_sub_obs_lon = obs_plan['planet_sub_obs_lon'][i]
            pl_sub_obs_lat = obs_plan['planet_sub_obs_lat'][i]
            orbit_radius_coeff = obs_plan['orbit_radius'][i]
            obs_time = obs_plan['time'][i] - obs_plan['time'][0]

            if (not self.params.gcm.is_staic) or self.__flags__.psg_needs_set:
                # enter if we need a reset or if there is time dependence
                upload = partial(self._upload_gcm, obstime=obs_time)
                if self.__flags__.psg_needs_set:  # do a reset if needed
                    upload(update=False)
                    self._set_static_config()
                    self.__flags__.psg_needs_set = False
                else:  # update if it's just time dependence.
                    upload(update=True)

            # Write updates to the config to change the phase value and ensure the star is of type 'StarType'
            update_config = partial(
                self._psg_all_call,
                phase=phase,
                orbit_radius_coeff=orbit_radius_coeff,
                sub_stellar_lon=sub_stellar_lon,
                sub_stellar_lat=sub_stellar_lat,
                pl_sub_obs_lon=pl_sub_obs_lon,
                pl_sub_obs_lat=pl_sub_obs_lat
            )
            
            path_dict = {
                'rad': Path(self.directories['psg_combined']),
                'noi': Path(self.directories['psg_noise']),
                'cfg': Path(self.directories['psg_configs']),
            }
            update_config(
                include_star=True,
                path_dict=path_dict,
                i=i
            )
            # write updates to config file to remove star flux
            
            path_dict = {
                'rad': Path(self.directories['psg_thermal']),
                'lyr': Path(self.directories['psg_layers'])
            }
            update_config(
                include_star=False,
                path_dict=path_dict,
                i=i
            )

    def _build_star(self):
        """
        Build a variable star model based on user-specified parameters.
        """
        self.star = self.params.star.to_star(
            rng=self.rng,
            seed=self.params.header.seed
        )

    def _warm_up_star(self, spot_warmup_time: u.Quantity = 0*u.day, facula_warmup_time: u.Quantity = 0*u.day):
        """
        "Warm up" the star. Generate spots, faculae, and/or flares for the star.
        The goal is to approach growth-decay equillibrium, something that is hard to
        do with a purely "hot star" method (like 
        `VPSEC.variable_star_model.Star.generate_mature_spots`).

        Parameters
        ----------
        spot_warm_up_time : astropy.units.Quantity [time], default=0*u.day
            The time to run to approach spot equillibrium.
        facula_warmup_time : astropy.units.Quantity [time], default=0*u.hr
            The time to run to approach faculae equillibrium.
        """
        if self.params.star.spots.initial_coverage > 0.0:
            self.star.generate_mature_spots(
                self.params.star.spots.initial_coverage)
            print(f'Generated {len(self.star.spots.spots)} mature spots')
        spot_warm_up_step = 1*u.day
        facula_warm_up_step = 1*u.hr
        N_steps_spot = int(
            round((spot_warmup_time/spot_warm_up_step).to(u.Unit('')).value))
        N_steps_facula = int(
            round((facula_warmup_time/facula_warm_up_step).to(u.Unit('')).value))
        if N_steps_spot > 0:
            for _ in self._wrap_iterator(range(N_steps_spot), desc='Spot Warmup', total=N_steps_spot):
                self.star.birth_spots(spot_warm_up_step)
                self.star.age(spot_warm_up_step)
        if N_steps_facula > 0:
            for _ in self._wrap_iterator(range(N_steps_facula), desc='Facula Warmup', total=N_steps_facula):
                self.star.birth_faculae(facula_warm_up_step)
                self.star.age(facula_warm_up_step)

        self.star.get_flares_over_observation(
            self.params.obs.observation_time)

    def _calculate_composite_stellar_spectrum(
        self,
        sub_obs_coords,
        tstart,
        tfinish,
        granulation_fraction: float = 0.0,
        orbit_radius: u.Quantity = 1*u.AU,
        planet_radius: u.Quantity = 1*u.R_earth,
        phase: u.Quantity = 90*u.deg,
        inclination: u.Quantity = 0*u.deg,
        transit_depth: np.ndarray | float = 0
    )->Tuple[u.Quantity, u.Quantity]:
        """
        Compute the stellar spectrum given an integration window and the
        side of the star facing the observer.

        Parameters
        ----------
        sub_obs_coords : dict
            A dictionary containing stellar sub-observer coordinates.
        tstart : astropy.units.Quantity [time]
            The starting time of the observation.
        tfinish : astropy.units.Quantity [time]
            The ending time of the observation.
        granulation_fraction : float
            The fraction of the quiet photosphere that has a lower Teff due to granulation

        Returns
        -------
        base_wave : astropy.units.Quantity [wavelength]
            The wavelength coordinates of the stellar spectrum.
        base_flux : astropy.units.Quantity [flambda]
            The composite stellar flux

        Raises
        ------
        ValueError
            If wavenelength coordinates do not match.
        """
        total, covered, pl_frac = self.star.calc_coverage(
            sub_obs_coords,
            granulation_fraction=granulation_fraction,
            orbit_radius=orbit_radius,
            planet_radius=planet_radius,
            phase=phase,
            inclination=inclination
        )
        visible_flares = self.star.get_flare_int_over_timeperiod(
            tstart, tfinish, sub_obs_coords)
        base_flux = self._get_model_spectrum(self.params.star.teff)
        base_flux = base_flux * 0
        # add up star flux before considering transit
        for teff, coverage in total.items():
            if coverage > 0:
                flux = self._get_model_spectrum(u.Quantity(teff))
                if not flux.shape == base_flux.shape:
                    raise ValueError('All arrays must have same shape.')
                base_flux = base_flux + flux * coverage
        # get flux of transited region
        transit_flux = base_flux*0
        for teff, coverage in covered.items():
            if coverage > 0:
                flux = self._get_model_spectrum(u.Quantity(teff))
                if not flux.shape == base_flux.shape:
                    raise ValueError('All arrays must have same shape.')
                transit_flux = transit_flux + flux * coverage
        # scale according to effective radius
        transit_flux = transit_flux * transit_depth
        base_flux = base_flux - transit_flux
        # add in flares
        for flare in visible_flares:
            teff = flare['Teff']
            timearea = flare['timearea']
            eff_area = (timearea/(tfinish-tstart)).to(u.km**2)
            correction = (eff_area/self.params.system.distance **
                          2).to_value(u.dimensionless_unscaled)
            flux = self.bb.evaluate(self._wl, teff) * correction
            base_flux = base_flux + flux

        return base_flux.to(config.flux_unit), pl_frac

    def _get_pyrad(
        self,
        kind: str,
        index: int
    )-> libpypsg.PyRad:
        """
        Read a rad file.
        """
        match kind:
            case 'thermal':
                path = self.directories['psg_thermal'] / get_filename(index, N_ZFILL, 'fits')
            case 'combined':
                path = self.directories['psg_combined'] / get_filename(index, N_ZFILL, 'fits')
            case 'noise':
                path = self.directories['psg_noise'] / get_filename(index, N_ZFILL, 'fits')
        return libpypsg.PyRad.read(path, format='fits')
    
    def _get_psg_interp(
        self,
        kind:str,
        name:str,
    ):
        """
        Create an interpolaor from PSG outputs.
        """
        observation_parameters = self.get_observation_parameters()
        observation_info = self._get_observation_plan(
            observation_parameters, planet=True)
        times = observation_info['time'].to_value(config.time_unit)
        spectra = [
            self._get_pyrad(kind, i)[name].to_value(config.flux_unit) for i in range(self.params.planet_total_images+1)
        ]
        return GridSpectra(
            native_wl=self._wl.to_value(config.wl_unit),
            params = {'time': times},
            spectra = np.array(spectra),
            impl='scipy'
        )
    
    def _calculate_reflected_spectra(
        self,
        start_time: float,
        end_time: float,
        reflected_interpolator: GridSpectra,
        stellar_interpolator: GridSpectra,
        sub_planet_flux: u.Quantity,
        pl_frac: float
    ) -> u.Quantity:
        """
        Calculate the reflected spectrum based on PSG output and
        our own stellar model. We scale the reflected spectra from PSG
        to our model.

        Parameters
        ----------
        start_time : float
            The start time of the observation.
        end_time : float
            The end time of the observation.
        _thermal_interpolator : GridSpectra
            The thermal interpolator.
        _combined_interpolator : GridSpectra
            The thermal+reflected interpolator.
        _stellar_interpolator : GridSpectra
            The stellar interpolator.
        sub_planet_flux : astropy.units.Quantity
            The sub-planet flux.
        pl_frac : float
            The planet fraction.

        Returns
        -------
        reflected_flux : astropy.units.Quantity
            Reflected flux.

        """
        reflected_spectrum = reflected_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        stellar_spectrum = stellar_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        
        planet_reflection_fraction = (reflected_spectrum / stellar_spectrum)
        
        return sub_planet_flux * planet_reflection_fraction * pl_frac
    
    def _get_thermal_interpolator(self)-> GridSpectra:
        """
        Get an interpolator for the thermal spectra.
        """
        observation_parameters = self.get_observation_parameters()
        observation_info = self._get_observation_plan(
            observation_parameters, planet=True)
        times = observation_info['time'].to_value(config.time_unit)
        spectra = []
        for i in range(self.params.planet_total_images+1):
            pyrad = self._get_pyrad('thermal', i)
            if 'Thermal' in pyrad.columns:
                spectra.append(pyrad['Thermal'].to_value(config.flux_unit))
            else:
                if 'Transit' in pyrad.columns:
                    spectra.append(
                        (pyrad[self._thermal_name] - pyrad['Transit']).to_value(config.flux_unit)
                    )
                else:
                    spectra.append(pyrad[self._thermal_name].to_value(config.flux_unit))
        return GridSpectra(
            native_wl=self._wl.to_value(config.wl_unit),
            params = {'time': times},
            spectra = np.array(spectra),
            impl='scipy'
        )
    def _get_reflected_interpolator(self)-> GridSpectra:
        """
        Get an interpolator for the reflected spectra.
        """
        observation_parameters = self.get_observation_parameters()
        observation_info = self._get_observation_plan(
            observation_parameters, planet=True)
        times = observation_info['time'].to_value(config.time_unit)
        spectra = []
        for i in range(self.params.planet_total_images+1):
            combined = self._get_pyrad('combined', i)
            thermal = self._get_pyrad('thermal', i)
            if 'Reflected' in combined.columns:
                spectra.append(combined['Reflected'].to_value(config.flux_unit))
            else:
                if 'Transit' in combined.columns:
                    cmb_spec = combined[self._thermal_name] - combined['Transit']
                    the_spec = thermal[self._thermal_name]
                    spectra.append((cmb_spec - the_spec).to_value(config.flux_unit))
                else:
                    cmb_spec = combined[self._thermal_name]
                    the_spec = thermal[self._thermal_name]
                    spectra.append((cmb_spec - the_spec).to_value(config.flux_unit))
        return GridSpectra(
            native_wl=self._wl.to_value(config.wl_unit),
            params = {'time': times},
            spectra = np.array(spectra),
            impl='scipy'
        )

    def _get_transit_interpolator(self)-> GridSpectra:
        """
        Get an interpolator for the transit spectra.
        """
        observation_parameters = self.get_observation_parameters()
        observation_info = self._get_observation_plan(
            observation_parameters, planet=True)
        times = observation_info['time'].to_value(config.time_unit)
        spectra = []
        for i in range(self.params.planet_total_images+1):
            try:
                pyrad = self._get_pyrad('combined', i)
                spectra.append(
                    (-pyrad['Transit']/pyrad['Stellar']).to_value(u.dimensionless_unscaled)
                )
            except KeyError:
                spectra.append(
                    np.zeros_like(self._wl.value)
                )
        return GridSpectra(
            native_wl=self._wl.to_value(config.wl_unit),
            params = {'time': times},
            spectra = np.array(spectra),
            impl='scipy'
        )
        

    def _get_transit(
        self,
        start_time: float,
        end_time: float,
        transit_interpolator: GridSpectra,
    ) -> np.ndarray:
        """
        Get the transit spectra calculated by PSG

        Parameters
        ----------
        start_time : float
            The start time of the observation.
        end_time : float
            The end time of the observation.
        transit_interpolator : GridSpectra
            The transit interpolator.
        star_interpolator : GridSpectra
            The star interpolator.

        Returns
        -------
        flux : np.ndarray
            The dimensionless fraction of light blocked by the planet at each wavelength.

        """
        transit = transit_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        
        depth_bare_rock: float = (self.params.planet.radius /
                                  self.params.star.radius).to_value(u.dimensionless_unscaled)**2
        return transit/depth_bare_rock

    def _calculate_noise(
        self,
        start_time: float,
        end_time: float,
        photon_noise_interpolator: GridSpectra,
        detector_noise_interpolator: GridSpectra,
        telescope_noise_interpolator: GridSpectra,
        background_noise_interpolator: GridSpectra,
        star_interpolator: GridSpectra,
        time_scale_factor: float,
        cmb_flux: u.Quantity
    ):
        """
        Calculate the noise in our model based on the noise output from PSG.

        Parameters
        ----------
        start_time : astropy.units.Quantity
            The start time of the observation.
        end_time : astropy.units.Quantity
            The end time of the observation.
        _photon_noise_interpolator : GridSpectra
            The interpolator for the photon noise.
        _detector_noise_interpolator : GridSpectra
            The interpolator for the detector noise.
        _telescope_noise_interpolator : GridSpectra
            The interpolator for the telescope noise.
        _background_noise_interpolator : GridSpectra
            The interpolator for the background noise.
        _star_interpolator : GridSpectra
            The interpolator for the star flux.
        time_scale_factor : float
            A scaling factor to apply to the noise at the end of the calculation.
            This is 1 if the planet and star sampling has the same cadence. Otherwise,
            it is usually `sqrt(self.planet_phase_binning)`.
        cmb_flux : astropy.units.Quantity [flambda]
            The flux of the combined spectrum.

        Returns
        -------
        noise : astropy.units.Quantity [flambda]
            The noise in our model.

        Raises
        ------
        ValueError
            If the PSG flux unit code is not recognized.
        ValueError
            If the wavelength coordinates from the loaded spectra do not match.
        """
        photon_noise = photon_noise_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        star = star_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        model_noise = photon_noise * np.sqrt(cmb_flux.to_value(config.flux_unit)/star)
        
        detector_noise = detector_noise_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        telescope_noise = telescope_noise_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        background_noise = background_noise_interpolator.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)

        noise_sq = (model_noise**2
                    + (detector_noise)**2
                    + (telescope_noise)**2
                    + (background_noise)**2)
        return np.sqrt(noise_sq) * time_scale_factor
    
    @property
    def _thermal_name(self):
        """
        The name of the thermal column in the output file.
        """
        if self.params.psg.use_molecular_signatures:
            return self.params.planet.name.replace(' ', '-')
        else:
            return 'Thermal'

    def _get_thermal_spectrum(
        self,
        start_time: float,
        end_time: float,
        _interp_thermal: GridSpectra,
        pl_frac: float
    )->u.Quantity:
        """
        Get the thermal emission spectra calculated by PSG

        Parameters
        ----------
        start_time : astropy.units.Quantity
            The start time of the integration
        end_time : astropy.units.Quantity
            The end time of the integration
        _interp_thermal : GridSpectra
            The thermal spectra interpolator object.
        pl_frac : float
            The fraction of the planet's surface that is visible

        Returns
        -------
        wavelength : astropy.units.Quantity [wavelength]
            The wavelength of the thermal emission.
        flux : astropy.units.Quantity [flambda]
            The flux of the thermal emission.

        Raises
        ------
        ValueError
            If the PSG flux unit code is not recognized.
        ValueError
            If the wavelength coordinates from the loaded spectra do not match.
        """
        
        spectra = _interp_thermal.evaluate((np.array([start_time,end_time]),),self._wl).mean(axis=0)
        return spectra * pl_frac * config.flux_unit

    def _get_layer_data(self, N1: int, N2: int, N1_frac: float) -> libpypsg.PyLyr:
        """
        Interpolate between two PSG .lyr files.

        Parameters
        ----------
        N1 : int
            The planet index immediately before the current epoch.
        N2 : int
            The planet index immediately after the current epoch.
        N1_frac : float
            The fraction of the `N1` epoch to use in interpolation.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the interpolated layer data.

        Raises
        ------
        ValueError
            If the layer file columns of layer numbers do not match.
        """
        psg_layers_path1 = Path(
            self.directories['psg_layers']) / get_filename(N1, N_ZFILL, 'fits')
        psg_layers_path2 = Path(
            self.directories['psg_layers']) / get_filename(N2, N_ZFILL, 'fits')
        layers1 = libpypsg.PyLyr.from_fits(psg_layers_path1)
        layers2 = libpypsg.PyLyr.from_fits(psg_layers_path2)

        if not np.all(layers1.prof.colnames == layers2.prof.colnames):
            raise ValueError(
                'Layer Profiles table files must have matching columns')
        if not np.all(layers1.cg.colnames == layers2.cg.colnames):
            raise ValueError(
                'Layer CG table files must have matching columns')

        prof_names = layers1.prof.colnames
        cg_names = layers1.cg.colnames

        prof_data = {}
        cg_data = {}
        for name in prof_names:
            prof_data[name] = layers1.prof[name] * \
                N1_frac + layers2.prof[name] * (1-N1_frac)
        for name in cg_names:
            cg_data[name] = layers1.cg[name] * \
                N1_frac + layers2.cg[name] * (1-N1_frac)

        return libpypsg.PyLyr(
            prof=QTable(data=prof_data, meta=layers1.prof.meta),
            cg=QTable(data=cg_data, meta=layers1.cg.meta)
        )

    def build_spectra(self):
        """
        Integrate our stellar model with PSG to produce a variable
        host + planet simulation.
        Follow the original Build_Spectra.py file to construct phase curve outputs.
        
        Examples
        --------
        >>> model = ObservationModel.from_yaml(CFG_PATH) # build_planet() has been run previously
        >>> model.build_spectra()
        Build Spectra: 100%|██████████| 36/36 [00:23<00:00,  1.53it/s]

        """
        if self.star is None:  # user can define a custom star before calling this function, e.g. for a specific spot pattern
            self._build_star()
            self._warm_up_star(spot_warmup_time=self.params.star.spots.burn_in,
                              facula_warmup_time=self.params.star.faculae.burn_in)
        observation_parameters = self.get_observation_parameters()
        observation_info = self._get_observation_plan(
            observation_parameters, planet=False)
        # write observation info to file
        obs_info_filename = Path(
            self.directories['all_model']) / 'observation_info.fits'
        observation_info.write(obs_info_filename, overwrite=True)

        planet_observation_info = self._get_observation_plan(
            observation_parameters, planet=True)
        planet_times = planet_observation_info['time']
        time_step = self.params.obs.integration_time
        planet_time_step = self.params.obs.observation_time / \
            (self.params.planet_total_images+1)
        granulation_fractions = self.star.get_granulation_coverage(
            observation_info['time'])
        print('Creating interpolators:', end='\n')
        s = 'thermal'
        print(s,end='\r')
        interp_thermal = self._get_thermal_interpolator()
        s+=', combined'
        print(s, end='\r')
        interp_reflected = self._get_reflected_interpolator()
        s+=', stellar'
        print(s, end='\r')
        interp_stellar = self._get_psg_interp('combined', 'Stellar')
        s+=', photon noise'
        print(s, end='\r')
        interp_noise_photon = self._get_psg_interp('noise', 'Source')
        s+=', detector noise'
        print(s, end='\r')
        interp_noise_detector = self._get_psg_interp('noise', 'Detector')
        s+=', telescope noise'
        print(s, end='\r')
        interp_noise_telescope = self._get_psg_interp('noise', 'Telescope')
        s+=', background noise'
        print(s, end='\r')
        interp_noise_background = self._get_psg_interp('noise', 'Background')
        s+=', transit'
        print(s)
        interp_transit = self._get_transit_interpolator()
        print('Finished!')

        for index in self._wrap_iterator(range(self.params.obs.total_images), desc='Build Spectra', total=self.params.obs.total_images, position=0, leave=True):
            tindex = observation_info['time'][index]
            tstart = tindex - observation_info['time'][0]
            tfinish = tstart + time_step
            planet_phase = observation_info['phase'][index]
            sub_obs_lon = observation_info['sub_obs_lon'][index]
            sub_obs_lat = observation_info['sub_obs_lat'][index]
            orbital_radius = observation_info['orbit_radius'][index] * \
                self.params.planet.semimajor_axis
            granulation_fraction = granulation_fractions[index]
            N1, N2 = get_planet_indicies(planet_times, tindex)
            N1_frac = (planet_times[N2] - tindex)/planet_time_step
            N1_frac = N1_frac.to_value(u.dimensionless_unscaled)

            sub_planet_lon = observation_info['sub_planet_lon'][index]
            sub_planet_lat = observation_info['sub_planet_lat'][index]
            
            transit_depth = self._get_transit(
                start_time=tstart.to_value(config.time_unit),
                end_time=tfinish.to_value(config.time_unit),
                transit_interpolator=interp_transit,
            )

            comp_flux, pl_frac = self._calculate_composite_stellar_spectrum(
                {'lat': sub_obs_lat, 'lon': sub_obs_lon}, tstart, tfinish,
                granulation_fraction=granulation_fraction,
                orbit_radius=orbital_radius,
                planet_radius=self.params.planet.radius,
                phase=planet_phase,
                inclination=self.params.system.inclination,
                transit_depth=transit_depth
            )
            true_star, _ = self._calculate_composite_stellar_spectrum(
                {'lat': sub_obs_lat, 'lon': sub_obs_lon}, tstart, tfinish,
                granulation_fraction=granulation_fraction,
                orbit_radius=orbital_radius,
                planet_radius=self.params.planet.radius,
                phase=planet_phase,
                inclination=self.params.system.inclination,
                transit_depth=0.
            )
            to_planet_flux, _ = self._calculate_composite_stellar_spectrum(
                {'lat': sub_planet_lat, 'lon': sub_planet_lon}, tstart, tfinish,
                granulation_fraction=granulation_fraction
            )

            reflection_flux_adj = self._calculate_reflected_spectra(
                start_time=tstart.to_value(config.time_unit),
                end_time=tfinish.to_value(config.time_unit),
                reflected_interpolator=interp_reflected,
                stellar_interpolator=interp_stellar,
                sub_planet_flux=to_planet_flux,
                pl_frac=pl_frac
            )
            thermal_spectrum = self._get_thermal_spectrum(
                start_time=tstart.to_value(config.time_unit),
                end_time=tfinish.to_value(config.time_unit),
                _interp_thermal=interp_thermal,
                pl_frac=pl_frac
            )

            combined_flux: u.Quantity = comp_flux + reflection_flux_adj + thermal_spectrum
            
            noise_flux_adj = self._calculate_noise(
                start_time=tstart.to_value(config.time_unit),
                end_time=tfinish.to_value(config.time_unit),
                photon_noise_interpolator=interp_noise_photon,
                detector_noise_interpolator=interp_noise_detector,
                telescope_noise_interpolator=interp_noise_telescope,
                background_noise_interpolator=interp_noise_background,
                star_interpolator=interp_stellar,
                time_scale_factor=np.sqrt((planet_time_step/time_step).to_value(u.dimensionless_unscaled)),
                cmb_flux=combined_flux
            )

            wl = self._wl
            data: Dict[str, u.Quantity] = {}
            data['wavelength'] = wl
            data['star'] = true_star
            data['star_towards_planet'] = to_planet_flux
            data['reflected'] = reflection_flux_adj
            data['planet_thermal'] = thermal_spectrum
            data['total'] = combined_flux
            data['noise'] = noise_flux_adj*config.flux_unit

            df = QTable(data)

            outfile = self.directories['all_model'] / \
                get_filename(index, N_ZFILL, 'fits')
            df.write(outfile, overwrite=True)

            # layers
            if self.params.psg.use_molecular_signatures:
                layerdat = self._get_layer_data(N1, N2, N1_frac)
                outfile = self.directories['all_model'] / \
                    f'layer{str(index).zfill(N_ZFILL)}.fits'
                layerdat.to_fits(outfile)

            self.star.birth_spots(time_step)
            self.star.birth_faculae(time_step)
            self.star.age(time_step)
