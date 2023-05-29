"""VSPEC main module

This module performs all of VSPEC's interaction with the user.
It contains the `ObservationModel` class, which has methods that
perform all of the model aggregation from the rest of the package
and PSG.
"""

from pathlib import Path
import typing

import numpy as np
import pandas as pd
from astropy import units as u
from tqdm.auto import tqdm
import warnings
from functools import partial

from VSPEC import stellar_spectra
from VSPEC import variable_star_model as vsm
from VSPEC.variable_star_model import granules
from VSPEC.config import PSG_CFG_MAX_LINES
from VSPEC.files import build_directories, N_ZFILL, get_filename
from VSPEC.geometry import SystemGeometry
from VSPEC.helpers import isclose, to_float, is_port_in_use, arrange_teff, get_surrounding_teffs
from VSPEC.helpers import plan_to_df, get_planet_indicies
from VSPEC.psg_api import call_api, PSGrad, get_reflected, cfg_to_bytes
from VSPEC.psg_api import change_psg_parameters, parse_full_output, cfg_to_dict
from VSPEC.analysis import read_lyr
from VSPEC.params.read import Parameters


class ObservationModel:
    """
    Main class that stores the information of this simulation.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path of the configuration file.
    verbose : int, default=1
        The verbosity level of the output.

    Attributes
    ----------
    verbose : int
        The verbosity level of the output.
    params : `VSPEC.read_info.ParamModel`
        The parameters for this simulation.
    dirs : dict
        The paths to model output directories.
    star : `VSPEC.variable_star_model.Star`
        The variable host star.
    rng : numpy.random.Generator
        A psudo-random number generator to be used in
        the simulation.
    """

    def __init__(self, config_path: Path, verbose=1):
        self.verbose = verbose
        self.params = Parameters.from_yaml(config_path)
        self.build_directories()
        self.star = None
        self.rng = np.random.default_rng(self.params.header.seed)

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

    def wrap_iterator(self, iterator, **kwargs):
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

    def build_directories(self):
        """
        Build the file system for this run.
        """
        self.dirs = build_directories(self.params.header.data_path)

    def bin_spectra(self):
        """
        Bins high-resolution spectra to required resolution.

        This method loads high-resolution spectra and bins them to the required resolution. The binned spectra are then
        written to a local file (`self.dirs['binned']/...`).
        """
        teffs = arrange_teff(
            self.params.header.teff_min,
            self.params.header.teff_max
        )
        for teff in self.wrap_iterator(teffs, desc='Binning Spectra', total=len(teffs)):
            def bin_spectrum(func):
                """
                func is a binning function.
                This way I don't have to type the same arguments twice.
                """
                func(
                    to_float(teff, u.K),
                    file_name_writer=stellar_spectra.get_binned_filename,
                    binned_path=self.dirs['binned'],
                    resolving_power=self.params.inst.bandpass.resolving_power,
                    lam1=self.params.inst.bandpass.wl_blue,
                    lam2=self.params.inst.bandpass.wl_red,
                    model_unit_wavelength=u.AA,
                    model_unit_flux=u.Unit(
                        'erg s-1 cm-2 cm-1'),
                    target_unit_wavelength=self.params.inst.bandpass.wavelength_unit,
                    target_unit_flux=self.params.inst.bandpass.flux_unit
                )
            try:
                bin_spectrum(stellar_spectra.bin_cached_model)
            except ValueError:
                bin_spectrum(stellar_spectra.bin_phoenix_model)

    def read_spectrum(self, teff: u.Quantity) -> typing.Tuple[u.Quantity, u.Quantity]:
        """
        Read a binned spectrum from file.

        Parameters
        ----------
        teff : astropy.units.Quantity [temperature]
            The effective temperature of the spectrum to read.

        Returns
        -------
        wavelengths : astropy.units.Quantity [wavelength]
            The binned wavelengths of the spectrum.
        flux : astropy.units.Quantity [flambda]
            The binned flux of the spectrum.
        """
        filename = stellar_spectra.get_binned_filename(to_float(teff, u.K))
        path = self.dirs['binned']
        return stellar_spectra.read_binned_spectrum(filename, path=path)

    def get_model_spectrum(self, Teff):
        """
        Interpolate between binned spectra to produce a model spectrum with a given Teff.

        Parameters
        ----------
        Teff : astropy.units.Quantity [temperature]
            The desired effective temperature of the spectrum

        Returns
        -------
        wavelengths : astropy.units.Quantity [wavelength]
            The wavelength coordinates of the spectrum.
        flux : astropy.units.Quantity [flambda]
            The flux of the spectrum, corrected for system distance.
        """
        if Teff == 0*u.K:  # for testing
            star_teff = self.params.star.teff
            wave1, flux1 = self.read_spectrum(
                star_teff - (star_teff % (100*u.K)))
            return wave1, flux1*0
        elif (Teff % (100*u.K) == 0*u.K):
            wave1, flux1 = self.read_spectrum(Teff)
            return wave1, flux1*self.params.flux_correction
        else:

            model_teffs = get_surrounding_teffs(Teff)
            if Teff in model_teffs:
                wave1, flux1 = self.read_spectrum(Teff)
                wave2, flux2 = wave1, flux1
            else:
                wave1, flux1 = self.read_spectrum(model_teffs[0])
                wave2, flux2 = self.read_spectrum(model_teffs[1])
            wavelength, flux = stellar_spectra.interpolate_spectra(Teff,
                                                                   model_teffs[0], wave1, flux1,
                                                                   model_teffs[1], wave2, flux2)
            return wavelength, flux*self.params.flux_correction

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
                              stellar_offset_amp=self.params.star.offset_magnitude,
                              stellar_offset_phase=self.params.star.offset_direction,
                              eccentricity=self.params.planet.eccentricity,
                              phase_of_periasteron=self.params.system.phase_of_periasteron,
                              system_distance=self.params.system.distance,
                              obliquity=self.params.planet.obliquity,
                              obliquity_direction=self.params.planet.obliquity_direction)

    def get_observation_plan(self, observation_parameters: SystemGeometry, planet=False):
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
        dict
            A dictionary of arrays describing the geometry at each
            epoch. Each dict value is an astropy.units.Quantity array.
        """
        if planet:
            N_obs = self.params.planet_total_images
        else:
            N_obs = self.params.star_total_images
        return observation_parameters.get_observation_plan(self.params.planet.init_phase,
                                                           self.params.obs.observation_time, N_obs=N_obs)

    def check_psg(self):
        """
        Check that PSG is running

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
        psg_url = self.params.psg.url
        if 'localhost' in psg_url:
            port = int(psg_url.split(':')[-1])
            if not is_port_in_use(port):
                raise RuntimeError('Local PSG is specified, but is not running.\n' +
                                   'Type `docker start psg` in the command line.')
        elif self.params.psg.api_key.value is None:
            msg = 'PSG is being called without an API key. '
            msg += 'After 100 API calls in a 24hr period you will need to get a key. '
            msg += 'We suggest installing PSG locally using docker. (see https://psg.gsfc.nasa.gov/help.php#handbook)'
            warnings.warn(msg, RuntimeWarning)

    def upload_gcm(self, obstime: u.Quantity = 0*u.s, update=False):
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
        if self.params.gcm.gcmtype == 'binary':
            kwargs = {}
        elif self.params.gcm.gcmtype == 'waccm':
            kwargs = {'obs_time': obstime}
        content = self.params.gcm.content(**kwargs)
        call_api(
            config_path=None,
            psg_url=self.params.psg.url,
            api_key=self.params.psg.api_key.value,
            output_type='upd' if update else 'set',
            app='globes',
            outfile=None,
            config_data=content
        )
        if not update:
            self.__flags__.psg_needs_set = False

    def set_static_config(self):
        """
        Upload the non-changing parts of the PSG config.
        """
        params = self.params.to_psg()
        content = cfg_to_bytes(params)
        call_api(
            config_path=None,
            psg_url=self.params.psg.url,
            api_key=self.params.psg.api_key.value,
            output_type='upd',
            app='globes',
            outfile=None,
            config_data=content
        )

    def update_config(
        self,
        phase: u.Quantity,
        orbit_radius_coeff: float,
        sub_stellar_lon: u.Quantity,
        sub_stellar_lat: u.Quantity,
        pl_sub_obs_lon: u.Quantity,
        pl_sub_obs_lat: u.Quantity,
        include_star: bool
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
        """
        params = change_psg_parameters(
            params=self.params,
            phase=phase,
            orbit_radius_coeff=orbit_radius_coeff,
            sub_stellar_lon=sub_stellar_lon,
            sub_stellar_lat=sub_stellar_lat,
            pl_sub_obs_lon=pl_sub_obs_lon,
            pl_sub_obs_lat=pl_sub_obs_lat,
            include_star=include_star
        )
        content = cfg_to_bytes(params)
        call_api(
            config_path=None,
            psg_url=self.params.psg.url,
            api_key=self.params.psg.api_key.value,
            output_type='upd',
            app='globes',
            outfile=None,
            config_data=content
        )

    def check_config(self, cfg_from_psg):
        """
        Validate the config file recieved from PSG to ensure that
        the parameters sent are the parameters used in the simulation.

        Parameters
        ----------
        cfg_from_psg : str
            A string containing the contents of the config file
            recieved from PSG

        Raises
        ------
        RuntimeError
            If the config recieved does not match the config sent.
        """
        n_lines = len(cfg_from_psg.split('\n'))
        if n_lines > PSG_CFG_MAX_LINES:
            self.__flags__.psg_needs_set = True
        cfg_dict = cfg_to_dict(cfg_from_psg)
        expected_cfg = self.params.to_psg()
        msg = ''
        for key, value in expected_cfg.items():
            try:
                assert cfg_dict[key] == value, f'key:{key} -- {cfg_dict[key]} != {value}'
            except KeyError as err:
                msg += f'Expected key not in config:{str(err)}\n'
            except AssertionError as err:
                if key == 'OBJECT-INCLINATION':
                    if np.sin(float(value)*u.deg) == np.sin(float(cfg_dict[key])*u.deg):
                        pass
                    else:
                        msg += f'{str(err)}\n'
                else:
                    msg += f'{str(err)}\n'
        if not msg == '':
            raise RuntimeError(f'PSG config validation error:\n{msg}')

    def run_psg(self, path_dict: dict, i: int):
        """
        Run PSG

        Parameters
        ----------
        path_dict : dict
            A dictionary that determines where each downloaded file
            gets written to.
        """
        content = bytes(
            f'<OBJECT-NAME>{self.params.planet.name}', encoding='UTF-8')
        response = call_api(
            config_path=None,
            psg_url=self.params.psg.url,
            api_key=self.params.psg.api_key.value,
            output_type='all',
            app='globes',
            outfile=None,
            config_data=content
        )
        output_data = parse_full_output(response)
        for key, path in path_dict.items():
            filename = get_filename(i, N_ZFILL, key)
            with open(path/filename, 'wt', encoding='UTF-8') as file:
                if not (key == 'lyr' and self.params.psg.use_molecular_signatures is False):
                    file.write(output_data[key])
                if key == 'cfg':
                    self.check_config(output_data[key])

    def build_planet(self):
        """
        Use the PSG GlobES API to construct a planetary phase curve.
        Follow steps in original PlanetBuilder.py file

        """
        # check that psg is running
        self.check_psg()
        # for not using globes, append all configurations instead of rewritting

        ####################################
        # Initial upload of GCM

        self.upload_gcm(
            obstime=0*u.s,
            update=False
        )
        ####################################
        # Set observation parameters that do not change
        self.set_static_config()

        ####################################
        # Calculate observation parameters
        observation_parameters = self.get_observation_parameters()
        obs_plan = self.get_observation_plan(
            observation_parameters, planet=True)

        obs_info_filename = Path(self.dirs['data']) / 'observation_info.csv'
        plan_to_df(obs_plan).to_csv(obs_info_filename, sep=',', index=False)

        if self.verbose > 0:
            print(
                f'Starting at phase {self.params.planet.init_phase}, observe for {self.params.obs.observation_time} in {self.params.planet_total_images} steps')
            print('Phases = ' +
                  str(np.round(np.asarray((obs_plan['phase']/u.deg).to(u.Unit(''))), 2)) + ' deg')
        ####################################
        # iterate through phases
        for i in self.wrap_iterator(range(self.params.planet_total_images), desc='Build Planet', total=self.params.planet_total_images):
            phase = obs_plan['phase'][i]
            sub_stellar_lon = obs_plan['sub_stellar_lon'][i]
            sub_stellar_lat = obs_plan['sub_stellar_lat'][i]
            pl_sub_obs_lon = obs_plan['planet_sub_obs_lon'][i]
            pl_sub_obs_lat = obs_plan['planet_sub_obs_lat'][i]
            orbit_radius_coeff = obs_plan['orbit_radius'][i]
            obs_time = obs_plan['time'][i] - obs_plan['time'][0]

            if (not self.params.gcm.is_staic) or self.__flags__.psg_needs_set:
                # enter if we need a reset or if there is time dependence
                upload = partial(self.upload_gcm,obstime=obs_time)
                if self.__flags__.psg_needs_set: # do a reset if needed
                    upload(update=False)
                    self.set_static_config()
                    self.__flags__.psg_needs_set = False
                else: # update if it's just time dependence.
                    upload(update=True)
            
            # Write updates to the config to change the phase value and ensure the star is of type 'StarType'
            update_config = partial(
                self.update_config,
                phase=phase,
                orbit_radius_coeff=orbit_radius_coeff,
                sub_stellar_lon=sub_stellar_lon,
                sub_stellar_lat=sub_stellar_lat,
                pl_sub_obs_lon=pl_sub_obs_lon,
                pl_sub_obs_lat=pl_sub_obs_lat
            )
            update_config(include_star=True)
            path_dict = {
                'rad': Path(self.dirs['psg_combined']),
                'noi': Path(self.dirs['psg_noise']),
                'cfg': Path(self.dirs['psg_configs'])
            }
            self.run_psg(path_dict, i)
            # write updates to config file to remove star flux
            update_config(include_star=False)
            path_dict = {
                'rad': Path(self.dirs['psg_thermal']),
                'lyr': Path(self.dirs['psg_layers'])
            }
            self.run_psg(path_dict, i)

    def build_star(self):
        """
        Build a variable star model based on user-specified parameters.
        """
        empty_spot_collection = vsm.SpotCollection(Nlat=self.params.star.Nlat,
                                                   Nlon=self.params.star.Nlon)
        empty_fac_collection = vsm.FaculaCollection(Nlat=self.params.star.Nlat,
                                                    Nlon=self.params.star.Nlon)
        flare_generator = vsm.FlareGenerator(
            self.params.star.teff, self.params.star.period, self.params.star.flares.group_probability,
            self.params.star.flares.teff_mean, self.params.star.flares.teff_sigma,
            np.log10(self.params.star.flares.fwhm_mean /
                     u.day), self.params.star.flares.fwhm_sigma,
            log_E_erg_max=np.log10(
                self.params.star.flares.E_max.to_value(u.erg)),
            log_E_erg_min=np.log10(
                self.params.star.flares.E_min.to_value(u.erg)),
            log_E_erg_Nsteps=self.params.star.flares.E_steps
        )

        spot_generator = vsm.SpotGenerator(
            self.params.star.spots.area_mean, self.params.star.spots.area_logsigma,
            self.params.star.spots.teff_umbra, self.params.star.spots.teff_penumbra,
            self.params.star.spots.growth_rate, self.params.star.spots.decay_rate,
            self.params.star.spots.initial_area, self.params.star.spots.distribution,
            self.params.star.spots.equillibrium_coverage,
            Nlat=self.params.star.Nlat, Nlon=self.params.star.Nlon
        )
        fac_generator = vsm.FaculaGenerator(
            R_peak=self.params.star.faculae.mean_radius, R_HWHM=self.params.star.faculae.hwhm_radius,
            T_peak=self.params.star.faculae.mean_timescale, T_HWHM=self.params.star.faculae.hwhm_timescale,
            coverage=self.params.star.faculae.equillibrium_coverage, dist=self.params.star.faculae.distribution,
            Nlat=self.params.star.Nlat, Nlon=self.params.star.Nlon
        )
        granulation = granules.Granulation(
            self.params.star.granulation.mean,
            self.params.star.granulation.amp,
            self.params.star.granulation.period,
            self.params.star.granulation.dteff
        )
        self.star = vsm.Star(self.params.star.teff, self.params.star.radius,
                             self.params.star.period, empty_spot_collection, empty_fac_collection,
                             distance=self.params.system.distance,
                             Nlat=self.params.star.Nlat, Nlon=self.params.star.Nlon, flare_generator=flare_generator,
                             spot_generator=spot_generator, fac_generator=fac_generator,
                             granulation=granulation, u1=self.params.star.ld.u1, u2=self.params.star.ld.u2)

    def warm_up_star(self, spot_warmup_time: u.Quantity[u.day] = 0*u.day, facula_warmup_time: u.Quantity[u.day] = 0*u.day):
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
            for _ in self.wrap_iterator(range(N_steps_spot), desc='Spot Warmup', total=N_steps_spot):
                self.star.birth_spots(spot_warm_up_step)
                self.star.age(spot_warm_up_step)
        if N_steps_facula > 0:
            for _ in self.wrap_iterator(range(N_steps_facula), desc='Facula Warmup', total=N_steps_facula):
                self.star.birth_faculae(facula_warm_up_step)
                self.star.age(facula_warm_up_step)

        self.star.get_flares_over_observation(
            self.params.obs.observation_time)

    def calculate_composite_stellar_spectrum(
        self,
        sub_obs_coords,
        tstart,
        tfinish,
        granulation_fraction:float=0.0,
        orbit_radius:u.Quantity = 1*u.AU,
        planet_radius:u.Quantity = 1*u.R_earth,
        phase:u.Quantity = 90*u.deg,
        inclination:u.Quantity = 0*u.deg,
        transit_depth:np.ndarray or float = 0
    ):
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
        total,covered,pl_frac = self.star.calc_coverage(
            sub_obs_coords,
            granulation_fraction=granulation_fraction,
            orbit_radius = orbit_radius,
            planet_radius = planet_radius,
            phase = phase,
            inclination = inclination
        )
        visible_flares = self.star.get_flare_int_over_timeperiod(
            tstart, tfinish, sub_obs_coords)
        base_wave, base_flux = self.get_model_spectrum(self.params.star.teff)
        base_flux = base_flux * 0
        # add up star flux before considering transit
        for teff, coverage in total.items():
            if coverage > 0:
                wave, flux = self.get_model_spectrum(teff)
                if not np.all(isclose(base_wave, wave, 1e-3*u.um)):
                    raise ValueError('All wavelength axes must be equivalent.')
                base_flux = base_flux + flux * coverage
        # get flux of transited region
        transit_flux = base_flux*0
        for teff, coverage in covered.items():
            if coverage > 0:
                wave, flux = self.get_model_spectrum(teff)
                if not np.all(isclose(base_wave, wave, 1e-3*u.um)):
                    raise ValueError('All wavelength axes must be equivalent.')
                transit_flux = transit_flux + flux * coverage
        # scale according to effective radius
        transit_flux = transit_flux * transit_depth
        base_flux = base_flux - transit_flux
        # add in flares
        for flare in visible_flares:
            teff = flare['Teff']
            timearea = flare['timearea']
            eff_area = (timearea/(tfinish-tstart)).to(u.km**2)
            flux = stellar_spectra.blackbody(base_wave, teff, eff_area, self.params.system.distance,
                                             target_unit_flux=self.params.inst.bandpass.flux_unit)
            base_flux = base_flux + flux

        return base_wave, base_flux, pl_frac

    def calculate_reflected_spectra(self, N1, N2, N1_frac,
                                    sub_planet_wavelength, sub_planet_flux,pl_frac:float):
        """
        Calculate the reflected spectrum based on PSG output and
        our own stellar model. We scale the reflected spectra from PSG
        to our model.

        Parameters
        ----------
        N1 : int
            The planet index immediately before the current epoch.
        N2 : int
            The planet index immediately after the current epoch.
        N1_frac : float
            The fraction of the `N1` epoch to use in interpolation.
        sub_planet_wavelength : astropy.units.Quantity [wavelength]
            Wavelengths for validation.
        sub_planet_flux : astropy.units.Quantity [flambda]
            Stellar flux to scale to.
        pl_frac : float
            The fraction of the planet that is visible (in case of
            eclipse)

        Returns
        -------
        reflected_wavelength : astropy.units.Quantity [wavelength]
            Reflected wavelength.
        reflected_flux : astropy.units.Quantity [flambda]
            Reflected flux.

        Raises
        ------
        ValueError
            If the PSG flux unit code is not recognized.
        ValueError
            If the wavelength coordinates from the loaded spectra do not match.
        """
        psg_combined_path1 = Path(
            self.dirs['psg_combined']) / get_filename(N1, N_ZFILL, 'rad')
        psg_thermal_path1 = Path(
            self.dirs['psg_thermal']) / get_filename(N1, N_ZFILL, 'rad')
        psg_combined_path2 = Path(
            self.dirs['psg_combined']) / get_filename(N2, N_ZFILL, 'rad')
        psg_thermal_path2 = Path(
            self.dirs['psg_thermal']) / get_filename(N2, N_ZFILL, 'rad')

        reflected = []

        for psg_combined_path, psg_thermal_path in zip([psg_combined_path1, psg_combined_path2],
                                                       [psg_thermal_path1, psg_thermal_path2]):
            combined = PSGrad.from_rad(psg_combined_path)
            thermal = PSGrad.from_rad(psg_thermal_path)

            # validate
            if not np.all(isclose(sub_planet_wavelength, combined.data['Wave/freq'], 1e-3*u.um)
                          & isclose(sub_planet_wavelength, thermal.data['Wave/freq'], 1e-3*u.um)):
                raise ValueError(
                    'The wavelength coordinates must be equivalent.')
            planet_reflection_only = get_reflected(
                combined, thermal, self.params.planet.name)
            planet_reflection_fraction = to_float(
                planet_reflection_only / combined.data['Stellar'], u.dimensionless_unscaled)

            planet_reflection_adj = sub_planet_flux * planet_reflection_fraction
            reflected.append(planet_reflection_adj)

        return sub_planet_wavelength, reflected[0] * N1_frac + reflected[1] * (1-N1_frac)*pl_frac

    def get_transit(
        self,
        N1:int,
        N2:int,
        N1_frac:float,
        phase:u.Quantity,
        orbit_radius:u.Quantity
        ):
        """
        Get the transit spectra calculated by PSG

        Parameters
        ----------
        N1 : int
            The planet index immediately before the current epoch.
        N2 : int
            The planet index immediately after the current epoch.
        N1_frac : float
            The fraction of the `N1` epoch to use in interpolation.
        pl_frac : float
            The fraction of the planet that is visible (not eclipsed)

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
        psg_cmb_path1 = Path(
            self.dirs['psg_combined']) / get_filename(N1, N_ZFILL, 'rad')
        psg_cmb_path2 = Path(
            self.dirs['psg_combined']) / get_filename(N2, N_ZFILL, 'rad')

        wavelength = []
        transit = []

        for psg_cmb_path in [psg_cmb_path1, psg_cmb_path2]:
            cmb_rad = PSGrad.from_rad(psg_cmb_path)

            wavelength.append(cmb_rad.data['Wave/freq'])
            try:
                transit.append(
                    -1*cmb_rad.data['Transit']/cmb_rad.data['Stellar']
                )
            except KeyError:
                transit.append(
                    0*cmb_rad.data['Stellar']/cmb_rad.data['Stellar']
                )

        if not np.all(isclose(wavelength[0], wavelength[1], 1e-3*u.um)):
            raise ValueError('The wavelength coordinates must be equivalent.')
        frac_absorbed = transit[0]*N1_frac + transit[1]*(1-N1_frac)
        pl_frac_covering = 1-self.star.get_pl_frac(
            phase+180*u.deg, orbit_radius,self.params.planet.radius,self.params.system.inclination
        ) * (self.params.planet.radius/self.params.star.radius).to_value(u.dimensionless_unscaled)**2
        if pl_frac_covering == 0:
            normalized_frac_absorbed = pl_frac_covering*0
        else:
            normalized_frac_absorbed = frac_absorbed/pl_frac_covering
        return wavelength[0], normalized_frac_absorbed
    
    def calculate_noise(self, N1: int, N2: int, N1_frac: float, time_scale_factor: float, cmb_wavelength, cmb_flux):
        """
        Calculate the noise in our model based on the noise output from PSG.

        Parameters
        ----------
        N1 : int
            The planet index immediately before the current epoch.
        N2 : int
            The planet index immediately after the current epoch.
        N1_frac : float
            The fraction of the `N1` epoch to use in interpolation.
        time_scale_factor : float
            A scaling factor to apply to the noise at the end of the calculation.
            This is 1 if the planet and star sampling has the same cadence. Otherwise,
            it is usually `sqrt(self.planet_phase_binning)`.
        cmb_wavelength : astropy.units.Quantity [wavelength]
            The wavelength of the combined spectra.
        cmb_flux : astropy.units.Quantity [flambda]
            The flux of the combined spectrum.

        Returns
        -------
        cmb_wavelength : astropy.units.Quantity [wavelength]
            The wavelength of the combined spectra
        noise : astropy.units.Quantity [flambda]
            The noise in our model.

        Raises
        ------
        ValueError
            If the PSG flux unit code is not recognized.
        ValueError
            If the wavelength coordinates from the loaded spectra do not match.
        """
        psg_combined_path1 = Path(
            self.dirs['psg_combined']) / get_filename(N1, N_ZFILL, 'rad')
        psg_noise_path1 = Path(
            self.dirs['psg_noise']) / get_filename(N1, N_ZFILL, 'noi')
        psg_combined_path2 = Path(
            self.dirs['psg_combined']) / get_filename(N2, N_ZFILL, 'rad')
        psg_noise_path2 = Path(
            self.dirs['psg_noise']) / get_filename(N2, N_ZFILL, 'noi')

        psg_noise_source = []
        psg_source = []

        for psg_combined_path, psg_noise_path in zip(
            [psg_combined_path1, psg_combined_path2],
            [psg_noise_path1, psg_noise_path2]
        ):
            combined = PSGrad.from_rad(psg_combined_path)
            noise = PSGrad.from_rad(psg_noise_path)

            # validate
            if not np.all(isclose(cmb_wavelength, combined.data['Wave/freq'], 1e-3*u.um)
                          & isclose(cmb_wavelength, noise.data['Wave/freq'], 1e-3*u.um)):
                raise ValueError(
                    'The wavelength coordinates must be equivalent.')
            psg_noise_source.append(noise.data['Source'])
            psg_source.append(combined.data['Total'])
        psg_noise_source = psg_noise_source[0] * \
            N1_frac + psg_noise_source[1] * (1-N1_frac)
        psg_source = psg_source[0]*N1_frac + psg_source[1] * (1-N1_frac)

        model_noise = psg_noise_source * np.sqrt(cmb_flux/psg_source)
        noise_sq = (model_noise**2
                    + (noise.data['Detector'])**2
                    + (noise.data['Telescope'])**2
                    + (noise.data['Background'])**2)
        return cmb_wavelength, np.sqrt(noise_sq) * time_scale_factor

    def get_thermal_spectrum(self, N1: int, N2: int, N1_frac: float,pl_frac:float):
        """
        Get the thermal emission spectra calculated by PSG

        Parameters
        ----------
        N1 : int
            The planet index immediately before the current epoch.
        N2 : int
            The planet index immediately after the current epoch.
        N1_frac : float
            The fraction of the `N1` epoch to use in interpolation.
        pl_frac : float
            The fraction of the planet that is visible (not eclipsed)

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
        psg_thermal_path1 = Path(
            self.dirs['psg_thermal']) / get_filename(N1, N_ZFILL, 'rad')
        psg_thermal_path2 = Path(
            self.dirs['psg_thermal']) / get_filename(N2, N_ZFILL, 'rad')

        wavelength = []
        thermal = []

        for psg_thermal_path in [psg_thermal_path1, psg_thermal_path2]:
            thermal_rad = PSGrad.from_rad(psg_thermal_path)

            wavelength.append(thermal_rad.data['Wave/freq'])
            try:
                thermal.append(thermal_rad.data[self.params.planet.name])
            except KeyError:
                thermal.append(thermal_rad.data['Thermal'])

        if not np.all(isclose(wavelength[0], wavelength[1], 1e-3*u.um)):
            raise ValueError('The wavelength coordinates must be equivalent.')

        return wavelength[0], thermal[0]*N1_frac + thermal[1]*(1-N1_frac)*pl_frac

    def get_layer_data(self, N1: int, N2: int, N1_frac: float) -> pd.DataFrame:
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
            self.dirs['psg_layers']) / get_filename(N1, N_ZFILL, 'lyr')
        psg_layers_path2 = Path(
            self.dirs['psg_layers']) / get_filename(N2, N_ZFILL, 'lyr')
        layers1 = read_lyr(psg_layers_path1)
        layers2 = read_lyr(psg_layers_path2)
        if not np.all(layers1.columns == layers2.columns) & (len(layers1) == len(layers2)):
            raise ValueError(
                'Layer files must have matching columns and number of layers')
        cols = layers1.columns
        dat = layers1.values * N1_frac + layers2.values * (1-N1_frac)
        df = pd.DataFrame(columns=cols, data=dat)
        return df

    def build_spectra(self):
        """
        Integrate our stellar model with PSG to produce a variable
        host + planet simulation.
        Follow the original Build_Spectra.py file to construct phase curve outputs.
        """
        if self.star is None:  # user can define a custom star before calling this function, e.g. for a specific spot pattern
            self.build_star()
            self.warm_up_star(spot_warmup_time=self.params.star.spots.warmup,
                              facula_warmup_time=self.params.star.faculae.warmup)
        observation_parameters = self.get_observation_parameters()
        observation_info = self.get_observation_plan(
            observation_parameters, planet=False)
        # write observation info to file
        obs_info_filename = Path(
            self.dirs['all_model']) / 'observation_info.csv'
        obs_df = plan_to_df(observation_info)
        obs_df.to_csv(obs_info_filename, sep=',', index=False)

        planet_observation_info = self.get_observation_plan(
            observation_parameters, planet=True)
        planet_times = planet_observation_info['time']

        time_step = self.params.obs.integration_time
        planet_time_step = self.params.obs.integration_time * self.params.psg.phase_binning
        granulation_fractions = self.star.get_granulation_coverage(
            observation_info['time'])

        for index in self.wrap_iterator(range(self.params.obs.total_images), desc='Build Spectra', total=self.params.obs.total_images, position=0, leave=True):
            tindex = observation_info['time'][index]
            tstart = tindex - observation_info['time'][0]
            tfinish = tstart + time_step
            planet_phase = observation_info['phase'][index]
            sub_obs_lon = observation_info['sub_obs_lon'][index]
            sub_obs_lat = observation_info['sub_obs_lat'][index]
            orbital_radius = observation_info['orbit_radius'][index] * self.params.planet.semimajor_axis
            granulation_fraction = granulation_fractions[index]
            N1, N2 = get_planet_indicies(planet_times, tindex)
            N1_frac = (planet_times[N2] - planet_times[index])/planet_time_step
            N1_frac = N1_frac.to_value(u.dimensionless_unscaled)
            # N1_frac = 1 - \
            #     to_float(
            #         (tindex - planet_times[N1])/planet_time_step, u.Unit(''))

            sub_planet_lon = observation_info['sub_planet_lon'][index]
            sub_planet_lat = observation_info['sub_planet_lat'][index]
            

            wave, transit_depth = self.get_transit(N1,N2,N1_frac,planet_phase,orbital_radius)

            comp_wave, comp_flux, pl_frac = self.calculate_composite_stellar_spectrum(
                {'lat': sub_obs_lat,'lon': sub_obs_lon}, tstart, tfinish,
                granulation_fraction=granulation_fraction,
                orbit_radius=orbital_radius,
                planet_radius = self.params.planet.radius,
                phase = planet_phase,
                inclination = self.params.system.inclination,
                transit_depth=transit_depth
            )
            wave, true_star, _ = self.calculate_composite_stellar_spectrum(
                {'lat': sub_obs_lat,'lon': sub_obs_lon}, tstart, tfinish,
                granulation_fraction=granulation_fraction,
                orbit_radius=orbital_radius,
                planet_radius = self.params.planet.radius,
                phase = planet_phase,
                inclination = self.params.system.inclination,
                transit_depth=0.
            )
            wave, to_planet_flux,_ = self.calculate_composite_stellar_spectrum(
                {'lat': sub_planet_lat,'lon': sub_planet_lon}, tstart, tfinish,
                granulation_fraction=granulation_fraction
            )
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            wave, reflection_flux_adj = self.calculate_reflected_spectra(
                N1, N2, N1_frac, comp_wave, to_planet_flux,pl_frac)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            wave, thermal_spectrum = self.get_thermal_spectrum(N1, N2, N1_frac,pl_frac)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            combined_flux = comp_flux + reflection_flux_adj + thermal_spectrum

            wave, noise_flux_adj = self.calculate_noise(N1, N2, N1_frac,
                                                        np.sqrt(
                                                            to_float(planet_time_step/time_step, u.Unit(''))),
                                                        comp_wave, combined_flux)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            df = pd.DataFrame({
                f'wavelength[{str(comp_wave.unit)}]': comp_wave.value,
                f'star[{str(true_star.unit)}]': true_star.value,
                f'star_towards_planet[{str(to_planet_flux.unit)}]': to_planet_flux.value,
                f'reflected[{str(reflection_flux_adj.unit)}]': reflection_flux_adj.value,
                f'planet_thermal[{str(thermal_spectrum.unit)}]': thermal_spectrum.value,
                f'total[{str(combined_flux.unit)}]': combined_flux.value,
                f'noise[{str(noise_flux_adj.unit)}]': noise_flux_adj.value
            })
            outfile = Path(self.dirs['all_model']) / \
                get_filename(index, N_ZFILL, 'csv')
            df.to_csv(outfile, index=False, sep=',')

            # layers
            if self.params.psg.use_molecular_signatures:
                layerdat = self.get_layer_data(N1, N2, N1_frac)
                outfile = Path(self.dirs['all_model']) / \
                    f'layer{str(index).zfill(N_ZFILL)}.csv'
                layerdat.to_csv(outfile, index=False, sep=',')

            self.star.birth_spots(time_step)
            self.star.birth_faculae(time_step)
            self.star.age(time_step)
