"""VSPEC main module

This module performs all of VSPEC's interaction with the user.
It contains the `ObservationModel` class, which has methods that
perform all of the model aggregation from the rest of the package
and PSG.
"""

from pathlib import Path
from os import system
import typing

import numpy as np
import pandas as pd
from astropy import units as u
from tqdm.auto import tqdm
import warnings

from VSPEC import stellar_spectra
from VSPEC import variable_star_model as vsm
from VSPEC.files import build_directories, N_ZFILL, get_filename
from VSPEC.geometry import SystemGeometry
from VSPEC.helpers import isclose, to_float, is_port_in_use, arrange_teff, get_surrounding_teffs
from VSPEC.psg_api import call_api, write_static_config, PSGrad, get_reflected
from VSPEC.read_info import ParamModel
from VSPEC.analysis import read_lyr


class ObservationModel:
    """
    Main class that stores the information of this simulation.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path of the configuration file.
    debug : bool, default=False
        Whether to enter debug mode.

    Attributes
    ----------
    debug : bool, default=False
        Whether to enter debug mode.
    params : `VSPEC.read_info.ParamModel`
        The parameters for this simulation.
    dirs : dict
        The paths to model output directories.
    star : `VSPEC.variable_star_model.Star`
        The variable host star.
    """

    def __init__(self, config_path, debug=False):
        self.debug = debug
        self.params = ParamModel(config_path)
        self.build_directories()
        self.star = None

    def build_directories(self):
        """
        Build the file system for this run.
        """
        self.dirs = build_directories(self.params.star_name)

    def bin_spectra(self):
        """
        Bins high-resolution spectra to required resolution.

        This method loads high-resolution spectra and bins them to the required resolution. The binned spectra are then
        written to a local file (`self.dirs['binned']/...`).
        """
        teffs = arrange_teff(self.params.star_teff_min,self.params.star_teff_max)
        for teff in tqdm(teffs, desc='Binning Spectra', total=len(teffs)):
            stellar_spectra.bin_phoenix_model(to_float(teff, u.K),
                                              file_name_writer=stellar_spectra.get_binned_filename,
                                              binned_path=self.dirs['binned'],
                                              resolving_power=self.params.resolving_power,
                                              lam1=self.params.lambda_min,
                                              lam2=self.params.lambda_max,
                                              model_unit_wavelength=u.AA,
                                              model_unit_flux=u.Unit('erg s-1 cm-2 cm-1'),
                                              target_unit_wavelength=self.params.target_wavelength_unit,
                                              target_unit_flux=self.params.target_flux_unit)

    def read_spectrum(self, teff: u.Quantity)->typing.Tuple[u.Quantity,u.Quantity]:
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
        if Teff == 0*u.K: # for testing
            star_teff = self.params.star_teff
            wave1, flux1 = self.read_spectrum(star_teff - (star_teff % (100*u.K)))
            return wave1, flux1*0
        elif (Teff % (100*u.K)==0*u.K):
            wave1, flux1 = self.read_spectrum(Teff)
            return wave1, flux1
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
            return wavelength, flux*self.params.distanceFluxCorrection

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
        return SystemGeometry(self.params.system_inclination,
                              0*u.deg,
                              self.params.planet_initial_phase,
                              self.params.star_rot_period,
                              self.params.planet_orbital_period,
                              self.params.planet_semimajor_axis,
                              self.params.planet_rotational_period,
                              self.params.planet_init_substellar_lon,
                              self.params.star_rot_offset_from_orbital_plane,
                              self.params.star_rot_offset_angle_from_pariapse,
                              self.params.planet_eccentricity,
                              self.params.system_phase_of_periasteron,
                              self.params.system_distance,
                              self.params.planet_obliquity,
                              self.params.planet_obliquity_direction)

    def get_observation_plan(self, observation_parameters: SystemGeometry):
        """
        Compute the locations and geometries of each object in this simulation.

        Parameters
        ----------
        observation_parameters : VSPEC.geometry.SystemGeometry
            An object containting the system geometry.

        Returns
        -------
        dict
            A dictionary of arrays describing the geometry at each
            epoch. Each dict value is an astropy.units.Quantity array.
        """
        return observation_parameters.get_observation_plan(self.params.planet_initial_phase,
                                                           self.params.total_observation_time, N_obs=self.params.total_images)

    def get_planet_observation_plan(self, observation_parameters: SystemGeometry):
        """
        Compute the locations and geometries of each object in this simulation.
        Bin in the phase dimension if planet phase binning is specified.

        Parameters
        ----------
        observation_parameters : VSPEC.geometry.SystemGeometry
            An object containting the system geometry.

        Returns
        -------
        dict
            A dictionary of arrays describing the geometry at each
            epoch. Each dict value is an astropy.units.Quantity array.
        """
        return observation_parameters.get_observation_plan(self.params.planet_initial_phase,
                                                           self.params.total_observation_time, N_obs=self.params.planet_images)

    def build_planet(self):
        """
        Use the PSG GlobES API to construct a planetary phase curve.
        Follow steps in original PlanetBuilder.py file

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

        # check that psg is running
        psg_url = self.params.psg_url
        if 'localhost' in psg_url:
            port = int(psg_url.split(':')[-1])
            if not is_port_in_use(port):
                raise RuntimeError('Local PSG is specified, but is not running.\n'+
                                   'Type `docker start psg` in the command line.')
        elif self.params.api_key_path is None:
            msg = 'PSG is being called without an API key. '
            msg += 'After 100 API calls in a 24hr period you will need to get a key. '
            msg += 'We suggest installing PSG locally using docker. (see https://psg.gsfc.nasa.gov/help.php#handbook)'
            warnings.warn(msg,RuntimeWarning)


        # for not using globes, append all configurations instead of rewritting

        if self.params.use_globes:
            file_mode = 'w'
        else:
            file_mode = 'a'

        ####################################
        # Initial upload of GCM
        gcm_path = self.params.gcm_path
        cfg_path = Path(self.dirs['data']) / 'cfg_temp.txt'
        if not self.params.use_globes:
            system(f'cp {gcm_path} {cfg_path}')
        url = self.params.psg_url
        call_type = 'set'
        if self.params.use_globes:
            app = 'globes'
        else:
            app = None
        outfile = None
        if self.params.api_key_path:
            with open(self.params.api_key_path, 'r', encoding='UTF-8') as file:
                api_key = file.read()
        else:
            api_key = None
        call_api(gcm_path, psg_url=url, api_key=api_key,
                 output_type=call_type, app=app, outfile=outfile, verbose=self.debug)
        ####################################
        # Set observation parameters that do not change
        cfg_path = Path(self.dirs['data']) / 'cfg_temp.txt'
        write_static_config(cfg_path,self.params,file_mode=file_mode)

        url = self.params.psg_url
        call_type = 'upd'
        if self.params.use_globes:
            app = 'globes'
        else:
            app = None
        outfile = None
        call_api(cfg_path, psg_url=url, api_key=api_key,
                 output_type=call_type, app=app, outfile=outfile, verbose=self.debug)
        # # debug
        # call_api(cfg_path,psg_url=url,api_key=api_key,
        #         output_type='all',app=app,outfile='temp_out.txt')

        ####################################
        # Calculate observation parameters
        observation_parameters = self.get_observation_parameters()
        obs_plan = self.get_planet_observation_plan(observation_parameters)

        obs_info_filename = Path(self.dirs['data']) / 'observation_info.csv'
        obs_df = pd.DataFrame()
        for key in obs_plan.keys():
            try:
                unit = obs_plan[key].unit
                name = f'{key}[{str(unit)}]'
                obs_df[name] = obs_plan[key].value
            except AttributeError:
                unit = ''
                name = f'{key}[{str(unit)}]'
                obs_df[name] = obs_plan[key]
        obs_df.to_csv(obs_info_filename, sep=',', index=False)

        print(
            f'Starting at phase {self.params.planet_initial_phase}, observe for {self.params.total_observation_time} in {self.params.planet_images} steps')
        print('Phases = ' +
              str(np.round(np.asarray((obs_plan['phase']/u.deg).to(u.Unit(''))), 2)) + ' deg')
        ####################################
        # iterate through phases
        for i in tqdm(range(self.params.planet_images), desc='Build Planet', total=self.params.planet_images):
            phase = obs_plan['phase'][i]
            sub_stellar_lon = obs_plan['sub_stellar_lon'][i]
            sub_stellar_lat = obs_plan['sub_stellar_lat'][i]
            orbit_radius_coeff = obs_plan['orbit_radius'][i]

            pl_sub_obs_lon = obs_plan['planet_sub_obs_lon'][i]
            pl_sub_obs_lat = obs_plan['planet_sub_obs_lat'][i]
            # Write updates to the config to change the phase value and ensure the star is of type 'StarType'
            with open(cfg_path, file_mode) as fr:
                fr.write('<OBJECT-STAR-TYPE>%s\n' %
                         self.params.psg_star_template)
                fr.write('<OBJECT-SEASON>%f\n' % to_float(phase, u.deg))
                fr.write('<OBJECT-STAR-DISTANCE>%f\n' %
                         to_float(orbit_radius_coeff*self.params.planet_semimajor_axis, u.AU))
                fr.write(
                    f'<OBJECT-SOLAR-LONGITUDE>{to_float(sub_stellar_lon,u.deg):.4f}\n')
                fr.write(
                    f'<OBJECT-SOLAR-LATITUDE>{to_float(sub_stellar_lat,u.deg)}\n')
                fr.write('<OBJECT-OBS-LONGITUDE>%f\n' %
                         to_float(pl_sub_obs_lon, u.deg))
                fr.write('<OBJECT-OBS-LATITUDE>%f\n' %
                         to_float(pl_sub_obs_lat, u.deg))
                # fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00')
                fr.close()
            # call api to get combined spectra
            url = self.params.psg_url
            call_type = None
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_combined']) / get_filename(i,N_ZFILL,'rad')
            call_api(cfg_path, psg_url=url, api_key=api_key,
                     output_type=call_type, app=app, outfile=outfile, verbose=self.debug)
            # call api to get noise
            url = self.params.psg_url
            call_type = 'noi'
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_noise']) / get_filename(i,N_ZFILL,'noi')
            call_api(cfg_path, psg_url=url, api_key=api_key,
                     output_type=call_type, app=app, outfile=outfile, verbose=self.debug)

            # call api to get config
            url = self.params.psg_url
            call_type = 'cfg'
            app = 'globes'
            outfile = Path(self.dirs['psg_configs']) / get_filename(i,N_ZFILL,'cfg')
            call_api(cfg_path, psg_url=url, api_key=api_key,
                     output_type=call_type, app=app, outfile=outfile, verbose=self.debug)

            # write updates to config file to remove star flux
            with open(cfg_path, file_mode) as fr:
                # phase *= -1
                fr.write('<OBJECT-STAR-TYPE>-\n')
                fr.write('<OBJECT-OBS-LONGITUDE>%f\n' %
                         to_float(pl_sub_obs_lon, u.deg))
                fr.write('<OBJECT-OBS-LATITUDE>%f\n' %
                         to_float(pl_sub_obs_lat, u.deg))
                fr.close()
            # call api to get thermal spectra
            url = self.params.psg_url
            call_type = None
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_thermal']) / get_filename(i,N_ZFILL,'rad')
            call_api(cfg_path, psg_url=url, api_key=api_key,
                     output_type=call_type, app=app, outfile=outfile, verbose=self.debug)
            # call api to get layers
            url = self.params.psg_url
            call_type = 'lyr'
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_layers']) / get_filename(i,N_ZFILL,'lyr')
            call_api(cfg_path, psg_url=url, api_key=api_key,
                     output_type=call_type, app=app, outfile=outfile, verbose=self.debug)

    def build_star(self):
        """
        Build a variable star model based on user-specified parameters.
        """
        empty_spot_collection = vsm.SpotCollection(Nlat=self.params.Nlat,
                                                   Nlon=self.params.Nlon)
        empty_fac_collection = vsm.FaculaCollection(Nlat=self.params.Nlat,
                                                    Nlon=self.params.Nlon)
        flare_generator = vsm.FlareGenerator(self.params.star_teff, self.params.star_rot_period, self.params.star_flare_group_prob,
                                             self.params.star_flare_mean_teff, self.params.star_flare_sigma_teff,
                                             self.params.star_flare_mean_log_fwhm_days, self.params.star_flare_sigma_log_fwhm_days,
                                             self.params.star_flare_log_E_erg_max, self.params.star_flare_log_E_erg_min, self.params.star_flare_log_E_erg_Nsteps)
        spot_generator = vsm.SpotGenerator(
            self.params.star_spot_mean_area, self.params.star_spot_sigma_area, self.params.star_spot_umbra_teff,
            self.params.star_spot_penumbra_teff, self.params.star_spot_growth_rate, self.params.star_spot_decay_rate,
            self.params.star_spot_initial_area, self.params.star_spot_distribution,
            self.params.star_spot_coverage, Nlat=self.params.Nlat, Nlon=self.params.Nlon
        )
        fac_generator = vsm.FaculaGenerator(
            R_peak=self.params.star_fac_mean_radius, R_HWHM=self.params.star_fac_HWHM_radius,
            T_peak=self.params.star_fac_mean_timescale, T_HWHM=self.params.star_fac_HWHM_timescale,
            coverage=self.params.star_fac_coverage, dist=self.params.star_fac_distribution,
            Nlat=self.params.Nlat, Nlon=self.params.Nlon
        )
        ld_params = [1-self.params.ld_a1-self.params.ld_a2,
                     self.params.ld_a1, self.params.ld_a2]
        self.star = vsm.Star(self.params.star_teff, self.params.star_radius,
                             self.params.star_rot_period, empty_spot_collection, empty_fac_collection,
                             name=self.params.star_name, distance=self.params.system_distance,
                             Nlat=self.params.Nlat, Nlon=self.params.Nlon, flare_generator=flare_generator,
                             spot_generator=spot_generator, fac_generator=fac_generator, ld_params=ld_params)

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
        if self.params.star_spot_initial_coverage > 0.0:
            self.star.generate_mature_spots(
                self.params.star_spot_initial_coverage)
            print(f'Generated {len(self.star.spots.spots)} mature spots')
        spot_warm_up_step = 1*u.day
        facula_warm_up_step = 1*u.hr
        N_steps_spot = int(
            round((spot_warmup_time/spot_warm_up_step).to(u.Unit('')).value))
        N_steps_facula = int(
            round((facula_warmup_time/facula_warm_up_step).to(u.Unit('')).value))
        if N_steps_spot > 0:
            for i in tqdm(range(N_steps_spot), desc='Spot Warmup', total=N_steps_spot):
                self.star.birth_spots(spot_warm_up_step)
                self.star.age(spot_warm_up_step)
        if N_steps_facula > 0:
            for i in tqdm(range(N_steps_facula), desc='Facula Warmup', total=N_steps_facula):
                self.star.birth_faculae(facula_warm_up_step)
                self.star.age(facula_warm_up_step)

        self.star.get_flares_over_observation(
            self.params.total_observation_time)

    def calculate_composite_stellar_spectrum(self, sub_obs_coords, tstart, tfinish):
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
        surface_dict = self.star.calc_coverage(sub_obs_coords)
        visible_flares = self.star.get_flare_int_over_timeperiod(
            tstart, tfinish, sub_obs_coords)
        base_wave, base_flux = self.get_model_spectrum(self.params.star_teff)
        base_flux = base_flux * 0
        for teff, coverage in surface_dict.items():
            if coverage > 0:
                wave, flux = self.get_model_spectrum(teff)
                if not np.all(isclose(base_wave, wave, 1e-3*u.um)):
                    raise ValueError('All wavelength axes must be equivalent.')
                base_flux = base_flux + flux * coverage
        for flare in visible_flares:
            teff = flare['Teff']
            timearea = flare['timearea']
            eff_area = (timearea/(tfinish-tstart)).to(u.km**2)
            flux = stellar_spectra.blackbody(base_wave, teff, eff_area, self.params.system_distance,
                                             target_unit_flux=self.params.target_flux_unit)
            base_flux = base_flux + flux

        return base_wave, base_flux

    def get_planet_indicies(self, planet_times: u.Quantity, tindex: u.Quantity) -> tuple[int, int]:
        """
        Get the incicies of the planet spectra to interpolate over.
        This is a helper function that allows for interpolation of planet spectra.
        Since the planet changes over much longer timescales than the star (flares, etc),
        it makes sense to only run PSG once for multiple "integrations".

        Parameters
        ----------
        planet_times : astropy.units.Quantity [time]
            The times (cast to since periasteron) at which the planet spectrum was taken.
        tindex : astropy.units.Quantity [time]
            The epoch of the current observation. The goal is to place this between
            two elements of `planet_times`

        Returns
        -------
        int
            The index of `planet_times` before `tindex`
        int
            The index of `planet_times` after `tindex`

        Raises
        ------
        ValueError
            If multiple elements of 'planet_times' are equal to 'tindex'.
        """
        after = planet_times > tindex
        equal = planet_times == tindex
        if equal.sum() == 1:
            N1 = np.argwhere(equal)[0][0]
            N2 = np.argwhere(equal)[0][0]
        elif equal.sum() > 1:
            raise ValueError('There must be a duplicate time')
        elif equal.sum() == 0:
            N2 = np.argwhere(after)[0][0]
            N1 = N2 - 1
        return N1, N2

    def calculate_reflected_spectra(self, N1, N2, N1_frac,
                                    sub_planet_wavelength, sub_planet_flux):
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
            self.dirs['psg_combined']) / get_filename(N1,N_ZFILL,'rad')
        psg_thermal_path1 = Path(
            self.dirs['psg_thermal']) / get_filename(N1,N_ZFILL,'rad')
        psg_combined_path2 = Path(
            self.dirs['psg_combined']) / get_filename(N2,N_ZFILL,'rad')
        psg_thermal_path2 = Path(
            self.dirs['psg_thermal']) / get_filename(N2,N_ZFILL,'rad')

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
            planet_reflection_only = get_reflected(combined,thermal)
            planet_reflection_fraction = to_float(planet_reflection_only / combined.data['Stellar'],u.dimensionless_unscaled)
            
            planet_reflection_adj = sub_planet_flux * planet_reflection_fraction
            reflected.append(planet_reflection_adj)

        return sub_planet_wavelength, reflected[0] * N1_frac + reflected[1] * (1-N1_frac)

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
            self.dirs['psg_combined']) / get_filename(N1, N_ZFILL,'rad')
        psg_noise_path1 = Path(
            self.dirs['psg_noise']) / get_filename(N1, N_ZFILL,'noi')
        psg_combined_path2 = Path(
            self.dirs['psg_combined']) / get_filename(N2, N_ZFILL,'rad')
        psg_noise_path2 = Path(
            self.dirs['psg_noise']) / get_filename(N2, N_ZFILL,'noi')

        psg_noise_source = []
        psg_source = []

        for psg_combined_path, psg_noise_path in zip(
            [psg_combined_path1, psg_combined_path2],
            [psg_noise_path1, psg_noise_path2]
        ):
            combined_df = pd.read_csv(psg_combined_path,
                                      comment='#',
                                      delim_whitespace=True,
                                      names=["Wave/freq", "Total", "Noise",
                                             "Stellar", "Planet", '_', '__'],
                                      )
            noise_df = pd.read_csv(psg_noise_path,
                                   comment='#',
                                   delim_whitespace=True,
                                   names=['Wave/freq', 'Total', 'Source',
                                          'Detector', 'Telescope', 'Background'],
                                   )
            if self.params.psg_rad_unit == 'Wm2um':
                flux_unit = u.Unit('W m-2 um-1')
            else:
                raise ValueError('That flux unit is not recognized')

            # validate
            if not np.all(isclose(cmb_wavelength, combined_df['Wave/freq'].values*self.params.target_wavelength_unit, 1e-3*u.um)
                          & isclose(cmb_wavelength, noise_df['Wave/freq'].values*self.params.target_wavelength_unit, 1e-3*u.um)):
                raise ValueError(
                    'The wavelength coordinates must be equivalent.')
            psg_noise_source.append(noise_df['Source'].values * flux_unit)
            psg_source.append(combined_df['Total'].values * flux_unit)
        psg_noise_source = psg_noise_source[0] * \
            N1_frac + psg_noise_source[1] * (1-N1_frac)
        psg_source = psg_source[0]*N1_frac + psg_source[1] * (1-N1_frac)

        model_noise = psg_noise_source * np.sqrt(cmb_flux/psg_source)
        noise_sq = (model_noise**2
                    + (noise_df['Detector'].values*flux_unit)**2
                    + (noise_df['Telescope'].values*flux_unit)**2
                    + (noise_df['Background'].values*flux_unit)**2)
        return cmb_wavelength, np.sqrt(noise_sq) * time_scale_factor

    def get_thermal_spectrum(self, N1: int, N2: int, N1_frac: float):
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
            self.dirs['psg_thermal']) / get_filename(N1, N_ZFILL,'rad')
        psg_thermal_path2 = Path(
            self.dirs['psg_thermal']) / get_filename(N2, N_ZFILL,'rad')

        wavelength = []
        thermal = []

        for psg_thermal_path in [psg_thermal_path1, psg_thermal_path2]:
            thermal_df = pd.read_csv(psg_thermal_path,
                                     comment='#',
                                     delim_whitespace=True,
                                     names=["Wave/freq", "Total",
                                            "Noise", "Planet", '_', '__'],
                                     )
            if self.params.psg_rad_unit == 'Wm2um':
                flux_unit = u.Unit('W m-2 um-1')
            else:
                raise ValueError('That flux unit is not recognized')

            wavelength.append(
                thermal_df['Wave/freq'].values * self.params.target_wavelength_unit)
            thermal.append(thermal_df['Planet'].values*flux_unit)

        if not np.all(isclose(wavelength[0], wavelength[1], 1e-3*u.um)):
            raise ValueError('The wavelength coordinates must be equivalent.')

        return wavelength[0], thermal[0]*N1_frac + thermal[1]*(1-N1_frac)

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
            self.dirs['psg_layers']) / get_filename(N1, N_ZFILL,'lyr')
        psg_layers_path2 = Path(
            self.dirs['psg_layers']) / get_filename(N2, N_ZFILL,'lyr')
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
            self.warm_up_star(spot_warmup_time=self.params.star_spot_warmup,
                              facula_warmup_time=self.params.star_fac_warmup)
        observation_parameters = self.get_observation_parameters()
        observation_info = self.get_observation_plan(observation_parameters)
        # write observation info to file
        obs_info_filename = Path(
            self.dirs['all_model']) / 'observation_info.csv'
        obs_df = pd.DataFrame()
        for key in observation_info.keys():
            try:
                unit = observation_info[key].unit
                name = f'{key}[{str(unit)}]'
                obs_df[name] = observation_info[key].value
            except AttributeError:
                unit = ''
                name = f'{key}[{str(unit)}]'
                obs_df[name] = observation_info[key]
        obs_df.to_csv(obs_info_filename, sep=',', index=False)

        planet_observation_info = self.get_planet_observation_plan(
            observation_parameters)
        planet_times = planet_observation_info['time']

        time_step = self.params.total_observation_time / self.params.total_images
        planet_time_step = self.params.total_observation_time / self.params.planet_images

        for index in tqdm(range(self.params.total_images), desc='Build Spectra', total=self.params.total_images, position=0, leave=True):

            tindex = observation_info['time'][index]
            tstart = tindex - observation_info['time'][0]
            tfinish = tstart + time_step
            planetPhase = observation_info['phase'][index]
            sub_obs_lon = observation_info['sub_obs_lon'][index]
            sub_obs_lat = observation_info['sub_obs_lat'][index]
            N1, N2 = self.get_planet_indicies(planet_times, tindex)
            N1_frac = 1 - \
                to_float(
                    (tindex - planet_times[N1])/planet_time_step, u.Unit(''))

            sub_planet_lon = observation_info['sub_planet_lon'][index]
            sub_planet_lat = observation_info['sub_planet_lat'][index]

            comp_wave, comp_flux = self.calculate_composite_stellar_spectrum({'lat': sub_obs_lat,
                                                                              'lon': sub_obs_lon}, tstart, tfinish)
            wave, to_planet_flux = self.calculate_composite_stellar_spectrum({'lat': sub_planet_lat,
                                                                              'lon': sub_planet_lon}, tstart, tfinish)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            wave, reflection_flux_adj = self.calculate_reflected_spectra(
                N1, N2, N1_frac, comp_wave, to_planet_flux)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            wave, thermal_spectrum = self.get_thermal_spectrum(N1, N2, N1_frac)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            combined_flux = comp_flux + reflection_flux_adj + thermal_spectrum

            wave, noise_flux_adj = self.calculate_noise(N1, N2, N1_frac,
                                                        np.sqrt(
                                                            to_float(planet_time_step/time_step, u.Unit(''))),
                                                        comp_wave, combined_flux)
            assert np.all(isclose(comp_wave, wave, 1e-3*u.um))

            df = pd.DataFrame({
                f'wavelength[{str(comp_wave.unit)}]': comp_wave.value,
                f'star[{str(comp_flux.unit)}]': comp_flux.value,
                f'star_towards_planet[{str(to_planet_flux.unit)}]': to_planet_flux.value,
                f'reflected[{str(reflection_flux_adj.unit)}]': reflection_flux_adj.value,
                f'planet_thermal[{str(thermal_spectrum.unit)}]': thermal_spectrum.value,
                f'total[{str(combined_flux.unit)}]': combined_flux.value,
                f'noise[{str(noise_flux_adj.unit)}]': noise_flux_adj.value
            })
            outfile = Path(self.dirs['all_model']) / get_filename(index, N_ZFILL,'csv')
            df.to_csv(outfile, index=False, sep=',')

            # layers
            if self.params.use_globes and self.params.use_molec_signatures:
                layerdat = self.get_layer_data(N1, N2, N1_frac)
                outfile = Path(self.dirs['all_model']) / \
                    f'layer{str(index).zfill(N_ZFILL)}.csv'
                layerdat.to_csv(outfile, index=False, sep=',')

            self.star.birth_spots(time_step)
            self.star.birth_faculae(time_step)
            self.star.age(time_step)
