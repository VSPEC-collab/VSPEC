from pathlib import Path
from os import system

import numpy as np
import pandas as pd
from astropy import units as u
from tqdm.auto import tqdm

from VSPEC import stellar_spectra
from VSPEC import variable_star_model as vsm
from VSPEC.files import build_directories
from VSPEC.geometry import SystemGeometry
from VSPEC.helpers import isclose, to_float
from VSPEC.psg_api import call_api
from VSPEC.read_info import ParamModel
from VSPEC.analysis import read_lyr


class ObservationModel:
    """Obsservation Model
    Main class that stores the information of this simulation

    Args:
        config_path (str or pathlib.Path): path of the configuration file
    
    Returns:
        None
    """

    def __init__(self,config_path, debug=False):
        self.debug=debug
        self.params = ParamModel(config_path)
        self.build_directories()
    
    def build_directories(self):
        """build directories
        Build the file system for this run
        """
        self.dirs = build_directories(self.params.star_name)
    
    def bin_spectra(self):
        teffs = 100*np.arange(np.floor(self.params.star_teff_min.to(u.K)/100/u.K),
                                np.ceil(self.params.star_teff_max.to(u.K)/100/u.K)+1) * u.K
        for teff in tqdm(teffs,desc='Binning Spectra',total=len(teffs)):
            stellar_spectra.bin_phoenix_model(to_float(teff,u.K),
                file_name_writer=stellar_spectra.get_binned_filename,
                binned_path=self.dirs['binned'],R=self.params.resolving_power,
                lam1=self.params.lambda_min,lam2=self.params.lambda_max,
                model_unit_wavelength=u.AA,model_unit_flux=u.Unit('erg s-1 cm-2 cm-1'),
                target_unit_wavelength=self.params.target_wavelength_unit,
                target_unit_flux = self.params.target_flux_unit)
    
    def read_spectrum(self,teff):
        filename = stellar_spectra.get_binned_filename(to_float(teff,u.K))
        path = self.dirs['binned']
        return stellar_spectra.read_binned_spectrum(filename,path=path)

    def get_model_spectrum(self,Teff):
        model_teffs = [to_float(np.round(Teff - Teff%(100*u.K)),u.K),
            to_float(np.round(Teff - Teff%(100*u.K) +(100*u.K)),u.K)] * u.K
        if Teff in model_teffs:
            wave1, flux1 = self.read_spectrum(Teff)
            wave2, flux2 = wave1, flux1
        else:
            wave1, flux1 = self.read_spectrum(model_teffs[0])
            wave2, flux2 = self.read_spectrum(model_teffs[1])
        wavelength, flux = stellar_spectra.interpolate_spectra(Teff,
                                model_teffs[0],wave1,flux1,
                                model_teffs[1],wave2,flux2)
        return wavelength, flux*self.params.distanceFluxCorrection
    
    def get_observation_parameters(self):
        return SystemGeometry(self.params.system_inclination_psg,
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
                            self.params.system_argument_of_pariapsis,
                            self.params.system_distance,
                            self.params.planet_obliquity,
                            self.params.planet_obliquity_direction)
    
    def get_observation_plan(self,observation_parameters:SystemGeometry):
        return observation_parameters.get_observation_plan(self.params.planet_initial_phase,
                self.params.total_observation_time,N_obs=self.params.total_images)
    def get_planet_observation_plan(self,observation_parameters:SystemGeometry):
        return observation_parameters.get_observation_plan(self.params.planet_initial_phase,
                self.params.total_observation_time,N_obs=self.params.planet_images)

    def build_planet(self):
        """build planet
        Follow steps in original PlanetBuilder.py file
        """

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
            api_key = open(self.params.api_key_path,'r').read()
        else:
            api_key = None
        call_api(gcm_path,psg_url=url,api_key=api_key,
                type=call_type,app=app,outfile=outfile,verbose=self.debug)
        ####################################
        # Set observation parameters that do not change
        cfg_path = Path(self.dirs['data']) / 'cfg_temp.txt'
        with open(cfg_path, file_mode) as fr:
            bool_to_str = {True:'Y',False:'N'}
            fr.write('<OBJECT>Exoplanet\n')
            fr.write('<OBJECT-NAME>Planet\n')
            fr.write('<OBJECT-DIAMETER>%f\n' % to_float(2*self.params.planet_radius,u.km))
            fr.write('<OBJECT-GRAVITY>%f\n' % self.params.planet_grav)
            fr.write(f'<OBJECT-GRAVITY-UNIT>{self.params.planet_grav_mode}\n')
            fr.write('<OBJECT-STAR-TYPE>%s\n' % self.params.psg_star_template)
            fr.write('<OBJECT-STAR-DISTANCE>%f\n' % to_float(self.params.planet_semimajor_axis,u.AU))
            fr.write('<OBJECT-PERIOD>%f\n' % to_float(self.params.planet_orbital_period,u.day))
            fr.write('<OBJECT-ECCENTRICITY>%f\n' % self.params.planet_eccentricity)
            fr.write('<OBJECT-PERIAPSIS>%f\n' % to_float(self.params.system_argument_of_pariapsis,u.deg))
            fr.write('<OBJECT-STAR-TEMPERATURE>%f\n' % to_float(self.params.star_teff,u.K))
            fr.write('<OBJECT-STAR-RADIUS>%f\n' % to_float(self.params.star_radius,u.R_sun))
            fr.write('<GEOMETRY>Observatory\n')
            fr.write('<GEOMETRY-OBS-ALTITUDE>%f\n' % to_float(self.params.system_distance,u.pc))
            fr.write('<GEOMETRY-ALTITUDE-UNIT>pc\n')
            fr.write('<GENERATOR-RANGE1>%f\n' % to_float(self.params.lambda_min,self.params.target_wavelength_unit))
            fr.write('<GENERATOR-RANGE2>%f\n' % to_float(self.params.lambda_max,self.params.target_wavelength_unit))
            fr.write(f'<GENERATOR-RANGEUNIT>{self.params.target_wavelength_unit}\n')
            fr.write('<GENERATOR-RESOLUTION>%f\n' % self.params.resolving_power)
            fr.write('<GENERATOR-RESOLUTIONUNIT>RP\n')
            fr.write('<GENERATOR-BEAM>%d\n' % self.params.beamValue)
            fr.write('<GENERATOR-BEAM-UNIT>%s\n'% self.params.beamUnit)
            fr.write('<GENERATOR-CONT-STELLAR>Y\n')
            fr.write('<OBJECT-INCLINATION>%s\n' % to_float(self.params.system_inclination_psg,u.deg))
            fr.write('<OBJECT-SOLAR-LATITUDE>0.0\n')
            fr.write('<OBJECT-OBS-LATITUDE>0.0\n')
            fr.write('<GENERATOR-RADUNITS>%s\n' % self.params.psg_rad_unit)
            fr.write('<GENERATOR-GCM-BINNING>%d\n' % self.params.gcm_binning)
            fr.write(f'<GENERATOR-GAS-MODEL>{bool_to_str[self.params.use_molec_signatures]}\n')
            fr.write(f'<GENERATOR-NOISE>{self.params.detector_type}\n')
            fr.write(f'<GENERATOR-NOISE2>{self.params.detector_dark_current}\n')
            fr.write(f'<GENERATOR-NOISETIME>{self.params.detector_integration_time}\n')
            fr.write(f'<GENERATOR-NOISEOTEMP>{self.params.detector_temperature}\n')
            fr.write(f'<GENERATOR-NOISEOEFF>{self.params.detector_throughput:.1f}\n')
            fr.write(f'<GENERATOR-NOISEOEMIS>{self.params.detector_emissivity:.1f}\n')
            fr.write(f'<GENERATOR-NOISEFRAMES>{self.params.detector_number_of_integrations}\n')
            fr.write(f'<GENERATOR-NOISEPIXELS>{self.params.detector_pixel_sampling}\n')
            fr.write(f'<GENERATOR-NOISE1>{self.params.detector_read_noise}\n')
            fr.write(f'<GENERATOR-DIAMTELE>{self.params.telescope_diameter:.1f}\n')
            fr.write(f'<GENERATOR-TELESCOPE>SINGLE\n')
            fr.write(f'<GENERATOR-TELESCOPE1>1\n')
            fr.write(f'<GENERATOR-TELESCOPE2>1.0\n')
            fr.write(f'<GENERATOR-TELESCOPE3>1.0\n')
        url = self.params.psg_url
        call_type = 'upd'
        if self.params.use_globes:
            app = 'globes'
        else:
            app = None
        outfile = None
        call_api(cfg_path,psg_url=url,api_key=api_key,
                type=call_type,app=app,outfile=outfile,verbose=self.debug)
        # # debug
        # call_api(cfg_path,psg_url=url,api_key=api_key,
        #         type='all',app=app,outfile='temp_out.txt')

        ####################################
        # Calculate observation parameters
        observation_parameters = self.get_observation_parameters()
        obs_plan = self.get_planet_observation_plan(observation_parameters)

        obs_info_filename = Path(self.dirs['data']) / 'observation_info.csv'
        obs_df = pd.DataFrame()
        for key in obs_plan.keys():
            try:
                unit  = obs_plan[key].unit
                name = f'{key}[{str(unit)}]'
                obs_df[name] = obs_plan[key].value
            except AttributeError:
                unit = ''
                name = f'{key}[{str(unit)}]'
                obs_df[name] = obs_plan[key]
        obs_df.to_csv(obs_info_filename,sep=',',index=False)

        print(f'Starting at phase {self.params.planet_initial_phase}, observe for {self.params.total_observation_time} in {self.params.planet_images} steps')
        print('Phases = ' + str(np.round(np.asarray((obs_plan['phase']/u.deg).to(u.Unit(''))),2)) + ' deg')
        ####################################
        # iterate through phases
        for i in tqdm(range(self.params.planet_images),desc='Build Planet',total=self.params.planet_images):
            phase = obs_plan['phase'][i]
            sub_stellar_lon = obs_plan['sub_stellar_lon'][i]
            sub_stellar_lat = obs_plan['sub_stellar_lat'][i]
            orbit_radius_coeff = obs_plan['orbit_radius'][i]
            
            
            if phase>178*u.deg and phase<182*u.deg:
                phase=182.0*u.deg # Add transit phase;
            if phase == 185*u.deg:
                phase = 186.0*u.deg
            
            pl_sub_obs_lon = obs_plan['planet_sub_obs_lon'][i]
            pl_sub_obs_lat =  obs_plan['planet_sub_obs_lat'][i]
            # Write updates to the config to change the phase value and ensure the star is of type 'StarType'
            with open(cfg_path, file_mode) as fr:
                fr.write('<OBJECT-STAR-TYPE>%s\n' % self.params.psg_star_template)
                fr.write('<OBJECT-SEASON>%f\n' % to_float(phase,u.deg))
                fr.write('<OBJECT-STAR-DISTANCE>%f\n' % to_float(orbit_radius_coeff*self.params.planet_semimajor_axis,u.AU))
                fr.write(f'<OBJECT-SOLAR-LONGITUDE>{to_float(sub_stellar_lon,u.deg):.4f}\n')
                fr.write(f'<OBJECT-SOLAR-LATITUDE>{to_float(0*u.deg,u.deg)}\n') # Add this option later
                fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % to_float(pl_sub_obs_lon,u.deg))
                fr.write('<OBJECT-OBS-LATITUDE>%f\n' % to_float(pl_sub_obs_lat,u.deg))
                # fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00')
                fr.close()
            # call api to get combined spectra
            url = self.params.psg_url
            call_type = None
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_combined']) / f'phase{str(i).zfill(3)}.rad'
            call_api(cfg_path,psg_url=url,api_key=api_key,
                    type=call_type,app=app,outfile=outfile,verbose=self.debug)
            # call api to get noise
            url = self.params.psg_url
            call_type = 'noi'
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_noise']) / f'phase{str(i).zfill(3)}.noi'
            call_api(cfg_path,psg_url=url,api_key=api_key,
                    type=call_type,app=app,outfile=outfile,verbose=self.debug)

            # call api to get config
            url = self.params.psg_url
            call_type = 'cfg'
            app = 'globes'
            outfile = Path(self.dirs['psg_configs']) / f'phase{str(i).zfill(3)}.cfg'
            call_api(cfg_path,psg_url=url,api_key=api_key,
                    type=call_type,app=app,outfile=outfile,verbose=self.debug)

            # write updates to config file to remove star flux
            with open(cfg_path, file_mode) as fr:
                # phase *= -1
                fr.write('<OBJECT-STAR-TYPE>-\n')
                fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % to_float(pl_sub_obs_lon,u.deg))
                fr.write('<OBJECT-OBS-LATITUDE>%f\n' % to_float(pl_sub_obs_lat,u.deg)) 
                fr.close()
            # call api to get thermal spectra
            url = self.params.psg_url
            call_type = None
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_thermal']) / f'phase{str(i).zfill(3)}.rad'
            call_api(cfg_path,psg_url=url,api_key=api_key,
                    type=call_type,app=app,outfile=outfile,verbose=self.debug)
            # call api to get layers
            url = self.params.psg_url
            call_type = 'lyr'
            if self.params.use_globes:
                app = 'globes'
            else:
                app = None
            outfile = Path(self.dirs['psg_layers']) / f'phase{str(i).zfill(3)}.lyr'
            call_api(cfg_path,psg_url=url,api_key=api_key,
                    type=call_type,app=app,outfile=outfile,verbose=self.debug)
    

    def build_star(self):
        empty_spot_collection = vsm.SpotCollection(Nlat = self.params.Nlat,
                                                    Nlon = self.params.Nlon)
        empty_fac_collection = vsm.FaculaCollection(Nlat = self.params.Nlat,
                                                    Nlon = self.params.Nlon)
        flare_generator = vsm.FlareGenerator(self.params.star_teff,self.params.star_rot_period,self.params.star_flare_group_prob,
                                            self.params.star_flare_mean_teff,self.params.star_flare_sigma_teff,
                                            self.params.star_flare_mean_log_fwhm_days,self.params.star_flare_sigma_log_fwhm_days,
                                            self.params.star_flare_log_E_erg_max,self.params.star_flare_log_E_erg_min,self.params.star_flare_log_E_erg_Nsteps)
        spot_generator = vsm.SpotGenerator(
            self.params.star_spot_mean_area,self.params.star_spot_sigma_area,self.params.star_spot_umbra_teff,
            self.params.star_spot_penumbra_teff,self.params.star_spot_growth_rate,self.params.star_spot_decay_rate,
            self.params.star_spot_initial_area,self.params.star_spot_distribution,
            self.params.star_spot_coverage,Nlat=self.params.Nlat,Nlon=self.params.Nlon
            )
        fac_generator = vsm.FaculaGenerator(
            R_peak = self.params.star_fac_mean_radius,R_HWHM = self.params.star_fac_HWHM_radius,
            T_peak=self.params.star_fac_mean_timescale,T_HWHM=self.params.star_fac_HWHM_timescale,
            coverage = self.params.star_fac_coverage,dist=self.params.star_fac_distribution,
            Nlat=self.params.Nlat,Nlon=self.params.Nlon
        )
        self.star = vsm.Star(self.params.star_teff,self.params.star_radius,
                            self.params.star_rot_period,empty_spot_collection,empty_fac_collection,
                            name = self.params.star_name,distance = self.params.system_distance,
                            Nlat = self.params.Nlat, Nlon = self.params.Nlon,flare_generator=flare_generator,
                            spot_generator=spot_generator, fac_generator=fac_generator)


    def warm_up_star(self, spot_warmup_time=30*u.day, facula_warmup_time=3*u.day):

        if self.params.star_spot_initial_coverage > 0.0:
            self.star.generate_mature_spots(self.params.star_spot_initial_coverage)
            print(f'Generated {len(self.star.spots.spots)} mature spots')
        spot_warm_up_step = 1*u.day
        facula_warm_up_step = 1*u.hr
        N_steps_spot = int(round((spot_warmup_time/spot_warm_up_step).to(u.Unit('')).value))
        N_steps_facula = int(round((facula_warmup_time/facula_warm_up_step).to(u.Unit('')).value))
        for i in tqdm(range(N_steps_spot),desc='Spot Warmup',total=N_steps_spot):
            self.star.birth_spots(spot_warm_up_step)
            self.star.age(spot_warm_up_step)
        for i in tqdm(range(N_steps_facula),desc='Facula Warmup',total=N_steps_facula):
            self.star.birth_faculae(facula_warm_up_step)
            self.star.age(facula_warm_up_step)

        self.star.get_flares_over_observation(self.params.total_observation_time)

    def calculate_composite_stellar_spectrum(self,sub_obs_coords,tstart,tfinish):
        surface_dict = self.star.calc_coverage(sub_obs_coords)
        visible_flares = self.star.get_flare_int_over_timeperiod(tstart,tfinish,sub_obs_coords)
        base_wave, base_flux = self.get_model_spectrum(self.params.star_teff)
        base_flux = base_flux * 0
        for teff, coverage in surface_dict.items():
            if coverage > 0:
                wave, flux = self.get_model_spectrum(teff)
                assert np.all(isclose(base_wave,wave,1e-3*u.um))
                base_flux = base_flux + flux * coverage
        for flare in visible_flares:
            teff = flare['Teff']
            timearea = flare['timearea']
            eff_area = (timearea/(tfinish-tstart)).to(u.km**2)
            flux = stellar_spectra.blackbody(base_wave,teff,eff_area,self.params.system_distance,
                                            target_unit_flux=self.params.target_flux_unit)
            base_flux = base_flux + flux

        return base_wave, base_flux

    def get_planet_indicies(self,planet_times,tindex):
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
        return N1,N2

    def calculate_reflected_spectra(self,N1,N2, N1_frac,
                                    sub_planet_wavelength,sub_planet_flux):

        psg_combined_path1 = Path(self.dirs['psg_combined']) / f'phase{str(N1).zfill(3)}.rad'
        psg_thermal_path1 = Path(self.dirs['psg_thermal']) / f'phase{str(N1).zfill(3)}.rad'
        psg_combined_path2 = Path(self.dirs['psg_combined']) / f'phase{str(N2).zfill(3)}.rad'
        psg_thermal_path2 = Path(self.dirs['psg_thermal']) / f'phase{str(N2).zfill(3)}.rad'

        reflected = []

        for psg_combined_path, psg_thermal_path in zip([psg_combined_path1,psg_combined_path2],
                                                        [psg_thermal_path1,psg_thermal_path2]):
            combined_df = pd.read_csv(psg_combined_path,
                comment='#',
                delim_whitespace=True,
                names=["Wave/freq", "Total", "Noise", "Stellar", "Planet",'_','__'],
                )
            thermal_df = pd.read_csv(psg_thermal_path,
                comment='#',
                delim_whitespace=True,
                names=["Wave/freq", "Total", "Noise", "Planet",'_','__'],
                )
            if self.params.psg_rad_unit == 'Wm2um':
                flux_unit = u.Unit('W m-2 um-1')
            else:
                raise NotImplementedError('That flux unit is not implemented')
            
            #validate
            assert np.all(isclose(sub_planet_wavelength,combined_df['Wave/freq'].values*self.params.target_wavelength_unit,1e-3*u.um)
                            & isclose(sub_planet_wavelength,thermal_df['Wave/freq'].values*self.params.target_wavelength_unit,1e-3*u.um))
            
            planet_reflection_only = combined_df['Planet'].values * flux_unit - thermal_df['Planet'].values*flux_unit
            planet_reflection_fraction = planet_reflection_only / (combined_df['Stellar'].values*flux_unit)
            planet_reflection_adj = sub_planet_flux * planet_reflection_fraction
            reflected.append(planet_reflection_adj)
        
        return sub_planet_wavelength,reflected[0] * N1_frac + reflected[1] * (1-N1_frac)
    def calculate_noise(self,N1,N2,N1_frac,time_scale_factor,cmb_wavelength,cmb_flux):
        psg_combined_path1 = Path(self.dirs['psg_combined']) / f'phase{str(N1).zfill(3)}.rad'
        psg_noise_path1 = Path(self.dirs['psg_noise']) / f'phase{str(N1).zfill(3)}.noi'
        psg_combined_path2 = Path(self.dirs['psg_combined']) / f'phase{str(N2).zfill(3)}.rad'
        psg_noise_path2 = Path(self.dirs['psg_noise']) / f'phase{str(N2).zfill(3)}.noi'
        
        psg_noise_source = []
        psg_source = []
        
        for psg_combined_path,psg_noise_path in zip(
            [psg_combined_path1,psg_combined_path2],
            [psg_noise_path1,psg_noise_path2]
            ):
            combined_df = pd.read_csv(psg_combined_path,
                comment='#',
                delim_whitespace=True,
                names=["Wave/freq", "Total", "Noise", "Stellar", "Planet",'_','__'],
                )
            noise_df = pd.read_csv(psg_noise_path,
                comment='#',
                delim_whitespace=True,
                names=['Wave/freq','Total','Source','Detector','Telescope','Background'],
                )
            if self.params.psg_rad_unit == 'Wm2um':
                flux_unit = u.Unit('W m-2 um-1')
            else:
                raise NotImplementedError('That flux unit is not implemented')

            # validate
            assert np.all(isclose(cmb_wavelength,combined_df['Wave/freq'].values*self.params.target_wavelength_unit,1e-3*u.um)
                            & isclose(cmb_wavelength,noise_df['Wave/freq'].values*self.params.target_wavelength_unit,1e-3*u.um))
            psg_noise_source.append(noise_df['Source'].values * flux_unit)
            psg_source.append(combined_df['Total'].values * flux_unit)
        psg_noise_source = psg_noise_source[0]*N1_frac + psg_noise_source[1] * (1-N1_frac)
        psg_source = psg_source[0]*N1_frac + psg_source[1] * (1-N1_frac)

        model_noise = psg_noise_source * np.sqrt(cmb_flux/psg_source)
        noise_sq = (model_noise**2
                    + (noise_df['Detector'].values*flux_unit)**2
                    + (noise_df['Telescope'].values*flux_unit)**2
                    + (noise_df['Background'].values*flux_unit)**2)
        return cmb_wavelength, np.sqrt(noise_sq) * time_scale_factor


    def get_thermal_spectrum(self,N1,N2,N1_frac):
        psg_thermal_path1 = Path(self.dirs['psg_thermal']) / f'phase{str(N1).zfill(3)}.rad'
        psg_thermal_path2 = Path(self.dirs['psg_thermal']) / f'phase{str(N2).zfill(3)}.rad'
        
        wavelength = []
        thermal = []

        for psg_thermal_path in [psg_thermal_path1,psg_thermal_path2]:
            thermal_df = pd.read_csv(psg_thermal_path,
                comment='#',
                delim_whitespace=True,
                names=["Wave/freq", "Total", "Noise", "Planet",'_','__'],
                )
            if self.params.psg_rad_unit == 'Wm2um':
                flux_unit = u.Unit('W m-2 um-1')
            else:
                raise NotImplementedError('That flux unit is not implemented')
            
            wavelength.append(thermal_df['Wave/freq'].values * self.params.target_wavelength_unit)
            thermal.append(thermal_df['Planet'].values*flux_unit)
        
        assert np.all(isclose(wavelength[0],wavelength[1],1e-3*u.um))
        
        return wavelength[0], thermal[0]*N1_frac + thermal[1]*(1-N1_frac)

    def get_layer_data(self,N1:int,N2:int,N1_frac:float)->pd.DataFrame:
        psg_layers_path1 = Path(self.dirs['psg_layers']) / f'phase{str(N1).zfill(3)}.lyr'
        psg_layers_path2 = Path(self.dirs['psg_layers']) / f'phase{str(N2).zfill(3)}.lyr'
        layers1 = read_lyr(psg_layers_path1)
        layers2 = read_lyr(psg_layers_path2)
        assert np.all(layers1.columns == layers2.columns) & (len(layers1)==len(layers2))
        cols = layers1.columns
        dat = layers1.values * N1_frac + layers2.values * (1-N1_frac)
        df = pd.DataFrame(columns=cols,data=dat)
        return df

    def build_spectra(self):
        """build spectra"""
        if not hasattr(self,'star'): # user can define a custom star before calling this function, e.g. for a specific spot pattern
            self.build_star()
            self.warm_up_star(spot_warmup_time=self.params.star_spot_warmup,
                            facula_warmup_time=self.params.star_fac_warmup)
        observation_parameters = self.get_observation_parameters()
        observation_info = self.get_observation_plan(observation_parameters)
        # write observation info to file
        obs_info_filename = Path(self.dirs['all_model']) / 'observation_info.csv'
        obs_df = pd.DataFrame()
        for key in observation_info.keys():
            try:
                unit  = observation_info[key].unit
                name = f'{key}[{str(unit)}]'
                obs_df[name] = observation_info[key].value
            except AttributeError:
                unit = ''
                name = f'{key}[{str(unit)}]'
                obs_df[name] = observation_info[key]
        obs_df.to_csv(obs_info_filename,sep=',',index=False)

        planet_observation_info = self.get_planet_observation_plan(observation_parameters)
        planet_times = planet_observation_info['time']

        time_step = self.params.total_observation_time / self.params.total_images
        planet_time_step = self.params.total_observation_time / self.params.planet_images

        for index in tqdm(range(self.params.total_images),desc='Build Spectra',total=self.params.total_images,position=0,leave=True):

            tindex = observation_info['time'][index]
            tstart = tindex - observation_info['time'][0]
            tfinish = tstart + time_step
            planetPhase = observation_info['phase'][index]
            sub_obs_lon = observation_info['sub_obs_lon'][index]
            sub_obs_lat = observation_info['sub_obs_lat'][index]
            N1,N2 = self.get_planet_indicies(planet_times,tindex)
            N1_frac = 1 - to_float((tindex - planet_times[N1])/planet_time_step,u.Unit(''))
            
            sub_planet_lon = observation_info['sub_planet_lon'][index]
            sub_planet_lat = observation_info['sub_planet_lat'][index]
        
            comp_wave, comp_flux = self.calculate_composite_stellar_spectrum({'lat':sub_obs_lat,
                                                            'lon':sub_obs_lon},tstart,tfinish)
            wave, to_planet_flux = self.calculate_composite_stellar_spectrum({'lat':sub_planet_lat,
                                                            'lon':sub_planet_lon},tstart,tfinish)
            assert np.all(isclose(comp_wave,wave,1e-3*u.um))

            wave, reflection_flux_adj = self.calculate_reflected_spectra(N1,N2,N1_frac,comp_wave,to_planet_flux)
            assert np.all(isclose(comp_wave,wave,1e-3*u.um))

            wave, thermal_spectrum = self.get_thermal_spectrum(N1,N2,N1_frac)
            assert np.all(isclose(comp_wave,wave,1e-3*u.um))

            combined_flux = comp_flux + reflection_flux_adj + thermal_spectrum


            wave, noise_flux_adj = self.calculate_noise(N1,N2,N1_frac,
                        np.sqrt(to_float(planet_time_step/time_step,u.Unit(''))),
                        comp_wave, combined_flux)
            assert np.all(isclose(comp_wave,wave,1e-3*u.um))

            df = pd.DataFrame({
                f'wavelength[{str(comp_wave.unit)}]': comp_wave.value,
                f'star[{str(comp_flux.unit)}]': comp_flux.value,
                f'star_towards_planet[{str(to_planet_flux.unit)}]': to_planet_flux.value,
                f'reflected[{str(reflection_flux_adj.unit)}]': reflection_flux_adj.value,
                f'planet_thermal[{str(thermal_spectrum.unit)}]': thermal_spectrum.value,
                f'total[{str(combined_flux.unit)}]': combined_flux.value,
                f'noise[{str(noise_flux_adj.unit)}]': noise_flux_adj.value
            })
            outfile = Path(self.dirs['all_model']) / f'phase{str(index).zfill(3)}.csv'
            df.to_csv(outfile,index=False,sep=',')

            #layers
            if self.params.use_globes and self.params.use_molec_signatures:
                layerdat = self.get_layer_data(N1,N2,N1_frac)
                outfile = Path(self.dirs['all_model']) / f'layer{str(index).zfill(3)}.csv'
                layerdat.to_csv(outfile,index=False,sep=',')


            self.star.birth_spots(time_step)
            self.star.birth_faculae(time_step)
            self.star.age(time_step)