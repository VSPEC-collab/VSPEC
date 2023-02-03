import configparser
from astropy import units as u, constants as c
from pathlib import Path

from VSPEC.helpers import to_float
from VSPEC.variable_star_model import MSH


class ParamModel:
    """Parameter Model
    Class to read and store information from the configuration file

    Args:
        filename (str): path of the configuration file

    Returns:
        None
    """

    def __init__(self,filename):
        configParser = configparser.RawConfigParser()
        configParser.read_file(open(filename))
        # Read in the information of the star from the config file
        self.star_name = configParser.get('Star', 'star_name')

        # Star Properties
        self.star_teff = configParser.getint('Star', 'star_teff') * u.K
        self.star_teff_min = configParser.getint('Star', 'star_teff_min') * u.K
        self.star_teff_max = configParser.getint('Star', 'star_teff_max') * u.K

        self.ld_a1 = configParser.getfloat('Star','limb_darkening_a1')
        self.ld_a2 = configParser.getfloat('Star','limb_darkening_a2')

        self.star_spot_initial_coverage = configParser.getfloat('Star','star_spot_initial_coverage')
        self.star_spot_distribution = configParser.get('Star','star_spot_distribution')
        self.star_spot_mean_area = configParser.getfloat('Star','star_spot_mean_area') * MSH
        self.star_spot_sigma_area = configParser.getfloat('Star','star_spot_sigma_area')
        self.star_spot_umbra_teff = configParser.getfloat('Star','star_spot_umbra_teff')*u.K
        self.star_spot_penumbra_teff = configParser.getfloat('Star','star_spot_penumbra_teff')*u.K
        self.star_spot_growth_rate = configParser.getfloat('Star','star_spot_growth_rate') / u.day
        self.star_spot_decay_rate = configParser.getfloat('Star','star_spot_decay_rate') * MSH / u.day
        self.star_spot_initial_area = configParser.getfloat('Star','star_spot_initial_area') * MSH
        self.star_spot_coverage = float(configParser.get('Star', 'star_spot_coverage'))
        self.star_spot_warmup = configParser.getfloat('Star','star_spot_warmup') * u.day

        self.star_fac_coverage = float(configParser.get('Star', 'star_fac_coverage'))
        self.star_fac_mean_radius = configParser.getfloat('Star','star_fac_mean_radius') * u.km
        self.star_fac_HWHM_radius = configParser.getfloat('Star','star_fac_HWMH_radius') * u.km
        self.star_fac_mean_timescale = configParser.getfloat('Star','star_fac_mean_timescale') * u.hr
        self.star_fac_HWHM_timescale = configParser.getfloat('Star','star_fac_HWMH_timescale') * u.hr
        self.star_fac_distribution = configParser.get('Star','star_fac_distribution')
        self.star_fac_warmup = configParser.getfloat('Star','star_fac_warmup') * u.hr

        self.star_flare_group_prob = configParser.getfloat('Star','star_flare_group_prob')
        self.star_flare_mean_teff = configParser.getfloat('Star','star_flare_mean_teff') * u.K
        self.star_flare_sigma_teff = configParser.getfloat('Star','star_flare_sigma_teff') * u.K
        self.star_flare_mean_log_fwhm_days = configParser.getfloat('Star','star_flare_mean_log_fwhm_days')
        self.star_flare_sigma_log_fwhm_days = configParser.getfloat('Star','star_flare_sigma_log_fwhm_days')
        self.star_flare_log_E_erg_max = configParser.getfloat('Star','star_flare_log_E_erg_max')
        self.star_flare_log_E_erg_min = configParser.getfloat('Star','star_flare_log_E_erg_min')
        self.star_flare_log_E_erg_Nsteps = configParser.getint('Star','star_flare_log_E_erg_Nsteps')



        self.star_mass = configParser.getfloat('Star', 'star_mass') * u.M_sun
        self.star_radius = configParser.getfloat('Star', 'star_radius') * u.R_sun

        self.star_rot_period = configParser.getfloat('Star', 'star_rot_period') * u.day
        self.star_rot_offset_from_orbital_plane = configParser.getfloat('Star','star_rot_offset_from_orbital_plane') * u.deg
        self.star_rot_offset_angle_from_pariapse = configParser.getfloat('Star','star_rot_offset_angle_from_pariapse') * u.deg

        self.psg_star_template = configParser.get('Star','psg_star_template')


        self.planet_name = configParser.get('Planet','planet_name')
        self.planet_initial_phase = configParser.getfloat('Planet','planet_initial_phase') * u.deg
        self.planet_init_substellar_lon = configParser.getfloat('Planet','planet_init_substellar_lon') * u.deg

        planet_radius_unit = configParser.get('Planet','planet_radius_unit')
        self.planet_radius = configParser.getfloat('Planet','planet_radius') * u.Unit(planet_radius_unit)

        planet_grav_mode = configParser.get('Planet','planet_mass_den_grav_mode')
        planet_grav_unit = u.Unit(configParser.get('Planet','planet_mass_den_grav_unit'))
        mode_parser = {
            'gravity': {'kw':'g','base_unit':u.Unit('m s-2')},
            'density': {'kw':'rho','base_unit':u.Unit('g cm-3')},
            'mass': {'kw':'kg','base_unit':u.Unit('kg')}
        }
        self.planet_grav_mode = mode_parser[planet_grav_mode]['kw']
        self.planet_grav = to_float(configParser.getfloat('Planet','planet_mass_den_grav') * planet_grav_unit,mode_parser[planet_grav_mode]['base_unit'])

        self.planet_semimajor_axis = configParser.getfloat('Planet','planet_semimajor_axis')*u.AU
        self.planet_orbital_period = configParser.getfloat('Planet','planet_orbital_period')*u.day
        self.planet_eccentricity = configParser.getfloat('Planet','planet_eccentricity')
        self.planet_rotational_period = configParser.getfloat('Planet','planet_rotational_period')*u.day
        self.planet_obliquity = configParser.getfloat('Planet','planet_obliquity')*u.deg
        self.planet_obliquity_direction = configParser.getfloat('Planet','planet_obliquity_direction')*u.deg

        if self.planet_obliquity != 0*u.deg:
            raise NotImplementedError('Currently non-zero obliquities are not supported. The Geometry is hard.')


        self.system_distance = configParser.getfloat('System','system_distance') * u.pc
        self.system_inclination = configParser.getfloat('System','system_inclination') * u.deg
        self.system_inclination_psg = 90*u.deg - self.system_inclination
        self.system_phase_of_periasteron = configParser.getfloat('System','system_phase_of_periasteron') * u.deg


        self.Nlat = configParser.getint('Model','map_Nlat')    
        self.Nlon = configParser.getint('Model','map_Nlon')
        gcm_path = configParser.get('Model','gcm_path')
        self.gcm_path = Path(filename).parent / gcm_path
        self.use_globes = configParser.getboolean('Model','use_globes')
        self.gcm_binning = configParser.getint('Model','gcm_binning')
        self.planet_phase_binning = configParser.getint('Model','planet_phase_binning')
        self.use_molec_signatures = configParser.getboolean('Model','use_molec_signatures')
        self.psg_url = configParser.get('Model','psg_url')
        try:
            self.api_key_path = configParser.get('PSG', 'api_key_path')
        except:
            self.api_key_path = None

        self.target_wavelength_unit = u.Unit(configParser.get('Observation','wavelength_unit'))
        self.target_flux_unit = u.Unit(configParser.get('Observation','flux_unit'))
        psg_rad_mapper = {u.Unit('W m-2 um-1'):'Wm2um'}
        self.psg_rad_unit = psg_rad_mapper[self.target_flux_unit]
        self.resolving_power = configParser.getfloat('Observation','resolving_power')
        self.lambda_min = configParser.getfloat('Observation','lambda_min') * self.target_wavelength_unit
        self.lambda_max = configParser.getfloat('Observation','lambda_max') * self.target_wavelength_unit

        image_integration_time_unit = u.Unit(configParser.get('Observation','image_integration_time_unit'))
        self.image_integration_time = configParser.getfloat('Observation','image_integration_time')*image_integration_time_unit

        total_observing_time_unit = u.Unit(configParser.get('Observation','total_observation_time_unit'))
        self.total_observation_time = configParser.getfloat('Observation','total_observation_time')*total_observing_time_unit


        # Noise
        self.detector_type = configParser.get('Observation','detector_type')
        self.detector_integration_time = configParser.getfloat('Observation','detector_integration_time')
        self.detector_pixel_sampling = configParser.getint('Observation','detector_pixel_sampling')
        self.detector_read_noise = configParser.getint('Observation','detector_read_noise')
        self.detector_dark_current = configParser.getint('Observation','detector_dark_current')
        self.detector_throughput = configParser.getfloat('Observation','detector_throughput')
        self.detector_emissivity = configParser.getfloat('Observation','detector_emissivity')
        self.detector_temperature = configParser.getfloat('Observation','detector_temperature')
        self.telescope_diameter = configParser.getfloat('Observation','telescope_diameter')

        self.total_images = int(round(float((self.total_observation_time/self.image_integration_time).to(u.Unit('')))))
        self.planet_images = int(round(float((self.total_observation_time
                            /(self.planet_phase_binning*self.image_integration_time)).to(u.Unit('')))))
        self.detector_number_of_integrations = int(round((self.image_integration_time/self.detector_integration_time/u.s).to(u.Unit('')).value))

        self.beamValue = configParser.getfloat('Observation', 'beamValue') # Beam value and unit used to also retrieve stellar flux values, not just planet
        self.beamUnit = configParser.get('Observation', 'beamUnit')
        # Some unit conversions
        self.distanceFluxCorrection = (self.star_radius.to(u.m)/self.system_distance.to(u.m)).to(u.Unit('')).value**2

        # Units are hard-fast, EDIT LATER
        self.cmTOum = 1e4
        self.cm2TOm2 = 1e-4
        self.erg_sTOwatts = 1e-7
        self.unit_conversion = (u.Unit('erg m-2 s-1 um-1')/u.Unit('W m-2 um-1')).to(u.Unit(''))
