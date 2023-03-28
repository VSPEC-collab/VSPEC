"""VSPEC read info module

Read and parse parameters from a configuration
file so they can be used in a `VSPEC` run.
"""
import configparser
from configparser import NoOptionError
from pathlib import Path
from astropy import units as u

from VSPEC.helpers import to_float, MSH

class ParamModel:
    """Parameter Model

    Class to read and store information from the configuration file.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the configuration file.

    Attributes
    ----------
    star_name : str
        The name of the star.
    star_teff : astropy.units.Quantity [temperature]
        The effective temperature of the star.
    star_teff_min : astropy.units.Quantity [temperature]
        The minimum effective temperature model to be binned.
    star_teff_max : astropy.units.Quantity [temperature]
        The maximum effective temperature model to be binned.
    self.ld_a1 : float
        Coefficient a1 for limb darkening of the star.
    self.ld_a2 : float
        Coefficient a2 for limb darkening of the star.
    self.star_spot_initial_coverage : float
        The initial fractional coverage of the star's surface by spots.
    self.star_spot_distribution : str
        The distribution function to be used for the spot positions. 'iso' or 'solar'.
    self.star_spot_mean_area : astropy.units.Quantity [area]
        The mean area of a spot on the star's surface.
    self.star_spot_sigma_area : float
        The standard deviation of the spot areas. This is a lognormal
        distribution, so the units of this value are dex
    self.star_spot_umbra_teff : astropy.units.Quantity [temperature]
        The effective temperature of the spot umbrae.
    self.star_spot_penumbra_teff : astropy.units.Quantity [temperature]
        The effective temperature of the spot penumbrae.
    self.star_spot_growth_rate : astropy.units.Quantity [frequency]
        The rate at which new spots grow.
    self.star_spot_decay_rate : astropy.units.Quantity [area per time]
        The rate at which existing spots decay.
    self.star_spot_initial_area : astropy.units.Quantity [area]
        The initial area of newly created spots.
    self.star_spot_coverage : float
        The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equillibrium, and different from the 'hot start' value given
        by `star_spot_initial_coverage`.
    self.star_spot_warmup : astropy.units.Quantity [time]
        The duration of the warmup period, during which the spot coverage approaches
        equillibrium.
    star_fac_coverage : float
        The fraction of the star's surface covered by the faculae at growth-decay equillibrium
    star_fac_mean_radius : astropy.units.quantity.Quantity [distance]
        The mean radius of the faculae.
    star_fac_HWHM_radius : astropy.units.quantity.Quantity [distance]
        The half-width at half-maximum radius of the faculae. Difference
        between the peak of the radius distribution and the half maximum
        in the positive direction.
    star_fac_mean_timescale : astropy.units.quantity.Quantity [time]
        The mean faculae lifetime.
    star_fac_HWHM_timescale : astropy.units.quantity.Quantity [time]
        The facula timescale distribution half-width-half-maximum in hr.
        Difference between the peak of the timescale distribution and the
        half maximum in the positive direction.
    star_fac_distribution : str
        The distribution used to generate the faculae on the star. Currently
        only 'iso' is supported
    star_fac_warmup : astropy.units.Quantity [time]
        The warmup time for the faculae on the star to reach equillibrium.
    star_flare_group_prob : float
        The probability that a given flare will be closely followed by another flare
    star_flare_mean_teff : astropy.units.quantity.Quantity [temperature]
        The mean temperature of the flare blackbody.
    star_flare_sigma_teff : astropy.units.quantity.Quantity [temperature]
        The standard deviation of the generated flare temperature.
    star_flare_mean_log_fwhm_days : float
        The mean logarithm of the full width at half maximum (FWHM) of the flare in days.
    star_flare_sigma_log_fwhm_days : float
        The standard deviation of the logarithm of the FWHM of the flare in days.
    star_flare_log_E_erg_max : float
        Log of the maximum energy flares to be considered in ergs.
    star_flare_log_E_erg_min : float
        Log of the minimum energy flares to be considered in ergs.
    star_flare_log_E_erg_Nsteps : int
        The number of flare energy steps to consider.
    star_mass : astropy.units.Quantity [mass]
        The mass of the star.
    star_radius : astropy.units.Quantity [distance]
        The radius of the star.
    star_rot_period : astropy.units.Quantity [time]
        Rotational period of the star.
    star_rot_offset_from_orbital_plane : astropy.units.Quantity [angle]
        Angle between the rotation axis of the star and the vector
        normal to the orbital plane.
    star_rot_offset_angle_from_pariapse : astropy.units.Quantity
        Angle between the projection of the rotational axis onto the
        orbital plane and the line between the star and planet at periasteron.
    psg_star_template : str
        Keyword describing the stellar spectrum for PSG to generate.
        E.g. 'A', 'G', 'K', 'M'.
    planet_name : str
        The name of the planet for PSG to use internally. This will
        affect `.rad` files, but no `VSPEC` output.
    planet_initial_phase : astropy.units.Quantity [angle]
        The initial phase of the planet's orbit.
    planet_init_substellar_lon :astropy.units.Quantity [angle]
        The initial longitude of the substellar point on the planet.
    planet_radius :  astropy.units.Quantity [angle]
        The radius of the planet
    planet_grav_mode : str
        The method to pass the planet gravity to PSG. One of 
        'g' -- gravity method, 'rho' -- density method, or
        'kg' -- mass method.
    planet_grav : float
        The planet gravity parameter to pass to PSG
    planet_semimajor_axis : astropy.units.quantity.Quantity [distance]
        The semi-major axis of the planet's orbit.
    planet_orbital_period : astropy.units.quantity.Quantity [time]
        The orbital period of the planet.
    planet_eccentricity : float
        The eccentricity of the planet's orbit.
    planet_rotational_period : astropy.units.quantity.Quantity [time]
        The rotational period of the planet.
    planet_obliquity : astropy.units.quantity.Quantity [angle]
        The obliquity (tilt) of the planet's rotation axis.
    planet_obliquity_direction : astropy.units.quantity.Quantity [angle]
        The true anomaly at which the planet's north pole faces away from the star.
    system_distance : astropy.units.quantity.Quantity [distance]
        The distance to the system.
    system_inclination : astropy.units.quantity.Quantity [angle]
        The inclination angle of the system. Transit occurs at 90 deg.
    system_phase_of_periasteron : astropy.units.quantity.Quantity [angle]
        The phase (as seen from the observer) of the planet when it reaches periasteron.
    Nlat : int
        Number of latitudes in the stellar surface.
    Nlon : int
        Number of longitudes in the stellar surface.
    gcm_path : pathlib.Path
        Path to the GCM data.
    use_globes : bool
        Whether to use GlobES or not.
    gcm_binning : int
        Number of spacial points to bin together in the GCM data.
        Use 3 for science.
    planet_phase_binning : int
        Number of phase epochs to bin together when simulating the planet.
        These are later interpolated to match the cadence of the variable
        star simulation.
    use_molec_signatures : bool
        Whether to use molecular signatures (PSG atmosphere) or not.
    psg_url : str
        URL of the Planetary Spectrum Generator.
    api_key_path : str or None
        Path to a file containing your own personal PSG API key. Keep this key private
        and do not commit it to any Git repository. None if you plan to run PSG
        locally.
    target_wavelength_unit : astropy.units.Quantity [flambda]
        The wavelength unit of the output.
    target_flux_unit : astropy.units.Quantity
        The flux unit of the output.
    psg_rad_unit : str
        The PSG-specific flux unit keyword.
    resolving_power : float
        The resolving power of the observation.
    lambda_min : astropy.units.Quantity [wavelength]
        The minimum wavelength of the observation.
    lambda_max : astropy.units.Quantity [wavelength]
        The maximum wavelength of the observation.
    image_integration_time : astropy.units.quantity.Quantity [time]
        The integration time of each epoch of observation
    total_observation_time : astropy.units.quantity.Quantity [time]
        The total duration of the observation.
    detector_type : str
        Type of detector used, passed to PSG.
    detector_integration_time : float
        Integration time of the detector is seconds.
    detector_pixel_sampling : int
        Pixel sampling of the detector.
    detector_read_noise : astropy.units.quantity.Quantity
        Read noise of the detector in electrons.
    detector_dark_current : astropy.units.quantity.Quantity
        Dark current of the detector in electrons/second.
    detector_throughput : float
        Throughput of the detector.
    detector_emissivity : float
        Emissivity of the detector.
    detector_temperature : float
        Temperature of the detector in K.
    telescope_diameter : float
        Diameter of the telescope in meters.
    total_images : int
        Total number of images to be taken during the observation.
    planet_images : int
        Number of epochs to simulate the planet.
    detector_number_of_integrations : int
        Number of detector integrations per image.
    beamValue : float
        Beam value used to retrieve stellar flux values, as well as planet.
    beamUnit : str
        Unit of beam value.
    distanceFluxCorrection : float
        Correction factor for distance from the star.
    cmTOum : float
        Conversion factor from centimeters to micrometers.
    cm2TOm2 : float
        Conversion factor from square centimeters to square meters.
    erg_sTOwatts : float
        Conversion factor from erg per second to watts.
    unit_conversion : float
        Unit conversion factor for flux units.

    Raises
    ------
    NotImplementedError
        If obliquity is not 0 deg.
    """

    def __init__(self, filename):
        configParser = configparser.RawConfigParser()
        configParser.read_file(open(filename, encoding='UTF-8'))
        # Read in the information of the star from the config file
        self.star_name = configParser.get('Star', 'star_name')

        # Star Properties
        self.star_teff = configParser.getint('Star', 'star_teff') * u.K
        self.star_teff_min = configParser.getint('Star', 'star_teff_min') * u.K
        self.star_teff_max = configParser.getint('Star', 'star_teff_max') * u.K

        self.ld_a1 = configParser.getfloat('Star', 'limb_darkening_a1')
        self.ld_a2 = configParser.getfloat('Star', 'limb_darkening_a2')

        self.star_spot_initial_coverage = configParser.getfloat(
            'Star', 'star_spot_initial_coverage')
        self.star_spot_distribution = configParser.get(
            'Star', 'star_spot_distribution')
        self.star_spot_mean_area = configParser.getfloat(
            'Star', 'star_spot_mean_area') * MSH
        self.star_spot_sigma_area = configParser.getfloat(
            'Star', 'star_spot_sigma_area')
        self.star_spot_umbra_teff = configParser.getfloat(
            'Star', 'star_spot_umbra_teff')*u.K
        self.star_spot_penumbra_teff = configParser.getfloat(
            'Star', 'star_spot_penumbra_teff')*u.K
        self.star_spot_growth_rate = configParser.getfloat(
            'Star', 'star_spot_growth_rate') / u.day
        self.star_spot_decay_rate = configParser.getfloat(
            'Star', 'star_spot_decay_rate') * MSH / u.day
        self.star_spot_initial_area = configParser.getfloat(
            'Star', 'star_spot_initial_area') * MSH
        self.star_spot_coverage = float(
            configParser.get('Star', 'star_spot_coverage'))
        self.star_spot_warmup = configParser.getfloat(
            'Star', 'star_spot_warmup') * u.day

        self.star_fac_coverage = float(
            configParser.get('Star', 'star_fac_coverage'))
        self.star_fac_mean_radius = configParser.getfloat(
            'Star', 'star_fac_mean_radius') * u.km
        self.star_fac_HWHM_radius = configParser.getfloat(
            'Star', 'star_fac_HWMH_radius') * u.km
        self.star_fac_mean_timescale = configParser.getfloat(
            'Star', 'star_fac_mean_timescale') * u.hr
        self.star_fac_HWHM_timescale = configParser.getfloat(
            'Star', 'star_fac_HWMH_timescale') * u.hr
        self.star_fac_distribution = configParser.get(
            'Star', 'star_fac_distribution')
        self.star_fac_warmup = configParser.getfloat(
            'Star', 'star_fac_warmup') * u.hr

        self.star_flare_group_prob = configParser.getfloat(
            'Star', 'star_flare_group_prob')
        self.star_flare_mean_teff = configParser.getfloat(
            'Star', 'star_flare_mean_teff') * u.K
        self.star_flare_sigma_teff = configParser.getfloat(
            'Star', 'star_flare_sigma_teff') * u.K
        self.star_flare_mean_log_fwhm_days = configParser.getfloat(
            'Star', 'star_flare_mean_log_fwhm_days')
        self.star_flare_sigma_log_fwhm_days = configParser.getfloat(
            'Star', 'star_flare_sigma_log_fwhm_days')
        self.star_flare_log_E_erg_max = configParser.getfloat(
            'Star', 'star_flare_log_E_erg_max')
        self.star_flare_log_E_erg_min = configParser.getfloat(
            'Star', 'star_flare_log_E_erg_min')
        self.star_flare_log_E_erg_Nsteps = configParser.getint(
            'Star', 'star_flare_log_E_erg_Nsteps')

        self.star_mass = configParser.getfloat('Star', 'star_mass') * u.M_sun
        self.star_radius = configParser.getfloat(
            'Star', 'star_radius') * u.R_sun

        self.star_rot_period = configParser.getfloat(
            'Star', 'star_rot_period') * u.day
        self.star_rot_offset_from_orbital_plane = configParser.getfloat(
            'Star', 'star_rot_offset_from_orbital_plane') * u.deg
        self.star_rot_offset_angle_from_pariapse = configParser.getfloat(
            'Star', 'star_rot_offset_angle_from_pariapse') * u.deg

        self.psg_star_template = configParser.get('Star', 'psg_star_template')
        self.planet_name = configParser.get('Planet', 'planet_name')

        self.planet_initial_phase = configParser.getfloat(
            'Planet', 'planet_initial_phase') * u.deg
        self.planet_init_substellar_lon = configParser.getfloat(
            'Planet', 'planet_init_substellar_lon') * u.deg

        planet_radius_unit = configParser.get('Planet', 'planet_radius_unit')
        self.planet_radius = configParser.getfloat(
            'Planet', 'planet_radius') * u.Unit(planet_radius_unit)

        planet_grav_mode = configParser.get(
            'Planet', 'planet_mass_den_grav_mode')
        planet_grav_unit = u.Unit(configParser.get(
            'Planet', 'planet_mass_den_grav_unit'))
        mode_parser = {
            'gravity': {'kw': 'g', 'base_unit': u.Unit('m s-2')},
            'density': {'kw': 'rho', 'base_unit': u.Unit('g cm-3')},
            'mass': {'kw': 'kg', 'base_unit': u.Unit('kg')}
        }
        self.planet_grav_mode = mode_parser[planet_grav_mode]['kw']
        self.planet_grav = to_float(configParser.getfloat(
            'Planet', 'planet_mass_den_grav') * planet_grav_unit, mode_parser[planet_grav_mode]['base_unit'])

        self.planet_semimajor_axis = configParser.getfloat(
            'Planet', 'planet_semimajor_axis')*u.AU
        self.planet_orbital_period = configParser.getfloat(
            'Planet', 'planet_orbital_period')*u.day
        self.planet_eccentricity = configParser.getfloat(
            'Planet', 'planet_eccentricity')
        self.planet_rotational_period = configParser.getfloat(
            'Planet', 'planet_rotational_period')*u.day
        self.planet_obliquity = configParser.getfloat(
            'Planet', 'planet_obliquity')*u.deg
        self.planet_obliquity_direction = configParser.getfloat(
            'Planet', 'planet_obliquity_direction')*u.deg

        if self.planet_obliquity != 0*u.deg:
            raise NotImplementedError(
                'Currently non-zero obliquities are not supported. The Geometry is hard.')

        self.system_distance = configParser.getfloat(
            'System', 'system_distance') * u.pc
        self.system_inclination = configParser.getfloat(
            'System', 'system_inclination') * u.deg
        self.system_phase_of_periasteron = configParser.getfloat(
            'System', 'system_phase_of_periasteron') * u.deg

        self.Nlat = configParser.getint('Model', 'map_Nlat')
        self.Nlon = configParser.getint('Model', 'map_Nlon')
        gcm_path = configParser.get('Model', 'gcm_path')
        self.gcm_path = Path(filename).parent / gcm_path
        self.use_globes = configParser.getboolean('Model', 'use_globes')
        self.gcm_binning = configParser.getint('Model', 'gcm_binning')
        self.planet_phase_binning = configParser.getint(
            'Model', 'planet_phase_binning')
        self.use_molec_signatures = configParser.getboolean(
            'Model', 'use_molec_signatures')
        self.psg_url = configParser.get('Model', 'psg_url')
        try:
            self.api_key_path = configParser.get('Model', 'api_key_path')
        except NoOptionError:
            self.api_key_path = None

        self.target_wavelength_unit = u.Unit(
            configParser.get('Observation', 'wavelength_unit'))
        self.target_flux_unit = u.Unit(
            configParser.get('Observation', 'flux_unit'))
        psg_rad_mapper = {u.Unit('W m-2 um-1'): 'Wm2um'}
        self.psg_rad_unit = psg_rad_mapper[self.target_flux_unit]
        self.resolving_power = configParser.getfloat(
            'Observation', 'resolving_power')
        self.lambda_min = configParser.getfloat(
            'Observation', 'lambda_min') * self.target_wavelength_unit
        self.lambda_max = configParser.getfloat(
            'Observation', 'lambda_max') * self.target_wavelength_unit

        image_integration_time_unit = u.Unit(configParser.get(
            'Observation', 'image_integration_time_unit'))
        self.image_integration_time = configParser.getfloat(
            'Observation', 'image_integration_time')*image_integration_time_unit

        total_observing_time_unit = u.Unit(configParser.get(
            'Observation', 'total_observation_time_unit'))
        self.total_observation_time = configParser.getfloat(
            'Observation', 'total_observation_time')*total_observing_time_unit

        # Noise
        self.detector_type = configParser.get('Observation', 'detector_type')
        self.detector_integration_time = configParser.getfloat(
            'Observation', 'detector_integration_time')
        self.detector_pixel_sampling = configParser.getint(
            'Observation', 'detector_pixel_sampling')
        self.detector_read_noise = configParser.getint(
            'Observation', 'detector_read_noise')
        self.detector_dark_current = configParser.getint(
            'Observation', 'detector_dark_current')
        self.detector_throughput = configParser.getfloat(
            'Observation', 'detector_throughput')
        self.detector_emissivity = configParser.getfloat(
            'Observation', 'detector_emissivity')
        self.detector_temperature = configParser.getfloat(
            'Observation', 'detector_temperature')
        self.telescope_diameter = configParser.getfloat(
            'Observation', 'telescope_diameter')

        self.total_images = int(round(
            float((self.total_observation_time/self.image_integration_time).to(u.Unit('')))))
        self.planet_images = int(round(float((self.total_observation_time
                                              / (self.planet_phase_binning*self.image_integration_time)).to(u.Unit('')))))
        self.detector_number_of_integrations = int(round(
            (self.image_integration_time/self.detector_integration_time/u.s).to(u.Unit('')).value))

        # Beam value and unit used to also retrieve stellar flux values, not just planet
        self.beamValue = configParser.getfloat('Observation', 'beamValue')
        self.beamUnit = configParser.get('Observation', 'beamUnit')
        # Some unit conversions
        self.distanceFluxCorrection = (self.star_radius.to(
            u.m)/self.system_distance.to(u.m)).to(u.Unit('')).value**2

        # Units are hard-fast, EDIT LATER
        self.cmTOum = 1e4
        self.cm2TOm2 = 1e-4
        self.erg_sTOwatts = 1e-7
        self.unit_conversion = (
            u.Unit('erg m-2 s-1 um-1')/u.Unit('W m-2 um-1')).to(u.Unit(''))
