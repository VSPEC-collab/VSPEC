import configparser
from astropy import units as u, constants as c
from pathlib import Path


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
        self.starName = configParser.get('Star', 'starName')
        self.spotCoverage = float(configParser.get('Star', 'spotCoverage'))
        self.facCoverage = float(configParser.get('Star', 'facCoverage'))

        # Star Properties
        self.teffStar = configParser.getint('PSG', 'starTemp')
        self.binningRange = int(configParser.get('Star', 'binningRange')) * u.K
        self.starRadius = float(configParser.get('Star', 'starRadius')) * u.R_sun
        self.starRadiusMeters = self.starRadius.to(u.m).value
        self.starDistance = float(configParser.get('Star', 'starDistance')) * u.pc
        self.starDistanceMeters = self.starDistance.to(u.m).value
        self.rotstar = float(configParser.get('Star', 'Rotstar')) * u.day

        #update this later
        self.Nlat = 500
        self.Nlon = 1000

        # Inclination of the star
        self.inclination = int(configParser.get('Star', 'Inclination')) * u.deg
        self.inclinationPSG = self.inclination + 90*u.deg
        self.offsetFromOrbitalPlane = int(configParser.get('Star','offsetFromOrbitalPlane')) * u.deg
        self.offsetDirection = int(configParser.get('Star','offsetDirection')) * u.deg

        # Planet periodic properties
        self.revPlanet = configParser.getfloat('PSG', 'revPlanet')*u.day
        # The planet rotation is equivalent to the planet revolution for tidally locked planets
        self.rotPlanet = self.revPlanet
        self.planetPhaseChange = configParser.getfloat('PSG', 'planetPhaseChange')*u.deg
        
        # Noise
        self.detector_type = configParser.get('PSG','detector_type')
        self.detector_integration_time = configParser.getfloat('PSG','detector_integration_time')
        self.detector_pixel_sampling = configParser.getint('PSG','detector_pixel_sampling')
        self.detector_read_noise = configParser.getint('PSG','detector_read_noise')
        self.detector_dark_current = configParser.getint('PSG','detector_dark_current')
        self.detector_throughput = configParser.getfloat('PSG','detector_throughput')
        self.detector_emissivity = configParser.getfloat('PSG','detector_emissivity')
        self.detector_temperature = configParser.getfloat('PSG','detector_temperature')
        self.telescope_diameter = configParser.getfloat('PSG','telescope_diameter')

        # Observation information
        self.time_between_exposures = int(configParser.get('HemiMap', 'time_between_exposures'))*u.min
        self.total_observation_time = int(configParser.get('HemiMap', 'observing_time'))*u.min
        self.total_images = int(round(float((self.total_observation_time/self.time_between_exposures).to(u.Unit('')))))

        # Binned spectrum
        self.loadData = configParser.getboolean('Star', 'loadData')
        self.lam1   = configParser.getfloat('PSG', 'lam1')         # Initial wavelength of the simulations (um)
        self.lam2   = configParser.getfloat('PSG', 'lam2')         # Final wavelength of the simulations (um)
        self.lamRP  = configParser.getfloat('PSG', 'lamRP')        # Resolving power
        self.target_wavelength_unit = u.Unit(configParser.get('Spectra','wavelengthUnit'))
        self.target_flux_unit = u.Unit(configParser.get('Spectra','desiredFluxUnit'))

        # PSG
        # Paramaters sent to PSG to retrieve planet spectra
        self.gcm_file_path = configParser.get('PSG', 'GCM')
        try:
            self.api_key_path = configParser.get('PSG', 'api_key_path')
        except:
            self.api_key_path = None
        self.planetName = configParser.get('PSG', 'planetName') # Name of planet, used for graphing/save name purposes
        self.initial_planet_phase = configParser.getfloat('PSG', 'phase1')         # Initial phase (degrees) for the simulation, 0 is sub-solar point, 180 is night-side
        self.binning= configParser.getfloat('PSG', 'binning')        # Binning applied to the GCM data for each radiative-transfer (greater is faster, minimum is 1)
        self.objDiam = configParser.getfloat('PSG', 'objDiam')       # Diamater of prox-cen b (km)
        self.objGrav = configParser.getfloat('PSG', 'objGrav')       # Surface Grav of prox cen b (m/s^2)
        self.starType = configParser.get('PSG', 'starType')         # Star type
        self.semMajAx = configParser.getfloat('PSG', 'semMajAx')   # Semi Major Axis of planet (AU)
        self.objPer = configParser.getfloat('PSG', 'objPer')       # Period of planet (days)
        self.objRev = self.objPer                             # planet revolution is equal to planet rev for tidally locked planets
        self.objEcc = configParser.getfloat('PSG', 'objEcc')       # Eccentricity of planet
        self.objArgOfPeriapsis = configParser.getfloat('PSG','objArgOfPeriapsis') * u.deg
        self.objDis = configParser.getfloat('PSG', 'objDis')       # Distance to system (uses distance to star) (pc)
        self.starRad = configParser.getfloat('PSG', 'starRad')     # Radius of the star
        self.beamValue = configParser.getfloat('PSG', 'beamValue') # Beam value and unit used to also retrieve stellar flux values, not just planet
        self.beamUnit = configParser.get('PSG', 'beamUnit')
        self.radunit = configParser.get('PSG', 'radunit')     # Desired spectral irradiance unit for planet and star combo
        self.psgurl = configParser.get('PSG', 'psgurl')       # URL of the PSG server

        self.PSGcombinedSpectraFolder = Path('.') / f'{self.starName}' / 'Data' / 'PSGCombinedSpectra'
        self.PSGthermalSpectraFolder = Path('.') / f'{self.starName}' / 'Data' / 'PSGThermalSpectra'
        self.PSGnoiseFolder = Path('.') / f'{self.starName}' / 'Data' / 'PSGNoise'
        self.PSGlayersFolder = Path('.') / f'{self.starName}' / 'Data' / 'PSGLayers'

        # Some unit conversions
        self.distanceFluxCorrection = (self.starRadiusMeters/self.starDistanceMeters)**2

        # Units are hard-fast, EDIT LATER
        self.cmTOum = 1e4
        self.cm2TOm2 = 1e-4
        self.erg_sTOwatts = 1e-7
        self.unit_conversion = (u.Unit('erg m-2 s-1 um-1')/u.Unit('W m-2 um-1')).to(u.Unit(''))

        # Plotting booleans; decide what plots to create during an execution of the program.
        self.plotLightCurve = configParser.getboolean('Plots', 'plotLightCurve')
        self.plotPlanetContrast = configParser.getboolean('Plots', 'plotPlanetContrast')
        self.plotPlanetVariationContrast = configParser.getboolean('Plots', 'plotPlanetVariationContrast')
        self.plotMaxFluxChange = configParser.getboolean('Plots', 'plotMaxFluxChange')
        self.plotStellarFlux = configParser.getboolean('Plots', 'plotStellarFlux')
        self.plotAdjustedPlanetFlux = configParser.getboolean('Plots', 'plotAdjustedPlanetFlux')
        self.plotPlanetFluxVariation = configParser.getboolean('Plots', 'plotPlanetFluxVariation')
        self.plotLightCurveHemiCombo = configParser.getboolean('Plots', 'plotLightCurveHemiMapCombo')
        self.plotLightCurveContrastCombo = configParser.getboolean('Plots', 'plotLightCurveContrastCombo')
        self.plotBinnedVariations = configParser.getboolean('Plots', 'plotBinnedVariations')
        
        self.makeGifs = configParser.getboolean('Gifs', 'makeGifs')