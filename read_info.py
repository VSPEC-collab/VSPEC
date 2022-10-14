import configparser
from contextlib import nullcontext
import h5py
import numpy as np
import math
import pandas as pd
from pathlib import Path
import time
from scipy.interpolate import interp2d
from astropy import units

class ReadStarModels():
    def __init__(self,
        starName,
        binnedWavelengthMin=None,
        binnedWavelengthMax=None,
        imageResolution=None,
        resolvingPower=None,
        filenames=None,
        topValues=None,
        cwValues=None
        ):

        self.starName = starName
        self.binnedWavelengthMin = binnedWavelengthMin
        self.binnedWavelengthMax = binnedWavelengthMax
        if binnedWavelengthMin:
            self.rangeMin = binnedWavelengthMin - (binnedWavelengthMin / 1000)
            self.rangeMax = binnedWavelengthMax + (binnedWavelengthMax / (binnedWavelengthMax * 1.5))
        self.ngridpoints = imageResolution
        self.resolvingPower = resolvingPower
        self.filenames = filenames
        self.topValues = topValues
        self.cwValues = cwValues
    # EDIT: Move Bin into it's own program file
    # Bin the photosphere, spot, and faculae stellar model flux values
    def bin(self, model):
        ultimaLam=0.0
        lastLam=0.0
        lam = 0.0
        ultimaVal=0.0
        lastVal=0.0
        val = 0.0
        summationVal=0.0
        numValues = 0
        binCount=0
        bottom = 0.0
        outputY = []
        outputYstring = []
        cwValuesString = []

        start_time = time.time()

        # Lambda index keeps track of which wavelength/flux value is being looked at in the lists
        lambdaIndex = 0
        # While there are still wavelengths to look at from the raw data, continue binning
        while lambdaIndex < len(model.wavelength):
            # lam is initially set to the first wavelength value in the raw data
            lam = model.wavelength.values[lambdaIndex]
            # val is initially set to the first flux value associated with the first wavelength value
            val = model.flux.values[lambdaIndex]

            # If lastLam is 0, this is the beginning of the binning process.


            if lastLam == 0:
                # To get things started, since lam is already the smallest wavelength value of the raw data,
                # lastLam is set to .000001 less than Lam.
                lastLam = lam - 1e-6
                # Then, ultimaLam (which is the variable for the wavelength value that is 2 positions
                # smaller than lam, and 1 position smaller than lastLam in the wavelength list) is set to
                # .000001 less then lastLam
                ultimaLam = lastLam - 1e-6
                # Since val is the first flux value, the lastVal (previous flux value) is simply set
                # to val
                lastVal = val
                # Then, ultimaVal (the flux value 2 positions before val in the flux list) is also
                # simply set to lastVal, which is set to val
                ultimaVal = lastVal
                summationVal=0.0
                numberOfValues=0
                # Initially, the bottom of the bin is set to the first center wavelength (.2 in my case)
                # minus the same amount as the difference between the cw and the bin's top value
                bottom = self.cwValues[0] - (self.topValues[0] - self.cwValues[0])
            # If lam (the current wavelength being looked at) is less than the bin's bottom (lower bound)
            # then a new bin gets created.
            # The meanVal of the bin and the number of values in the bin gets reset
            if lam < bottom:
                numberOfValues = 0
                summationVal = 0.0

            # Now the actual binning process
            # binCount keeps track fo which bin is being created, also how many center wavelengths
            # have been looked at since each bin has 1 center wavelength
            # The loop breaks when all center wavelengths (bins) have been examined
            # or if the current bin's top value is less than the lam being examined. If that happens,
            # it moves to the next bin
            while binCount < len(self.cwValues) and self.topValues[binCount] < lam:
                # These conditionals cover all of the possible scenarios while binning and how
                # to handle them

                # If there are more than 1 values that have been found to fit in the current bin,
                # calculate the mean of the bin's values
                if numberOfValues > 1:
                    temp = summationVal/numberOfValues
                    outputY.append(temp)
                    outputYstring.append("{:.7e}".format(temp))
                # The below conditions are if there were 1 or less values found to fit in the current bin

                # If the top value of the bin is greater than the previous lambda's (lastLam) flux
                # value, calculate an average flux value between the last flux value (inside the bin)
                # and the current flux value 
                elif self.topValues[binCount] > lastLam:
                    # Take the slope of the line between the current flux value (outside of bin) and
                    # the previous flux value (inside of bin), multiply it by the difference between
                    # the bin's center wavelength and the previous wavelength
                    temp = (val  - lastVal)*(self.cwValues[binCount] - lastLam)/( lam - lastLam) + lastVal
                    outputY.append(temp)
                    outputYstring.append("{:.7e}".format(temp))
                elif self.topValues[binCount] > ultimaLam:
                    temp = (lastVal - ultimaVal) * (self.cwValues[binCount] - ultimaLam) / (lastLam - ultimaLam) + ultimaVal
                    outputY.append(temp)
                    outputYstring.append("{:.7e}".format(temp))
                else:
                    outputY.append(ultimaVal)
                    outputYstring.append("{:.7e}".format(ultimaVal))
                bottom = self.topValues[binCount]
                cwValuesString.append("{:.9e}".format(self.cwValues[binCount]))
                binCount += 1
                numberOfValues = 0
                summationVal = 0.0

            ultimaLam = lastLam
            lastLam = lam
            ultimaVal = lastVal
            lastVal = val
            summationVal += lastVal
            numberOfValues += 1
            lambdaIndex += 1

            percent = (lambdaIndex/len(model.wavelength)) * 100
            # print(percent)
            if  percent % 1 == 0:
                print("%.1f" % percent + "%" + "Complete")

        print("Time after binning")
        print("--- %s seconds ---" % (time.time() - start_time))
        model = pd.DataFrame(list(zip(self.cwValues, outputY)), columns =['wavelength', 'flux'])
        modelStrings = pd.DataFrame(list(zip(cwValuesString, outputYstring)), columns =['wavelength', 'flux'])

        return model, modelStrings

    def read_model(self, Teffs, loadData = False):
        
        modelStrings = None

        # # Works for reading ASCII files
        # model = pd.read_csv(
        #     filename, skiprows=7, delim_whitespace=True, names=["wavelength", "flux"]
        # )

        # Works for reading h5 files
        # print('Reading File: <'+self.photModelFile+'>')
        print('Model Teffs to load: ', Teffs)
        for Teff in Teffs:
            filename = Path('.') / 'NextGenModels' / 'RawData' / f'lte0{Teff:.0f}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
            fh5 = h5py.File(filename,'r')
            wl = fh5['PHOENIX_SPECTRUM/wl'][()]
            fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()]
            data = pd.DataFrame({'wavelength': wl, 'flux': fl})
            data.wavelength /= 1.0e4
            data = data.loc[
            (data.wavelength >= self.rangeMin) & (data.wavelength <= self.rangeMax)
        ]
            print(f"\nBinning Teff = {Teff} Data to Specified Resolving Power...")
            print("----------------------------------------------------------")
            photModel, photModelStrings = self.bin(data)
            binned_path = Path('.') / 'NextGenModels' / 'BinnedData'
            if not binned_path.exists():
                binned_path.mkdir()
            binnedPhotStringCSV = photModelStrings.to_csv(index=False, header=['WAVELENGTH (MICRONS)','FLUX (ERG/CM2/S/A)'], sep=' ')
            file_to_open = Path('.') / 'NextGenModels' / 'BinnedData' / f'binned{Teff}StellarModel.txt'
            file = open(file_to_open,'w')
            file.write(binnedPhotStringCSV)
            file.close()
        return self

class ParamModel():
    
    def calculate_observation_parameters(self, observation_param_dict):
        # This scenario is if the user has elected to image the planet every 'n' minutes
        if 'time_between_exposures' in observation_param_dict:
            delta_time = observation_param_dict['time_between_exposures']
            
            # Convert minutes to days
            self.delta_time = delta_time * (1/ 60) * (1/24)
            
            # The delta_phase_planet between images will be calculated based on the rotation of the planet
            # given the amount of time between exposures.

            self.delta_phase_planet = float((self.delta_time / self.rotPlanet) * 360)
            print(f'delta_time = {self.delta_time}')
            print(f'rotPlanet = {self.rotPlanet}')
            print("delta_phase_planet = ", self.delta_phase_planet)
            
            self.delta_phase_star = float((self.delta_time / self.rotstar) * 360)
        
        # This scenario is if the uer has elected to image the planet every 'n' degrees of its rotation
        elif 'image_per_degree_rotation_planet' in observation_param_dict:
            degrees_rotate = observation_param_dict['image_per_degree_rotation_planet']
            
            self.delta_phase_planet = float(degrees_rotate)
            print(f"Image every {degrees_rotate} planet rotation degrees")
            
            # First, take the percentage of the planet delta phase out of 360 and apply it to the planet's rotation period.
            # This gives the number of days the planet has taken to rotate delta_phase_planet degrees.
            # Take that number of days and apply it to the star's rotation period to get the percent that these days take up.
            # Multiply that by 360 degrees to see how many degrees the star rotates during the same time period as the
            # delta_planet_phase.
            delta_planet_phase_in_days = (self.delta_phase_planet / 360) * self.rotPlanet
            self.delta_phase_star = float((delta_planet_phase_in_days / self.rotstar) * 360)
            self.delta_time = self.delta_phase_planet / 360.0 * self.rotPlanet
            
            print('done')
        
        # This scenario is if the uer has elected to image the planet every 'n' degrees of the star's
        # rotation    
        elif 'image_per_degree_rotation_star' in observation_param_dict:
            degrees_rotate = observation_param_dict['image_per_degree_rotation_star']
            
            self.delta_phase_star = float(degrees_rotate)
            print(f"Image every {degrees_rotate} star rotation degrees")
            
            # First, take the percentage of the star delta phase out of 360 and apply it to the star's rotation period.
            # This gives the number of days the star has taken to rotate delta_phase_star degrees.
            # Take that number of days and apply it to the planet's rotation period to get the percent that these days take up.
            # Multiply that by 360 degrees to see how many degrees the planet rotates during the same time period as the
            # delta_planet_star.
            delta_star_phase_in_days = (self.delta_phase_star / 360) * self.rotstar
            self.delta_phase_planet = float((delta_star_phase_in_days / self.rotPlanet) * 360)
            self.delta_time = self.delta_phase_star / 360.0 * self.rotstar


            print('done')
            
        if 'observing_time' in observation_param_dict:
            total_time = observation_param_dict['observing_time'] * (1/60) * (1/24)
            
            if 'time_between_exposures' in observation_param_dict:
                total_images = math.floor(total_time / self.delta_time)
                print("TOTAL IMAGES = ", total_images)

            elif 'image_per_degree_rotation_planet' in observation_param_dict:
                total_images = math.floor(total_time / ((self.delta_phase_planet / 360) * self.rotPlanet))
                print("TOTAL IMAGES = ", total_images)
                
            elif 'image_per_degree_rotation_star' in observation_param_dict:
                total_images = math.floor(total_time / ((self.delta_phase_star / 360) * self.rotstar))
                print("TOTAL IMAGES = ", total_images)
            
            self.total_images = total_images
            print(f'total_image = {self.total_images}')
        elif 'num_planet_rotations' in observation_param_dict:
            total_rotations = observation_param_dict['num_planet_rotations']
            total_images = math.floor((360 / self.delta_phase_planet) * total_rotations)
            self.total_images = total_images + 1
        
        elif 'num_star_rotations' in observation_param_dict:
            total_rotations = observation_param_dict['num_star_rotations']
            total_images = math.floor((360 / self.delta_phase_star) * total_rotations)
            self.total_images = total_images + 1
        self.detector_number_of_integrations = self.delta_time * 24*60*60/self.detector_integration_time

    def __init__(self):
        configParser = configparser.RawConfigParser()
        while True:
            try:
                fileName = input("Config File Name (Located in the Config/Stellar Folder): ")
                p = Path('.')
                # file_to_open = p / 'Configs' / 'Stellar' / f'{fileName}.cfg'
                file_to_open = Path('.') / 'Configs' / 'Stellar' / f'{fileName}.cfg'
                configParser.read_file(open(file_to_open))
                break
            except FileNotFoundError:
                print("There is no config file by that name, please try again. (Note: Do not include .cfg)")
        
        # Read in the information of the star from the config file
        self.starName = configParser.get('Star', 'starName')

        self.spotCoverage = float(configParser.get('Star', 'spotCoverage'))
        
        self.facCoverage = float(configParser.get('Star', 'facCoverage'))
        # Load in 
        self.defaultModelType = configParser.getboolean('Star', 'defaultModelType')

        # Temperature values for the star, spots, and faculae
        self.teffStar = int(configParser.get('Star', 'teffStar'))
        self.binningRange = int(configParser.get('Star', 'binningRange'))

        self.starRadius = float(configParser.get('Star', 'starRadius'))
        self.starRadiusMeters = self.starRadius * 6.957e8

        self.starDistance = float(configParser.get('Star', 'starDistance'))
        self.starDistanceMeters = self.starDistance * 3.086e16

        # Rotation of star in days
        self.rotstar = float(configParser.get('Star', 'Rotstar'))

        # Inclination of the star
        self.inclination = int(configParser.get('Star', 'Inclination'))
        self.inclinationPSG = self.inclination + 90

        # Total number of exposures to be takn
        # self.total_images = int(configParser.get('HemiMap', 'total_images'))
        
        # self.final_stellar_phase = int(configParser.get('HemiMap', 'final_stellar_phase'))
        
        self.revPlanet = configParser.getfloat('PSG', 'revPlanet')
        # The planet rotation is equivalent to the planet revolution for tidally locked planets
        self.rotPlanet = self.revPlanet
        self.planetPhaseChange = configParser.getfloat('PSG', 'planetPhaseChange')

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

        self.observation_style = str(configParser.get('HemiMap', 'Chosen_Observation_Style'))
        if self.observation_style == 'time_between_exposures':
            self.observation_param_dict = {'time_between_exposures':int(configParser.get('HemiMap', 'time_between_exposures'))}
        elif self.observation_style == 'image_per_degree_rotation_planet':
            self.observation_param_dict = {'image_per_degree_rotation_planet':int(configParser.get('HemiMap', 'image_per_degree_rotation_planet'))}
        elif self.observation_style == 'image_per_degree_rotation_star':
            self.observation_param_dict = {'image_per_degree_rotation_star':int(configParser.get('HemiMap', 'image_per_degree_rotation_star'))}
        
        self.observation_length = str(configParser.get('HemiMap', 'Chosen_Observation_Length'))
        if self.observation_length == 'observing_time':
            self.observation_param_dict['observing_time'] = int(configParser.get('HemiMap', 'observing_time'))
        elif self.observation_length == 'num_planet_rotations':
            self.observation_param_dict['num_planet_rotations'] = int(configParser.get('HemiMap', 'num_planet_rotations'))
        elif self.observation_length == 'num_star_rotations':
            self.observation_param_dict['num_star_rotations'] = int(configParser.get('HemiMap', 'num_star_rotations'))
        
        # Based on user input, this method will look at the chosen observation style and calculate how many
        # degrees the planet and star will rotate per image.
        # Then, also based on the user's input, this method will look at the observation length
        # and calculate how many images will be taken.
        self.calculate_observation_parameters(self.observation_param_dict)

        # # Time (in minutes) between exposures. Converted to days.
        # self.time_between_exposures = float(configParser.get('HemiMap', 'time_between_exposures'))

        # # Calculates what percent the time between exposures is compared to the stellar rotation period
        # # Used to calculate the change in phase between images 
        # self.exposure_time_percent = self.time_between_exposures * (100 / self.rotstar)
        # # print("Exposure Time percent = ", self.exposure_time_percent)
        # self.deltaPhase = self.exposure_time_percent / 100
        # # print("Delta Phase = ", self.deltaPhase)
        # self.deltaStellarPhase = self.deltaPhase * 360

        # Load in the NextGen Stellar Info
        if self.defaultModelType:
            self.teffs = 100*np.arange(np.floor(self.teffStar/100 - self.binningRange/100),
                                np.ceil(self.teffStar/100 + self.binningRange/100)+1)
            self.model_files = [Path('.') / 'NextGenModels' / 'RawData' / f'lte0{t:.0f}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5' for t in self.teffs]
        else:
            raise NotImplementedError
            # phot_model_file = configParser.get('Star', 'phot_model_file')
            # self.phot_model_file = phot_model_file.strip('"') # configParser adds extra "" that I remove
            # spot_model_file = configParser.get('Star', 'spot_model_file')
            # self.spot_model_file = spot_model_file.strip('"')
            # fac_model_file = configParser.get('Star', 'fac_model_file')
            # self.fac_model_file = fac_model_file.strip('"')

        # Boolean which says whether or not the user wishes to bin the data to a certain resolving power
        self.binData = configParser.getboolean('Star', 'binData')
        self.loadData = configParser.getboolean('Star', 'loadData')
        
        # The resolving power to bin to
        self.resolvingPower = int(configParser.get('Star', 'resolvingPower'))
        self.binnedWavelengthMin = float(configParser.get('Star', 'binnedWavelengthMin'))
        self.binnedWavelengthMax = float(configParser.get('Star', 'binnedWavelengthMax'))


        # Paramaters sent to PSG to retrieve planet spectra
        self.planetName = configParser.get('PSG', 'planetName') # Name of planet, used for graphing/save name purposes
        # self.noStar = configParser.getboolean('PSG', 'noStar')  # Set to true if you want to retrieve strictly the planet thermal values
        self.phase1 = configParser.getfloat('PSG', 'phase1')         # Initial phase (degrees) for the simulation, 0 is sub-solar point, 180 is night-side
        self.phase2 = configParser.getfloat('PSG', 'phase2')         # Final phase (degrees)
        self.binning= configParser.getfloat('PSG', 'binning')        # Binning applied to the GCM data for each radiative-transfer (greater is faster, minimum is 1)
        self.objDiam = configParser.getfloat('PSG', 'objDiam')       # Diamater of prox-cen b (km)
        self.objGrav = configParser.getfloat('PSG', 'objGrav')       # Surface Grav of prox cen b (m/s^2)
        self.starType = configParser.get('PSG', 'starType')         # Star type
        self.semMajAx = configParser.getfloat('PSG', 'semMajAx')   # Semi Major Axis of planet (AU)
        self.objPer = configParser.getfloat('PSG', 'objPer')       # Period of planet (days)
        self.objRev = self.objPer                             # planet revolution is equal to planet rev for tidally locked planets
        self.objEcc = configParser.getfloat('PSG', 'objEcc')       # Eccentricity of planet
        self.objDis = configParser.getfloat('PSG', 'objDis')       # Distance to system (uses distance to star) (pc)
        self.starTemp = configParser.getint('PSG', 'starTemp')   # Temperature of star; ProxCen's temp is really 3042, but need to use 3000 for later conversion
        self.starRad = configParser.getfloat('PSG', 'starRad')     # Radius of the star
        self.lam1   = configParser.getfloat('PSG', 'lam1')         # Initial wavelength of the simulations (um)
        self.lam2   = configParser.getfloat('PSG', 'lam2')         # Final wavelength of the simulations (um)
        self.lamRP  = configParser.getfloat('PSG', 'lamRP')        # Resolving power
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
        self.unit_conversion = (units.Unit('erg m-2 s-1 um-1')/units.Unit('W m-2 um-1')).to(units.Unit(''))

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