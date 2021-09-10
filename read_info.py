import configparser
import h5py
import pandas as pd
import time

class ReadStarModels():
    def __init__(self,
        starName,
        binnedWavelengthMin,
        binnedWavelengthMax,
        imageResolution,
        resolvingPower,
        topValues,
        cwValues,
        photModelFile,
        spotModelFile,
        facModelFile,
        ):

        self.starName = starName
        self.binnedWavelengthMin = binnedWavelengthMin
        self.binnedWavelengthMax = binnedWavelengthMax
        self.rangeMin = binnedWavelengthMin - (binnedWavelengthMin / 1000)
        self.rangeMax = binnedWavelengthMax + (binnedWavelengthMax / (binnedWavelengthMax * 1.5))
        self.ngridpoints = imageResolution
        self.resolvingPower = resolvingPower
        self.topValues = topValues
        self.cwValues = cwValues
        self.photModelFile = photModelFile
        self.spotModelFile = spotModelFile
        self.facModelFile = facModelFile

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
                    print("lam = ", lam)
                    print("lastLam = ", lastLam)
                    print("val = ", val)
                    print("lastVal = ", lastVal)
                    print("CW = ", self.cwValues[binCount])
                    temp = (val  - lastVal)*(self.cwValues[binCount] - lastLam)/( lam - lastLam) + lastVal
                    print("temp = ", temp)
                    outputY.append(temp)
                    outputYstring.append("{:.7e}".format(temp))
                elif self.topValues[binCount] > ultimaLam:
                    print("lastLam = ", lastLam)
                    print("ultimaLam = ", ultimaLam)
                    print("lastVal = ", lastVal)
                    print("ultimaVal = ", ultimaVal)
                    print("CW = ", self.cwValues[binCount])
                    temp = (lastVal - ultimaVal) * (self.cwValues[binCount] - ultimaLam) / (lastLam - ultimaLam) + ultimaVal
                    print("temp = ", temp)
                    outputY.append(temp)
                    outputYstring.append("{:.7e}".format(temp))
                else:
                    print("ultimaVal = ", ultimaVal)
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
            if lambdaIndex % 100000 == 0:
                print(lambdaIndex)

        print("Time after binning")
        print("--- %s seconds ---" % (time.time() - start_time))
        model = pd.DataFrame(list(zip(self.cwValues, outputY)), columns =['wavelength', 'flux'])
        modelStrings = pd.DataFrame(list(zip(cwValuesString, outputYstring)), columns =['wavelength', 'flux'])

        return model, modelStrings

    def read_model(self, teffStar, teffSpot, teffFac):
        
        modelStrings = None

        # # Works for reading ASCII files
        # model = pd.read_csv(
        #     filename, skiprows=7, delim_whitespace=True, names=["wavelength", "flux"]
        # )

        # Works for reading h5 files
        print('Reading File: <'+self.photModelFile+'>')
        fh5 = h5py.File(self.photModelFile,'r')
        wl = fh5['PHOENIX_SPECTRUM/wl'][()]
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()]
        data = {'wavelength': wl, 'flux': fl}
        self.photModel = pd.DataFrame(data)
        fh5.close()

        # Works for reading h5 files
        print('Reading File: <'+self.spotModelFile+'>')
        fh5 = h5py.File(self.spotModelFile,'r')
        wl = fh5['PHOENIX_SPECTRUM/wl'][()]
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()]
        data = {'wavelength': wl, 'flux': fl}
        self.spotModel = pd.DataFrame(data)
        fh5.close()

        # Works for reading h5 files
        print('Reading File: <'+self.facModelFile+'>')
        fh5 = h5py.File(self.facModelFile,'r')
        wl = fh5['PHOENIX_SPECTRUM/wl'][()]
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()]
        data = {'wavelength': wl, 'flux': fl}
        self.facModel = pd.DataFrame(data)
        fh5.close()

        # lets convert to micron
        self.photModel.wavelength /= 1.0e4
        self.spotModel.wavelength /= 1.0e4
        self.facModel.wavelength /= 1.0e4
        
        # cut the wavelength values tot he specified cut range (.1999-20.5 microns)
        self.photModel = self.photModel.loc[
            (self.photModel.wavelength >= self.rangeMin) & (self.photModel.wavelength <= self.rangeMax)
        ]
        self.spotModel = self.spotModel.loc[
            (self.spotModel.wavelength >= self.rangeMin) & (self.spotModel.wavelength <= self.rangeMax)
        ]
        self.facModel = self.facModel.loc[
            (self.facModel.wavelength >= self.rangeMin) & (self.facModel.wavelength <= self.rangeMax)
        ]

        self.photModel, self.photModelStrings = self.bin(self.photModel)
        self.spotModel, self.spotModelStrings = self.bin(self.spotModel)
        self.facModel, self.facModelStrings = self.bin(self.facModel)

        binnedPhotStringCSV = self.photModelStrings.to_csv(index=False, header=['WAVELENGTH (MICRONS)','FLUX (ERG/CM2/S/A)'], sep=' ')
        file = open(r'./NextGenModels/BinnedData/binned%sStellarModel.txt' % teffStar,'w')
        file.write(binnedPhotStringCSV)
        file.close()

        binnedSpotspectrumCSV = self.spotModelStrings.to_csv(index=False, header=['WAVELENGTH (MICRONS)','FLUX (ERG/CM2/S/A)'], sep=' ')
        file = open(r'./NextGenModels/BinnedData/binned%sStellarModel.txt' % teffSpot,'w')
        file.write(binnedSpotspectrumCSV)
        file.close()

        binnedFaculaespectrumCSV = self.facModelStrings.to_csv(index=False, header=['WAVELENGTH (MICRONS)','FLUX (ERG/CM2/S/A)'], sep=' ')
        file = open(r'./NextGenModels/BinnedData/binned%sStellarModel.txt' % teffFac,'w')
        file.write(binnedFaculaespectrumCSV)
        file.close()

        # # This code plots the difference in wavelength values of the raw data. Shows a resolution across the model
        # wavelengthDifference = []
        # wavelengthResolution = []
        # frequencyList = []
        # frequencyDiff = []
        # c = 2.99e14
        # count = 0
        # for value in model.wavelength:
        #     if count == len(model.wavelength) - 1 or count == 0:
        #         wavelengthDifference.append(float('nan'))
        #         wavelengthResolution.append(float('nan'))
        #         frequencyDiff.append(float('nan'))
        #         frequencyList.append(float('nan'))
        #         # print("End of file")
        #     else:
        #         frequency = c/value
        #         diffWave = model.wavelength[count + 1] - value
        #         diffFreq = (frequency) - (c/model.wavelength[count + 1])
        #         wavelengthDifference.append(diffWave)
        #         wavelengthResolution.append(value/diffWave)
        #         frequencyList.append(frequency)
        #         frequencyDiff.append(diffFreq)

        #         # print(diffWave)
        #         # print(value/diffWave)
        #         # print(diffFreq)
        #         # if diffWave > 0.05:
        #         #     print(model.wavelength[count])
        #         #     print("Greater than 0.05")
        #         #     print(diffWave)
        #     count += 1
        #     if count % 100000 == 0:
        #         print(count)
        
        # fig = plt.figure()
        # plt.yscale('log')
        # plt.plot(model.wavelength, wavelengthDifference)
        # plt.title("Wavelength Spacing of NextGen Data")
        # plt.ylabel("Delta Wavelength")
        # plt.xlabel("Wavelength (um)")
        # if bin_data:
        #     plt.savefig('./%s/BinnedNextGenModels/VariabilityGraphs/wavelengthSpacingOfRawData.png' % starName, bbox_inches='tight')
        # else:
        #     plt.savefig('./%s/NotBinnedNextGenModels/VariabilityGraphs/wavelengthSpacingOfRawData.png' % starName, bbox_inches='tight')
        # # plt.show()
        # plt.close("all")
        # print("WavelengthDiffPlot done")

        # fig = plt.figure()
        # # plt.yscale('log')
        # plt.ticklabel_format(useOffset=False, style='plain')
        # # plt.ylim(resolvingPower - 1, resolvingPower + 1)
        # plt.yscale('log')
        # plt.plot(model.wavelength, wavelengthResolution)
        # plt.title("Wavelength Resolution of NextGen Data")
        # plt.ylabel("Wavelength/Delta Wavelength (Wavelength Resolving Power)")
        # plt.xlabel("Wavelength (um)")
        # if bin_data:
        #     plt.savefig('./%s/BinnedNextGenModels/VariabilityGraphs/wavelengthResolvingPower.png' % starName, bbox_inches='tight')
        # else:
        #     plt.savefig('./%s/NotBinnedNextGenModels/VariabilityGraphs/wavelengthResolvingPower.png' % starName, bbox_inches='tight')
        # # plt.show()
        # plt.close("all")
        # print("WavelengthResolvingPowerPlot done")

        # fig = plt.figure()
        # plt.yscale('log')
        # plt.plot(frequencyList, frequencyDiff)
        # plt.title("Delta Frequency of NextGen Data")
        # plt.ylabel("Delta Frequency")
        # plt.xlabel("Frequency (um/s)")
        # if bin_data:
        #     plt.savefig('./%s/BinnedNextGenModels/VariabilityGraphs/frequencyDiff.png' % starName, bbox_inches='tight')
        # else:
        #     plt.savefig('./%s/NotBinnedNextGenModels/VariabilityGraphs/frequencyDiff.png' % starName, bbox_inches='tight')
        # # plt.show()
        # plt.close("all")
        # print("FrequencyDiffPlot done")

        return self

class ParamModel():

    def __init__(self):
        configParser = configparser.RawConfigParser()
        while True:
            try:
                fileName = input("Config File Path: ./Config/")
                configParser.read_file(open("./Configs/%s" % fileName))
                break
            except FileNotFoundError:
                print("There is no config file by that name, please try again.")
        
        # Read in the information of the star from the config file
        self.starName = configParser.get('Star', 'starName')

        self.spotCoverage = int(configParser.get('Star', 'spotCoverage'))
        # Turn spot coverage into a percentage
        self.spotCoverage = self.spotCoverage/100
        self.spotNumber = int(configParser.get('Star', 'spotNumber'))
        
        self.facCoverage = int(configParser.get('Star', 'facCoverage'))
        # Turn facCoverage into a percentage
        self.facCoverage = self.facCoverage/100
        self.facNumber = int(configParser.get('Star', 'facNumber'))

        # Temperature values for the star, spots, and faculae
        self.teffStar = int(configParser.get('Star', 'teffStar'))
        self.teffSpot = int(configParser.get('Star', 'teffSpot'))
        self.teffFac = int(configParser.get('Star', 'teffFac'))

        # Booleaan that decides whether to generate the hemisphere data or not (if already generated for example)
        self.generateHemispheres = configParser.getboolean('Star', 'generateHemispheres')

        # Boolean that decides whether to create high resolution hemisphere maps or not
        # Default imageResolution of 300
        self.highResHemispheres = configParser.getboolean('HemiMap', 'high_res')
        self.imageResolution = 300

        # Rotation of star in days
        self.rotstar = float(configParser.get('Star', 'Rotstar'))

        # Inclination of the star
        self.inclination = int(configParser.get('Star', 'Inclination'))

        # Total number of exposures to be takn
        self.num_exposures = int(configParser.get('HemiMap', 'num_exposures'))

        # Time (in days) between exposures
        self.time_between_exposures = float(configParser.get('HemiMap', 'time_between_exposures'))

        # Calculates what percent the time between exposures is compared to the stellar rotation period
        # Used to calculate the change in phase between images 
        self.exposure_time_percent = self.time_between_exposures * (100 / self.rotstar)
        # print("Exposure Time percent = ", self.exposure_time_percent)
        self.deltaPhase = self.exposure_time_percent / 100
        # print("Delta Phase = ", self.deltaPhase)

        # Load in the NextGen Stellar Info
        phot_model_file = configParser.get('Star', 'phot_model_file')
        self.phot_model_file = phot_model_file.strip('"') # configParser adds extra "" that I remove
        spot_model_file = configParser.get('Star', 'spot_model_file')
        self.spot_model_file = spot_model_file.strip('"')
        fac_model_file = configParser.get('Star', 'fac_model_file')
        self.fac_model_file = fac_model_file.strip('"')

        # Boolean which says whether or not the user wishes to bin the data to a certain resolving power
        self.binData = configParser.getboolean('Star', 'binData')
        self.loadData = configParser.getboolean('Star', 'loadData')
        
        # The resolving power to bin to
        self.resolvingPower = int(configParser.get('Star', 'resolvingPower'))
        self.binnedWavelengthMin = float(configParser.get('Star', 'binnedWavelengthMin'))
        self.binnedWavelengthMax = float(configParser.get('Star', 'binnedWavelengthMax'))

        # # Whether or not a planet is being considered; creates spectra of the reflection
        # self.planetBool = configParser.getboolean('Planet', 'hasPlanet')

        self.planetName = configParser.get('Planet', 'planetName')
        self.revPlanet = float(configParser.get('Planet', 'revPlanet'))
        # The planet rotation is equivalent to the planet revolution for tidally locked planets
        self.rotPlanet = self.revPlanet
        self.planetPhaseChange = int(configParser.get('Planet', 'planetPhaseChange'))