import ast
import csv
import read_info
import numpy as np
import os
import pandas as pd

def calculate_combined_spectrum(allModels, Params, percentagesDict, percentagesDictTowardsPlanet):
    photFrac = percentagesDict['phot']
    spotFrac = percentagesDict['spot']
    facFrac = percentagesDict['fac']
    location = len(allModels.allModelSpectra.axes[1])
    allModels.allModelSpectra.insert(location, 'sumflux', (allModels.allModelSpectra.photflux * photFrac) +
                                                          (allModels.allModelSpectra.spotflux * spotFrac) +
                                                          (allModels.allModelSpectra.facflux * facFrac))

    photFracTowardsPlanet = percentagesDictTowardsPlanet['phot']
    spotFracTowardsPlanet = percentagesDictTowardsPlanet['spot']
    facFracTowardsPlanet = percentagesDictTowardsPlanet['fac']
    location = len(allModels.allModelSpectra.axes[1])
    allModels.allModelSpectra.insert(location, 'sumfluxTowardsPlanet', (allModels.allModelSpectra.photflux * photFracTowardsPlanet) +
                                    (allModels.allModelSpectra.spotflux * spotFracTowardsPlanet) +
                                    (allModels.allModelSpectra.facflux * facFracTowardsPlanet))

if __name__ == "__main__":
    # 1) Read in all of the user-defined config parameters into a class, called Params.
    Params = read_info.ParamModel()

    # Create an object to store all of the spectra flux values
    allModels = read_info.ReadStarModels(Params.starName)

    # Load in the NextGen Stellar Data for photosphere, spot, and faculae temperatures
    allModels.photModel = pd.read_csv('./NextGenModels/BinnedData/binned%dStellarModel.txt' % Params.teffStar,
                                names=['wavelength', 'flux'], delimiter=' ', skiprows=1)

    allModels.spotModel = pd.read_csv('./NextGenModels/BinnedData/binned%dStellarModel.txt' % Params.teffSpot,
                                names=['wavelength', 'flux'], delimiter=' ', skiprows=1)
    
    allModels.facModel = pd.read_csv('./NextGenModels/BinnedData/binned%dStellarModel.txt' % Params.teffFac,
                                    names=['wavelength', 'flux'], delimiter=' ', skiprows=1)

    if not np.all(allModels.photModel.wavelength == allModels.spotModel.wavelength) or not np.all(allModels.photModel.wavelength == allModels.facModel.wavelength):
        raise ValueError("The star, spot, and faculae spectra should be on the same wavelength scale and currently are not.")
    data = {'wavelength': allModels.photModel.wavelength, 'photflux': allModels.photModel.flux, 'spotflux': allModels.spotModel.flux, 'facflux': allModels.facModel.flux}
    allModels.allModelSpectra = pd.DataFrame(data)

    # EDIT LATER: Currently hard-coded to convert into W/m2/um
    conversion = Params.erg_sTOwatts * Params.cm2TOm2 * Params.cmTOum * Params.distanceFluxCorrection

    allModels.allModelSpectra.photflux *= conversion
    allModels.allModelSpectra.spotflux *= conversion
    allModels.allModelSpectra.facflux *= conversion

    # Load in the dictionary of surface coverage percentages for each of the star's phases
    surfaceCoverageDict = {}
    with open('%s/Data/SurfaceCoveragePercentage/surfaceCoveragePercentageDict.csv' % Params.starName, newline='') as csvfile:
        # reader = csv.DictReader(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            valueDict = ast.literal_eval(row[1])
            surfaceCoverageDict[float(row[0])] = valueDict

        print("Done")

    # For loop here to run through each "image"/number of exposures as specified in the config file
    for index in range(Params.num_exposures):
        # The current phase of the planet is the planet phase change value (between exposures) multiplied
        # by the nuber of exposures taken so far (index)
        # Example: 180
        # Planet phase change is specified by the user; how many degrees it turns between "images" of the star
        allModels.planetPhase = (Params.planetPhaseChange * index) % 360

        # PSG can't calculate the planet's values at phase 180 (in front of star, no reflection), so it calculates them at phase 182.
        if allModels.planetPhase == 180:
            allModels.planetPhase = 182

        # EDIT LATER
        # The way this GCM was created, Phase 176-185 are calculated as if in transit, so we must use phase 186
        # in place of 185 or else the lower wavelength flux values will still be 0.
        if allModels.planetPhase == 185:
            allModels.planetPhase = 186

        # The current phase of the star is the star phase change value (between exposures) multiplied
        # by the nuber of exposures taken so far (index)
        # Example: 30
        allModels.starPhase = (Params.deltaStellarPhase * index) % 360

        # In PSG's models, phase 0 for the planet is "behind" the star from the viewer's perspective,
        # in secondary eclipse.
        # In the variable stellar code, phase 0 of the star is the side of the star facing the observer.
        # This means that, when starting the time-series simulation, the face of the star facing the observer is phase 0,
        # but the face of the star facing the planet is whatever half the total number of exposures is.
        # In the default example, there are 252 exposures taken, so the initial face of the star looking towards the planet
        # is phase 252/2 = 126

        # The star phase currently facing the planet has to take into account where the planet is in its orbit.
        # This is calculated by taking how far the planet has rotated/revolved (one in the same for tidally locked like this),
        # plus 180 degrees (an offset necessary as explained above), then simply subtract how far the star has
        # turned to figure out what phase of the star is facing the planet.
        # Modulo 360 ensures it is never above that value, and dividing by delta stellar phase 
        allModels.starPhaseFacingPlanet = round((((allModels.planetPhase + 180) - allModels.starPhase) % 360) / Params.deltaStellarPhase)
        if allModels.starPhaseFacingPlanet == 252:
            allModels.starPhaseFacingPlanet = 0

        # Read in the planet's reflected spectrum (in W/sr/m^2/um) for the current phase
        allModels.planetReflectionModel = pd.read_csv(
            './%s/Data/PSGCombinedSpectra/phase%d.txt' % (Params.starName, allModels.planetPhase),
            skiprows=23,
            delim_whitespace=True,
            names=["wavelength", "total", "stellar", "planet"],
            )

        # Read in the planet's thermal spectrum (in W/m^2/um) for the current phase
        allModels.planetThermalModel = pd.read_csv(
            './%s/Data/PSGThermalSpectra/phase%d.txt' % (Params.starName, allModels.planetPhase),
            skiprows=23,
            delim_whitespace=True,
            names=["wavelength", "total", "planet"],
            )

        location = len(allModels.allModelSpectra.axes[1])
        allModels.allModelSpectra.insert(location, 'planetReflection', allModels.planetReflectionModel.planet)
        location = len(allModels.allModelSpectra.axes[1])
        allModels.allModelSpectra.insert(location, 'planetThermal', allModels.planetThermalModel.planet)

        # Calculate the total output flux of this star's phase by computing a linear combination of the photosphere,
        # spot, and flux models based on what percent of the surface area those components take up
        tempPhase = Params.deltaPhase * index
        percentagesDict = surfaceCoverageDict[tempPhase]
        percentagesDictTowardsPlanet = []
        calculate_combined_spectrum(allModels, Params, percentagesDict, percentagesDictTowardsPlanet)

        print("done")