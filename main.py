import read_info, star_functions
import numpy as np
import os
import pandas as pd

if __name__ == "__main__":
    # Main program of my spotty-PSG connection code

    # 1) create a program that reads in all of the user-defined config parameters
    #    into a class, called params.
    Params = read_info.ParamModel()

    # 2) Check that all directories used in the program are created
    if not os.path.isdir('./%s/' % Params.starName):
        os.mkdir('./%s/' % Params.starName)
        os.mkdir('./%s/Data' % Params.starName)
        os.mkdir('./%s/Data/HemiMapArrays/' % Params.starName)
        os.mkdir('./%s/Data/PSGCombinedSpectra/' % Params.starName)
        os.mkdir('./%s/Data/PSGThermalSpectra/' % Params.starName)
        os.mkdir('./%s/Data/SumfluxArraysByPhase/' % Params.starName)
        os.mkdir('./%s/Figures' % Params.starName)
        os.mkdir('./%s/Figures/VariabilityGraphs' % Params.starName)
        os.mkdir('./%s/Figures/Hemi+LightCurve/' % Params.starName)
        os.mkdir('./%s/Figures/HemiMapImages/' % Params.starName)
        os.mkdir('./%s/Figures/IntegralPhasePlot/' % Params.starName)
        os.mkdir('./%s/Figures/LightCurves' % Params.starName)
        os.mkdir('./%s/Figures/PlanetFluxCalculation/' % Params.starName)
        os.mkdir('./%s/Figures/SpectrumGraphs/' % Params.starName)

    # 3) If not already created, create the 2D Spot map of the star's surface
    #    that plots the locations of the spots and faculae. Used later to create
    #    the 3D hemisphere models
    if not os.path.isfile('./%s/Figures/FlatMap.png' % Params.starName):
        # Create a 2D spotmmodel from the star_flatmap.py file
        StarModel2D = star_functions.StarModel2D(
            Params.spotCoverage,
            Params.spotNumber,
            Params.facCoverage,
            Params.facNumber,
            Params.starName,
        )

                    # Generate the spots on the star
        surface_map = StarModel2D.generate_spots()

        # Convert the 1's and 0's in the ndarray, which store the spot locations, to a smaller data type
        surface_map = surface_map.astype(np.int8)

        # Saves the numpy array version of the flat surface map to be loaded while creating the hemisphere views
        np.save('./%s/Data/flatMap.npy' % Params.starName, surface_map)

    # 4) This section generates the hemispheres of the star, based on the 2D surface map.
         
    # Name of the .npy array which contains the flat surface map
    # This array is used to create the hemisphere maps
    surface_map = np.load('./%s/Data/flatMap.npy' % Params.starName)

    # If the generate hemispheres boolean set by the user-defined config is true, the program will generate
    # (or re-generate with different inclination for example) the star hemispheres.
    if Params.generateHemispheres:

        # Can be made high res by user-specified config. 3000x3000 array
        if Params.highResHemispheres:
            Params.imageResolution = 3000

        HM = star_functions.HemiModel(
            Params.teffStar,
            Params.rotstar,
            surface_map,
            Params.inclination,
            Params.imageResolution,
            Params.starName,
        )

        # Tyler's Suggestion
        # create array of phases
        phases = [i * Params.deltaPhase for i in range(Params.num_exposures)]

        # Begin at phase = 0
        # Count keeps track of which hemisphere map image is being looked at currently
        phase = 0
        count = 0
        print("GENERATING HEMISPHERES")
        print("----------------------")
        for phase in phases:
            hemi_map = HM.generate_hemisphere_map(phase, count)
            print("Percent Complete = ", phase * 100, "%")
            count += 1
        
        deltaStellarPhase = Params.deltaPhase * 360


    topValues = []
    cwValues = []
    CW = Params.binnedWavelengthMin
    # Calculate the center wavelenghts (CW) and upper values (top) of each bin
    while CW < Params.binnedWavelengthMax:
        deltaLambda = CW / Params.resolvingPower
        topValue = CW + (deltaLambda / 2)
        topValues.append(topValue)
        cwValues.append(CW)
        CW += deltaLambda

    # If the stellar data was previously binned, simply load in the saved binned stellar model.
    ###### Does saving these as .npy arrays save storage space/loading time rather than .txt files?
    if Params.loadData:
        starspectrum = pd.read_csv('./NextGenModels/BinnedData/binned%dStellarModel.txt' % Params.teffStar,
                                  names=['wavelength', 'flux'], delimiter=' ', skiprows=1)

        spotspectrum = pd.read_csv('./BinnedNextGenModels/binned%dStellarModel.txt' % Params.teffSpot,
                                   names=['wavelength', 'flux'], delimiter=' ', skiprows=1)
        
        faculaespectrum = pd.read_csv('./BinnedNextGenModels/binned%dStellarModel.txt' % Params.binDatateffFac,
                                      names=['wavelength', 'flux'], delimiter=' ', skiprows=1)
    else:
        allModels = read_info.ReadStarModels(Params.starName, Params.binnedWavelengthMin, Params.binnedWavelengthMax,
                                             Params.imageResolution, Params.resolvingPower, topValues, cwValues,
                                             Params.phot_model_file, Params.spot_model_file, Params.fac_model_file)
        allModels.read_model(Params.teffStar, Params.teffSpot, Params.teffFac)

    print("done")
    # GCM file type should be a net cdf file typ
    # Most common gcf file type