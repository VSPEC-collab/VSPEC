from multiprocessing import Pool, cpu_count, Array
import read_info, star_functions
import csv
import numpy as np
import os
import pandas as pd
from pathlib import Path

import time

import faulthandler; faulthandler.enable()

# 1st file to run.

if __name__ == "__main__":
    
    # 1) Read in all of the user-defined config parameters into a class, called Params.
    Params = read_info.ParamModel()
    
    parent_folder = Path('.') / f'{Params.starName}/'
    figures_folder = parent_folder / 'Figures'

    # CHANGE ALL % FormattinG to f"text {variable}"
    # 2) Check that all directories used in the program are created
    if not parent_folder.exists():
        parent_folder.mkdir()
        data_folder = parent_folder / 'Data'
        data_folder.mkdir()
        all_spectra_values_folder = data_folder / 'AllModelSpectraValues'
        all_spectra_values_folder.mkdir()
        hemi_map_arrays_folder = data_folder / 'HemiMapArrays'
        hemi_map_arrays_folder.mkdir()
        PSG_combined_spectra_folder = data_folder / 'PSGCombinedSpectra'
        PSG_combined_spectra_folder.mkdir()
        PSG_thermal_spectra_folder = data_folder / 'PSGThermalSpectra'
        PSG_thermal_spectra_folder.mkdir()
        PSG_noise_folder = data_folder / 'PSGNoise'
        PSG_noise_folder.mkdir()
        PSG_layers_folder = data_folder / 'PSGLayers'
        PSG_layers_folder.mkdir()
        sumflux_arrays_towards_observer_folder = data_folder / 'SumfluxArraysTowardsObserver'
        sumflux_arrays_towards_observer_folder.mkdir()
        sumflux_arrays_towards_planet_folder = data_folder / 'SumfluxArraysTowardsPlanet'
        sumflux_arrays_towards_planet_folder.mkdir()
        surface_coverage_percentage_folder = data_folder / 'SurfaceCoveragePercentage'
        surface_coverage_percentage_folder.mkdir()
        variable_planet_flux_folder = data_folder / 'VariablePlanetFlux'
        variable_planet_flux_folder.mkdir()
        figures_folder.mkdir()
        gifs_folder = figures_folder / 'GIFs'
        gifs_folder.mkdir()
        hemi_and_light_curve_folder = figures_folder / 'Hemi+LightCurve'
        hemi_and_light_curve_folder.mkdir()
        hemi_map_images_folder = figures_folder / 'HemiMapImages'
        hemi_map_images_folder.mkdir()
        integral_phase_plot_folder = figures_folder / 'IntegralPhasePlot'
        integral_phase_plot_folder.mkdir()
        light_curves_observer_folder = figures_folder / 'LightCurvesObserver'
        light_curves_observer_folder.mkdir()
        light_curves_planet_folder = figures_folder / 'LightCurvesPlanet'
        light_curves_planet_folder.mkdir()
        planet_contrast_and_light_curve_folder = figures_folder / 'PlanetContrast+LightCurve'
        planet_contrast_and_light_curve_folder.mkdir()
        planet_plots_folder = figures_folder / 'PlanetPlots'
        planet_plots_folder.mkdir()
        planet_flux_folder = planet_plots_folder / 'PlanetFlux'
        planet_flux_folder.mkdir()
        planet_phase_0_contrast_folder = planet_plots_folder / 'PlanetPhase0Contrast'
        planet_phase_0_contrast_folder.mkdir()
        planet_phase_90_contrast_folder = planet_plots_folder / 'PlanetPhase90Contrast'
        planet_phase_90_contrast_folder.mkdir()
        planet_phase_180_contrast_folder = planet_plots_folder / 'PlanetPhase180Contrast'
        planet_phase_180_contrast_folder.mkdir()
        variable_and_PSG_contrast_folder = planet_plots_folder / 'VariableAndPSGContrast'
        variable_and_PSG_contrast_folder.mkdir()
        variable_H2O_feature_folder = planet_plots_folder / 'VariableH2OFeature'
        variable_H2O_feature_folder.mkdir()
        variable_CO2_feature_folder = planet_plots_folder / 'VariableCO2Feature'
        variable_CO2_feature_folder.mkdir()
        stellar_plots_folder = figures_folder / 'StellarPlots'
        stellar_plots_folder.mkdir()
        max_sumflux_changes_folder = stellar_plots_folder / 'MaxSumfluxChanges'
        max_sumflux_changes_folder.mkdir()
        star_flux_folder = stellar_plots_folder / 'StarFlux'
        star_flux_folder.mkdir()
        variability_graphs_folder = figures_folder / 'VariabilityGraphs'
        variability_graphs_folder.mkdir()

    # 3) If not already created, create the 2D Spot map of the star's surface
    #    that plots the locations of the spots and faculae. Used later to create
    #    the 3D hemisphere models
    # flat_map_image = figures_folder / 'FlatMap.png'
    # if not flat_map_image.exists():
    #     # Create a 2D spotmmodel from the star_flatmap.py file
    #     StarModel2D = star_functions.StarModel2D(
    #         Params.spotCoverage,
    #         Params.spotNumber,
    #         Params.facCoverage,
    #         Params.facNumber,
    #         Params.starName,
    #     )

    #                 # Generate the spots on the star
    #     surface_map = StarModel2D.generate_spots()

    #     # Convert the 1's and 0's in the ndarray, which store the spot locations, to a smaller data type
    #     surface_map = surface_map.astype(np.int8)

        # Saves the numpy array version of the flat surface map to be loaded while creating the hemisphere views
        
        # flat_map_file = data_folder / 'flatMap.npy'
        # np.save(flat_map_file, surface_map)

    # 4) This section "Bins" the stellar models to be a uniform resolving power and wavelength range.

    # If the stellar data was previously binned, simply load in the saved binned stellar model.
    ###### Does saving these as .npy arrays save storage space/loading time rather than .txt files?
    if Params.loadData:
        allModels = read_info.ReadStarModels(Params.starName)

        # file_to_read = Path('.') / 'NextGenModels' / 'BinnedData' / f'binned{Params.teffStar}StellarModel.txt'
        # allModels.photModel = pd.read_csv(file_to_read, names=['wavelength', 'flux'], delimiter=' ', skiprows=1)

        # file_to_read = Path('.') / 'NextGenModels' / 'BinnedData' / f'binned{Params.teffSpot}StellarModel.txt'
        # allModels.spotModel = pd.read_csv(file_to_read, names=['wavelength', 'flux'], delimiter=' ', skiprows=1)
        
        # file_to_read = Path('.') / 'NextGenModels' / 'BinnedData' / f'binned{Params.teffFac}StellarModel.txt'
        # allModels.facModel = pd.read_csv(file_to_read, names=['wavelength', 'flux'], delimiter=' ', skiprows=1)
        
        # if not np.all(allModels.photModel.wavelength == allModels.spotModel.wavelength) or not np.all(allModels.photModel.wavelength == allModels.facModel.wavelength):
        #     raise ValueError("The star, spot, and faculae spectra should be on the same wavelength scale and currently are not. Have you binned the date yet? Check 'binData' in your config file.")
        # data = {'wavelength': allModels.photModel.wavelength, 'photflux': allModels.photModel.flux, 'spotflux': allModels.spotModel.flux, 'facflux': allModels.facModel.flux}
        # allModels.mainDataFrame = pd.DataFrame(data)
    else:
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

        allModels = read_info.ReadStarModels(Params.starName, Params.binnedWavelengthMin, Params.binnedWavelengthMax,
                                             Params.imageResolution, Params.resolvingPower, Params.phot_model_file,
                                             Params.spot_model_file, Params.fac_model_file, topValues, cwValues)
        allModels.read_model(Params.teffs)
        print("Done")

    # 5) This section generates the hemispheres of the star, based on the 2D surface map.
         
    # Name of the .npy array which contains the flat surface map
    # This array is used to create the hemisphere maps
    # surface_map = np.load(Path('.') / f'{Params.starName}' / 'Data' / 'flatMap.npy')

    # If the generate hemispheres boolean set by the user-defined config is true, the program will generate
    # (or re-generate with different inclination for example) the star hemispheres.
    # if Params.generateHemispheres:

    #     # Can be made high res by user-specified config. 3000x3000 (SLOW). Array by default is 180x180.
    #     # This is the smallest number if 'pixels' in the image that can be used without a noticeable los of accuracy
    #     if Params.highResHemispheres:
    #         Params.imageResolution = 3000

    #     HM = star_functions.HemiModel(
    #         Params,
    #         Params.teffStar,
    #         Params.rotstar,
    #         surface_map,
    #         Params.inclination,
    #         Params.imageResolution,
    #         Params.starName,
    #     )

    #     # Begin at phase = 0
    #     print("\nGENERATING HEMISPHERES")
    #     print("----------------------")
        
    #     HM.multiprocessing_worker()
        
    #     # Writes the surface coverage percentage of the photosphere, spots, and faculae for each image taken to file.
    #     # Allows for a speedy way to read the data further down the line.
    #     w = csv.writer(open(Path('.') / f'{Params.starName}' / 'Data' / 'SurfaceCoveragePercentage' / 'surfaceCoveragePercentageDict.csv', 'w'))
    #     for key, val in HM.surfaceCoverageDictionary.items():
    #         w.writerow([key, val])

    # GCM file type should be a net cdf file type
    # Most common gcf file type

    print("done")

