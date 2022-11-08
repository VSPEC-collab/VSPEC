import VSPEC.read_info as read_info
from pathlib import Path

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

    # If the stellar data was previously binned, simply load in the saved binned stellar model.
    if Params.loadData:
        allModels = read_info.ReadStarModels(Params.starName)
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

        allModels = read_info.ReadStarModels(Params.starName, binnedWavelengthMin=Params.binnedWavelengthMin,
                binnedWavelengthMax=Params.binnedWavelengthMax,filenames=Params.model_files,
                resolvingPower = Params.resolvingPower, topValues=topValues, cwValues=cwValues)
        allModels.read_model(Params.teffs)
        print("Done")

    print("done")

