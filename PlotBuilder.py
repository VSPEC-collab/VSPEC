import SpectraBuilder
import matplotlib.pyplot as plt
import pandas as pd
import statistics

import read_info

# data to save for use in plots:
# 1) adjusted total planet flux (combination of thermal and variable stellar reflected planet)
# 2) sumflux of star for each phase facing observer
# 3) sumflux of star for each phase facing planet
# 4) 

# Strategy for saving this info
# Save in seperate folders with all phase's labeled as "descriptiveName_phase.txt" where phase is the current phase
# in integer form

# QUESTIONS:
# 1) What graphs/images should be produced from this data?
# 2) When graphing the lightcurve of the star, what value do I use? Currently, I calculate the average flux across
#    all wavelengths for that phase. Is this best?

# Plots I want to create
# 1) Lightcurve timeseries
# 2) Plots showing the percent difference change between PSG's reflected spectrum at each phase compared to the
#    reflected spectrum of the variable star's planet of the same phase
# 3) A time-series plot showing the max? average? percent change across wavelengths
# 4) plot of the stellar sumflux by itself for each phase
# 5) plot of the total planet flux by itself for each phase
# 6) plot of the planet flux and stellar flux on the same y axis, shows seperation of ~6 magnitudes of 10
# 7) Plot of planet flux and stellar flux on 2 different y axes, shows the similar shape of the lower wavelengths

# Initial plot in PPM on the y-axis

def max_flux_change(Params, allModels, phase, x, y):
    # The X-Axis represents the number of the images taken. Presents itself as a time series, from first image
    # to last
    adjustment = 360 / Params.num_exposures
    x.append(phase * adjustment)
    y.append(allModels.maxChange)
    print(allModels.maxChange)
    plt.yscale('log')
    plt.plot(x, y)
    plt.title("Variable Planet Flux / PSG Blackbody Planet Flux")
    plt.xlabel("Wavelength (um) Phase %d" % (phase * adjustment))
    plt.ylabel("Average Output Flux (W/um/m2")
    plt.savefig('./%s/Figures/StellarPlots/MaxSumfluxChanges/maxSumfluxChange_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()

    return x, y
    plt.close('all')

def planet_contrast(Params, allModels, phase):
    contrast = allModels.allModelSpectra.planetReflection / allModels.planetReflectionModel.planet
    plt.plot(allModels.allModelSpectra.wavelength, contrast)
    plt.title("Variable Star Planet Flux / PSG Blackbody Planet Flux")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux Contrast")
    # Come up with a directory to store this in ################################
    plt.savefig('./%s/Figures/PlanetPlots/VariableAndPSGContrast/variablAndPSGContrast_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()
    plt.close('all')

# Plots the contrast between the current phase 0 of the planet as it rotates with the initial phase 0.
# Shows the difference caused by the variable surface of the star rather than the spectral models or revolution
# of the planet.
def planet_variation_contrast(Params, allModels, phase):
    contrast = allModels.allModelSpectra.planetReflection / allModels.allModelSpectra0.planetReflection
    plt.plot(allModels.allModelSpectra.wavelength, contrast)
    plt.title("Phase 0 Planet Reflection Flux / Initial Phase 0 Planet Reflection Flux")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux Contrast")
    # Come up with a directory to store this in ################################
    plt.savefig('./%s/Figures/PlanetPlots/PlanetPhase0Contrast/planetPhase0Contrast_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()
    plt.close('all')

# Plot a time-series light curve as the star rotates of the observed stellar flux
def plot_light_curve(allModels, Params, phase, x, y):
    # Calculate the mean value for the current phase, for the current star hemisphere
    avg_sumflux = statistics.mean(allModels.allModelSpectra.sumflux.values)

    # The X-Axis represents the number of the images taken. Presents itself as a time series, from first image
    # to last
    adjustment = 360 / Params.num_exposures
    x.append(phase * adjustment)

    # The y value is the flux value of the current hemisphere image
    y.append(avg_sumflux)

    plt.plot(x, y)
    plt.yscale("log")
    plt.title("Light Curve")
    plt.ylabel("Flux (W/m^2/um)")
    plt.xlabel("Phase (0-360)")
    plt.savefig('./%s/Figures/LightCurves/LightCurve_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()
    plt.close("all")

    return x, y

if __name__ == "__main__":
    Params = read_info.ParamModel()

    allModels = read_info.ReadStarModels(Params.starName)

    # Instantiate x and y data lists used for the lightcurve
    x_lightCurve = []
    y_lightCurve  = []
    x_maxChange = []
    y_maxChange = []
    zero_flag = True
    for phase in range(Params.num_exposures):
        allModels.allModelSpectra = pd.read_csv('./%s/Data/AllModelSpectraValues/phase%d.csv' % (Params.starName, phase), sep=",")
        
        if zero_flag:
            allModels.allModelSpectra0 = allModels.allModelSpectra
            allModels.maxChange = 0
            zero_flag = False

        allModels.planetPhase = (Params.planetPhaseChange * phase) % 360

        # PSG can't calculate the planet's values at phase 180 (in front of star, no reflection), so it calculates them at phase 182.
        if allModels.planetPhase == 180:
            allModels.planetPhase = 182

        # EDIT LATER
        # The way this GCM was created, Phase 176-185 are calculated as if in transit, so we must use phase 186
        # in place of 185 or else the lower wavelength flux values will still be 0.
        if allModels.planetPhase == 185:
            allModels.planetPhase = 186
        allModels.planetReflectionModel = pd.read_csv(
                                          './%s/Data/PSGCombinedSpectra/phase%d.txt' % (Params.starName,
                                                                                        allModels.planetPhase),
                                          skiprows=23,
                                          delim_whitespace=True,
                                          names=["wavelength", "total", "stellar", "planet"],
                                          )

        if Params.plotLightCurve:
            x, y = plot_light_curve(allModels, Params, phase, x, y)

        if Params.plotPlanetContrast:
            planet_contrast(Params, allModels, phase)

        if allModels.planetPhase == 0 and Params.plotPlanetVariationContrast:
            planet_variation_contrast(Params, allModels, phase)

        # Calculate the average sumflux of the current phase and subtract the initial sumflux to determine
        # the difference. Highlights how much the stellar flux is affected by the variable surface.
        change = abs(statistics.mean(allModels.allModelSpectra.sumflux.values) -
                 statistics.mean(allModels.allModelSpectra0.sumflux.values))
        if Params.plotMaxFluxChange and change > allModels.maxChange:
            x_maxChange, y_maxChange = max_flux_change(Params, allModels, phase, x_maxChange, y_maxChange)
            allModels.maxChange = change