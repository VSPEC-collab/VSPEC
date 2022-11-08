from copyreg import constructor
from csv import writer
from os import remove
import VSPEC.SpectraBuilder as SpectraBuilder
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import statistics
from PIL import Image
from pathlib import Path
import glob

import VSPEC.read_info as read_info

# 4th file to run.

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

# Re-create the PlanetVariationFlux plot at specified wavelengths: optical and 10 um; locked at 90 degrees planet phase
# Also re-create this at eclipse and transit

# Give options to specify wavelength, planet-phase constants, stellar phase contants, etc. Super customizeable.

# Start with images that are paper-worthy

def max_flux_change(Params, allModels, phase, x, y):
    # The X-Axis represents the number of the images taken. Presents itself as a time series, from first image
    # to last
    # EDIT LATER: This adjustment won't work for total_images>360 AND stellar revolutions > 1
    adjustment = 360 / Params.total_images
    x.append(phase * adjustment)
    # y.append(allModels.maxChange)
    y.append((allModels.maxChange / allModels.minFlux) * 100)
    # plt.yscale('log')
    plt.plot(x, y)
    plt.title("Max Stellar Flux Change Over Time")
    plt.xlabel("Phase %d" % (phase * adjustment))
    plt.ylabel("Percent Change in Flux")
    plt.savefig('./%s/Figures/StellarPlots/MaxSumfluxChanges/maxSumfluxChange_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()
    plt.close('all')

    return x, y

# Divides the flux of the planet revolving around the variable star by flux of the planet around a static star.
# Highlights how much the stellar variability changes the planet's reflected flux compared to a non-variable star.
def planet_contrast(Params, allModels, phase):
    contrast = allModels.allModelSpectra.planetReflection / allModels.allModelSpectra0Spots.planetReflection
    # plt.yscale('log')
    plt.xlim(19, 20)
    # plt.xlim(.2, 20)
    plt.ylim(0.999, 1.001)
    plt.plot(allModels.allModelSpectra.wavelength, contrast, label = 'Contrast')
    # Original Values from .4-5 um.
    # Also collecting min and max flux percent change from 3-5 um, 14-15um, and 19-20um
    tempSliceStart = np.where(allModels.allModelSpectra.wavelength >= 19)
    tempSliceStartInt = tempSliceStart[0][0]
    tempSliceEnd = np.where(allModels.allModelSpectra.wavelength <= 20)
    tempSliceEndInt = tempSliceEnd[0][-1]
    tempSliceWavelength = allModels.allModelSpectra.wavelength.to_numpy()
    tempSliceWavelength = tempSliceWavelength[tempSliceStartInt:tempSliceEndInt + 1]
    tempSlice = contrast.to_numpy()#[tempSliceStartInt:tempSliceEnd + 1]
    tempSlice = tempSlice[tempSliceStartInt:tempSliceEndInt + 1]
    if min(tempSlice) > 0 and min(tempSlice) < allModels.contrastMin:
        allModels.contrastMin = min(tempSlice)
    plt.hlines(allModels.contrastMin, 19, 20, colors='red', linestyles=':', label='Min Contrast')
    if max(tempSlice) > allModels.contrastMax:
        allModels.contrastMax = max(tempSlice)
    plt.hlines(allModels.contrastMax, 19, 20, colors='green', linestyles=':', label='Max Contrast')
    # tempArray = allModels.allModelSpectra.planetReflection[indexStart[0]:indexEnd[indexEnd.length]]
    plt.title("Var. Star Planet Reflection/Stat. Star Planet Reflection")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux Contrast")
    plt.legend()
    plt.savefig('./%s/Figures/PlanetPlots/VariableAndPSGContrast/variableAndPSGContrast_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()
    plt.close('all')

# Plots the contrast between the current phase 0 of the planet as it rotates with the initial phase 0.
# Same for phase 90 and 180.
# Shows the difference caused by the variable surface of the star rather than the spectral models or revolution
# of the planet.
def planet_variation_contrast(Params, allModels, current_phase, initial_phase, revolution, flag):
    if initial_phase == 0:
        contrast = allModels.allModelSpectra.planetReflection / allModels.allModelSpectra0.planetReflection
        linestyle = '-'
        # linestyle = '-'
    elif initial_phase == 90:
        contrast = allModels.allModelSpectra.planetReflection / allModels.allModelSpectra90.planetReflection
        linestyle = '-'
    elif initial_phase == 180:
        contrast = allModels.allModelSpectra.planetReflection / allModels.allModelSpectra180.planetReflection
    
    if revolution == 0:
        return
    elif revolution == 1:
        color = 'orange'
        flag=True
    elif revolution == 2:
        color = 'green'
        flag=True
    elif revolution == 3:
        color = 'blue'
        flag=True
    elif revolution > 3:
        return
    
    contrast -= 1
    contast = abs(contrast)
    label = f'Phase {initial_phase}, Revolution {revolution + 1}'
    
    plt.xlim(0.2, 7)
    plt.ylim(10e-8, 1)
    plt.yscale('log')
    plt.plot(allModels.allModelSpectra.wavelength, contrast, linestyle, color=color, label=label)
    # plt.show()
    if flag:
        plt.hlines(.001, .2, 7, colors='purple', linestyles=':', label='1000 ppm')
        plt.hlines(.00001, .2, 7, colors='blue', linestyles=':', label='10 ppm')
        plt.hlines(.000001, .2, 7, colors='red', linestyles=':', label='1 ppm')
    else:
        plt.hlines(.001, .2, 7, colors='cyan', linestyles=':')
        plt.hlines(.00001, .2, 7, colors='blue', linestyles=':')
        plt.hlines(.000001, .2, 7, colors='red', linestyles=':')
    # plt.plot(label='10 ppm')
    plt.title("Comparison of Planetary Phases")
    plt.legend()
    plt.xlabel("Wavelength (um)")
    # plt.ylabel("log10 (Delta-F/F0)")
    plt.ylabel("Flux Contrast")
    plt.savefig('./%s/Figures/PlanetPlots/PlanetPhase%dContrast/planetPhaseContrast_%d' % (Params.starName, initial_phase, current_phase), bbox_inches='tight')
    # plt.show()
    plt.close('all')

# Plot a time-series light curve of the stellar flux as the star rotates.
# This method runs twice, once from the observer's perspective, and once from the planet's perspective.
def plot_light_curve(allModels, Params, phase, x, y, towardsPlanet=False):
    fig, ax = plt.subplots()
    
    # Calculate the mean value for the current phase, for the current star hemisphere, either facing the observer or planet
    # The X-Axis represents the number of the images taken. Presents itself as a time series, from first image
    # to last
    if towardsPlanet:
        avg_sumflux = statistics.mean(allModels.allModelSpectra.sumfluxTowardsPlanet)
        x.append(phase * Params.delta_phase_star)
    else:
        avg_sumflux = statistics.mean(allModels.allModelSpectra.sumflux.values)
        x.append(phase * Params.delta_phase_star)

    # The y value is the flux value of the current hemisphere image
    y.append(avg_sumflux)
    ax.plot(x, y, label="Phase %d" % phase)
    plt.yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.ylabel("Flux (W/m^2/um)")
    plt.xlabel("Star Phase (0-%d)" % Params.final_stellar_phase)
    if towardsPlanet:
        plt.title("Light Curve Towards Planet")
        plt.savefig('./%s/Figures/LightCurvesPlanet/LightCurve_%d' % (Params.starName, phase), bbox_inches='tight')
    else:
        plt.title("Light Curve Towards Observer")
        plt.savefig('./%s/Figures/LightCurvesObserver/LightCurve_%d' % (Params.starName, phase), bbox_inches='tight')
    # ax.show()
    plt.close("all")

    return x, y

# Plot the stellar flux from the observers perspective
def plot_stellar_flux(Params, allModels, phase):
    plt.plot(allModels.allModelSpectra.wavelength, allModels.allModelSpectra.sumflux)
    plt.title("Star's Output Flux")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux (W/um/m2)")
    plt.savefig('./%s/Figures/StellarPlots/StarFlux/totalStellarFlux_%d' % (Params.starName, phase), bbox_inches='tight')
    plt.close('all')
    
# Plot the reflected flux spectrum of the planet revolving the variable star
def plot_adjusted_planet_flux(Params, allModels, phase):
    plt.ylim(1e-23, 50e-19)
    plt.yscale('log')
    plt.plot(allModels.allModelSpectra.wavelength, allModels.allModelSpectra.planetReflection)
    plt.title("Planet Flux Affected by a Variable Star")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux (W/um/m2)")
    plt.savefig('./%s/Figures/PlanetPlots/PlanetFlux/planetPhaseFlux_%d' % (Params.starName, phase), bbox_inches='tight')
    # plt.show()
    plt.close('all')
    
# Plot the reflected flux spectrum of the planet revolving around the variable star ON TOP of the reflected flux of the planet
# revolving around a static star
def plot_planet_flux_variation(Params, allModels, phase):
    plt.yscale('log')
    plt.ylim(10e-24, 10e-18)
    plt.plot(allModels.allModelSpectra0Spots.wavelength, allModels.allModelSpectra0Spots.planetReflection, label='Static')
    plt.plot(allModels.allModelSpectra.wavelength, allModels.allModelSpectra.planetReflection, label='Variable')
    plt.legend(loc="lower right")
    plt.title("Static Star Planet Flux Compared to Variable Star Planet Flux")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux (W/um/m2)")
    plt.savefig('./%s/Figures/PlanetPlots/PlanetFlux/PSGandVariablePlanetFluxComparison_%d' % (Params.starName, phase), bbox_inches='tight')
    plt.close('all')
    
# Combine the observer's lightcurve plot with the image of the stellar hemisphere into one image
def plot_lc_hemi_combo(Params, allModels, phase):
    for image in range(Params.total_images):
        star_phase = (Params.delta_phase_star * image) % 360
        star_phase_string = str("%.3f" % star_phase)
        #Read the two images
        hemi = Image.open("./%s/Figures/HemiMapImages/hemiMap_%s.png" % (Params.starName, star_phase_string))
        # hemi.show()
        lc = Image.open("./%s/Figures/LightCurvesObserver/LightCurve_%s.png" % (Params.starName, image))
        # lc.show()
        
        hemiSize = hemi.size
        lc = lc.resize(hemiSize)
        
        # In the size parameter (2nd parameter), the second value is the height
        newImage = Image.new('RGB',(2*hemiSize[0], 1*hemiSize[1]), (250,250,250))
        newImage.paste(hemi,(0,0))
        newImage.paste(lc,(hemiSize[0],0))
        
        # newImage.show()
        newImage.save("./%s/Figures/Hemi+LightCurve/Hemi+LightCurve_%d.png" % (Params.starName, image))
        
# Combine the lightcurve experienced by the planet with the graph of the VariableStarPlanet/PSGPlanet Contrast graph into
# one image
def plot_lc_planet_contrast_combo(Params, allModels, phase):
    for image in range(Params.total_images):
        star_phase = (Params.delta_phase_star * image) % 360
        star_phase_string = str("%.3f" % star_phase)
        #Read the two images
        planet_contrast = Image.open("./%s/Figures/PlanetPlots/VariableAndPSGContrast/variableAndPSGContrast_%s.png" % (Params.starName, image))
        # hemi.show()
        lc = Image.open("./%s/Figures/LightCurvesPlanet/LightCurve_%d.png" % (Params.starName, image))
        # lc.show()
        
        contrastSize = planet_contrast.size
        lc = lc.resize(contrastSize)
        
        # In the size parameter (2nd parameter), the second value is the height
        newImage = Image.new('RGB',(2*contrastSize[0], 1*contrastSize[1]), (250,250,250))
        newImage.paste(planet_contrast,(0,0))
        newImage.paste(lc,(contrastSize[0],0))
        
        # newImage.show()
        newImage.save("./%s/Figures/PlanetContrast+LightCurve/PlanetContrast+LightCurve_%d.png" % (Params.starName, image))
        
def plot_PSG_planet_reflection(Params, allModels, phase):
    plt.yscale('log')
    plt.ylim(10e-24, 10e-18)
    plt.plot(allModels.PSGPlanetReflectionModel.wavelength, allModels.PSGPlanetReflectionModel.planet, label='PSG Planet')
    plt.legend(loc="lower right")
    plt.title("PSG Planet Flux")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux (W/um/m2)")
    plt.savefig('./%s/Figures/PlanetPlots/PlanetFlux/PSGPlanetFlux_%d' % (Params.starName, phase), bbox_inches='tight')
    plt.close('all')

        
def plot_wavelength_bin_flux_variation(Params, allModels, phase, bin, revolution):
    
    if bin == 'H2O':
        # wl = 2.75
        wl = 2.756438874
        bin_min = 2.302785193
        bin_max = 2.903415065
    elif bin == 'CO2':
        wl = 4.2
        wl = 4.201821600
        bin_min = 3.902401165
        bin_max = 4.551415357
    
    if phase == 9 and revolution == 0:
        allModels.allModelSpectra0Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra0Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra0Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        
        allModels.allModelSpectra1Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra1Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra1Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        
        allModels.allModelSpectra10Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
        
        tempSliceStart = np.where(allModels.allModelSpectra.wavelength >= bin_min)
        tempSliceStartInt = tempSliceStart[0][0]
        
        tempSliceEnd = np.where(allModels.allModelSpectra.wavelength <= bin_max)
        tempSliceEndInt = tempSliceEnd[0][-1]

        tempSliceWavelength = allModels.allModelSpectra.wavelength.to_numpy()
        tempSliceWavelength = tempSliceWavelength[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced0Pct2300 = allModels.allModelSpectra0Pct2300.planetReflection.to_numpy()
        sliced0Pct2300 = sliced0Pct2300[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced0Pct2600 = allModels.allModelSpectra0Pct2600.planetReflection.to_numpy()
        sliced0Pct2600 = sliced0Pct2600[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced0Pct2900 = allModels.allModelSpectra0Pct2900.planetReflection.to_numpy()
        sliced0Pct2900 = sliced0Pct2900[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced1Pct2300 = allModels.allModelSpectra1Pct2300.planetReflection.to_numpy()
        sliced1Pct2300 = sliced1Pct2300[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced1Pct2600 = allModels.allModelSpectra1Pct2600.planetReflection.to_numpy()
        sliced1Pct2600 = sliced1Pct2600[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced1Pct2900 = allModels.allModelSpectra1Pct2900.planetReflection.to_numpy()
        sliced1Pct2900 = sliced1Pct2900[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced10Pct2300 = allModels.allModelSpectra10Pct2300.planetReflection.to_numpy()
        sliced10Pct2300 = sliced10Pct2300[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced10Pct2600 = allModels.allModelSpectra10Pct2600.planetReflection.to_numpy()
        sliced10Pct2600 = sliced10Pct2600[tempSliceStartInt:tempSliceEndInt + 1]
        
        sliced10Pct2900 = allModels.allModelSpectra10Pct2900.planetReflection.to_numpy()
        sliced10Pct2900 = sliced10Pct2900[tempSliceStartInt:tempSliceEndInt + 1]
        
        # plt.yscale('log')
        plt.xlim(bin_min, bin_max)
        plt.ylim(97.5, 100.5)
        plt.plot(tempSliceWavelength, (((sliced0Pct2300 - sliced0Pct2300) / max(sliced0Pct2300)) * 100) + 100, label='0 % Spots')
        plt.plot(tempSliceWavelength, (((sliced1Pct2300 - sliced0Pct2300) / max(sliced0Pct2300)) * 100) + 100, label='1 % Spots')
        plt.plot(tempSliceWavelength, (((sliced10Pct2300 - sliced0Pct2300) / max(sliced0Pct2300)) * 100) + 100, label='10 % Spots')
        plt.legend()
        plt.title("%s Feature Variation" % bin)
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Percent Difference Compared to Static Star")
        plt.savefig('./AbSciConPhaseCurvePlots/Figures/2300SpotTemp/%s_phase%d.png' % (bin, phase), bbox_inches='tight')
        # plt.show()
        plt.close('all')
        
def plot_planet_phasecurve_one_WL(Params, allModels, phase, wl, phase_curve_x, phase_curve_y_0, phase_curve_y_1, phase_curve_y_10, spotTemp, temp_star_x, temp_star_y, temp_planet_y):
    # Find location of the wavelength in the data array
    loc = np.where(allModels.allModelSpectra.wavelength >= wl)
    loc = loc[0][0]
    
    # Load in the data for star model with 0, 1, and 10 pct coverage with spot
    # temps of 2300, 2600, or 2900 (depending on spotTemp) for the current phase
    phase_curve_x.append((phase * Params.delta_phase_planet) - 180)
    
    if spotTemp == 2300:
        # plt.ylim(1e-19, 20e-19) # wl 0um
        # plt.ylim(1e-19, 20e-19) # wl 1um
        # plt.ylim(1e-20, 1e-19) # wl 1.36um
        # plt.ylim(1e-23, 30e-23) # wl 2um
        # plt.ylim(1e-19, 20e-19) # wl 4um UNNECCESSARY
        # plt.ylim(1e-19, 20e-19) # wl 6um
        # plt.ylim(1e-19, 20e-19) # wl 8um
        allModels.allModelSpectra0Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra1Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
        
        phase_curve_y_0.append(allModels.allModelSpectra0Pct2300.planetReflection[loc])
        phase_curve_y_1.append(allModels.allModelSpectra1Pct2300.planetReflection[loc])
        phase_curve_y_10.append(allModels.allModelSpectra10Pct2300.planetReflection[loc])
        
        temp_planet_y.append(allModels.allModelSpectra10Pct2300.planetReflection[loc])
        temp_star_y.append(allModels.allModelSpectra10Pct2300.sumflux[loc])
        temp_star_x.append(Params.delta_phase_star * phase)
        
        plt.yscale('log')
        plt.plot(temp_star_x, temp_star_y, label='Star')
        plt.title('Stellar Phase Curve: %.3f um | %d Spot Temp' % (wl, spotTemp))
        plt.xlabel('Phase (Degrees)')
        plt.ylabel('Flux (W/um/m^2)')
        plt.legend(loc='lower left')
        wl_string = int(wl)
        plt.savefig('./AbSciConPhaseCurvePlots/Figures/%sSpotTemp/PhaseCurve_%s_um/StarOnly_phase%d.png' % (spotTemp, wl_string, phase), bbox_inches='tight')
        plt.close("all")
        
        plt.yscale('log')
        plt.plot(phase_curve_x, temp_planet_y, label='Planet')
        plt.title('Planet Phase Curve: %.3f um | %d Spot Temp' % (wl, spotTemp))
        plt.xlabel('Phase (Degrees)')
        plt.ylabel('Flux (W/um/m^2)')
        plt.legend(loc='lower left')
        wl_string = int(wl)
        plt.savefig('./AbSciConPhaseCurvePlots/Figures/%sSpotTemp/PhaseCurve_%s_um/PlanetOnly_phase%d.png' % (spotTemp, wl_string, phase), bbox_inches='tight')
        plt.close("all")
    
    elif spotTemp == 2600:
        # plt.ylim(1e-19, 20e-19) # wl 0um UNNECCESARY
        # plt.ylim(1e-19, 20e-19) # wl 1um
        # plt.ylim(1e-20, 1e-19) # wl 1.36um
        # plt.ylim(1e-23, 30e-23) # wl 2um
        # plt.ylim(1e-19, 20e-19) # wl 4um UNNECCESSARY
        # plt.ylim(1e-19, 20e-19) # wl 6um
        # plt.ylim(1e-19, 20e-19) # wl 8um
        allModels.allModelSpectra0Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra1Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
    
        phase_curve_y_0.append(allModels.allModelSpectra0Pct2600.planetReflection[loc])
        phase_curve_y_1.append(allModels.allModelSpectra1Pct2600.planetReflection[loc])
        phase_curve_y_10.append(allModels.allModelSpectra10Pct2600.planetReflection[loc])
        
    elif spotTemp == 2900:
        # plt.ylim(1e-19, 20e-19) # wl 0um
        # plt.ylim(1e-19, 20e-19) # wl 1um
        # plt.ylim(1e-20, 1e-19) # wl 1.36um
        # plt.ylim(1e-23, 30e-23) # wl 2um
        # plt.ylim(1e-19, 20e-19) # wl 4um UNNECCESSARY
        # plt.ylim(1e-19, 20e-19) # wl 6um
        # plt.ylim(1e-19, 20e-19) # wl 8um
        allModels.allModelSpectra0Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra1Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
    
        phase_curve_y_0.append(allModels.allModelSpectra0Pct2900.planetReflection[loc])
        phase_curve_y_1.append(allModels.allModelSpectra1Pct2900.planetReflection[loc])
        phase_curve_y_10.append(allModels.allModelSpectra10Pct2900.planetReflection[loc])
    
    # Load in the data for star model with 0, 1, and 10 pct coverage with spot
    # temps of 2300, 2600, and 2900 for the current phase
    
    # phase_curve_x.append((phase * Params.delta_phase_planet) - 180)
    
    plt.yscale('log')
    plt.plot(phase_curve_x, phase_curve_y_0, label='0% Spots')
    plt.plot(phase_curve_x, phase_curve_y_1, label='1% Spots')
    plt.plot(phase_curve_x, phase_curve_y_10, label='10% Spots')
    plt.title('Planet Phase Curve: %.3f um | %d Spot Temp' % (wl, spotTemp))
    plt.xlabel('Planet Phase (Degrees)')
    plt.ylabel('Flux (W/um/m^2)')
    plt.legend(loc='lower left')
    wl_string = int(wl)
    plt.savefig('./AbSciConPhaseCurvePlots/Figures/%sSpotTemp/PhaseCurve_%s_um/phase%d.png' % (spotTemp, wl_string, phase), bbox_inches='tight')
    plt.close("all")
    
    return phase_curve_x, phase_curve_y_0, phase_curve_y_1, phase_curve_y_10, temp_star_x, temp_star_y, temp_planet_y

def plot_planet_PS_Ratio_one_WL(Params, allModels, phase, wl, phase_curve_x, phase_curve_y_0, phase_curve_y_1, phase_curve_y_10, spotTemp):
    # Find location of the wavelength in the data array
    loc = np.where(allModels.allModelSpectra.wavelength >= wl)
    loc = loc[0][0]
    
    # Load in the data for star model with 0, 1, and 10 pct coverage with spot
    # temps of 2300, 2600, or 2900 (depending on spotTemp) for the current phase
    
    if spotTemp == 2300:
        # plt.ylim(1e-19, 20e-19) # wl 0um
        # plt.ylim(1e-19, 20e-19) # wl 1um
        # plt.ylim(1e-20, 1e-19) # wl 1.36um
        # plt.ylim(1e-23, 30e-23) # wl 2um
        # plt.ylim(1e-19, 20e-19) # wl 4um UNNECCESSARY
        # plt.ylim(1e-19, 20e-19) # wl 6um
        # plt.ylim(1e-19, 20e-19) # wl 8um
        allModels.allModelSpectra0Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra1Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2300 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2300SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
        
        if allModels.planetPhase == 182 and allModels.zero_flag:
            allModels.allModelSpectra_phase0_0pct_2300 = allModels.allModelSpectra0Pct2300
            allModels.allModelSpectra_phase0_1pct_2300 = allModels.allModelSpectra1Pct2300
            allModels.allModelSpectra_phase0_10pct_2300 = allModels.allModelSpectra10Pct2300
        
        phase_curve_y_0.append((allModels.allModelSpectra0Pct2300.planetReflection[loc] + allModels.allModelSpectra0Pct2300.sumflux[loc]) / allModels.allModelSpectra_phase0_0pct_2300.sumflux[loc])
        phase_curve_y_1.append((allModels.allModelSpectra1Pct2300.planetReflection[loc] + allModels.allModelSpectra1Pct2300.sumflux[loc]) / allModels.allModelSpectra_phase0_1pct_2300.sumflux[loc])
        phase_curve_y_10.append((allModels.allModelSpectra10Pct2300.planetReflection[loc] + allModels.allModelSpectra10Pct2300.sumflux[loc]) / allModels.allModelSpectra_phase0_10pct_2300.sumflux[loc])
    
    elif spotTemp == 2600:
        # plt.ylim(1e-19, 20e-19) # wl 0um UNNECCESARY
        # plt.ylim(1e-19, 20e-19) # wl 1um
        # plt.ylim(1e-20, 1e-19) # wl 1.36um
        # plt.ylim(1e-23, 30e-23) # wl 2um
        # plt.ylim(1e-19, 20e-19) # wl 4um UNNECCESSARY
        # plt.ylim(1e-19, 20e-19) # wl 6um
        # plt.ylim(1e-19, 20e-19) # wl 8um
        allModels.allModelSpectra0Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra1Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2600 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2600SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")

        if allModels.planetPhase == 182 and allModels.zero_flag:
            allModels.allModelSpectra_phase0_0pct_2600 = allModels.allModelSpectra0Pct2600
            allModels.allModelSpectra_phase0_1pct_2600 = allModels.allModelSpectra1Pct2600
            allModels.allModelSpectra_phase0_10pct_2600 = allModels.allModelSpectra10Pct2600
        
        phase_curve_y_0.append((allModels.allModelSpectra0Pct2600.planetReflection[loc] + allModels.allModelSpectra0Pct2600.sumflux[loc]) / allModels.allModelSpectra_phase0_0pct_2600.sumflux[loc])
        phase_curve_y_1.append((allModels.allModelSpectra1Pct2600.planetReflection[loc] + allModels.allModelSpectra1Pct2600.sumflux[loc]) / allModels.allModelSpectra_phase0_1pct_2600.sumflux[loc])
        phase_curve_y_10.append((allModels.allModelSpectra10Pct2600.planetReflection[loc] + allModels.allModelSpectra10Pct2600.sumflux[loc]) / allModels.allModelSpectra_phase0_10pct_2600.sumflux[loc])
        
    elif spotTemp == 2900:
        # plt.ylim(1e-19, 20e-19) # wl 0um
        # plt.ylim(1e-19, 20e-19) # wl 1um
        # plt.ylim(1e-20, 1e-19) # wl 1.36um
        # plt.ylim(1e-23, 30e-23) # wl 2um
        # plt.ylim(1e-19, 20e-19) # wl 4um UNNECCESSARY
        # plt.ylim(1e-19, 20e-19) # wl 6um
        # plt.ylim(1e-19, 20e-19) # wl 8um
        allModels.allModelSpectra0Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/0PctSpots/phase%d.csv' % phase, sep=",")
        allModels.allModelSpectra1Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/1PctSpots/phase%d.csv' % (phase), sep=",")
        allModels.allModelSpectra10Pct2900 = pd.read_csv('./AbSciConPhaseCurvePlots/Data/2900SpotTemp/10PctSpots/phase%d.csv' % (phase), sep=",")
    
        if allModels.planetPhase == 182 and allModels.zero_flag:
            allModels.allModelSpectra_phase0_0pct_2900 = allModels.allModelSpectra0Pct2900
            allModels.allModelSpectra_phase0_1pct_2900 = allModels.allModelSpectra1Pct2900
            allModels.allModelSpectra_phase0_10pct_2900 = allModels.allModelSpectra10Pct2900
    
        phase_curve_y_0.append((allModels.allModelSpectra0Pct2900.planetReflection[loc] + allModels.allModelSpectra0Pct2900.sumflux[loc]) / allModels.allModelSpectra_phase0_0pct_2900.sumflux[loc])
        phase_curve_y_1.append((allModels.allModelSpectra1Pct2900.planetReflection[loc] + allModels.allModelSpectra1Pct2900.sumflux[loc]) / allModels.allModelSpectra_phase0_1pct_2900.sumflux[loc])
        phase_curve_y_10.append((allModels.allModelSpectra10Pct2900.planetReflection[loc] + allModels.allModelSpectra10Pct2900.sumflux[loc]) / allModels.allModelSpectra_phase0_10pct_2900.sumflux[loc])
    
    # Load in the data for star model with 0, 1, and 10 pct coverage with spot
    # temps of 2300, 2600, and 2900 for the current phase
    
    phase_curve_x.append((phase * Params.delta_phase_planet) - 180)
    
    # plt.yscale('log')
    plt.ylim(.990, 1.0175)
    plt.plot(phase_curve_x, phase_curve_y_0, label='0% Spots')
    plt.plot(phase_curve_x, phase_curve_y_1, label='1% Spots')
    plt.plot(phase_curve_x, phase_curve_y_10, label='10% Spots')
    plt.title('(P flux + S flux) / Initial S flux: %.3f um | %d Spot Temp' % (wl, spotTemp))
    plt.xlabel('Planet Phase (Degrees)')
    plt.ylabel('(Pf + Sf) / Sf0')
    plt.legend(loc='lower left')
    wl_string = int(wl)
    plt.savefig('./AbSciConPhaseCurvePlots/Figures/%sSpotTemp/PhaseCurve_%s_um/PS_phase%d.png' % (spotTemp, wl_string, phase), bbox_inches='tight')
    plt.close("all")
    
    return phase_curve_x, phase_curve_y_0, phase_curve_y_1, phase_curve_y_10

def make_gifs(Params, filePath, saveName, removeFirst, loop, duration, phaseType):
    frames = []
    for image in range(Params.total_images):
        star_phase = (Params.delta_phase_star * image) % 360
        star_phase_string = str("%.3f" % star_phase)
        try:
            if phaseType == 'int':
                frames.append(Image.open(filePath + "_%d.png" % image))
            elif phaseType == 'string':
                frames.append(Image.open(filePath + "_%s.png" % star_phase_string))
        except FileNotFoundError:
            return None
    # frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/PSGandVariablePlanetFluxComparison*.png")]
    if removeFirst:
        del frames[0]
    frame_one = frames[0]
    frame_one.save(saveName, format="GIF", append_images=frames, save_all=True, duration=duration, loop=loop)

if __name__ == "__main__":
    Params = read_info.ParamModel()

    allModels = read_info.ReadStarModels(Params.starName)

    # Instantiate x and y data lists used for the lightcurve, both seen by the observer and the planet
    x_lightCurve = []
    x_lightCurvePlanet = []
    y_lightCurve  = []
    y_lightCurvePlanet = []
    x_maxChange = []
    y_maxChange = []

    allModels.maxChange = 0
    allModels.maxFlux = 0
    allModels.minFlux = 1e999
    allModels.contrastMin = 999
    allModels.contrastMax = 1e-999
    
    phase_curve_x_2300 = []
    phase_curve_y_2300_0 = []
    phase_curve_y_2300_1 = []
    phase_curve_y_2300_10 = []
    
    phase_curve_x_2600 = []
    phase_curve_y_2600_0 = []
    phase_curve_y_2600_1 = []
    phase_curve_y_2600_10 = []
    
    phase_curve_x_2900 = []
    phase_curve_y_2900_0 = []
    phase_curve_y_2900_1 = []
    phase_curve_y_2900_10 = []
    
    phase_curve_x_PS_2300 = []
    phase_curve_y_PS_2300_0 = []
    phase_curve_y_PS_2300_1 = []
    phase_curve_y_PS_2300_10 = []
    
    temp_star_x_2300 = []
    temp_star_y_2300 = []
    temp_planet_y_2300 = []
    
    phase_curve_x_PS_2600 = []
    phase_curve_y_PS_2600_0 = []
    phase_curve_y_PS_2600_1 = []
    phase_curve_y_PS_2600_10 = []
    
    phase_curve_x_PS_2900 = []
    phase_curve_y_PS_2900_0 = []
    phase_curve_y_PS_2900_1 = []
    phase_curve_y_PS_2900_10 = []

    allModels.zero_flag = True
    ninety_flag = True
    eclipse_flag = True
    
    # Tracks the current revolution number of the planet
    revolution=0
    
    # Turns true first time we hit 180
    flag180_first = False
    flag180_second = False
    done = False

    for phase in range(Params.total_images):
        
        # Read in all of the appropriate information regarding thevariable star and planet fluxes
        allModels.allModelSpectra = pd.read_csv(Path('.') / f'{Params.starName}' / 'Data' / 'AllModelSpectraValues' / f'phase{phase}.csv', sep=",")
        allModels.allModelSpectra0Spots = pd.read_csv(Path('.') / 'test' / 'Data' / 'AllModelSpectraValues' / f'phase{phase}.csv', sep=",")

        # The phase of the planet, according to observer, compared to starting point (0 degrees)
        allModels.planetPhase = (Params.phase1 + Params.delta_phase_planet * phase)
        if allModels.planetPhase >= 360:
            # revolution += 1
            pass
            
        allModels.planetPhase = allModels.planetPhase % 360

        # Record the flux spectra of the initial 0, 90, and 180 phases of the planet. Compare subsequent 0, 90, and 180
        # degree planet reflection flux to this original to see how the planet varies due to the star's variation.
        if allModels.planetPhase == 0 and allModels.zero_flag:
            allModels.allModelSpectra0 = allModels.allModelSpectra
            # zero_flag = False

        if allModels.planetPhase == 90 and ninety_flag:
            allModels.allModelSpectra90 = allModels.allModelSpectra
            # ninety_flag = False
        print(allModels.planetPhase)
        if allModels.planetPhase == 180 and eclipse_flag:
            allModels.allModelSpectra180 = allModels.allModelSpectra
            eclipse_flag = False

        # PSG can't calculate the planet's values at phase 180 (in front of star, no reflection), so it calculates them at phase 182.
        if round(allModels.planetPhase,3) == 180.00:
            allModels.planetPhase = 182.00

        # EDIT LATER
        # The way this GCM was created, Phase 176-185 are calculated as if in transit, so we must use phase 186
        # in place of 185 or else the lower wavelength flux values will still be 0.
        if allModels.planetPhase == 185:
            allModels.planetPhase = 186
        
        # Load in the initial values from PSG, with a static star
        allModels.PSGPlanetReflectionModel = pd.read_csv(
                                             Path('.') / f'{Params.starName}' / 'Data' / 'PSGCombinedSpectra' / f'phase{allModels.planetPhase:.3f}.txt',
                                            comment='#',
                                            delim_whitespace=True,
                                            names=["wavelength", "total", "stellar", "planet"],
                                            )

        if Params.plotLightCurve:
            x_lightCurve, y_lightCurve = plot_light_curve(allModels, Params, phase, x_lightCurve, y_lightCurve)
            x_lightCurvePlanet, y_lightCurvePlanet = plot_light_curve(allModels, Params, phase, x_lightCurvePlanet,
                                                                      y_lightCurvePlanet, towardsPlanet=True)
        
        if Params.plotBinnedVariations:
            # Plot water feature
            plot_wavelength_bin_flux_variation(Params, allModels, phase, 'H2O', revolution)
            # Plot CO2 feature
            plot_wavelength_bin_flux_variation(Params, allModels, phase, 'CO2', revolution)
        
        # Plots the varying planet reflection flux divided by the original planet flux
        # from PSG which was created with a static star.
        if Params.plotPlanetContrast:
            planet_contrast(Params, allModels, phase)

        if allModels.planetPhase == 0 and Params.plotPlanetVariationContrast:
            planet_variation_contrast(Params, allModels, phase, initial_phase=0, revolution=revolution, flag=allModels.zero_flag)
            # zero_flag = False

        if allModels.planetPhase == 90 and Params.plotPlanetVariationContrast:
            planet_variation_contrast(Params, allModels, phase, initial_phase=90, revolution=revolution, flag=ninety_flag)
            ninety_flag= False
            revolution += 1
            
        # if allModels.planetPhase == 182 and Params.plotPlanetVariationContrast:
        #     planet_variation_contrast(Params, allModels, phase, initial_phase=180, revolution=revolution, flag=eclipse_flag)
        #     eclipse_flag = False
        #     revolution += 1

        # Calculate the average sumflux of the current phase and subtract the initial sumflux to determine
        # the difference. Highlights how much the stellar flux is affected by the variable surface.
        currentPhaseAverageFlux = statistics.mean(allModels.allModelSpectra.sumflux.values)
        if currentPhaseAverageFlux > allModels.maxFlux:
            allModels.maxFlux = currentPhaseAverageFlux
        if currentPhaseAverageFlux < allModels.minFlux:
            allModels.minFlux = currentPhaseAverageFlux
        change = abs(allModels.maxFlux - allModels.minFlux)
        if Params.plotMaxFluxChange and change > allModels.maxChange:
            allModels.maxChange = change
            x_maxChange, y_maxChange = max_flux_change(Params, allModels, phase, x_maxChange, y_maxChange)
            

        if Params.plotStellarFlux:
            plot_stellar_flux(Params, allModels, phase)
            
        if Params.plotAdjustedPlanetFlux:
            plot_adjusted_planet_flux(Params, allModels, phase)
            
        if Params.plotPlanetFluxVariation:
            plot_planet_flux_variation(Params, allModels, phase)
        
        # MOVED ABOVE planet_variation_contrast
        # if Params.plotBinnedVariations:
        #     # Plot water feature
        #     plot_wavelength_bin_flux_variation(Params, allModels, phase, 'H2O', revolution)
        #     # Plot CO2 feature
        #     plot_wavelength_bin_flux_variation(Params, allModels, phase, 'CO2', revolution)
        
        # Diagnostic plot to view the unaltered values from PSG
        # plot_PSG_planet_reflection(Params, allModels, phase)
        
        # AbSciCon Presentation plots dealing with phase curve of a planet at a single wavelength
        # 0.5 um
        # wl = .5003964022
        # 1um
        # wl = 1.000952687
        # wl = 1.361578439
        # 2.75 um
        # wl = 2.750937
        # 4.22 um
        wl = 4.218645693
        # 6.27 um
        # wl = 6.265877722999999
        # 8 um
        # wl = 7.995468972999995
        
        if allModels.planetPhase == 182:
            if flag180_first:
                flag180_second = True
            flag180_first = True
            
        
        # Plot Planet Phase Curves across 1 rotation?
        if flag180_first and not done:
            # phase_curve_x_2300, phase_curve_y_2300_0, phase_curve_y_2300_1, phase_curve_y_2300_10, temp_star_x_2300, temp_star_y_2300, temp_planet_y_2300 = plot_planet_phasecurve_one_WL(Params, allModels, phase, wl, phase_curve_x_2300, phase_curve_y_2300_0, phase_curve_y_2300_1, phase_curve_y_2300_10, 2300, temp_star_x_2300, temp_star_y_2300, temp_planet_y_2300)
            # phase_curve_x_2600, phase_curve_y_2600_0, phase_curve_y_2600_1, phase_curve_y_2600_10 = plot_planet_phasecurve_one_WL(Params, allModels, phase, wl, phase_curve_x_2600, phase_curve_y_2600_0, phase_curve_y_2600_1, phase_curve_y_2600_10, 2600)
            # phase_curve_x_2900, phase_curve_y_2900_0, phase_curve_y_2900_1, phase_curve_y_2900_10 = plot_planet_phasecurve_one_WL(Params, allModels, phase, wl, phase_curve_x_2900, phase_curve_y_2900_0, phase_curve_y_2900_1, phase_curve_y_2900_10, 2900)
        
            # phase_curve_x_PS_2300, phase_curve_y_PS_2300_0, phase_curve_y_PS_2300_1, phase_curve_y_PS_2300_10 = plot_planet_PS_Ratio_one_WL(Params, allModels, phase, wl, phase_curve_x_PS_2300, phase_curve_y_PS_2300_0, phase_curve_y_PS_2300_1, phase_curve_y_PS_2300_10, 2300)
            # phase_curve_x_PS_2600, phase_curve_y_PS_2600_0, phase_curve_y_PS_2600_1, phase_curve_y_PS_2600_10 = plot_planet_PS_Ratio_one_WL(Params, allModels, phase, wl, phase_curve_x_PS_2600, phase_curve_y_PS_2600_0, phase_curve_y_PS_2600_1, phase_curve_y_PS_2600_10, 2600)
            # phase_curve_x_PS_2900, phase_curve_y_PS_2900_0, phase_curve_y_PS_2900_1, phase_curve_y_PS_2900_10 = plot_planet_PS_Ratio_one_WL(Params, allModels, phase, wl, phase_curve_x_PS_2900, phase_curve_y_PS_2900_0, phase_curve_y_PS_2900_1, phase_curve_y_PS_2900_10, 2900)

            allModels.zero_flag = False
            
        if flag180_second == True:
            done = True
        

            
    if Params.plotLightCurveHemiCombo:
        plot_lc_hemi_combo(Params, allModels, phase)
        
    if Params.plotLightCurveContrastCombo:
        plot_lc_planet_contrast_combo(Params, allModels, phase)
    
    if Params.makeGifs:
        # The 'duration' parameter is the number of milliseconds each image appears in the GIF. 150ms is the default, or normal,
        # value. 300 is a longer duration that creates a 'slower' gif.
        
        # make_gifs(Params, "./%s/Figures/PlanetPlots/PlanetFlux/PSGandVariablePlanetFluxComparison" % Params.starName,
        #           "./%s/Figures/GIFs/StaticAndVariablePlanetFluxComparison.gif" % Params.starName, removeFirst=False, loop=0, duration=100, phaseType='int')
        # make_gifs(Params, "./%s/Figures/PlanetPlots/PlanetFlux/PlanetPhaseFlux" % Params.starName,
        #           "./%s/Figures/GIFs/VariablePlanetPhaseFlux.gif" % Params.starName, removeFirst=False, loop=0, duration=100, phaseType='int')
        # The first light curve graph created is blank; remove the first image of the gif before saving
        make_gifs(Params, "./%s/Figures/LightCurvesObserver/LightCurve" % Params.starName,
                  "./%s/Figures/GIFs/LightCurveObserver.gif" % Params.starName, removeFirst=True, loop=0, duration=150, phaseType='int')
        make_gifs(Params, "./%s/Figures/LightCurvesPlanet/LightCurve" % Params.starName,
                  "./%s/Figures/GIFs/LightCurvePlanet.gif" % Params.starName, removeFirst=True, loop=0, duration=100, phaseType='int')
        make_gifs(Params, "./%s/Figures/HemiMapImages/HemiMap" % Params.starName,
                  "./%s/Figures/GIFs/StarRotation.gif" % Params.starName, removeFirst=False, loop=0, duration=150, phaseType='string')
        make_gifs(Params, "./%s/Figures/Hemi+LightCurve/Hemi+LightCurve" % Params.starName,
                  "./%s/Figures/GIFs/LCHemiMapCombo.gif" % Params.starName, removeFirst=False, loop=0, duration=50, phaseType='int')
        make_gifs(Params, "./%s/Figures/PlanetPlots/VariableAndPSGContrast/variableAndPSGContrast" % Params.starName,
                  "./%s/Figures/GIFs/PSGandVariablePlanetFluxContrast.gif" % Params.starName, removeFirst=False, loop=0, duration=150, phaseType='int')
        make_gifs(Params, "./%s/Figures/PlanetContrast+LightCurve/PlanetContrast+LightCurve" % Params.starName,
                  "./%s/Figures/GIFs/LCPlanetContrastCombo.gif" % Params.starName, removeFirst=False, loop=0, duration=150, phaseType='int')
        # make_gifs(Params, "./%s/Figures/PlanetPlots/PlanetFlux/PSGandVariablePlanetFluxComparison" % Params.starName,
        #           "./%s/Figures/GIFs/LCPlanetContrastCombo2.gif" % Params.starName, removeFirst=False, loop=0, duration=100, phaseType='int')