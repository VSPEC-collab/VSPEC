import SpectraBuilder
import matplotlib.pyplot as plt
import statistics

# data to save for use in plots:
# 1) adjusted total planet flux (combination of thermal and variable stellar reflected planet)
# 2) sumflux of star for each phase facing observer
# 3) sumflux of star for each phase facing planet
# 4) 

# Strategy for saving this info
# Save in seperate folders with all phase's labeled as "descriptiveName_phase.txt" where phase is the current phase
# in integer form

def planet_plots(allModels):
    # Create a copy of the original PSG planet flux values to compare the variable stellar adjusted planet values to
    # at the end
    # UNFINISHED, JUST A PLACE HOLDER/REMINDER
    originalPSGPlanetFlux = allModels.allModelSpectra.planet.copy(deep=True)

# Plot a time-series light curve as the star rotates
def plot_light_curve(self, count, x, y):
    # Calculate the mean value for the current phase, for the current star hemisphere
    avg_sumflux = statistics.mean(self.modelspectra.sumflux)

    # The X-Axis represents the number of the images taken. Presents itself as a time series, from first image
    # to last
    adjustment = 360 / self.num_exposures
    x.append(count * adjustment)

    # The y value is the flux value of the current hemisphere image
    y.append(avg_sumflux)

    plt.plot(x, y)
    plt.yscale("log")
    plt.ylabel("Flux (W/m^2/um)")
    plt.xlabel("Phase (0-360)")
    plt.savefig('./%s/LightCurves/LightCurve_%d' % (self.starName, count), bbox_inches='tight')
    # plt.show()
    plt.close("all")

    return x, y