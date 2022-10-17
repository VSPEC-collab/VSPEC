import ast
import csv
import read_info
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d, interp2d
import variable_star_model as vsm
from astropy import units as u, constants as c

# 3rd file to run.

def calculate_combined_spectrum(allModels, Params, percentagesDict, percentagesDictTowardsPlanet, phase):
    # Creates a new column of the allModels.allModelSpectra dataframe that contains 'sumflux,' the linearly combined
    # flux output of the photosphere, spot, and faculae based on their respective surface coverage percent.
    # Essentially, it is the total flux output by the star as seen from the observer.
    photFrac = percentagesDict['phot']
    spotFrac = percentagesDict['spot']
    facFrac = percentagesDict['fac']
    allModels.allModelSpectra["sumflux"] = ((allModels.allModelSpectra.photflux * photFrac) +
                                            (allModels.allModelSpectra.spotflux * spotFrac) +
                                            (allModels.allModelSpectra.facflux * facFrac))

    # allModels.allModelSpectra.sumflux.to_csv('./%s/Data/SumfluxArraysTowardsObserver/phase%d.txt' % (Params.starName, phase))

    # Creates a new column of the allModels.allModelSpectra dataframe that contains 'sumfluxTowardsPlanet,' the linearly combined
    # flux output of the photosphere, spot, and faculae based on their respective surface coverage percent.
    # Essentially, it is the total flux output by the star as seen from the planet.
    photFracTowardsPlanet = percentagesDictTowardsPlanet['phot']
    spotFracTowardsPlanet = percentagesDictTowardsPlanet['spot']
    facFracTowardsPlanet = percentagesDictTowardsPlanet['fac']
    allModels.allModelSpectra["sumfluxTowardsPlanet"] = ((allModels.allModelSpectra.photflux * photFracTowardsPlanet) +
                                        (allModels.allModelSpectra.spotflux * spotFracTowardsPlanet) +
                                        (allModels.allModelSpectra.facflux * facFracTowardsPlanet))

    # allModels.allModelSpectra.sumfluxTowardsPlanet.to_csv('./%s/Data/SumfluxArraysTowardsPlanet/phase%d.txt' % (Params.starName, phase))
    # allModels.allModelSpectra.to_csv('./%s/Data/AllModelSpectraValues/phase%d.csv' % (Params.starName, phase), index=False, sep=',')
    # print(allModels.allModelSpectra)
    # print('wait')

def calculate_planet_flux(allModels, phase):
    # Produce only PSG's reflection flux values by subtracting the thermal flux values out, removing the planet's
    # thermal radiance spectrum
    planetReflectionOnly = abs(allModels.PSGplanetReflectionModel.planet.values - allModels.planetThermalModel.planet.values)

    # Calculate the fraction (contrast) of the PSG planet's reflected flux to PSG's stellar flux.
    # Will apply this fraction to the NextGen stellar flux to obtain the equivalent planet reflection flux
    # values as if calculated while the planet was around the NextGen star
    planetFluxFraction = np.array(planetReflectionOnly / allModels.PSGplanetReflectionModel.stellar.values)

    # Multiply the reflection fraction from the PSG data by the NextGen star's variable sumflux values
    # This simulates the planet's reflection flux if it were created around this varaiable NextGen star,
    # rather than the star used in PSG.
    # Must multiply by the phase of the star facing the planet.
    adjustedReflectionFlux = planetFluxFraction * allModels.allModelSpectra.sumfluxTowardsPlanet.values

    # Add back on the planet's thermal flux values to the adjusted reflection flux values
    allModels.allModelSpectra["planetReflection"] = adjustedReflectionFlux + allModels.planetThermalModel.planet.values
    # allModels.allModelSpectra.planetReflection.to_csv('./%s/Data/VariablePlanetFlux/phase%d.txt' % (Params.starName, phase), index=False)
    allModels.allModelSpectra['sourceTotal'] = allModels.allModelSpectra['sumflux'].values + adjustedReflectionFlux + allModels.planetThermalModel.planet.values
    # Add the planet's thermal value to the allModelSpectra dataframe
    allModels.allModelSpectra["planetThermal"] = allModels.planetThermalModel.planet.values


def calculate_noise(allModels,phase):
    """
    Load in the noise model from PSG. Scale the source noise with the
    new stellar model and add the other sources.
    """
    PSGnoise = allModels.noiseModel
    PSGsource = np.array(allModels.PSGplanetReflectionModel.total.values)
    Modelsource = np.array(allModels.allModelSpectra['sourceTotal'].values)
    Modelnoise_source = PSGnoise['Source'].values * np.sqrt(Modelsource/PSGsource)
    # Now add in quadrature
    Noise_sq = Modelnoise_source**2 + PSGnoise['Detector']**2 + PSGnoise['Telescope']**2 + PSGnoise['Background']**2
    allModels.allModelSpectra['Noise'] = np.sqrt(Noise_sq)

def interpolate_stellar_spectrum(Teff):
    model_teffs = [int(Teff - Teff%100),int(Teff - Teff%100 +100)]
    interp_data = {}
    for T in model_teffs:
        dat = pd.read_csv(Path('.') / 'NextGenModels' / 'BinnedData' / f'binned{T}StellarModel.txt',
                                names=['wavelength', 'flux'], delimiter=' ', skiprows=1)
        interp_data[T] = dat
    interp = interp2d(interp_data[model_teffs[0]]['wavelength'],model_teffs,
    [interp_data[model_teffs[0]]['flux'],interp_data[model_teffs[1]]['flux']])
    return pd.DataFrame({'wavelength': interp_data[model_teffs[0]]['wavelength'],
                        'flux': interp(interp_data[model_teffs[0]]['wavelength'],Teff)
                        })


# 3rd file to run.

if __name__ == "__main__":
    # 1) Read in all of the user-defined config parameters into a class, called Params.
    Params = read_info.ParamModel()

    # Create an object to store all of the spectra flux values
    allModels = read_info.ReadStarModels(Params.starName)

    # create the star
    vstar = vsm.Star(Params.teffStar*u.K,Params.starRadius,Params.rotstar,vsm.SpotCollection(),
    vsm.FaculaCollection(),Params.starName,Params.starDistance)
    vstar.spot_generator.coverage=Params.spotCoverage
    vstar.fac_generator.coverage=Params.facCoverage

    # create some spots --> want a steady state before we take spectra
    spot_warm_up_step = 1*u.day
    prev_nspots = 0
    while True:
        vstar.birth_spots(spot_warm_up_step)
        vstar.age(spot_warm_up_step)
        if len(vstar.spots.spots) < prev_nspots:
            break
        prev_nspots = len(vstar.spots.spots)
    
    # create some faculae
    facula_warm_up_step = 1*u.hr
    prev_nfac = 0
    while True:
        vstar.birth_faculae(facula_warm_up_step)
        vstar.age(facula_warm_up_step)
        if len(vstar.faculae.faculae) < prev_nfac:
            break
        prev_nfac = len(vstar.faculae.faculae)



    # Load in the NextGen Stellar Data for photosphere, spot, and faculae temperatures
    allModels.photModel = interpolate_stellar_spectrum(Params.teffStar)
    allModels.spotModel = interpolate_stellar_spectrum(Params.teffSpot)
    allModels.facModel = interpolate_stellar_spectrum(Params.teffFac)
    if not np.all(allModels.photModel.wavelength == allModels.spotModel.wavelength) or not np.all(allModels.photModel.wavelength == allModels.facModel.wavelength):
        raise ValueError("The star, spot, and faculae spectra should be on the same wavelength scale, and currently are not.")
    # EDIT LATER: Currently hard-coded to convert into W/m2/um

    conversion = Params.unit_conversion * Params.distanceFluxCorrection
    data = {'wavelength': allModels.photModel.wavelength,
            'photflux': allModels.photModel.flux*conversion,
            'spotflux': allModels.spotModel.flux*conversion,
            'facflux': allModels.facModel.flux*conversion}
    allModels.allModelSpectra = pd.DataFrame(data)

    # allModels.allModelSpectra.photflux *= conversion
    # allModels.allModelSpectra.spotflux *= conversion
    # allModels.allModelSpectra.facflux *= conversion

    # Load in the dictionary of surface coverage percentages for each of the star's phases
    # Actually, load into arrays for scipy.interp
    starphases = []
    phot_pct = []
    spot_pct = []
    fac_pct = []
    with open(Path('.') / f'{Params.starName}' / 'Data' / 'SurfaceCoveragePercentage' / 'surfaceCoveragePercentageDict.csv', newline='') as csvfile:
        # reader = csv.DictReader(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            valueDict = ast.literal_eval(row[1])
            starphases.append(float(row[0]))
            phot_pct.append(valueDict['phot'])
            spot_pct.append(valueDict['spot'])
            fac_pct.append(valueDict['fac'])
    #and get the last one to interp up to 360 degrees
    i_min = np.argmin(starphases)
    starphases.append(starphases[i_min]+360)
    phot_pct.append(phot_pct[i_min])
    spot_pct.append(spot_pct[i_min])
    fac_pct.append(fac_pct[i_min])

    phot_interp = interp1d(starphases,phot_pct)
    spot_interp = interp1d(starphases,spot_pct)
    fac_interp = interp1d(starphases,fac_pct)

    # For loop here to run through each "image"/number of exposures as specified in the config file
    print("\nCalculating Total System Output, Stellar, and Planetary Reflection Flux Values")
    print("------------------------------------------------------------------------------")
    for index in range(Params.total_images):
        # The current phase of the planet is the planet phase change value (between exposures) multiplied
        # by the nuber of exposures taken so far (index)
        # Example: 180
        # Planet phase change is specified by the user; how many degrees it turns between "images" of the star
        planetPhase = (Params.phase1 + Params.delta_phase_planet * index) % (360*u.deg)
        sub_obs_lon = (0*u.deg + Params.delta_phase_star*index) % (360*u.deg)
        sub_obs_lat = 90*u.deg - Params.inclinationPSG
        if planetPhase>178*u.deg and planetPhase<182*u.deg:
            planetPhase=182.0*u.deg # Add transit phase;
        if planetPhase == 185*u.deg:
            planetPhase = 186.0*u.deg


        allModels.planetPhase = planetPhase
        # The current phase of the star is the star phase change value (between exposures) multiplied
        # by the nuber of exposures taken so far (index)
        # Example: 30
        allModels.starPhase = (Params.delta_phase_star * index) % (360*u.deg)
        
        # In PSG's models, phase 0 for the planet is "behind" the star from the viewer's perspective,
        # in secondary eclipse.
        # In the variable stellar code, phase 0 of the star is the side of the star facing the observer.
        # This means that, when starting the time-series simulation, the face of the star facing the observer is phase 0,
        # but the face of the star facing the planet is whatever half the total number of exposures is.
        # If in the default example, there are 252 exposures taken, the initial face of the star looking towards the planet
        # is phase 252/2 = 126

        # The star phase currently facing the planet has to take into account where the planet is in its orbit.
        # This is calculated by taking how far the planet has rotated/revolved (one in the same for tidally locked like this),
        # plus roughly 180 degrees (an offset necessary as explained above), then simply subtract how far the star has
        # turned to figure out what phase of the star is facing the planet.
        # Modulo 360 ensures it is never above that value, and dividing by delta stellar phase
        
        # temp = int(180 / Params.delta_phase_planet)
        # allModels.starPhaseFacingPlanet = ((Params.delta_phase_planet * (temp + 1)) - allModels.starPhase) % 360
        # allModels.starPhaseFacingPlanet = ((allModels.planetPhase + 180) - (allModels.starPhase % 360)) % 360
        allModels.starPhaseFacingPlanet = (180 - allModels.starPhase + allModels.planetPhase) % 360
        percent = int((index/Params.total_images) * 100)
        # print(percent)
        if percent % 10 == 0:
            print(f'{percent:.1f}% Complete')
            print(f'planet phase = {allModels.planetPhase}')
            print(f'star phase = {allModels.starPhase}')
            print(f'star phase facing planet = {allModels.starPhaseFacingPlanet}')
        # Example:
        # deltaPlanetPhase = 10
        # deltaStellarPhase = 6.666
        # Planet starts at 180, star starts at 0, from observers perspective
        # planet phase 190, starPhase 6.6666
        # Star phase facing planet = 190 - 6.6666 = 183.3333
        # the deltaStarPhaseFacingPlanet = 3.333, 1/2 of delta stellar phase
        # Given these parameters, 1/2 of the deltaStellarPhase = the deltaStellarPhaseFacingPlanet
        
        # PSG can't calculate the planet's values at phase 180 (in front of star, no reflection), so it calculates them at phase 182.
        # if allModels.planetPhase == 180:
        #     allModels.planetPhase = 182

        # # EDIT LATER
        # # The way this GCM was created, Phase 176-185 are calculated as if in transit, so we must use phase 186
        # # in place of 185 or else the lower wavelength flux values will still be 0.
        # if allModels.planetPhase == 185:
        #     allModels.planetPhase = 186
        
        # # allModels.starPhaseFacingPlanet = round((((allModels.planetPhase + 180) - allModels.starPhase) % 360) / Params.deltaStellarPhase)
        # if allModels.starPhaseFacingPlanet == Params.total_images + 1:
        #     allModels.starPhaseFacingPlanet = 0

        # Read in the planet's reflected spectrum (in W/sr/m^2/um) for the current phase
        allModels.PSGplanetReflectionModel = pd.read_csv(
            Path('.') / f'{Params.starName}' / 'Data' / 'PSGCombinedSpectra' / f'phase{allModels.planetPhase:.3f}.rad',
            comment='#',
            delim_whitespace=True,
            names=["wavelength", "total", "noise", "stellar", "planet"],
            )

        # Read in the planet's thermal spectrum (in W/m^2/um) for the current phase
        allModels.planetThermalModel = pd.read_csv(
            Path('.') / f'{Params.starName}' / 'Data' / 'PSGThermalSpectra' / f'phase{allModels.planetPhase:.3f}.rad',
            comment='#',
            delim_whitespace=True,
            names=["wavelength", "total", "noise", "planet"],
            )
        
        # Read in the noise model from PSG
        allModels.noiseModel = pd.read_csv(
            Path('.') / f'{Params.starName}' / 'Data' / 'PSGNoise' / f'phase{allModels.planetPhase:.3f}.noi',
            comment='#',
            delim_whitespace=True,
            names=['Wave/freq','Total','Source','Detector','Telescope','Background'],
            )

        # Calculate the total output flux of this star's phase by computing a linear combination of the photosphere,
        # spot, and flux models based on what percent of the surface area those components take up
        tempPhase = round(allModels.starPhase, 3) % 360
        tempPhaseFacingPlanet = round(allModels.starPhaseFacingPlanet, 3)
        if tempPhaseFacingPlanet == 360:
            tempPhaseFacingPlanet = 0
        percentagesDict = {
            'phot': phot_interp(tempPhase),
            'spot': spot_interp(tempPhase),
            'fac': fac_interp(tempPhase)
        }
        # now normalize (because of the interpolation)
        total = np.array(list(percentagesDict.values())).sum()
        for key in percentagesDict:
            percentagesDict[key] = percentagesDict[key] /total
        percentagesDictTowardsPlanet = {
            'phot': phot_interp(tempPhaseFacingPlanet),
            'spot': spot_interp(tempPhaseFacingPlanet),
            'fac': fac_interp(tempPhaseFacingPlanet)
        }
        # now normalize (because of the interpolation)
        total = np.array(list(percentagesDictTowardsPlanet.values())).sum()
        for key in percentagesDictTowardsPlanet:
            percentagesDictTowardsPlanet[key] = percentagesDictTowardsPlanet[key] /total
            
        calculate_combined_spectrum(allModels, Params, percentagesDict, percentagesDictTowardsPlanet, index)

        calculate_planet_flux(allModels, index)
        calculate_noise(allModels, index)

        allModels.allModelSpectra.to_csv(Path('.') / f'{Params.starName}' / 'Data' / 'AllModelSpectraValues' / f'phase{index}.csv', index=False, sep=',')

print("Done")