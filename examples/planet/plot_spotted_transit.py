"""
Observe a transit of a spotted star.
========================================

This example demonstrates stellar contamination of a transit spectrum.

A transit can change the spot coverage of a star and produce a signal
that is difficult to distinguish from atmospheric absorption. We aim to simulate
data from :cite:t:`2023ApJ...948L..11M`.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pathlib import Path

from VSPEC import ObservationModel,PhaseAnalyzer
from VSPEC import params
from VSPEC.config import MSH

SEED = 10


# %%
# Create the needed configurations
# --------------------------------
# 
# :cite:t:`2023ApJ...948L..11M` observed super-Earth GJ 486b with
# JWST NIRSpec/G395H

# Instrument

inst = params.InstrumentParameters(
    telescope=params.SingleDishParameters.jwst(),
    bandpass=params.BandpassParameters(
        wl_blue=2.87*u.um,
        wl_red=5.14*u.um,
        resolving_power=200,
        wavelength_unit=u.um,
        flux_unit=u.Unit('W m-2 um-1')
    ),
    detector=params.DetectorParameters(
        beam_width=5*u.arcsec,
        integration_time=0.5*u.s,
        ccd=params.ccdParameters(
            pixel_sampling=8,
            read_noise=16.8*u.electron,
            dark_current=0.005*u.electron/u.s,
            throughput=0.3,
            emissivity=0.1,
            temperature=50*u.K
        )
    )
)

# Observation

observation = params.ObservationParameters(
    observation_time=3.53*u.hr,
    integration_time=8*u.min
)

# PSG

psg_kwargs = dict(
    gcm_binning=5,
    phase_binning=1,
    nmax=0,
    lmax=0,
    continuum=['Rayleigh', 'Refraction'],
    url='http://localhost:3000',
    api_key=params.APIkey.none()
)
psg_params = params.psgParameters(
    use_molecular_signatures=True,
    **psg_kwargs
)
psg_no_atm = params.psgParameters(
    use_molecular_signatures=False,
    **psg_kwargs
)

# Star and Planet

star_teff = 3291*u.K
star_rad = 0.339*u.R_sun
planet_rad = 1.305*u.R_earth
orbit_rad = 0.01734*u.AU
orbit_period = 1.467119*u.day
planet_rot_period = orbit_period
star_rot_period = np.inf*u.s # assume the star does not change.
planet_mass = 2.82*u.M_earth
star_mass = 0.323*u.M_sun
inclination = 88.4*u.deg

observation_angle = (2*np.pi*u.rad * observation.observation_time/orbit_period).to(u.deg)
initial_phase = 180*u.deg - 0.5*observation_angle

planet_params = params.PlanetParameters(
    name='GJ486b',
    radius=planet_rad,
    gravity=params.GravityParameters('kg',planet_mass),
    semimajor_axis=orbit_rad,
    orbit_period=orbit_period,
    rotation_period=planet_rot_period,
    eccentricity=0,
    obliquity=0*u.deg,
    obliquity_direction=0*u.deg,
    init_phase=initial_phase,
    init_substellar_lon=0*u.deg
)

system_params = params.SystemParameters(
    distance=8.07*u.pc,
    inclination=inclination,
    phase_of_periasteron=0*u.deg
)


star_dict = {
    'teff': star_teff,
    'radius': star_rad
}
planet_dict = {'semimajor_axis': orbit_rad}

gcm_dict = {
    'nlayer': 30,
    'nlon': 30,
    'nlat': 15,
    'epsilon': 100,
    'albedo': 0.3,
    'emissivity': 1.0,
    'gamma': 1.4,
    'psurf': 1*u.bar,
    'ptop': 1e-8*u.bar,
    'wind': {'U': '0 m/s','V':'0 m/s'},
    'molecules':{'H2O':0.99}
}

# Create two sets of GCM Parameters

h2o_atm = {'molecules':{'H2O':0.99}}
gcm_h2o = params.gcmParameters.from_dict({
    'star':star_dict,
    'planet':planet_dict,
    'gcm':{'vspec':dict(gcm_dict,**h2o_atm),'mean_molec_weight':18}
})
star_kwargs = dict(
    psg_star_template='M',
    teff=star_teff,
    mass=star_mass,
    radius=star_rad,
    period=star_rot_period,
    misalignment=0*u.deg,
    misalignment_dir=0*u.deg,
    ld=params.LimbDarkeningParameters.proxima(),
    faculae=params.FaculaParameters.none(),
    flares=params.FlareParameters.none(),
    granulation=params.GranulationParameters.none(),
    Nlat=500,Nlon=1000
)
quiet_star = params.StarParameters(
    spots=params.SpotParameters.none(),
    **star_kwargs
)
spotted_star = params.StarParameters(
    spots=params.SpotParameters(
            'iso', 0.025, 0., 0.*u.s,
            300*MSH, 0.1,
            2700*u.K, 2700*u.K,
            0./u.day, 0*MSH/u.day,
            10*MSH),
    **star_kwargs
)

# Set parameters for simulation
header_kwargs = dict(
    teff_min=2300*u.K,teff_max=3400*u.K,
    seed = SEED
)
internal_params_kwargs = dict(
    planet=planet_params,
    system=system_params,
    obs=observation,
    gcm=gcm_h2o,
    inst=inst
)

# Make the three cases

params_rock_quiet = params.InternalParameters(
    header=params.Header(data_path=Path('rock_quiet'),**header_kwargs),
    star = quiet_star,
    psg=psg_no_atm,
    **internal_params_kwargs
)
params_h2o_quiet = params.InternalParameters(
    header=params.Header(data_path=Path('h2o_quiet'),**header_kwargs),
    star = quiet_star,
    psg=psg_params,
    **internal_params_kwargs
)

params_rock_spotted = params.InternalParameters(
    header=params.Header(data_path=Path('rock_spotted'),**header_kwargs),
    star = spotted_star,
    psg=psg_no_atm,
    **internal_params_kwargs
)

# %%
# Run VSPEC for the simplest case
# -------------------------------
#
# We read in the config file and run the model.

model_rock_quiet = ObservationModel(params_rock_quiet)
model_rock_quiet.bin_spectra()
model_rock_quiet.build_planet()
model_rock_quiet.build_spectra()

# %%
# Load in the data
# ----------------
#
# We can use VSPEC to read in the synthetic
# data we just created.

data_rock_quiet = PhaseAnalyzer(model_rock_quiet.dirs['all_model'])

# %%
# Plot the transit
# ----------------
#
# Lets plot the lightcurve of each channel.

def plot_transit(data:PhaseAnalyzer,title:str,color:str):
    time_from_mid_transit = (data.time-0.5*(data.time[-1]+data.time[0])).to(u.hour)

    fig,axes = plt.subplots(2,1,tight_layout=True)
    axes[0].scatter(
        time_from_mid_transit,
        data.lightcurve('total',(0,-1),normalize=0,noise=False),
        label = 'white light curve',c=color
    )
    axes[0].set_xlabel('Time past mid-transit (hour)')
    axes[0].set_ylabel('Transit depth (ppm)')
    axes[0].legend()
    axes[0].set_title(title)

    # standardize the epochs to use for analysis
    pre_transit = 8
    in_transit = 11

    unocculted_spectrum = data.spectrum('total',pre_transit)
    occulted_spectrum = data.spectrum('total',in_transit)
    lost_to_transit = unocculted_spectrum-occulted_spectrum
    transit_depth = (lost_to_transit/unocculted_spectrum).to_value(u.dimensionless_unscaled)

    axes[1].plot(data.wavelength, 1e6*(transit_depth),c=color)
    axes[1].set_xlabel(f'Wavelength ({data.wavelength.unit})')
    axes[1].set_ylabel('Transit depth (ppm)')
    ylo,yhi = axes[1].get_ylim()
    if yhi-ylo < 0.5:
        mean = 0.5*(ylo+yhi)
        axes[1].set_ylim(mean-0.25,mean+0.25)

    return fig

plot_transit(data_rock_quiet,'Spotless Star and Bare Rock','xkcd:lavender').show()


# %%
# Run the other models
# --------------------
#
# Let's do the same analysis for the other cases.
#
# Spotless Star, H2O Planet
# +++++++++++++++++++++++++

model_h2o_quiet = ObservationModel(params_h2o_quiet)
model_h2o_quiet.bin_spectra()
model_h2o_quiet.build_planet()
model_h2o_quiet.build_spectra()

data_h2o_quiet = PhaseAnalyzer(model_h2o_quiet.dirs['all_model'])

plot_transit(data_h2o_quiet,'Spotless Star and 1 bar H2O Atmosphere','xkcd:azure').show()

# %%
# Spotted Star, CO2 Planet
# ++++++++++++++++++++++++

model_rock_spotted = ObservationModel(params_rock_spotted)
model_rock_spotted.bin_spectra()
model_rock_spotted.build_planet()
model_rock_spotted.build_spectra()

data_rock_spotted = PhaseAnalyzer(model_rock_spotted.dirs['all_model'])

plot_transit(data_rock_spotted,'Spotted Star and Bare Rock','xkcd:golden yellow').show()


# %%
# Plot the observed spectra
# -------------------------
#
# Let's compare the transits. We also load in the actual JWST data.

# %%
# Get the data
# ++++++++++++
#
# Reduced data from :cite:t:`2023ApJ...948L..11M` is publicly available.
# However, you must download it from the figure caption of the online version.

import pandas as pd

filename = 'moran2023_fig3.txt'
df = pd.read_fwf(filename,colspecs=[(0,8),(9,14),(15,20),(21,25),(26,28)],
    header=20,names=['Reduction','Wave','Width','Depth','e_Depth'])
used_eureka = df['Reduction']=='Eureka'

# %%
# Make the figure
# +++++++++++++++
#

fig, ax = plt.subplots(1,1)

for data,label,color in zip(
    [data_rock_quiet,data_h2o_quiet,data_rock_spotted],
    ['Rock', 'H2O', 'Rock+Spots'],
    ['xkcd:lavender','xkcd:azure','xkcd:golden yellow']
):
    pre_transit = 8
    in_transit = 11
    unocculted_spectrum = data.spectrum('total',pre_transit)
    occulted_spectrum = data.spectrum('total',in_transit)
    lost_to_transit = unocculted_spectrum-occulted_spectrum
    transit_depth = (lost_to_transit/unocculted_spectrum).to_value(u.dimensionless_unscaled)
    ax.plot(data.wavelength,transit_depth*1e6,label=label,color=color)

# ax.errorbar(df.loc[used_eureka,'Wave'],df.loc[used_eureka,'Depth'],yerr=df.loc[used_eureka,'e_Depth'],
#     fmt='o',color='k')

ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Transit depth (ppm)')
ax.legend()





# %%
# Plot the star
# -------------
#
transit_phase = data_rock_spotted.phase[11]
model_rock_spotted.star.plot_surface(
    lat0=0*u.deg,lon0=0*u.deg,
    orbit_radius=orbit_rad,radius=planet_rad,
    phase=transit_phase,inclination = inclination
)
