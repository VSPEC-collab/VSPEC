"""
Plot the lightcurve stellar granulation
=======================================

This example plots the lightcurve effects caused by granulation
on the surface of a star.
"""

from astropy import units as u
import matplotlib.pyplot as plt
from pathlib import Path

from VSPEC import ObservationModel,PhaseAnalyzer
from VSPEC import params

SEED = 32

# %%
# Initialize the VSPEC run parameters
# -----------------------------------
#
# For this example, we will create the
# parameter objects explicitly. This can also
# be done using a YAML file.

header = params.Header(
    data_path=Path('.vspec/granulation_lightcurve'),
    teff_min=3000*u.K,
    teff_max=3400*u.K,
    seed=SEED,verbose=0
)
star = params.StarParameters(
    psg_star_template='M',
    teff=3300*u.K,
    mass = 0.1*u.M_sun,
    radius=0.15*u.R_sun,
    period = 10*u.day,
    misalignment_dir=0*u.deg,
    misalignment=0*u.deg,
    ld = params.LimbDarkeningParameters.solar(),
    faculae=params.FaculaParameters.none(),
    spots=params.SpotParameters.none(),
    flares=params.FlareParameters.none(),
    granulation=params.GranulationParameters(
        mean=0.2,
        amp=0.005,
        period=6*u.hr,
        dteff=300*u.K
    ),
    Nlat=500,Nlon=1000
)
planet = params.PlanetParameters.std(init_phase=180*u.deg,init_substellar_lon=0*u.deg)
system = params.SystemParameters(
    distance=1.3*u.pc,
    inclination=30*u.deg,
    phase_of_periasteron=0*u.deg
)
observation = params.ObservationParameters(
    observation_time=3*u.day,
    integration_time=30*u.min
)
psg_params = params.psgParameters(
    gcm_binning=200,
    phase_binning=1,
    use_molecular_signatures=True,
    nmax=0,
    lmax=0,
    continuum=['Rayleigh', 'Refraction', 'CIA_all'],
    url='http://localhost:3000',
    api_key=params.APIkey.none()
)
instrument = params.InstrumentParameters.niriss_soss()

gcm = params.gcmParameters(
    gcm=params.vspecGCM.earth(molecules={'CO2':1e-4}),
    mean_molec_weight=28
)

parameters = params.InternalParameters(
    header = header,
    star = star,
    planet = planet,
    system = system,
    obs=observation,
    psg = psg_params,
    inst=instrument,
    gcm = gcm
)

#%%
# Run the simulation
# ------------------
#

model = ObservationModel(params=parameters)
model.build_planet()
model.build_spectra()

# %%
# Load in the data
# ----------------
#
# We can use VSPEC to read in the synthetic
# data we just created.

data = PhaseAnalyzer(model.directories['all_model'])

wl_pixels = [0,300,500,700]
time = data.time.to(u.day)
for i in wl_pixels:
    wl = data.wavelength[i]
    lc = data.lightcurve(
        source='star',
        pixel=i,
        normalize=0
    )
    plt.plot(time,lc,label=f'{wl:.1f}')
plt.legend()
plt.xlabel(f'time ({time.unit})')
_=plt.ylabel('Flux (normalized)')