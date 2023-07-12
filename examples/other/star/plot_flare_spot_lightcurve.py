"""
Plot the lightcurve of a flaring star with spots
================================================

This example plots the lightcurve caused by a
flaring star when it also has spots.
"""

from astropy import units as u
import matplotlib.pyplot as plt
from pathlib import Path

from VSPEC import ObservationModel,PhaseAnalyzer
from VSPEC import params
from VSPEC import config

SEED = 42

# %%
# Initialize the VSPEC run parameters
# -----------------------------------
#
# For this example, we will create the
# parameter objects explicitly. This can also
# be done using a YAML file.

header = params.Header(
    data_path=Path('.vspec/flare_spot_lightcurve'),
    teff_min=2900*u.K,
    teff_max=3400*u.K,
    seed=SEED,verbose=0
)

star = params.StarParameters(
    psg_star_template='M',
    teff=3300*u.K,
    mass = 0.1*u.M_sun,
    radius=0.15*u.R_sun,
    period = 6*u.day,
    misalignment_dir=0*u.deg,
    misalignment=0*u.deg,
    ld = params.LimbDarkeningParameters.solar(),
    faculae=params.FaculaParameters.none(),
    spots=params.SpotParameters(
        distribution='iso',
        initial_coverage=0.1,
        area_mean=300*config.MSH,
        area_logsigma=0.2,
        teff_umbra=2900*u.K,
        teff_penumbra=3000*u.K,
        equillibrium_coverage=0.1,
        burn_in=0*u.day,
        growth_rate=0/u.day,
        decay_rate=0*config.MSH/u.day,
        initial_area=10*config.MSH
        ),
    flares=params.FlareParameters(
        dist_teff_mean=9000*u.K,
        dist_teff_sigma=500*u.K,
        dist_fwhm_mean=3*u.hr,
        dist_fwhm_logsigma=0.4,
        alpha=-0.829,
        beta=26.87,
        min_energy=1e32*u.erg,
        cluster_size=3
    ),
    granulation=params.GranulationParameters.none(),
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

wl_pixels = [0,300,700]
colors = ['xkcd:azure','xkcd:coral','xkcd:lavender']
time = data.time.to(u.day)
time = time-time[0]
for i,c in zip(wl_pixels,colors):
    wl = data.wavelength[i]
    lc = data.lightcurve(
        source='star',
        pixel=i,
        normalize=0
    )
    plt.plot(time,lc,label=f'{wl:.1f}',c=c)
plt.legend()
plt.xlabel(f'time ({time.unit})')
_=plt.ylabel('Flux (normalized)')