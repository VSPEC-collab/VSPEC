"""
Plot a map of the stellar surface
=================================

This example initializes a ``Star`` object and plots it.
"""
from astropy import units as u
import numpy as np

from VSPEC import variable_star_model as vsm
from VSPEC.helpers import MSH

SEED = 10
rng = np.random.default_rng(SEED)

# %%
# Initialize the star
# -------------------
#
# First, let's initialize a ``Star`` object.
# 
# It needs to be populated by spots and faculae.

n_spots = 10
n_faculae = 10
spot_area = 1000*MSH
facula_radius = 10000*u.km
facula_depth = 10000*u.km

spots = vsm.SpotCollection(
    *[
        vsm.StarSpot(
            lat=(rng.random() - 0.5)*120*u.deg,
            lon = rng.random()*360*u.deg,
            Amax = spot_area,
            A0 = spot_area,
            Teff_umbra=2700*u.K,
            Teff_penumbra=2900*u.K,
            r_A = 5.,
            growing = False,
            growth_rate=0./u.day,
            decay_rate=0*MSH/u.day
        ) for _ in range(n_spots)
    ]
)

faculae = vsm.FaculaCollection(
    *[
        vsm.Facula(
            lat=(rng.random() - 0.5)*120*u.deg,
            lon = rng.random()*360*u.deg,
            Rmax = facula_radius,
            R0 = facula_radius,
            Teff_floor=2500*u.K,
            Teff_wall=3700*u.K,
            lifetime=5*u.hr,
            growing=False,
            floor_threshold=200*u.km,
            Zw=facula_depth
        ) for _ in range(n_faculae)
    ]
)

star_teff = 3300*u.K
star_radius = 0.15*u.R_sun
star_period = 40*u.day
Nlat = 500
Nlon = 1000
ld_params = dict(u1=0.3,u2=0.1)

star = vsm.Star(
    Teff=star_teff,
    radius=star_radius,
    period = star_period,
    spots=spots,
    faculae=faculae,
    Nlat=Nlat,
    Nlon=Nlon,
    **ld_params
)
# %%
# Simulate the disk
# -----------------
#
# Now let's decide a viewing angle and get an image of the surface.

lon0 = 0*u.deg
lat0 = 0*u.deg

star.plot_surface(lat0,lon0)

# %%
# Add a transit
# -------------
#
# Let's throw in a transiting planet just for fun.

pl_radius = 1*u.R_earth
pl_orbit = 0.05*u.AU
inclination = 89.8*u.deg
phase = 180.4*u.deg

star.plot_surface(lat0,lon0,None,pl_orbit,pl_radius,phase,inclination)
