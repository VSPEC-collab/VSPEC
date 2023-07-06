"""
Plot the effect of facula depth on it's lightcurve
==================================================

This example create toy lightcurves for a set of faculae.
"""

from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from VSPEC.variable_star_model import Facula
from VSPEC.helpers import round_teff

# %%
# Create a default facula
# -----------------------
#
# Let's make it so that our facula model only has a single free
# parameter: the depth-to-radius ratio.


def make_fac(depth_over_rad: float) -> Facula:
    """Return a default facula"""
    radius = 0.005*u.R_sun
    return Facula(
        lat=0*u.deg,
        lon=0*u.deg,
        r_max=radius,
        r_init=radius,
        depth=radius*depth_over_rad,
        # None of the below parameters will affect this
        # example, but we must set them to something
        lifetime=1*u.day,
        floor_teff_slope=0*u.K/u.km,
        floor_teff_min_rad=1*u.km,
        floor_teff_base_dteff=-100*u.K,
        wall_teff_intercept=100*u.K,
        wall_teff_slope=0*u.K/u.km
    )

# %%
# Make a toy flux model
# ---------------------
#
# We can avoid using an actual stellar spectrum and
# running the ``VSPEC.main`` module if we assume a few things.
#
# Suppose the wall is 10% brigher than the photosphere and the floor
# is 10% dimmer. Also suppose that the star has a radius of 0.15 solar radii


rad_star = 0.15*u.R_sun
wall_brightness = 1.1
floor_brightness = 0.9


def relative_flux(facula: Facula, angle: u.Quantity) -> float:
    """Get the contrast from a toy flux model"""
    effective_area = facula.effective_area(angle)
    area_floor = effective_area[round_teff(facula.floor_dteff)]
    area_wall = effective_area[round_teff(facula.wall_dteff)]
    area_of_disk = np.pi*rad_star**2
    floor_fraction = (
        area_floor/area_of_disk).to_value(u.dimensionless_unscaled)
    wall_fraction = (area_wall/area_of_disk).to_value(u.dimensionless_unscaled)
    return 1 + floor_fraction*(floor_brightness - 1) + wall_fraction*(wall_brightness-1)

# %%
# Plot the brightness as a function of angle
# ------------------------------------------
#
# Let's choose a parameter space.


angles = np.linspace(0, 90, 50)*u.deg
depth_over_rad = np.logspace(-1, 1, 5)

for dor in depth_over_rad:
    facula = make_fac(dor)
    flux = np.array([
        relative_flux(facula, angle) for angle in angles
    ])
    x = np.concatenate([np.flip(-angles), angles])
    y = (np.concatenate([np.flip(flux), flux]) - 1)*1e6
    log_dor = np.log10(dor)
    color = cm.get_cmap('viridis')(0.5*(log_dor+1))
    plt.plot(x, y, label=f'log(D/R) = {log_dor:.1f}', c=color)
plt.xlabel('angle from disk center (deg)')
plt.ylabel('Relative flux (ppm)')
_ = plt.legend()
