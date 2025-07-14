"""
Day-night mix model
===================

This example shows how to use the build-in 'two-face' planetary model.

"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cartopy.crs as ccrs
from astropy import units as u
from VSPEC import ObservationModel,PhaseAnalyzer
import libpypsg as psg

try:
    YAML_PATH = Path(__file__).parent / 'twoface.yaml'
except NameError:
    YAML_PATH = Path('twoface.yaml')

#%%
# Let's look at the GCM portion of the input file:

with open(YAML_PATH) as f:
    d = yaml.safe_load(f)['gcm']

print(yaml.dump(d))

#%%
# Initialize the VSPEC run
#
# We read in the config file.

model = ObservationModel.from_yaml(YAML_PATH)

#%%
# First, let's look at the planetary structure.

gcm = model.params.gcm.get_gcm()

#%%
# Map the planetary surface
# -------------------------

tsurf = gcm.tsurf.dat.to_value(u.K)

fig = plt.figure()
proj = ccrs.Mollweide(central_longitude=0)
ax = fig.add_subplot(projection=proj)

lats = gcm.lat_start + np.arange(gcm.shape[2]) * gcm.dlat.to_value(u.deg)
lons = gcm.lon_start + np.arange(gcm.shape[1]) * gcm.dlon.to_value(u.deg)

im=ax.pcolormesh(lons,lats,tsurf.T,cmap='inferno',transform=ccrs.PlateCarree())
ax.coastlines()
_=fig.colorbar(im,ax=ax,label='Surface Temperature (K)')

#%%
# Look at the profiles

ilon_substellar = gcm.shape[1] // 2
ilon_antistellar = 0
ilon_terminator = gcm.shape[1] // 4

ilat_equator = gcm.shape[2] // 2

plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(wspace=0.4)

for ilon, label,c in zip([ilon_substellar, ilon_antistellar, ilon_terminator], ['Substellar', 'Antistellar', 'Terminator'], ['b', 'k', 'r']):
    pressure = gcm.pressure.dat[:, ilon, ilat_equator]
    temperature = gcm.temperature.dat[:, ilon, ilat_equator]
    h2o = gcm.molecules[0].dat[:, ilon, ilat_equator]
    ax1.plot(temperature, pressure, label=label, color=c)
    ax2.plot(h2o*100, pressure, color=c)

ax1.set_yscale('log')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Pressure (bar)')
ax1.set_ylim(np.flip(ax1.get_ylim()))
ax1.legend(loc='upper left')

ax2.set_xlabel('H2O (%)')
ax2.set_ylabel('Pressure (bar)')
ax2.set_yscale('log')
_=ax2.set_ylim(np.flip(ax2.get_ylim()))


# %%
# Make the phase curve

psg.docker.set_url_and_run()

model.build_planet()
model.build_spectra()

data = PhaseAnalyzer.from_model(model)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

im = ax.pcolormesh(
    data.wavelength.to_value(u.um),
    data.time.to_value(u.hour),
    (data.thermal/data.total*1e6).T,
    cmap='viridis'
)
ax.set_xlabel('Wavelength ($\\mu m$)')
ax.set_ylabel('Time (hours)')

_=fig.colorbar(im, ax=ax, label='Thermal Emission (ppm)')

# %%
# Look at spectra

i_day = 0
i_night = data.time.shape[0] // 2
i_quad = data.time.shape[0] // 4

fig,ax = plt.subplots(1,1,figsize=(4,3))

ax.plot(data.wavelength.to_value(u.um),data.spectrum('thermal',i_day),label='Day')
ax.plot(data.wavelength.to_value(u.um),data.spectrum('thermal',i_night),label='Night')
ax.plot(data.wavelength.to_value(u.um),data.spectrum('thermal',i_quad),label='Quadrature')
ax.set_xlabel('Wavelength ($\\mu m$)')
ax.set_ylabel('Thermal Emission (W/m$^2$/$\\mu m$)')
_=ax.legend()
