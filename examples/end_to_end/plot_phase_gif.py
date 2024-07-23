"""
Make a phase curve GIF
======================

This example turns a phase curve into a gif.
"""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from imageio.v2 import imread, mimsave
from cartopy import crs as ccrs
import pypsg
from pypsg.globes import PyGCM

from VSPEC import ObservationModel,PhaseAnalyzer
from VSPEC.geometry import SystemGeometry

try:
    CONFIG_PATH = Path(__file__).parent / 'phase_gif.yaml'
except NameError:
    CONFIG_PATH = Path('phase_gif.yaml')

pypsg.docker.set_url_and_run()


#%%
# Load in the configuration
# -------------------------
#
# It is stored in a YAML file.

model = ObservationModel.from_yaml(CONFIG_PATH)
model.build_planet()
model.build_spectra()

#%%
# Write a figure making function
# ------------------------------
#
# So we can make a GIF later.

def make_fig(data:PhaseAnalyzer,geo:SystemGeometry,gcm:PyGCM,s:tuple):
    """
    data is the simulation data
    s is the phase index (start, stop)
    """
    i = int(np.mean(s)) # int representation of s
    p,q = s
    
    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(1,2)
    prof_ax = fig.add_subplot(gs[0,0])

    pressure = np.mean(data.get_layer('Pressure')[p:q,:],axis=0)
    temp = np.mean(data.get_layer('Temp')[p:q,:],axis=0)

    prof_ax.plot(temp,pressure)
    prof_ax.set_yscale('log')
    prof_ax.set_xlabel('Temperature (K)')
    prof_ax.set_ylabel('Pressure (bar)')
    prof_ax.set_ylim(np.flip(prof_ax.get_ylim()))
    prof_ax.set_xlim(-5,290)

    phase = data.phase[i]
    inax = prof_ax.inset_axes([0.5,0.5,0.4,0.4])
    inax.set_aspect(1)
    geo.get_system_visual(phase,ax=inax)

    pl_spec = data.spectrum('thermal',s,noise=False)
    star_spec = data.spectrum('star',s,noise=False)
    noi_spec = data.spectrum('noise',s,noise=False)
    wl = data.wavelength.to_value(u.um)

    cont = ((pl_spec)/star_spec).to_value(u.dimensionless_unscaled)
    contp = ((pl_spec+noi_spec)/star_spec).to_value(u.dimensionless_unscaled)
    contm = ((pl_spec-noi_spec)/star_spec).to_value(u.dimensionless_unscaled)

    spec_ax = fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())

    spec_ax.plot(wl,cont*1e6,c='k')
    spec_ax.fill_between(wl,contp*1e6,contm*1e6,color='k',alpha=0.2)

    spec_ax.set_xlabel('Wavelength (um)')
    spec_ax.set_ylabel('Thermal emission (ppm)')
    spec_ax.set_aspect('auto')
    spec_ax.tick_params(axis='both',which='major',direction='out')
    spec_ax.set_xticks(np.arange(1,19,2))
    spec_ax.set_yticks(np.arange(-1,10,2)*10)

    lat = geo.get_pl_sub_obs_lat(phase)
    time = data.time[i]
    lon = geo.get_pl_sub_obs_lon(time,phase)
    proj = ccrs.Orthographic(
        central_latitude=lat.to_value(u.deg),
        central_longitude=lon.to_value(u.deg)
    )
    mapax = spec_ax.inset_axes([0.05,0.5,0.4,0.4],projection=proj)
    cbarax = spec_ax.inset_axes([0.5,0.5,0.1,0.4],projection=ccrs.PlateCarree())
    cbarax.set_axis_off()

    tsurf = gcm.tsurf.dat.to_value(u.K)
    lats = gcm.lats
    lons = gcm.lons
    im = mapax.pcolormesh(lons,lats,tsurf.T,cmap='gist_heat',transform=ccrs.PlateCarree())
    fig.colorbar(im,ax=cbarax,label='$T_{\\rm surf}$ (K)')
    return fig

#%%
# Get the data
# ------------

data = PhaseAnalyzer(model.directories['all_model'])
geometry = model.get_observation_parameters()
gcm = model.params.gcm.get_gcm()
# gcm = GCMdecoder.from_psg(model.params.gcm.content())

#%%
# Display one frame
# -----------------

fig = make_fig(data,geometry,gcm,(0,20))
fig.show()

#%%
# Make a GIF
# ----------


def gif_image(i):
    s = (max(0,i-10),min(data.n_images-1,i+10))
    return make_fig(data,geometry,gcm,s)

images = []
fname='temp.png'
for i in range(data.n_images):
    fig = gif_image(i)
    fig.savefig(fname)
    plt.close(fig)
    images.append(imread(fname))
    os.remove(fname)

filename = 'phase_curve.gif'
_=mimsave(filename, images,fps=20)
