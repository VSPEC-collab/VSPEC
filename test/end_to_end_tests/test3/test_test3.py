"""
End-to-end test 3

Eclipse
"""

from pathlib import Path
import matplotlib.pyplot as plt
from os import chdir
from astropy import units as u
import numpy as np

import pypsg

import VSPEC

chdir(Path(__file__).parent)

pypsg.docker.set_url_and_run()

CFG_PATH = Path(__file__).parent / 'test3.yaml'
FIG1_PATH = Path(__file__).parent / 'out1.png'
FIG2_PATH = Path(__file__).parent / 'out2.png'
FIG3_PATH = Path(__file__).parent / 'out3.png'
u_flux = u.Unit('W m-2 um-1')
u_time = u.day
u_wl = u.um

def make_fig(data:VSPEC.PhaseAnalyzer):
    fig = plt.figure()
    gs = fig.add_gridspec(2,2)


    spec = fig.add_subplot(gs[:,:],projection='3d')
    wl = data.wavelength.to_value(u.um)
    time = data.time.to_value(u.hour)
    for i,w in enumerate(wl):
        lc = (data.lightcurve('total',i,0,False) - 1)*1e6
        spec.plot(np.ones_like(time)*w,time,lc,lw=0.5,c='k')
    spec.set_xlabel('Wavelength (um)')
    spec.set_ylabel('time (hours)')
    spec.set_zlabel('Eclipse depth (ppm)')
    spec.view_init(30, 100)
    fig.savefig(FIG1_PATH,facecolor='w')

    fig, ax  = plt.subplots(1,1)
    for i in [0,40,90,120,145]:
        lc = (data.lightcurve('total',i,normalize=0) - 1)*1e6
        ax.plot(data.time.to(u.hour),lc,label=f'{data.wavelength[i]:.1f}')
    ax.set_ylabel('Flux (ppm)')
    ax.set_xlabel('Time (hour)')
    ax.legend()
    fig.savefig(FIG2_PATH,facecolor='w')

    fig,ax = plt.subplots(1,1)
    observed = data.total.value
    maxs = np.max(observed,axis=1)
    mins = np.min(observed,axis=1)
    excess = (maxs-mins)/maxs
    
    ax.plot(data.wavelength,excess*1e6)
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('excess (ppm)')
    fig.savefig(FIG3_PATH,facecolor='w')

def test_run():
    model = VSPEC.ObservationModel.from_yaml(CFG_PATH)
    model.build_planet()
    model.build_spectra()

    data = VSPEC.PhaseAnalyzer(model.directories['all_model'])
    make_fig(data)

if __name__ in '__main__':
    test_run()