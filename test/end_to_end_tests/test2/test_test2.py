"""
End-to-end test 2

Transit.
"""
from os import chdir
from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
import pypsg
from pypsg.globes.waccm.waccm import download_test_data

import VSPEC

pypsg.docker.set_url_and_run()
download_test_data(rewrite=False)
chdir(Path(__file__).parent)

CFG_PATH = Path(__file__).parent / 'test2.yaml'
FIG1_PATH = Path(__file__).parent / 'out1.png'
FIG2_PATH = Path(__file__).parent / 'out2.png'
FIG3_PATH = Path(__file__).parent / 'out3.png'
FIG4_PATH = Path(__file__).parent / 'out4.png'
u_flux = u.Unit('W m-2 um-1')
u_time = u.day
u_wl = u.um


def make_fig(data: VSPEC.PhaseAnalyzer):
    """
    Make the figure
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    spec = fig.add_subplot(gs[:, :], projection='3d')
    wl = data.wavelength.to_value(u.um)
    time = data.time.to_value(u.hour)
    for i, w in enumerate(wl):
        lc = (data.lightcurve('total', i, 0, False) - 1)*1e6
        spec.plot(np.ones_like(time)*w, time, lc, lw=0.5, c='k')
    spec.set_xlabel('Wavelength (um)')
    spec.set_ylabel('time (hours)')
    spec.set_zlabel('transit depth (ppm)')
    spec.view_init(30, 120)
    fig.savefig(FIG1_PATH, facecolor='w')

    fig, ax = plt.subplots(1, 1)
    for i in [0, 40, 90, 125, 145]:
        lc = (data.lightcurve('total', i, normalize=0) - 1)*1e6
        ax.scatter(data.time.to(u.minute), lc,
                   label=f'{data.wavelength[i]:.1f}')
    ax.set_ylabel('Flux (ppm)')
    ax.set_xlabel('Time (hour)')
    ax.legend()
    fig.savefig(FIG2_PATH, facecolor='w')

    fig, ax = plt.subplots(1, 1)
    observed = data.total.value
    maxs = np.max(observed, axis=1)
    mins = np.min(observed, axis=1)
    tr_depth = (maxs-mins)/maxs
    rp_rs = np.sqrt(tr_depth)
    ax.plot(data.wavelength, rp_rs*100)
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Rp/Rs (%)')
    fig.savefig(FIG3_PATH, facecolor='w')

    fig, ax = plt.subplots(1, 1)
    for i in [0, 40, 90, 125, 145]:
        lc = data.lightcurve('thermal', i)
        ax.scatter(data.time.to(u.minute), lc,
                   label=f'{data.wavelength[i]:.1f}')
    ax.set_ylabel('Flux (ppm)')
    ax.set_xlabel('Time (hour)')
    ax.legend()
    fig.savefig(FIG4_PATH, facecolor='w')


def test_run():
    """
    Run the model.
    """
    model = VSPEC.ObservationModel.from_yaml(CFG_PATH)
    model.build_planet()
    model.build_spectra()

    data = VSPEC.PhaseAnalyzer(model.directories['all_model'])
    make_fig(data)


if __name__ in '__main__':
    test_run()
