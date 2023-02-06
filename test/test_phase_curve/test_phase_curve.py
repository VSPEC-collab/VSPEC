#
# TEST GEOMETRY
# --------------
# Objective 1
# Create a gif with some random parameters

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy import units as u
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from os import system


from os import chdir, remove


from VSPEC import ObservationModel, PhaseAnalyzer
from VSPEC.helpers import isclose,to_float
from VSPEC import variable_star_model
from pathlib import Path



def lambertian(beta:u.Quantity[u.deg]):
    """
    beta is the angle, centered on the planet, between the star and observer
    """
    beta = to_float(beta,u.rad)
    return (np.sin(beta) + (np.pi-beta)*np.cos(beta))/np.pi

def calc_beta(nu,omega,i):
    """
    nu: true anomaly
    omega: arg of periasteron?
    i: inclination
    """
    x = -np.sin(nu)*np.sin(omega) + np.cos(nu)*np.cos(omega)
    y = np.cos(nu)*np.cos(i)*np.sin(omega) + np.sin(nu)*np.cos(i)*np.cos(omega)
    z = -np.cos(omega)*np.sin(i)*np.sin(nu) - np.cos(nu)*np.sin(i)*np.sin(omega)
    beta = np.arctan(-np.sqrt(x**2 + y**2)/z)
    return beta


if __name__ in '__main__':
    CONFIG_FILENAME = 'test_phase_curve.cfg'
    WORKING_DIRECTORY = Path(__file__).parent
    CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

    chdir(WORKING_DIRECTORY)


    model = ObservationModel(CONFIG_FILENAME)

    # model.bin_spectra()
    fig,ax = plt.subplots(1,1)
    i_list = [0,30,60,90]*u.deg
    colorlist = [f'C{i}' for i in range(len(i_list))]
    for i,c in zip(i_list,colorlist):
        model.params.system_inclination = i
        model.build_planet()
        model.build_spectra()

        data = PhaseAnalyzer(model.dirs['all_model'])

        nu = data.unique_phase - 360*u.deg
        omega = 90*u.deg

        expected = lambertian(calc_beta(nu,omega,i)% (np.pi*u.rad))
        k = max(expected)
        observed = data.lightcurve('reflected',0,normalize='max')*k

        ax.plot(nu,expected,label=f'Expected, i={i}',c=c)
        ax.scatter(nu,observed,marker='s',label=f'VSPEC, i={i}',c=c)
    ax.set_xlabel('phase (deg)')
    ax.set_ylabel('lambertian factor')
    ax.legend()
    wl = data.wavelength[0]
    ax.set_title(f'Reflected light at {wl.value:.2f} {wl.unit}')
    fig.savefig('phase_curve.png',facecolor='w',dpi=120)


   
