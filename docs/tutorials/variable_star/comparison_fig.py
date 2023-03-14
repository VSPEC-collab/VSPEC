# Make a figure comparing the different kinds of variability
import VSPEC
from astropy import units as u
import matplotlib.pyplot as plt
from os import chdir
from pathlib import Path


if __name__ in '__main__':
    WORKING_DIRECTORY = Path(__file__).parent
    chdir(WORKING_DIRECTORY)

    spot_data = VSPEC.PhaseAnalyzer('spots/Data/AllModelSpectraValues')
    fac_data = VSPEC.PhaseAnalyzer('faculae/Data/AllModelSpectraValues')
    flare_data = VSPEC.PhaseAnalyzer('flares/Data/AllModelSpectraValues')

    fig,axes = plt.subplots(3,1,figsize=(4,7),sharex=True)
    for ax,data,scale in zip(axes,[spot_data,fac_data,flare_data],[100,1e6,100]):
        ax.plot((data.time - data.time[0]).to(u.day),(data.lightcurve('star',0,normalize=0)-1)*scale,
        label=f'{data.wavelength[0]:.1f}')
        ax.plot((data.time-data.time[0]).to(u.day),(data.lightcurve('star',-6,normalize=0)-1)*scale,
        label = f'{data.wavelength[-6]:.1f}')
        ax.set_ylabel(f'contrast ({"pct" if scale==100 else "ppm"})',fontsize=12)
        ax.tick_params(direction='in',right=True,top=True)


    axes[2].set_xlabel('Time (days)',fontsize=12)
    axes[0].legend(prop=dict(size=12))
    plt.subplots_adjust(hspace=0,right=0.95,top=0.95,left=0.19)

    fig.savefig('var_comparison.png',facecolor='w',dpi=120)