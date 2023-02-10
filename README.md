---
jupyter:
  kernelspec:
    display_name: Python 3.9.12 (\'base\')
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.12
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: a881c44cacd4bbbf839be660c688ced33edf73c9abf125cc2b16ee77cb5e3123
---

::: {.cell .markdown}
# VSPEC: Variable Star PhasE Curve

## A python package to simulate Planetary Infrared Excess (PIE) observations of rocky planets around variable M dwarfs

#### Cameron Kelehan and Ted Johnson

VSPEC uses a dynamic model of stellar spots, faculae, and flares
combined with simultations from the Planetary Spectrum Generator (PSG,
[Villanueva et al.,
2018](https://ui.adsabs.harvard.edu/abs/2018JQSRT.217...86V/abstract))
to simulate phase resolved observations of planetary thermal emission
spectra. This package was designed for the Mid-IR Exoplanet CLimate
Explorer mission concept (MIRECLE, [Mandell et al.,
2022](https://ui.adsabs.harvard.edu/abs/2022AJ....164..176M/abstract)),
but was built to be used more generally.

The primary goal of this software is to simulate combined planet-host
spectra in order to develop techniques to remove the star using the
Planetary Infrared Excess (PIE) technique. For more info on PIE, see
[Stevenson
(2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...898L..35S/abstract)
and [Lustig-Yaeger et al.
(2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...921L...4L/abstract).
:::

::: {.cell .markdown}
### Installation

For now it is best to clone this repository, but we would like to use
pypi in the future.

`git clone https://github.com/tedjohnson12/VSPEC.git`

`cd VSPEC`

`pip install -e .`
:::

::: {.cell .markdown}
### Using VSPEC

#### Quick start guide

The parameters of a VSPEC model are specified in a configuration file.
Before we run the model we will read these parameters into memory. The
fundamental object in VSPEC is `VSPEC.ObservationalModel`.
:::

::: {.cell .code execution_count="1"}
``` python
import VSPEC
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy import units as u
import numpy as np
```
:::

::: {.cell .code execution_count="2"}
``` python
config_path = '../test/default.cfg'
model = VSPEC.ObservationModel(config_path)
```
:::

::: {.cell .markdown}
The GCM we will use is located here:
:::

::: {.cell .code execution_count="3"}
``` python
print(Path(config_path).parent / 'test_gcms/modernearth.gcm')
```

::: {.output .stream .stdout}
    ../test/test_gcms/modernearth.gcm
:::
:::

::: {.cell .markdown}
This created a new directory called `example_star/` that will store all
the data from this model run.

Let\'s look at where our model parameters are stored. For example we can
look at the effective temperature of quiet photosphere.
:::

::: {.cell .code execution_count="4"}
``` python
model.params.star_teff
```

::: {.output .execute_result execution_count="4"}
```{=latex}
$3300 \; \mathrm{K}$
```
:::
:::

::: {.cell .markdown}
Now we need to bin the spectra to the desired resolution. In the
configureation file we specified:
:::

::: {.cell .code execution_count="5"}
``` python
print(f'starting wavelength = {model.params.lambda_min}')
print(f'ending wavelength = {model.params.lambda_max}')
print(f'resolving power = {model.params.resolving_power}')
```

::: {.output .stream .stdout}
    starting wavelength = 1.0 um
    ending wavelength = 18.0 um
    resolving power = 50.0
:::
:::

::: {.cell .markdown}
We can ask the model to bin spectra from a grid of PHOENIX models. The
$T_{eff}$ range is specified in the configuration file.
:::

::: {.cell .code execution_count="6"}
``` python
model.bin_spectra()
```

::: {.output .display_data}
``` json
{"model_id":"d1f2c87531f84035b69165ac5585044c","version_major":2,"version_minor":0}
```
:::
:::

::: {.cell .markdown}
Now we make a series of API calls to PSG to retrive spectra of the
planet model. The configuration file specifies a GCM file that is
uploaded to PSG in `model.params.gcm_file_path`.

I am running PSG locally using Rancher Desktop. To run PSG on your
machine there are detailed instructions in the [PSG
handbook](https://psg.gsfc.nasa.gov/help.php#handbook). It is very easy
and allows you to avoid the need for an API key.
:::

::: {.cell .code execution_count="7"}
``` python
model.build_planet()
```

::: {.output .stream .stdout}
    Starting at phase 180.0 deg, observe for 10.0 d in 20 steps
    Phases = [180.   198.95 217.89 236.84 255.79 274.74 293.68 312.63 331.58 350.53
       9.47  28.42  47.37  66.32  85.26 104.21 123.16 142.11 161.05 180.  ] deg
:::

::: {.output .display_data}
``` json
{"model_id":"29eb787f38914af092a005f1b31513d8","version_major":2,"version_minor":0}
```
:::
:::

::: {.cell .markdown}
Lastly, we need to run our variable star model. PSG uses its own stellar
templates, but we will replace those with our own model. This allows us
to accurately model the affect that variability has on reflected light
as well as noise. We will more finely sample the time over which we
observe because the star can change much faster than the planet.
:::

::: {.cell .code execution_count="8"}
``` python
model.build_spectra()
```

::: {.output .stream .stdout}
    Generated 39 mature spots
:::

::: {.output .display_data}
``` json
{"model_id":"8da5c57821af4e969042ab0e00004d26","version_major":2,"version_minor":0}
```
:::
:::

::: {.cell .markdown}
#### Analysis

All of our data produced by this model run is stored in
`example_star/Data/AllModelSpectraValues`. We can store that data in the
`PhaseAnayzer` object.
:::

::: {.cell .code execution_count="9"}
``` python
data_path = Path('default/Data/AllModelSpectraValues')
sim_data = VSPEC.PhaseAnalyzer(data_path)
```
:::

::: {.cell .code execution_count="10"}
``` python
plt.contourf(sim_data.unique_phase-360*u.deg,sim_data.wavelength,sim_data.thermal,levels=30)
plt.xlabel('phase (deg)')
plt.ylabel('wavelength (um)')
plt.title('Planetary Thermal Emission')
cbar = plt.colorbar()
cbar.set_label('flux (W m-2 um-1)')
```

::: {.output .display_data}
![](readme_files/92832a48796efea0c5fb0e6449436a64c4bf8062.png)
:::
:::

::: {.cell .markdown}
#### NOTE: this code uses the PSG convention for planet phase. Primary transit occurs at phase = 180 deg, and secondary ecllipse occurs at phase = 0 deg (for planets that transit). {#note-this-code-uses-the-psg-convention-for-planet-phase-primary-transit-occurs-at-phase--180-deg-and-secondary-ecllipse-occurs-at-phase--0-deg-for-planets-that-transit}
:::

::: {.cell .markdown}
We can also create lightcurves
:::

::: {.cell .code execution_count="11"}
``` python
pixel = (0,40)
bandpass = sim_data.wavelength[slice(*pixel)]
plt.plot(sim_data.time.to(u.day), sim_data.lightcurve('total',pixel))
plt.xlabel('time (day)')
plt.ylabel('flux (W m-2 um-1)')
plt.title(f'Source total with\nbandpass from {bandpass.min()} to {bandpass.max()}')
```

::: {.output .execute_result execution_count="11"}
    Text(0.5, 1.0, 'Source total with\nbandpass from 1.0 um to 2.164745 um')
:::

::: {.output .display_data}
![](readme_files/8acb6cd8049bf559d09be6410e5ea103b9545d4a.png)
:::
:::

::: {.cell .code execution_count="12"}
``` python
pixel = (120,150)
bandpass = sim_data.wavelength[slice(*pixel)]
plt.plot(sim_data.time.to(u.day),sim_data.lightcurve('thermal',pixel))
plt.xlabel('time (day)')
plt.ylabel('flux (W m-2 um-1)')
plt.title(f'thermal with\nbandpass from {bandpass.min()} to {bandpass.max()}')
```

::: {.output .execute_result execution_count="12"}
    Text(0.5, 1.0, 'thermal with\nbandpass from 10.76516 um to 17.66139 um')
:::

::: {.output .display_data}
![](readme_files/3506cf22080b436662fe572645c111496cd7837c.png)
:::
:::

::: {.cell .code execution_count="14"}
``` python
pixel = (0,40)
bandpass = sim_data.wavelength[slice(*pixel)]
plt.plot(sim_data.time.to(u.day),sim_data.lightcurve('reflected',pixel))
plt.xlabel('time (day)')
plt.ylabel('flux (W m-2 um-1)')
plt.title(f'Reflected with\nbandpass from {bandpass.min()} to {bandpass.max()}')
```

::: {.output .execute_result execution_count="14"}
    Text(0.5, 1.0, 'Reflected with\nbandpass from 1.0 um to 2.164745 um')
:::

::: {.output .display_data}
![](readme_files/666de8db276cb7925b531037237f667903dec256.png)
:::
:::

::: {.cell .markdown}
We can also look at the spectra produced at each phase step and combine
multiple steps to improve SNR. The code below creates a figure that
shows the system geometry, the thermal profile, and the spectrum with
error envelope
:::

::: {.cell .code execution_count="17"}
``` python
i=0
binning = 50
def makefig(i):
    # plt.style.use('dark_background')
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    # color = cm.viridis(np.cos(data.unique_phase[i])*0.5 + 0.5)
    color = 'k'
    sli = (max(0,i-binning),min(sim_data.N_images,i+binning))
    ax[0].plot(sim_data.layers.loc[sim_data.unique_phase[i],:,'Temp[K]'],(sim_data.layers.loc[sim_data.unique_phase[i],:,'Pressure[bar]']),c=color)
    ax[0].set_yscale('log')
    ylim = ax[0].get_ylim()

    ax[0].text(210,5e-5,f'phase={sim_data.unique_phase[i].value - 360:.1f} deg',fontsize=14,fontfamily='serif')

    ax[0].set_ylim(ylim[1],ylim[0])
    ax[0].set_xlim(150,292)
    ax[0].set_ylabel('Pressure (bar)',fontsize=14,fontfamily='serif')
    ax[0].set_xlabel('Temperature (K)',fontsize=14,fontfamily='serif')

    inax = ax[0].inset_axes([0.5,0.3,0.4,0.4])
    inax.set_aspect(1)
    inax.scatter(0,0,c='xkcd:tangerine',s=150)
    
    theta = np.linspace(0,360,180,endpoint=False)*u.deg
    r_dist = (1-model.params.planet_eccentricity**2)/(1+model.params.planet_eccentricity*np.cos(theta- model.params.system_phase_of_periasteron - 90*u.deg))
    curr_theta = sim_data.phase[i] + 90*u.deg
    x_dist = model.params.planet_semimajor_axis * np.cos(theta)*r_dist
    y_dist = model.params.planet_semimajor_axis * np.sin(theta)*r_dist*np.cos(model.params.system_inclination)
    current_r = (1-model.params.planet_eccentricity**2)/(1+model.params.planet_eccentricity*np.cos(curr_theta- model.params.system_phase_of_periasteron - 90*u.deg))
    current_x_dist = model.params.planet_semimajor_axis * np.cos(curr_theta)*current_r
    current_y_dist = model.params.planet_semimajor_axis * np.sin(curr_theta)*current_r*np.cos(model.params.system_inclination)
    behind = np.sin(theta) >= 0
    x_angle = np.arctan(x_dist/model.params.system_distance).to(u.mas)
    y_angle = np.arctan(y_dist/model.params.system_distance).to(u.mas)
    plotlim = np.arctan(model.params.planet_semimajor_axis/model.params.system_distance).to(u.mas).value * (1+model.params.planet_eccentricity)*1.05
    current_x_angle = np.arctan(current_x_dist/model.params.system_distance).to(u.mas)
    current_y_angle = np.arctan(current_y_dist/model.params.system_distance).to(u.mas)
    z_order_mapper = {True:-99,False:100}
    inax.plot(x_angle[behind],y_angle[behind],zorder=-100,c='C0',alpha=1,ls=(0,(2,2)))
    inax.plot(x_angle[~behind],y_angle[~behind],zorder=99,c='C0')
    inax.scatter(current_x_angle,current_y_angle,zorder = z_order_mapper[np.sin(curr_theta) >= 0],c='k')
    inax.set_xlim(-plotlim,plotlim)
    inax.set_ylim(-plotlim,plotlim)
    inax.set_xlabel('sep (mas)')
    inax.set_ylabel('sep (mas)')


    contrast = sim_data.combine('thermal',sli)/sim_data.combine('total',sli)
    noise = sim_data.combine('noise',sli)/sim_data.combine('total',sli)
    ax[1].plot(sim_data.wavelength,contrast*1e6,c=color)
    ax[1].fill_between(sim_data.wavelength.value,(contrast + noise)*1e6,(contrast-noise)*1e6,color='k',alpha=0.3)
    ax[1].set_ylim(-1,100)
    ax[1].set_ylabel('Contrast (ppm)',fontsize=14,fontfamily='serif')
    ax[1].set_xlabel('Wavelength ($\mu$m)',fontsize=14,fontfamily='serif')
    plt.style.use('default')
    return fig
```
:::

::: {.cell .code execution_count="18"}
``` python
makefig(300).show()
```

::: {.output .stream .stderr}
    /var/folders/b4/9tq5p8g95dl0144rmgktxrvh0000gp/T/ipykernel_6698/1612521938.py:1: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      makefig(300).show()
:::

::: {.output .display_data}
![](readme_files/776dcab8b7322ece31a20e69136185c027d71829.png)
:::
:::

::: {.cell .code execution_count="19"}
``` python
makefig(60).show()
```

::: {.output .stream .stderr}
    /var/folders/b4/9tq5p8g95dl0144rmgktxrvh0000gp/T/ipykernel_6698/1915159874.py:1: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      makefig(60).show()
:::

::: {.output .display_data}
![](readme_files/0ad88f64d4c8dbff8ef83c03c443a2ed399e8642.png)
:::
:::

::: {.cell .code execution_count="20"}
``` python
plt.contourf(sim_data.time.to(u.day),sim_data.wavelength,sim_data.total,levels=30)
plt.colorbar(label='flux (W m-2 um-1)')
plt.yscale('log')
plt.xlabel('time (day)')
plt.ylabel('wavelength (um)')
```

::: {.output .execute_result execution_count="20"}
    Text(0, 0.5, 'wavelength (um)')
:::

::: {.output .display_data}
![](readme_files/278ec7bbc6d15ea90a4d2fc8516309c7a5fe9680.png)
:::
:::

::: {.cell .code}
``` python
```
:::
