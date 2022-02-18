# Exoplanet Phase Curve Variability Simulator

This code links variable stellar model output to GCM output and creates realistic, synthetic data of a star/planet system. It returns spectra of the star, planet, and combined star/planet. The data sets will be used to create a method of separating the planet's flux (reflected and thermal) from the combined system flux as observed by a telescope, in order to study the planet's atmosphere based on its spectra.


## Requirements

This code was designed using Python 3.8

## Installation

The code doesn't require any special installation, and can be directly downloaded from the Github page or, in alternative, the repository can be cloned via

    https://github.com/cameronkelahan/Exoplanet-Phase-Curve-Variability-Simulator

    
## Directories and files

The code is made up of 4 main programs:
* StarBuilder.py
* PlanetBuilder.py
* SpectraBuilder.py
* PlotBuilder.py

The code is executed based on a user-specified/curated config file:
* Config files are stored in `/Configs`
* Users can create their own configs based on the template provided in the repository (ProxCenTemplate)
* The code also relies on a GCM config file which can be found in `/Configs/GCMs/`
* Users are free to create new configs for the star based on the template or introduce new GCM configs

All files are created when the code is run, based on the chosen stellar config file:
* A folder with the user-defined name of the star (based on config) is created.
* This folder contains two major sub-directories: Data saves all necessary data arrays and Figures stores all produced images/plots.


## Using the code

The Code is split into 4 main executable programs, all named ____Builder.py, that you run individually and, to start, in this order.

1. StarBuilder.py
   - Creates the variable star model with spots and faculae, must be run first as the program needs a star.
   - Calculates the coverage fractions of the photosphere, spots, and faculae for each phase of the star.
   - It also bins the supplied stellar flux models (NextGen stellar dataset by default) to a resolving power specified in the config.
2. PlanetBuilder.py
   - Calls the Globes application of the Planetary Spectrum Generator web-tool.
   - Run second, after building th star; this produces a planet for the program.
   - The program sends information based on the user-defined config file including stellar and planetray parameters to PSG, starting with a General Circulation Model, and returns theoretical flux spectra of the planet's reflected and thermal flux.
3. SpectraBuilder.py
   - Run third; it needs saved planetary spectra and stellar spectra.
   - Uses the output of the StarBuilder.py and PlanetBuilder.py to create a synthetic timeseries of flux data that applies the planet flux from PSG to the star model created in StarBuilder.py, so the data we see is of the planet's flux as if it were revolving around the newly created, variable star.
4. GraphBuilder.py
   - Creates many graphs showing lightcurves, stellar flux output, planet flux output, planet thermal flux output, total system output, etc. across a timeseries.

Once you have built and saved all of the models, re-running any part of the code can be done individually.
* Modularization of this code makes it easy to quickly correct graphs without re-running the whole process, or re-creating the variable 3D stellar model with a different inclination.
* Any time the StarBuilder or PlanetBuilder are changed, the SpectraBuilder must also be re-run, since it relies on the output of those two programs.



<!-- ## Output files

## References
<a id="1">[1]</a> 
Caldiroli, A., Haardt, F., Gallo, E., Spinelli, R., Malsky, I., Rauscher, E., 2021, "Irradiation-driven escape of primordial planetary atmospheres I. The ATES photoionization hydrodynamics code", arXiv:2106.10294
 -->
