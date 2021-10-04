# VSPCA
Variable Stellar Phase-Curve Analysis: Code that links variable stellar model output to GCM output to create a realistic synthetic data of a star/planet system. Returns spectra of the star, planet, and combined star/planet. The data sets will be used to create a method of separating the planet's flux (reflected and thermal) from the combined system flux as observed by a telescope, in order to study the planet's atmosphere based on its spectra.

TUTORIAL
  I) Clone the code into a repository.

 II) The code comes with 2 example config files, each containing the necessary information to run the code
    A) Read through one of the examples to understand the parameters used throughout the whole code.

III) Running the code
    A) The Code is split into 4 main executable programs, all named ____Builder.py, that you run individually and, to start, in 
       this order.
        1) StarBuilder.py
            i) creates the variable star model with spots and faculae, must be run first as the program needs a star.
           ii) it also bins the supplied stellar flux models (NextGen stellar dataset by default) to a resolving power specified 
               in the config.
        2) PlanetBuilder.py
            i) calls the Globes application of the Planetary Spectrum Generator web-tool.
           ii) Run second, after building th star; the program needs a planet.
          iii) the program sends information based on the user-defined config file including stellar and planetray parameters to 
               PSG, starting with a General Circulation Model, and returns a theoretical flux spectrum of the planet's reflected and thermal flux.
        3) SpectraBuilder.py
            i) run third; needs a saved planetary spectrum and stellar spectrum.
           ii) uses the outputs of the first two programs to create a synthetic timeseries of flux data that applies the planet 
               flux from PSG to the star model created in StarBuilder.py, so the data we see is of the planet's flux as if it were revolving around the newly created, variable star.
        4) GraphBuilder.py
            i) Creates many graphs showing lightcurves, 3D stellar model rotation with spots/faculae, stellar flux output, planet 
               flux output, planet thermal flux output, total system output, etc. across a timeseries.
    B) Once you have built and saved all of the models, re-running any part of the code can be done individually.
        1) Modularization of this code makes it easy to quickly correct graphs without re-running the whole process, or 
           re-creating the variable 3D stellar model with a different inclination.
        2) Any time the StarBuilder or PlanetBuilder are changed, the SpectraBuilder must also be re-run, since it relies on the 
           output of those two programs.

# VSPCA

Variable Stellar Phase-Curve Analysis: Code that links variable stellar model output to GCM output to create a realistic synthetic data of a star/planet system. Returns spectra of the star, planet, and combined star/planet. The data sets will be used to create a method of separating the planet's flux (reflected and thermal) from the combined system flux as observed by a telescope, in order to study the planet's atmosphere based on its spectra.


## Requirements

This code was designed using Python 3.8

## Installation

The code doesn't require any special installation, and can be directly downloaded from the Github page or, in alternative, the repository can be cloned via

    git clone https://github.com/AndreaCaldiroli/ATES-Code

    
## Directories and files

The code is made up of 4 main programs:
* StarBuilder.py
* PlanetBuilder.py
* SpectraBuilder.py
* GraphBuilder.py

The code is executed based on a user-specified/curated config file:
* Config files are stored in `VSPCA/Configs`
* Users can create their own configs based on the template provided in the repository (ProxCenTemplate)
* The code also relies on a GCM config file which can be found in `VSPCA/Configs/GCMs/`
* Users are free to create new configs for the star based on the template or introduce new GCM configs

All files are created when the code is run, based on the chosen stellar config file:
* A folder with the user-defined name of the star (based on config) is created.
* This folder contains two major sub-directories: Data and Figures


## Using the code

The Code is split into 4 main executable programs, all named ____Builder.py, that you run individually and, to start, in this order.

1. StarBuilder.py
   - Creates the variable star model with spots and faculae, must be run first as the program needs a star.
   - Calculates the coverage fractions of the photosphere, spots, and faculae for each phase of the star.
   - It also bins the supplied stellar flux models (NextGen stellar dataset by default) to a resolving power specified in the config.
2. PlanetBuilder.py
   - Calls the Globes application of the Planetary Spectrum Generator web-tool.
   - Run second, after building th star; the program needs a planet.
   - The program sends information based on the user-defined config file including stellar and planetray parameters to PSG, starting with a General Circulation Model, and returns a theoretical flux spectrum of the planet's reflected and thermal flux.
3. SpectraBuilder.py
   - Run third; needs a saved planetary spectrum and stellar spectrum.
   - Uses the outputs of the first two programs to create a synthetic timeseries of flux data that applies the planet flux from PSG to the star model created in StarBuilder.py, so the data we see is of the planet's flux as if it were revolving around the newly created, variable star.
4. GraphBuilder.py
   - Creates many graphs showing lightcurves, 3D stellar model rotation with spots/faculae, stellar flux output, planet flux output, planet thermal flux output, total system output, etc. across a timeseries.

Once you have built and saved all of the models, re-running any part of the code can be done individually.
* Modularization of this code makes it easy to quickly correct graphs without re-running the whole process, or re-creating the variable 3D stellar model with a different inclination.
* Any time the StarBuilder or PlanetBuilder are changed, the SpectraBuilder must also be re-run, since it relies on the output of those two programs.



<!-- ## Output files

The code writes the current output of the simulations on two file saved in the `$MAIN/output` directory. The `$MAIN/output/Hydro_ioniz.txt` file stores the hydrodynamical variables, which are saved in column vectors in the following order:
1. radial distance (in unit of the planetary radius)
2. mass density (in unit of the proton mass)
3. velocity (in unit of the scale velocity - see [[1]](#1))
4. pressure (in CGS units)
5. Temperature (in Kelvin)
6. Radiative heating rate (in CGS units)
7. Radiative cooling rate (in CGS units)
8. Heating efficiency (adimensional)


The ionization profiles are saved in the `$MAIN/output/Ion_species.txt` file. The columns of the file correspond to the number densities of HI, HII, HeI, HeII, HeIII in <img src="https://render.githubusercontent.com/render/math?math=\text{cm}^{-3}">.

The post-processed profile are written on the `$MAIN/output/Hydro_ioniz_adv.txt` and `$MAIN/output/Ion_species_adv.txt` files. The data are formatted as the `$MAIN/output/Hydro_ioniz.txt` and `$MAIN/output/Ion_species.txt` files.

If the `Load IC` flag is active in the input window, the code automatically chooses the last saved `$MAIN/output/Hydro_ioniz.txt` and `$MAIN/output/Ion_species.txt`files in the `$MAIN/output` directory and copies them onto two new files named, by default,`$MAIN/output/Hydro_ioniz_IC.txt` and `$MAIN/output/Ion_species_IC.txt`, which are loaded by the code. For the writing/reading formats consult the `$MAIN/src/modules/file_IO/load_IC.f90` and `$MAIN/src/modules/file_IO/write_output.f90` files.

## Plotting results

The `$MAIN/ATES_plots.py` file can be used to plot the current status of the simulation or to follow the evolution of the profiles with a live animation. The script can be executed with the following syntax:

    python3 $MAIN/ATES_plots.py --live n
    
The `--live n` arguments are optional, and can therefore be omitted. If so, the content of the current `$MAIN/output/Hydro_ioniz.txt` and `$MAIN/output/Ion_species.txt` is plotted. If only the `--live` flag is used, the figure is updated by default every 4 seconds with the content of the current output files (which ATES, by defaults, overwrites every 1000th temporal iteration). To set the time update interval, specify the `n` argument with the desired number of seconds between the updates. Finally, a second figure with the post-processed profiles is created if the corresponding files (`$MAIN/output/Hydro_ioniz_adv.txt`and `$MAIN/output/Ion_species_adv.txt`) are found in the `$MAIN/output` directory.


## References
<a id="1">[1]</a> 
Caldiroli, A., Haardt, F., Gallo, E., Spinelli, R., Malsky, I., Rauscher, E., 2021, "Irradiation-driven escape of primordial planetary atmospheres I. The ATES photoionization hydrodynamics code", arXiv:2106.10294
 -->
