# VSPCA
Variable Stellar Phase-Curve Analysis: Code that links variable stellar model output to GCM output to create a realistic synthetic data of a star/planet system. Returns spectra of the star, planet, and combined star/planet. The data sets will be used to create a method of separating the planet's flux (reflected and thermal) from the combined system flux as observed by a telescope, in order to study the planet's atmosphere based on its spectra.

TUTORIAL
  I) Clone the code into a repository

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
    
3) Run the main.py program and input the name of the config file you wish to use, then hit enter.
    3a) Example: ProxCenTest.cfg
    
    3b) The program will create all the necessary folders in the locally cloned directory.

    On the first run, the program will have to create the star map, the 3D hemisphere star models, and bin the NextGen stellar model code. This will take some time
    4a) By default, the star hemisphere map images are low resolution simply because it is faster. Changing them to high res can be done in the config file.
    
    4b) Once the hemisphere images are generated, they do not have to be generated again the the "GenerateHemispheres" boolean value in the cofig file can be set to False.
    
    4c) Re-generating the hemispheres is only necessary of you change the inclination