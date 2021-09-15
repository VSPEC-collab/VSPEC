# VSPCA
Variable Stellar Phase-Curve Analysis: Code that links variable stellar model output to GCM output to create a realistic synthetic data of a star/planet system. Returns spectra of the star, planet, and combined star/planet. The data sets will be used to create a method of separating the planet's flux (reflected and thermal) from the combined system flux as observed by a telescope, in order to study the planet's atmosphere based on its spectra.

TUTORIAL
1) Clone the code into a repository

2) The code comes with 2 example config files, each containing the necessary information to run the code

3) Run the main.py program and input the name of the config file you wish to use, then hit enter.
    3a) Example: ProxCenTest.cfg
    
    3b) The program will create all the necessary folders in the locally cloned directory.

4) On the first run, the program will have to create the star map, the 3D hemisphere star models, and bin the NextGen stellar model code. This will take some time
    4a) By default, the star hemisphere map images are low resolution simply because it is faster. Changing them to high res can be done in the config file.
    
    4b) Once the hemisphere images are generated, they do not have to be generated again the the "GenerateHemispheres" boolean value in the cofig file can be set to False.
    
    4c) Re-generating the hemispheres is only necessary of you change the inclination