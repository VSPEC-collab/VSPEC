##### GERONIMO'S CODE #####
# ---------------------------------------------------------------
# Script to compute phase curves with PSG/GlobES
# Villanueva, Suissa - NASA Goddard Space Flight Center
# February 2021
# ---------------------------------------------------------------

import numpy as np
import os
import read_info

if __name__ == "__main__":
    
    # 1) Read in all of the user-defined config parameters into a class, called Params.
    Params = read_info.ParamModel()

    # KEEP FOR NOW MAYBE EDIT; Geronimo said most common gcf file is netCDF. Will need to be able to convet this
    # # Convert netCDF file to PSG/GCM format
    # from gcm_exocam import convertgcm
    # convertgcm(ncfile, 'gcm_psg.dat')

    # EDIT: this key belongs to me, need to remove it after testing. Then test again to ensure one full run can work
    # GlobES/API calls can be sequentially, and PSG will remember the previous values
    # This means that we can upload parameters step-by-step. To reset your config for GlobES (use type=set), and to simply update (use type=upd)
    print("\nUploading GCM to PSG...")
    os.system('curl -s -d key=3c8f608c3c5059f79a59 -d app=globes -d type=set --data-urlencode file@./Configs/GCMs/modernearth.gcm %s/api.php' % Params.psgurl)

    # Define parameters of this run
    with open("./Configs/GCMs/config.txt", "w") as fr:
        fr.write('<OBJECT-DIAMETER>%f\n' % Params.objDiam)
        fr.write('<OBJECT-GRAVITY>%f\n' % Params.objGrav)
        fr.write('<OBJECT-STAR-TYPE>%s\n' % Params.starType)
        fr.write('<OBJECT-STAR-DISTANCE>%f\n' % Params.semMajAx)
        fr.write('<OBJECT-PERIOD>%f\n' % Params.objPer)
        fr.write('<OBJECT-ECCENTRICITY>%f\n' % Params.objEcc)
        fr.write('<OBJECT-STAR-TEMPERATURE>%f\n' % Params.starTemp)
        fr.write('<OBJECT-STAR-RADIUS>%f\n' % Params.starRad)
        fr.write('<GEOMETRY-OBS-ALTITUDE>%f\n' % Params.objDis)
        fr.write('<GENERATOR-RANGE1>%f\n' % Params.lam1)
        fr.write('<GENERATOR-RANGE2>%f\n' % Params.lam2)
        fr.write('<GENERATOR-RANGEUNIT>um\n')
        fr.write('<GENERATOR-RESOLUTION>%f\n' % Params.lamRP)
        fr.write('<GENERATOR-RESOLUTIONUNIT>RP\n')
        fr.write('<GENERATOR-BEAM>%d\n' % Params.beamValue)
        fr.write('<GENERATOR-BEAM-UNIT>%s\n'% Params.beamUnit)
        fr.write('<OBJECT-INCLINATION>90\n')
        fr.write('<OBJECT-SOLAR-LATITUDE>0.0\n')
        fr.write('<OBJECT-OBS-LATITUDE>0.0\n')
        fr.write('<GENERATOR-RADUNITS>%s\n' % Params.radunit)
        fr.write('<GENERATOR-GCM-BINNING>%d\n' % Params.binning)
        fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00')
        fr.close()
    print("\nUploading specified planet and star data...")
    os.system('curl -s -d key=3c8f608c3c5059f79a59 -d app=globes -d type=upd --data-urlencode file@./Configs/GCMs/config.txt %s/api.php' % Params.psgurl)

    print("\nPERCENT DONE")
    print("=================")
    for phase in np.arange(Params.phase1,Params.phase2+Params.dphase,Params.dphase):
        if phase>178 and phase<182: phase=182 # Add transit phase;
        if phase == 185:
            phase = 186
        
        # Write updates to the config to change the phase value and ensure the star is of type 'StarType'
        with open("./Configs/Stellar/config.txt", "w") as fr:
            fr.write('<OBJECT-STAR-TYPE>%s\n' % Params.starType)
            fr.write('<OBJECT-SEASON>%f\n' % phase)
            fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % phase)
            fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00')
            fr.close()
        
        # First call PSG to retrieve the stellar and planet reflection values; saves the output
        os.system('curl -s -d key=3c8f608c3c5059f79a59 -d app=globes --data-urlencode file@./Configs/Stellar/config.txt %s/api.php > %s/phase%d.txt'
                  % (Params.psgurl, Params.PSGcombinedSpectraFolder, phase))
        
        # Second call to PSG to retrieve strictly the planet's thermal values, by setting the star to Null (-)
        with open("./Configs/Stellar/config.txt", "w") as fr:
            phase *= -1
            fr.write('<OBJECT-STAR-TYPE>-\n')
            fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % phase)
            phase *= -1
            fr.close()
        
        os.system('curl -s -d key=3c8f608c3c5059f79a59 -d app=globes --data-urlencode file@./Configs/Stellar/config.txt %s/api.php > %s/phase%d.txt'
            % (Params.psgurl, Params.PSGthermalSpectraFolder, phase))

        print(round(((phase/360) * 100), 2), "%")



    print('done')