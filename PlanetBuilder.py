##### GERONIMO'S CODE #####
# ---------------------------------------------------------------
# Script to compute phase curves with PSG/GlobES
# Villanueva, Suissa - NASA Goddard Space Flight Center
# February 2021
# ---------------------------------------------------------------

from sys import api_version
import numpy as np
import os
from pathlib import Path
import read_info

api_key = '5c48d8163cc183e79ac3'

# 2nd file to run.

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
    # gcm = 'modernearth.gcm'
    gcm = 'ProxCenb_PSG.dat'
    print("\nUploading GCM to PSG...")
    file_path = Path('.') / 'Configs' / 'GCMs' / f'{gcm}'
    # os.system(f'curl -s -d key=3c8f608c3c5059f79a59 -d app=globes -d type=set --data-urlencode file@{file_path} {Params.psgurl}/api.php')
    os.system(f'curl -s -d key={api_key} -d app=globes -d type=set --data-urlencode file@{file_path} {Params.psgurl}/api.php')

    # Define parameters of this run
    with open(Path('.') / 'Configs' / 'GCMs' / 'config.txt', "w") as fr:
        fr.write('<OBJECT-DIAMETER>%f\n' % Params.objDiam)
        fr.write('<OBJECT-GRAVITY>%f\n' % Params.objGrav)
        fr.write('<OBJECT-GRAVITY-UNIT>g\n')
        fr.write('<OBJECT-STAR-TYPE>%s\n' % Params.starType)
        fr.write('<OBJECT-STAR-DISTANCE>%f\n' % Params.semMajAx)
        fr.write('<OBJECT-PERIOD>%f\n' % Params.objPer)
        fr.write('<OBJECT-ECCENTRICITY>%f\n' % Params.objEcc)
        fr.write('<OBJECT-STAR-TEMPERATURE>%f\n' % Params.starTemp)
        fr.write('<OBJECT-STAR-RADIUS>%f\n' % Params.starRad)
        fr.write('<GEOMETRY>Observatory\n')
        # fr.write('<GEOMETRY-OFFSET-NS>0.0\n')
        # fr.write('<GEOMETRY-OFFSET-EW>0.0\n')
        # fr.write('<GEOMETRY-OFFSET-UNIT>arcsec\n')
        fr.write('<GEOMETRY-OBS-ALTITUDE>%f\n' % Params.objDis)
        fr.write('<GEOMETRY-ALTITUDE-UNIT>pc\n')
        fr.write('<GENERATOR-RANGE1>%f\n' % Params.lam1)
        fr.write('<GENERATOR-RANGE2>%f\n' % Params.lam2)
        fr.write('<GENERATOR-RANGEUNIT>um\n')
        fr.write('<GENERATOR-RESOLUTION>%f\n' % Params.lamRP)
        fr.write('<GENERATOR-RESOLUTIONUNIT>RP\n')
        fr.write('<GENERATOR-BEAM>%d\n' % Params.beamValue)
        fr.write('<GENERATOR-BEAM-UNIT>%s\n'% Params.beamUnit)
        fr.write('<GENERATOR-CONT-STELLAR>Y\n')
        fr.write('<OBJECT-INCLINATION>%s\n' % Params.inclinationPSG)
        fr.write('<OBJECT-SOLAR-LATITUDE>0.0\n')
        fr.write('<OBJECT-OBS-LATITUDE>0.0\n')
        fr.write('<GENERATOR-RADUNITS>%s\n' % Params.radunit)
        fr.write('<GENERATOR-GCM-BINNING>%d\n' % Params.binning)
        # fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00\n')
        # REMOVE/EDIT LATER: Temporary fix to avoid dramatic C02 emissions at 4.5 and 12 um, due to Rayleigh/PSGDORT/nmax stream pairs
        # fr.write('<ATMOSPHERE-NMAX>0')
        fr.close()
        
    file_path = Path('.') / 'Configs' / 'GCMs' / 'config.txt'
    # file_path = Path('.') / 'Configs' / 'GCMs' / 'test.txt'
    print("\n\n\n\n\n\n\n\n")
    print("\nUploading specified planet and star data...")
    # os.system(f'curl -s -d key=3c8f608c3c5059f79a59 -d app=globes -d type=upd --data-urlencode file@{file_path} {Params.psgurl}/api.php')
    os.system(f'curl -s -d key={api_key} -d app=globes -d type=upd --data-urlencode file@{file_path} {Params.psgurl}/api.php')
    # os.system(f'curl -v -d type=cfg -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {file_path}')

    print("\nPERCENT DONE")
    print("=================")
    # Calculate the total number of planet rotations given the observing time frame
    final_phase = Params.total_images * Params.delta_phase_planet
    # print(np.arange(0,final_phase,Params.delta_phase_planet))
    print(np.linspace(Params.phase1,Params.phase1+final_phase,Params.delta_phase_planet) % 360)
    count = 0
    for phase in (np.linspace(Params.phase1,Params.phase1+final_phase,Params.delta_phase_planet) % 360):
        if phase>178 and phase<182:
            phase=182.0 # Add transit phase;
        if phase == 185:
            phase = 186.0
        
        # Write updates to the config to change the phase value and ensure the star is of type 'StarType'
        with open(Path('.') / 'Configs' / 'Stellar' / 'config.txt', 'w') as fr:
            fr.write('<OBJECT-STAR-TYPE>%s\n' % Params.starType)
            fr.write('<OBJECT-SEASON>%f\n' % phase)
            fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % phase)
            fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00')
            fr.close()
        
        # First call PSG to retrieve the stellar and planet reflection values; saves the output
        file_path = Path('.') / 'Configs' / 'Stellar' / 'config.txt'
        # -d wgeo=y (After type=cfg)
        # os.system(f'curl -s -d key=3c8f608c3c5059f79a59 -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGcombinedSpectraFolder}/phase{phase:.3f}.txt')
        if ((count/Params.total_images) * 100) >= 200: #changed from 71
            os.system(f'curl -s -d key={api_key} -d type=cfg -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGcombinedSpectraFolder}/phase{phase:.3f}.txt')
        else:
            os.system(f'curl -s -d key={api_key} -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGcombinedSpectraFolder}/phase{phase:.3f}.txt')
        # os.system(f'curl -v -d type=all -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGcombinedSpectraFolder}/phase{phase:.3f}.txt')
        
        # with open(f'{file_path}', 'r') as fr:
        #     lines =fr.readlines()
        
        # Second call to PSG to retrieve strictly the planet's thermal values, by setting the star to Null (-)
        with open(f'{file_path}', 'w') as fr:
            phase *= -1
            fr.write('<OBJECT-STAR-TYPE>-\n')
            fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % phase)
            phase *= -1
            fr.close()
        
        if ((count/Params.total_images) * 100) >= 71:
            os.system(f'curl -s -d key={api_key} -d type=all -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGthermalSpectraFolder}/phase{phase:.3f}.txt')
        # os.system(f'curl -s -d key=3c8f608c3c5059f79a59 -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGthermalSpectraFolder}/phase{phase:.3f}.txt')
        os.system(f'curl -s -d key={api_key} -d app=globes --data-urlencode file@{file_path} {Params.psgurl}/api.php > {Params.PSGthermalSpectraFolder}/phase{phase:.3f}.txt')

        print(round(((count/Params.total_images) * 100), 2), "%")
        print(count)
        if ((count/Params.total_images) * 100) >= 71:
            print("Pause")
        count += 1



    print('done')