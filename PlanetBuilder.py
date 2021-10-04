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
    os.system('curl -v -d key=3c8f608c3c5059f79a59 -d app=globes -d type=set --data-urlencode file@./GCMs/modernearth.gcm %s/api.php' % Params.psgurl)
    # print('curl -v -d app=globes -d type=set --data-urlencode file@./ProxCen+TOI700d/modernearth.gcm %s/api.php' % psgurl)
    # exit()

    # # Calls just regular old PSG, not globes
    # if update: os.system(f'curl -v -d -m 45 type=set --data-urlencode file@./ProxCen+TOI700d/baseConfigStellarTemplates.txt %s/api.php' % psgurl)

    print("wait")
    # Define parameters of this run
    with open("./GCMs/config.txt", "w") as fr:
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
    os.system('curl -v -d key=3c8f608c3c5059f79a59 -d app=globes -d type=upd --data-urlencode file@./GCMs/config.txt %s/api.php' % Params.psgurl)
    # if update: os.system('curl -v -d key=3c8f608c3c5059f79a59 --data-urlencode file@config.txt %s/api.php > spectra/halfwayPointHandUpdate.txt' % psgurl)
    # if update: os.system('curl -v -d key=3c8f608c3c5059f79a59 --data-urlencode file@./psg_cfg_master_config_after_changes.txt %s/api.php > spectra/halfwayPoint.txt' % psgurl)
    # print('curl -v -d app=globes -d type=upd --data-urlencode file@config.txt %s/api.php' % psgurl)
    # exit()

    # Calculate the spectra across the phases
    if not os.path.isdir('spectra'): os.system('mkdir ./PSGSpectra/')
    for phase in np.arange(Params.phase1,Params.phase2+Params.dphase,Params.dphase):
        if phase>178 and phase<182: phase=182 # Add transit phase;
        if phase == 185:
            phase = 186
        with open("./GCMs/config.txt", "w") as fr:
            if Params.noStar:
                phase *= -1
                fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % phase)
                phase *= -1
            else:
                fr.write('<OBJECT-SEASON>%f\n' % phase)
                fr.write('<OBJECT-OBS-LONGITUDE>%f\n' % phase)
                fr.write('<GEOMETRY-STAR-DISTANCE>0.000000e+00')
            # fr.write('<GENERATOR-CONT-STELLAR>Y\n')
            # fr.write('<GEOMETRY-PHASE>-%f\n' % phase)
            fr.close()
        if phase == 45:
            print("45")
        os.system('curl -v -d key=3c8f608c3c5059f79a59 -d app=globes --data-urlencode file@config.txt %s/api.php > %s/phase%d.txt'
                  % (Params.psgurl, Params.PSGcombinedSpectraFolder, phase))
        # if update: os.system('curl -v -d key=3c8f608c3c5059f79a59 -d type=upd --data-urlencode file@config.txt %s/api.php > spectra/phase%d.txt' % (psgurl,phase))
        # if update: os.system('curl -v -d key=3c8f608c3c5059f79a59 --data-urlencode file@psg_cfg_master_config_after_changes.txt %s/api.php > spectra/phase%d.txt' % (psgurl,phase))
        # print('curl -v -d app=globes --data-urlencode file@config.txt %s/api.php' % psgurl)
        # exit()
        # data = np.genfromtxt('spectra/phase%d.txt' % phase)
        # plt.plot(data[:,0],data[:,1],label=phase)
        print(phase)
        # f = "%d" % int(phase)
        # outfile = "%s.txt" % f
        # os.chdir(mypath)
        # os.system("curl -d -m 45 type=cfg --data-urlencode file@%s https://psg.gsfc.nasa.gov/api.php > ./proxCenRealistic/PSGConfigs/%s " %(f, outfile))
        #Dope! 

    # #Endfor phase
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    # plt.show()
    print('done')