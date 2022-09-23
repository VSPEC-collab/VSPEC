from matplotlib.ft2font import LOAD_NO_SCALE
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
from numba import jit, njit
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from statistics import mean
import matplotlib
import configparser
import os
import multiprocessing
import time
import platform
from pathlib import Path

import sys
np.set_printoptions(threshold=sys.maxsize)

# Class used to create a 2D Spot/Fac model used to create the 3D hemi models
class StarModel2D():

    def __init__(self,
        spotCoverage,
        spotNumber,
        facCoverage,
        facNumber,
        starName,):
        
        self.spotCoverage = spotCoverage
        self.spotNumber = spotNumber
        self.facCoverage = facCoverage
        self.facNumber = facNumber
        self.starName = starName
        self.rEarth = 6370.0  # km

    def generate_spots(self, randomSeed = None):
        randomSeed = 26
        # 15 8.8 spot 5.83 fac
        # 26 8.24 spot 6.1 fac
        if randomSeed is not None:
            np.random.seed(randomSeed)
            
        # Unit radius is chosen as 1
        surface_area = 4.0 * np.pi * 1.0 ** 2

        # Picks all radii of spots. All random between 0 and 1
        spot_radius = np.random.random_sample(self.spotNumber)
        # Picks all radii of faculae. All random between 0 and 1
        fac_radius = np.random.random_sample(self.facNumber)

        # Area of each spot added up to give TOTAL spot coverage
        total_spot_coverage = np.sum(np.pi * spot_radius ** 2)
        # Area of each fac added up to give TOTAL fac coverage
        total_fac_coverage = np.sum(np.pi * fac_radius ** 2)

        # Calcualates the normalization value for spots and fac, used later to achive the proper spot/fac coverage
        spotNormalization = surface_area * self.spotCoverage / total_spot_coverage
        facNormalization = surface_area * self.facCoverage / total_fac_coverage

        true_spot_radius = spot_radius * spotNormalization ** 0.5
        true_spot_coverage = np.sum(np.pi * true_spot_radius ** 2)
        true_fac_radius = fac_radius * facNormalization ** 0.5
        true_fac_coverage = np.sum(np.pi * true_fac_radius ** 2)

        # Limits latitude to between 60 and -60? Based on Butterfly effect?
        spotLat = -60 + 120 * np.random.random_sample(self.spotNumber)
        facLat = -60 + 120 * np.random.random_sample(self.facNumber)
        # Limits Longitude to between -180 and 180
        spotLon = -180 + 360 * np.random.random_sample(self.spotNumber)
        facLon = -180 + 360 * np.random.random_sample(self.facNumber)

        surface_map = self.generate_flat_surface_map(true_spot_radius, spotLon, spotLat,
                                                     true_fac_radius, facLon, facLat,)
        # print("Type of surface_map = ", type(surface_map))

        flat_image = surface_map.flatten()
        length = len(flat_image)
        spot_pixels = np.where(flat_image == 1)
        fac_pixels = np.where(flat_image == 2)
        # summ = sum(flat_image)
        surfaceMapSpotCoverage = len(spot_pixels[0]) / length
        surfaceMapFacCoverage = len(fac_pixels[0]) / length

        print("Total SpotCoverage = ", surfaceMapSpotCoverage)
        print("Total FacCoverage = ", surfaceMapFacCoverage)

        if os.path.isfile(Path('.') / f'{self.starName}' / 'Data' / 'surfaceMapInfo.txt'):
            os.remove(Path('.') / f'{self.starName}' / 'Data' / 'surfaceMapInfo.txt')
        f = open(Path('.') / f'{self.starName}' / 'Data' / 'surfaceMapInfo.txt', 'a')
        f.write('Total Spot Coverage Percentage = {:.2f}%\n'.format(surfaceMapSpotCoverage * 100))
        f.write('Total Fac Coverage Percentage = {:.2f}%\n'.format(surfaceMapFacCoverage * 100))
        f.write(f'RandomSeed = {randomSeed}')
        f.close()

        plt.close("all")
        return surface_map

    def generate_flat_surface_map(self, spot_radii, spotLon, spotLat, fac_radii, facLon, facLat):
        # we create an image using matplotlib (!!)
        fig = plt.figure(figsize=[5.00, 2.5], dpi=1200)
        proj = ccrs.PlateCarree()
        # fc stand for facecolor = (1, .714, .129)
        ax = plt.axes(projection=proj, fc=(1, .714, .129))
        canvas = FigureCanvas(fig)
        plt.gca().set_position([0, 0, 1, 1])

        ax.set_global()
        ax.outline_patch.set_linewidth(0.0)
        ax.set_extent([-180, 180, -90, 90])

        # loop through each spot, adding it to the image
        # tissot assume the sphere is earth, so multiply by radius of earth
        for spot in range(self.spotNumber):
            add_spots = ax.tissot(
                rad_km=spot_radii[spot] * self.rEarth,
                lons=spotLon[spot],
                lats=spotLat[spot],
                n_samples=1000,
                fc="k",
                alpha=1,
            )
        
        canvas.draw()
        buf = canvas.buffer_rgba()
        surface_map_image = np.asarray(buf)
        # print(surface_map_image[1000][1000])
        # 0 = photosphere
        # 1 = spot
        # 2 = fac
        # If the red value of the pixel is 255, it's a part of the photosphere, so
        # set the corresponding value in the surface map array to 0
        # If it isn't 255 (in this context then, that means it is a spot with RED value of 0),
        # set the corresponding surface map value to 1 to indicate it's a spot
        surface_map = np.where(surface_map_image[:, :, 0] == 255, 0, 1)

        # fac_map = np.where(surface_map_image[:, :, 2] == 255)
        # surface_map[fac_map] = 2
        # print(surface_map)

        plt.savefig(Path('.') / f'{self.starName}' / 'Figures' / 'SpotsOnlyFlatMap.png')

        # fc = (.953, .129, .035) or 'b'
        for fac in range(self.facNumber):
            add_facs = ax.tissot(
                rad_km=fac_radii[fac] * self.rEarth,
                lons=facLon[fac],
                lats=facLat[fac],
                n_samples=1000,
                fc=(.953, .129, .035),
                alpha=1,
            )
        
        canvas.draw()
        buf = canvas.buffer_rgba()
        surface_map_image = np.asarray(buf)
        # 0 = photosphere
        # 1 = spot
        # 2 = fac
    
        # alt from Tyler
        #full_map = fac_map (top layer) + spots_map (lower) # [0, 1, 2 , 3]
        # for every overlay of spot and fac = 2
        # full_map[np.where(full_map == 3)] = 2
        # 0 = photo 
        # 1 = spot
        # 2 = fac
        # 3 = spot + fac overlay
        
        # if fac color 'b', -> 2] == 255
        fac_map = np.where(surface_map_image[:, :, 0] == 243, 2, 0)
        for row in range(len(fac_map)):
            for col in range(len(fac_map[row])):
                if fac_map[row][col] == 2:
                    surface_map[row][col] = 2
        # print(surface_map)
        
        
        # Save and show the surface map values (0, 1, or 2) that create the 2D map image (rectangular shape)
        # NOTE: the plt.show() function does not scale well with this plot, must view from file
        plt.savefig(Path('.') / f'{self.starName}' / 'Figures/FlatMap.png')
        # plt.show()
        return surface_map

class HemiModel():
    def __init__(self,
        Params,
        teffStar,
        rotstar,
        surface_map,
        inclination,
        imageResolutiopn,
        starName,):
        
        self.Params = Params
        self.teffStar = teffStar
        self.rotstar = rotstar
        self.surface_map = surface_map
        self.inclination = inclination
        self.imageResolution = imageResolutiopn
        self.starName = starName
        self.surfaceCoverageDictionary = {}

    def multiprocessing_worker(self):
        start_time = time.time()
        ncpus = multiprocessing.cpu_count()
        
        with multiprocessing.Pool(ncpus - 1) as pool:
            print(f'total images = {self.Params.total_images}')
            results = pool.map(self.generate_hemisphere_map, range(self.Params.total_images))
        total_time = time.time() - start_time
        #360 secondsm, 6 mins
        print("TOTAL TIME (sec) = ", total_time)
        
        # Add all of the values to a dictionary that will be written to a CSV file for future, efficient use.
        for res in results:
            tempDict = {'phot':res[0], 'spot':res[1], 'fac':res[2]}
            self.surfaceCoverageDictionary[res[3]] = tempDict

    def generate_hemisphere_map(self, image_id):
        if (image_id % 10) == 0:
            print(f'id = {image_id}')
        planet_phase = (self.Params.phase1 + self.Params.delta_phase_planet * image_id) % 360
        star_phase = (self.Params.delta_phase_star * image_id) % 360
        star_phase_string = str("%.3f" % star_phase)
        # phase is between [0 and 360)
        if planet_phase > 359.9 and planet_phase <360:
            planet_phase = 360
        lonStrPlanet = str("%.3f" % planet_phase)
        self.lonStrStar = str("%.3f" % star_phase)
        if np.abs(star_phase - 180) < 0.01:
            star_phase += 0.01  # needs a litle push at 180 degrees
        image_lon_min = -180 + star_phase
        image_lon_max = 180 + star_phase
        proj = ccrs.Orthographic(
            central_longitude=0.0, central_latitude=self.inclination)
        fig = plt.figure(figsize=(5, 5), dpi=100, frameon=False)

        ax = plt.gca(projection=proj, fc="r")
        ax.outline_patch.set_linewidth(0.0)
        hemi_map = ax.imshow(
            self.surface_map,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[image_lon_min, image_lon_max, -90, 90],
            interpolation="none",
            regrid_shape=self.imageResolution
        ).get_array()

        # Calulate and save the photosphere, spot, and facuale coverage percentages for this hemisphere
        phot_frac, spot_frac, fac_frac, star_phase = self.calculate_coverage(hemi_map, star_phase_string)

        plt.title(f"Hemishpere Map: {self.starName}\nT={self.teffStar}K; Log g=5; Met=0")

        # Saves each hemishpere map image to file, indexed by the longitude
        # Number of saved images will be equal to Params.total_images value, calcualted in read_info.py
        plt.savefig(Path('.') / f'{self.starName}' / 'Figures' / 'HemiMapImages' / f'hemiMap_{star_phase_string}.png')
        # plt.show()
        plt.close("all")

        # Option to save the pickled numpy array version of the hemisphere map to load in later. Not used by default.
        # hemi_map.dump(Path('.') / f'{self.starName}' / 'Data' / 'HemiMapArrays' / f'hemiArray_{star_phase_string}')

        if ((image_id / self.Params.total_images) * 100) % 10 == 0:
            print(f"{(image_id / self.Params.total_images) * 100}% Complete")
        return phot_frac, spot_frac, fac_frac, star_phase_string

    def calculate_coverage(self, hemi_map, star_phase):
        # add where the 3rd dimension index is certain color
        flat_image = hemi_map[~hemi_map.mask].flatten()
        total_size = flat_image.shape[0]
        # print(total_size)
        photo = np.where(flat_image == 0)[0].shape[0]
        spot = np.where(flat_image == 1)[0].shape[0]
        fac = np.where(flat_image == 2)[0].shape[0]

        total_size_mod = total_size

        photo_frac =  photo / total_size_mod
        spot_frac = spot / total_size_mod
        fac_frac = fac / total_size_mod
        
        return photo_frac, spot_frac, fac_frac, star_phase