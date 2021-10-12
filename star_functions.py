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

        if os.path.isfile('./%s/Data/surfaceMapInfo.txt' % self.starName):
            os.remove('./%s/Data/surfaceMapInfo.txt' % self.starName)
        f = open('./%s/Data/surfaceMapInfo.txt' % self.starName, 'a')
        f.write('Total Spot Coverage Percentage = {:.2f}%\n'.format(surfaceMapSpotCoverage * 100))
        f.write('Total Fac Coverage Percentage = {:.2f}%\n'.format(surfaceMapFacCoverage * 100))
        f.write('RandomSeed = %s' % randomSeed)
        f.close()

        plt.close("all")
        return surface_map

    def generate_flat_surface_map(self, spot_radii, spotLon, spotLat, fac_radii, facLon, facLat):
        # we create an image using matplotlib (!!)
        fig = plt.figure(figsize=[5.00, 2.5], dpi=1200)
        proj = ccrs.PlateCarree()
        ax = plt.axes(projection=proj, fc="r")
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
        surface_map = np.where(surface_map_image[:, :, 0] == 255, 0, 1)

        # fac_map = np.where(surface_map_image[:, :, 2] == 255)
        # surface_map[fac_map] = 2
        # print(surface_map)

        plt.savefig('./%s/Figures/SpotsOnlyFlatMap.png' % self.starName)

        for fac in range(self.facNumber):
            add_facs = ax.tissot(
                rad_km=fac_radii[fac] * self.rEarth,
                lons=facLon[fac],
                lats=facLat[fac],
                n_samples=1000,
                fc="b",
                alpha=1,
            )
        
        canvas.draw()
        buf = canvas.buffer_rgba()
        surface_map_image = np.asarray(buf)
        # 0 = photosphere
        # 1 = spot
        # 2 = fac
        fac_map = np.where(surface_map_image[:, :, 2] == 255, 2, 0)
        for row in range(len(fac_map)):
            for col in range(len(fac_map[row])):
                if fac_map[row][col] == 2:
                    surface_map[row][col] = 2
        # print(surface_map)

        
        # Save and show the surface map values (1 or 0) that create the red/black map image (rectangular shape)
        # NOTE: the plt.show() function does not scale well with this plot, must view from file
        plt.savefig('./%s/Figures/FlatMap.png' % self.starName)
        # plt.show()
        return surface_map

class HemiModel():
    def __init__(self,
        teffStar,
        rotstar,
        surface_map,
        inclination,
        imageResolutiopn,
        starName,):
        
        self.teffStar = teffStar
        self.rotstar = rotstar
        self.surface_map = surface_map
        self.inclination = inclination
        self.imageResolution = imageResolutiopn
        self.starName = starName
        self.surfaceCoverageDictionary = {}

    def generate_hemisphere_map(self, phase, count):
        # phase is between 0 and 1
        lon = phase * 360
        if np.abs(lon - 180) < 0.01:
            lon += 0.01  # needs a litle push at 180 degrees
        image_lon_min = -180 + lon
        image_lon_max = 180 + lon
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
            # # Optional regrid_shape of 100 runs significantly faster
            # regrid_shape=100
        ).get_array()

        # Calulate and save the photosphere, spot, and facuale coverage percentages for this hemisphere
        self.calculate_coverage(hemi_map, phase)

        plt.title("Hemishpere Map: %s\nT=%dK; Log g=5; Met=0" % (self.starName, self.teffStar))

        # Saves each hemishpere map image to file
        # Number of saved images will be equal to num_exposures value in config file
        plt.savefig("./%s/Figures/HemiMapImages/hemiMap_%d.png" % (self.starName, count))
        # plt.show()
        plt.close("all")

        # Save the pickle numpy array version of the hemisphere map to load in other programs later
        # UNUSED? EDIT
        hemi_map.dump('./%s/Data/HemiMapArrays/hemiArray_%d' % (self.starName, count))

        return hemi_map

    def calculate_coverage(self, hemi_map, phase, ignore_planet=False):
        # add where the 3rd dimension index is certain color
        flat_image = hemi_map[~hemi_map.mask].flatten()
        total_size = flat_image.shape[0]
        # print(total_size)
        photo = np.where(flat_image == 0)[0].shape[0]
        spot = np.where(flat_image == 1)[0].shape[0]
        fac = np.where(flat_image == 2)[0].shape[0]

        # print(photo)
        # print(spot)
        # print(fac)
        # print(photo+spot+fac)

        # planet  = np.where(flat_image == 2)[0].shape[0]

        if ignore_planet:
        #    total_size_mod = total_size - planet
            pass
        else:
            total_size_mod = total_size

        photo_frac =  photo / total_size_mod
        spot_frac = spot / total_size_mod
        fac_frac = fac / total_size_mod

        # Save these values to a dictionary
        # The dictionary key is the phase, the dictionary value is another dictionary that contatins the key-value pairs of 
        # photosphere: coverage_percentage; spot: coverage_percentage; faculae: coverage_percentage
        tempDict = {'phot':photo_frac, 'spot':spot_frac, 'fac':fac_frac}
        self.surfaceCoverageDictionary[phase] = tempDict