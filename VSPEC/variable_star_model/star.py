"""VSPEC star

This module contains the code to govern the
behavior of a variable star.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.units.quantity import Quantity

from VSPEC.helpers import CoordinateGrid
from VSPEC.variable_star_model import MSH
from VSPEC.variable_star_model.spots import StarSpot, SpotCollection, SpotGenerator
from VSPEC.variable_star_model.faculae import Facula, FaculaCollection, FaculaGenerator
from VSPEC.variable_star_model.flares import StellarFlare, FlareCollection, FlareGenerator


class Star:
    """
    Star object representing a variable star.

    Parameters
    ----------
    Teff : astropy.units.Quantity 
        Effective temperature of the stellar photosphere.
    radius : astropy.units.Quantity 
        Stellar radius.
    period : astropy.units.Quantity 
        Stellar rotational period.
    spots : SpotCollection
        Initial spots on the stellar surface.
    faculae : FaculaCollection
        Initial faculae on the stellar surface.
    name : str, default=''
        Name of the star.
    distance : astropy.units.Quantity , default=1*u.pc
        Distance to the star.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    flare_generator : FlareGenerator, default=None
        Flare generator object.
    spot_generator : SpotGenerator, default=None
        Spot generator object.
    fac_generator : FaculaGenerator, default=None
        Facula generator object.
    ld_params : list, default=[0, 1, 0]
        Limb-darkening parameters.

    Attributes
    ----------
    name : str
        Name of the star.
    Teff : astropy.units.Quantity 
        Effective temperature of the stellar photosphere.
    radius : astropy.units.Quantity 
        Stellar radius.
    distance : astropy.units.Quantity 
        Distance to the star.
    period : astropy.units.Quantity 
        Stellar rotational period.
    spots : SpotCollection
        Spots on the stellar surface.
    faculae : FaculaCollection
        Faculae on the stellar surface.
    gridmaker : CoordinateGrid
        Object to create the coordinate grid of the surface.
    map : astropy.units.Quantity 
        Pixel map of the stellar surface.
    flare_generator : FlareGenerator
        Flare generator object.
    spot_generator : SpotGenerator
        Spot generator object.
    fac_generator : FaculaGenerator
        Facula generator object.
    ld_params : list
        Limb-darkening parameters.
    """

    def __init__(self, Teff: u.Quantity,
                 radius: u.Quantity,
                 period: u.Quantity,
                 spots: SpotCollection,
                 faculae: FaculaCollection,
                 name: str = '',
                 distance: u.Quantity = 1*u.pc,
                 Nlat: int = 500,
                 Nlon: int = 1000,
                 gridmaker: CoordinateGrid = None,
                 flare_generator: FlareGenerator = None,
                 spot_generator: SpotGenerator = None,
                 fac_generator: FaculaGenerator = None,
                 ld_params: list = [0, 1, 0]):
        self.name = name
        self.Teff = Teff
        self.radius = radius
        self.distance = distance
        self.period = period
        self.spots = spots
        self.faculae = faculae
        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        self.map = self.get_pixelmap()
        self.faculae.gridmaker = self.gridmaker
        self.spots.gridmaker = self.gridmaker

        if flare_generator is None:
            self.flare_generator = FlareGenerator(self.Teff, self.period)
        else:
            self.flare_generator = flare_generator

        if spot_generator is None:
            self.spot_generator = SpotGenerator(500*MSH, 200*MSH, umbra_teff=self.Teff*0.75,
                                                penumbra_teff=self.Teff*0.85, Nlon=Nlon, Nlat=Nlat, gridmaker=self.gridmaker)
        else:
            self.spot_generator = spot_generator

        if fac_generator is None:
            self.fac_generator = FaculaGenerator(
                R_peak=300*u.km, R_HWHM=100*u.km, Nlon=Nlon, Nlat=Nlat)
        else:
            self.fac_generator = fac_generator
        self.ld_params = ld_params

    def get_pixelmap(self):
        """
        Create a map of the stellar surface based on spots.

        Returns
        -------
        pixelmap : astropy.units.Quantity , Shape(self.gridmaker.Nlon,self.gridmaker.Nlat)
            Map of stellar surface with effective temperature assigned to each pixel.
        """
        return self.spots.map_pixels(self.radius, self.Teff)

    def age(self, time):
        """
        Age the spots and faculae on the stellar surface according
        to their own `age` methods. Remove the spots that have decayed.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Length of time to age the features on the stellar surface.
            For most realistic behavior, `time` should be much less than
            spot or faculae lifetime.
        """
        self.spots.age(time)
        self.faculae.age(time)
        self.map = self.get_pixelmap()

    def add_spot(self, spot):
        """
        Add one or more spots to the stellar surface.

        Parameters
        ----------
        spot : StarSpot or sequence of StarSpot
            The `StarSpot` object(s) to add.
        """
        self.spots.add_spot(spot)
        self.map = self.get_pixelmap()

    def add_fac(self, facula):
        """
        Add one or more faculae to the stellar surface.

        Parameters
        ----------
        facula : Facula or sequence of Facula
            The Facula object(s) to add.

        """
        self.faculae.add_faculae(facula)

    def calc_coverage(self, sub_obs_coords):
        """
        Calculate coverage

        Calculate coverage fractions of various Teffs on stellar surface
        given coordinates of the sub-observation point.

        Parameters
        ----------
        sub_obs_coord : dict
            A dictionary giving coordinates of the sub-observation point.
            This is the point that is at the center of the stellar disk from the view of
            an observer. Format: {'lat':lat,'lon':lon} where lat and lon are
            `astropy.units.Quantity` objects.

        Returns
        -------
        dict
            Dictionary with Keys as Teff quantities and Values as surface fraction floats.
        """
        latgrid, longrid = self.gridmaker.grid()
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                 + np.cos(sub_obs_coords['lat']) * np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid))
        ld = cos_c**0 * self.ld_params[0] + cos_c**1 * \
            self.ld_params[1] + cos_c**2 * self.ld_params[2]
        ld[cos_c < 0] = 0
        jacobian = np.sin(latgrid + 90*u.deg)

        int_map, map_keys = self.faculae.map_pixels(
            self.map, self.radius, self.Teff)

        Teffs = np.unique(self.map)
        data = {}
        # spots and photosphere
        for teff in Teffs:
            pix = self.map == teff
            pix_sum = ((pix.astype('float32') * ld * jacobian)
                       [int_map == 0]).sum()
            data[teff] = pix_sum
        for i in map_keys.keys():
            facula = self.faculae.faculae[i]
            angle = 2 * np.arcsin(np.sqrt(np.sin(0.5*(facula.lat - sub_obs_coords['lat']))**2
                                          + np.cos(facula.lat)*np.cos(sub_obs_coords['lat']) * np.sin(0.5*(facula.lon - sub_obs_coords['lon']))**2))
            frac_area_dict = facula.fractional_effective_area(angle)
            loc = int_map == map_keys[i]
            pix_sum = (loc.astype('float32') * ld * jacobian).sum()
            for teff in frac_area_dict.keys():
                if teff in data:
                    data[teff] = data[teff] + pix_sum * frac_area_dict[teff]
                else:
                    data[teff] = pix_sum * frac_area_dict[teff]
        total = 0
        for teff in data.keys():
            total += data[teff]
        # normalize
        for teff in data.keys():
            data[teff] = data[teff]/total
        return data

    def calc_orthographic_mask(self, sub_obs_coords):
        """
        Calculate orthographic mask.

        Get the value of the orthographic mask at each point on the stellar surface when
        viewed from the specified sub-observation point. 


        Parameters
        ----------
        sub_obs_coords : dict
            A dictionary containing coordinates of the sub-observation point. This is the 
            point that is at the center of the stellar disk from the view of an observer. 
            Format: {'lat':lat,'lon':lon} where lat and lon are `astropy.units.Quantity` objects.

        Returns
        -------
        numpy.ndarray
            The effective pixel size when projected onto an orthographic map.
        """

        latgrid, longrid = self.gridmaker.grid()
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                 + np.cos(sub_obs_coords['lat']) * np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid))
        ld = cos_c**0 * self.ld_params[0] + cos_c**1 * \
            self.ld_params[1] + cos_c**2 * self.ld_params[2]
        ld[cos_c < 0] = 0
        return ld

    def birth_spots(self, time):
        """
        Create new spots from a spot generator.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Time over which these spots should be created.

        """
        self.spots.add_spot(self.spot_generator.birth_spots(time, self.radius))
        self.map = self.get_pixelmap()

    def birth_faculae(self, time):
        """
        Create new faculae from a facula generator.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Time over which these faculae should be created.


        """
        self.faculae.add_faculae(
            self.fac_generator.birth_faculae(time, self.radius, self.Teff))

    def average_teff(self, sub_obs_coords):
        """
        Calculate the average Teff of the star given a sub-observation point
        using the Stephan-Boltzman law. This can approximate a lightcurve for testing.

        Parameters
        ----------
        sub_obs_coords : dict
            A dictionary containing coordinates of the sub-observation point. This is the 
            point that is at the center of the stellar disk from the view of an observer. 
            Format: {'lat':lat,'lon':lon} where lat and lon are `astropy.units.Quantity` objects.

        Returns
        -------
        astropy.units.Quantity 
            Bolometric average Teff of stellar disk.

        """
        dat = self.calc_coverage(sub_obs_coords)
        num = 0
        den = 0
        for teff in dat.keys():
            num += teff**4 * dat[teff]
            den += dat[teff]
        return ((num/den)**(0.25)).to(u.K)

    def plot_spots(self, view_angle, sub_obs_point=None):
        """
        Plot spots on a map using the orthographic projection.

        Parameters
        ----------
        view_angle: dict
            Dictionary with two keys, 'lon' and 'lat', representing the longitude and
            latitude of the center of the projection in degrees.
        sub_obs_point: tuple, default=None
            Tuple with two elements, representing the longitude and latitude of the
            sub-observer point in degrees. If provided, a gray overlay is plotted
            indicating the regions that are visible from the sub-observer point.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The resulting figure object.

        Notes
        -----
        This method uses the numpy and matplotlib libraries to plot a map of the spots
        on the stellar surface using an orthographic projection centered at the
        coordinates provided in the `view_angle` parameter. The pixel map is obtained
        using the `get_pixelmap` method of `Star`. If the `sub_obs_point` parameter
        is provided, a gray overlay is plotted indicating the visible regions from the
        sub-observer point.
        """
        # This makes cartopy and optional dependency
        import cartopy.crs as ccrs

        pmap = self.get_pixelmap().value
        proj = ccrs.Orthographic(
            central_longitude=view_angle['lon'], central_latitude=view_angle['lat'])
        fig = plt.figure(figsize=(5, 5), dpi=100, frameon=False)
        ax = plt.axes(projection=proj, fc="r")
        ax.outline_patch.set_linewidth(0.0)
        ax.imshow(
            pmap.T,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[0, 360, -90, 90],
            interpolation="none",
            cmap='viridis',
            regrid_shape=(self.gridmaker.Nlat, self.gridmaker.Nlon)
        )
        if sub_obs_point is not None:
            mask = self.calc_orthographic_mask(sub_obs_point)
            ax.imshow(
                mask.T,
                origin="lower",
                transform=ccrs.PlateCarree(),
                extent=[0, 360, -90, 90],
                interpolation="none",
                regrid_shape=(self.gridmaker.Nlat, self.gridmaker.Nlon),
                cmap='gray',
                alpha=0.7
            )
        return fig

    def plot_faculae(self, view_angle):
        """
        Plot faculae on a map using orthographic projection.

        Parameters
        ----------
        view_angle: dict
            Dictionary with two keys, 'lon' and 'lat', representing the longitude and
            latitude of the center of the projection in degrees.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The resulting figure object.

        Notes
        -----
        This method uses the numpy and matplotlib libraries to plot a map of the faculae
        on the stellar surface using an orthographic projection centered at the
        coordinates provided in the `view_angle` parameter. The faculae are obtained from
        the `Star`'s faculae attribute and are mapped onto pixels using the `map_pixels`
        method. The resulting map is plotted using an intensity map with faculae pixels
        represented by the value 1 and non-faculae pixels represented by the value 0.
        """
        # This makes cartopy and optional dependency
        import cartopy.crs as ccrs

        int_map, map_keys = self.faculae.map_pixels(
            self.map, self.radius, self.Teff)
        is_fac = ~(int_map == 0)
        int_map[is_fac] = 1
        proj = ccrs.Orthographic(
            central_longitude=view_angle['lon'], central_latitude=view_angle['lat'])
        fig = plt.figure(figsize=(5, 5), dpi=100, frameon=False)
        ax = plt.axes(projection=proj, fc="r")
        ax.outline_patch.set_linewidth(0.0)
        ax.imshow(
            int_map.T,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[0, 360, -90, 90],
            interpolation="none",
            regrid_shape=(self.gridmaker.Nlat, self.gridmaker.Nlon)
        )
        return fig

    def get_flares_over_observation(self, time_duration: Quantity[u.hr]):
        """
        Generate a collection of flares over a specified observation period.

        Parameters
        ----------
        time_duration: astropy.units.Quantity 
            The duration of the observation period.

        Notes
        -----
        This method uses the `FlareGenerator` attribute of the `Star` object to generate
        a distribution of flare energies, and then generates a series of flares over the
        specified observation period using these energies. The resulting collection of
        flares is stored in the `Star`'s `flares` attribute.
        """
        energy_dist = self.flare_generator.generage_E_dist()
        flares = self.flare_generator.generate_flare_series(
            energy_dist, time_duration)
        self.flares = FlareCollection(flares)

    def get_flare_int_over_timeperiod(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr], sub_obs_coords):
        """
        Compute the total flare integral over a specified time period and sub-observer point.

        Parameters
        ----------
        tstart: astropy.units.Quantity 
            The start time of the period.
        tfinish: astropy.units.Quantity 
            The end time of the period.
        sub_obs_coords : dict
            A dictionary containing coordinates of the sub-observation point. This is the 
            point that is at the center of the stellar disk from the view of an observer. 
            Format: {'lat':lat,'lon':lon} where lat and lon are `astropy.units.Quantity` objects.


        Returns
        -------
        flare_timeareas: list of dict
            List of dictionaries containing flare temperatures and integrated
            time-areas. In the format [{'Teff':9000*u.K,'timearea'=3000*u.Unit('km2 hr)},...]

        Notes
        -----
        This method computes the total flare integral over each flare in the `flares` 
        attribute of the `Star` object that falls within the specified time period and is visible
        from the sub-observer point defined by `sub_obs_coords`. The result is returned
        as a list of dictionaries representing the teff and total flare integral of each flare.
        """
        flare_timeareas = self.flares.get_flare_integral_in_timeperiod(
            tstart, tfinish, sub_obs_coords)
        return flare_timeareas

    def generate_mature_spots(self, coverage: float):
        """
        Generate new mature spots with a specified coverage.

        Parameters
        ----------
        coverage: float
            The coverage of the new spots.

        Notes
        -----
        This method uses the `SpotGenerator` attribute of the current object to generate a
        set of new mature spots with the specified coverage. The new spots are added to 
        the object's `spots` attribute and the pixel map is updated using the new spots.
        """
        new_spots = self.spot_generator.generate_mature_spots(
            coverage, self.radius)
        self.spots.add_spot(new_spots)
        self.map = self.get_pixelmap()
