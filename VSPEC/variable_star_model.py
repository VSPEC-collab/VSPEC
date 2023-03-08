"""VSPEC variable star module

This module describes the stellar variability
contianed in `VSPEC`'s model.
"""

from copy import deepcopy
from typing import List, Dict
import typing as Typing

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u, constants as const
from astropy.units.quantity import Quantity
import cartopy.crs as ccrs
from xoflares.xoflares import _flareintegralnp as flareintegral, get_light_curvenp
from VSPEC.helpers import to_float


MSH = u.def_unit('micro solar hemisphere', 1e-6 * 0.5 * 4*np.pi*u.R_sun**2)
"""Micro-solar hemisphere

This is a standard unit in heliophysics that
equals one millionth of one half the surface area of the Sun.
"""


class CoordinateGrid:
    """
    Class to standardize the creation of latitude and longitude grids.

    Parameters
    ----------
    Nlat : int, default=500
        Number of latitude points.
    Nlon : int, default=1000
        Number of longitude points.

    Attributes
    ----------
    Nlat : int, default=500
        Number of latitude points.
    Nlon : int, default=1000
        Number of longitude points.

    """

    def __init__(self, Nlat=500, Nlon=1000):
        assert isinstance(Nlat, int)
        assert isinstance(Nlon, int)
        self.Nlat = Nlat
        self.Nlon = Nlon

    def oned(self):
        """
        Create one dimensional arrays of latitude and longitude points.

        Returns
        -------
        lats : `~astropy.unit.quantity.Quantity` [angle], shape=(Nlat,)
            Array of latitude points.
        lons : `~astropy.unit.quantity.Quantity` [angle], shape=(Nlon,)
            Array of longitude points.

        """
        lats = np.linspace(-90, 90, self.Nlat)*u.deg
        lons = np.linspace(0, 360, self.Nlon)*u.deg
        return lats, lons

    def grid(self):
        """
        Create a 2 dimensional grid of latitude and longitude points.

        Returns
        -------
        lats : astropy.unit.quantity.Quantity [angle], shape=(Nlat,Nlon)
            Array of latitude points.
        lons : astropy.unit.quantity.Quantity [angle], shape=(Nlat,Nlon)
            Array of longitude points.

        """
        lats, lons = self.oned()
        return np.meshgrid(lats, lons)

    def zeros(self, dtype='float32'):
        """
        Get a grid of zeros.

        Parameters
        ----------
        dtype : str, default='float32
            Data type to pass to np.zeros.

        Returns
        -------
        arr : np.ndarray, shape=(Nlon, Nlat)
            Grid of zeros.

        """
        return np.zeros(shape=(self.Nlon, self.Nlat), dtype=dtype)

    def __eq__(self, other):
        """
        Check to see if two CoordinateGrid objects are equal.

        Parameters
        ----------
        other : CoordinateGrid
            Another CoordinateGrid object.

        Returns
        -------
        bool
            Whether the two objects have equal properties.

        """
        if not isinstance(other, CoordinateGrid):
            return False
        else:
            return (self.Nlat == other.Nlat) & (self.Nlon == other.Nlon)


class StarSpot:
    """
    Star Spot

    Class to govern behavior of spots on a star's surface.

    Parameters
    ----------
    lat : `~astropy.units.quantity.Quantity` [angle]
        Latitude of spot center. North is positive.
    lon : `~astropy.units.quantity.Quantity` [angle]
        Longitude of spot center. East is positive.
    Amax : `~astropy.units.quantity.Quantity` [area]
        The maximum area a spot reaches before it decays.
    A0 : `~astropy.units.quantity.Quantity` [area]
        The current spot area.
    Teff_umbra : `~astropy.units.quantity.Quantity` [temperature]
        The effective temperature of the spot umbra.
    Teff_penumbra : `~astropy.units.quantity.Quantity` [temperature]
        The effective temperature of spot penumbra.
    r_A : float
        The ratio of total spot area to umbra area. 5+/-1 according to [1]_.
    growing : bool
        Whether or not the spot is growing.
    growth_rate : `~astropy.units.quantity.Quantity` [frequency]
        Fractional growth of the spot for a given unit time.
        From from sunspot literature, can be 0.52/day to 1.83/day [1]_.
        According to M dwarf literature, can effectively be 0 [2]_.
    decay_rate : `~astropy.units.quantity.Quantity` [area per time]
        The rate at which a spot linearly decays. From sunspot
        literature, this is 10.89 MSH/day [1]_. According to M dwarf
        literature, this can be 0 [2]_.
    Nlat : int, default=500
        Number of latitude points.
    Nlon : int, default=1000
        Number of longitude points.
    gridmaker : CoordinateGrid, default=None
        Coordinate grid object to produce points in the
        stellar surface. Ideally, this is passed from a
        container object (such as `SpotCollection`).

    Attributes
    ----------
    coords : dict
        A dictionary containing the latitude and longitude of the spot's center.
    area_max : `~astropy.units.quantity.Quantity` [area]
        The maximum area a spot reaches before it decays.
    area_current : `~astropy.units.quantity.Quantity` [area]
        The current area of the spot.
    Teff_umbra : `~astropy.units.quantity.Quantity` [temperature]
        The effective temperature of the spot umbra.
    Teff_penumbra : `~astropy.units.quantity.Quantity` [temperature]
        The effective temperature of the spot penumbra.
    decay_rate : `~astropy.units.quantity.Quantity` [area per time]
        The rate at which a spot linearly decays.
    total_area_over_umbra_area : float
        The ratio of total spot area to umbra area. 5+/-1 according to [1]_.
    is_growing : bool
        Whether or not the spot is growing.
    growth_rate : `~astropy.units.quantity.Quantity` [frequency]
        Fractional growth of the spot for a given unit time.
    gridmaker : `~CoordinateGrid` or None
        A `CoordinateGrid` object used to produce points on the stellar surface. If None,
        a `CoordinateGrid` object is created with default parameters.
    r : `~np.ndarray`
        An array of points on the stellar surface with their pre-computed
        distance from the center of the spot.


    References
    ----------
    .. [1] Goulding, N. T. 2013, PhD thesis, University of
        Hertfordshire, UK
    .. [2] Davenport, J. R. A., Hebb, L., & Hawley, S. L. 2015, ApJ,
        806, 212
    """

    def __init__(
        self, lat: Quantity[u.deg], lon: Quantity[u.deg], Amax: Quantity[MSH], A0: Quantity[MSH],
        Teff_umbra: Quantity[u.K], Teff_penumbra: Quantity[u.K], r_A: float = 5, growing: bool = True,
        growth_rate: Quantity[1/u.day] = 0.52/u.day, decay_rate: Quantity[MSH/u.day] = 10.89 * MSH/u.day,
        Nlat: int = 500, Nlon: int = 1000, gridmaker=None
    ):

        self.coords = {'lat': lat, 'lon': lon}
        self.area_max = Amax
        self.area_current = A0
        self.Teff_umbra = Teff_umbra
        self.Teff_penumbra = Teff_penumbra
        self.decay_rate = decay_rate
        self.total_area_over_umbra_area = r_A
        self.is_growing = growing
        self.growth_rate = growth_rate

        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        latgrid, longrid = self.gridmaker.grid()
        lat0 = self.coords['lat']
        lon0 = self.coords['lon']
        self.r = 2 * np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
                                       + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))

    def __str__(self):
        s = 'StarSpot with '
        s += f'Teff = ({self.Teff_umbra:.0f},{self.Teff_penumbra:.0f}), '
        s += f'area = {self.area_current.to(MSH):.0f}, '
        s += f'lat = {self.coords["lat"]:.1f}, lon = {self.coords["lon"]:.1f}'
        return s

    def radius(self) -> Quantity[u.km]:
        """
        Radius

        Get the radius of the spot.

        Returns
        -------
        `~astropy.units.quantity.Quantity` [distance]
            The radius of the spot.
        """
        return np.sqrt(self.area_current/np.pi).to(u.km)

    def angular_radius(self, star_rad: Quantity[u.R_sun]) -> Quantity[u.deg]:
        """
        Angular radius

        Get the angular radius of the spot on the stellar surface.

        Parameters
        ----------
        star_rad : `~astropy.units.quantity.Quantity` [distance]
            The radius of the star.

        Returns
        -------
        `~astropy.units.quantity.Quantity` [angle]
            The angular radius of the spot.
        """
        cos_angle = 1 - self.area_current/(2*np.pi*star_rad**2)
        return (np.arccos(cos_angle)).to(u.deg)

    def map_pixels(self, star_rad: Quantity[u.R_sun]) -> dict:
        """
        Map latitude and longituide points continaing the umbra and penumbra

        Parameters
        ----------
        star_rad : `~astropy.units.quantity.Quantity` [length]
            The radius of the star.

        Returns
        -------
        dict
            Dictionary of points covered by the umbra and penumbra. Keys are the
            effective temperature (Teff) of each region. Values are numpy boolean
            arrays.
        Notes
        -----
        This function calculates the angular radius of the spot based on the
        provided `star_rad`, and uses the `total_area_over_umbra_area` attribute to
        determine the size of the spot. Then, it returns a dictionary with two
        keys, `Teff_umbra` and `Teff_penumbra`, whose values are boolean arrays
        indicating which points are covered by each region.
        """
        radius = self.angular_radius(star_rad)
        radius_umbra = radius/np.sqrt(self.total_area_over_umbra_area)
        return {self.Teff_umbra: self.r < radius_umbra,
                self.Teff_penumbra: self.r < radius}

    def surface_fraction(self, sub_obs_coords: dict,
                         star_rad: Quantity[u.R_sun], N: int = 1001) -> float:
        """
        Determine the surface fraction covered by a spot from a given 
        angle of observation using the orthographic projection.

        Parameters
        ----------
        sub_obs_coords : dict
            Dictionary giving coordinates of the sub-observation point. This
            is the point that is at the center of the stellar disk from the
            view of an observer. Format: {'lat': lat, 'lon': lon} where lat and
            lon are `~astropy.units.quantity.Quantity` objects.
        star_rad : `~astropy.units.quantity.Quantity` [length]
            Radius of the star.
        N : int, optional
            Number of points to use in numerical integration.
            N=1000 is not so different from N=100000.

        Returns
        -------
        float
            Fraction of observed disk covered by the spot.

        """
        cos_c0 = (np.sin(sub_obs_coords['lat']) * np.sin(self.coords['lat'])
                  + np.cos(sub_obs_coords['lat']) * np.cos(self.coords['lat'])
                  * np.cos(sub_obs_coords['lon']-self.coords['lon']))
        c0 = np.arccos(cos_c0)
        c = np.linspace(-90, 90, N)*u.deg
        a = self.angular_radius(star_rad).to(u.deg)
        rad = a**2 - (c-c0)**2
        rad[rad < 0] = 0
        integrand = 2 * np.cos(c)*np.sqrt(rad)
        return to_float(np.trapz(integrand, x=c)/(2*np.pi*u.steradian), u.Unit(''))

    def age(self, time: Quantity[u.s]) -> None:
        """
        Age a spot according to its growth timescale and decay rate

        Parameters
        ----------
        time : `~astropy.units.quantity.Quantity` [time]
            Length of time to age the spot. For most realistic behavior,
            time should be << spot lifetime

        Notes
        -----
        This method updates the `area_current` attribute of the `Sunspot`
        object to simulate the growth and decay of a sunspot.

        If the spot is growing, it calculates the time required to reach
        the maximum area (`time_to_max`) using the growth rate (`growth_rate`)
        and the maximum area (`area_max`). If `time_to_max` is greater than
        the input time (`time`), it grows the spot and updates the `area_current` 
        attribute. If `time_to_max` is less than or equal to the input time, it
        sets the `is_growing` attribute to False and calculates the area that the
        spot should decay (`area_decay`). If the decayed area (`area_decay`) is
        greater than the maximum area (`area_max`), the `area_current` attribute
        is set to zero. Otherwise, it updates the `area_current` attribute accordingly.

        If the spot is not growing, it calculates the area that the spot should
        decay (`area_decay`). If the decayed area (`area_decay`) is greater than
        the maximum area (`area_max`), the `area_current` attribute is set to zero.
        Otherwise, it updates the `area_current` attribute accordingly.
        """
        if self.is_growing:
            tau = np.log((self.growth_rate * u.day).to(u.Unit('')) + 1)
            if tau == 0:
                time_to_max = np.inf*u.day
            else:
                time_to_max = np.log(
                    self.area_max/self.area_current)/tau * u.day
            if time_to_max > time:
                new_area = self.area_current * np.exp(tau * time/u.day)
                self.area_current = new_area
            else:
                self.is_growing = False
                decay_time = time - time_to_max
                area_decay = decay_time * self.decay_rate
                if area_decay > self.area_max:
                    self.area_current = 0*MSH
                else:
                    self.area_current = self.area_max - area_decay
        else:
            area_decay = time * self.decay_rate
            if area_decay > self.area_max:
                self.area_current = 0*MSH
            else:
                self.area_current = self.area_current - area_decay


class SpotCollection:
    """
    Container holding StarSpot objects

    Parameters
    ----------
    *spots : tuple of StarSpot objects
        A series of StarSpot objects to be added to the collection.
    Nlat : int, default=500
        Number of latitude points.
    Nlon : int, default=1000
        Number of longitude points.
    gridmaker : CoordinateGrid, default=None
        `CoordinateGrid` object, probably passed from a
        `Star`

    Notes
    -----
    This class is a container for `StarSpot` objects. It can be
    used to store a series of spots and to apply operations to the entire collection.

    Attributes
    ----------
    spots : tuple of StarSpot objects
        Series of `StarSpot` objects in the collection.
    gridmaker : CoordinateGrid object
        `CoordinateGrid` object used to calculate the grid of the stellar surface.
    """

    def __init__(self, *spots: tuple[StarSpot], Nlat: int = 500, Nlon: int = 1000, gridmaker=None):
        self.spots = spots
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        for spot in spots:
            spot.gridmaker = self.gridmaker

    def add_spot(self, spot: Typing.Union[StarSpot, list[StarSpot]]):
        """
        Add a `StarSpot` object or a list of `StarSpot` objects to the collection.

        Parameters
        ----------
        spot : `StarSpot` or list of `StarSpot` objects
            `StarSpot` object(s) to be added to the collection.

        Notes
        -----
        This method adds a `StarSpot` object or a list of `StarSpot`
        objects to the `spots` attribute of the `SpotCollection` object.

        For every spot added to the `spots` attribute, each first has it's
        own `gridmaker` attribute set to be identical to the `SpotColelction`'s
        own `gridmaker` attribute.
        """
        if isinstance(spot, StarSpot):
            spot.gridmaker = self.gridmaker
        else:
            for s in spot:
                s.gridmaker = self.gridmaker
        self.spots += tuple(spot)

    def clean_spotlist(self):
        """
        Remove spots that have decayed to 0 area from
        the `spots` attribute of the `SpotCollection` object.

        Notes
        -----
        This method iterates over the `StarSpot` objects in the `spots`
        attribute of the `SpotCollection` object and removes those that
        have decayed to 0 area and are not growing. The remaining
        `StarSpot` objects are stored back in the `spots` attribute of
        the `SpotCollection` object.
        """
        spots_to_keep = []
        for spot in self.spots:
            if (spot.area_current <= 0*MSH) and (not spot.is_growing):
                pass
            else:
                spots_to_keep.append(spot)
        self.spots = spots_to_keep

    def map_pixels(self, star_rad: Quantity[u.R_sun], star_teff: Quantity[u.K]):
        """
        Map latitude and longitude points containing the umbra and penumbra
        of each spot. For pixels with coverage from multiple spots, assign
        the coolest Teff to that pixel.

        Parameters
        ----------
        latgrid : `~astropy.units.quantity.Quantity` [angle], shape(M,N)
            Grid of latitude points to map.
        longrid : `~astropy.units.quantity.Quantity` [angle], shape(M,N)
            Grid of longitude points to map.
        star_rad : `~astropy.units.quantity.Quantity` [length]
            Radius of the star.
        star_teff : `~astropy.units.quantity.Quantity` [temperature]
            Temperature of the star.

        Returns
        -------
        surface_map : array of astropy.units.quantity.Quantity [temperature], shape(M,N)
            Map of the stellar surface with Teff assigned to each pixel
        """
        surface_map = self.gridmaker.zeros()*star_teff.unit + star_teff
        for spot in self.spots:
            teff_dict = spot.map_pixels(star_rad)
            # penumbra
            assign = teff_dict[spot.Teff_penumbra] & (
                surface_map > spot.Teff_penumbra)
            surface_map[assign] = spot.Teff_penumbra
            # umbra
            assign = teff_dict[spot.Teff_umbra] & (
                surface_map > spot.Teff_umbra)
            surface_map[assign] = spot.Teff_umbra
        return surface_map

    def age(self, time: Quantity[u.day]) -> None:
        """
        Age spots according to its growth timescale and decay rate.

        Remove spots that have decayed.

        Parameters
        ----------
        time : `~astropy.units.quantity.Quantity` [time]
            Length of time to age the spot. For most realistic
            behavior, time should be << spot lifetime.
        """
        for spot in self.spots:
            spot.age(time)
        self.clean_spotlist()


class SpotGenerator:
    """Spot Generator

    Class controling the birth rates and properties of new spots.
    This class is based on various studies, but since in general starspots cannot
    be resolved, lots of gaps are filled in with studies of sunspots.

    Parameters
    ----------
    average_area : `~astropy.units.quantity.Quantity` [area]
        The average peak spot area.
    area_spread : float
        The standard deviation, in dex, of the lognormal peak spot area distribution
    umbra_teff : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of umbra.
    penumbra_teff : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of penumbra.
    growth_rate : `~astropy.units.quantity.Quantity` [frequency], default=0.52/u.day
        The spot growth rate.
    decay_rate : astropy.units.quantity.Quantity [area/time], default = 10.89 * MSH/u.day
        The spot decay rate.
    starting_size : astropy.units.quantity.Quantity [area], default=10*MSH
        The area of each spot at birth.
    distribution : str, default='solar'
        The spot distribution method. 'iso' or 'solar'.
    coverage : float, default=0.2
        The factional coverage of surface by spots in growth-decay equillibrium.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.

    Attributes
    ----------
    average_spot_area : `~astropy.units.quantity.Quantity` [area]
        The average peak spot area.
    spot_area_spread : float
        The standard deviation, in dex, of the lognormal peak spot area distribution.
    umbra_teff : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of umbra.
    penumbra_teff : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of penumbra.
    growth_rate : `~astropy.units.quantity.Quantity` [frequency]
        The spot growth rate.
    decay_rate : `~astropy.units.quantity.Quantity` [area/time]
        The spot decay rate.
    starting_size : `~astropy.units.quantity.Quantity` [area]
        The area of each spot at birth.
    distribution : str
        The spot distribution method. Choose from 'iso' or 'solar'.
    average_spot_lifetime : `~astropy.units.quantity.Quantity` [time]
        The average lifetime of a spot.
    coverage : float
        The fractional coverage of the surface by spots in growth-decay equilibrium.
    gridmaker : `~CoordinateGrid` or subclass
        A `CoordinateGrid` object to create the stellar surface grid.

    Notes
    ----
    The `distribution` parameter can have values of 'iso' of 'solar'. The 'iso' keyword
    distributes spots evenly across the surface. The 'solar' keyword, on the other hand,
    distributes spots according to their know clustering around +/- 15 degrees solar
    latitude [1]_.

    References
    ----------
    .. [1] Mandal, S., Karak, B. B., & Banerjee, D. 2017, ApJ, 851, 70

    """

    def __init__(self,
                 average_area: Quantity[MSH],
                 area_spread: float,
                 umbra_teff: Quantity[u.K],
                 penumbra_teff: Quantity[u.K],
                 growth_rate: Quantity[1/u.day] = 0.52/u.day,
                 decay_rate: Quantity[MSH/u.day] = 10.89 * MSH/u.day,
                 starting_size: Quantity[MSH] = 10*MSH,
                 distribution='solar',
                 coverage: float = 0.2,
                 Nlat: int = 500,
                 Nlon: int = 1000,
                 gridmaker=None
                 ):
        self.average_spot_area = average_area
        self.spot_area_spread = area_spread
        self.umbra_teff = umbra_teff
        self.penumbra_teff = penumbra_teff
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.starting_size = starting_size
        self.distribution = distribution
        self.average_spot_lifetime = 2 * \
            (self.average_spot_area / self.decay_rate).to(u.hr)
        self.coverage = coverage
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker

    def generate_spots(self, N: int) -> tuple[StarSpot]:
        """
        Create a specified number of `StarSpot` objects.

        Parameters
        ----------
        N : int
            Number of spots to create.

        Returns
        -------
        tuple[StarSpot]
            Tuple of new `StarSpot` objects.

        Raises
        ------
        ValueError
            If an unknown value is given for distribution.
        """
        new_max_areas = np.random.lognormal(mean=np.log(
            self.average_spot_area/MSH), sigma=self.spot_area_spread, size=N)*MSH
        new_r_A = np.random.normal(loc=5, scale=1, size=N)
        while np.any(new_r_A <= 0):
            new_r_A = np.random.normal(loc=5, scale=1, size=N)
        # now assign lat and lon (dist approx from 2017ApJ...851...70M)
        if self.distribution == 'solar':
            hemi = np.random.choice([-1, 1], size=N)
            lat = np.random.normal(15, 5, size=N)*hemi*u.deg
            lon = np.random.random(size=N)*360*u.deg
        elif self.distribution == 'iso':
            lon = np.random.random(size=N)*360*u.deg
            lats = np.arange(90)
            w = np.cos(lats*u.deg)
            lat = (np.random.choice(lats, p=w/w.sum(), size=N) +
                   np.random.random(size=N))*u.deg * np.random.choice([1, -1], size=N)
        else:
            raise ValueError(
                f'Unknown value {self.distribution} for distribution')

        penumbra_teff = self.penumbra_teff
        umbra_teff = self.umbra_teff

        spots = []
        for i in range(N):
            spots.append(StarSpot(
                lat[i], lon[i], new_max_areas[i], self.starting_size, umbra_teff, penumbra_teff,
                growth_rate=self.growth_rate, decay_rate=self.decay_rate,
                r_A=new_r_A[i], Nlat=self.gridmaker.Nlat, Nlon=self.gridmaker.Nlon, gridmaker=self.gridmaker
            ))
        return tuple(spots)

    def birth_spots(self, time: Quantity[u.day], rad_star: Quantity[u.R_sun],) -> tuple[StarSpot]:
        """
        Generate new `StarSpot` objects to be birthed over a given time duration.

        Parameters
        ----------
        time : `~astropy.units.quantity.Quantity` [time]
            Amount of time in which to birth spots.
            The total number of new spots will consider this time and the birthrate.
        rad_star : `~astropy.units.quantity.Quantity` [length]
            The radius of the star.

        Returns
        -------
        Tuple[StarSpot]
            New `StarSpot` objects.
        """
        N_exp = (self.coverage * 4*np.pi*rad_star**2 / self.average_spot_area
                 * time/self.average_spot_lifetime).to(u.Unit(''))
        # N_exp is the expectation value of N, but this is a poisson process
        N = max(0, round(np.random.normal(loc=N_exp, scale=np.sqrt(N_exp))))

        return self.generate_spots(N)

    def generate_mature_spots(self, coverage: float, R_star: Quantity[u.R_sun]) -> List[StarSpot]:
        """Generate mature StarSpot objects to cover a given fraction of the star's surface.

        This method generates mature spots such that the total solid angle subtended by the spots
        covers a specified fraction of the star's surface.

        Parameters
        ----------
        coverage : float
            The fraction of the star's surface to be covered by the spots.
        R_star : `~astropy.units.quantity.Quantity` [length]
            The radius of the star.

        Returns:
            List[StarSpot]: A list of mature spots generated by this method.

        Raises:
            ValueError: If the coverage is greater than 1 or less than 0.
        """
        if coverage > 1 or coverage < 0:
            raise ValueError('Coverage must be between 0 and 1.')
        spots = []
        current_omega = 0*(u.deg**2)
        target_omega = (4*np.pi*coverage*u.steradian).to(u.deg**2)
        while current_omega < target_omega:
            new_spot = self.generate_spots(1)[0]
            const_spot = (new_spot.decay_rate == 0*MSH /
                          u.day) or (new_spot.growth_rate == 0/u.day)
            if const_spot:
                area0 = self.starting_size
                area_range = new_spot.area_max - area0
                area = np.random.random()*area_range + area0
                new_spot.area_current = area
            else:
                decay_lifetime = (new_spot.area_max /
                                  new_spot.decay_rate).to(u.day)
                tau = new_spot.growth_rate
                grow_lifetime = (np.log(
                    to_float(new_spot.area_max/self.starting_size, u.Unit('')))/tau).to(u.day)
                lifetime = grow_lifetime+decay_lifetime
                age = np.random.random() * lifetime
                new_spot.age(age)
            spots.append(new_spot)
            spot_solid_angle = new_spot.angular_radius(R_star)**2 * np.pi
            current_omega += spot_solid_angle
        return spots


class Facula:
    """
    Class containing model parameters of stellar faculae using the 'hot wall' model.

    Parameters
    ----------
    lat : `~astropy.units.quantity.Quantity` [angle]
        Latitude of facula center
    lon : `~astropy.units.quantity.Quantity` [angle]
        Longitude of facula center
    Rmax : `~astropy.units.quantity.Quantity` [length]
        Maximum radius of facula
    R0 : `~astropy.units.quantity.Quantity` [length]
        Current radius of facula
    Zw : `~astropy.units.quantity.Quantity` [length]
        Depth of the depression.
    Teff_floor : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of the 'cool floor'
    Teff_wall : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of the 'hot wall'
    lifetime : `~astropy.units.quantity.Quantity` [time]
        Facula lifetime
    growing : bool, default=True
        Whether or not the facula is still growing.
    floor_threshold : `~astropy.units.quantity.Quantity` [length], default=20*u.km
        Facula radius under which the floor is no longer visible.
        Small faculae appear as bright points regardless of their
        distance to the limb.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.

    Attributes
    ----------
    lat : `~astropy.units.quantity.Quantity` [angle]
        Latitude of facula center.
    lon : `~astropy.units.quantity.Quantity` [angle]
        Longitude of facula center.
    Rmax : `~astropy.units.quantity.Quantity` [length]
        Maximum radius of facula.
    current_R : `~astropy.units.quantity.Quantity` [length]
        Current radius of facula.
    Zw : `~astropy.units.quantity.Quantity` [length]
        Depth of the depression.
    Teff_floor : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of the 'cool floor'.
    Teff_wall : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of the 'hot wall'.
    lifetime : `~astropy.units.quantity.Quantity` [time]
        Facula lifetime.
    is_growing : bool
        Whether or not the facula is still growing.
    floor_threshold : `~astropy.units.quantity.Quantity` [length]
        Facula radius under which the floor is no longer visible.
        Small faculae appear as bright points regardless of their
        distance to the limb.
    gridmaker : `CoordinateGrid` object
        A `CoordinateGrid` object to create the stellar sufrace grid.
    r : `~astropy.units.quantity.Quantity` [distance]
        The distance between the center of the faculae and each point on the stellar surface.

    Notes
    -----
    The "Hot wall" model of solar facule describes them as a depression on the
    stellar surface with a hot wall and cool floor [1]_. Because if this, faculae
    appear bright when they are near the limb (hot wall is visible) and dark when near
    the disk center (cool floor is visible).

    References
    ----------
    .. [1] Spruit, H. C. 1976, SoPh, 50, 269
    """

    def __init__(self,
                 lat: Quantity[u.deg], lon: Quantity[u.deg], Rmax: Quantity[u.km], R0: Quantity[u.km],
                 Teff_floor: Quantity[u.K], Teff_wall: Quantity[u.K], lifetime: Quantity[u.day],
                 growing: bool = True, floor_threshold: Quantity[u.km] = 20*u.km, Zw: Quantity[u.km] = 100*u.km,
                 Nlat: int = 500, Nlon: int = 1000, gridmaker=None
                 ):
        self.lat = lat
        self.lon = lon
        self.Rmax = Rmax
        self.current_R = R0
        self.Zw = Zw
        self.Teff_floor = self.round_teff(Teff_floor)
        self.Teff_wall = self.round_teff(Teff_wall)
        self.lifetime = lifetime
        self.is_growing = growing
        self.floor_threshold = floor_threshold

        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker

        latgrid, longrid = self.gridmaker.grid()
        self.r = 2 * np.arcsin(np.sqrt(np.sin(0.5*(lat-latgrid))**2
                                       + np.cos(latgrid)*np.cos(lat)*np.sin(0.5*(lon - longrid))**2))

    def age(self, time: Quantity[u.day]):
        """
        Progress the development of the facula by an amount of time.

        Parameters
        ----------
        time : `~astropy.units.quantity.Quantity` [time]
            The amount of time to age facula.

        Notes
        -----
        This method calculates the new radius of the facula based on the amount of
        time elapsed since the last time it was updated. If the facula is still growing,
        it checks if it has reached the maximum radius and sets the `is_growing` attribute
        to False if so. If the facula is no longer growing, it shrinks over time.

        """
        if self.is_growing:
            T_from_max = -1*np.log(self.current_R/self.Rmax)*self.lifetime*0.5
            if T_from_max <= time:
                self.is_growing = False
                time = time - T_from_max
                self.current_R = self.Rmax * np.exp(-2*time/self.lifetime)
            else:
                self.current_R = self.current_R * np.exp(2*time/self.lifetime)
        else:
            self.current_R = self.current_R * np.exp(-2*time/self.lifetime)

    def round_teff(self, teff):
        """
        Round the effective temperature to the nearest integer.
        The goal is to reduce the number of unique effective temperatures
        while not affecting the accuracy of the model.

        Parameters
        ----------
        teff : `~astropy.units.quantity.Quantity` [temperature]
            The temperature to round.

        Returns
        -------
        `~astropy.units.quantity.Quantity` [temperature]
            The rounded temperature.
        """
        val = teff.value
        unit = teff.unit
        return int(round(val)) * unit

    def effective_area(self, angle, N=101):
        """
        Calculate the effective area of the floor and walls when projected on a disk.

        Parameters
        ----------
        angle : `~astropy.units.quantity.Quantity` [angle]
            Angle from disk center.
        N : int, optional
            Number of points to sample the facula with. Default is 101.

        Returns
        -------
        dict
            Effective area of the wall and floor. The keys are the Teff, the
            values are the area. Both are `astropy.units.quantity.Quantity` objects.
        """
        if self.current_R < self.floor_threshold:
            return {self.round_teff(self.Teff_floor): 0.0 * u.km**2, self.round_teff(self.Teff_wall): np.pi*self.current_R**2 * np.cos(angle)}
        else:
            # distance from center along azmuth of disk
            x = np.linspace(0, 1, N) * self.current_R
            # effective radius of the 1D facula approximation
            h = np.sqrt(self.current_R**2 - x**2)
            critical_angles = np.arctan(2*h/self.Zw)
            Zeffs = np.sin(angle)*np.ones(N) * self.Zw
            Reffs = np.cos(angle)*h*2 - self.Zw * np.sin(angle)
            no_floor = critical_angles < angle
            Zeffs[no_floor] = h[no_floor]*np.cos(angle)
            Reffs[no_floor] = 0

            return {self.round_teff(self.Teff_wall): np.trapz(Zeffs, x), self.round_teff(self.Teff_floor): np.trapz(Reffs, x)}

    def fractional_effective_area(self, angle: Quantity[u.deg],
                                  N: int = 101) -> Dict[Quantity[u.K], Quantity]:
        """
        Calculate the fractional effective area as a fraction of the
        projected area of a region of quiet photosphere with
        the same radius and distance from limb.

        Parameters
        ----------
        angle : `~astropy.units.quantity.Quantity` [angle]
            Angle from disk center.
        N : int, default=101
            Number of points to sample the facula with.

        Returns
        -------
        dict
            Fractional effective area of the wall and floor. Keys are Teff.

        """
        effective_area = self.effective_area(angle, N=N)
        frac_eff_area = {}
        total = 0
        for teff in effective_area.keys():
            total = total + effective_area[teff]
        for teff in effective_area.keys():
            frac_eff_area[teff] = (
                effective_area[teff]/total).to(u.dimensionless_unscaled)
        return frac_eff_area

    def angular_radius(self, star_rad: u.Quantity):
        """
        Calculate the angular radius of the facula.

        Parameters
        ----------
        star_rad : `~astropy.units.Quantity` [length]
            The radius of the star.

        Returns
        -------
        `~astropy.units.Quantity` [angle]
            The angular radius of the facula.
        """
        return self.current_R/star_rad * 180/np.pi * u.deg

    def map_pixels(self, star_rad):
        """
        Map pixels onto the surface of the facula.

        Parameters
        ----------
        star_rad : `~astropy.units.quantity.Quantity`
            The radius of the star.

        Returns
        -------
        `~numpy.ndarray`
            Boolean array indicating whether each pixel is within the facula radius.
        """
        rad = self.angular_radius(star_rad)
        pix_in_fac = self.r <= rad
        return pix_in_fac


class FaculaCollection:
    """
    Container class to store faculae.

    Parameters
    ----------
    *faculae : tuple
        A series of faculae objects.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.

    Attributes
    ----------
    faculae : tuple
        Series of faculae objects.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    """

    def __init__(self, *faculae: tuple,
                 Nlat: int = 500,
                 Nlon: int = 1000,
                 gridmaker: CoordinateGrid = None):
        self.faculae = tuple(faculae)

        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        for facula in faculae:
            facula.gridmaker = self.gridmaker

    def add_faculae(self, facula):
        """
        Add a facula or faculae

        Parameters
        ----------
        facula : Facula or series of Facula
            Facula object(s) to add.
        """
        if isinstance(facula, Facula):
            facula.gridmaker = self.gridmaker
        else:
            for fac in facula:
                fac.gridmaker = self.gridmaker
        self.faculae += tuple(facula)

    def clean_faclist(self) -> None:
        """
        Remove faculae that have decayed to Rmax/e**2 radius.
        """
        faculae_to_keep = []
        for facula in self.faculae:
            if (facula.current_R <= facula.Rmax/np.e**2) and (not facula.is_growing):
                pass
            else:
                faculae_to_keep.append(facula)
        self.faculae = faculae_to_keep

    def age(self, time: u.Quantity) -> None:
        """
        Age spots according to their growth timescale and decay rate.
        Remove spots that have decayed.

        Parameters
        ----------
        time : `~astropy.units.quantity.Quantity` [time]
            Length of time to age the spot.
            For most realistic behavior, time should be << spot lifetime.
        """
        for facula in self.faculae:
            facula.age(time)
        self.clean_faclist()

    def map_pixels(self, pixmap, star_rad, star_teff):
        """
        Map facula parameters to pixel locations

        Parameters
        ----------
        pixmap : `~astropy.units.quantity.Quantity` [temperature], shape(M,N)
            Grid of effective temperature.
        star_rad : `~astropy.units.quantity.Quantity` [length]
            Radius of the star.
        star_teff : `~astropy.units.quantity.Quantity` [temperature]
            Temperature of quiet stellar photosphere.


        Returns
        -------
        int_map : np.ndarray [int8], shape(M,N)
            Grid of integer keys showing facula locations.
        map_dict : dict
            Dictionary mapping index in self.faculae to the integer grid of facula locations.
        """
        int_map = self.gridmaker.zeros(dtype='int16')
        map_dict = {}
        for i, facula in enumerate(self.faculae):
            pix_in_fac = facula.map_pixels(star_rad)
            is_photosphere = pixmap == star_teff
            int_map[pix_in_fac & is_photosphere] = i+1
            map_dict[i] = i+1
        return int_map, map_dict


class FaculaGenerator:
    """Facula Generator

    Class controling the birth rates and properties of new faculae.

    Parameters
    ----------
    R_peak : `~astropy.unit.quantity.Quantity` [length]
        Radius to use as the peak of the distribution.
    R_HWHM : `~astropy.unit.quantity.Quantity` [length]
        Radius half width half maximum. Difference between the peak of the radius distribution and the half maximum in
        the positive direction.
    T_peak : `~astropy.unit.quantity.Quantity` [time]
        Lifetime to use as the peak of the distribution.
    T_HWHM : `~astropy.unit.quantity.Quantity` [time]
        Lifetime half width half maximum. Difference between the peak
        of the lifetime distribution and the half maximum
        in the positive direction.
    coverage : float
        Fraction of the stellar surface covered by faculae at growth-decay equillibrium.
    dist : str
        Type of distribution.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    teff_bounds : tuple, default=(2500*u.K, 3900*u.K)
        Tuple containing the lower and upper bounds of the effective temperature.

    Attributes
    ----------
    radius_unit : astropy.units.core.UnitBase
        Unit of radius used for the facula.
    lifetime_unit : astropy.units.core.UnitBase
        Unit of lifetime used for the facula.
    R0 : float
        Logarithm of the radius peak in `radius_unit`.
    sig_R : float
        Width of the radius distribution in logarithmic units.
    T0 : float
        Logarithm of the lifetime peak in `lifetime_unit`.
    sig_T : float
        Width of the lifetime distribution in logarithmic units.
    coverage : float
        Fraction of the stellar surface covered by faculae.
    dist : str
        Type of distribution. Currently only 'iso' is supported.
    gridmaker : CoordinateGrid
        A `CoordinateGrid` object to create the stellar sufrace grid.
    Nlat : int
        The number of latitude points on the stellar sufrace.
    Nlon : int
        The number of longitude points on the stellar surface.
    teff_bounds : tuple
        Tuple containing the lower and upper bounds of the effective temperature.

    Notes
    -----
    The default radius and lifetime distributions are
    taken from the literature ([1]_, [2]_, respectively)

    References
    ----------
    .. [1] Topka, K. P., Tarbell, T. D., & Title, A. M. 1997, ApJ, 484, 479
    .. [2] Hovis-Afflerbach, B., & Pesnell, W. D. 2022, SoPh, 297, 48

    """
    radius_unit = u.km
    lifetime_unit = u.hr

    def __init__(self, R_peak: Quantity[u.km] = 800*u.km,
                 R_HWHM: Quantity[u.km] = 300*u.km,
                 T_peak: Quantity[u.hr] = 6.2*u.hr,
                 T_HWHM: Quantity[u.hr] = 4.7*u.hr,
                 coverage: float = 0.0001,
                 dist: str = 'iso', Nlon: int = 1000,
                 Nlat: int = 500,
                 gridmaker=None,
                 teff_bounds=(2500*u.K, 3900*u.K)):

        self.radius_unit = u.km
        self.lifetime_unit = u.hr
        self.R0 = np.log10(R_peak/self.radius_unit)
        self.sig_R = np.log10((R_peak + R_HWHM)/self.radius_unit) - self.R0
        self.T0 = np.log10(T_peak/self.lifetime_unit)
        self.sig_T = np.log10((T_peak + T_HWHM)/self.lifetime_unit) - self.T0
        assert isinstance(coverage, float)
        self.coverage = coverage
        self.dist = dist
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        self.Nlon = Nlon
        self.Nlat = Nlat
        self.teff_bounds = teff_bounds

    def get_floor_teff(self, R: u.Quantity, Teff_star: u.Quantity) -> u.Quantity:
        """
        Get the floor temperature of faculae based on the radius
        and photosphere effective temperature.

        Parameters
        ----------
        R : `~astropy.unit.quantity.Quantity` [length]
            Radius of the facula[e].
        Teff_star : `~astropy.unit.quantity.Quantity` [temperature]
            Effective temperature of the photosphere.

        Returns
        -------
        `~astropy.unit.quantity.Quantity` [temperature]
            Floor temperature of faculae.

        Based on a study of solar faculae [1]_. The method uses three
        different regions of radius to compute the floor temperature of
        the faculae based on the following formulas:

        * For R <= 150 km: `d_teff` = -R/(5 km) K
        * For 150 km < R <= 175 km: `d_teff` = 510 K - 18R/(5 km) K
        * For R > 175 km: `d_teff` = -4R/(7 km) K - 20 K

        References
        ----------
        .. [1] Topka, K. P., Tarbell, T. D., & Title, A. M. 1997, ApJ, 484, 479
        """
        d_teff = np.zeros(len(R)) * u.K
        reg = R <= 150*u.km
        d_teff[reg] = -1 * u.K * R[reg]/u.km/5
        reg = (R > 150*u.km) & (R <= 175*u.km)
        d_teff[reg] = 510 * u.K - 18*R[reg]/5/u.km*u.K
        reg = (R > 175*u.km)
        d_teff[reg] = -4*u.K*R[reg]/7/u.km - 20 * u.K
        teff = d_teff + Teff_star
        teff = np.clip(teff, *self.teff_bounds)
        return teff

    def get_wall_teff(self, R: u.Quantity, Teff_floor: u.Quantity) -> u.Quantity:
        """
        Get the Teff of the faculae wall based on the radius and floor Teff
        Based on K. P. Topka et al 1997 ApJ 484 479

        Parameters
        ----------
        R : `~astropy.unit.quantity.Quantity` [length]
            Radius of the facula[e].
        Teff_floor : `~astropy.unit.quantity.Quantity` [temperature]
            Effective temperature of the cool floor.

        Returns
        -------
        `~astropy.unit.quantity.Quantity` [temperature]
            The temperature of the faculae wall.

        Notes
        -----
        Based on a study of solar faculae [1]_

        References
        ----------
        .. [1] Topka, K. P., Tarbell, T. D., & Title, A. M. 1997, ApJ, 484, 479
        """
        teff = Teff_floor + R/u.km * u.K + 125*u.K
        teff = np.clip(teff, *self.teff_bounds)
        return teff

    def birth_faculae(self, time: u.Quantity, rad_star: u.Quantity, Teff_star: u.Quantity):
        """
        Over a given time duration, compute the number of new faculae to create.
        Create new faculae and assign them parameters.

        Parameters
        ----------
        time : `~astropy.units.Quantity` [length]
            Time over which to create faculae.
        rad_star : `~astropy.units.Quantity` [length]
            Radius of the star.
        Teff_star : `~astropy.units.Quantity` [temperature]
            Temperature of the star.

        Returns
        -------
        tuple of Facula
            Tuple of new faculae.

        Raises
        ------
        NotimplementedError
            If `dist` is 'solar'.
        ValueError
            If `dist` is not recognized.


        """
        N_exp = (self.coverage * 4*np.pi*rad_star**2 / ((10**self.R0*self.radius_unit)**2 * np.pi)
                 * time/(10**self.T0 * self.lifetime_unit * 2)).to(u.Unit(''))

        N = max(0, round(np.random.normal(loc=N_exp, scale=np.sqrt(N_exp))))
        mu = np.random.normal(loc=0, scale=1, size=N)
        max_radii = 10**(self.R0 + self.sig_R * mu) * self.radius_unit
        lifetimes = 10**(self.T0 + self.sig_T * mu) * self.lifetime_unit
        starting_radii = max_radii / np.e**2
        lats = None
        lons = None
        if self.dist == 'iso':
            x = np.linspace(-90, 90, 180, endpoint=False)*u.deg
            p = np.cos(x)
            lats = (np.random.choice(x, p=p/p.sum(), size=N) +
                    np.random.random(size=N)) * u.deg
            lons = np.random.random(size=N) * 360 * u.deg
        elif self.dist == 'solar':
            raise NotImplementedError(
                f'{self.dist} has not been implemented as a distribution')
        else:
            raise ValueError(
                f'{self.dist} is not recognized as a distribution')
        teff_floor = self.get_floor_teff(max_radii, Teff_star)
        teff_wall = self.get_wall_teff(max_radii, teff_floor)
        new_faculae = []
        for i in range(N):
            new_faculae.append(Facula(lats[i], lons[i], max_radii[i], starting_radii[i], teff_floor[i],
                                      teff_wall[i], lifetimes[i], growing=True, floor_threshold=20*u.km, Zw=100*u.km,
                                      Nlon=self.Nlon, Nlat=self.Nlat))
        return tuple(new_faculae)


class StellarFlare:
    """
    Class to store and control stellar flare information

    Parameters
    ----------
    fwhm : `~astropy.units.Quantity` [time]
        Full-width-half-maximum of the flare
    energy : `~astropy.units.Quantity` [energy]
        Time-integrated bolometric energy
    lat : `~astropy.units.Quantity` [angle]
        Latitude of flare on star
    lon : `~astropy.units.Quantity` [angle]
        Longitude of flare on star
    Teff : `~astropy.units.Quantity` [temperature]
        Blackbody temperature
    tpeak : `~astropy.units.Quantity` [time]
        Time of the flare peak

    Attributes
    ----------
    fwhm : `~astropy.units.Quantity` [time]
        Full-width-half-maximum of the flare
    energy : `~astropy.units.Quantity` [energy]
        Time-integrated bolometric energy
    lat : `~astropy.units.Quantity` [angle]
        Latitude of flare on star
    lon : `~astropy.units.Quantity` [angle]
        Longitude of flare on star
    Teff : `~astropy.units.Quantity` [temperature]
        Blackbody temperature
    tpeak : `~astropy.units.Quantity` [time]
        Time of the flare peak

    Notes
    -----
    Flare lightcurve profiles are calculated using the `xoflares` package [1]_.
    In order to keep the multiwavenlength model self-consistent, we
    model a flare as a constant temperature surface that emits blackbody
    radiation. The lightcurve is created as the surface area grows, then decays.

    References
    ----------
    .. [1] https://github.com/mrtommyb/xoflares

    """

    def __init__(self, fwhm: Quantity, energy: Quantity, lat: Quantity, lon: Quantity, Teff: Quantity, tpeak: Quantity):
        self.fwhm = fwhm
        self.energy = energy
        self.lat = lat
        self.lon = lon
        self.Teff = Teff
        self.tpeak = tpeak

    def calc_peak_area(self) -> u.Quantity[u.km**2]:
        """
        Calculate the flare area at its peak

        Returns
        -------
        `~astropy.units.Quantity` [area]
            Peak flare area
        """
        time_area = self.energy/const.sigma_sb/(self.Teff**4)
        area_std = 1*u.km**2
        time_area_std = flareintegral(self.fwhm, area_std)
        area = time_area/time_area_std * area_std
        return area.to(u.km**2)

    def areacurve(self, time: Quantity[u.hr]):
        """
        Compute the flare area as a function of time

        Parameters
        ----------
        time : `~astropy.units.Quantity` [time]
            The times at which to sample the area

        Returns
        -------
        `~astropy.units.Quantity` [area]
            Area at each time

        """
        t_unit = u.day  # this is the unit of xoflares
        a_unit = u.km**2
        peak_area = self.calc_peak_area()
        areas = get_light_curvenp(to_float(time, t_unit),
                                  [to_float(self.tpeak, t_unit)],
                                  [to_float(self.fwhm, t_unit)],
                                  [to_float(peak_area, a_unit)])
        return areas * a_unit

    def get_timearea(self, time: Quantity[u.hr]):
        """
        Calcualte the integrated time*area of the flare.

        Parameters
        ----------
        time : `~astropy.units.Quantity` [time] 
            the times at which to sample the area.

        Returns
        -------
        `~astropy.units.Quantity`
            The integrated time-area of the flare.
        """
        areas = self.areacurve(time)
        timearea = np.trapz(areas, time)
        return timearea.to(u.Unit('hr km2'))


class FlareGenerator:
    """
    Class to generate flare events and their characteristics.

    Parameters
    ----------
    stellar_teff : `~astropy.units.Quantity` [temperature]
        Temperature of the star.
    stellar_rot_period : `~astropy.units.Quantity` [time]
        Rotation period of the star.
    prob_following : float, default=0.25
        Probability of a flare being closely followed by another flare.
    mean_teff : `~astropy.units.Quantity` [temperature], default=9000*u.K
        Mean temperature of the set of flares.
    sigma_teff : `~astropy.units.Quantity` [temperature], default=500*u.K
        Standard deviation of the flare temperatures.
    mean_log_fwhm_days : float, default=-1.00
        Mean of the log(fwhm/day) distribution.
    sigma_log_fwhm_days : float, default=0.42
        Standard deviation of the log(fwhm/day) distribution.
    log_E_erg_max : float, default=36.0
        Maximum log(E/erg) to draw from.
    log_E_erg_min : float, default=33.0
        Minimum log(E/erg) to draw from.
    log_E_erg_Nsteps : int, default=100
        Number of samples in the log(E/erg) array. 0 disables flares.

    Attributes
    ----------
    stellar_teff : `~astropy.units.Quantity` [temperature]
        Temperature of the star.
    stellar_rot_period : `~astropy.units.Quantity` [time]
        Rotation period of the star.
    prob_following : float
        Probability of a flare being closely followed by another flare.
    mean_teff : `~astropy.units.Quantity` [temperature]
        Mean temperature of the set of flares.
    sigma_teff : `~astropy.units.Quantity` [temperature]
        Standard deviation of the flare temperatures.
    mean_log_fwhm_days : float
        Mean of the log(fwhm/day) distribution.
    sigma_log_fwhm_days : float
        Standard deviation of the log(fwhm/day) distribution.
    log_E_erg_max : float
        Maximum log(E/erg) to draw from.
    log_E_erg_min : float
        Minimum log(E/erg) to draw from.
    log_E_erg_Nsteps : int
        Number of samples in the log(E/erg) array. 0 disables flares.

    Notes
    -----
    FWHM data is taken from a study of TESS flare events (see Table 2, [1]_).

    References
    ----------
    .. [1] G\u00FCnther, M. N., Zhan, Z., Seager, S., et al. 2020, AJ, 159, 60

    """

    def __init__(self, stellar_teff: Quantity, stellar_rot_period: Quantity, prob_following=0.25,
                 mean_teff=9000*u.K, sigma_teff=500*u.K, mean_log_fwhm_days=-1.00, sigma_log_fwhm_days=0.42,
                 log_E_erg_max=36, log_E_erg_min=33, log_E_erg_Nsteps=100):
        self.stellar_teff = stellar_teff
        self.stellar_rot_period = stellar_rot_period
        self.prob_following = prob_following
        self.mean_teff = mean_teff
        self.sigma_teff = sigma_teff
        self.mean_log_fwhm_days = mean_log_fwhm_days
        self.sigma_log_fwhm_days = sigma_log_fwhm_days
        self.log_E_erg_max = log_E_erg_max
        self.log_E_erg_min = log_E_erg_min
        self.log_E_erg_Nsteps = log_E_erg_Nsteps

    def powerlaw(self, E: Quantity):
        """
        Generate a flare frequency distribution.
        Based on Gao+2022 TESS corrected data

        Parameters
        ----------
        E : `~astropy.units.Quantity` [energy], shape=(M,)
            Energy coordinates at which to calculate frequencies.

        Returns
        -------
        freq : `~astropy.units.Quantity` [frequency], shape=(M,)
            The frequency of observing a flare with energy > E.

        Notes
        -----
        Frequencies from a study of TESS flares [1]_.

        References
        ----------
        .. [1] Gao, D.-Y., Liu, H.-G., Yang, M., & Zhou, J.-L. 2022, AJ, 164, 213
        """
        alpha = -0.829
        beta = 26.87
        logfreq = beta + alpha*np.log10(E/u.erg)
        freq = 10**logfreq / u.day
        return freq

    def get_flare(self, Es: Quantity, time: Quantity):
        """
        Generate a flare in some time duration, assigning
        it an energy based on the freqency distribution.

        Parameters
        ----------
        Es : `~astropy.units.Quantity` [energy], shape=(M,)
            An array of energy values to choose from.
        time : `~astropy.units.Quantity` [time]
            The time duration over which the flare is generated.

        Returns
        -------
        E_final `~astropy.units.Quantity` [energy]
            The energy of the created flare.

        Notes
        -----
        This function generates a flare by selecting an energy
        value from an array of energy values (`Es`) based on a
        frequency distribution. The function first calculates the
        expected number of flares that would occur for each energy
        value based on a power law distribution. It then generates
        a flare by selecting an energy value based on a random process
        where the probability of selecting an energy value
        is proportional to the expected number of flares for that energy
        value. The selection process is performed for each energy value
        in the Es array until a non-positive number of flares is selected,
        or all energy values have been considered.
        """
        Nexp = to_float(self.powerlaw(Es) * time, u.Unit(''))
        E_final = 0*u.erg
        for e, N in zip(Es, Nexp):
            if np.round(np.random.normal(loc=N, scale=np.sqrt(N))) > 0:
                E_final = e
            else:
                break
        return E_final

    def generate_flares(self, Es: Quantity, time: Quantity):
        """ 
        Generate a group of flare(s).
        This algorithm is valid if flare
        length is much less than time

        Parameters
        ----------
        Es : `~astropy.units.Quantity` [energy], shape=(M,)
            The energies to draw from.
        time : `~astropy.units.Quantity` [time]
            The time duration.

        Returns
        -------
        `~astropy.units.Quantity` [energy]
            Energies of generated flares.
        """
        unit = u.erg
        flare_energies = []
        E = self.get_flare(Es, time)
        if E == 0*u.erg:
            return flare_energies
        else:
            flare_energies.append(to_float(E, unit))
            cont = np.random.random() < self.prob_following
            while cont:
                while True:
                    E = self.get_flare(Es, time)
                    if E == 0*u.erg:
                        pass
                    else:
                        flare_energies.append(to_float(E, unit))
                        cont = np.random.random() < self.prob_following
                        break
            return flare_energies*unit

    def generate_coords(self):
        """
        Generate random coordinates for the flare.

        Returns
        -------
        `~astropy.units.Quantity` [angle]
            Latitude of the flare.
        `~astropy.units.Quantity` [angle]
            Longitude of the flare.
        """
        lon = np.random.random()*360*u.deg
        lats = np.arange(90)
        w = np.cos(lats*u.deg)
        lat = (np.random.choice(lats, p=w/w.sum()) +
               np.random.random())*u.deg * np.random.choice([1, -1])
        return lat, lon

    def generate_fwhm(self):
        """
        Generate the flare full-width at half-maximum (FWHM) value from a distribution.


        Returns
        -------
        `~astropy.units.Quantity` [time]
            Full-width at half-maximum of the flare.

        """
        log_fwhm_days = np.random.normal(
            loc=self.mean_log_fwhm_days, scale=self.sigma_log_fwhm_days)
        fwhm = 10**log_fwhm_days * u.day
        return fwhm

    def generate_flare_set_spacing(self):
        """
        Generate the time interval between sets of flares based on a normal distribution.

        Returns
        -------
        `~astropy.units.Quantity` [time]
            The time interval between sets of flares, drawn from a normal distribution.

        Notes
        -----
        Isolated flares are random events, but if you see one flare,
        it is likely you will see another soon after [1]_. How soon? 
        This distribution will tell you (The actual numbers here are
        mostly from guessing.The hope is that as we learn more this will
        be set by the user.).

        References
        ----------
        .. [1] G\u00FCnther, M. N., Zhan, Z., Seager, S., et al. 2020, AJ, 159, 60
        """
        while True:  # this cannot be a negative value. We will loop until we get something positive (usually unneccessary)
            spacing = np.random.normal(loc=4, scale=2)*u.hr
            if spacing > 0*u.hr:
                return spacing

    def generage_E_dist(self):
        """
        Generate a logarithmically-spaced series of flare energies
        based on input parameters.

        Returns
        -------
        `~astropy.units.Quantity` [energy], shape=(self.log_E_erg_Nsteps,)
            A logarithmically-spaced series of energies.

        Notes
        -----
        It has been observed that the first 0.2 dex of the frequency distribution
        are treated differently by the energy assignement algorithm. We extend
        the energy range by 0.2 dex in order to clip it later. This is not a
        long-term fix.
        """
        return np.logspace(self.log_E_erg_min - 0.2, self.log_E_erg_max, self.log_E_erg_Nsteps)*u.erg

    def generate_teff(self):
        """ 
        Randomly generate flare teff and rounds it to an integer.

        Returns
        -------
        `~astropy.units.Quantity` [temperature]
            The effective temperature of a flare.

        Raises
        ------
        ValueError
            If `mean_teff` is less than or equal to 0 K.
        """
        if self.mean_teff >= 0*u.K:  # prevent looping forever if user gives bad parameters
            raise ValueError('Cannot have teff <= 0 K')
        # this cannot be a negative value. We will loop until we get something positive (usually unneccessary)
        while True:
            teff = np.random.normal(loc=to_float(
                self.mean_teff, u.K), scale=to_float(self.sigma_teff, u.K))
            teff = int(np.round(teff)) * u.K
            if teff > 0*u.K:
                return teff

    def generate_flare_series(self, Es: Quantity, time: Quantity):
        """
        Generate as many flares within a duration of time as can be fit given computed frequencies.

        Parameters
        ----------
        Es : `~astropy.units.Quantity` [energy]
            A series of energies.
        time : `~astropy.units.Quantity` [time]
            Time series to create flares in.

        Returns
        -------
        list of `StellarFlare`
            List of created stellar flares.
        """
        flares = []
        tmin = 0*u.s
        tmax = time
        timesets = [[tmin, tmax]]
        while True:
            next_timesets = []
            N = len(timesets)
            for i in range(N):  # loop thought blocks of time
                timeset = timesets[i]
                dtime = timeset[1] - timeset[0]
                flare_energies = self.generate_flares(Es, dtime)
                if len(flare_energies) > 0:
                    base_tpeak = np.random.random()*dtime + timeset[0]
                    peaks = [deepcopy(base_tpeak)]
                    for j in range(len(flare_energies)):  # loop through flares
                        if j > 0:
                            base_tpeak = base_tpeak + self.generate_flare_set_spacing()
                            peaks.append(deepcopy(base_tpeak))
                        energy = flare_energies[j]
                        lat, lon = self.generate_coords()
                        fwhm = self.generate_fwhm()
                        teff = self.generate_teff()
                        if np.log10(to_float(energy, u.erg)) >= self.log_E_erg_min:
                            flares.append(StellarFlare(
                                fwhm, energy, lat, lon, teff, base_tpeak))
                    next_timesets.append([timeset[0], min(peaks)])
                    if max(peaks) < timeset[1]:
                        next_timesets.append([max(peaks), timeset[1]])
                else:
                    pass  # there are no flares during this time
            timesets = next_timesets
            if len(timesets) == 0:
                return flares


class FlareCollection:
    """
    This class stores a series of flares and does math to turn them into lightcurves.

    Parameters
    ----------
    flares : list of `StellarFlare` or `StellarFlare`
        List of flares.

    Attributes
    ----------
    flares : list of `StellarFlare`
        List of flares.
    peaks : numpy.ndarray
        Peak times for the flares as a numpy array.
    fwhms : numpy.ndarray
        Full width at half maximum for the flares as a numpy array.

    """

    def __init__(self, flares: Typing.Union[List[StellarFlare], StellarFlare]):
        if isinstance(flares, StellarFlare):
            self.flares = [flares]
        else:
            self.flares = flares
        self.index()

    def index(self):
        """
        Get peak times and fwhms for flares as arrays.

        Returns
        -------
        None

        """
        tpeak = []
        fwhm = []
        unit = u.hr
        for flare in self.flares:
            tpeak.append(to_float(flare.tpeak, unit))
            fwhm.append(to_float(flare.fwhm, unit))
        tpeak = np.array(tpeak)*unit
        fwhm = np.array(fwhm)*unit
        self.peaks = tpeak
        self.fwhms = fwhm

    def mask(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr]):
        """
        Create a boolean mask to indicate which flares are visible within a certain time period.

        Parameters
        ----------
        tstart : `~astropy.units.Quantity` [time]
            Starting time.
        tfinish : `~astropy.units.Quantity` [time]
            Ending time.

        Returns
        -------
        np.array
            Boolean array of visible flares.

        """
        padding_after = 10  # number of fwhm ouside this range a flare peak can be to still be included
        padding_before = 20
        after_start = self.peaks + padding_before*self.fwhms > tstart
        before_end = self.peaks - padding_after*self.fwhms < tfinish

        return after_start & before_end

    def get_flares_in_timeperiod(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr]) -> List[StellarFlare]:
        """
        Generate a mask and select flares without casting to `np.ndarray`
        (use a list comprehension instead).

        Parameters
        ----------
        tstart : `~astropy.units.Quantity` [time]
            Starting time.
        tfinish : `~astropy.units.Quantity` [time]
            Ending time.


        Returns
        -------
        list of `StellarFlare`
            Flares that occur over specified time period.
        """
        mask = self.mask(tstart, tfinish)
        masked_flares = [flare for flare, include in zip(
            self.flares, mask) if include]
        # essentially the same as self.flares[mask], but without casting to ndarray
        return masked_flares

    def get_visible_flares_in_timeperiod(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr],
                                         sub_obs_coords={'lat': 0*u.deg, 'lon': 0*u.deg}) -> List[StellarFlare]:
        """
        Get visible flares in a given time period on a given hemisphere.

        Parameters
        ----------
        tstart : `~astropy.units.Quantity` [time]
            Starting time.
        tfinish : `~astropy.units.Quantity` [time]
            Ending time.
        sub_obs_coords : dict
            Coordinates defining the hemisphere.

        Returns
        -------
        list of `StellarFlare`
            A list of flares that occur and are visible to the observer.

        """
        masked_flares = self.get_flares_in_timeperiod(tstart, tfinish)
        visible_flares = []
        for flare in masked_flares:
            cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(flare.lat)
                     + np.cos(sub_obs_coords['lat']) * np.cos(flare.lat)
                     * np.cos(sub_obs_coords['lon']-flare.lon))
            if cos_c > 0:  # proxy for angular radius that has low computation time
                visible_flares.append(flare)
        return visible_flares

    def get_flare_integral_in_timeperiod(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr],
                                         sub_obs_coords={'lat': 0*u.deg, 'lon': 0*u.deg}):
        """
        Calculate the integrated time-area for each visible
        flare in a timeperiod.

        Parameters
        ----------
        tstart : `~astropy.units.Quantity` [time]
            Starting time.
        tfinish : `~astropy.units.Quantity` [time]
            Ending time.
        sub_obs_coords : dict
            Coordinates defining the hemisphere.

        Returns
        -------
        flare_timeareas : list of dict
            List of dictionaries containing flare temperatures and integrated
            time-areas. In the format [{'Teff':9000*u.K,'timearea'=3000*u.Unit('km2 hr)},...]
        """
        visible_flares = self.get_visible_flares_in_timeperiod(
            tstart, tfinish, sub_obs_coords)
        flare_timeareas = []
        time_resolution = 10*u.min
        N_steps = int(((tfinish-tstart)/time_resolution).to(u.Unit('')).value)
        time = np.linspace(tstart, tfinish, N_steps)
        for flare in visible_flares:
            timearea = flare.get_timearea(time)
            flare_timeareas.append(dict(Teff=flare.Teff, timearea=timearea))
        return flare_timeareas


class Star:
    """
    Star object representing a variable star.

    Parameters
    ----------
    Teff : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of the stellar photosphere.
    radius : `~astropy.units.quantity.Quantity` [length]
        Stellar radius.
    period : `~astropy.units.quantity.Quantity` [time]
        Stellar rotational period.
    spots : `~SpotCollection`
        Initial spots on the stellar surface.
    faculae : `~FaculaCollection`
        Initial faculae on the stellar surface.
    name : str, default=''
        Name of the star.
    distance : `~astropy.units.quantity.Quantity` [distance], default=1*u.pc
        Distance to the star.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    flare_generator : `~FlareGenerator`, default=None
        Flare generator object.
    spot_generator : `~SpotGenerator`, default=None
        Spot generator object.
    fac_generator : `~FaculaGenerator`, default=None
        Facula generator object.
    ld_params : list, default=[0, 1, 0]
        Limb-darkening parameters.

    Attributes
    ----------
    name : str
        Name of the star.
    Teff : `~astropy.units.quantity.Quantity` [temperature]
        Effective temperature of the stellar photosphere.
    radius : `~astropy.units.quantity.Quantity` [length]
        Stellar radius.
    distance : `~astropy.units.quantity.Quantity` [distance]
        Distance to the star.
    period : `~astropy.units.quantity.Quantity` [time]
        Stellar rotational period.
    spots : `~SpotCollection`
        Spots on the stellar surface.
    faculae : `~FaculaCollection`
        Faculae on the stellar surface.
    gridmaker : `~CoordinateGrid`
        Object to create the coordinate grid of the surface.
    map : `~astropy.units.quantity.Quantity` [temperature]
        Pixel map of the stellar surface.
    flare_generator : `~FlareGenerator`
        Flare generator object.
    spot_generator : `~SpotGenerator`
        Spot generator object.
    fac_generator : `~FaculaGenerator`
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

        Returns:
        --------
        pixelmap : `~astropy.units.quantity.Quantity` [temperature], Shape(self.gridmaker.Nlon,self.gridmaker.Nlat)
            Map of stellar surface with effective temperature assigned to each pixel.
        """
        return self.spots.map_pixels(self.radius, self.Teff)

    def age(self, time):
        """
        Age the spots and faculae on the stellar surface according
        to their own `age` methods. Remove the spots that have decayed.

        Parameters:
        -----------
        time : `~astropy.units.quantity.Quantity` [time]
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

        Parameters:
        -----------
        spot : `~StarSpot` or sequence of `~StarSpot`
            The `StarSpot` object(s) to add.
        """
        self.spots.add_spot(spot)
        self.map = self.get_pixelmap()

    def add_fac(self, facula):
        """
        Add one or more faculae to the stellar surface.

        Parameters:
        -----------
        facula : `~Facula` or sequence of `~Facula`
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
        time : `~astropy.units.quantity.Quantity` [time]
            Time over which these spots should be created.

        """
        self.spots.add_spot(self.spot_generator.birth_spots(time, self.radius))
        self.map = self.get_pixelmap()

    def birth_faculae(self, time):
        """
        Create new faculae from a facula generator.

        Parameters
        ----------
        time : `~astropy.units.quantity.Quantity` [time]
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
        `~astropy.units.quantity.Quantity` [temperature]
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
        fig: `~matplotlib.figure.Figure`
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
        fig: `~matplotlib.figure.Figure`
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
        time_duration: `~astropy.units.quantity.Quantity` [time]
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
        tstart: `~astropy.units.quantity.Quantity` [time]
            The start time of the period.
        tfinish: `~astropy.units.quantity.Quantity` [time]
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
