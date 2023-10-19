"""VSPEC Spots module

This contains all the code governing
the behavior of spots.




"""
from typing import List
import typing as Typing

import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity

from VSPEC.helpers import CoordinateGrid
from VSPEC.config import MSH, starspot_initial_area
from VSPEC import config
from VSPEC.params import SpotParameters


class StarSpot:
    """
    A Star Spot

    Our star spot model is nearly entirely based on
    observations of the sun. Sun spots can be resolved
    and are well-studied, whereas spots on other stars
    (e.g., M dwarfs) can only observed indirectly. We
    therefore designed our spot model to mimic sun spots
    but with parameterized values for spot temperature and 
    lifetime that can be matched to observations of other 
    stellar types and ages.

    Parameters
    ----------
    lat : astropy.units.Quantity 
        Latitude of spot center. North is positive.
    lon : astropy.units.Quantity 
        Longitude of spot center. East is positive.
    Amax : astropy.units.Quantity 
        The maximum area a spot reaches before it decays.
    A0 : astropy.units.Quantity 
        The current spot area.
    Teff_umbra : astropy.units.Quantity 
        The effective temperature of the spot umbra.
    Teff_penumbra : astropy.units.Quantity 
        The effective temperature of spot penumbra.
    r_A : float
        The ratio of total spot area to umbra area. 5+/-1 according to :cite:t:`2003A&ARv..11..153S`.
    growing : bool
        Whether or not the spot is growing.
    growth_rate : astropy.units.Quantity 
        Fractional growth of the spot for a given unit time.
        From from sunspot literature, can be 0.52/day to 1.83/day :cite:p:`2003A&ARv..11..153S`.
        According to M dwarf literature, can effectively be 0 :cite:p:`2015ApJ...806..212D`.
    decay_rate : astropy.units.Quantity
        The rate at which a spot linearly decays. From sunspot
        literature, this is 10.89 MSH/day :cite:p:`2003A&ARv..11..153S`. According to M dwarf
        literature, this can be 0 :cite:p:`2015ApJ...806..212D`.
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
    radius
    coords : dict
        A dictionary containing the latitude and longitude of the spot's center.
    area_max : astropy.units.Quantity
        The maximum area a spot reaches before it decays.
    area_current : astropy.units.Quantity
        The current area of the spot.
    Teff_umbra : astropy.units.Quantity
        The effective temperature of the spot umbra.
    Teff_penumbra : astropy.units.Quantity
        The effective temperature of the spot penumbra.
    decay_rate : astropy.units.Quantity
        The rate at which a spot linearly decays.
    total_area_over_umbra_area : float
        The ratio of total spot area to umbra area. 5+/-1 according
        to :cite:t:`2003A&ARv..11..153S`.
    is_growing : bool
        Whether or not the spot is growing.
    growth_rate : astropy.units.Quantity
        Fractional growth of the spot for a given unit time.
    gridmaker : CoordinateGrid or None
        A `CoordinateGrid` object used to produce points on the stellar surface. If None,
        a `CoordinateGrid` object is created with default parameters.
    r : np.ndarray
        An array of points on the stellar surface with their pre-computed
        distance from the center of the spot.

    References
    ----------
    :cite:t:`2003A&ARv..11..153S`
    :cite:t:`2015ApJ...806..212D`
    """

    def __init__(
        self, lat: Quantity, lon: Quantity, Amax: Quantity, A0: Quantity,
        Teff_umbra: Quantity, Teff_penumbra: Quantity, r_A: float = 5, growing: bool = True,
        growth_rate: Quantity = 0.52/u.day, decay_rate: Quantity = 10.89 * MSH/u.day,
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
            self.set_gridmaker(CoordinateGrid(Nlat, Nlon))
        else:
            self.set_gridmaker(gridmaker)

    def set_gridmaker(self, gridmaker: CoordinateGrid):
        """
        Set the `gridmaker` attribute safely.

        Parameters
        ----------
        gridmaker : VSPEC.helpers.CoordinateGrid
            The `CoordinateGrid` object to set
        """
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

    @property
    def radius(self) -> Quantity:
        """
        Radius

        Get the radius of the spot.

        Returns
        -------
        astropy.units.Quantity
            The radius of the spot.
        """
        return np.sqrt(self.area_current/np.pi).to(u.km)

    def angular_radius(self, star_rad: Quantity) -> Quantity:
        """
        Angular radius

        Get the angular radius of the spot on the stellar surface.

        Parameters
        ----------
        star_rad : astropy.units.Quantity 
            The radius of the star.

        Returns
        -------
        astropy.units.Quantity 
            The angular radius of the spot.
        """
        cos_angle = 1 - self.area_current/(2*np.pi*star_rad**2)
        return (np.arccos(cos_angle)).to(u.deg)

    def map_pixels(self, star_rad: Quantity) -> dict:
        """
        Map latitude and longituide points continaing the umbra and penumbra

        Parameters
        ----------
        star_rad : astropy.units.Quantity 
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
                         star_rad: Quantity, N: int = 1001) -> float:
        """
        Determine the surface fraction covered by a spot from a given 
        angle of observation using the orthographic projection.

        Parameters
        ----------
        sub_obs_coords : dict
            Dictionary giving coordinates of the sub-observation point. This
            is the point that is at the center of the stellar disk from the
            view of an observer. Format: {'lat': lat, 'lon': lon} where lat and
            lon are astropy.units.Quantity objects.
        star_rad : astropy.units.Quantity 
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
        return (np.trapz(integrand, x=c)/(2*np.pi*u.steradian)).to_value(u.dimensionless_unscaled)

    def age(self, time: Quantity) -> None:
        """
        Age a spot according to its growth timescale and decay rate

        Parameters
        ----------
        time : astropy.units.Quantity
            Length of time to age the spot. For most realistic behavior,
            time should be << spot lifetime

        Notes
        -----
        This method updates the `area_current` attribute of the `StarSpot`
        object to simulate the growth and decay of a sunspot.

        .. math::
            A(t) = \\left\\{
                \\begin{array}{lr}
                    A_0 e^{(t-t_0)/\\tau}, & \\text{if } t \\leq t_0 \\\\
                    A_0 - W(t-t_0), & \\text{if } t > t_0
                \\end{array}
                \\right\\}


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

    def __init__(self, *spots: StarSpot, Nlat: int = 500, Nlon: int = 1000, gridmaker=None):
        self.spots = spots
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        for spot in self.spots:
            spot.set_gridmaker(self.gridmaker)

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
            spot = [spot]
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

    def map_pixels(self, star_rad: Quantity, star_teff: Quantity):
        """
        Map latitude and longitude points containing the umbra and penumbra
        of each spot. For pixels with coverage from multiple spots, assign
        the coolest Teff to that pixel.

        Parameters
        ----------
        latgrid : astropy.units.Quantity , shape(M,N)
            Grid of latitude points to map.
        longrid : astropy.units.Quantity , shape(M,N)
            Grid of longitude points to map.
        star_rad : astropy.units.Quantity
            Radius of the star.
        star_teff : astropy.units.Quantity
            Temperature of the star.

        Returns
        -------
        surface_map : array of astropy.units.Quantity , shape(M,N)
            Map of the stellar surface with Teff assigned to each pixel
        """
        surface_map = self.gridmaker.zeros()*star_teff.unit + star_teff
        for spot in self.spots:
            teff_dict = spot.map_pixels(star_rad)
            # penumbra
            penumbra = teff_dict[spot.Teff_penumbra] & (
                surface_map > spot.Teff_penumbra)
            # umbra
            umbra = teff_dict[spot.Teff_umbra] & (
                surface_map > spot.Teff_umbra)
            surface_map = np.where(umbra, spot.Teff_umbra, np.where(
                penumbra, spot.Teff_penumbra, surface_map))
        return surface_map

    def age(self, time: Quantity) -> None:
        """
        Age spots according to its growth timescale and decay rate.

        Remove spots that have decayed.

        Parameters
        ----------
        time : astropy.units.Quantity
            Length of time to age the spot. For most realistic
            behavior, time should be << spot lifetime.
        """
        for spot in self.spots:
            spot.age(time)
        self.clean_spotlist()

    def get_coverage(
        self,
        r_star: u.Quantity,
    ):
        """
        Get the fractional coverage of star spots.

        Parameters
        ----------
        r_star : astropy.units.Quantity
            The radius of the star.

        Returns
        -------
        coverage : float
            The fraction of the stellar surface covered by spots.
        """
        teff_star = np.inf*u.K
        lats2d, _ = self.gridmaker.grid()
        jacobian = np.cos(lats2d).value
        tmap = self.map_pixels(r_star, teff_star)
        is_spot = tmap != teff_star
        return np.sum(jacobian*is_spot)/np.sum(jacobian)


class SpotGenerator:
    """Spot Generator

    Class controling the birth rates and properties of new spots.
    This class is based on various studies, but since in general starspots cannot
    be resolved, lots of gaps are filled in with studies of sunspots.

    Parameters
    ----------
    average_area : astropy.units.Quantity 
        The average peak spot area.
    area_spread : float
        The standard deviation, in dex, of the lognormal peak spot area distribution
    umbra_teff : astropy.units.Quantity 
        Effective temperature of umbra.
    penumbra_teff : astropy.units.Quantity 
        Effective temperature of penumbra.
    growth_rate : astropy.units.Quantity , default=0.52/u.day
        The spot growth rate.
    decay_rate : astropy.units.Quantity [area/time], default = 10.89 * MSH/u.day
        The spot decay rate.
    starting_size : astropy.units.Quantity , default=10*MSH
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
    rng : numpy.random.generator, default = numpy.random.default_rng()
        The random number generator to use.

    Attributes
    ----------
    average_spot_area : astropy.units.Quantity
        The average peak spot area.
    spot_area_spread : float
        The standard deviation, in dex, of the lognormal peak spot area distribution.
    umbra_teff : astropy.units.Quantity
        Effective temperature of umbra.
    penumbra_teff : astropy.units.Quantity
        Effective temperature of penumbra.
    growth_rate : astropy.units.Quantity
        The spot growth rate.
    decay_rate : astropy.units.Quantity
        The spot decay rate.
    starting_size : astropy.units.Quantity
        The area of each spot at birth.
    distribution : str
        The spot distribution method. Choose from 'iso' or 'solar'.
    coverage : float
        The fractional coverage of the surface by spots in growth-decay equilibrium.
    gridmaker : CoordinateGrid or subclass
        A `CoordinateGrid` object to create the stellar surface grid.
    rng : numpy.random.generator
        The random number generator to use.
    is_static
    mean_lifetime
    mean_area

    Notes
    -----
    The `distribution` parameter can have values of 'iso' of 'solar'. The 'iso' keyword
    distributes spots evenly across the surface. The 'solar' keyword, on the other hand,
    distributes spots according to their know clustering around +/- 15 degrees solar
    latitude [1]_.

    References
    ----------
    .. [1] Mandal, S., Karak, B. B., & Banerjee, D. 2017, ApJ, 851, 70

    """

    def __init__(self,
                 dist_area_mean: Quantity,
                 dist_area_logsigma: float,
                 umbra_teff: Quantity,
                 penumbra_teff: Quantity,
                 growth_rate: Quantity = 0.52/u.day,
                 decay_rate: Quantity = 10.89 * MSH/u.day,
                 init_area: Quantity = 10*MSH,
                 distribution='solar',
                 coverage: float = 0.2,
                 Nlat: int = 500,
                 Nlon: int = 1000,
                 gridmaker=None,
                 rng: np.random.Generator = np.random.default_rng()
                 ):
        self.dist_area_mean = dist_area_mean
        self.dist_area_logsigma = dist_area_logsigma
        self.umbra_teff = umbra_teff
        self.penumbra_teff = penumbra_teff
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.init_area = init_area
        self.distribution = distribution
        self.coverage = coverage
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        self.rng = rng

    @classmethod
    def from_params(
        cls,
        spotparams: SpotParameters,
        nlat: int = config.nlat,
        nlon: int = config.nlon,
        gridmaker: CoordinateGrid = None,
        rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Construct a ``SpotGenerator`` object from a ``SpotParameters`` object.

        Parameters
        ----------
        spotparams : SpotParameters
            The parameters to build the instance from.
        nlat : int, default=VSPEC.config.nlat
            The number of latitude points. Default defined in `VSPEC.config`.
        nlon : int, default=VSPEC.config.nlon
            The number of longitude points. Default defined in `VSPEC.config`.
        gridmaker : CoordinateGrid, default=None
            The ``CoordianteGrid`` object to create the surface array.
        rng : numpy.random.Generator, default=np.random.default_rng()
            The random number generator to use.

        Notes
        -----
        ``init_area`` is set to ``VSPEC.config.starspot_initial_area``.
        """
        return cls(
            dist_area_mean=spotparams.area_mean,
            dist_area_logsigma=spotparams.area_logsigma,
            umbra_teff=spotparams.teff_umbra,
            penumbra_teff=spotparams.teff_penumbra,
            growth_rate=spotparams.growth_rate,
            decay_rate=spotparams.decay_rate,
            init_area=starspot_initial_area,
            distribution=spotparams.distribution,
            coverage=spotparams.equillibrium_coverage,
            Nlat=nlat,
            Nlon=nlon,
            gridmaker=gridmaker,
            rng=rng
        )

    @property
    def is_static(self) -> bool:
        """
        True if the spots do not decay.

        Returns
        -------
        is_static : bool
            True if the spots do not decay.
        """
        return self.decay_rate == 0*MSH/u.day

    @property
    def mean_lifetime(self) -> u.Quantity:
        """
        The mean lifetime of the spots.

        Returns
        -------
        mean_lifetime : astropy.units.Quantity
            The mean lifetime of spots.
        """
        if self.is_static:
            return np.inf * u.day
        else:
            return ((self.dist_area_mean/self.decay_rate)
                    - np.log(self.init_area/self.dist_area_mean)/self.growth_rate).to(u.day)

    @property
    def mean_area(self) -> u.Quantity:
        """
        The time-averaged area of a typical spot.

        Returns
        -------
        mean_area : astropy.units.Quantity
            The time-averaged area of a typical spot.
        """
        if self.is_static:
            return self.dist_area_mean
        else:
            Adt = (self.dist_area_mean / self.growth_rate) \
                + (self.init_area / self.growth_rate) \
                + (self.dist_area_mean**2/2/self.decay_rate)
            return Adt / self.mean_lifetime

    def get_coordinates(self, N: int):
        """
        Get coordinates for a `StarSpot` distribution.

        Parameters
        ----------
        N : int
            Number of spots to create.

        Returns
        -------
        lat : astropy.units.Quantity
            Latitude coordinates of the new spots
        lon : astropy.units.Quantity
            Longitude coordinates of the new spots

        Notes
        -----
        In order to draw latitude points for the isotropic case, we
        use inverse transform sampling to account for the jacobian.
        """
        if self.distribution == 'solar':
            # (dist approx from 2017ApJ...851...70M)
            hemi = self.rng.choice([-1, 1], size=N)
            lat = self.rng.normal(15, 5, size=N)*hemi*u.deg
            lon = self.rng.random(size=N)*360*u.deg
        elif self.distribution == 'iso':
            lon = self.rng.random(size=N)*360*u.deg
            # use inverse transform to generate lats
            X = self.rng.random(size=N)
            lat = np.arcsin(2*X - 1)/np.pi * 180*u.deg
        else:
            raise ValueError(
                f'Unknown value {self.distribution} for distribution')
        return lat, lon

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
        new_max_areas = self.rng.lognormal(mean=np.log(
            self.dist_area_mean/MSH), sigma=self.dist_area_logsigma, size=N)*MSH
        new_r_A = self.rng.normal(loc=5, scale=1, size=N)
        while np.any(new_r_A <= 0):
            new_r_A = self.rng.normal(loc=5, scale=1, size=N)
        lat, lon = self.get_coordinates(N)

        penumbra_teff = self.penumbra_teff
        umbra_teff = self.umbra_teff

        spots = []
        for i in range(N):
            spots.append(StarSpot(
                lat[i], lon[i], new_max_areas[i], self.init_area, umbra_teff, penumbra_teff,
                growth_rate=self.growth_rate, decay_rate=self.decay_rate,
                r_A=new_r_A[i], Nlat=self.gridmaker.Nlat, Nlon=self.gridmaker.Nlon, gridmaker=self.gridmaker
            ))
        return tuple(spots)

    def get_N_spots_to_birth(self, time: Quantity, rad_star: Quantity) -> float:
        """
        Calculate how many new `StarSpot` objects to birth over a given time duration (expectation value).

        Parameters
        ----------
        time : astropy.units.Quantity 
            Amount of time in which to birth spots.
            The total number of new spots will consider this time and the birthrate.
        rad_star : astropy.units.Quantity 
            The radius of the star.

        Returns
        -------
        N_exp : float
            Expected number of new `StarSpot` objects.
        """
        N_exp = (self.coverage * 4*np.pi*rad_star**2 / self.mean_area *
                 time / self.mean_lifetime).to_value(u.dimensionless_unscaled)
        return N_exp

    def birth_spots(self, time: Quantity, rad_star: Quantity) -> tuple[StarSpot]:
        """
        Generate new `StarSpot` objects to be birthed over a given time duration.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Amount of time in which to birth spots.
            The total number of new spots will consider this time and the birthrate.
        rad_star : astropy.units.Quantity 
            The radius of the star.

        Returns
        -------
        Tuple[StarSpot]
            New `StarSpot` objects.
        """
        N_exp = self.get_N_spots_to_birth(time, rad_star)
        # N_exp is the expectation value of N, but this is a poisson process
        # N = max(0, round(np.random.normal(loc=N_exp, scale=np.sqrt(N_exp))))
        N = self.rng.poisson(lam=N_exp)

        return self.generate_spots(N)

    def generate_mature_spots(self, coverage: float, R_star: Quantity) -> List[StarSpot]:
        """Generate mature StarSpot objects to cover a given fraction of the star's surface.

        This method generates mature spots such that the total solid angle subtended by the spots
        covers a specified fraction of the star's surface.

        Parameters
        ----------
        coverage : float
            The fraction of the star's surface to be covered by the spots.
        R_star : astropy.units.Quantity 
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
            if self.is_static:
                area0 = self.init_area
                area_range = new_spot.area_max - area0
                area = self.rng.random()*area_range + area0
                new_spot.area_current = area
            else:
                age = self.rng.random() * self.mean_lifetime
                new_spot.age(age)
            spots.append(new_spot)
            spot_solid_angle = new_spot.angular_radius(R_star)**2 * np.pi
            current_omega += spot_solid_angle
        return spots
