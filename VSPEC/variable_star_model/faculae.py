"""
Faculae are magnetically-generated regions of the solar
surface that usually appear as bright points near the limb;
we employ the ``hot wall'' model :cite:p:`1976SoPh...50..269S` where
faculae are described as three-dimensional pores in the stellar
surface with a hot, bright wall, and a cool, dark floor.

Their three-dimensional structure causes faculae to behave differently
depending on their angle from disk-center. Close to the limb, the hot wall
is visible to the observer, and faculae appear as bright points. Near the center,
however, the cool floor is exposed and faculae appear dark. To consider this effect
in the faculae lightcurve, we compute the fraction of the facula's normalized
area -- the area on the disk it would occupy as a flat spot -- that is occupied by
each the hot wall and cool floor. This is done via numerical integral along the radius of the spot.
"""
from typing import Dict, Tuple

import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity
import warnings

from VSPEC.helpers import CoordinateGrid, round_teff
from VSPEC import config
from VSPEC.params import FaculaParameters


class Facula:
    """
    A small magnetic depression with a cool floor and hot walls.

    Parameters
    ----------
    lat : astropy.units.Quantity
        Latitude of facula center
    lon : astropy.units.Quantity
        Longitude of facula center
    r_max : astropy.units.Quantity
        Maximum radius of facula
    r_init : astropy.units.Quantity
        Current radius of facula
    depth : astropy.units.Quantity
        Depth of the depression.
    floor_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    floor_teff_min_rad : astropy.units.Quantity
        The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    floor_teff_base_dteff : astropy.units.Quantity
        The Teff of the floor at the minimum radius.
    wall_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    wall_teff_intercept : astropy.units.Quantity
        The Teff of the wall when :math:`R = 0`.
    lifetime : astropy.units.Quantity
        Facula lifetime
    growing : bool, default=True
        Whether or not the facula is still growing.
    nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A ``CoordinateGrid`` object to create the stellar sufrace grid.

    Attributes
    ----------
    _r
    wall_dteff
    floor_dteff
    lat : astropy.units.Quantity
        Latitude of facula center.
    lon : astropy.units.Quantity
        Longitude of facula center.
    r_max : astropy.units.Quantity
        Maximum radius of facula.
    radius : astropy.units.Quantity
        Current radius of facula.
    depth : astropy.units.Quantity
        Depth of the depression.
    floor_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    floor_teff_min_rad : astropy.units.Quantity
        The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    floor_teff_base_dteff : astropy.units.Quantity
        The Teff of the floor at the minimum radius.
    wall_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    wall_teff_intercept : astropy.units.Quantity
        The Teff of the wall when :math:`R = 0`.
    lifetime : astropy.units.Quantity
        Facula lifetime.
    is_growing : bool
        Whether or not the facula is still growing.
    gridmaker : `CoordinateGrid` object
        A `CoordinateGrid` object to create the stellar sufrace grid.

    Notes
    -----
    The "Hot wall" model of solar facule describes them as a depression on the
    stellar surface with a hot wall and cool floor :cite:p:`1976SoPh...50..269S`. Because if this, faculae
    appear bright when they are near the limb (hot wall is visible) and dark when near
    the disk center (cool floor is visible).

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        lat: Quantity,
        lon: Quantity,
        r_max: Quantity,
        r_init: Quantity,
        depth: Quantity,
        lifetime: Quantity,
        floor_teff_slope: Quantity,
        floor_teff_min_rad: Quantity,
        floor_teff_base_dteff: Quantity,
        wall_teff_slope: Quantity,
        wall_teff_intercept: Quantity,
        growing: bool = True,
        nlat: int = 500,
        nlon: int = 1000,
        gridmaker=None
    ):
        self.lat = lat
        self.lon = lon
        self.r_max = r_max
        self.radius = r_init
        self.depth = depth
        self.floor_teff_slope = floor_teff_slope
        self.floor_teff_min_rad = floor_teff_min_rad
        self.floor_teff_base_dteff = floor_teff_base_dteff
        self.wall_teff_slope = wall_teff_slope
        self.wall_teff_intercept = wall_teff_intercept
        self.lifetime = lifetime
        self.is_growing = growing

        if gridmaker is None:
            self.set_gridmaker(CoordinateGrid(nlat, nlon))
        else:
            self.set_gridmaker(gridmaker)

    @property
    def floor_dteff(self) -> u.Quantity:
        """
        The effective temperature difference between the photosphere and cool floor.

        :type: astropy.units.Quantity
        """
        dteff = (self.radius - self.floor_teff_min_rad) * \
            self.floor_teff_slope + self.floor_teff_base_dteff
        return dteff.to(u.K)

    @property
    def wall_dteff(self) -> u.Quantity:
        """
        The effective temperature difference between the photosphere and hot wall

        :type: astropy.units.Quantity
        """
        dteff = self.wall_teff_slope*self.radius + self.wall_teff_intercept
        return dteff.to(u.K)

    @property
    def _r(self):
        """
        The angular radius to every point on the ``CoordinateGrid``.

        :type: astropy.units.Quantity
        """
        latgrid, longrid = self.gridmaker.grid()
        lat0 = self.lat
        lon0 = self.lon
        return 2 * np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
                                     + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))

    def set_gridmaker(self, gridmaker: CoordinateGrid):
        """
        .. deprecated:: 0.1
            Now that radius ``_r`` is a property is is not needed. The skeleton of this
            function is kept for compatibility.

        Set the `gridmaker` attribute safely.

        Parameters
        ----------
        gridmaker : VSPEC.helpers.CoordinateGrid
            The `CoordinateGrid` object to set
        """
        self.gridmaker = gridmaker
        msg = 'The `set_gridmaker` method of Facula is no longer necessary.'
        warnings.warn(msg, DeprecationWarning)

    def age(self, time: Quantity[u.day]):
        """
        Progress the development of the facula by an amount of time.

        Parameters
        ----------
        time : astropy.units.Quantity
            The amount of time to age facula.

        Notes
        -----
        This method calculates the new radius of the facula based on the amount of
        time elapsed since the last time it was updated. If the facula is still growing,
        it checks if it has reached the maximum radius and sets the `is_growing` attribute
        to False if so. If the facula is no longer growing, it shrinks over time.

        """
        if self.is_growing:
            T_from_max = -1*np.log(self.radius/self.r_max)*self.lifetime*0.5
            if T_from_max <= time:
                self.is_growing = False
                time = time - T_from_max
                self.radius = self.r_max * np.exp(-2*time/self.lifetime)
            else:
                self.radius = self.radius * np.exp(2*time/self.lifetime)
        else:
            self.radius = self.radius * np.exp(-2*time/self.lifetime)

    def effective_area(self, angle, N=201):
        """
        Calculate the effective area of the floor and walls when projected on a disk.

        Parameters
        ----------
        angle : astropy.units.Quantity 
            Angle from disk center.
        N : int, optional
            Number of points to sample the facula with. Default is 101.

        Returns
        -------
        dict
            Effective area of the wall and floor. The keys are the Teff, the
            values are the area. Both are `astropy.units.Quantity` objects.

        Notes
        -----
        The effective area is computed by numerical integration. The visible area of the
        hot wall is:

        .. math::
            \\int _{-R}^{R} Z_{\\rm eff} dr

        and the visible area of the cool floor is:

        .. math::
            \\int _{-R}^{R} Z_{\\rm eff} dr

        Where

        .. math::
            Z_{\\rm eff} = \\left\\{
                \\begin{array}{lr}
                    Z_w \\sin{\\alpha}, & \\text{if } \\alpha \\leq \\alpha_{\\rm crit} \\\\
                    2\\sqrt{R^2 - r^2}\\cos{\\alpha}, & \\text{if } \\alpha > \\alpha_{\\rm crit}
                \\end{array}
                \\right\\}

        and

        .. math::
            R_{\\rm eff} = \\left\\{
                \\begin{array}{lr}
                    2\\sqrt{R^2 - r^2} - Z_w\\sin{\\alpha}, & \\text{if } \\alpha \\leq \\alpha_{\\rm crit} \\\\
                    0, & \\text{if } \\alpha > \\alpha_{\\rm crit}
                \\end{array}
                \\right\\}

        for facula radius :math:`R`, depth :math:`Z_w`, and angle from disk-center :math:`\\alpha`. :math:`r` in
        this numerical scheme is defined as the distance from the center of the facula along the radial line
        connecting the facula center to the disk center. :math:`\\alpha_{\\rm crit}` is the value of alpha at
        which the floor is no longer visible and is defined to be :math:`\\arctan{\\frac{2\\sqrt{R^2-r^2}}{Z_w}}`. 
        """
        if self.floor_dteff == self.wall_dteff:
            raise ValueError(
                'Wall and Floor teff are equal. This is not allowed.')
        if self.radius < self.floor_teff_min_rad:
            return {
                round_teff(self.wall_dteff): np.pi*self.radius**2 * np.cos(angle),
                round_teff(self.floor_dteff): 0.0 * u.km**2
            }
        else:
            # distance from center along azmuth of disk
            x = np.linspace(-1, 1, N) * self.radius
            # effective radius of the 1D facula approximation
            h = np.sqrt(self.radius**2 - x**2)
            critical_angles = np.arctan(2*h/self.depth)
            Zeffs = np.sin(angle)*np.ones(N) * self.depth
            Reffs = np.cos(angle)*h*2 - self.depth * np.sin(angle)
            no_floor = critical_angles < angle
            Zeffs[no_floor] = 2*h[no_floor]*np.cos(angle)
            Reffs[no_floor] = 0

            return {
                round_teff(self.wall_dteff): np.trapz(Zeffs, x),
                round_teff(self.floor_dteff): np.trapz(Reffs, x)
            }

    def fractional_effective_area(self, angle: Quantity[u.deg],
                                  N: int = 101) -> Dict[Quantity[u.K], Quantity]:
        """
        Calculate the fractional effective area as a fraction of the
        projected area of a region of quiet photosphere with
        the same radius and distance from limb.

        Parameters
        ----------
        angle : astropy.units.Quantity
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
        star_rad : astropy.units.Quantity
            The radius of the star.

        Returns
        -------
        astropy.units.Quantity
            The angular radius of the facula.
        """
        return self.radius/star_rad * 180/np.pi * u.deg

    def map_pixels(self, star_rad):
        """
        Map pixels onto the surface of the facula.

        Parameters
        ----------
        star_rad : astropy.units.Quantity
            The radius of the star.

        Returns
        -------
        numpy.ndarray
            Boolean array indicating whether each pixel is within the facula radius.
        """
        rad = self.angular_radius(star_rad)
        pix_in_fac = self._r <= rad
        return pix_in_fac


class FaculaCollection:
    """
    Container class to store faculae.

    Parameters
    ----------
    *faculae : tuple
        A series of faculae objects.
    nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    nlon : int, default=1000
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

    def __init__(self, *faculae: Tuple[Facula],
                 nlat: int = config.nlat,
                 nlon: int = config.nlon,
                 gridmaker: CoordinateGrid = None):
        self.faculae: Tuple[Facula] = tuple(faculae)

        if not gridmaker:
            self.gridmaker = CoordinateGrid(nlat, nlon)
        else:
            self.gridmaker = gridmaker
        for facula in faculae:
            facula: Facula
            if not facula.gridmaker == self.gridmaker:
                facula.gridmaker = gridmaker

    def add_faculae(self, facula: Tuple[Facula] or Facula) -> None:
        """
        Add a facula or faculae

        Parameters
        ----------
        facula : Facula or series of Facula
            Facula object(s) to add.
        """
        if isinstance(facula, Facula):
            facula = (facula,)
        for fac in facula:
            fac: Facula
            if not fac.gridmaker == self.gridmaker:
                fac.gridmaker = self.gridmaker
        self.faculae += tuple(facula)

    def clean_faclist(self) -> None:
        """
        Remove faculae that have decayed to Rmax/e**2 radius.
        """
        faculae_to_keep = []
        for facula in self.faculae:
            facula: Facula
            if (facula.radius <= facula.r_max/np.e**2) and (not facula.is_growing):
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
        time : astropy.units.Quantity 
            Length of time to age the spot.
            For most realistic behavior, time should be << spot lifetime.
        """
        for facula in self.faculae:
            facula.age(time)
        self.clean_faclist()

    def map_pixels(self, pixmap, star_rad, star_teff):
        """
        .. deprecated:: 0.1
            This functionality is now performed by the ``Star`` object.

        Map facula parameters to pixel locations

        Parameters
        ----------
        pixmap : astropy.units.Quantity , shape(M,N)
            Grid of effective temperature.
        star_rad : astropy.units.Quantity
            Radius of the star.
        star_teff : astropy.units.Quantity
            Temperature of quiet stellar photosphere.

        Returns
        -------
        int_map : np.ndarray
            Grid of integer keys showing facula locations.
        map_dict : dict
            Dictionary mapping index in self.faculae to the integer grid of facula locations.
        """
        warnings.warn('Use `Star.add_faculae_to_map` method instead.')
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
    dist_r_peak : astropy.units.Quantity
        The peak of the radius distribution.
    dist_r_logsigma : float
        The standard deviation of :math:`\\log{r_0}`
    depth : astropy.units.Quantity
        The depth of the facula depression.
    dist_life_peak : astropy.units.Quantity
        The peak of the lifetime distribution.
    dist_life_logsigma : float
        The standard deviation of :math:`\\log{\\tau}`
    floor_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    floor_teff_min_rad : astropy.units.Quantity
        The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    floor_teff_base_dteff : astropy.units.Quantity
        The Teff of the floor at the minimum radius.
    wall_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    wall_teff_intercept : astropy.units.Quantity
        The Teff of the wall when :math:`R = 0`.
    coverage : float
        The fraction of the surface covered by faculae in growth-deacay equilibrium.
    dist : str
        Type of distribution. Currently only `iso` is supported.
    nlat : int, default=VSPEC.config.nlat
        The number of latitude points on the stellar sufrace.
    nlon : int, default=VSPEC.config.nlon
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    rng : numpy.random.Generator, default=numpy.random.default_rng()
        The random number generator to use.

    Attributes
    ----------
    mean_area
    dist_r_peak : astropy.units.Quantity
        The peak of the radius distribution.
    dist_r_logsigma : float
        The standard deviation of :math:`\\log{r_0}`
    depth : astropy.units.Quantity
        The depth of the facula depression.
    dist_life_peak : astropy.units.Quantity
        The peak of the lifetime distribution.
    dist_life_logsigma : float
        The standard deviation of :math:`\\log{\\tau}`
    floor_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    floor_teff_min_rad : astropy.units.Quantity
        The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    floor_teff_base_dteff : astropy.units.Quantity
        The Teff of the floor at the minimum radius.
    wall_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    wall_teff_intercept : astropy.units.Quantity
        The Teff of the wall when :math:`R = 0`.
    coverage : float
        The fraction of the surface covered by faculae in growth-deacay equilibrium.
    dist : str
        Type of distribution. Currently only `iso` is supported.
    nlat : int
        The number of latitude points on the stellar sufrace.
    nlon : int
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid or None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    rng : numpy.random.Genrator
        The random number generator to use.

    """

    def __init__(
        self,
        dist_r_peak: Quantity,
        dist_r_logsigma: float,
        depth: Quantity,
        dist_life_peak: Quantity,
        dist_life_logsigma: float,
        floor_teff_slope: Quantity,
        floor_teff_min_rad: Quantity,
        floor_teff_base_dteff: Quantity,
        wall_teff_slope: Quantity,
        wall_teff_intercept: Quantity,
        coverage: float,
        dist: str = 'iso',
        nlon: int = config.nlon,
        nlat: int = config.nlat,
        gridmaker=None,
        rng: np.random.Generator = np.random.default_rng()
    ):
        self.dist_r_peak = dist_r_peak
        self.dist_r_logsigma = dist_r_logsigma
        self.depth = depth
        self.dist_life_peak = dist_life_peak
        self.dist_life_logsigma = dist_life_logsigma
        self.floor_teff_slope = floor_teff_slope
        self.floor_teff_min_rad = floor_teff_min_rad
        self.floor_teff_base_dteff = floor_teff_base_dteff
        self.floor_teff_base_dteff = floor_teff_base_dteff
        self.wall_teff_slope = wall_teff_slope
        self.wall_teff_intercept = wall_teff_intercept
        if not isinstance(coverage, (float, int)):
            raise TypeError('coverage must be float or int.')
        self.coverage = coverage
        if dist == 'solar':
            raise NotImplementedError(
                '`solar` distribution only allowed for spots.')
        elif dist != 'iso':
            raise ValueError(f'Unknown distribution `{dist}`.')
        self.dist = dist
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(nlat, nlon)
        else:
            self.gridmaker = gridmaker
        self.nlon = nlon
        self.nlat = nlat
        self.rng = rng

    @classmethod
    def from_params(
        cls,
        facparams: FaculaParameters,
        nlat: int = config.nlat,
        nlon: int = config.nlon,
        gridmaker: CoordinateGrid = None,
        rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Construct an instance from a ``FaculaParameters`` object.

        Parameters
        ----------
        facparams : FaculaParameters
            The set of parameters to construct from.
        nlat : int, default=VSPEC.config.nlat
            The number of latitude points on the stellar sufrace.
        nlon : int, default=VSPEC.config.nlon
            The number of longitude points on the stellar surface.
        gridmaker : CoordinateGrid, default=None
            A `CoordinateGrid` object to create the stellar sufrace grid.
        rng : numpy.random.Genrator, default=numpy.random.default_rng()
            The random number generator to use.
        """
        return cls(
            dist_r_peak=facparams.mean_radius,
            dist_r_logsigma=facparams.logsigma_radius,
            depth=facparams.depth,
            dist_life_peak=facparams.mean_timescale,
            dist_life_logsigma=facparams.logsigma_timescale,
            floor_teff_slope=facparams.floor_teff_slope,
            floor_teff_min_rad=facparams.floor_teff_min_rad,
            floor_teff_base_dteff=facparams.floor_teff_base_dteff,
            wall_teff_slope=facparams.wall_teff_slope,
            wall_teff_intercept=facparams.wall_teff_intercept,
            coverage=facparams.equillibrium_coverage,
            dist=facparams.distribution,
            nlat=nlat,
            nlon=nlon,
            gridmaker=gridmaker,
            rng=rng
        )

    @property
    def mean_area(self):
        """
        The time-averaged area of a typical facula.

        .. math::
            \\bar{A} = \\frac{ \\pi r_0 ^2 (1-e^{-2}) }{2}

        :type: astropy.units.Quantity
        """
        return 0.5 * np.pi * (1-np.e**-2) * self.dist_r_peak**2

    def get_n_faculae_expected(self, time: u.Quantity, rad_star: u.Quantity) -> float:
        """
        Over a given time duration, compute the number of new faculae to create.

        Parameters
        ----------
        time : astropy.units.Quantity
            Time over which to create faculae.
        rad_star : astropy.units.Quantity
            Radius of the star.

        Returns
        -------
        n_exp : float
            The expected number of faculae to be birthed in the given time.

        Notes
        -----
        .. math::
            N_{exp} = \\frac{X 4 \\pi R_* ^2 t}{2\\tau \\bar{A}}

        For coverage fraction :math:`X`, stellar radius :math:`R_*`, time :math:`t`,
        lifetime :math:`\\tau`, and time-averaged area :math:`\\bar{A}`.
        """
        n_exp = self.coverage * 4*np.pi*rad_star**2 / \
            self.mean_area * time / (2*self.dist_life_peak)
        return n_exp.to_value(u.dimensionless_unscaled)

    def get_coords(self, N: int):
        """
        Generate random coordinates for new Faculae to be centered at.

        Parameters
        ----------
        N : int
            The number of lat/lon pairs to create.

        Returns
        -------
        lats : astropy.units.Quantity
            The latitude coordinates of the new faculae.
        lons : astropy.units.Quantity
            The longitude coordinates of the new faculae.

        Raises
        ------
        NotimplementedError
            If `dist` is 'solar'.
        ValueError
            If `dist` is not recognized.
        """
        if self.dist == 'iso':
            X = self.rng.random(size=N)
            lats = np.arcsin(2*X - 1)/np.pi * 180*u.deg
            lons = self.rng.random(size=N) * 360 * u.deg
            return lats, lons
        elif self.dist == 'solar':
            raise NotImplementedError(
                f'{self.dist} has not been implemented as a distribution')
        else:
            raise ValueError(
                f'{self.dist} is not recognized as a distribution')

    def generate_faculae(self, N: int):
        """
        Generate a given number of new Faculae

        Parameters
        ----------
        N : int
            The number of faculae to generate.
        Teff_star : astropy.units.Quantity
            Temperature of the star.

        Returns
        -------
        tuple of Facula
            Tuple of new faculae.
        """
        mu = self.rng.normal(loc=0, scale=1, size=N)
        max_radii = self.dist_r_peak * 10**(mu*self.dist_r_logsigma)
        lifetimes = self.dist_life_peak * 10**(mu*self.dist_life_logsigma)
        starting_radii = max_radii / np.e**2
        lats, lons = self.get_coords(N)
        new_faculae = []
        for i in range(N):
            new_faculae.append(
                Facula(
                    lat=lats[i],
                    lon=lons[i],
                    r_max=max_radii[i],
                    r_init=starting_radii[i],
                    depth=self.depth,
                    lifetime=lifetimes[i],
                    floor_teff_slope=self.floor_teff_slope,
                    floor_teff_min_rad=self.floor_teff_min_rad,
                    floor_teff_base_dteff=self.floor_teff_base_dteff,
                    wall_teff_slope=self.wall_teff_slope,
                    wall_teff_intercept=self.wall_teff_intercept,
                    growing=True,
                    nlat=self.nlat,
                    nlon=self.nlon,
                    gridmaker=None
                )
            )
        return tuple(new_faculae)

    def birth_faculae(self, time: u.Quantity, rad_star: u.Quantity):
        """
        Over a given time duration, compute the number of new faculae to create.
        Create new faculae and assign them parameters.

        Parameters
        ----------
        time : astropy.units.Quantity
            Time over which to create faculae.
        rad_star : astropy.units.Quantity
            Radius of the star.
        Teff_star : astropy.units.Quantity
            Temperature of the star.

        Returns
        -------
        tuple of Facula
            Tuple of new faculae.
        """
        N_exp = self.get_n_faculae_expected(time, rad_star)
        N = self.rng.poisson(lam=N_exp)
        return self.generate_faculae(N)
