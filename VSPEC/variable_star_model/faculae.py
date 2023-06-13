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
    teff_floor : astropy.units.Quantity
        Effective temperature of the 'cool floor'
    teff_wall : astropy.units.Quantity
        Effective temperature of the 'hot wall'
    lifetime : astropy.units.Quantity
        Facula lifetime
    growing : bool, default=True
        Whether or not the facula is still growing.
    floor_threshold : astropy.units.Quantity , default=20*u.km
        Facula radius under which the floor is no longer visible.
        Small faculae appear as bright points regardless of their
        distance to the limb.
    nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A ``CoordinateGrid`` object to create the stellar sufrace grid.

    Attributes
    ----------
    _r
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
    teff_floor : astropy.units.Quantity
        Effective temperature of the 'cool floor'.
    teff_wall : astropy.units.Quantity
        Effective temperature of the 'hot wall'.
    lifetime : astropy.units.Quantity
        Facula lifetime.
    is_growing : bool
        Whether or not the facula is still growing.
    floor_threshold : astropy.units.Quantity
        Facula radius under which the floor is no longer visible.
        Small faculae appear as bright points regardless of their
        distance to the limb.
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
        teff_floor: Quantity,
        teff_wall: Quantity,
        lifetime: Quantity,
        floor_threshold: Quantity,
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
        self.teff_floor = round_teff(teff_floor)
        self.teff_wall = round_teff(teff_wall)
        self.lifetime = lifetime
        self.is_growing = growing
        self.floor_threshold = floor_threshold

        if gridmaker is None:
            self.set_gridmaker(CoordinateGrid(nlat, nlon))
        else:
            self.set_gridmaker(gridmaker)
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
        warnings.warn(msg,DeprecationWarning)
        

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
        if self.radius < self.floor_threshold:
            return {round_teff(self.teff_floor): 0.0 * u.km**2, round_teff(self.teff_wall): np.pi*self.radius**2 * np.cos(angle)}
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

            return {round_teff(self.teff_wall): np.trapz(Zeffs, x), round_teff(self.teff_floor): np.trapz(Reffs, x)}

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
        self.faculae:Tuple[Facula] = tuple(faculae)

        if not gridmaker:
            self.gridmaker = CoordinateGrid(nlat, nlon)
        else:
            self.gridmaker = gridmaker
        for facula in faculae:
            facula:Facula
            if not facula.gridmaker == self.gridmaker:
                facula.gridmaker = gridmaker

    def add_faculae(self, facula:Tuple[Facula] or Facula)->None:
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
            fac:Facula
            if not fac.gridmaker == self.gridmaker:
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
        time : astropy.units.Quantity 
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
    R_peak : astropy.units.Quantity
        Radius to use as the peak of the distribution.
    R_HWHM : astropy.units.Quantity
        Radius half width half maximum. Difference between the peak of
        the radius distribution and the half maximum in
        the positive direction.
    T_peak : astropy.units.Quantity 
        Lifetime to use as the peak of the distribution.
    T_HWHM : astropy.units.Quantity 
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
    """
    astropy.units.Unit
        Unit of radius used for the facula.
    
    """
    lifetime_unit = u.hr
    """
    astropy.units.Unit
        Unit of lifetime used for the facula.
    """

    def __init__(
        self,
        dist_r_peak: Quantity,
        dist_r_hwhm: Quantity,
        dist_life_peak: Quantity,
        dist_life_hwhm: Quantity,
        floor_teff_slope: Quantity,
        coverage: float,
        dist: str = 'iso',
        nlon: int = config.nlon,
        nlat: int = config.nlat,
        gridmaker=None,
        teff_bounds=config.grid_teff_bounds
    ):
        self.dist_logr_peak = np.log10(
            dist_r_peak/self.radius_unit).to_value(u.dimensionless_unscaled)
        self.dist_logr_sigma = np.log10(
            (dist_r_peak + dist_r_hwhm)/self.radius_unit).to_value(u.dimensionless_unscaled) - self.dist_logr_peak
        self.dist_loglife_peak = np.log10(
            dist_life_peak/self.lifetime_unit).to_value(u.dimensionless_unscaled)
        self.dist_loglife_sigma = np.log10(
            (dist_life_peak + dist_life_hwhm)/self.lifetime_unit).to_value(u.dimensionless_unscaled) - self.dist_loglife_peak
        self.floor_teff_slope = floor_teff_slope
        if not isinstance(coverage,(float,int)):
            raise TypeError('coverage must be float or int.')
        self.coverage = coverage
        if dist == 'solar':
            raise NotImplementedError('`solar` distribution only allowed for spots.')
        elif dist != 'iso':
            raise ValueError(f'Unknown distribution `{dist}`.')
        self.dist = dist
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(nlat, nlon)
        else:
            self.gridmaker = gridmaker
        self.nlon = nlon
        self.nlat = nlat
        self.teff_bounds = teff_bounds

    def get_floor_teff(self, R: u.Quantity, Teff_star: u.Quantity) -> u.Quantity:
        """
        Get the floor temperature of faculae based on the radius
        and photosphere effective temperature.

        Parameters
        ----------
        R : astropy.units.Quantity 
            Radius of the facula[e].
        Teff_star : astropy.units.Quantity 
            Effective temperature of the photosphere.

        Returns
        -------
        astropy.units.Quantity 
            Floor temperature of faculae.

        Notes
        -----
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
        R : astropy.units.Quantity 
            Radius of the facula[e].
        Teff_floor : astropy.units.Quantity 
            Effective temperature of the cool floor.

        Returns
        -------
        astropy.units.Quantity 
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
        float
            The expected number of faculae to be birthed in the given time.
        """
        N_exp = (self.coverage * 4*np.pi*rad_star**2 / ((10**self.dist_logr_peak*self.radius_unit)**2 * np.pi)
                 * time/(10**self.dist_loglife_peak * self.lifetime_unit * 2)).to_value(u.Unit(''))
        return N_exp

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
            X = np.random.random(size=N)
            lats = np.arcsin(2*X - 1)/np.pi * 180*u.deg
            lons = np.random.random(size=N) * 360 * u.deg
            return lats, lons
        elif self.dist == 'solar':
            raise NotImplementedError(
                f'{self.dist} has not been implemented as a distribution')
        else:
            raise ValueError(
                f'{self.dist} is not recognized as a distribution')

    def generate_faculae(self, N: int, Teff_star: u.Quantity):
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
        mu = np.random.normal(loc=0, scale=1, size=N)
        max_radii = 10**(self.dist_logr_peak + self.dist_logr_sigma * mu) * self.radius_unit
        lifetimes = 10**(self.dist_loglife_peak + self.dist_loglife_sigma * mu) * self.lifetime_unit
        starting_radii = max_radii / np.e**2
        lats, lons = self.get_coords(N)
        teff_floor = self.get_floor_teff(max_radii, Teff_star)
        teff_wall = self.get_wall_teff(max_radii, teff_floor)
        new_faculae = []
        for i in range(N):
            new_faculae.append(Facula(lats[i], lons[i], max_radii[i], starting_radii[i], teff_floor[i],
                                      teff_wall[i], lifetimes[i], growing=True, floor_threshold=20*u.km, depth=100*u.km,
                                      nlon=self.nlon, nlat=self.nlat))
        return tuple(new_faculae)

    def birth_faculae(self, time: u.Quantity, rad_star: u.Quantity, Teff_star: u.Quantity):
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
        N = np.random.poisson(lam=N_exp)
        return self.generate_faculae(N, Teff_star)
