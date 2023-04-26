"""VSPEC Facula module

This code describes the behavior of stellar faculae.
"""
from typing import Dict

import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity

from VSPEC.helpers import CoordinateGrid, round_teff


class Facula:
    """
    Class containing model parameters of stellar faculae using the 'hot wall' model.

    Parameters
    ----------
    lat : astropy.units.Quantity 
        Latitude of facula center
    lon : astropy.units.Quantity 
        Longitude of facula center
    Rmax : astropy.units.Quantity 
        Maximum radius of facula
    R0 : astropy.units.Quantity 
        Current radius of facula
    Zw : astropy.units.Quantity 
        Depth of the depression.
    Teff_floor : astropy.units.Quantity 
        Effective temperature of the 'cool floor'
    Teff_wall : astropy.units.Quantity 
        Effective temperature of the 'hot wall'
    lifetime : astropy.units.Quantity 
        Facula lifetime
    growing : bool, default=True
        Whether or not the facula is still growing.
    floor_threshold : astropy.units.Quantity , default=20*u.km
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
    lat : astropy.units.Quantity 
        Latitude of facula center.
    lon : astropy.units.Quantity 
        Longitude of facula center.
    Rmax : astropy.units.Quantity 
        Maximum radius of facula.
    current_R : astropy.units.Quantity 
        Current radius of facula.
    Zw : astropy.units.Quantity 
        Depth of the depression.
    Teff_floor : astropy.units.Quantity 
        Effective temperature of the 'cool floor'.
    Teff_wall : astropy.units.Quantity 
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
    r : astropy.units.Quantity 
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
        self.Teff_floor = round_teff(Teff_floor)
        self.Teff_wall = round_teff(Teff_wall)
        self.lifetime = lifetime
        self.is_growing = growing
        self.floor_threshold = floor_threshold

        if gridmaker is None:
            self.set_gridmaker(CoordinateGrid(Nlat, Nlon))
        else:
            self.set_gridmaker(gridmaker)
        
    def set_gridmaker(self,gridmaker:CoordinateGrid):
        """
        Set the `gridmaker` attribute safely.

        Parameters
        ----------
        gridmaker : VSPEC.helpers.CoordinateGrid
            The `CoordinateGrid` object to set
        """
        self.gridmaker = gridmaker
        latgrid, longrid = self.gridmaker.grid()
        lat0 = self.lat
        lon0 = self.lon
        self.r = 2 * np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
                                       + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))


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
            T_from_max = -1*np.log(self.current_R/self.Rmax)*self.lifetime*0.5
            if T_from_max <= time:
                self.is_growing = False
                time = time - T_from_max
                self.current_R = self.Rmax * np.exp(-2*time/self.lifetime)
            else:
                self.current_R = self.current_R * np.exp(2*time/self.lifetime)
        else:
            self.current_R = self.current_R * np.exp(-2*time/self.lifetime)


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
        """
        if self.current_R < self.floor_threshold:
            return {round_teff(self.Teff_floor): 0.0 * u.km**2, round_teff(self.Teff_wall): np.pi*self.current_R**2 * np.cos(angle)}
        else:
            # distance from center along azmuth of disk
            x = np.linspace(-1, 1, N) * self.current_R
            # effective radius of the 1D facula approximation
            h = np.sqrt(self.current_R**2 - x**2)
            critical_angles = np.arctan(2*h/self.Zw)
            Zeffs = np.sin(angle)*np.ones(N) * self.Zw
            Reffs = np.cos(angle)*h*2 - self.Zw * np.sin(angle)
            no_floor = critical_angles < angle
            Zeffs[no_floor] = 2*h[no_floor]*np.cos(angle)
            Reffs[no_floor] = 0

            return {round_teff(self.Teff_wall): np.trapz(Zeffs, x), round_teff(self.Teff_floor): np.trapz(Reffs, x)}

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
        return self.current_R/star_rad * 180/np.pi * u.deg

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
