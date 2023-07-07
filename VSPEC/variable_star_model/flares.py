"""VSPEC Flares module

This code governs the behavior of flares.
"""
from typing import List
import typing as Typing

import numpy as np
from astropy import units as u, constants as const
from astropy.units.quantity import Quantity

from VSPEC.params import FlareParameters

from xoflares.xoflares import _flareintegralnp as flareintegral, get_light_curvenp


class StellarFlare:
    """
    Class to store and control stellar flare information

    Parameters
    ----------
    fwhm : astropy.units.Quantity
        Full-width-half-maximum of the flare
    energy : astropy.units.Quantity
        Time-integrated bolometric energy
    lat : astropy.units.Quantity
        Latitude of flare on star
    lon : astropy.units.Quantity
        Longitude of flare on star
    Teff : astropy.units.Quantity
        Blackbody temperature
    tpeak : astropy.units.Quantity
        Time of the flare peak

    Attributes
    ----------
    fwhm : astropy.units.Quantity
        Full-width-half-maximum of the flare
    energy : astropy.units.Quantity
        Time-integrated bolometric energy
    lat : astropy.units.Quantity
        Latitude of flare on star
    lon : astropy.units.Quantity
        Longitude of flare on star
    Teff : astropy.units.Quantity
        Blackbody temperature
    tpeak : astropy.units.Quantity
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
        astropy.units.Quantity
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
        time : astropy.units.Quantity 
            The times at which to sample the area

        Returns
        -------
        astropy.units.Quantity 
            Area at each time

        """
        t_unit = u.day  # this is the unit of xoflares
        a_unit = u.km**2
        peak_area = self.calc_peak_area()
        areas = get_light_curvenp(
            time.to_value(t_unit),
            [self.tpeak.to_value(t_unit)],
            [self.fwhm.to_value(t_unit)],
            [peak_area.to_value(a_unit)]
        )
        return areas * a_unit

    def get_timearea(self, time: Quantity[u.hr]):
        """
        Calcualte the integrated time*area of the flare.

        Parameters
        ----------
        time : astropy.units.Quantity
            the times at which to sample the area.

        Returns
        -------
        astropy.units.Quantity
            The integrated time-area of the flare.
        """
        areas = self.areacurve(time)
        timearea = np.trapz(areas, time)
        return timearea.to(u.Unit('hr km2'))


class FlareGenerator:
    """
    Generator for flare events and their characteristics.

    Parameters
    ----------
    dist_teff_mean : astropy.units.Quantity
        The mean of the temperature distribution.
    dist_teff_sigma : astropy.units.Quantity
        The standard deviation of the temperature distribution.
    dist_fwhm_mean : astropy.units.Quantity
        The mean of the FWHM distribution.
    dist_fwhm_logsigma : float
        The standard deviation of the logorithm of the FWHM distribution in dex.
    alpha : float, default=-0.829
        The slope of the log frequency - log energy relationship.
    beta : float, default=26.87
        The y-intercept of the log frequency - log energy relationship.
    min_energy : astropy.units.Quantity, default=1e33 erg
        The minimum energy to consider. Set to ``np.inf*u.erg`` to disable flares.
    cluster_size : int, default=2
        The typical size of flare clusters.
    rng : numpy.random.Generator, default=numpy.random.default_rng()
        The random number generator instance to use.

    Attributes
    ----------
    energy_unit
    time_unit
    temperature_unit
    dist_teff_mean : astropy.units.Quantity
        The mean of the temperature distribution.
    dist_teff_sigma : astropy.units.Quantity
        The standard deviation of the temperature distribution.
    dist_fwhm_mean : astropy.units.Quantity
        The mean of the FWHM distribution.
    dist_fwhm_logsigma : float
        The standard deviation of the logorithm of the FWHM distribution in dex.
    alpha : float
        The slope of the log frequency - log energy relationship.
    beta : float
        The y-intercept of the log frequency - log energy relationship.
    min_energy : astropy.units.Quantity
        The minimum energy to consider. Set to ``np.inf*u.erg`` to disable flares.
    cluster_size : int
        The typical size of flare clusters.
    rng : numpy.random.Generator
        The random number generator instance to use.

    Notes
    -----
    Relationship between flare frequency is defined as

    .. math::
        \\log{(f(E \\ge E_0)~\\text{[day]})} = \\beta + \\alpha \\log{(E_0/\\text{[erg]})}

    by :cite:t:`2022AJ....164..213G`. Their study of TESS flare rates fits :math:`\\alpha = -0.829`
    and :math:`\\beta = 26.87`, which we use as the default values. This relationship is valid for
    :math:`E > 10^{33}` erg, which we use as the default minimum considered energy.

    To disable flares, set ``min_energy`` to infinity so that the expected number of flares is 0.

    """
    energy_unit = u.erg
    """
    The energy unit to standardize coversions to floats and logs.

    :type: astropy.units.Quantity
    """
    time_unit = u.day
    """
    The time unit to standardize coversions to floats and logs.

    :type: astropy.units.Quantity
    """
    temperature_unit = u.K
    """
    The temperature unit to standardize coversions to floats and logs.

    :type: astropy.units.Quantity
    """

    def __init__(
        self,
        dist_teff_mean: u.Quantity,
        dist_teff_sigma: u.Quantity,
        dist_fwhm_mean: u.Quantity,
        dist_fwhm_logsigma: float,
        alpha: float = -0.829,
        beta: float = 26.87,
        min_energy: u.Quantity = 1e33*u.erg,
        cluster_size: int = 2,
        rng: np.random.Generator = np.random.default_rng()
    ):
        self.dist_teff_mean = dist_teff_mean
        self.dist_teff_sigma = dist_teff_sigma
        self.dist_fwhm_mean = dist_fwhm_mean
        self.dist_fwhm_logsigma = dist_fwhm_logsigma
        self.min_energy = min_energy
        self.cluster_size = cluster_size
        self.alpha = alpha
        self.beta = beta
        self.rng = rng

    @classmethod
    def from_params(cls, flareparams: FlareParameters, rng: np.random.Generator):
        """
        Load a ``FlareGenerator`` from a ``FlareParameters`` instance.

        Parameters
        ----------
        flareparams : FlareParameters
            The object to load from.
        rng : numpy.random.Generator
            The random number generator to use.
        """
        return cls(
            dist_teff_mean=flareparams.dist_teff_mean,
            dist_teff_sigma=flareparams.dist_teff_sigma,
            dist_fwhm_mean=flareparams.dist_fwhm_mean,
            dist_fwhm_logsigma=flareparams.dist_fwhm_logsigma,
            alpha=flareparams.alpha,
            beta=flareparams.beta,
            min_energy=flareparams.min_energy,
            cluster_size=flareparams.cluster_size,
            rng=rng
        )

    def frequency_greater_than(self, energy: u.Quantity):
        """
        The frequency of flares greater than an energy.

        Parameters
        ----------
        energy : astropy.units.Quantity
            The energy to compute the frequency of.

        Returns
        -------
        astropy.units.Quantity
            The frequency of flares with energy greater than `energy`.
        """
        return 10**self.beta * (energy.to_value(self.energy_unit))**self.alpha / self.time_unit

    def get_nexp_greater_than(self, energy: u.Quantity, time: u.Quantity) -> float:
        """
        Get the expected number of flares with energy greater than `energy` in a
        time period `time`

        Parameters
        ----------
        energy : astropy.units.Quantity
            The energy to compute the frequency of.
        time : astropy.units.Quantity
            The time duration.

        Returns
        -------
        float
            The number of flares with energy greater than `energy`.
        """
        return (self.frequency_greater_than(energy)*time).to_value(u.dimensionless_unscaled)

    def get_ntotal(self, time) -> float:
        """
        Get the total number of flares in a time period.

        Parameters
        ----------
        time : astropy.units.Quantity
            The time duration.

        Returns
        -------
        float
            The number of flares.
        """
        return self.get_nexp_greater_than(self.min_energy, time)

    def pdf(self, energy: u.Quantity):
        """
        The probability density function of flare energies.

        Parameters
        ----------
        energy : astropy.units.Quantity
            The energies at which to sample the pdf.

        Returns
        -------
        pdf : float or np.ndarray
            The probability density of energy
        """
        return (-self.alpha * energy**(self.alpha-1)/self.min_energy**self.alpha).to_value(u.dimensionless_unscaled)

    def cdf(self, energy: u.Quantity):
        """
        Get the cumulative density function.

        Parameters
        ----------
        energy : astropy.units.Quantity
            The energies at which to sample the cdf.

        Returns
        -------
        cdf : float or np.ndarray
            The cumulative density of energy
        """
        return 1 - (energy/self.min_energy).to_value(u.dimensionless_unscaled)**self.alpha

    def quantile_func(self, X: np.ndarray):
        """
        The quantile function of this flare energy distribution.

        This function maps a variable on the range [0,1) to the distribution
        defined by :math:`\\beta` and :math:`\\alpha`.

        Parameters
        ----------
        X : numpy.ndarray
            A random variable with domain [0,1)

        Returns
        -------
        energy : astropy.Units.Quantity
            The energies represented by ``X``.
        """
        X = np.atleast_1d(X)
        if not (np.all(X >= 0.) and np.all(X < 1.)):
            raise ValueError('`X` must be in [0,1)')
        return self.min_energy * (1-X)**(1/self.alpha)

    def get_peaks(self, n_flares):
        """
        Generate the times at which each flare reaches its peak.

        Parameters
        ----------
        n_flares : int
            The number of peak times to generate.

        Returns
        -------
        tpeaks : astropy.units.Quantity
            The times that the flares reach their peak.
        """
        cluster_sizes = self.rng.poisson(lam=self.cluster_size, size=n_flares)
        cluster_sizes = cluster_sizes[np.cumsum(cluster_sizes) < n_flares]
        n_leftover = n_flares - np.sum(cluster_sizes)
        cluster_sizes = np.append(cluster_sizes, [1]*n_leftover)

        n_clusters = len(cluster_sizes)

        cluster_timescale = 1/n_clusters

        dtimes = []
        for size in cluster_sizes:
            dtimes.append(self.rng.exponential(cluster_timescale))
            flare_timescale = cluster_timescale/size/2
            for _ in range(size-1):
                dtimes.append(self.rng.exponential(flare_timescale))
        dtimes = np.array(dtimes)
        tpeaks = np.cumsum(dtimes)
        return tpeaks

    def gen_fwhm(self, n_flares: int):
        """
        Generate a FWHM set in a lognormal distribution.

        Parameters
        ----------
        n_flares : int
            The number of random FWHM lengths to generate.

        Returns
        -------
        fwhm : astropy.units.Quantity
            The FWHMs of the flares
        """
        dist_logmean_fwhm = np.log10(
            self.dist_fwhm_mean.to_value(self.time_unit))
        log_fwhm = self.rng.normal(
            loc=dist_logmean_fwhm,
            scale=self.dist_fwhm_logsigma,
            size=n_flares
        )
        fwhm = 10**log_fwhm * self.time_unit
        return fwhm

    def gen_teffs(self, n_flares):
        """
        Generate a random set of flare temperatures.

        Parameters
        ----------
        n_flares : int
            The number of random temperatures to generate.

        Returns
        -------
        teffs : astropy.units.Quantity
            The temperatures of the flares
        """

        teffs = self.rng.normal(
            loc=self.dist_teff_mean.to_value(self.temperature_unit),
            scale=self.dist_teff_sigma.to_value(self.temperature_unit),
            size=n_flares
        )*self.temperature_unit
        return teffs

    def gen_coords(self, n_flares):
        """
        Generate a set of random coordinates.

        Parameters
        ----------
        n_flares : int
            The number of random temperatures to generate.

        Returns
        -------
        lats : astropy.units.Quantity
            The stellar latitudes of the flares.
        lons : astropy.units.Quantity
            The stellar longitudes of the flares.

        Notes
        -----
        ``lats`` is generated using a quantile function for latitude.

        .. math::
            \\text{lat} = \\arcsin(2X - 1)

        For a random variable :math:`X` on [0,1).
        """
        lons = self.rng.random(size=n_flares)*360*u.deg
        lats = np.arcsin(2*self.rng.random(size=n_flares) - 1) / \
            np.pi * 180*u.deg
        return lats, lons

    def generate_flare_series(self, time: u.Quantity):
        """
        Generate a series of flares based on the distribution parameters of this
        generator.

        Parameters
        ----------
        time : astropy.units.Quantity
            The time over which the flares are observed.

        Returns
        -------
        flares : list of StellarFlare
            The flares that occur over ``time``.
        """
        nexp = self.get_ntotal(time)
        n_flares = self.rng.poisson(nexp)
        if n_flares == 0:
            return []

        energies = self.quantile_func(self.rng.random(n_flares))
        tpeaks = self.get_peaks(n_flares)*time
        fwhms = self.gen_fwhm(n_flares)
        teffs = self.gen_teffs(n_flares)
        lats, lons = self.gen_coords(n_flares)

        flares = [
            StellarFlare(
                fwhm=f,
                energy=e,
                lat=lat,
                lon=lon,
                Teff=teff,
                tpeak=t
            ) for f, e, lat, lon, teff, t in zip(fwhms, energies, lats, lons, teffs, tpeaks)
        ]
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
            tpeak.append(flare.tpeak.to_value(unit))
            fwhm.append(flare.fwhm.to_value(unit))
        tpeak = np.array(tpeak)*unit
        fwhm = np.array(fwhm)*unit
        self.peaks = tpeak
        self.fwhms = fwhm

    def mask(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr]):
        """
        Create a boolean mask to indicate which flares are visible within a certain time period.

        Parameters
        ----------
        tstart : astropy.units.Quantity 
            Starting time.
        tfinish : astropy.units.Quantity 
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
        tstart : astropy.units.Quantity 
            Starting time.
        tfinish : astropy.units.Quantity 
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
        tstart : astropy.units.Quantity 
            Starting time.
        tfinish : astropy.units.Quantity 
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
        tstart : astropy.units.Quantity 
            Starting time.
        tfinish : astropy.units.Quantity 
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
