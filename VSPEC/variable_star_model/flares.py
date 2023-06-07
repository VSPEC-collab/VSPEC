"""VSPEC Flares module

This code governs the behavior of flares.
"""
from copy import deepcopy
from typing import List
import typing as Typing

import numpy as np
from astropy import units as u, constants as const
from astropy.units.quantity import Quantity

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
    Class to generate flare events and their characteristics.

    Parameters
    ----------
    stellar_teff : astropy.units.Quantity 
        Temperature of the star.
    stellar_rot_period : astropy.units.Quantity 
        Rotation period of the star.
    prob_following : float, default=0.25
        Probability of a flare being closely followed by another flare.
    mean_teff : astropy.units.Quantity , default=9000*u.K
        Mean temperature of the set of flares.
    sigma_teff : astropy.units.Quantity , default=500*u.K
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
    stellar_teff : astropy.units.Quantity 
        Temperature of the star.
    stellar_rot_period : astropy.units.Quantity 
        Rotation period of the star.
    prob_following : float
        Probability of a flare being closely followed by another flare.
    mean_teff : astropy.units.Quantity 
        Mean temperature of the set of flares.
    sigma_teff : astropy.units.Quantity 
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
        E : astropy.units.Quantity , shape=(M,)
            Energy coordinates at which to calculate frequencies.

        Returns
        -------
        freq : astropy.units.Quantity , shape=(M,)
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
        Es : astropy.units.Quantity , shape=(M,)
            An array of energy values to choose from.
        time : astropy.units.Quantity 
            The time duration over which the flare is generated.

        Returns
        -------
        E_final astropy.units.Quantity 
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
        Nexp = (self.powerlaw(Es) * time).to_value(u.dimensionless_unscaled)
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
        Es : astropy.units.Quantity , shape=(M,)
            The energies to draw from.
        time : astropy.units.Quantity 
            The time duration.

        Returns
        -------
        astropy.units.Quantity 
            Energies of generated flares.
        """
        unit = u.erg
        flare_energies = []
        E = self.get_flare(Es, time)
        if E == 0*u.erg:
            return flare_energies
        else:
            flare_energies.append(E.to_value(unit))
            cont = np.random.random() < self.prob_following
            while cont:
                while True:
                    E = self.get_flare(Es, time)
                    if E == 0*u.erg:
                        pass
                    else:
                        flare_energies.append(E.to_value(unit))
                        cont = np.random.random() < self.prob_following
                        break
            return flare_energies*unit

    def generate_coords(self):
        """
        Generate random coordinates for the flare.

        Returns
        -------
        astropy.units.Quantity 
            Latitude of the flare.
        astropy.units.Quantity 
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
        astropy.units.Quantity 
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
        astropy.units.Quantity 
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
        astropy.units.Quantity , shape=(self.log_E_erg_Nsteps,)
            A logarithmically-spaced series of energies.

        Notes
        -----
        It has been observed that the first 0.2 dex of the frequency distribution
        are treated differently by the energy assignement algorithm. We extend
        the energy range by 0.2 dex in order to clip it later. This is not a
        long-term fix.
        """
        return np.logspace(self.log_E_erg_min, self.log_E_erg_max, self.log_E_erg_Nsteps)*u.erg

    def generate_teff(self):
        """ 
        Randomly generate flare teff and rounds it to an integer.

        Returns
        -------
        astropy.units.Quantity 
            The effective temperature of a flare.

        Raises
        ------
        ValueError
            If `mean_teff` is less than or equal to 0 K.
        """
        if self.mean_teff <= 0*u.K:  # prevent looping forever if user gives bad parameters
            raise ValueError('Cannot have teff <= 0 K')
        # this cannot be a negative value. We will loop until we get something positive (usually unneccessary)
        while True:
            teff = np.random.normal(
                loc=self.mean_teff.to_value(u.K),
                scale=self.sigma_teff.to_value(u.K)
            )
            teff = int(np.round(teff)) * u.K
            if teff > 0*u.K:
                return teff

    def generate_flare_series(self, Es: Quantity, time: Quantity):
        """
        Generate as many flares within a duration of time as can be fit given computed frequencies.

        Parameters
        ----------
        Es : astropy.units.Quantity 
            A series of energies.
        time : astropy.units.Quantity 
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
                    for j, energy in enumerate(flare_energies):
                        if j > 0:
                            base_tpeak = base_tpeak + self.generate_flare_set_spacing()
                            peaks.append(deepcopy(base_tpeak))
                        lat, lon = self.generate_coords()
                        fwhm = self.generate_fwhm()
                        teff = self.generate_teff()
                        if np.log10(energy.to_value(u.erg)) >= self.log_E_erg_min:
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
