"""VSPEC geometry module

This module stores and computes all of the geometry necessary
to simulate a `VSPEC` observation.
"""
import numpy as np
from astropy import units as u
from scipy.optimize import newton
from astropy.table import QTable

import matplotlib.pyplot as plt

class SystemGeometry:
    """System Geometry

    Class to store and calculate information on the geometry of a star-planet-observer system.

    Parameters
    ----------
    inclination : astropy.units.quantity.Quantity [angle], default=0 deg
        The inclination of the system, defined the same as in PSG. Transit is i=90 deg.
    init_stellar_lon : astropy.units.quantity.Quantity [angle], default=0 deg
        The sub-observer longitude at the beginning of observation.
    init_planet_phase : astropy.units.quantity.Quantity [angle], default=0 deg
        The planet phase at beginning of observation.
    stellar_period : astropy.units.quantity.Quantity [time], default=80 day
        The rotational period of the star.
    orbital_period : astropy.units.quantity.Quantity [time], default=11 day
        The orbital period of the planet.
    semimajor_axis : astropy.units.quantity.Quantity [distance], default=0.05 AU
        The semimajor axis of the planet's orbit.
    planetary_rot_period : astropy.units.quantity.Quantity [time], default=11 day
        The rotational period of the planet.
    planetary_init_substellar_lon : astropy.units.quantity.Quantity [angle], default=0 deg
        The sub-stellar longitude of the planet at the beginning of observation.
    stellar_offset_amp : astropy.units.quantity.Quantity [angle], default=0 deg
        The offset between stellar rotation axis and normal to orbital plane.
    stellar_offset_phase : astropy.units.quantity.Quantity [angle], default=0 deg
        The direction of the stellar offset. 0 is defined as facing the observer, and
        right hand direction is positive.
    eccentricity : float, default=0
        The orbital eccentricity of the planet.
    phase_of_periasteron : astropy.units.quantity.Quantity [angle], default=0 deg
        The phase at which the planet reaches periasteron.
    system_distance : astropy.units.quantity.Quantity [distance], default=1.3 pc
        The distance to the system.
    obliquity : astropy.units.quantity.Quantity [angle], default=0 deg
        The planet obliquity magnitude.
    obliquity_direction : astropy.units.quantity.Quantity [angle], default=0 deg
        The true anomaly at which the planet's north pole faces away from the star.


    Attributes
    ----------
    inclination : astropy.units.quantity.Quantity [angle]
        Inclination of the planet's orbit, defined as the angle between the line of
        sight and the orbital plane. Default is 0 deg.
    init_stellar_lon : astropy.units.quantity.Quantity [angle]
        Sub-observer longitude at the beginning of observation. Default is 0 deg.
    init_planet_phase : astropy.units.quantity.Quantity [angle]
        Phase of the planet at the beginning of observation. Default is 0 deg.
    stellar_period : astropy.units.quantity.Quantity [time]
        Rotational period of the star. Default is 80 day.
    orbital_period : astropy.units.quantity.Quantity [time]
        Orbital period of the planet. Default is 11 day.
    semimajor_axis : astropy.units.quantity.Quantity [length]
        Semimajor axis of the planet's orbit. Default is 0.05 AU.
    planetary_rot_period : astropy.units.quantity.Quantity [time]
        Rotational period of the planet. Default is 11 day.
    planetary_init_substellar_lon : astropy.units.quantity.Quantity [angle]
        Sub-stellar longitude of the planet at the beginning of observation. Default is 0 deg.
    alpha : astropy.units.quantity.Quantity [angle]
        Offset between the stellar rotation axis and normal to orbital plane. Default is 0 deg.
    beta : astropy.units.quantity.Quantity [angle]
        Direction of the stellar offset, defined as the angle between the line connecting
        the observer and the system barycenter and the projection of the stellar offset
        vector onto the plane perpendicular to the line of sight. Default is 0 deg.
    eccentricity : float
        Eccentricity of the planet's orbit. Default is 0.
    phase_of_periasteron : astropy.units.quantity.Quantity [angle]
        Phase at which the planet reaches periasteron. Default is 0 deg.
    system_distance : astropy.units.quantity.Quantity [length]
        Distance to the system. Default is 1.3 pc.
    obliquity : astropy.units.quantity.Quantity [angle]
        Obliquity of the planet, defined as the angle between the planet's equator
        and its orbital plane. Default is 0 deg.
    obliquity_direction : astropy.units.quantity.Quantity [angle]
        The true anomaly at which the planet's North pole faces away from the star.
        Default is 0 deg.
    init_time_since_periasteron : astropy.units.quantity.Quantity [angle]
        Time since periasteron at the beginning of observation. Default is the time
        since periasteron at `init_planet_phase`.
    init_true_anomaly : astropy.units.quantity.Quantity [angle]
        True anomaly at the beginning of observation, computed from `init_time_since_periasteron`.
    """

    def __init__(self, inclination=0*u.deg,
                 init_stellar_lon=0*u.deg,
                 init_planet_phase=0*u.deg,
                 stellar_period=80*u.day,
                 orbital_period=11*u.day,
                 semimajor_axis=0.05*u.AU,
                 planetary_rot_period=11*u.day,
                 planetary_init_substellar_lon=0*u.deg,
                 stellar_offset_amp=0*u.deg,
                 stellar_offset_phase=0*u.deg,
                 eccentricity=0,
                 phase_of_periasteron=0*u.deg,
                 system_distance=1.3*u.pc,
                 obliquity=0*u.deg,
                 obliquity_direction=0*u.deg):
        self.inclination = inclination
        self.init_stellar_lon = init_stellar_lon
        self.init_planet_phase = init_planet_phase
        self.stellar_period = stellar_period
        self.orbital_period = orbital_period
        self.semimajor_axis = semimajor_axis
        self.planetary_rot_period = planetary_rot_period
        self.planetary_init_substellar_lon = planetary_init_substellar_lon
        self.alpha = stellar_offset_amp
        self.beta = stellar_offset_phase
        self.eccentricity = eccentricity
        self.phase_of_periasteron = phase_of_periasteron
        self.system_distance = system_distance
        self.obliquity = obliquity
        self.obliquity_direction = obliquity_direction
        self.init_time_since_periasteron = self.get_time_since_periasteron(
            self.init_planet_phase)
        self.init_true_anomaly = self.true_anomaly(
            self.init_time_since_periasteron)

    def sub_obs(self, time):
        """
        Calculate the point on the stellar surface that is facing the observer.

        Parameters
        ----------
        time astropy.units.Quantity [time]
            The time elapsed since the observation began.

        Returns
        -------
        dict
            Coordinates of the sub-observer point in the form {'lat':lat,'lon':lon}
        """
        lon = self.init_stellar_lon - time * 360*u.deg/self.stellar_period
        lat = -1*(90*u.deg - self.inclination) + self.alpha*np.cos(self.beta)
        return {'lat': lat, 'lon': lon % (360*u.deg)}

    def mean_motion(self):
        """
        Get the mean motion of the planet's orbit.

        Returns
        -------
        astropy.units.Quantity [angular frequency]
            The mean motion of the orbit.
        """
        return 360*u.deg / self.orbital_period

    def mean_anomaly(self, time):
        """
        Get the mean anomaly of the orbit at a given time.

        Parameters
        ----------
        time : astropy.units.Quantity [time]
            The time elapsed since periasteron.

        Returns
        -------
        astropy.units.Quantity [angle]
            The mean anomaly.
        """
        return (time * self.mean_motion()) % (360*u.deg)

    def eccentric_anomaly(self, time):
        """
        Calculate the eccentric anomaly of the system
        at a given time

        Parameters
        ----------
        time : astropy.units.Quantity [time]
            The time elapsed since periasteron.

        Returns
        -------
        astropy.units.Quantity [angle]
            The eccentric anomaly.
        """
        mean_anom = self.mean_anomaly(time)

        def func(eccentric_anom: float):
            return mean_anom.to_value(u.rad) - (eccentric_anom*u.deg).to_value(u.rad) + self.eccentricity*np.sin((eccentric_anom*u.deg).to_value(u.rad))
        eccentric_anom = newton(func, x0=30)*u.deg
        return eccentric_anom

    def true_anomaly(self, time):
        """
        Calculate the true anomaly.

        Parameters
        ----------
        time : astropy.units.Quantity [time]
            The time elapsed since periasteron.

        Returns
        -------
        astropy.units.Quantity [angle]
            The true anomaly.
        """
        eccentric_anomaly = self.eccentric_anomaly(time) % (360*u.deg)
        if np.abs((180*u.deg - eccentric_anomaly)/u.deg) < 0.1:
            return eccentric_anomaly
        elif np.abs((0*u.deg - eccentric_anomaly)/u.deg) < 0.1:
            return eccentric_anomaly
        elif np.abs((360*u.deg - eccentric_anomaly)/u.deg) < 0.1:
            return eccentric_anomaly
        else:
            nu0 = 180*u.deg - (180*u.deg-eccentric_anomaly)*0.1

            def func(nu):
                eq = (1-self.eccentricity) * np.tan(nu*u.deg/2)**2 - \
                    (1+self.eccentricity) * np.tan(eccentric_anomaly/2)**2
                return eq
            nu = newton(func, x0=nu0.to_value(u.deg))*u.deg
            return nu

    def phase(self, time):
        """
        Calculate the phase at a given time.
        
        Phase is defined using the PSG convention.
        Transit occurs at phase=180 degrees.

        Parameters
        ----------
        time : astropy.units.Quantity [time]
            The time elapsed since periasteron.

        Returns
        -------
        astropy.units.Quantity [angle]
            The phase.
        """
        return (self.true_anomaly(time) + self.phase_of_periasteron) % (360*u.deg)

    def sub_planet(self, time, phase=None):
        """
        Get the coordinates of the sub-planet point on the star.

        Parameters
        ----------
        time : astropy.units.Quantity [time]
            The time elapsed since periasteron.
        phase : astropy.units.Quantity [angle], default=None
            The current phase, if known. Otherwise phase is calculated based on `time`.
            It is best practice to specify `phase` when possible to avoid using the Newtonian
            solver to calculate it.

        Returns
        -------
        dict
            Coordinates of the sub-planet point in the form {'lat':lat,'lon':lon}
        """
        sub_obs = self.sub_obs(time)
        if phase is None:
            phase = self.phase(time)
        lon = (sub_obs['lon'] + phase + 180*u.deg) % (360*u.deg)
        lat = -1*self.alpha * np.cos(self.beta + phase)
        return {'lat': lat, 'lon': lon}

    def get_time_since_periasteron(self, phase):
        """
        Calculate the time since the last periasteron for a given phase.
        
        This calculation is costly, so it is prefered that the user avoid
        calling this function more than necessary. The output is stored as
        `self.time_since_periasteron` upon initialization, so it can be
        accessed easily without recalculating (so long as no attributes change).

        Parameters
        ----------
        phase : astropy.units.Quantity [angle]
            The current phase of the planet.

        Returns
        -------
        astropy.units.Quantity [time]
            The time elapsed since periasteron.
        """
        true_anomaly = phase - self.phase_of_periasteron
        true_anomaly = true_anomaly % (360*u.deg)
        guess = true_anomaly/360/u.deg * self.orbital_period

        def func(guess):
            val = (self.true_anomaly(guess*u.day) -
                   true_anomaly).to(u.rad).value
            return val
        time = newton(func, (guess/u.day).to(u.Unit(''))) * u.day
        return time.to(u.day)

    def get_substellar_lon_at_periasteron(self) -> u.Quantity:
        """
        Compute the sub-stellar longitude at the previous periasteron
        given the rotation period, orbital period, and initial
        substellar longitude

        Returns
        -------
        astropy.units.Quantity [angle]
            The sub-stellar longitude at periasteron.
        """
        init_time_since_periasteron = self.get_time_since_periasteron(
            self.init_planet_phase)
        init_deg_rotated = 360*u.deg * init_time_since_periasteron/self.planetary_rot_period
        return self.planetary_init_substellar_lon + init_deg_rotated - self.init_true_anomaly

    def get_substellar_lon(self, time_since_periasteron) -> u.quantity.Quantity:
        """
        Calculate the sub-stellar longitude at a particular time since periasteron.

        Parameters
        ----------
        time_since_periasteron : astropy.units.Quantity [time]
            The time elapsed since periasteron.

        Returns
        -------
        astropy.units.Quantity [angle]
            The current sub-stellar longitude of the planet.

        Raises
        ------
        NotImplementedError
            If `self.obliquity` is not 0 deg.
        """
        substellar_lon_at_periasteron = self.get_substellar_lon_at_periasteron()
        deg_rotated = 360*u.deg * time_since_periasteron/self.planetary_rot_period
        true_anom = self.true_anomaly(time_since_periasteron)
        # how long since summer in N
        north_season = (true_anom - self.obliquity_direction) % (360*u.deg)
        if self.obliquity != 0*u.deg:
            raise NotImplementedError(
                'VSPEC does not currently support non-zero obliquity')
        lon = substellar_lon_at_periasteron + true_anom - \
            deg_rotated - self.obliquity*np.cos(north_season)
        # lon = self.planetary_init_substellar_lon - dphase + rotated
        # lon = self.planetary_init_substellar_lon + dphase - rotated
        return lon % (360.0*u.deg)

    def get_substellar_lat(self, phase: u.Quantity) -> u.Quantity:
        """
        Calculate the sub-stellar latitude of the planet at a particular phase.

        Parameters
        ----------
        phase : astropy.units.Quantity [angle]
            The current phase of the planet.

        Returns
        -------
        astropy.units.Quantity [angle]
            The current sub-stellar longitude of the planet.

        Raises
        ------
        NotImplementedError
            If `self.obliquity` is not 0 deg.
        """
        true_anomaly = phase - self.phase_of_periasteron
        north_season = (true_anomaly - self.obliquity_direction) % (360*u.deg)
        if self.obliquity != 0*u.deg:
            raise NotImplementedError(
                'VSPEC does not currently support non-zero obliquity')
        lat = 0*u.deg + self.obliquity*np.cos(north_season)
        return lat

    def get_pl_sub_obs_lon(self, time_since_periasteron: u.quantity.Quantity, phase: u.quantity.Quantity) -> u.quantity.Quantity:
        """
        Compute the sub-observer longitude of the planet.

        Parameters
        ----------
        time_since_periasteron : astropy.units.Quantity [time]
            The time elapsed since periasteron.
        phase : astropy.units.Quantity [angle]
            The current phase of the planet.

        Returns
        -------
        astropy.units.Quantity [angle]
            The current sub-observer longitude of the planet.
        """
        lon = self.get_substellar_lon(time_since_periasteron) - phase
        return lon

    def get_pl_sub_obs_lat(self, phase: u.Quantity) -> u.Quantity:
        """
        Compute the sub-observer latitude of the planet.

        Parameters
        ----------
        phase : astropy.units.Quantity [angle]
            The current phase of the planet.

        Returns
        -------
        astropy.units.Quantity [angle]
            The current sub-observer latitude of the planet.

        Raises
        ------
        NotImplementedError
            If `self.obliquity` is not 0 deg.
        """
        true_anomaly = phase - self.phase_of_periasteron
        north_season = (true_anomaly - self.obliquity_direction) % (360*u.deg)
        if self.obliquity != 0*u.deg:
            raise NotImplementedError(
                'VSPEC does not currently support non-zero obliquity')
        lat = 0*u.deg - self.obliquity * \
            np.cos(north_season) - (90*u.deg-self.inclination)
        return lat

    def get_radius_coeff(self, phase: u.quantity.Quantity) -> float:
        """
        Compute the orbital radius coefficient that depends on eccentricity and phase.

        Parameters
        ----------
        phase : astropy.units.Quantity [angle]
            The current phase of the planet.

        Returns
        -------
        float
            The orbital radius coefficient.
        """
        true_anomaly = phase - self.phase_of_periasteron
        num = 1 - self.eccentricity**2
        den = 1 + self.eccentricity*np.cos(true_anomaly)
        return (num/den).to_value(u.dimensionless_unscaled)

    def get_observation_plan(self,start_times: u.quantity.Quantity,
                             ) -> QTable:
        """
        Calculate information describing the state of the system
        for a series of observations.

        Parameters
        ----------
        start_times : astropy.units.Quantity [time]
            The time of each observation.

        Returns
        -------
        QTable
            The geometry of each observation.
        """
        start_times = start_times.to(u.s)
        
        phases = []
        sub_obs_lats = []
        sub_obs_lons = []
        sub_planet_lats = []
        sub_planet_lons = []
        sub_stellar_lons = []
        sub_stellar_lats = []
        pl_sub_obs_lons = []
        pl_sub_obs_lats = []
        orbit_radii = []
        u_angle = u.deg
        for time in start_times + self.init_time_since_periasteron:
            phase = self.phase(time).to_value(u.deg)  # % (360*u.deg)
            phases.append(phase)
            sub_obs = self.sub_obs(time)
            sub_obs_lats.append(sub_obs['lat'].to_value(u_angle))
            sub_obs_lons.append(sub_obs['lon'].to_value(u_angle))
            sub_planet = self.sub_planet(time, phase=phase*u_angle)
            sub_planet_lats.append(sub_planet['lat'].to_value(u_angle))
            sub_planet_lons.append(sub_planet['lon'].to_value(u_angle))
            sub_stellar_lon = self.get_substellar_lon(time)
            sub_stellar_lat = self.get_substellar_lat(phase*u_angle)
            sub_stellar_lons.append(sub_stellar_lon.to_value(u_angle))
            sub_stellar_lats.append(sub_stellar_lat.to_value(u_angle))
            pl_sub_obs_lon = self.get_pl_sub_obs_lon(time, phase*u_angle)
            pl_sub_obs_lat = self.get_pl_sub_obs_lat(phase*u_angle)
            pl_sub_obs_lons.append(pl_sub_obs_lon.to_value(u_angle))
            pl_sub_obs_lats.append(pl_sub_obs_lat.to_value(u_angle))
            orbit_rad = self.get_radius_coeff(phase*u_angle)
            orbit_radii.append(orbit_rad)
        return QTable(data={'time': start_times,
                'phase': phases*u_angle,
                'sub_obs_lat': sub_obs_lats*u_angle,
                'sub_obs_lon': sub_obs_lons*u_angle,
                'sub_planet_lat': sub_planet_lats*u_angle,
                'sub_planet_lon': sub_planet_lons*u_angle,
                'sub_stellar_lon': sub_stellar_lons*u_angle,
                'sub_stellar_lat': sub_stellar_lats*u_angle,
                'planet_sub_obs_lon': pl_sub_obs_lons*u_angle,
                'planet_sub_obs_lat': pl_sub_obs_lats*u_angle,
                'orbit_radius': orbit_radii})


    def get_system_visual(self,phase:u.Quantity,ax=None) -> plt.Axes:
        """
        Create a graphical representation of the host+planet geometry.

        Parameters
        ----------
        phase : astropy.units.Quantity
            The current phase of the planet.
        ax : matplotlib.axes.Axes, optional
            The axis to draw the figure to, by default None.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the graphics drawn.
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.scatter(0, 0, c='xkcd:tangerine', s=150)
        theta = np.linspace(0, 360, 180, endpoint=False)*u.deg
        r_dist = (1-self.eccentricity**2)/(1+self.eccentricity*np.cos(theta -
                                                                      self.phase_of_periasteron-90*u.deg))
        curr_theta = phase + 90*u.deg
        x_dist = self.semimajor_axis * np.cos(theta)*r_dist
        y_dist = -1*self.semimajor_axis * \
            np.sin(theta)*r_dist*np.cos(self.inclination)

        current_r = (1-self.eccentricity**2)/(1+self.eccentricity*np.cos(curr_theta -
                                                                         self.phase_of_periasteron - 90*u.deg))
        current_x_dist = self.semimajor_axis * np.cos(curr_theta)*current_r
        current_y_dist = -1*self.semimajor_axis * \
            np.sin(curr_theta)*current_r*np.cos(self.inclination)
        behind = np.sin(theta) >= 0
        x_angle = np.arctan(x_dist/self.system_distance).to(u.mas)
        y_angle = np.arctan(y_dist/self.system_distance).to(u.mas)
        plotlim = np.arctan(
            self.semimajor_axis/self.system_distance).to(u.mas).value * (1+self.eccentricity)*1.05
        current_x_angle = np.arctan(
            current_x_dist/self.system_distance).to(u.mas)
        current_y_angle = np.arctan(
            current_y_dist/self.system_distance).to(u.mas)
        z_order_mapper = {True: -99, False: 100}
        ax.plot(x_angle[behind], y_angle[behind],
                           zorder=-100, c='C0', alpha=1, ls=(0, (2, 2)))
        ax.plot(
            x_angle[~behind], y_angle[~behind], zorder=99, c='C0')
        ax.scatter(current_x_angle, current_y_angle,
                              zorder=z_order_mapper[np.sin(curr_theta) >= 0], c='k')

        ax.set_xlim(-plotlim, plotlim)
        ax.set_ylim(-plotlim, plotlim)
        ax.set_xlabel('sep (mas)')
        ax.set_ylabel('sep (mas)')
        return ax

    def get_planet_visual(self,phase:u.Quantity,ax=None):
        """
        Draw a visualization of the planet's geometry from the view
        of the observer.

        Parameters
        ----------
        phase : astropy.units.Quantity
            The current phase.
        ax : matplotlib.axes.Axes, optional
            The Axes to draw on, by default None

        Returns
        -------
        matplotlib.axes.Axes
            The drawn figure.
        """
        import cartopy.crs as ccrs
        from cartopy.geodesic import Geodesic
        if ax is None:
            ax = plt.gca()
        time_since_periasteron = self.get_time_since_periasteron(phase)
        substellar_lon = self.get_substellar_lon(time_since_periasteron)
        substellar_lat = self.get_substellar_lat(phase)
        
        ax.stock_img()
        lats = np.linspace(-90, 90, 181)*u.deg
        lons = np.linspace(0, 360, 181)*u.deg
        latgrid, longrid = np.meshgrid(lats, lons)
        latgrid = latgrid
        longrid = longrid
        cos_c = (np.sin(substellar_lat) * np.sin(latgrid)
                 + np.cos(substellar_lat) * np.cos(latgrid)
                 * np.cos(substellar_lon-longrid))
        dayside = (cos_c > 0).astype('int')

        # sub_stellar point
        rad_meters = 1e6
        circle_points = Geodesic().circle(lon=substellar_lon.to_value(u.deg), lat=substellar_lat.to_value(u.deg),
                                          radius=rad_meters, n_samples=200, endpoint=False)
        circ_lons = np.array(circle_points[:, 0])
        circ_lats = np.array(circle_points[:, 1])
        hemi_1 = circ_lons > 0
        ax.plot(circ_lons[hemi_1], circ_lats[hemi_1],
                            c='r', transform=ccrs.PlateCarree())
        ax.plot(circ_lons[~hemi_1], circ_lats[~hemi_1],
                            c='r', transform=ccrs.PlateCarree())

        # sub_stellar hemisphere
        rad_meters = (1*u.R_earth).to_value(u.m) * np.pi * 0.5 * 0.99
        circle_points = Geodesic().circle(lon=substellar_lon.to_value(u.deg), lat=substellar_lat.to_value(u.deg),
                                          radius=rad_meters, n_samples=200, endpoint=False)
        circ_lons = np.array(circle_points[:, 0])
        circ_lats = np.array(circle_points[:, 1])
        hemi_1 = circ_lons > 0
        ax.plot(circ_lons[hemi_1], circ_lats[hemi_1],
                            c='r', transform=ccrs.PlateCarree())
        ax.plot(circ_lons[~hemi_1], circ_lats[~hemi_1],
                            c='r', transform=ccrs.PlateCarree())

        ax.plot(
            lons, lons*0,
            transform=ccrs.PlateCarree(),
            c='k'
        )
        ax.plot(
            lons, lons*0+85*u.deg,
            transform=ccrs.PlateCarree(),
            c='C0'
        )
        ax.plot(
            lons, lons*0-85*u.deg,
            transform=ccrs.PlateCarree(),
            c='C1'
        )
        lats = np.linspace(-90, 90)
        ax.plot(lats*0, lats, transform=ccrs.PlateCarree(), c='C2')
        return ax
        
            


    def plot(self, phase: u.Quantity):
        """
        Create a plot of the geometry at a particular phase.

        Parameters
        ----------
        phase : astropy.units.Quantity [angle]
            The current phase of the planet.

        Returns
        -------
        matplotlib.figure.Figure
            A figure containing the plot.
        """
        # import cartopy. This way it is an optional dependencey
        import cartopy.crs as ccrs

        fig = plt.figure()
        axes = {}
        axes['orbit'] = fig.add_subplot(1, 2, 1)
        
        self.get_system_visual(phase,axes['orbit'])
        time_since_periasteron = self.get_time_since_periasteron(phase)
        subobs_lon = self.get_pl_sub_obs_lon(time_since_periasteron, phase)
        subobs_lat = self.get_pl_sub_obs_lat(phase)
        proj = ccrs.Orthographic(
            central_longitude=subobs_lon, central_latitude=subobs_lat)
        axes['planet'] = fig.add_subplot(1, 2, 2, projection=proj)
        self.get_planet_visual(phase,axes['planet'])
        
        return fig
