import numpy as np
from astropy import units as u, constants as c
from scipy.optimize import newton
import pandas as pd

def to_float(quant,unit):
    return (quant/unit).to(u.Unit('')).value

class SystemGeometry:
    """System Geometry

    Class to store and calculate information on the geometry of a star-planet-observer system

    Args:
        inclination (astropy.units.quantity.Quantity [angle]): Defined the same as in PSG. Transit is i=90 deg
        init_stellar_lon (astropy.units.quantity.Quantity [angle]): sub-observer longitude at the beginning of observation
        init_phase_planet (astropy.units.quantity.Quantity [angle]): planet phase at beginning of observation
        stellar_period (astropy.units.quantity.Quantity [time]): rotational period of the star
        orbital_period (astropy.units.quantity.Quantity [time]): orbital period of the planet
        planetary_rot_period (astropy.units.quantity.Quantity [time]): rotational period of the planet
        stellar_offset_amp (astropy.units.quantity.Quantity [angle]): offset between stellar rotation axis and normal to orbital plane
        stellar_offset_phase (astropy.units.quantity.Quantity [angle]): direction of stellar offset, 0 defined as facing observer. Right hand direction is positive
        eccentricity (float): orbital eccentricity of the planet
        argument_of_pariapsis (astropy.units.quantity.Quantity [angle]): Angle between the observer and the point of pariapsis
    
    Returns:
        None
    """
    def __init__(self,inclination=0*u.deg,
                    init_stellar_lon = 0*u.deg,
                    init_planet_phase = 0*u.deg,
                    stellar_period = 80*u.day,
                    orbital_period = 11*u.day,
                    planetary_rot_period = 11*u.day,
                    stellar_offset_amp = 0*u.deg,
                    stellar_offset_phase = 0*u.deg,
                    eccentricity = 0,
                    argument_of_pariapsis = 0*u.deg):
        self.i = inclination
        self.init_stellar_lon = init_stellar_lon
        self.init_planet_phase = init_planet_phase
        self.stellar_period = stellar_period
        self.orbital_period = orbital_period
        self.planetary_rot_period = planetary_rot_period
        self.alpha = stellar_offset_amp
        self.beta = stellar_offset_phase
        self.e = eccentricity
        self.omega = argument_of_pariapsis
    
    def sub_obs(self,time):
        """sub-obs
        Get the coordinates of the sub-observer point

        Args:
            time (astropy.units.quantity.Quantity [time]): time since start of observations
        
        Returns:
            (dict): Coordinates in the form {'lat':lat,'lon':lon}
        """
        lon = self.init_stellar_lon + time *360*u.deg/self.stellar_period + self.beta
        lat = 90*u.deg - self.i + self.alpha*np.cos(self.beta)
        return {'lat':lat,'lon':lon}

    def mean_motion(self):
        """mean motion
        Get the mean motion of the orbit

        Args:
            None

        Returns:
            (astropy.units.quantity.Quantity [angular frequency]): the mean motion of the orbit
        """
        return 360*u.deg / self.orbital_period
    def mean_anomaly(self, time):
        """mean anomaly
        Get the mean anomaly of the orbit at a given time

        Args:
            time (astropy.units.quantity.Quantity [time]): time since periasteron

        Returns:
            (astropy.units.quantity.Quantity [angle]): the mean anomaly
        """
        return (time * self.mean_motion()) % (360*u.deg)
    def eccentric_anomaly(self,time):
        """Eccentric Anomaly
        Calculate the eccentric anomaly of the system
        Args:
            time (astropy.units.quantity.Quantity [time]): time since periasteron
        
        Returns:
            (astropy.units.quantity.Quantity [angle]): the eccentric anomaly
        """
        M = self.mean_anomaly(time)
        def func(E):
            return to_float(M,u.rad) - to_float(E*u.deg,u.rad) + self.e*np.sin(to_float(E*u.deg,u.rad))
        E = newton(func,x0=30)*u.deg
        return E
    def true_anomaly(self,time):
        """true anomaly
        Calculate the true anomaly.

        Args:
            time (astropy.units.quantity.Quantity [time]): time since periasteron
        
        Returns:
            (astropy.units.quantity.Quantity [angle]): the true anomaly
        """
        E = self.eccentric_anomaly(time) % (360*u.deg)
        if np.abs((180*u.deg - E)/u.deg) < 0.1:
            return E
        elif np.abs((0*u.deg - E)/u.deg) < 0.1:
            return E
        elif np.abs((360*u.deg - E)/u.deg) < 0.1:
            return E
        else:
            nu0 = 180*u.deg - (180*u.deg-E)*0.1

            def func(nu):
                eq = (1-self.e) * np.tan(nu*u.deg/2)**2 - (1+self.e) * np.tan(E/2)**2
                return eq
            nu = newton(func,x0=to_float(nu0,u.deg))*u.deg
            return nu


    def phase(self,time):
        """phase
        Calculate the phase at a given time

        Args:
            time (astropy.units.quantity.Quantity [time]): time since periasteron
        
        Returns:
            (astropy.units.quantity.Quantity [angle]): the phase
        """
        return (self.true_anomaly(time) + self.omega + 90*u.deg) % (360*u.deg)

    def sub_planet(self,time,phase = None):
        """sub-planet point
        Get the coordinates of the sub-planet point

        Args:
            time (astropy.units.quantity.Quantity [time]): time since periasteron
            phase = None (astropy.units.quantity.Quantity [angle]): if known, the current phase,
                otherwise the phase is calculated based on `time`
            
        Returns:
            (dict): Coordinates in the form {'lat':lat,'lon':lon}
        """
        sub_obs = self.sub_obs(time)
        if isinstance(phase,type(None)):
            phase = self.phase(time)
        lon = sub_obs['lon'] + phase - 90*u.deg + self.beta
        lat = -1*self.alpha * np.cos(self.beta + phase)
        return {'lat':lat,'lon':lon}

    def get_time_since_periasteron(self,phase):
        """get time since periasteron
        Calculate the time since the last periasteron for a given phase

        Args:
            phase (astropy.units.quantity.Quantity [angle]): current phase of the planet
        
        Returns:
            (astropy.units.quantity.Quantity [time]): time since periasteron
        """
        true_anomaly = phase - 90*u.deg - self.omega
        true_anomaly = true_anomaly % (360*u.deg)
        guess = true_anomaly/360/u.deg * self.orbital_period
        def func(guess):
            val = (self.true_anomaly(guess*u.day) - true_anomaly).to(u.rad).value
            return val
        time = newton(func,(guess/u.day).to(u.Unit(''))) * u.day
        return time.to(u.day)
    
    def get_observation_plan(self, phase0,total_time, time_step = None, N_obs = 10):
        """get observation plan
        Calculate information describing the state of the system for a series of observations

        Args:
            phase0 (astropy.units.quantity.Quantity [angle]): initial phase of the planet
            total_time (astropy.units.quantity.Quantity [time]): time over which the full observation is carried out
            time_step = None (astropy.units.quantity.Quantity [time]): step between each epoch of observation
            N_obs (int): number of epochs to observe
        
        Returns:
            (dict): dict where values are Quantity array objects giving the state of the system at each epoch
        """
        if isinstance(time_step, type(None)):
            N_obs = int(N_obs)
        else:
            N_obs = int(total_time/time_step)
        t0 = self.get_time_since_periasteron(phase0)
        start_times = np.linspace(to_float(t0,u.s),to_float(t0+total_time,u.s),N_obs,endpoint=False)*u.s
        phases = []
        sub_obs_lats = []
        sub_obs_lons = []
        sub_planet_lats = []
        sub_planet_lons = []
        u_angle = u.deg
        for time in start_times:
            phase = to_float(self.phase(time),u_angle) #% (360*u.deg)
            phases.append(phase)
            sub_obs = self.sub_obs(time)
            sub_obs_lats.append(to_float(sub_obs['lat'],u_angle))
            sub_obs_lons.append(to_float(sub_obs['lon'],u_angle))
            sub_planet = self.sub_planet(time,phase=phase*u_angle)
            sub_planet_lats.append(to_float(sub_planet['lat'],u_angle))
            sub_planet_lons.append(to_float(sub_planet['lon'],u_angle))
        return {'time':start_times,
                            'phase':phases*u_angle,
                            'sub_obs_lat':sub_obs_lats*u_angle,
                            'sub_obs_lon': sub_obs_lons*u_angle,
                            'sub_planet_lat': sub_planet_lats*u_angle,
                            'sub_planet_lon': sub_planet_lons*u_angle}