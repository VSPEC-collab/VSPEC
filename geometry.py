import numpy as np
from astropy import units as u, constants as c
from scipy.optimize import newton
import pandas as pd

def to_float(quant,unit):
    return (quant/unit).to(u.Unit('')).value

class SystemGeometry:
    """
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
        lon = self.init_stellar_lon + time *360*u.deg/self.stellar_period + self.beta
        lat = 90*u.deg - self.i + self.alpha*np.cos(self.beta)
        return {'lat':lat,'lon':lon}

    def mean_motion(self):
        return 360*u.deg / self.orbital_period
    def mean_anomaly(self, time):
        return time * self.mean_motion()
    def eccentric_anomaly(self,time):
        M = self.mean_anomaly(time)
        def func(E):
            return to_float(M,u.rad) - to_float(E*u.deg,u.rad) + self.e*np.sin(to_float(E*u.deg,u.rad))
        E = newton(func,x0=30)*u.deg
        return E
    def true_anomaly(self,time):
        E = self.eccentric_anomaly(time)
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
        """
        FUTURE: make this work correctly for e > 0
        """
        return self.true_anomaly(time) + self.omega + 90*u.deg

    def sub_planet(self,time,phase = None):
        sub_obs = self.sub_obs(time)
        if not phase:
            phase = self.phase(time)
        lon = sub_obs['lon'] + phase - 90*u.deg + self.beta
        lat = -1*self.alpha * np.cos(self.beta + phase)
        return {'lat':lat,'lon':lon}

    def get_time_since_periasteron(self,phase):
        true_anomaly = phase - 90*u.deg - self.omega
        true_anomaly = true_anomaly % (360*u.deg)
        guess = true_anomaly/360/u.deg * self.orbital_period
        def func(guess):
            val = (self.true_anomaly(guess*u.day) - true_anomaly).to(u.rad).value
            return val
        time = newton(func,(guess/u.day).to(u.Unit(''))) * u.day
        return time.to(u.day)
    
    def get_observation_plan(self, phase0,total_time, time_step = None, N_obs = 10):
        if time_step is None:
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
        for time in start_times:
            phase = self.phase(time) % (360*u.deg)
            phases.append(phase)
            sub_obs = self.sub_obs(time)
            sub_obs_lats.append(sub_obs['lat'])
            sub_obs_lons.append(sub_obs['lon'])
            sub_planet = self.sub_planet(time,phase=phase)
            sub_planet_lats.append(sub_planet['lat'])
            sub_planet_lons.append(sub_planet['lon'])
        return pd.DataFrame({'time':start_times,
                            'phase':phases,
                            'sub_obs_lat':sub_obs_lats,
                            'sub_obs_lon': sub_obs_lons,
                            'sub_planet_lat': sub_planet_lats,
                            'sub_planet_lon': sub_planet_lons})