import numpy as np
from astropy import units as u, constants as c
from scipy.optimize import newton

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

    def sub_planet(self,time):
        sub_obs = self.sub_obs(time)
        phase = self.phase(time)
        lon = sub_obs['lon'] + phase - 90*u.deg + self.beta
        lat = -1*self.alpha * np.cos(self.beta + phase)
        return {'lat':lat,'lon':lon}