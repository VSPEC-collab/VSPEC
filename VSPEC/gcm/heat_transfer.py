"""
Physics of heat transport


Based on :cite:t:`2011ApJ...726...82C`

"""
from astropy import units as u, constants as c
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import warnings
from copy import deepcopy
from scipy.interpolate import interp1d


def get_flux(
    teff_star:u.Quantity,
    r_star:u.Quantity,
    r_orbit:u.Quantity
)->u.Quantity:
    """
    Get the flux of a star at a certain orbital radius

    Parameters
    ----------
    teff_star : astropy.units.Quantity
        The effective temperature of the star.
    r_star : astropy.units.Quantity
        The radius of the star.
    r_orbit : astropy.units.Quantity
        The orbital radius.
    
    Returns
    -------
    flux : astropy.units.Quantity
        The flux of the star at the planet.
    
    """
    # pylint: disable-next:no-member
    flux = c.sigma_sb * teff_star**4 * (r_star/r_orbit)**2
    return flux.to(u.Unit('W m-2'))

def get_psi(lon:u.Quantity):
    """
    Translate between longitude and Psi.
    
    Since the math we are doing for the energy balance requires
    longituide on [-180,180] deg, we cast everything to this range.
    
    Parameters
    ----------
    lon : astropy.units.Quantity
        The longituide to cast.
    
    Returns
    -------
    np.ndarray
        The longitude coordinate in radians cast to [-pi,pi]
    """
    unit = u.deg
    alon = np.atleast_1d(lon.to_value(unit))
    return (np.where(alon<=180,alon,alon-360)*unit).to_value(u.rad)

def pcos(x:u.Quantity):
    """
    'Positive cosine' -- :math:`\\max{\\cos{x},0}`

    Parameters
    ----------
    x : astropy.units.Quantity
        The angle to take the cosine of.
    
    Returns
    -------
    np.ndarray
        The maximum of 0 and cosine `x`.
    """
    cos = np.cos(x)
    if isinstance(cos,u.Quantity):
        cos = cos.to_value(u.dimensionless_unscaled)
    return np.where(cos>0,cos,0)

def colat(lat:u.Quantity):
    """
    Get the colatitude.

    Parameters
    ----------
    lat : astropy.units.Quantity
        The latitude.

    Returns
    -------
    colat : astropy.units.Quantity
        The colatitude.
    """
    return 90*u.deg - lat

def get_t0(
    star_teff:u.Quantity,
    albedo:float,
    r_star:u.Quantity,
    r_orbit:u.Quantity
):
    """
    Get the fiducial temperature for a planet's atmosphere, as
    defined by :cite:t:`2011ApJ...726...82C`, equation 4.

    Parameters
    ----------
    teff_star : astropy.units.Quantity
        The effective temperature of the star.
    albedo : float
        The Bond albedo of the planet.
    r_star : astropy.units.Quantity
        The radius of the star.
    r_orbit : astropy.units.Quantity
        The orbital radius.
    
    Returns
    -------
    t0: astropy.units.Quantity
        The fiducial temperature of the planet.
    """
    return (star_teff * (1-albedo)**0.25 * np.sqrt(r_star/r_orbit)).to(u.K)


def get_equator_curve(epsilon:float,n_points:int,mode:str='ivp_reflect'):
    """
    Get the temperature along the equator given thermal
    inertia `epsilon`. This is computed by integrating
    equation 10 of :cite:t:`2011ApJ...726...82C`.

    Parameters
    ----------
    epsilon : float
        The thermal inertial of the planet.
    n_points : int
        The numer of longitude points to return in the final array.
    mode : str
        The method to use to find the solution. This can be one of
        'ivp_reflect', 'bvp', 'ivp_iterate', 'analytic'.
    
    Returns
    -------
    lons : np.ndarray
        The longitude points in radians, starting at -pi
    tsurf : np.ndarray
        The unitless ratio between the surface temperature and the fiducial temperature.

    Warns
    -----
    RuntimeWarning
        If the specified `mode` is not valid with the specified `epsilon`.

    Notes
    -----
    Of the four solving methods, each has a region in which it is valid. Some are optimized for
    `epsilon` ~ 1, some for small `epsilon`, and some for when `epsilon` is very large.

    `ivp_reflect`: Integrate from pi to -pi, to find an initial condition to integrate
    from -pi to pi. For small `epsilon`, this method is very robust. Valid for `epsilon` < 1

    `bvp`: Solve the boundary condition problem for T(-pi) = T(pi). Valid for `epsilon` > 1

    `ivp_interate`: Integrate the ODE until the boundaries are within 1%. Valid for `epsilon` < 0.5

    `analytic`: Use an analytic approximation. Very fast, valid for `epsilon` > 10

    """
    def get_lons(n):
        return np.linspace(-np.pi,np.pi,n)
    def func(phi,T):
        return (pcos(phi) - T**4)/epsilon
    def minus_func(phi,T):
        return -(pcos(phi) - T**4)/epsilon
    if mode == 'ivp_reflect':
        if epsilon > 1:
            msg = 'Using method `ivp_reflect` with `epsilon` > 1. Energy balance may be invalid.'
            warnings.warn(msg,RuntimeWarning)
        lons = get_lons(n_points)
        y0 = np.atleast_1d(1)
        result = solve_ivp(minus_func,(np.pi,-np.pi),y0=y0)
        y_final = np.atleast_1d(np.mean(result.y[:,-1]))
        
        result = solve_ivp(func,(-np.pi,np.pi),y0=y_final,t_eval=(lons))
        if not result.success:
            raise ValueError(result.message)
        return result.t,result.y.T
    elif mode == 'bvp':
        if epsilon < 1:
            msg = 'Using method `bvp` with `epsilon` < 1. Energy balance may be invalid.'
            warnings.warn(msg,RuntimeWarning)
        lons = get_lons(n_points)
        def bfunc(phi,T):
            return np.vstack((func(phi,T)))
        def bc(ya,yb):
            return ya-yb
        y0 = np.atleast_2d(np.exp(-lons**2))
        result = solve_bvp(bfunc,bc,x=lons,y=y0)
        return result.x,result.y.T
    elif mode == 'ivp_iterate':
        if epsilon > 0.5:
            msg = 'Using method `ivp_iterate` with `epsilon` > 0.5. Energy balance may be invalid.'
            warnings.warn(msg,RuntimeWarning)
        lons = get_lons(n_points)
        y0 = np.atleast_1d(0.5)
        for _ in range(10):
            result = solve_ivp(func,(-np.pi,np.pi),y0=y0,t_eval=lons)
            phi = result.t
            T = result.y.T
            if np.abs(T[0]-T[-1])/T[-1] < 1e-2:
                return phi,T
        return phi,T
    elif mode == 'analytic':
        if epsilon < 10:
            msg = 'Using method `analytic` with `epsilon` < 10. Energy balance may be invalid.'
            warnings.warn(msg,RuntimeWarning)
        lons = get_lons(n_points)
        lon_prime = np.where(lons<-np.pi/2,lons+2*np.pi,lons)
        T_tild0 = np.pi**(-0.25)
        gamma = 4*T_tild0**3/epsilon
        T_tild_day = 3/4*T_tild0 + \
        (gamma*np.cos(lon_prime) + np.sin(lon_prime)) / (epsilon * (1+gamma**2)) + \
            np.exp(-gamma * lon_prime) / (2*epsilon*(1+gamma**2)*np.sinh(np.pi*gamma/2))
        T_tild_night = 3/4*T_tild0 + np.exp(-gamma*(lon_prime-np.pi)) / (2*epsilon*(1+gamma**2)*np.sinh(np.pi*gamma/2))
        day = (lon_prime >= -np.pi/2) & (lon_prime <= np.pi/2)
        T = deepcopy(T_tild_day)
        T[~day] = T_tild_night[~day]
        return lons,np.atleast_2d(T).T
    else:
        raise ValueError(f'Unknown mode: {mode}')

class TemperatureMap:
    """
    A map of the surface temperature.

    Parameters
    ----------
    epsilon : float
        The thermal inertia.
    t0 : astropy.units.Quantity
        The fiducial temperature.
    """
    min_temp = 40*u.K
    def __init__(
        self,
        epsilon,
        t0:u.Quantity
    ):
        n_points = 180
        if epsilon < 1:
            mode = 'ivp_reflect'
        elif epsilon < 10:
            mode = 'bvp'
        else:
            mode = 'analytic'

        phi,T = get_equator_curve(epsilon,n_points,mode)
        
        self.equator = interp1d(phi,T[:,0])
        self.epsilon = epsilon
        self.t0 = t0
    def eval(
        self,
        lon:u.Quantity,
        lat:u.Quantity
    )->u.Quantity:
        """
        Evaluate the temperature map at a point of points.

        Parameters
        ----------
        lon : astropy.units.Quantity
            The longitude of the points to evaluate.
        lat : astropy.units.Quantity
            The latitude of the points to evaluate.

        Returns
        -------
        astropy.units.Quantity
            The surface temperature at the desired points.
        """
        if isinstance(lon,u.Quantity):
            lon = get_psi(lon)
        tmap = self.equator(lon)*np.cos(lat)**0.25 * self.t0
        tmap:u.Quantity = np.where(tmap > self.min_temp,tmap,self.min_temp)
        return tmap
    @classmethod
    def from_planet(
        cls,
        epsilon:float,
        star_teff:u.Quantity,
        albedo:float,
        r_star:u.Quantity,
        r_orbit:u.Quantity
    ):
        """
        Generate a `TemperatureMap` given the properties of a planet.

        Parameters
        ----------
        epsilon : float
            The thermal inertia of the planet.
        star_teff : astropy.units.Quantity
            The effective temperature of the host star.
        albedo : float
            The Bond albedo of the planet.
        r_star : astropy.units.Quantity
            The radius of the host star.
        r_orbit : astropy.units.Quantity
            The planet's orbital radius.

        """
        t0 = get_t0(star_teff,albedo,r_star,r_orbit)
        return cls(epsilon,t0)

def validate_energy_balance(
    tmap:u.Quantity,
    lons:np.ndarray,
    lats:np.ndarray,
    star_teff:u.Quantity,
    albedo:float,
    r_star:u.Quantity,
    r_orbit:u.Quantity
)->bool:
    """
    Validate the energy balance.

    Parameters
    ----------
    tmap : astropy.units.Quantity
        The temperature map.
    lons : np.ndarray
        The longitude points of the map.
    lats : np.ndarray
        The latitude points of the map.
    star_teff : astropy.units.Quantity
        The effective temperature of the star.
    albedo : float
        The Bond albedo of the planet.
    r_star : astropy.units.Quantity
        The radius of the star.
    r_orbit : astropy.units.Quantity
        The orbital radius of the planet.

    Returns
    -------
    bool
        True if the energy is balanced to within 1%.
    """
    r_planet = 1*u.R_earth # for units. This will divide out later.
    incident_flux = get_flux(star_teff,r_star,r_orbit) # W/m2
    absorbed = (1-albedo) * incident_flux * np.pi * r_planet**2
    absorbed = absorbed.to(u.W)
    planet_area = 4*np.pi * r_planet**2
    mean_flux = (absorbed/planet_area).to(u.Unit('W m-2'))
    mean_T = ((mean_flux/c.sigma_sb)**0.25).to(u.K)

    llats,_ = np.meshgrid(lats,lons)
    jacobian = np.cos(llats)
    dlon = 2*np.pi/len(lons)
    dlat = np.pi/len(lats)
    integrand = tmap**4 * jacobian
    outbound_flux = (c.sigma_sb * dlon * dlat * np.sum(integrand)).to(u.Unit('W m-2'))
    eq_outbound_flux = np.sum(c.sigma_sb * mean_T**4 * jacobian * dlon*dlat)
    if np.abs(eq_outbound_flux-outbound_flux)/eq_outbound_flux < 1e-2:
        return True
    else:
        return False