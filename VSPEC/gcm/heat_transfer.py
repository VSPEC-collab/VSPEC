"""
Physics of heat transport


Based on :cite:t:`2011ApJ...726...82C`

"""
from astropy import units as u, constants as c
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
# from scipy.optimize import fsolve
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
    flux = c.sigma_sb * teff_star**4 * (r_star/r_orbit)**2
    return flux.to(u.Unit('W m-2'))

def get_psi(lon:u.Quantity):
    """
    Translate between longitude and Psi
    """
    unit = u.deg
    alon = np.atleast_1d(lon.to_value(unit))
    return np.where(alon<=180,alon,alon-360)*unit

def pcos(x:u.Quantity):
    cos = np.cos(x)
    if isinstance(cos,u.Quantity):
        cos = cos.to_value(u.dimensionless_unscaled)
    return np.where(cos>0,cos,0)

def colat(lat:u.Quantity):
    return 90*u.deg - lat


# def get_Teq(
#     lon:u.Quantity,
#     lat:u.Quantity,
#     bond_albedo:float,
#     teff_star:u.Quantity,
#     r_star:u.Quantity,
#     r_orbit:u.Quantity
# ):
#     num = (1-bond_albedo) * get_flux(teff_star,r_star,r_orbit)*np.sin(colat(lat))*pcos(get_psi(lon))
#     return ((num/c.sigma_sb)**0.25).to(u.K)

def get_t0(
    star_teff:u.Quantity,
    albedo:float,
    r_star:u.Quantity,
    r_orbit:u.Quantity
):
    return (star_teff * (1-albedo)**0.25 * np.sqrt(r_star/r_orbit)).to(u.K)


def get_equillibrium_temp(
    star_teff:u.Quantity,
    albedo:float,
    r_star:u.Quantity,
    r_orbit:u.Quantity
):
    return (star_teff * (1-albedo)**0.25 * np.sqrt(r_star/2/r_orbit)).to(u.K)

def get_equator_curve(epsilon,n_points):
    """
    Best and most robust solver of the three.
    I have not found a value of epsilon that breaks this.
    """
    lons = np.linspace(-np.pi,np.pi,n_points)
    y0 = np.atleast_1d(1)
    def func(phi,T):
        return -(pcos(phi) - T**4)/epsilon
    result = solve_ivp(func,(np.pi,-np.pi),y0=y0)
    y_final = np.atleast_1d(np.mean(result.y[:,-1]))
    def func(phi,T):
        return (pcos(phi) - T**4)/epsilon
    result = solve_ivp(func,(-np.pi,np.pi),y0=y_final,t_eval=(lons))
    return result.t,result.y.T

def get_equator_curve2(epsilon,n_points):
    """
    breaks for small epsilon
    However, I think this is the 'correct' approach.
    """
    def func(phi,T):
        return np.vstack((
            (pcos(phi) - T**4)/epsilon
        ))
        # return (pcos(phi) - T**4)/epsilon
    def bc(ya,yb):
        return ya-yb
    x0 = np.linspace(-np.pi,np.pi,n_points)
    y0 = np.atleast_2d(np.exp(-x0**2))
    result = solve_bvp(func,bc,x=x0,y=y0)
    return result.x,result.y.T

def get_equator_curve3(epsilon,n_points,tol=1e-3,niter=10):
    """
    breaks for small epsilon
    """
    def func(phi, T):
        return (pcos(phi) - T**4) / epsilon
    lons = np.linspace(-np.pi,np.pi,n_points)
    y0 = np.atleast_1d(0.5)
    for _ in range(niter):
        result = solve_ivp(func,(-np.pi,np.pi),y0=y0,t_eval=lons)
        phi = result.t
        T = result.y.T
        if np.abs(T[0]-T[-1]) < tol:
            return phi,T
        y0 = T[-1]
    raise RuntimeError('No Convergence')
def get_equator_curve4(epsilon,n_points):
    """
    breaks for small epsilon
    """
    lons = np.linspace(-np.pi/2,3*np.pi/2,n_points)
    T_tild0 = np.pi**(-0.25)
    gamma = 4*T_tild0**3/epsilon
    T_tild_day = 3/4*T_tild0 + \
    (gamma*np.cos(lons) + np.sin(lons)) / (epsilon * (1+gamma**2)) + \
        np.exp(-gamma * lons) / (2*epsilon*(1+gamma**2)*np.sinh(np.pi*gamma/2))
    T_tild_night = 3/4*T_tild0 + np.exp(-gamma*(lons-np.pi)) / (2*epsilon*(1+gamma**2)*np.sinh(np.pi*gamma/2))
    day = (lons >= -np.pi/2) & (lons <= np.pi/2)
    T = deepcopy(T_tild_day)
    T[~day] = T_tild_night[~day]
    return lons,T

class TemperatureMap:
    """
    A map of the surface temperature.

    Parameters
    ----------
    epsilon : float
        The thermal inertia.
    t0 : u.Quantity
        The fiducial temperature.
    """
    min_temp = 40*u.K
    def __init__(
        self,
        epsilon,
        t0:u.Quantity
    ):
        n_points = 180
        if epsilon < 0.5:
            phi,T = get_equator_curve(epsilon,n_points)
        else:
            phi,T = get_equator_curve2(epsilon,n_points)
        self.equator = interp1d(phi,T[:,0])
        self.epsilon = epsilon
        self.t0 = t0
    def eval(self,lon,lat)->u.Quantity:
        tmap = self.equator(lon)*np.cos(lat)**0.25 * self.t0
        tmap = np.where(tmap > self.min_temp,tmap,self.min_temp)
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
):  
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