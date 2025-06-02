"""
Two-faced planet model

"""

from typing import Tuple
import numpy as np
from astropy import units as u
import libpypsg as psg


COS2 = 'cos2'
DISCONTINUOUS = 'discont'

def _clean_interp(phis: u.Quantity, thetas: u.Quantity, p0d: u.Quantity, q0d: u.Quantity):
    """
    Get various metadata to use before and after interpolation.
    """
    pressure_unit = p0d.unit
    quant_unit = q0d.unit
    if not isinstance(phis, u.Quantity):
        phis = phis*u.rad
    if not isinstance(thetas, u.Quantity):
        thetas = thetas*u.rad
    angle_unit = phis.unit
    phi_arr, theta_arr = np.meshgrid(
        phis.to_value(angle_unit),
        thetas.to_value(angle_unit),
        indexing='ij'
    )
    return pressure_unit, quant_unit, angle_unit, phi_arr, theta_arr


def _do_interp(
    day_contrib: np.ndarray,
    night_contrib: np.ndarray,
    p0d: u.Quantity,
    p1d: u.Quantity,
    q0d: u.Quantity,
    q1d: u.Quantity,
    p0n: u.Quantity,
    p1n: u.Quantity,
    q0n: u.Quantity,
    q1n: u.Quantity,
    n_linear: int,
    n_const: int,
    p_top: u.Quantity,
    pressure_unit: u.Unit,
    quant_unit: u.Unit
) -> Tuple[u.Quantity, u.Quantity]:
    p0 = day_contrib*p0d.to_value(pressure_unit) + \
        night_contrib*p0n.to_value(pressure_unit)
    p1 = day_contrib*p1d.to_value(pressure_unit) + \
        night_contrib*p1n.to_value(pressure_unit)
    q0 = day_contrib*q0d.to_value(quant_unit) + \
        night_contrib*q0n.to_value(quant_unit)
    q1 = day_contrib*q1d.to_value(quant_unit) + \
        night_contrib*q1n.to_value(quant_unit)

    ln_pressures = np.linspace(0, np.log(p1/p0), n_linear, endpoint=False)
    ln_pressures = np.vstack([ln_pressures, np.linspace(np.log(
        p1/p0), np.log(p_top.to_value(pressure_unit)/p0), n_const, endpoint=True)])

    a = q0
    b = (q0-q1)/np.log(p1/p0)

    return np.exp(ln_pressures)*p0*pressure_unit, quant_unit*np.where(
        ln_pressures < np.log(p1/p0),
        q1,
        (a - b*ln_pressures)
    )


def interp_quantity_cos2(
    phis: u.Quantity,
    thetas: u.Quantity,
    p0d: u.Quantity,
    p1d: u.Quantity,
    q0d: u.Quantity,
    q1d: u.Quantity,
    p0n: u.Quantity,
    p1n: u.Quantity,
    q0n: u.Quantity,
    q1n: u.Quantity,
    n_linear: int,
    n_const: int,
    p_top: u.Quantity
) -> Tuple[u.Quantity, u.Quantity]:
    """
    Interpolate a quantity smoothly using the cosine squared of twice its arclength from the
    substellar point. This is quite computationally efficient as we use some clever
    trigonometry.

    Parameters
    ----------
    phis : astropy.units.Quantity (N,)
        Azimuthal angles
    thetas : astropy.units.Quantity (M,)
        Polar angles
    p0d : astropy.units.Quantity
        Pressure at :math:`P_0` on the dayside
    p1d : astropy.units.Quantity
        Pressure at :math:`P_1` on the dayside
    q0d : astropy.units.Quantity
        Quantity at :math:`P_0` on the dayside
    q1d : astropy.units.Quantity
        Quantity at :math:`P_1` on the dayside
    p0n : astropy.units.Quantity
        Pressure at :math:`P_0` on the nightside
    p1n : astropy.units.Quantity
        Pressure at :math:`P_1` on the nightside
    q0n : astropy.units.Quantity
        Quantity at :math:`P_0` on the nightside
    q1n : astropy.units.Quantity
        Quantity at :math:`P_1` on the nightside
    n_linear : int
        Number of pressure cells in the linear region
    n_const : int
        Number of pressure cells in the constant region
    p_top : astropy.units.Quantity
        Pressure at the top of the atmosphere

    Returns
    -------
    pressure : astropy.units.Quantity (n_linear+n_const, N, M)
        The pressure interpolated across the atmosphere.
    astropy.units.Quantity (n_linear+n_const, N, M)
        The quantity interpolated across the atmosphere.
    """
    pressure_unit, quant_unit, _, phi_arr, theta_arr = _clean_interp(
        phis, thetas, p0d, q0d)
    day_contrib = 0.5*(1 + np.sin(theta_arr)*np.cos(phi_arr))
    night_contrib = 0.5*(1 - np.sin(theta_arr)*np.cos(phi_arr))

    return _do_interp(
        day_contrib, night_contrib,
        p0d, p1d, q0d, q1d,
        p0n, p1n, q0n, q1n,
        n_linear, n_const, p_top,
        pressure_unit, quant_unit
    )


def interp_quantity_discontinuous(
    phis: u.Quantity,
    thetas: u.Quantity,
    p0d: u.Quantity,
    p1d: u.Quantity,
    q0d: u.Quantity,
    q1d: u.Quantity,
    p0n: u.Quantity,
    p1n: u.Quantity,
    q0n: u.Quantity,
    q1n: u.Quantity,
    n_linear: int,
    n_const: int,
    p_top: u.Quantity
) -> Tuple[u.Quantity, u.Quantity]:
    """
    Interpolate a quantity but with a sharp discontinuity at the terminator

    Parameters
    ----------
    phis : astropy.units.Quantity (N,)
        Azimuthal angles
    thetas : astropy.units.Quantity (M,)
        Polar angles
    p0d : astropy.units.Quantity
        Pressure at :math:`P_0` on the dayside
    p1d : astropy.units.Quantity
        Pressure at :math:`P_1` on the dayside
    q0d : astropy.units.Quantity
        Quantity at :math:`P_0` on the dayside
    q1d : astropy.units.Quantity
        Quantity at :math:`P_1` on the dayside
    p0n : astropy.units.Quantity
        Pressure at :math:`P_0` on the nightside
    p1n : astropy.units.Quantity
        Pressure at :math:`P_1` on the nightside
    q0n : astropy.units.Quantity
        Quantity at :math:`P_0` on the nightside
    q1n : astropy.units.Quantity
        Quantity at :math:`P_1` on the nightside
    n_linear : int
        Number of pressure cells in the linear region
    n_const : int
        Number of pressure cells in the constant region
    p_top : astropy.units.Quantity
        Pressure at the top of the atmosphere

    Returns
    -------
    pressure : astropy.units.Quantity (n_linear+n_const, N, M)
        The pressure interpolated across the atmosphere.

    q : astropy.units.Quantity (n_linear+n_const, N, M)
        The quantity interpolated across the atmosphere.
    """
    pressure_unit, quant_unit, _, phi_arr, theta_arr = _clean_interp(
        phis, thetas, p0d, q0d)
    cos_alpha = np.sin(theta_arr)*np.cos(phi_arr)
    day_contrib = np.where(
        cos_alpha > 0,
        1,
        np.where(cos_alpha == 0, 0.5, 0)
    )
    night_contrib = 1 - day_contrib

    return _do_interp(
        day_contrib, night_contrib,
        p0d, p1d, q0d, q1d,
        p0n, p1n, q0n, q1n,
        n_linear, n_const, p_top,
        pressure_unit, quant_unit
    )





def gen_planet(
    p0d: u.Quantity,
    p1d: u.Quantity,
    t0d: u.Quantity,
    t1d: u.Quantity,
    p0n: u.Quantity,
    p1n: u.Quantity,
    t0n: u.Quantity,
    t1n: u.Quantity,
    n_linear: int,
    n_const: int,
    p_top: u.Quantity,
    nphi: int,
    ntheta: int,
    scheme: str,
    h2o_d0: u.Quantity,
    h2o_d1: u.Quantity,
    h2o_n0: u.Quantity,
    h2o_n1: u.Quantity,
    co2: u.Quantity,
    o3: u.Quantity,
    no2: u.Quantity,
    albedo: float
) -> psg.globes.PyGCM:
    """
    Generate a planet using the two-face model.

    Parameters
    ----------
    p0d : astropy.units.Quantity
        Pressure at :math:`P_0` on the dayside
    p1d : astropy.units.Quantity
        Pressure at :math:`P_1` on the dayside
    t0d : astropy.units.Quantity
        Temperature at :math:`T_0` on the dayside
    t1d : astropy.units.Quantity
        Temperature at :math:`T_1` on the dayside
    p0n : astropy.units.Quantity
        Pressure at :math:`P_0` on the nightside
    p1n : astropy.units.Quantity
        Pressure at :math:`P_1` on the nightside
    t0n : astropy.units.Quantity
        Temperature at :math:`T_0` on the nightside
    t1n : astropy.units.Quantity
        Temperature at :math:`T_1` on the nightside
    n_linear : int
        Number of linear regime grid points.
    n_const : int
        Number of constant regime grid points.
    p_top : astropy.units.Quantity
        Pressure at the top of the atmosphere.
    nphi : int
        Number of azimuthal grid points.
    ntheta : int
        Number of polar grid points.
    scheme : str
        Interpolation scheme to use.
    h2o_d0 : astropy.units.Quantity
        H2O vmr abundance on the dayside at :math:`P_0`
    h2o_d1 : astropy.units.Quantity
        H2O vmr abundance on the dayside at :math:`P_1`
    h2o_n0 : astropy.units.Quantity
        H2O vmr abundance on the nightside at :math:`P_0`
    h2o_n1 : astropy.units.Quantity
        H2O vmr abundance on the nightside at :math:`P_1`
    co2 : astropy.units.Quantity
        CO2 vmr abundance
    o3 : astropy.units.Quantity
        Ozone vmr abundance
    no2 : astropy.units.Quantity
        Nitrogen dioxide vmr abundance
    albedo : float
        Albedo value.

    Returns
    -------
    psg.globes.PyGCM
        The PyGCM object.
    """
    shape3d = (n_linear+n_const, nphi, ntheta)
    phis = np.linspace(-np.pi, np.pi, nphi, endpoint=True)*u.rad
    thetas = np.linspace(0, np.pi, ntheta, endpoint=True)*u.rad
    if scheme == COS2:
        interpolator = interp_quantity_cos2
    elif scheme == DISCONTINUOUS:
        interpolator = interp_quantity_discontinuous
    else:
        raise ValueError(f'Unknown interpolation scheme: {scheme}. \
            Allowed values are {COS2} and {DISCONTINUOUS}.')
    pressure, temperature = interpolator(
        phis, thetas,
        p0d, p1d, t0d, t1d,
        p0n, p1n, t0n, t1n,
        n_linear, n_const, p_top
    )
    if pressure.shape != shape3d:
        raise ValueError(f'Pressure shape {pressure.shape} \
            does not match expected shape {shape3d}.')
    if temperature.shape != shape3d:
        raise ValueError(f'Temperature shape {temperature.shape} \
            does not match expected shape {shape3d}.')
    _, h2o = interpolator(
        phis, thetas,
        p0d, p1d, h2o_d0, h2o_d1,
        p0n, p1n, h2o_n0, h2o_n1,
        n_linear, n_const, p_top
    )
    if h2o.shape != shape3d:
        raise ValueError(f'H2O shape {h2o.shape} \
            does not match expected shape {shape3d}.')

    press = psg.globes.structure.Pressure(pressure)
    psurf = psg.globes.structure.SurfacePressure.from_pressure(press)
    temp = psg.globes.structure.Temperature(temperature)
    tsurf = psg.globes.structure.SurfaceTemperature(dat=temp.dat[0, :, :])
    h2o = psg.globes.structure.Molecule('H2O', h2o)
    co2 = psg.globes.structure.Molecule.constant('CO2', co2, shape=h2o.shape)
    o3 = psg.globes.structure.Molecule.constant('O3', o3, shape=h2o.shape)
    no2 = psg.globes.structure.Molecule.constant('NO2', no2, shape=h2o.shape)
    albedo = psg.globes.structure.Albedo.constant(albedo, shape=h2o.shape[1:])
    _wind_u = psg.globes.structure.Wind.zero('wind_u', h2o.shape)
    _wind_v = psg.globes.structure.Wind.zero('wind_v', h2o.shape)

    return psg.globes.PyGCM(
        press, temp,
        h2o,
        co2, o3, no2,
        wind_u=_wind_u, wind_v=_wind_v,
        tsurf=tsurf, psurf=psurf,
        albedo=albedo,
        lon_start=-180.,
        lat_start=-90.
    )
