"""
Helpers for math
"""

import numpy as np
from astropy import units as u


def get_angle_between(
    lat1: u.Quantity,
    lon1: u.Quantity,
    lat2: u.Quantity,
    lon2: u.Quantity
):
    """
    Compute the angle between two coordintates in
    lat/lon space.

    Parameters
    ----------
    lat1 : astropy.units.Quantity
        The latitude of the first coordinate
    lon1 : astropy.units.Quantity
        The longitude of the first coordinate
    lat2 : astropy.units.Quantity
        The latitude of the second coordinate
    lon2 : astropy.units.Quantity
        The longitude of the second coordinate

    Returns
    -------
    alpha : astropy.units.Quantity
        The angle between the coordinates.

    Notes
    -----
    This function computes the angle between two coordinates specified in latitude and longitude space using the spherical law of cosines:

    .. math::
        \\mu = \\sin{\\phi_1} \\sin{\\phi_2} + \\cos{\\phi_1} \\cos{\\phi_2} \\cos{(\\lambda_1 - \\lambda_2)}

    where :math:`(\\phi_1, \\lambda_1)` and :math:`(\\phi_2, \\lambda_2)` are the latitude and longitude pairs of the two coordinates, respectively. The angle between the coordinates is then computed as:

    .. math::
        \\alpha = \\arccos{\\mu}

    The computed angle is returned as an `astropy.units.Quantity` object.

    Examples
    --------
    >>> lat1 = 30 * u.deg
    >>> lon1 = 45 * u.deg
    >>> lat2 = 40 * u.deg
    >>> lon2 = 60 * u.deg
    >>> angle = get_angle_between(lat1, lon1, lat2, lon2)
    >>> print(angle)
    0.27581902582503454 rad

    """
    mu = (np.sin(lat1) * np.sin(lat2)
          + np.cos(lat1) * np.cos(lat2)
          * np.cos(lon1-lon2))
    return np.arccos(mu)


def proj_ortho(
    lat0: u.Quantity,
    lon0: u.Quantity,
    lats: u.Quantity,
    lons: u.Quantity
):
    """
    Use the spherical law of cosines to project each
    lat/lon point onto the x-y plane.

    Parameters
    ----------
    lat0 : astropy.units.quantity
        The latitude of the central point.
    lon0 : astropy.units.quantity
        The longitude of the central point.
    lats : astropy.units.quantity
        The latitude of the other points.
    lons : astropy.units.quantity
        The longitude of the other points.

    Returns
    -------
    x : np.ndarray
        The x coordinates of the points in the
        orthographic projection
    y : np.ndarray
        The y coordinates of the points in the
        orthographic projection

    Notes
    -----
    This function uses the spherical law of cosines to project each lat/lon point
    onto the x-y plane of an orthographic projection.

    Spherical law of cosines:

    .. math::

        \\cos{c} = \\cos{a} \\cos{b} + \\sin{a} \\sin{b} \\cos{C}

    We want to find :math:`C`.
        - :math:`b` is the colatitude of the central point.
        - :math:`c` is the colatitude of the other points.
        - :math:`a` is the distance of each point from the center (i.e. :math:`\\cos{\\mu}`)
    This can be difficult for special cases, but this code attempts
    to account for those.

    Examples
    --------
    >>> lat0 = 30 * u.deg
    >>> lon0 = 45 * u.deg
    >>> lats = [40, 50, 60] * u.deg
    >>> lons = [60, 70, 80] * u.deg
    >>> x, y = proj_ortho(lat0, lon0, lats, lons)
    >>> print(x)
    [0.19825408 0.27164737 0.28682206]
    >>> print(y)
    [0.18671294 0.37213692 0.54519419]

    """
    if not isinstance(lon0, u.Quantity):
        raise TypeError('`lon0` must be a Quantity')
    if not isinstance(lat0, u.Quantity):
        raise TypeError('`lat0` must be a Quantity')
    if not isinstance(lons, u.Quantity):
        raise TypeError('`lons` must be a Quantity')
    if not isinstance(lats, u.Quantity):
        raise TypeError('`lats` must be a Quantity')
    # Cast all to radians
    lon0: float = lon0.to_value(u.rad)
    lat0: float = lat0.to_value(u.rad)
    lons: np.ndarray = lons.to_value(u.rad)
    lats: np.ndarray = lats.to_value(u.rad)
    b = np.pi/2 - lat0
    c = np.pi/2 - lats
    if b == 0:  # centered on north pole
        a = c
        C = np.where(c <= np.pi/2, np.pi-lons, np.nan)
    elif b == np.pi:  # centered on south pole
        a = c
        C = np.where(c >= np.pi/2, lons, np.nan)
    else:  # centered elsewhere
        mu = (np.sin(lat0) * np.sin(lats)
              + np.cos(lat0) * np.cos(lats)
              * np.cos(lon0-lons))
        a = np.arccos(mu)
        cos_C = np.where(
            mu < 1,
            (np.cos(c) - np.cos(a)*np.cos(b)) / (np.sin(a)*np.sin(b)),
            1
        )
        cos_C = np.round(cos_C, 4)
        C = np.where(
            lons == lon0,
            0, np.arccos(cos_C)
        )
        C = np.where(
            (lons == lon0) & (lats < lat0),
            np.pi, C
        )
        C = np.where(
            mu >= 0,
            C,
            np.nan
        )
        C = np.where(
            np.sin(lons-lon0) < 0,
            -C, C
        )
    x = np.sin(C)*np.sin(a)
    y = np.cos(C)*np.sin(a)
    return x, y


def calc_circ_fraction_inside_unit_circle(x, y, r):
    """
    Calculate what fraction of a circle sits inside
    the unit circle.

    This is useful for modeling secondary eclipse.

    Parameters
    ----------
    x : float
        The x coordinate of the circle's center.
    y : float
        The y coordinate of the circle's center.
    r : float
        The circle's radius.

    Returns
    -------
    float
        The fraction of the circle that lies inside the unit circle.

    Raises
    ------
    ValueError
        If the circle's radius exceeds 1.


    Notes
    -----
    The calculation is based on the circle-circle intersection math presented in [1].

    References
    ----------
    [1].  Weisstein, Eric W. "Circle-Circle Intersection."
        From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Circle-CircleIntersection.html 
    """
    x0, y0, r0 = 0, 0, 1
    d = np.sqrt((x-x0)**2 + (y-y0)**2)
    if r > 1:
        raise ValueError('R must be less than 1.')

    if d > (r0 + r):
        return 0.0
    if d <= (r0 - r):
        return 1.0
    if d == 0 and r == 1:  # it is the unit circle
        return 1.0

    R = r0
    d1 = (d**2 - r**2 + R**2) / (2*d)
    d2 = (d**2 + r**2 - R**2) / (2*d)

    def area(_R, _d):
        return _R**2 * np.arccos(_d/_R) - _d * np.sqrt(_R**2 - _d**2)
    A1 = area(R, d1)
    A2 = area(r, d2)
    area_of_intersection = A1+A2

    area_of_circle = np.pi*r**2
    return area_of_intersection/area_of_circle
