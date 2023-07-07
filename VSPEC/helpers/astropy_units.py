from astropy import units as u
from numpy import isclose as np_isclose


def isclose(quant1: u.Quantity, quant2: u.Quantity, tol: u.Quantity) -> bool:
    """
    Check if two quantities are close

    Use `numpy.isclose` on two quantity objects.
    This function safely casts them to floats first.

    Parameters
    ----------
    param1 : astropy.units.Quantity
        First object for comparison
    param2 : astropy.units.Quantity
        Second object for comparison
    tol : astropy.units.Quantity
        Error tolerance between `param1` and `param2`

    Returns
    -------
    bool
        Whether `param1` and `param2` are within `tol`

    Raises
    ------
    astropy.units.UnitConversionError
        If `param1` and `param2` have different physical types.


    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np

    >>> values1 = np.array([1.0, 2.0, 3.0]) * u.m
    >>> values2 = np.array([1.01, 2.02, 3.03]) * u.m
    >>> tol = 0.05 * u.m
    >>> isclose(values1, values2, tol)
    array([ True,  True, False])

    >>> temperatures1 = np.array([25.0, 30.0, 35.0]) * u.K
    >>> temperatures2 = np.array([25.5, 30.2, 34.8]) * u.K
    >>> tol = 0.3 * u.K
    >>> isclose(temperatures1, temperatures2, tol)
    array([ True,  True,  True])

    """
    unit = tol.unit
    return np_isclose(quant1.to_value(unit), quant2.to_value(unit), atol=tol.to_value(unit))
