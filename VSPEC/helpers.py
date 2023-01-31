from astropy import units as u
from numpy import isclose as np_isclose

def to_float(quant,unit):
    """
    to float

    Cast a quantity to a float given a unit

    Args:
        quant (Quantity): Quantity to be cast
        unit (Unit): Unit to be cast with
    
    Returns:
        (float): Cast quantity
    """
    return (quant/unit).to(u.Unit('')).value

def isclose(a,b,tol):
    """
    is close

    Use numpy.isclose on two quantity objects. This function safely casts them to floats first.

    Args:
        a (Quantity): array to be compared
        b (Quantity): array to be compared
        tol (Quantity): tolerance
    """
    unit = tol.unit
    return np_isclose(to_float(a,unit),to_float(b,unit),rtol=to_float(tol,unit))