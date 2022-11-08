from astropy import units as u, constants as c
from numpy import isclose as np_isclose

def to_float(quant,unit):
    return (quant/unit).to(u.Unit('')).value

def isclose(a,b,tol):
    unit = tol.unit
    return np_isclose(to_float(a,unit),to_float(b,unit),rtol=to_float(tol,unit))