from astropy import units as u, constants as c

def to_float(quant,unit):
    return (quant/unit).to(u.Unit('')).value