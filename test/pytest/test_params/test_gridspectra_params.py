"""
Tests for `VSPEC.params.read.AbstractGridSpectraParameters` and decendants

"""
from astropy import units as u
from GridPolator import GridSpectra

from VSPEC.spectra import ForwardSpectra
from VSPEC.params.read import AbstractGridParameters
from VSPEC.params import(
    VSPECGridParameters,
    BlackbodyGridParameters
)

def test_basic_functionality():
    def fun(a,b,c,d):
        return a+b+c+d
    gr = AbstractGridParameters(fun,a='a',c='c')
    assert gr.build(b='b',d='d') == 'abcd'

def test_abstract_from_dict():
    d = {
        'name': 'vspec',
        'max_teff': 3200*u.K,
        'min_teff': 3000*u.K
    }
    assert isinstance(AbstractGridParameters.from_dict(d), VSPECGridParameters)
    
    d = {
        'name': 'bb',
    }
    assert isinstance(AbstractGridParameters.from_dict(d), BlackbodyGridParameters)

def test_vspec_basic():
    grp = VSPECGridParameters(
        max_teff=3200*u.K,
        min_teff=3000*u.K,
    )
    gr = grp.build(
        w1=3*u.um,
        w2=5*u.um,
        resolving_power=100.
    )
    assert isinstance(gr, GridSpectra)

def test_bb_basic():
    gr = BlackbodyGridParameters().build()
    assert isinstance(gr, ForwardSpectra)