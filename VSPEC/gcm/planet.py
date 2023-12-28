"""
VSPEC GCM Planet Container

"""
import numpy as np
from typing import Tuple
from astropy import units as u

import pypsg
from pypsg.cfg.globes import GCM

from VSPEC.gcm.structure import Wind, Molecule, Aerosol, AerosolSize
from VSPEC.gcm import structure as st
from VSPEC.config import atmosphere_type_dict as mtype, aerosol_type_dict as atype

class Winds:
    """
    Represent wind speed in a GCM.
    
    Wind in PSG is represented by a 2 component vector :math:1(U,V)` where
    :math:`U` is West-to-East and :math:`V` is South-to-North.
    
    Parameters
    ----------
    wind_u : VSPEC.gcm.structure.Wind
        The West-to-East wind.
    wind_v : VSPEC.gcm.structure.Wind
        The South-to-North wind.
    
    Attributes
    ----------
    wind_u : VSPEC.gcm.structure.Wind
        The West-to-East wind.
    wind_v : VSPEC.gcm.structure.Wind
        The South-to-North wind.
    
    """
    def __init__(
        self,
        wind_u:Wind,
        wind_v:Wind
    ):
        self.wind_u = wind_u
        self.wind_v = wind_v
    @property
    def flat(self):
        """
        Get a flat version of this variable to write to a binary file.

        Returns
        -------
        np.ndarray
            The flattened array.
        """
        # pylint: disable-next:unexpected-keyword-arg
        return np.concatenate([self.wind_u.flat,self.wind_v.flat],dtype='float32')

class Molecules:
    """
    GCM Molecules container.
    
    Parameters
    ----------
    molecules : VSPEC.gcm.structure.Molecule
        The molecules that exist in the atmosphere.
    
    Attributes
    ----------
    molecules : VSPEC.gcm.structure.Molecule
        The molecules that exist in the atmosphere.
    """
    def __init__(
        self,
        molecules:Tuple[Molecule]
    ):
        self.molecules = molecules
    @property
    def flat(self)->np.ndarray:
        """
        1D representation of the data.

        Returns
        -------
        np.ndarray
            The flattened data.
        """
        # pylint: disable-next:unexpected-keyword-arg
        return np.concatenate([mol.flat for mol in self.molecules],dtype='float32')
    @classmethod
    def _from_dict(cls,d:dict,shape:tuple):
        """
        Create a `Molecules` object from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary describing the molecules.
        shape : tuple
            The shape of the GCM array.

        """
        molecules = []
        for key, val in d.items():
            name = key
            value = u.Quantity(val)
            molecules.append(
                st.Molecule.constant(name,value,shape)
            )
        return cls(tuple(molecules))
    @classmethod
    def from_dict(cls,d:dict,shape:tuple):
        """
        Create a `Molecules` object from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary describing the molecules.
        shape : tuple
            The shape of the GCM array.

        """
        return cls._from_dict(d,shape)

class Aerosols:
    """
    GCM Aerosols container.
    
    Parameters
    ----------
    aerosols : tuple of VSPEC.gcm.structure.Aerosol
        The aerosol abundances in the atmosphere.
    sizes : tuple of VSPEC.gcm.structure.AerosolSize
        The aerosol sizes in the atmosphere.
    """
    def __init__(
        self,
        aerosols:Tuple[Aerosol],
        sizes:Tuple[AerosolSize]
    ):
        self.aerosols = aerosols
        self.sizes = sizes
    @property
    def flat(self)->np.ndarray:
        """
        A flat representation of the array.

        Returns
        -------
        np.ndarray
            The flattened array.
        """
        # pylint: disable-next:unexpected-keyword-arg
        return np.concatenate([aero.flat for aero in self.aerosols]+[size.flat for size in self.sizes],dtype='float32')
    @classmethod
    def _from_dict(cls,d:dict,shape:tuple):
        """
        Load an `Aerosols` object from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary describing the aerosols.
        shape : tuple
            The shape of the GCM array.
        """
        aersols = []
        sizes = []
        for key,value in d.items():
            name = key
            abn = u.Quantity(value['abn'])
            size = u.Quantity(value['size'])
            aersols.append(st.Aerosol.constant(name,abn,shape))
            sizes.append(st.AerosolSize.constant(f'{name}_size',size,shape))
    @classmethod
    def from_dict(cls,d:dict,shape:tuple):
        """
        Load an `Aerosols` object from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary describing the aerosols.
        shape : tuple
            The shape of the GCM array.
        """
        return cls._from_dict(d,shape)

class Planet:
    """
    A simple planet GCM.

    Parameters
    ----------
    wind : Winds
        The wind speeds in the GCM represented by a 2-component vector (U, V),
        where U is the West-to-East wind and V is the South-to-North wind.
    tsurf : st.SurfaceTemperature
        The surface temperature of the planet.
    psurf : st.SurfacePressure
        The surface pressure of the planet.
    albedo : st.Albedo
        The albedo of the planet's surface.
    emissivity : st.Emissivity
        The emissivity of the planet's surface.
    temperature : st.Temperature
        The temperature profile of the planet's atmosphere.
    pressure : st.Pressure
        The pressure profile of the planet's atmosphere.
    molecules : Molecules
        The molecules present in the planet's atmosphere.
    aerosols : Aerosols
        The aerosols present in the planet's atmosphere.

    Attributes
    ----------
    wind : Winds
        The wind speeds in the GCM.
    tsurf : st.SurfaceTemperature
        The surface temperature of the planet.
    psurf : st.SurfacePressure
        The surface pressure of the planet.
    albedo : st.Albedo
        The albedo of the planet's surface.
    emissivity : st.Emissivity
        The emissivity of the planet's surface.
    temperature : st.Temperature
        The temperature profile of the planet's atmosphere.
    pressure : st.Pressure
        The pressure profile of the planet's atmosphere.
    molecules : Molecules
        The molecules present in the planet's atmosphere.
    aerosols : Aerosols
        The aerosols present in the planet's atmosphere.

    Raises
    ------
    TypeError
        If the pressure attribute is not provided or is None.
    AssertionError
        If the shapes of the attributes are inconsistent.

    """
    def __init__(
        self,
        wind:Winds,
        tsurf:st.SurfaceTemperature,
        psurf:st.SurfacePressure,
        albedo:st.Albedo,
        emissivity:st.Emissivity,
        temperature:st.Temperature,
        pressure:st.Pressure,
        molecules:Molecules,
        aerosols:Aerosols
    ):
        self.wind = wind
        self.tsurf = tsurf
        self.psurf = psurf
        self.albedo = albedo
        self.emissivity = emissivity
        self.temperature = temperature
        self.pressure = pressure
        self.molecules = molecules
        self.aerosols = aerosols
        self.validate()
    
    def validate(self):
        """
        Validates the consistency of the planet's attributes.

        Raises
        ------
        TypeError
            If the pressure attribute is not provided or is None.
        AssertionError
            If the shapes of the attributes are inconsistent.
        """

        if self.pressure is None:
            raise TypeError('pressure must be provided!')
        shape3d = self.pressure.shape
        shape2d = shape3d[1:]
        if self.wind is not None:
            assert self.wind.wind_u.shape == shape3d
            assert self.wind.wind_v.shape == shape3d
        assert self.psurf.shape == shape2d
        if self.tsurf is not None:
            assert self.tsurf.shape == shape2d
        if self.albedo is not None:
            assert self.albedo.shape == shape2d
        if self.emissivity is not None:
            assert self.emissivity.shape == shape2d
        if self.temperature is not None:
            assert self.temperature.shape == shape3d
        if self.molecules is not None:
            for molecule in self.molecules.molecules:
                assert molecule.shape == shape3d
        if self.aerosols is not None:
            for aerosol in self.aerosols.aerosols:
                assert aerosol.shape == shape3d
            for size in self.aerosols.sizes:
                assert size.shape == shape3d
    @property
    def shape(self)->Tuple[int,int,int]:
        """
        Returns the shape of the planet's arrays (nlayers, nlon, lat).

        Returns
        -------
        shape : Tuple[int, int, int]
            The shape of the planet's arrays.
        """

        nlayers,nlon,lat = self.pressure.shape
        return nlayers,nlon,lat
    @property
    def lons(self)->u.Quantity:
        """
        Returns the longitudes of the planet's grid.

        Returns
        -------
        lons : u.Quantity
            The longitudes of the planet's grid.
        """

        _,nlon,_ = self.shape
        return np.linspace(-180,180,nlon,endpoint=False)*u.deg
    @property
    def dlon(self)->u.Quantity:
        """
        Returns the longitudinal grid spacing.

        Returns
        -------
        dlon : u.Quantity
            The longitudinal grid spacing.
        """

        _,nlon,_ = self.shape
        return 360*u.deg / nlon
    @property
    def dlat(self)->u.Quantity:
        """
        Returns the latitudinal grid spacing.

        Returns
        -------
        dlat : u.Quantity
            The latitudinal grid spacing.
        """

        _,_,nlat = self.shape
        return 180*u.deg / nlat
    @property
    def lats(self)->u.Quantity:
        """
        Returns the latitudes of the planet's grid.

        Returns
        -------
        lats : u.Quantity
            The latitudes of the planet's grid.
        """

        _,_,nlat = self.shape
        return np.linspace(-90,90,nlat,endpoint=True)*u.deg
    @property
    def gcm_properties(self)->str:
        """
        Constructs the string value of the PSG ``<ATMOSPHERE-GCM-PROPERTIES>`` parameter.

        Returns
        -------
        gcm_properties : str
            The value of the ``<ATMOSPHERE-GCM-PROPERTIES>`` parameter.
        """

        nlayer,nlon,nlat = self.shape
        coords = f'{nlon},{nlat},{nlayer},-180.0,-90.0,{self.dlon.to_value(u.deg):.2f},{self.dlat.to_value(u.deg):.2f}'
        vars = []
        if self.wind is not None:
            vars.append('Winds')
        if self.tsurf is not None:
            vars.append(self.tsurf.name)
        vars.append(self.psurf.name)
        if self.albedo is not None:
            vars.append(self.albedo.name)
        if self.emissivity is not None:
            vars.append(self.emissivity.name)
        if self.temperature is not None:
            vars.append(self.temperature.name)
        vars.append(self.pressure.name)
        if self.molecules is not None:
            for molecule in self.molecules.molecules:
                vars.append(molecule.name)
        if self.aerosols is not None:
            for aerosol in self.aerosols.aerosols:
                vars.append(aerosol.name)
            for size in self.aerosols.sizes:
                vars.append(size.name)
        return f'{coords},{",".join(vars)}'
    @property
    def flat(self)->np.ndarray:
        """
        Returns a flattened representation of the planet's data.

        Returns
        -------
        flat_data : np.ndarray
            A flattened representation of the planet's data.
        """
        # pylint: disable-next:unexpected-keyword-arg
        return np.concatenate([
            [] if self.wind is None else self.wind.flat,
            [] if self.tsurf is None else self.tsurf.flat,
            self.psurf.flat,
            [] if self.albedo is None else self.albedo.flat,
            [] if self.emissivity is None else self.emissivity.flat,
            [] if self.temperature is None else self.temperature.flat,
            self.pressure.flat,
            [] if self.molecules is None else self.molecules.flat,
            [] if self.aerosols is None else self.aerosols.flat
        ],dtype='float32')
    
    @property
    def psg_params(self)->dict:
        """
        Returns a dictionary of PSG parameters representing the planet.

        Returns
        -------
        params : dict
            A dictionary of PSG parameters representing the planet.
        """

        nlayers,_,_ = self.shape
        gases = [molec.name for molec in self.molecules.molecules]
        if self.aerosols is not None:
            aerosols = [aero.name for aero in self.aerosols.aerosols]
        else:
            aerosols = []
        gas_types = [f'HIT[{mtype[gas]}]' if isinstance(mtype[gas],int) else mtype[gas] for gas in gases]
        aerosol_types = [atype[aerosol] for aerosol in aerosols]
        gcm_params = self.gcm_properties
        params = {
            'ATMOSPHERE-DESCRIPTION': 'Variable Star Phase CurvE (VSPEC) default GCM',
            'ATMOSPHERE-STRUCTURE': 'Equilibrium',
            'ATMOSPHERE-LAYERS': f'{nlayers}',
            'ATMOSPHERE-NGAS': f'{len(gases)}',
            'ATMOSPHERE-GAS': ','.join(gases),
            'ATMOSPHERE-TYPE': ','.join(gas_types),
            'ATMOSPHERE-ABUN': ','.join(['1']*len(gases)),
            'ATMOSPHERE-UNIT': ','.join(['scl']*len(gases)),
            'ATMOSPHERE-GCM-PARAMETERS': gcm_params
        }
        if len(aerosols) > 0:
            params.update({
                'ATMOSPHERE-NAERO': f'{len(aerosols)}',
                'ATMOSPHERE-AEROS': ','.join(aerosols),
                'ATMOSPHERE-ATYPE': ','.join(aerosol_types),
                'ATMOSPHERE-AABUN': ','.join(['1']*len(aerosols)),
                'ATMOSPHERE-AUNIT': ','.join(['scl']*len(aerosols)),
                'ATMOSPHERE-ASIZE': ','.join(['1']*len(aerosols)),
                'ATMOSPHERE-ASUNI': ','.join(['scl']*len(aerosols))
            })
        return params
    @property
    def content(self)->bytes:
        """
        Returns the content of the planet as a PSG-compatible bytes object.

        Returns
        -------
        content : bytes
            The content of the planet as a PSG-compatible bytes object.
        """

        config = '\n'.join([f'<{param}>{value}' for param,value in self.psg_params.items()])
        content = bytes(config,encoding='UTF-8')
        dat = self.flat.tobytes('C')
        return content + b'\n<BINARY>' + dat + b'</BINARY>'
    @property
    def pycfg(self)->pypsg.PyConfig:
        """
        Get a `pypsg.PyConfig` object representing the planet.
        """
        return pypsg.PyConfig(
            atmosphere=pypsg.cfg.EquilibriumAtmosphere.from_cfg(self.psg_params),
            gcm=GCM(
                header=self.psg_params['ATMOSPHERE-GCM-PARAMETERS'],
                dat=self.flat
            )
        )
        
    @classmethod
    def from_dict(cls,d:dict):
        """
        Creates a `Planet` instance from a dictionary representation.

        Parameters
        ----------
        d : dict
            The dictionary representing the planet.

        Returns
        -------
        planet : Planet
            The `Planet` instance created from the dictionary.
        """

        nlayer = int(d['shape']['nlayer'])
        nlon = int(d['shape']['nlon'])
        nlat = int(d['shape']['nlat'])
        shape2d = (nlon,nlat)
        shape3d = (nlayer,nlon,nlat)
        tsurf = st.SurfaceTemperature.from_map(
            shape=shape2d,
            epsilon=float(d['planet']['epsilon']),
            star_teff=u.Quantity(d['planet']['teff_star']),
            albedo=float(d['planet']['albedo']),
            r_star=u.Quantity(d['planet']['r_star']),
            r_orbit=u.Quantity(d['planet']['r_orbit'])
        )
        pressure = st.Pressure.from_limits(
            high=u.Quantity(d['planet']['pressure']['psurf']),
            low=u.Quantity(d['planet']['pressure']['ptop']),
            shape=shape3d
        )
        wind = Winds(
            wind_u=st.Wind.contant(
                name='U',value=u.Quantity(d['planet']['wind']['U']),shape=shape3d
            ),
            wind_v=st.Wind.contant(
                name='V',value=u.Quantity(d['planet']['wind']['V']),shape=shape3d
            )
        )
        psurf = st.SurfacePressure.from_pressure(pressure)
        albedo = st.Albedo.constant(
            val=u.Quantity(d['planet']['albedo']),shape=shape2d
        )
        emissivity = st.Emissivity.constant(
            val=u.Quantity(d['planet']['emissivity']),shape=shape2d
        )
        gamma = float(d['planet']['gamma'])
        temperature = st.Temperature.from_adiabat(gamma,tsurf,pressure)
        molecules = Molecules.from_dict(d['molecules'],shape3d)
        aerosols = None if d.get('aerosols',None) is None else Aerosols.from_dict(d['aerosols'],shape3d)
        return cls(
            wind=wind,
            tsurf=tsurf,
            psurf=psurf,
            albedo=albedo,
            emissivity=emissivity,
            temperature=temperature,
            pressure=pressure,
            molecules=molecules,
            aerosols=aerosols
        )

