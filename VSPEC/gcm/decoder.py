"""
VSPEC gcm decoder module
"""

from pathlib import Path
import json
import numpy as np
from astropy import units as u, constants as c

from VSPEC.config import MOLEC_DATA_PATH

def get_gcm_binary(config:str or Path or bytes):
    """
    Separate a GCM into it's text and binary components.

    Parameters
    ----------
    config : str, pathlib.Path, bytes
        The filename to read or the bytes to parse
    
    Returns
    -------
    head : str
        The value of the 'ATMOSPHERE-GCM-PARAMETERS' option.
    dat : np.ndarray
        The data between '<BINARY></BINARY>' tags.

    """
    key = '<ATMOSPHERE-GCM-PARAMETERS>'
    start = b'<BINARY>'
    end = b'</BINARY>'
    if isinstance(config,(str,Path)):
        with open(config,'rb') as file:
            fdat = file.read()
    else:
        fdat = config
    header, dat = fdat.split(start)
    dat = dat.replace(end,b'')
    dat = np.frombuffer(dat,dtype='float32')
    for line in str(header).split(r'\n'):
        if key in line:
            return line.replace(key,''), np.array(dat)
def sep_header(header):
    fields = header.split(',')
    coords = fields[:7]
    var = fields[7:]
    return coords,var

class GCMdecoder:
    DOUBLE = ['Winds']
    FLAT = ['Tsurf','Psurf','Albedo','Emissivity']
    def __init__(self,header,dat):
        self.header=header
        self.dat=dat
    @classmethod
    def from_psg(cls,config:str or Path or bytes):
        """
        Construct a ``GCMdecoder`` from PSG.

        Parameters
        ----------
        config : str, pathlib.Path, bytes
            The filename to read or the bytes to parse
        """
        head,dat = get_gcm_binary(config)
        return cls(head,dat)
    def rename_var(self,oldname,newname):
        coords,vars = sep_header(self.header)
        if oldname in vars:
            vars = [newname if var==oldname else var for var in vars]
        else:
            raise KeyError(f'Variable {oldname} not in header.')
        new_header = ','.join(coords+vars)
        self.header=new_header
    def get_shape(self):
        coord,_ = sep_header(self.header)
        Nlon,Nlat,Nlayer, _,_,_,_ = coord
        return int(Nlon),int(Nlat),int(Nlayer)
    def get_3d_size(self):
        Nlon,Nlat,Nlayer = self.get_shape()
        return Nlon*Nlat*Nlayer
    def get_2d_size(self):
        Nlon,Nlat,_ = self.get_shape()
        return Nlon*Nlat
    def get_lats(self):
        coord,_ = sep_header(self.header)
        _,Nlat,_,_,lat0,_,dlat = coord
        return np.arange(int(Nlat))*float(dlat) + float(lat0)
    def get_lons(self):
        coord,_ = sep_header(self.header)
        Nlon,_,_,lon0,_,dlon,_ = coord
        return np.arange(int(Nlon))*float(dlon) + float(lon0)
    def get_molecules(self):
        with open(MOLEC_DATA_PATH, 'rt',encoding='UTF-8') as file:
            molec_data = json.loads(file.read())
        _,variables = sep_header(self.header)
        molecs = [var for var in variables if var in molec_data.keys()]
        return molecs
    def get_aerosols(self):
        _,variables = sep_header(self.header)
        aerosols = [var for var in variables if var+'_size' in variables]
        aerosol_sizes = [aero+'_size' for aero in aerosols]
        return aerosols, aerosol_sizes
    def __getitem__(self,item):
        _, variables = sep_header(self.header)
        if not item in variables:
            raise KeyError(f'{item} not found. Acceptable keys are {variables}')
        else:
            start = 0
            def get_array_length(var):
                if var in self.DOUBLE:
                    return 2*self.get_3d_size(), 'double'
                elif var in self.FLAT:
                    return self.get_2d_size(), 'flat'
                else:
                    return self.get_3d_size(), 'single'
            def package_array(dat,key):
                if key == 'single':
                    Nlat,Nlon,Nlayer = self.get_shape()
                    return dat.reshape(Nlayer,Nlon,Nlat)
                elif key == 'flat':
                    Nlat,Nlon,Nlayer = self.get_shape()
                    return dat.reshape(Nlon,Nlat)
                elif key == 'double':
                    Nlat,Nlon,Nlayer = self.get_shape()
                    return dat.reshape(2,Nlayer,Nlon,Nlat)
                else:
                    raise ValueError(f'Unknown value {key}')
            for var in variables:
                size,key = get_array_length(var)
                if item==var:
                    dat = self.dat[start:start+size]
                    return package_array(dat,key)
                else:
                    start+=size
    def __setitem__(self,item:str,new_value:np.ndarray):
        """
        set an array
        """
        old_value = self.__getitem__(item)
        if not old_value.shape == new_value.shape:
            raise ValueError('New shape must match old shape.')
        new_value = new_value.astype(old_value.dtype)
        _, variables = sep_header(self.header)
        def get_array_length(var):
            if var in self.DOUBLE:
                return 2*self.get_3d_size(), 'double'
            elif var in self.FLAT:
                return self.get_2d_size(), 'flat'
            else:
                return self.get_3d_size(), 'single'
        start = 0
        for var in variables:
            size,_ = get_array_length(var)
            if item==var:
                dat = new_value.flatten(order='C')
                self.dat[start:start+size] = dat
                return None
            else:
                start+=size
        
    def remove(self,item):
        """
        remove an item from the gcm
        """
        coords, variables = sep_header(self.header)
        if item not in variables:
            return ValueError(f'Unknown {item}')
        def get_array_length(var):
            if var in self.DOUBLE:
                return 2*self.get_3d_size(), 'double'
            elif var in self.FLAT:
                return self.get_2d_size(), 'flat'
            else:
                return self.get_3d_size(), 'single'
        start = 0
        for var in variables:
            size,_ = get_array_length(var)
            if item==var:
                s = slice(start,start+size)
                self.dat = np.delete(self.dat,s)
            else:
                start+=size
        new_variables = [var for var in variables if item != var]
        self.header = ','.join(coords+new_variables)
        
    def copy_config(self,path_to_copy:Path,path_to_write:Path,NMAX=2,LMAX=2,mean_mass=28):
        """
        Copy a PSG config file but overwrite all GCM parameters and data
        """
        def replace_line(line):
            if b'<ATMOSPHERE-GCM-PARAMETERS>' in line:
                return bytes('<ATMOSPHERE-GCM-PARAMETERS>' + self.header + '\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-LAYERS>' in line:
                _,_,Nlayer = self.get_shape()
                return bytes(f'<ATMOSPHERE-LAYERS>{Nlayer}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-NGAS>' in line:
                n_molecs = len(self.get_molecules())
                return bytes(f'<ATMOSPHERE-NGAS>{n_molecs}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-GAS>' in line:
                molecs = ','.join(self.get_molecules())
                return bytes(f'<ATMOSPHERE-GAS>{molecs}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-TYPE>' in line:
                with open(MOLEC_DATA_PATH, 'rt',encoding='UTF-8') as file:
                    molec_data = json.loads(file.read())
                molecs = self.get_molecules()
                atm_types = ','.join([f'HIT[{molec_data[mol]["ID"]}]' for mol in molecs])
                return bytes(f'<ATMOSPHERE-TYPE>{atm_types}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-ABUN>' in line:
                n_molecs = len(self.get_molecules())
                return bytes(f'<ATMOSPHERE-ABUN>{",".join(["1"]*n_molecs)}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-UNIT>' in line:
                n_molecs = len(self.get_molecules())
                return bytes(f'<ATMOSPHERE-UNIT>{",".join(["scl"]*n_molecs)}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-NAERO>' in line:
                n_aero = len(self.get_aerosols()[0])
                return bytes(f'<ATMOSPHERE-NAERO>{n_aero}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-AEROS>' in line:
                aeros = ','.join(self.get_aerosols()[0])
                return bytes(f'<ATMOSPHERE-AEROS>{aeros}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-ATYPE>' in line:
                dat = {
                    'Water': 'AFCRL_Water_HRI',
                    'WaterIce': 'Warren_ice_HRI'
                }
                atypes = ','.join([dat[aero] for aero in self.get_aerosols()[0]])
                return bytes(f'<ATMOSPHERE-ATYPE>{atypes}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-AABUN>' in line:
                n_aero = len(self.get_aerosols()[0])
                return bytes(f'<ATMOSPHERE-AABUN>{",".join(["1"]*n_aero)}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-AUNIT>' in line:
                n_aero = len(self.get_aerosols()[0])
                return bytes(f'<ATMOSPHERE-AUNIT>{",".join(["scl"]*n_aero)}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-ASIZE>' in line:
                n_aero = len(self.get_aerosols()[0])
                return bytes(f'<ATMOSPHERE-ASIZE>{",".join(["1"]*n_aero)}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-ASUNI>' in line:
                n_aero = len(self.get_aerosols()[0])
                return bytes(f'<ATMOSPHERE-ASUNI>{",".join(["scl"]*n_aero)}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-NMAX>' in line:
                return bytes(f'<ATMOSPHERE-NMAX>{NMAX}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-LMAX>' in line:
                return bytes(f'<ATMOSPHERE-LMAX>{LMAX}\n',encoding='UTF-8')
            elif b'<ATMOSPHERE-WEIGHT>' in line:
                return bytes(f'<ATMOSPHERE-WEIGHT>{mean_mass}\n',encoding='UTF-8')
            else:
                return line + b'\n'

        with open(path_to_copy,'rb') as infile:
            with open(path_to_write, 'wb') as outfile:
                contents = infile.read()
                t,b = contents.split(b'<BINARY>')
                b = b.replace(b'</BINARY>',b'')
                lines = t.split(b'\n')
                for line in lines:
                    outfile.write(replace_line(line))
                outfile.write(b'<BINARY>')
                outfile.write(np.asarray(self.dat,dtype='float32',order='C'))
                outfile.write(b'</BINARY>')
        

    def get_mean_molec_mass(self):
        """
        Get the mean molecular mass at every point on the GCM
        """
        with open(MOLEC_DATA_PATH, 'rt',encoding='UTF-8') as file:
            molec_data = json.loads(file.read())
        Nlon,Nlat,Nlayer = self.get_shape()
        mean_molec_mass = np.zeros(shape=(Nlayer,Nlat,Nlon))*u.g/u.mol
        for mol, dat in molec_data.items():
            mass = dat['mass']
            try:
                data = 10**self[mol]
                mean_molec_mass += data*mass*u.g/u.mol
            except KeyError:
                pass
        return mean_molec_mass
    def get_alt(self,M:u.Quantity,R:u.Quantity):
        """
        Get the altitude of each GCM point.
        """
        P = 10**self['Pressure']*u.bar
        T = self['Temperature']*u.K
        m = self.get_mean_molec_mass()
        Nlon, Nlat, Nlayers = self.get_shape()
        z_unit = u.km
        z = [np.zeros(shape=(Nlat,Nlon))]
        for i in range(Nlayers-1):
            dP = P[i+1,:,:] - P[i,:,:]
            rho = m[i,:,:]*(P[i,:,:]+ 0.5*dP)/c.R/T[i,:,:]
            r = z[-1]*z_unit + R
            g = M*c.G/r**2
            dz = -dP/rho/g
            z.append((z[-1]*z_unit+dz).to(z_unit).value)
        return z*z_unit
    def get_column_density(self,mol:str,M:u.Quantity,R:u.Quantity,):
        """
        Get the column density of a gas at each point on the gcm.
        """
        abn = 10**self[mol]*u.mol/u.mol
        P = 10**self['Pressure']*u.bar
        T = self['Temperature']*u.K
        partial_pressure = P*abn
        alt = self.get_alt(M,R)
        heights = np.diff(alt,axis=0)
        density:u.Quantity = np.sum(partial_pressure[:-1]*heights/c.R/T[:-1],axis=0)
        return density.to(u.mol/u.cm**2)

    def get_column_clouds(self,var:str,M:u.Quantity,R:u.Quantity,):
        """
        Get the column density of a cloud at each point on the gcm.
        """
        mass_frac = 10**self[var]*u.kg/u.kg
        P = 10**self['Pressure']*u.bar
        T = self['Temperature']*u.K
        molar_mass = self.get_mean_molec_mass()
        alt = self.get_alt(M,R)
        heights = np.diff(alt,axis=0)
        gas_mass_density = P[:-1]*heights/c.R/T[:-1]*molar_mass[:-1] # g cm-2
        mass_density = np.sum(mass_frac[:-1]*gas_mass_density,axis=0).cgs
        return mass_density.to(u.kg/u.cm**2)