"""
Base parameter class
"""
import numpy as np
import yaml
from pathlib import Path

from VSPEC.config import PRESET_PATH

class BaseParameters:
    """
    Base class for Parameters
    """
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            self.__setattr__(key,value)
    @classmethod
    def _from_dict(cls,d:dict):
        return cls(**d)
    @classmethod
    def from_dict(cls,d:dict,*args):
        """
        Construct a BaseParameters (or subclass) instance from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary containing the parameters for `LimbDarkeningParameters`.

        Returns
        -------
        BaseParameters or subclass
            An instance of `BaseParameters` initialized with the provided parameters.

        Notes
        -----
        If the dictionary contains a key named 'preset', the corresponding class method
        will be called to create an instance with preset parameters.
        If 'preset' key is not present, the dictionary is expected to contain the values
        needed by the constructor.

        """
        if 'preset' in d.keys():
            return getattr(cls,d['preset'].replace('-','_'))()
        else:
            return cls._from_dict(d,*args)
    @classmethod
    def from_preset(cls,name):
        """
        Load a ``BaseParameters`` instance from a preset file.

        Parameters
        ----------
        name : str
            The name of the preset to load.
        
        Returns
        -------
        BaseParameters
            The class instance loaded from a preset.
        """
        if hasattr(cls,'_PRESET_PATH'):
            with open(cls._PRESET_PATH, 'r',encoding='UTF-8') as file:
                data = yaml.safe_load(file)
                return cls.from_dict(data[name])
        else:
            raise NotImplementedError('This class does not have a ``_PRESET_PATH`` attribute.')


class PSGtable(BaseParameters):
    """
    Class to store Table data for PSG
    """
    def __init__(self,x:np.ndarray,y:np.ndarray):
        self.x = np.array(x,dtype='float32')
        self.y = np.array(y,dtype='float32')
    def __str__(self):
        pairs = []
        for x,y in zip(self.x,self.y):
            pairs.append(f'{y:.2e}@{x:.2e}')
        return ','.join(pairs)
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            x = d['x'],
            y = d['y']
        )
    def to_psg(self):
        return self.__str__()

def parse_table(val,cls):
    if isinstance(val,dict):
        return PSGtable.from_dict(val['table'])
    else:
        return cls(val)
