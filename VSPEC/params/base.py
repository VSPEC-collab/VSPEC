"""
Base parameter class
"""
import yaml
from astropy import units as u
from libpypsg.cfg.base import Table

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
    def __init__(
        self,
        x:u.Quantity,
        y:u.Quantity
    ):
        self.x = x
        self.y = y
    def __str__(self):
        pairs = []
        for x,y in zip(self.x,self.y):
            pairs.append(f'{y:.2e}@{x:.2e}')
        return ','.join(pairs)
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            x = d['x']*u.Unit(d.get('xunit','')),
            y = d['y']*u.Unit(d.get('yunit',''))
        )
    @classmethod
    def from_dict(cls, d: dict, *args):
        """
        Construct a ``PSGtable`` object from a dictionary.
        
        Parameters
        ----------
        d : dict
            The dictionary to use to construct the class.

        Returns
        -------
        PSGtable
            The constructed class instance.

        """
        return super().from_dict(d, *args)
    def to_psg(self):
        """
        Convert to libpypsg table
        """
        return Table(
            x = self.x.to_value(u.dimensionless_unscaled) if self.x.unit is u.dimensionless_unscaled else self.x,
            y = self.y.to_value(u.dimensionless_unscaled) if self.y.unit is u.dimensionless_unscaled else self.y
        )

def parse_table(val,cls):
    """
    Parse some input that could potentially construct a ``PSGtable`` object.
    
    If ``val`` is a dictionary, use it to construct a table. Otherwise, construct
    an object of type ``cls`` from ``val`` (which is probably a string).

    Parameters
    ----------
    val : any
        The input to parse.
    cls : type
        The class to cast ``val`` to if it is not a dictionary.

    Returns
    -------
    any
        The parsed input.
    """
    if isinstance(val,dict):
        return PSGtable.from_dict(val['table']).to_psg()
    else:
        return cls(val)
