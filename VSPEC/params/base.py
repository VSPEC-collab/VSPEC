"""
Base parameter class
"""
import numpy as np

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
    def from_dict(cls,d:dict):
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

        Examples
        --------
        >>> params_dict = {'preset': 'solar'}
        >>> params = LimbDarkeningParameters.from_dict(params_dict)

        In the example above, the 'solar' preset configuration is used to create an instance
        of LimbDarkeningParameters.

        >>> params_dict = {'u1': 0.3, 'u2': 0.1}
        >>> params = LimbDarkeningParameters.from_dict(params_dict)

        In the example above, custom values for 'u1' and 'u2' are provided to create an instance
        of LimbDarkeningParameters.
        """
        if 'preset' in d.keys():
            return getattr(cls,d['preset'].replace('-','_'))()
        else:
            return cls._from_dict(d)

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
