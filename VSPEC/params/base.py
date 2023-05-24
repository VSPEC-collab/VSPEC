"""
Base parameter class
"""

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
            return getattr(cls,d['preset'])()
        else:
            return cls._from_dict(d)