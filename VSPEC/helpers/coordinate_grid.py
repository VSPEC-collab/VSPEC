"""
Coordinate Grid class
"""

import numpy as np
from astropy import units as u


class CoordinateGrid:
    """
    Class to standardize the creation of latitude and longitude grids.

    This class provides a convenient way to create latitude and longitude grids of specified dimensions. It allows the creation of both one-dimensional arrays and two-dimensional grids of latitude and longitude points.

    Parameters
    ----------
    Nlat : int, optional (default=500)
        Number of latitude points.
    Nlon : int, optional (default=1000)
        Number of longitude points.

    Raises
    ------
    TypeError
        If Nlat or Nlon is not an integer.


    Attributes
    ----------
    Nlat : int
        Number of latitude points.
    Nlon : int
        Number of longitude points.

    Examples
    --------
    >>> grid = CoordinateGrid(Nlat=100, Nlon=200)
    >>> lats, lons = grid.oned()
    >>> print(lats.shape, lons.shape)
    (100,) (200,)
    >>> grid_arr = grid.grid()
    >>> print(grid_arr.shape)
    (100, 200)
    >>> zeros_arr = grid.zeros()
    >>> print(zeros_arr.shape)
    (200, 100)
    >>> other_grid = CoordinateGrid(Nlat=100, Nlon=200)
    >>> print(grid == other_grid)
    True

    """

    def __init__(self, Nlat=500, Nlon=1000):
        if not isinstance(Nlat, int):
            raise TypeError('Nlat must be int')
        if not isinstance(Nlon, int):
            raise TypeError('Nlon must be int')
        self.Nlat = Nlat
        self.Nlon = Nlon

    def oned(self):
        """
        Create one dimensional arrays of latitude and longitude points.

        Returns
        -------
        lats : astropy.units.Quantity , shape=(Nlat,)
            Array of latitude points.
        lons : astropy.units.Quantity , shape=(Nlon,)
            Array of longitude points.

        """
        lats = np.linspace(-90, 90, self.Nlat)*u.deg
        lons = np.linspace(0, 360, self.Nlon,endpoint=False)*u.deg
        return lats, lons

    def grid(self):
        """
        Create a 2 dimensional grid of latitude and longitude points.

        Returns
        -------
        lats : astropy.units.Quantity , shape=(Nlat,Nlon)
            Array of latitude points.
        lons : astropy.units.Quantity , shape=(Nlat,Nlon)
            Array of longitude points.

        """
        lats, lons = self.oned()
        return np.meshgrid(lats, lons)

    def zeros(self, dtype='float32'):
        """
        Get a grid of zeros.

        Parameters
        ----------
        dtype : str, default='float32'
            Data type to pass to np.zeros.

        Returns
        -------
        arr : np.ndarray, shape=(Nlon, Nlat)
            Grid of zeros.

        """
        return np.zeros(shape=(self.Nlon, self.Nlat), dtype=dtype)

    def __eq__(self, other):
        """
        Check to see if two CoordinateGrid objects are equal.

        Parameters
        ----------
        other : CoordinateGrid
            Another CoordinateGrid object.

        Returns
        -------
        bool
            Whether the two objects have equal properties.

        Raises
        ------
        TypeError
            If `other` is not a CoordinateGrid object.

        """
        if not isinstance(other, CoordinateGrid):
            raise TypeError('other must be of type CoordinateGrid')
        else:
            return (self.Nlat == other.Nlat) & (self.Nlon == other.Nlon)
