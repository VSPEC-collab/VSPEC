
import pytest

from VSPEC import helpers


def test_CoordinateGrid():
    """
    Test `VSPEC.helpers.CoordinateGrid`
    """
    helpers.CoordinateGrid()
    with pytest.raises(TypeError):
        helpers.CoordinateGrid(100, 100.)
    with pytest.raises(TypeError):
        helpers.CoordinateGrid(100., 100)
    i, j = 100, 50
    grid = helpers.CoordinateGrid(i, j)
    lat, lon = grid.oned()
    assert len(lat) == i
    assert len(lon) == j

    lats, lons = grid.grid()
    # switch if meshgrid index changes to `ij`
    assert lats.shape == (j, i)
    assert lons.shape == (j, i)

    zeros = grid.zeros(dtype='int')
    assert zeros.shape == (j, i)
    assert zeros.sum() == 0

    other = ''
    with pytest.raises(TypeError):
        grid == other

    other = helpers.CoordinateGrid(i+1, j)
    assert grid != other

    other = helpers.CoordinateGrid(i, j)
    assert grid == other
