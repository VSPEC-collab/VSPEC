"""
File Helpers
"""
from pathlib import Path
import warnings


def check_and_build_dir(path: Path) -> None:
    """
    Check and build directory.

    Check if a path exists, if not, create it.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to check and create
    """
    if isinstance(path, str):
        path = Path(str)
    if path.exists():
        pass
    else:
        path.mkdir(parents=True)


def get_filename(N: int, n_zfill: int, ext: str):
    """
    Get the filename that a PSG output is written to.
    This function unifies all of the filename handling to a single piece of code.

    Parameters
    ----------
    N : int
        The 'phase number' of the file. Represents the iteration of the simulation.
    n_zfill : int
        The ``zfill()`` convention to use for zero-padding the phase number.
    ext : str
        The file extension, such as ``'.rad'`` for rad files.

    Returns
    -------
    str
        The filename

    Warns
    -----
    RuntimeWarning
        If `N` has more digits than `n_zfill` can accommodate.
    """
    if N > (10**(n_zfill)-1):
        warnings.warn(
            f'zfill of {n_zfill} not high enough for phase {N}', RuntimeWarning)
    return f'phase{str(N).zfill(n_zfill)}.{ext}'
