Introduction
============
``VSPEC`` (Variable Star PhasE Curve) is a powerful tool 
designed to simulate observations of exoplanets orbiting variable stars.

``VSPEC`` uses a dynamic model of stellar spots, faculae, and 
flares combined with simultations from the Planetary Spectrum Generator 
(PSG, `Villanueva et al., 2018 <https://ui.adsabs.harvard.edu/abs/2018JQSRT.217...86V/abstract>`_)
to simulate phase resolved observations of planetary thermal emission spectra.
This package was designed for the Mid-IR Exoplanet CLimate Explorer mission concept 
(MIRECLE, `Mandell et al., 2022 <https://ui.adsabs.harvard.edu/abs/2022AJ....164..176M/abstract>`_),
but was built to be used more generally.

The primary goal of this software is to simulate combined planet-host spectra
in order to develop techniques to remove the star using the Planetary Infrared Excess
(PIE) technique. For more info on PIE, see `Stevenson (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...898L..35S/abstract>`_
and `Lustig-Yaeger et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...921L...4L/abstract>`_.

Installation
************

``VSPEC`` can be installed via pip from our github repository. In the future,
stable versions will be on PyPI.

.. code-block:: shell

    pip install git+https://github.com/VSPEC-collab/VSPEC.git@main


You can also clone our repository. If you are interested in contributing, please reach out. 

.. code-block:: shell
    
    git clone https://github.com/VSPEC-collab/VSPEC.git
    cd VSPEC
    pip install -e .

Required dependanceies of ``VSPEC`` include ``numpy``, ``pandas``, ``matplotlib``, ``scipy``,
``astropy``, ``tqdm``, ``h5py``, and ``xoflares``.

The ``cartopy`` package is optional, but is necessary for some built-in plotting funcitonality.

``xoflares`` is the only non-standard python package that ``VSPEC`` requires, and there are a
number of ways to install it depending on your python version. One option is to clone it to your
local machine:

.. code-block:: shell
    
    git clone https://github.com/mrtommyb/xoflares.git@master
    cd xoflares
    pip install -e .

However, since we only utilize a small portion of ``xoflares`` (which is designed to integrate
into the ``exoplanet`` package), it is also possible to install a stripped-down
branch that only contains the imports and functions that ``VSPEC`` requires:

.. code-block:: shell

    pip install git+https://github.com/tedjohnson12/xoflares.git@numpy-only


Running PSG
***********

While it is not 100% necessary to run PSG locally in order to use ``VSPEC``, it is
highly recommended. Luckly, PSG easy to install and run. Detailed instructions can be
found in the `PSG handbook <https://psg.gsfc.nasa.gov/help.php#handbook>_` (see page 153).

We recommend using `Rancher Desktop <rancherdesktop.io>_` to run the PSG Docker container,
as it is free for all use. If you do not have administrative (sudo) access to your
computer, uncheck 'Administrative Access' in Rancher Desktop settings.

``VSPEC`` assumes you have installed PSG using the commands (Handbook page 154):

.. code-block:: shell

    docker logout
    docker pull nasapsg/psg
    docker tag nasapsg/psg psg

    docker run -d --name psg -p 3000:80 psg

Importantly, we assume that a container named ``psg`` can be accessed
though local port ``3000``.

``VSPEC`` has been tested with the following PSG packages installed:

- BASE
- SURFACES
- ATMOSPHERES
- LINES
- EXO
- CORRKLOWMAIN
