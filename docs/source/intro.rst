Introduction
============
``VSPEC`` (Variable Star PhasE Curve) is a powerful tool 
designed to simulate observations of exoplanets orbiting variable stars.

``VSPEC`` uses a dynamic model of stellar spots, faculae, 
flares, and granulation combined with simultations from the Planetary Spectrum Generator
(PSG, `Villanueva et al., 2018 <https://ui.adsabs.harvard.edu/abs/2018JQSRT.217...86V/abstract>`_)
:cite:empty:`2018JQSRT.217...86V`
to simulate phase resolved observations of planetary spectra.

Recent observations with JWST have shown stellar contamination can cause signals similar to exoplanet
atmospheres :cite:p:`2023ApJ...948L..11M`. Similarly, the future Habitable Worlds Observatory gather
reflected-light spectra of earth-like exoplanets over a very long baseline. To understand these challenges
and develop data analysis that can mitigate them, we need a robust and flexible modeling suite.

This package was initially designed to simulate data for the Mid-IR Exoplanet CLimate Explorer mission concept 
(MIRECLE, `Mandell et al., 2022 <https://ui.adsabs.harvard.edu/abs/2022AJ....164..176M/abstract>`_),
:cite:empty:`2022AJ....164..176M`
but has since been refactored to be a general-use tool. It builds off of PSG and supports reflcted,
thermal, and transmission spectroscopy as well as the use of a coronagraph for direct imaging spectroscopy.

Installation
************

``VSPEC`` can be installed via pip.

.. code-block:: shell

    pip install vspec


You can also clone our repository. This would be great if you are interested in contributing. 

.. code-block:: shell
    
    git clone https://github.com/VSPEC-collab/VSPEC.git
    cd VSPEC
    pip install -e .[dev,plot]

Note that adding ``[dev,plot]`` will install additional dependecies for development and plotting.


Running PSG
***********

While it is not 100% necessary to run PSG locally in order to use ``VSPEC``, it is
highly recommended. Luckly, PSG is easy to install and run. Detailed instructions can be
found in the `PSG handbook <https://psg.gsfc.nasa.gov/help.php#handbook>`_ (see page 153).

We recommend using `Rancher Desktop <rancherdesktop.io>`_ to run the PSG Docker container,
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

.. note::
    Users who wish to run simulations with resolving powers higher than ``R=500`` must
    install the CORRKMEDMAIN package (up to ``R=5000``).