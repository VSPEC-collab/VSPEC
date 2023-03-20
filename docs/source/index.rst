.. VSPEC documentation master file, created by
   sphinx-quickstart on Wed Mar  8 13:08:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VSPEC's documentation!
=================================

VSPEC (Variable Star PhasE Curve) is an exoplanet modeling suite that combines NASA's Planetary Spectrum Generator
(`PSG <https://psg.gsfc.nasa.gov>`_) with a custom variable star
(based on the `spotty <https://github.com/mrtommyb/spotty>`_ and `xoflares <https://github.com/mrtommyb/xoflares>`_ packages).

The goal of VSPEC is to simulate observations of non-transiting rocky planets around variable M dwarfs
in order to build tools to disentangle the light from the planet from that of the host. However, this code
has been designed to be used more generally, with planetary, stellar, and variability scenarios
that can be designed by the user.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro
   components
   tutorials
   input_params



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
