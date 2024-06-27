.. VSPEC documentation master file, created by
   sphinx-quickstart on Wed Mar  8 13:08:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/images/vspec_logo.png
   :scale: 50 %
   :alt: VSPEC logo
   :align: center



Welcome!
========

VSPEC (Variable Star PhasE Curve) is an exoplanet modeling suite that combines NASA's Planetary Spectrum Generator
(`PSG <https://psg.gsfc.nasa.gov>`_) with a custom variable star
(based on the `spotty <https://github.com/mrtommyb/spotty>`_ and `xoflares <https://github.com/mrtommyb/xoflares>`_ packages).

``VSPEC`` is a general exoplanet observation modeling suite that is built to be modular. ``VSPEC`` can simulate
phase curves of non-transiting planets, transits, eclipses, and direct-imaging spectroscopy. It uses
a noise model from PSG, which is sophisticated enough to simulate realistic data for a variety of scenarios.

The original motivation for ``VSPEC`` was to create synthetic datasets of Planetary Infrared Excess (PIE) observations,
but the code is much more general.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro
   getting_started
   components
   modules/params
   references

Examples
========
.. toctree::
   :maxdepth: 0

   auto_examples/end_to_end/index
   auto_examples/other/index

Authors
=======

``VSPEC`` is maintained by Ted Johnson and the original code was written by Cameron Kelahan. Additional contributions come from
Tobi Hammond and the PSG team. For more information, see our `author list <https://github.com/VSPEC-collab/VSPEC/blob/main/AUTHORS.md>`_.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
