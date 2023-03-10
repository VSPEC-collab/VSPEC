Introduction
============
``VSPEC`` (Variable Star PhasE Curve) is a powerful tool 
designed to simulate observations of exoplanets orbiting variable stars.

VSPEC uses a dynamic model of stellar spots, faculae, and 
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
For now it is best to clone from github, but we would like to use PyPI in the future.

.. code-block:: shell
    
    git clone https://github.com/VSPEC-collab/VSPEC.git
    cd VSPEC
    pip install -e .
