"""
VSPEC: Variable Star Phase Curve

VSPEC is designed as an observation simulation suite for rocky
exoplanets orbiting variable stars. Built for the Mid-IR Exoplanet
CLimate Explorer (MIRECLE) mission concept, this package is flexible
enough to simulate observations over a wide range of conditions.

VSPEC utilizes the Planetary Spectrum Generator (PSG, psg.gsfc.nasa.gov)
to perform radiative transfer calculations
"""
from VSPEC.main import ObservationModel
from VSPEC.analysis import PhaseAnalyzer