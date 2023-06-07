Input Parameters
=============================


Simulation-level Parameters
---------------------------

Options the control the ``VSPEC`` run at the highest level and
act as a container for all the parameters below.

.. currentmodule:: VSPEC.params.read

.. autosummary:: InternalParameters

.. autosummary:: Header


Stellar Parameters
------------------

The intrinsic properties of the star.

.. currentmodule:: VSPEC.params.stellar

.. autosummary:: StarParameters

.. autosummary::

    SpotParameters
    FaculaParameters
    FlareParameters
    GranulationParameters
    LimbDarkeningParameters



Bulk Planet and System Parameters
---------------------------------

Bulk (i.e. non-GCM) properties of the planet and properties of the
system in relation to the observer.

.. currentmodule:: VSPEC.params.planet

.. autosummary::

    PlanetParameters
    GravityParameters

.. autosummary::

    SystemParameters


GCM Parameters
--------------

Properties of the Global Circulation Model (GCM) that describe the planet.

.. currentmodule:: VSPEC.params.gcm

.. autosummary:: gcmParameters

.. autosummary::

    binaryGCM
    vspecGCM
    waccmGCM

PSG/GlobES Parameters
---------------------

Options that are specific to the opperation of PSG/GlobES

.. currentmodule:: VSPEC.params.gcm

.. autosummary:: psgParameters

.. autosummary:: APIkey


Observation Parameters
----------------------

These properties control the length of the observation and the cadence of integrations.

.. currentmodule:: VSPEC.params.observation

.. autosummary:: ObservationParameters


Instrument Parameters
---------------------

A description of the instrument that controls the bandpass, resolving power, and noise.

.. currentmodule:: VSPEC.params.observation

.. autosummary:: InstrumentParameters

Telescope
^^^^^^^^^

.. autosummary:: TelescopeParameters

.. autosummary::

    SingleDishParameters
    CoronagraphParameters

Bandpass
^^^^^^^^

.. autosummary:: BandpassParameters


Detector
^^^^^^^^

.. autosummary:: DetectorParameters

.. autosummary:: ccdParameters


The ``PSGtable`` Class
----------------------

Some parameters in PSG can take a table of input in the place of a single value (e.g.
instrument thoughput can be a function of wavelength). An advanced ``VSPEC`` user can use the
``PSGtable`` class to specify these inputs. Note that the user must be careful to place values
in the units that PSG is expecting.

.. currentmodule:: VSPEC.params.base

.. autosummary:: PSGtable

The Base Parameter Class
------------------------

This is the base class for input parameters.

.. currentmodule:: VSPEC.params.base

.. autosummary:: BaseParameters

