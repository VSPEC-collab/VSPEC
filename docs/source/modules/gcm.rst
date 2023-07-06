``VSPEC.gcm``
=============================


.. currentmodule:: VSPEC.gcm

Interface with PSG
++++++++++++++++++

.. autosummary::
    :toctree: ../api
    :template: custom_template.rst

    Planet
    GCMdecoder

Energy Balance and Heat Transport
+++++++++++++++++++++++++++++++++

.. currentmodule:: VSPEC.gcm.heat_transfer

.. autosummary::
    :toctree: ../api
    :template: custom_template.rst

    TemperatureMap

.. autosummary::
    :toctree: ../api
    :template: custom_func.rst

    get_flux
    get_psi
    pcos
    colat
    get_t0
    get_equator_curve
    validate_energy_balance



GCM Structure
+++++++++++++

.. currentmodule:: VSPEC.gcm.structure

.. autosummary::
    :toctree: ../api
    :template: custom_template.rst

    Variable
    Wind
    Pressure
    SurfacePressure
    SurfaceTemperature
    Temperature
    Molecule
    Aerosol
    AerosolSize
    Albedo
    Emissivity

.. currentmodule:: VSPEC.gcm.planet

.. autosummary::
    :toctree: ../api
    :template: custom_template.rst

    Winds
    Molecules
    Aerosols