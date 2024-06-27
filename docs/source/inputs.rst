Inputs
======

``VSPEC`` can read in inputs in two ways:

#. From a YAML file
#. From a ``VSPEC.params.InternalParameters`` object

YAML is convenient because it is human-readable and can be easily ported between programs. Creating an ``InternalParameters``
instance directly in Python can also be convenient, especially when running many near-identical models.

The ``VSPEC.params`` YAML Format
---------------------------------

The most basic way to create an ``ObservationalModel`` object instance is through a YAML file:

.. code-block:: python

    import VSPEC
    model = VSPEC.ObservationModel.from_yaml('my_config.yaml')

The file ``my_config.yaml`` contains all the necessary information to run ``VSPEC``.
YAML files have a hierarchical structure, so for example the header section looks like this:

.. code-block:: yaml

    header:
        data_path: transit
        teff_min: 2300 K
        teff_max: 3900 K
        desc: This is a VSPEC example.
        verbose: 0

The sections of a ``VSPEC`` YAML configuration file are below.

.. note::

    Because the configuration file is parsed by a parameter object that knows
    what type to expect, it is possible to parse directly to an ``astropy.units.Quantity``
    object. Any parameter that expects a quantity will read in the user input as:
    
    .. code-block:: python
        
        u.Quantity(my_input)
    
    So encoding the quantity as a string is very easy. For example the ``teff_min``
    parameter can be set to ``2300 K``, and so after parsing it will be equivalent
    to having set it to ``2300 * u.K``.

``header``
~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Key
     - Parses to
     - Description
   * - ``data_path``
     - ``str``
     - The name of the directory inside ``.vspec`` where outputs will be saved.
   * - ``teff_min``
     - ``astropy.units.Quantity``
     - The minimum effective temperature to include in the spectrum grid.
   * - ``teff_max``
     - ``astropy.units.Quantity``
     - The maximum effective temperature to include in the spectrum grid.
   * - ``desc``
     - ``str``
     - A description of the model.
   * - ``verbose``
     - ``int``
     - The level of verbosity.
   * - ``seed``
     - ``int``
     - The seed for the random number generator.

``star``
~~~~~~~~

.. note::

    Available presets include ``static_proxima``, ``spotted_proxima``, ``flaring_proxima``, and ``proxima``.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Key
     - Parses to
     - Description
   * - ``psg_star_template``
     - ``str``
     - The template for the star. See the PSG Handbook for options.
   * - ``teff``
     - ``astropy.units.Quantity``
     - The effective temperature of the star.
   * - ``mass``
     - ``astropy.units.Quantity``
     - The mass of the star.
   * - ``radius``
     - ``astropy.units.Quantity``
     - The radius of the star.
   * - ``period``
     - ``astropy.units.Quantity``
     - The rotational period of the star.
   * - ``misalignment``
     - ``astropy.units.Quantity``
     - The misalignment between the stellar rotation axis and the orbital axis. This is the planet's "mutual inclination".
   * - ``misalignment_dir``
     - ``astropy.units.Quantity``
     - The direction of stellar rotation axis misalignment, relative to the argument of periapsis.
   * - ``ld``
     - See :ref:`subsec_ld`
     - The limb darkening parameters of the star.
   * - ``spots``
     - See :ref:`subsec_spots`
     - The parameters to create star spots.
   * - ``faculae``
     - See :ref:`subsec_faculae`
     - The parameters to create faculae.
   * - ``flares``
     - See :ref:`subsec_flares`
     - The parameters to create flares.
   * - ``granulation``
     - See :ref:`subsec_granulation`
     - The parameters to create granulation.
   * - ``grid_params``
     - ``int`` or ``tuple``
     - Stellar surface grid parameters. If ``tuple``, the number of points in the
       latitude and longitude directions. If ``int``, the number of total grid points
       for a Fibonacci spiral grid. In general the spiral grid is more efficient.
   * - ``spectral_grid``
     - ``str``
     - The spectral grid to use. Either ``default`` for the default stellar grid from
       ``Gridpolator``, or ``bb`` to use a blackbody forward model (very fast).

.. _subsec_ld:
``ld``
++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``u1``
      - ``float``
      - The first limb darkening coefficient.
    * - ``u2``
      - ``float``
      - The second limb darkening coefficient.

.. _subsec_spots:
``spots``
+++++++++

.. note::

    Available presets include ``none``, ``mdwarf``, and ``solar``.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``distribution``
      - ``str``
      - The distribution of the spots. Either ``iso`` for an isotropic distribution or ``solar`` for two bands at :math:`\\pm 15^{\\circ}` latitude.
    * - ``initial_coverage``
      - ``float``
      - The spot coverage created initially by generating spots at random stages of life.
    * - ``equillibrium_coverage``
      - ``float``
      - The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equilibrium.
    * - ``burn_in``
      - ``astropy.units.Quantity``
      - The duration of the burn-in period, during which the spot coverage approaches equilibrium.
    * - ``area_mean``
      - ``astropy.units.Quantity``
      - The mean area of a spot on the star's surface in MSH.
    * - ``area_logsigma``
      - ``float``
      - The standard deviation of the spot areas. This is a lognormal
        distribution, so the units of this value are dex.
    * - ``teff_umbra``
      - ``astropy.units.Quantity``
      - The effective temperature of the spot umbrae.
    * - ``teff_penumbra``
      - ``astropy.units.Quantity``
      - The effective temperature of the spot penumbrae.
    * - ``growth_rate``
      - ``astropy.units.Quantity``
      - The rate at which new spots grow. Spots grow exponentially, so this quantity has units of 1/time.
    * - ``decay_rate``
      - ``astropy.units.Quantity``
      - The rate at which existing spots decay [area/time].
    * - ``initial_area``
      - ``astropy.units.Quantity``
      - The area of a spot at birth.

.. _subsec_faculae:
``faculae``
++++++++++++

.. note::

    Available presets include ``none`` and ``std``.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``distribution``
      - ``str``
      - The distribution of the spots. Currently only ``iso`` is supported.
    * - ``Equillibrium_coverage``
      - ``float``
      - The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equilibrium.
    * - ``burn_in``
      - ``astropy.units.Quantity``
      - The duration of the burn-in period, during which the spot coverage approaches
        equilibrium.
    * - ``mean_radius``
      - ``astropy.units.Quantity``
      - The mean radius of the faculae.
    * - ``logsigma_radius``
      - ``float``
      - The standard deviation of the radius in dex.
    * - ``mean_timescale``
      - ``astropy.units.Quantity``
      - The mean faculae lifetime.
    * - ``logsigma_timescale``
      - ``float``
      - The standard deviation of the lifetime in dex.\
    * - ``depth``
      - ``float``
      - The depth of the facula depression.
    * - ``floor_teff_slope``
      - ``astropy.units.Quantity``
      - The slope of the radius-Teff relationship for the cool floor.
    * - ``floor_teff_min_rad``
      - ``astropy.units.Quantity``
      - The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    * - ``floor_teff_base_dteff``
      - ``astropy.units.Quantity``
      - The Teff of the floor at the minimum radius.
    * - ``wall_teff_slope``
      - ``astropy.units.Quantity``
      - The slope of the radius-Teff relationship for the hot wall.
    * - ``wall_teff_intercept``
      - ``astropy.units.Quantity``
      - The intercept of the radius-Teff relationship for the hot wall.

.. note::

    The temperatures of a facula wall or floor are a function of the radius.
    See `the vspec-vsm source <https://github.com/VSPEC-collab/vspec-vsm/blob/main/vspec_vsm/faculae.py>`_
    for information on how these values are calculated.

.. _subsec_flares:
``flares``
++++++++++

.. note::

    Available presets include ``none`` and ``std``.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``dist_teff_mean``
      - ``astropy.units.Quantity``
      - The mean temperature of the flares.
    * - ``dist_teff_sigma``
      - ``astropy.units.Quantity``
      - The standard deviation of the flare temperatures.
    * - ``dist_fwhm_mean``
      - ``astropy.units.Quantity``
      - The mean FWHM of the flare lightcurves [time].
    * - ``dist_fwhm_logsigma``
      - ``float``
      - The standard deviation of the FWHM in dex.
    * - ``alpha``
      - ``float``
      - The slope of the frequency-energy powerlaw.
    * - ``beta``
      - ``float``
      - The intercept of the frequency-energy powerlaw.
    * - ``min_energy``
      - ``astropy.units.Quantity``
      - The minimum energy of the flares. Set to infinity to disable.
    * - ``cluster_size``
      - ``int``
      - The typical number of flares in each cluster.

.. _subsec_granulation:
``granulation``
+++++++++++++++

.. note::

    Available presets include ``none`` and ``std``.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``mean``
      - ``float``
      - The mean coverage of low-teff granulation.
    * - ``amp``
      - ``float``
      - The amplitude of granulation oscillations.
    * - ``period``
      - ``astropy.units.Quantity``
      - The period of granulation oscillations.
    * - ``dteff``
      - ``astropy.units.Quantity``
      - The difference between the quiet photosphere and the low-teff granulation region.


.. automodapi:: VSPEC.params