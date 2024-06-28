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

``planet``
~~~~~~~~~~

.. note::

    Available presets include ``proxcenb`` and ``std``.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``name``
      - ``str``
      - The name of the planet.
    * - ``radius``
      - ``astropy.units.Quantity``
      - The radius of the planet.
    * - ``gravity``
      - see :ref:`subsec_gravity`
      - The mass/surface gravity/density of the planet.
    * - ``semimajor_axis``
      - ``astropy.units.Quantity``
      - The semi-major axis of the planet's orbit.
    * - ``orbit_period``
      - ``astropy.units.Quantity``
      - The period of the planet's orbit.
    * - ``rotation_period``
      - ``astropy.units.Quantity``
      - The rotation period of the planet.
    * - ``eccentricity``
      - ``float``
      - The eccentricity of the planet's orbit
    * - ``obliquity``
      - ``astropy.units.Quantity``
      - The obliquity (tilt) of the planet. Not currently implemented.
    * - ``obliquity_direction``
      - ``astropy.units.Quantity``
      - The direction of the planet's obliquity. The true anomaly
        at which the planet's north pole faces away from the star.
    * - ``init_phase``
      - ``astropy.units.Quantity``
      - The phase of the planet at the beginning of the simulation.
    * - ``init_substellar_lon``
      - ``astropy.units.Quantity``
      - The initial substellar longitude of the planet.

.. _subsec_gravity:
``gravity``
+++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``mode``
      - ``str``
      - The mode of the gravity parameter. Valid options are 'g', 'rho', and 'kg'.
    * - ``value``
      - ``astropy.units.Quantity``
      - The value of the gravity parameter. Can be any unit so long as the physical type is correct.

``system``
~~~~~~~~~~

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``distance``
      - ``astropy.units.Quantity``
      - The distance to the system.
    * - ``inclination``
      - ``astropy.units.Quantity``
      - The inclination of the system. Transit occurs when :math:`i=90^{\circ}`.
    * - ``phase_of_periasteron``
      - ``astropy.units.Quantity``
      - The phase of the planet when it reaches periasteron. Only necessary for planets
        with non-zero obliquity or non-zero eccentricity.

``obs``
~~~~~~~

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``observation_time``
      - ``astropy.units.Quantity``
      - The total time of the observation.
    * - ``integration_time``
      - ``astropy.units.Quantity``
      - The integration time of each epoch of observation.

``inst``
~~~~~~~~

.. note::

    Available presets include ``mirecle``, ``miri_lrs``, and ``niriss_soss``.
  
.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``single`` | ``coronagraph``
      - see :ref:`subsec_single` or :ref:`subsec_coronagraph`
      - The type of telescope to use.
    * - ``bandpass``
      - see :ref:`subsec_bandpass`
      - The bandpass & resolution of the observation.
    * - ``detector``
      - see :ref:`subsec_detector`
      - The detector properties of the observation.


.. _subsec_single:
``single``
++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``aperture``
      - ``astropy.units.Quantity``
      - The aperture size of the telescope.
    * - ``zodi``
      - ``float``
      - The level of the zodiacal background. See the PSG Handbook for details.

.. _subsec_coronagraph:
``coronagraph``
+++++++++++++++

.. warning::

    Use this instrument type with caution. It is not well tested and
    not fully implemented. Pull requests are welcome.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``aperture``
      - ``astropy.units.Quantity``
      - The aperture size of the telescope.
    * - ``zodi``
      - ``float``
      - The level of the zodiacal background. See the PSG Handbook for details.
    * - ``contrast``
      - ``float``
      - The contrast of the coronagraphic system.
    * - ``iwa``
      - ``PSGtable``
      - The inner working angle of the coronagraph.

.. _subsec_bandpass:
``bandpass``
++++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``wl_blue``
      - ``astropy.units.Quantity``
      - The minimum wavelength.
    * - ``wl_red``
      - ``astropy.units.Quantity``
      - The maximum wavelength.
    * - ``resolving_power``
      - ``int``
      - The resolving power of the observation.
    * - ``wavelength_unit``
      - ``astropy.units.Unit``
      - The unit to be used on the spectral axis.
    * - ``flux_unit``
      - ``astropy.units.Unit``
      - The unit to be used on the flux axis.

.. _subsec_detector:
``detector``
++++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``beam_width``
      - ``astropy.units.Quantity``
      - The width of the field of view.
    * - ``integration_time``
      - ``astropy.units.Quantity``
      - The integration time on the detector for saturation purposes.
        This effects the read noise of the observation.
    * - ``ccd``
      - see :ref:`subsubsec_ccd`
      - The CCD used for the observation.

.. _subsubsec_ccd:
``ccd``
"""""""

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``pixel_sampling``
      - ``int``
      - The number of pixels comprising a resolution element.
    * - ``read_noise``
      - ``astropy.units.Quantity``
      - The read noise of the CCD in electrons.
    * - ``dark_current``
      - ``astropy.units.Quantity``
      - The dark current of the CCD in electrons/second.
    * - ``throughput``
      - ``float``
      - The throughput of the optical system.
    * - ``emissivity``
      - ``float``
      - The emissivity of the optical system.
    * - ``temperature``
      - ``astropy.units.Quantity``
      - The temperature of the optical system.

``psg``
~~~~~~~

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``gcm_binning``
      - ``int``
      - The number of GCM points to bin together.
        Use 200 for testing and 3 for science (depending on the native resolution of your gcm).
    * - ``phase_binning``
      - ``int``
      - Planetary epoch binning. Useful if your star changes on a shorter timescale than your planet.
        For example, if set to 4, a planet spectrum will be calculated for every fourth stellar spectrum
        and will be interpolated otherwise.
    * - ``use_molecular_signatures``
      - ``bool``
      - Whether to have PSG consider the planetary atmosphere. If False, PSG will return a blackbody spectrum.
        Useful for testing.
    * - ``use_continuum_stellar`` (optional)
      - ``bool``
      - Whether to have PSG use a stellar atmosphere model. Otherwise, the stellar spectrum is a blackbody.
        While not guaranteed, there is no expectation that this parameter has any effect on VSPEC outputs.
        Default is ``True``.
    * - ``nmax``
      - ``int``
      - The number of n-stream pairs for scattering aerosols calculations. See the PSG Handbook for details.
    * - ``lmax``
      - ``int``
      - The number of Legendre polynomials for scattering aerosols calculations. See the PSG Handbook for details.
    * - ``continuum``
      - ``list of str``
      - The continuum opacities to include in the radiative transfer calculation. Typically these include
        ``Rayleigh``, ``Refraction``, and ``CIA_all``.

``gcm``
~~~~~~~

.. note::

    In order to support many types of GCM input this section can be configured many ways.\
    Read the following carefully.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``mean_molecular_weight``
      - ``float``
      - The mean molecular weight of the GCM in amu.
    * - ``binary`` | ``waccm`` | ``exocam`` | ``exoplasim`` | ``vspec``
      - see :ref:`subsec_binary`, :ref:`subsec_waccm`, :ref:`subsec_exocam`, :ref:`subsec_exoplasim`, :ref:`subsec_vspec`
      - The GCM type and parameters.
.. _subsec_binary:
``binary``
++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``path``
      - ``pathlib.Path``
      - The path to the PSG-readable GCM file. The contents of this file are parsed
        by ``pypsg.globes.GCMdecoder`` into a ``pypsg.globes.PyGCM`` object.

.. _subsec_waccm:
``waccm``
++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``path``
      - ``pathlib.Path``
      - The path to the WACCM netCDF file.
    * - ``itime`` (optional)
      - ``int``
      - The index of the time axis to use. If this is specified, then the GCM is static. Otherwise,
        it is assumed that the GCM changes as a function of time, and the ``tstart`` parameter must be specified.
    * - ``tstart`` (optional)
      - ``astropy.units.Quantity``
      - The start time of the GCM. Only used if ``itime`` is not specified.
    * - ``molecules``
      - ``list of str``
      - Which molecular species to look for in the GCM file. Use the same names as in PSG.
    * - ``aerosols`` (optional)
      - ``list of str``
      - Which aerosol species to look for in the GCM file. Use the same names as in PSG.
    * - ``background`` (optional)
      - ``str``
      - The background molecular species. The abundances in ``molecules`` will be subtracted from unity and the
        difference will be included as this background species. If this species is already in ``molecules``, then
        an error will be raised.
    * - ``lon_start`` (optional)
      - ``float``
      - The longitude of the first point on the longitudinal axis. Default is ``-180``.
    * - ``lat_start`` (optional)
      - ``float``
      - The latitude of the first point on the latitudinal axis. Default is ``-90``.

.. _subsec_exocam:
``exocam``
++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``path``
      - ``pathlib.Path``
      - The path to the ExoCAM netCDF file.
    * - ``itime`` (optional)
      - ``int``
      - The index of the time axis to use. If this is specified, then the GCM is static. Otherwise,
        it is assumed that the GCM changes as a function of time, and the ``tstart`` parameter must be specified.
    * - ``tstart`` (optional)
      - ``astropy.units.Quantity``
      - The start time of the GCM. Only used if ``itime`` is not specified.
    * - ``molecules``
      - ``list of str``
      - Which molecular species to look for in the GCM file. Use the same names as in PSG.
    * - ``aerosols`` (optional)
      - ``list of str``
      - Which aerosol species to look for in the GCM file. Use the same names as in PSG.
    * - ``background`` (optional)
      - ``str``
      - The background molecular species. The abundances in ``molecules`` will be subtracted from unity and the
        difference will be included as this background species. If this species is already in ``molecules``, then
        an error will be raised.
    * - ``lon_start`` (optional)
      - ``float``
      - The longitude of the first point on the longitudinal axis. Default is ``-180``.
    * - ``lat_start`` (optional)
      - ``float``
      - The latitude of the first point on the latitudinal axis. Default is ``-90``.
    * - ``mean_molecular_mass`` (optional)
      - ``float``
      - The mean molecular mass of the GCM. Must be specified if ``H2O`` is in ``molecules``. This is 
        because ExoCAM treats water as humidity.

.. _subsec_exoplasim:
``exoplasim``
+++++++++++++

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``path``
      - ``pathlib.Path``
      - The path to the ExoPlasim netCDF file.
    * - ``itime``
      - ``int``
      - The index of the time axis to use.
    * - ``molecules``
      - ``list of str``
      - Which molecular species to look for in the GCM file. Use the same names as in PSG.
    * - ``aerosols`` (optional)
      - ``list of str``
      - Which aerosol species to look for in the GCM file. Use the same names as in PSG.
    * - ``background`` (optional)
      - ``str``
      - The background molecular species. The abundances in ``molecules`` will be subtracted from unity and the
        difference will be included as this background species. If this species is already in ``molecules``, then
        an error will be raised.
    * - ``lon_start`` (optional)
      - ``float``
      - The longitude of the first point on the longitudinal axis. Default is ``-180``.
    * - ``lat_start`` (optional)
      - ``float``
      - The latitude of the first point on the latitudinal axis. Default is ``-90``.
    * - ``mean_molecular_mass`` (optional)
      - ``float``
      - The mean molecular mass of the GCM. Must be specified if ``H2O`` is in ``molecules``. This is 
        because ExoPlasim treats water as humidity.

.. _subsec_vspec:
``vspec``
+++++++++++++

.. warning::

    This is a minimal atmosphere model. It is useful for testing or when a
    sophisticated model is not necessary.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Key
      - Parses to
      - Description
    * - ``nlayer``
      - ``int``
      - The number of layers in the GCM.
    * - ``nlon``
      - ``int``
      - The number of longitude points in the GCM.
    * - ``nlat``
      - ``int``
      - The number of latitude points in the GCM.
    * - ``epsilon``
      - ``float``
      - The thermal inertia of the GCM. See :cite:t:`2011ApJ...726...82C`.
    * - ``p_surf``
      - ``astropy.units.Quantity``
      - The surface pressure of the GCM.
    * - ``p_stop``
      - ``astropy.units.Quantity``
      - The pressure at the top of the atmosphere.
    * - ``wind_u``
      - ``astropy.units.Quantity``
      - The U component (west-to-east) of the GCM's wind.
    * - ``wind_v``
      - ``astropy.units.Quantity``
      - The V component (south-to-north) of the GCM's wind.
    * - ``gamma``
      - ``float``
      - The adiabatic index of the GCM atmosphere.
    * - ``albedo``
      - ``float``
      - The surface albedo of the GCM.
    * - ``emissivity``
      - ``float``
      - The surface emissivity of the GCM.
    * - ``molecules``
      - ``list of str``
      - The molecular species in the GCM.
    * - ``lat_redistribution`` (optional)
      - ``float``
      - The latitudinal redistribution factor of the GCM. If ``1``, the poles have the
        same temperature as the equator. If ``0``, then there is no heat redistribution.
        Default is ``0.0``.





.. automodapi:: VSPEC.params