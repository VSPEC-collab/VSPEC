Getting Started
===============

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

Most use cases for VSPEC involve this simple workflow:

#. Write a configuration file. For example: ``my_config.yaml``
#. Run the model:
    * :python:`>>> import VSPEC`
    * :python:`>>> model = VSPEC.ObservationModel.from_yaml('my_config.yaml')`
    * :python:`>>> model.build_planet()`
    * :python:`>>> model.build_spectra()`
#. Read the data and analyze.

.. note::
    ``VSPEC`` must have some way to call the PSG API. We interface with PSG using the
    ``libpypsg`` package. The easiest thing you can do is add to the top of your script
    :python:`import libpypsg; libpypsg.docker.set_url_and_run()`. This will check if PSG
    is installed, start the container if necessary, and set the URL appropriately.

.. note::
    If you have an API key for PSG you can set it using the ``libpypsg`` package. In a terminal type
    :bash:`$ python -c "import libpypsg;libpypsg.save_settings(api_key='YOUR_API_KEY')"`. This will save
    you key to `~/.libpypsg/settings.json` where it can be read safely in the future.
    
    .. warning::
        Never commit your API key to a public repository.

Configuration Files
-------------------

Because the model running portion is so straightforward, most of the 'work' comes from
writing a good configuration file. These files are useful because they allow all the
paramaters of a model to be set statically together, but they are nothing more than a YAML
representation of a ``VSPEC.params.InternalParameters`` object. See the :doc:`inputs`
page for descriptions of every ``VSPEC`` parameter.

In general the files look like this:

.. code-block:: yaml
    
    header:
        [ simulation header ]
    star:
        [ bulk star paramters ]
        ld:
            [ limb-darkening parameters ]
        spots:
            [ spot paramters ]
        faculae:
            [ facula paramters ]
        flares:
            [ flare parameters ]
        granulation:
            [ granulation parameters ]
    planet:
        [ planet-specific parameters ]
    system:
        [ system-specific parameters ]
    obs:
        [ observation-specific parameters ]
    inst:
        [ instrument-specific parameters ]
    psg:
        [ PSG-specific parameters ]
    gcm:
        [ GCM parameters ]


Running the Model
-----------------

As shown above, running a configured model is very easy.

.. code-block:: python

    import libpypsg
    libpypsg.docker.set_url_and_run()
    
    from VSPEC import ObservationModel
    path = 'my_config.yaml'
    model = ObservationModel.from_yaml(path)
    # run the model
    model.build_planet()
    model.build_spectra()

The simulated observation will now be saved to a local directory specified in the header. Except in cases where
the ``Header`` is custom written by the user (i.e. not constructed from a YAML file), all simulation output is
stored in a directory called ``.vspec``.

Reading the Data
----------------

``VSPEC`` data is stored in the popular `.fits` format, however, we
have included a built-in analysis class for convenience. This ``PhaseAnalyzer`` object reads in the final
data products, which already live in the directory ``model.dirs['all_model']``.

.. code-block:: python
    
    from VSPEC import PhaseAnalyzer
    data = PhaseAnalyzer(model.directories['all_model'])

See the :doc:`auto_examples/end_to_end/index` page for real use cases.