Getting Started
===============

Most use cases for VSPEC involve this simple workflow:

#. Write a configuration file. For example: ``my_config.yaml``
#. Run the model:
    * ``import VSPEC``
    * ``model = VSPEC.ObservationModel.from_yaml('my_config.yaml')``
    * ``model.bin_spectra()``
    * ``model.build_planet()``
    * ``model.build_spectra()``
#. Read the data and analyze.

Configuration Files
-------------------

Because the model running portion is so straightforward, most of the 'work' comes from
writing a good configuration file. These files are usefull because they allow all the
paramters of a model to be set statically together, but they are nothing more than a YAML
representation of a ``VSPEC.params.InternalParameters`` object. See the :doc:`modules/params`
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

    from VSPEC import ObservationModel
    from pathlib import Path
    path = Path('my_config.yaml')
    model = ObservationModel.from_yaml(path)
    # run the model
    model.bin_spectra()
    model.build_planet()
    model.build_spectra()

The simulated observation will now be saved to a local directory specified in the header.

Reading the Data
----------------

``VSPEC`` data should be easy to read using standard Python libraries such as ``pandas``, however, we
have included a built-in analysis class for convenience. This ``PhaseAnalyzer`` object reads in the final
data products, which already live in the directory ``model.dirs['all_model']``.

.. code-block:: python
    
    from VSPEC import PhaseAnalyzer
    data = PhaseAnalyzer(model.dirs['all_model'])

See the :doc:`auto_examples/index` page for real use cases.