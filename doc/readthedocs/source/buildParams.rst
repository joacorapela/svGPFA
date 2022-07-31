
## svGPFA parameters The following types of parameter values need to be
specified in a configuration file and/or in Python dictionaries.

.. toctree::
   :maxdepth: 1

   modelStructureParams
   dataStructureParams
   paramsInitialValues
   optimParams

We provide two utility functions (:func:`svGPFA.utils.initUtils.getInitialAndQuadParamsAndKernelsTypes`, :func:`svGPFA.utils.initUtils.getOptimParams`) to build svGPFA parameter data structures (used as arguments to svGPFA functions) from a first dictionary of dynamics parameters, a second dictionary with the contents of the configuration file and a third dictionary with default parameters values.  

For a given parameter, these functions first search for the parameter value in
the dictionary of dynamics parameters. If not found, they then search in
the dictionary with the contents of the configuration file. If still not found,
they use the parameter value in the default dictionary.

The default dictionary contains parameter values common across all models, the configuration file can contain parameter values for a specific model (e.g., a model to estimate the latent structure in recordings from a monkey motor cortex), and the dictionary of dynamic parameters can contain paremeters set from the command line in a parameter sweep).

Configuration files group parameter values into sections, given by the above parmeter types. See :download:`parameters.ini <examples/parameters.ini>` for an example configuration file.

The input parameter dictionaries to the above utility functions should be nested (e.g., ``params[section_name][param_name]`` contains the value of parameter ``param_name`` of section ``section_name`` in the ``params`` dictionary). When calling these utility functions the dictionaries with dynamic parameters or with the contents of the configuration file can be ``None``.
