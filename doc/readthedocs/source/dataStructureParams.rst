Data structure parameters
=========================

Data structure parameters can be specified in longer and shorter formats. If both are specified, the longer format takes precedence.

Longer format
^^^^^^^^^^^^^
Two items need to be specified:

* ``trials_start_times`` should provide a list of length *n_trials*, with float values, such that *trials_start_times[i]* gives the start time of the ith trial.

* ``trials_end_times`` should provide a list of length *n_trials*, with float values, such that *trials_end_times[i]* gives the end time of the ith trial.

Example:

    .. code-block:: none
       :caption: example section [data_structure_params] of the configuration in the longer format

        [data_structure_params]
        trials_start_times = [0.0, 0.01, 0.0]
        trials_end_times = [1.0, 1.01, 1.0]

Shorter format
^^^^^^^^^^^^^^
Two items need to be specified:

* ``trials_start_time`` should provide the start time (float value, secs) of all trials.

* ``trials_end_time`` should provide the end time (float value, secs) of all trials.

Example:

    .. code-block:: none
       :caption: example section [data_structure_params] of the configuration file in the shorter format

        [data_structure_params]
        trials_start_time = 0.0
        trials_end_time = 1.0

