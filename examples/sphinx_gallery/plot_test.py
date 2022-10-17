"""
Simulated data and default params
=================================
In this notebook we use simulated data to estimate an svGPFA model using the default initial parameters.

1. Estimate model
-----------------

1.1 Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pickle

# import svGPFA.utils.test
import svGPFA.utils.initUtils

#%%
# 1.2 Load spikes times
# ~~~~~~~~~~~~~~~~~~~~~
# The spikes times of all neurons in all trials should be stored in nested lists. ``spikes_times[r][n]`` should contain a list of spikes times of neuron ``n`` in trial ``r``.

sim_res_filename = "../../examples/data/32451751_simRes.pickle" # simulation results filename
with open(sim_res_filename, "rb") as f:
    sim_res = pickle.load(f)
spikes_times = sim_res["spikes"]


#%%
# 1.3 Set estimation hyperparameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_latents = 2                                      # number of latents
trials_start_time = 0.0                            # trials start time
trials_end_time = 1.0                              # triasl end time
em_max_iter = 30                                   # maximum number of EM iterations
n_trials = len(spikes_times)                       # n_trials
n_neurons = len(spikes_times[0])                   # n_neurons


#%%
# 1.4 Get default parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# build default parameter specificiations                                                                                                                                              
# default_params_spec = svGPFA.utils.test.getDefaultParamsDict(
default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(
    n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents,
    trials_start_time=trials_start_time, trials_end_time=trials_end_time,
    em_max_iter=em_max_iter)
