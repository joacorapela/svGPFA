
"""
Simulated data and default params
=================================

In this notebook we use simulated data to estimate an svGPFA model using the default initial parameters.
"""
#%%
# Estimate model
# --------------
#
# Import required packages
# ~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import time
import warnings
import torch
import pickle

import gcnu_common.stats.pointProcesses.tests
import svGPFA.stats.kernels
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import svGPFA.plot.plotUtilsPlotly


#%%
# Get spikes times
# ~~~~~~~~~~~~~~~~~
# The spikes times of all neurons in all trials should be stored in nested lists. ``spikes_times[r][n]`` should contain a list of spikes times of neuron ``n`` in trial ``r``.

sim_res_filename = "../../examples/data/32451751_simRes.pickle" # simulation results filename
with open(sim_res_filename, "rb") as f:
    sim_res = pickle.load(f)
spikes_times = sim_res["spikes"]
n_trials = len(spikes_times)
n_neurons = len(spikes_times[0])
trials_start_time = 0.0
trials_end_time = 1.0
trials_start_times = [trials_start_time] * n_trials
trials_end_times = [trials_end_time] * n_trials

#%%
# Check that spikes have been epoched correctly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%%
# Plot spikes 
# ~~~~~~~~~~~
# Plot the spikes of all trials of a randomly chosen neuron. Most trials should
# contain at least one spike.

neuron_to_plot_index = torch.randint(low=0, high=n_neurons, size=(1,)).item()
fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneNeuron(
    spikes_times=spikes_times,
    neuron_index=neuron_to_plot_index,
    title=f"Neuron index: {neuron_to_plot_index}",
)
fig

#%%
# Run some simple checks on spikes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The function ``checkEpochedSpikesTimes`` tests that:
#
#   a. every neuron fired at least one spike across all trials,
#   b. for each trial, the spikes times of every neuron are between the trial
#      start and end times.
#
# If any check fails, a ``ValueError`` will be raised. Otherwise a checks
# passed message should be printed.

try:
    gcnu_common.utils.neural_data_analysis.checkEpochedSpikesTimes(
        spikes_times=spikes_times, trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
    )
except ValueError:
    raise
print("Checks passed")

#%%
# Set estimation hyperparameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

n_latents = 2
em_max_iter = 30
model_save_filename = "../results/simulation_model.pickle"

#%%
# Get parameters
# ~~~~~~~~~~~~~~

#%%
# Build default parameters specificiation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(
    n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents,
    em_max_iter=em_max_iter)

#%%
# Get parameters and kernels types from the parameters specification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
    n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
    trials_start_times=trials_start_times,
    trials_end_times=trials_end_times,
    default_params_spec=default_params_spec)

#%%
# Create kernels, a model and set its initial parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%%
# Build kernels
# ^^^^^^^^^^^^^
kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
kernels = svGPFA.utils.miscUtils.buildKernels(
    kernels_types=kernels_types, kernels_params=kernels_params0)

#%%
# Create model
# ^^^^^^^^^^^^
model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.\
    buildModelPyTorch(kernels=kernels)

#%%
# Set initial parameters
# ^^^^^^^^^^^^^^^^^^^^^^
model.setParamsAndData(
    measurements=spikes_times,
    initial_params=params["initial_params"],
    eLLCalculationParams=params["ell_calculation_params"],
    priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])


#%%
# Maximize the Lower Bound
# ~~~~~~~~~~~~~~~~~~~~~~~~
# (Warning: with the parameters above, this step takes around 5 minutes for 30 em_max_iter)
#

svEM = svGPFA.stats.svEM.SVEM_PyTorch()
tic = time.perf_counter()
lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
    svEM.maximize(model=model, optim_params=params["optim_params"],
                  method=params["optim_params"]["optim_method"],
                  out=sys.stdout)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

resultsToSave = {"lowerBoundHist": lowerBoundHist,
                 "elapsedTimeHist": elapsedTimeHist,
                 "terminationInfo": terminationInfo,
                 "iterationModelParams": iterationsModelParams,
                 "model": model}
with open(model_save_filename, "wb") as f:
    pickle.dump(resultsToSave, f)
print("Saved results to {:s}".format(model_save_filename))

#%%
# ..  with open(model_save_filename, "rb") as f:
#        load_res = pickle.load(f)
#    lowerBoundHist = load_res["lowerBoundHist"]
#    elapsedTimeHist = load_res["elapsedTimeHist"]
#    model = load_res["model"]

#%%
# Goodness-of-fit analysis
# ------------------------

#%%
# Set goodness-of-fit variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ks_test_gamma = 10                                 # number of simulations for the KS test numerical correction
trial_for_gof = 0
cluster_id_for_gof = 1
n_time_steps_IF = 100

trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_IF)

#%%
# Calculate expected intensity function values (for KS test and IF plots)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with torch.no_grad():
    epm_cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
cif_values_GOF = epm_cif_values[trial_for_gof][cluster_id_for_gof]

#%%
# KS time-rescaling GOF test
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
trial_times_GOF = trials_times[trial_for_gof, :, 0]
spikes_times_GOF = spikes_times[trial_for_gof][cluster_id_for_gof].numpy()
if len(spikes_times_GOF) == 0:
    raise ValueError("No spikes found for goodness-of-fit analysis")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
        gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(
            spikes_times=spikes_times_GOF, cif_times=trial_times_GOF,
            cif_values=cif_values_GOF, gamma=ks_test_gamma)
title = "Trial {:d}, Neuron {:d}".format(trial_for_gof, cluster_id_for_gof)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig

#%%
# ROC predictive analysis
# ~~~~~~~~~~~~~~~~~~~~~~~

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
        spikes_times=spikes_times_GOF,
        cif_times=trial_times_GOF,
        cif_values=cif_values_GOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
    fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
fig

#%%
# Plotting
# --------

#%%
# Imports for plotting
# ~~~~~~~~~~~~~~~~~~~~

import numpy as np
import plotly.express as px

#%%
# Set plotting parameters
# ~~~~~~~~~~~~~~~~~~~~~~~
neuron_to_plot = 0
latent_to_plot = 0
trials_colorscale = "hot"

#%%
# Set trials colors
# ^^^^^^^^^^^^^^^^^
trials_colors = px.colors.sample_colorscale(
    colorscale=trials_colorscale, samplepoints=n_trials,
    colortype="rgb")
trials_colors_patterns = [f"rgba{trial_color[3:-1]}, {{:f}})"
                          for trial_color in trials_colors]

#%%
# Set trials ids
# ^^^^^^^^^^^^^^
trials_ids = [r for r in range(n_trials)]


#%%
# Lower bound history
# ~~~~~~~~~~~~~~~~~~~
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
    lowerBoundHist=lowerBoundHist)
fig

#%%
# Latent across trials
# ~~~~~~~~~~~~~~~~~~~~
test_mu_k, test_var_k = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
    times=trials_times.numpy(), latentsMeans=test_mu_k,
    latentsSTDs=torch.sqrt(test_var_k), latentToPlot=latent_to_plot,
    trials_colors_patterns=trials_colors_patterns, xlabel="Time (msec)")
fig

#%%
# Orthonormalized latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, testVarK = model.predictLatents(times=trials_times)
testMuK_np = [testMuK[r].detach().numpy() for r in range(len(testMuK))]
estimatedC, estimatedD = model.getSVEmbeddingParams()
estimatedC_np = estimatedC.detach().numpy()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
    trials_times=trials_times,
    latentsMeans=testMuK_np,
    C=estimatedC_np,
    trials_ids=trials_ids,
    latentToPlot=latent_to_plot,
    trials_colors=trials_colors,
    xlabel="Time (msec)")
fig

#%%
# Embedding
# ~~~~~~~~~
embedding_means, embedding_vars = model.predictEmbedding(times=trials_times)
embedding_means = embedding_means.detach().numpy()
embedding_vars = embedding_vars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(times=trials_times.numpy(), embeddingsMeans=embedding_means[:,:,neuron_to_plot], embeddingsSTDs=np.sqrt(embedding_vars[:,:,neuron_to_plot]), trials_colors_patterns=trials_colors_patterns, title=title)
fig

#%%
# IFs
# ~~~~
with torch.no_grad():
    ePos_IF_values = model.computeExpectedPosteriorCIFs(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times, cif_values=ePos_IF_values,
    trials_ids=trials_ids, neuron_index=neuron_to_plot,
    trials_colors=trials_colors)
fig


#%%
# Embedding parameters
# ~~~~~~~~~~~~~~~~~~~~
estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingParams(C=estimatedC.numpy(), d=estimatedD.numpy())
fig


#%%
# Kernels parameters
# ~~~~~~~~~~~~~~~~~~

kernelsParams = model.getKernelsParams()
kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
    kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
fig

#%%
# .. raw:: html
#
#    <h3><font color="red">To run the Python script or Jupyter notebook below,
#    please download them to the <i>examples/sphinx_gallery</i> folder of the
#    repository and execute them from there.</font></h3>

# sphinx_gallery_thumbnail_path = '_static/model.png'
