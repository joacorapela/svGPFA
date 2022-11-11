
"""
Simulated data and default params
=================================

In this notebook we use simulated data to estimate an svGPFA model using the default initial parameters.

1. Estimate model
-----------------

1.1 Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import sys
import time
import warnings
import torch
import pickle

import svGPFA.stats.kernels
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import gcnu_common.stats.pointProcesses.tests


#%%
# 1.2 Get spikes times
# ~~~~~~~~~~~~~~~~~~~~~
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
# 1.3 Set estimation hyperparameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

n_latents = 2
em_max_iter = 30

#%%
# 1.4 Get parameters
# ~~~~~~~~~~~~~~~~~~

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
# 1.5 Create kernels, a model and set its initial parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
# 1.6 Maximize the Lower Bound
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#%%
# ..
#     with open("/tmp/estimationResults.pickle", "rb") as f:
#         load_res = pickle.load(f)
#     lowerBoundHist = load_res["lowerBoundHist"]
#     elapsedTimeHist = load_res["elapsedTimeHist"]
#     model = load_res["model"]

#%%
# 2 Plotting
# ----------

#%%
# 2.1 Imports for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import plotly.express as px
import svGPFA.plot.plotUtilsPlotly

#%%
# 2.2 Set plotting parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
neuron_to_plot = 0
latent_to_plot = 0
n_time_steps_CIF = 100
trials_colorscale = "hot"

#%%
# Build times to plot
# ^^^^^^^^^^^^^^^^^^^
trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_CIF)

#%%
# Set trials colors
# ^^^^^^^^^^^^^^^^^
trials_colors = px.colors.sample_colorscale(
    colorscale=trials_colorscale, samplepoints=n_trials,
    colortype="rgb")
trials_colors_patterns = [f"rgba{trial_color[3:-1]}, {{:f}})" for trial_color in trials_colors]

#%%
# Set trials labels
# ^^^^^^^^^^^^^^^^^
trials_labels = [str(r) for r in range(n_trials)]


#%%
# 2.3 Lower bound history
# ~~~~~~~~~~~~~~~~~~~~~~~
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
fig

#%%
# 2.4 Latent across trials
# ~~~~~~~~~~~~~~~~~~~~~~~~
test_mu_k, test_var_k = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(times=trials_times.numpy(), latentsMeans=test_mu_k, latentsSTDs=torch.sqrt(test_var_k), latentToPlot=latent_to_plot, trials_colors_patterns=trials_colors_patterns, xlabel="Time (msec)")
fig

#%%
# 2.5 Orthonormalized latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, testVarK = model.predictLatents(times=trials_times)
testMuK_np = [testMuK[r].detach().numpy() for r in range(len(testMuK))]
estimatedC, estimatedD = model.getSVEmbeddingParams()
estimatedC_np = estimatedC.detach().numpy()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
    trials_times=trials_times,
    latentsMeans=testMuK_np, latentToPlot=latent_to_plot,
    C=estimatedC_np,
    trials_labels=trials_labels,
    trials_colors=trials_colors,
    xlabel="Time (msec)")
fig

#%%
# 2.6 Embedding
# ~~~~~~~~~~~~~
embedding_means, embedding_vars = model.predictEmbedding(times=trials_times)
embedding_means = embedding_means.detach().numpy()
embedding_vars = embedding_vars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(times=trials_times.numpy(), embeddingsMeans=embedding_means[:,:,neuron_to_plot], embeddingsSTDs=np.sqrt(embedding_vars[:,:,neuron_to_plot]), trials_colors_patterns=trials_colors_patterns, title=title)
fig

#%%
# 2.7 CIFs
# ~~~~~~~~
with torch.no_grad():
    ePos_CIF_values = model.computeExpectedPosteriorCIFs(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times, cif_values=ePos_CIF_values,
    neuron_index=neuron_to_plot, trials_colors=trials_colors)
fig


#%%
# 2.8 Embedding parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~
estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingParams(C=estimatedC.numpy(), d=estimatedD.numpy())
fig


#%%
# 2.9 Kernels parameters
# ~~~~~~~~~~~~~~~~~~~~~~

kernelsParams = model.getKernelsParams()
kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
    kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
fig

#%%
# 3 Goodness of fit (GOF)
# -----------------------
trial_GOF = 0
neuron_GOF = 0
trial_times_GOF = trials_times[trial_GOF, :, 0]
spikes_times_GOF = spikes_times[trial_GOF][neuron_GOF].numpy()

#%%
# 3.1 KS time-rescaling GOF test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ksTest_gamma = 20                                 # number of simulations for the KS test numerical correction
with torch.no_grad():
    epm_cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
cif_values_GOF = epm_cif_values[trial_GOF][neuron_GOF]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF, cif_times=trial_times_GOF, cif_values=cif_values_GOF, gamma=ksTestGamma)
title = "Trial {:d}, Neuron {:d}".format(trial_GOF, neuron_GOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig

#%%
# 3.2 ROC predictive analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
    spikes_times=spikes_times_GOF,
    cif_times=trial_times_GOF,
    cif_values=cif_values_GOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
fig

# sphinx_gallery_thumbnail_path = '_static/model.png'
