
"""
Recordings from the primary motor and dorsal premotor cortex striatum of a monkey performing a delayed-reach task
=================================================================================================================

In this notebook we download publically available data from Dandi, epoch it, run svGPFA and plot its results.

"""

#%%
# Setup environment
# -----------------

#%%
# Import required packages
# ^^^^^^^^^^^^^^^^^^^^^^^^

import sys
import warnings
import pickle
import time
import configparser
import numpy as np
import pandas as pd
import torch

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import gcnu_common.utils.neural_data_analysis
import gcnu_common.stats.pointProcesses.tests
import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import svGPFA.plot.plotUtilsPlotly

#%%
# Set data parameters
# ^^^^^^^^^^^^^^^^^^^
dandiset_id = "000140"
filepath = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
epoch_event_name = "move_onset_time"

#%%
# Set estimation hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_latents = 10
em_max_iter_dyn = 100
common_n_ind_points = 15
est_init_number = 0
est_init_config_filename_pattern = "../init/{:08d}_jenkins_small_estimation_metaData.ini"
model_save_filename = \
    f"../results/jenkins_small_model_emMaxIter{em_max_iter_dyn}.pickle"

#%%
# Epoch
# -----

#%%
# Download data
# ^^^^^^^^^^^^^
with DandiAPIClient() as client:
	asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
	s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

io = NWBHDF5IO(s3_path, mode="r", driver="ros3")
nwbfile = io.read()
units = nwbfile.units
units_df = units.to_dataframe()

# n_neurons
n_neurons = units_df.shape[0]

# continuous spikes times
continuous_spikes_times = [None for r in range(n_neurons)]
for n in range(n_neurons):
    continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

# trials
trials_df = nwbfile.intervals["trials"].to_dataframe()

# n_trials
n_trials = trials_df.shape[0]

#%%
# Epoch spikes times
# ^^^^^^^^^^^^^^^^^^
trials_start_times = [None for r in range(n_trials)]
trials_end_times = [None for r in range(n_trials)]
spikes_times = [[None for n in range(n_neurons)] for r in range(n_trials)]
for n in range(n_neurons):
    for r in range(n_trials):
        epoch_start_time = trials_df.iloc[r]["start_time"]
        epoch_end_time = trials_df.iloc[r]["stop_time"]
        epoch_time = trials_df.iloc[r][epoch_event_name]
        spikes_times[r][n] = (continuous_spikes_times[n][
            np.logical_and(epoch_start_time <= continuous_spikes_times[n],
                           continuous_spikes_times[n] <= epoch_end_time)] -
            epoch_time)
        trials_start_times[r] = epoch_start_time - epoch_time
        trials_end_times[r] = epoch_end_time - epoch_time

#%%
# Check that spikes have been epoched correctly
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# Plot spikes 
# ^^^^^^^^^^^
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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
# Get parameters
# --------------
# Details on how to specify svGPFA parameters are provided `here <../params.html>`_

#%%
# Dynamic parameters specification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dynamic_params_spec = {
    "optim_params": {"em_max_iter": em_max_iter_dyn},
    "ind_points_locs_params0": {"common_n_ind_points": common_n_ind_points},
}

#%%
# Config file parameters specification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The configuration file appears `here <https://github.com/joacorapela/svGPFA/blob/master/examples/init/00000000_IBL_estimation_metaData.ini>`_

args_info = svGPFA.utils.initUtils.getArgsInfo()
est_init_config_filename = est_init_config_filename_pattern.format(
    est_init_number)
est_init_config = configparser.ConfigParser()
est_init_config.read(est_init_config_filename)

strings_dict = gcnu_common.utils.config_dict.GetDict(
    config=est_init_config).get_dict()
config_file_params_spec = \
    svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials,
        strings_dict=strings_dict, args_info=args_info)

#%%
# Get the parameters from the dynamic and configuration file parameter specifications
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
    n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
    trials_start_times=trials_start_times,
    trials_end_times=trials_end_times,
    dynamic_params_spec=dynamic_params_spec,
    config_file_params_spec=config_file_params_spec)

#%%
# Estimate svGPFA model
# ---------------------

#%%
# Create kernels, a model and set its initial parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# Build kernels
# ^^^^^^^^^^^^^
kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
kernels = svGPFA.utils.miscUtils.buildKernels(
    kernels_types=kernels_types, kernels_params=kernels_params0)

#%%
# Create model
# ^^^^^^^^^^^^
kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
    conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
    linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
    embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
    kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
    indPointsCovRep=indPointsCovRep)

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
# ^^^^^^^^^^^^^^^^^^^^^^^^
# (Warning: with the parameters above, this step takes around 5 minutes for 30 em_max_iter)

svEM = svGPFA.stats.svEM.SVEM_PyTorch()
tic = time.perf_counter()
lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
svEM.maximize(model=model, optim_params=params["optim_params"],
              method=params["optim_params"]["optim_method"], out=sys.stdout)
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
#        estResults = pickle.load(f)
#    lowerBoundHist = estResults["lowerBoundHist"]
#    elapsedTimeHist = estResults["elapsedTimeHist"]
#    model = estResults["model"]

#%%
# Goodness-of-fit analysis
# ------------------------

#%%
# Set goodness-of-fit variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ks_test_gamma = 10
trial_for_gof = 0
neuron_for_gof = 0
n_time_steps_IF = 100

trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_IF)

#%%
# Calculate expected intensity function values (for KS test and IF plots)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with torch.no_grad():
    cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
cif_values_GOF = cif_values[trial_for_gof][neuron_for_gof]

#%%
# KS time-rescaling GOF test
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
trial_times_GOF = trials_times[trial_for_gof, :, 0]
spikes_times_GOF = spikes_times[trial_for_gof][neuron_for_gof]
if len(spikes_times_GOF) == 0:
    raise ValueError("No spikes found for goodness-of-fit analysis")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
        gcnu_common.stats.pointProcesses.tests.\
        KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF,
            cif_times=trial_times_GOF, cif_values=cif_values_GOF,
            gamma=ks_test_gamma)

title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
    trial_for_gof, neuron_for_gof, len(spikes_times_GOF))
fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig

#%%
# ROC predictive analysis
# ^^^^^^^^^^^^^^^^^^^^^^^
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
# ^^^^^^^^^^^^^^^^^^^^

import numpy as np
import pandas as pd

#%%
# Set plotting variables
# ^^^^^^^^^^^^^^^^^^^^^^

latent_to_plot = 0
latents_to_3D_plot = [0, 1, 2]
neuron_to_plot = 0
trial_to_plot = 0
trials_ids = np.arange(n_trials)
neurons_ids = np.arange(n_neurons)
choices_colors_patterns = ["rgba(0,0,255,{:f})", "rgba(255,0,0,{:f})"]
align_event_name = "response_times"
events_names = ["target_on_time", "go_cue_time", "move_onset_time"]
events_colors = ["magenta", "green", "black"]
events_markers = ["circle", "circle", "circle"]

#%%
# Plot lower bound history
# ^^^^^^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
    elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
fig

#%%
# Plot estimated latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, testVarK = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
    times=trials_times.numpy(),
    latentsMeans=testMuK,
    latentsSTDs=torch.sqrt(testVarK),
    trials_ids=trials_ids,
    latentToPlot=latent_to_plot,
    xlabel="Time (msec)")
fig

#%%
# Plot orthonormalized estimated latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, _ = model.predictLatents(times=trials_times)
test_mu_k_np = [testMuK[r].detach().numpy() for r in range(len(testMuK))]
estimatedC, estimatedD = model.getSVEmbeddingParams()
estimatedC_np = estimatedC.detach().numpy()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
    trials_times=trials_times, latentsMeans=test_mu_k_np, latentToPlot=latent_to_plot,
    C=estimatedC_np, trials_ids=trials_ids, xlabel="Time (msec)")
fig

#%%
# Plot 3D scatter plot of orthonormalized latents
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
    trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
    C=estimatedC_np, trials_ids=trials_ids,
    latentsToPlot=latents_to_3D_plot)
fig

#%%
# Plot embedding
# ^^^^^^^^^^^^^^
embeddingMeans, embeddingVars = model.predictEmbedding(times=trials_times)
embeddingMeans = embeddingMeans.detach().numpy()
embeddingVars = embeddingVars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
    times=trials_times.numpy(),
    embeddingsMeans=embeddingMeans[:, :, neuron_to_plot],
    embeddingsSTDs=np.sqrt(embeddingVars[:, :, neuron_to_plot]),
    title=title)
fig

#%%
# Plot intensity functions for one neuron and all trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
title = f"Neuron: {neuron_to_plot}"
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times,
    cif_values=cif_values,
    neuron_index=neuron_to_plot,
    spikes_times=spikes_times,
    trials_ids=trials_ids,
    title=title)
fig

#%%
# Plot orthonormalized embedding parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
hovertemplate = "value: %{y}<br>" + \
                "neuron index: %{x}<br>" + \
                "%{text}"
text = [f"neuron: {neuron}" for neuron in neurons_ids]
estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedEmbeddingParams(
    C=estimatedC.numpy(), d=estimatedD.numpy(),
    hovertemplate=hovertemplate, text=text)
fig

#%%
# Plot kernel parameters
# ^^^^^^^^^^^^^^^^^^^^^^
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

# sphinx_gallery_thumbnail_path = '_static/npsl_logo.jpg'
