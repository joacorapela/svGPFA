
"""
IBL's Recordings from the striatum of a mouse performing a visual discrimination task
=====================================================================================

In this notebook we download publically available data from the International
Brain Laboratory, epoch it, run svGPFA and plot its results.

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

from one.api import ONE
import brainbox.io.one
import iblUtils

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
eID = "ebe2efe3-e8a1-451a-8947-76ef42427cc9"
probe_id = "probe00"
epoch_event_name = "response_times"
clusters_ids_filename = "../init/clustersIDs_40_64.csv"
trials_ids_filename = "../init/trialsIDs_0_89.csv"
min_neuron_trials_avg_firing_rate = 0.1

#%%
# Set estimation hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_latents = 10
em_max_iter_dyn = 200
common_n_ind_points = 15
est_init_number = 0
est_init_config_filename_pattern = "../init/{:08d}_IBL_estimation_metaData.ini"
model_save_filename = "../results/stiatum_ibl_model.pickle"

#%%
# Epoch
# -----

#%%
# Download data
# ^^^^^^^^^^^^^
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
spikes = one.load_object(eID, 'spikes', 'alf/probe00/pykilosort')
clusters = one.load_object(eID, "clusters", f"alf/{probe_id}/pykilosort")
trials = one.load_object(eID, 'trials')

#%%
# Extract variables of interest
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
clusters_ids = np.unique(spikes.clusters.tolist())
n_clusters = len(clusters_ids)
channels_for_clusters_ids = clusters.channels
els = brainbox.io.one.load_channel_locations(eID, one=one)
locs_for_clusters_ids = els[probe_id]["acronym"][channels_for_clusters_ids].tolist()

#%%
# Epoch spikes times
# ^^^^^^^^^^^^^^^^^^
epoch_times = trials[epoch_event_name]
n_trials = len(epoch_times)

epoch_start_times = [trials["intervals"][r][0] for r in range(n_trials)]
epoch_end_times = [trials["intervals"][r][1] for r in range(n_trials)]

spikes_times_by_neuron = []
for cluster_id in clusters_ids:
    print(f"Processing cluster {cluster_id}")
    neuron_spikes_times = spikes.times[spikes.clusters==cluster_id]
    n_epoched_spikes_times = iblUtils.epoch_neuron_spikes_times(
        neuron_spikes_times=neuron_spikes_times,
        epoch_times = epoch_times,
        epoch_start_times=epoch_start_times,
        epoch_end_times=epoch_end_times)
    spikes_times_by_neuron.append(n_epoched_spikes_times)
spikes_times = [[spikes_times_by_neuron[n][r] for n in range(n_clusters)]
                for r in range(n_trials)]

trials_start_times = [epoch_start_times[r]-epoch_times[r] for r in range(n_trials)]
trials_end_times = [epoch_end_times[r]-epoch_times[r] for r in range(n_trials)]
n_neurons = len(spikes_times[0])

#%%
# Subset epoched spikes times
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# subset selected_clusters_ids
selected_clusters_ids = np.genfromtxt(clusters_ids_filename, dtype=np.uint64)

spikes_times = iblUtils.subset_clusters_ids_data(
    selected_clusters_ids=selected_clusters_ids,
    clusters_ids=clusters_ids,
    spikes_times=spikes_times,
)
n_neurons = len(spikes_times[0])
n_trials = len(spikes_times)
trials_ids = np.arange(n_trials)

# subset selected_trials_ids
selected_trials_ids = np.genfromtxt(trials_ids_filename, dtype=np.uint64)
spikes_times, trials_start_times, trials_end_times = \
        iblUtils.subset_trials_ids_data(
            selected_trials_ids=selected_trials_ids,
            trials_ids=trials_ids,
            spikes_times=spikes_times,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)
n_trials = len(spikes_times)

# remove units with low spike rate
neurons_indices = torch.arange(n_neurons)
trials_durations = [trials_end_times[i] - trials_start_times[i]
                    for i in range(n_trials)]
spikes_times, neurons_indices = \
    gcnu_common.utils.neural_data_analysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
        spikes_times=spikes_times, neurons_indices=neurons_indices,
        trials_durations=trials_durations,
        min_neuron_trials_avg_firing_rate=min_neuron_trials_avg_firing_rate)
selected_clusters_ids = [selected_clusters_ids[i] for i in neurons_indices]

n_trials = len(spikes_times)
n_neurons = len(spikes_times[0])

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
cluster_id_for_gof = 41
n_time_steps_IF = 100

cluster_id_for_gof_index = torch.nonzero(torch.IntTensor(selected_clusters_ids)==cluster_id_for_gof)
trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_IF)

#%%
# Calculate expected intensity function values (for KS test and IF plots)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with torch.no_grad():
    cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
cif_values_GOF = cif_values[trial_for_gof][cluster_id_for_gof_index]

#%%
# KS time-rescaling GOF test
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
trial_times_GOF = trials_times[trial_for_gof, :, 0]
spikes_times_GOF = spikes_times[trial_for_gof][cluster_id_for_gof_index]
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
    trial_for_gof, cluster_id_for_gof, len(spikes_times_GOF))
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
latents_to_3D_plot = [0, 2, 4]
cluster_id_to_plot = 41
trial_to_plot = 0
choices_colors_patterns = ["rgba(0,0,255,{:f})", "rgba(255,0,0,{:f})"]
align_event_name = "response_times"
events_names = ["stimOn_times", "response_times", "stimOff_times"]
events_colors = ["magenta", "green", "black"]
events_markers = ["circle", "circle", "circle"]

cluster_id_to_plot_index = torch.nonzero(torch.IntTensor(selected_clusters_ids)==cluster_id_to_plot)


n_trials = len(spikes_times)

trials_choices = [trials["choice"][trial_id] for trial_id in selected_trials_ids]
trials_rewarded = [trials["feedbackType"][trial_id] for trial_id in selected_trials_ids]
trials_contrast = [trials["contrastRight"][trial_id] 
                   if not np.isnan(trials["contrastRight"][trial_id])
                   else trials["contrastLeft"][trial_id]
                   for trial_id in selected_trials_ids]
trials_colors_patterns = [choices_colors_patterns[0]
                          if trials_choices[r] == -1
                          else choices_colors_patterns[1]
                          for r in range(n_trials)]
trials_colors = [trial_color_pattern.format(1.0)
                 for trial_color_pattern in trials_colors_patterns]
trials_annotations = {"choice": trials_choices,
                      "rewarded": trials_rewarded,
                      "contrast": trials_contrast,
                      "choice_prev": np.insert(trials_choices[:-1], 0, np.NAN),
                      "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
                                                 np.NAN)}

events_times = []
for event_name in events_names:
    events_times.append([trials[event_name][trial_id]
                         for trial_id in selected_trials_ids])

marked_events_times, marked_events_colors, marked_events_markers = \
    iblUtils.buildMarkedEventsInfo(events_times=events_times,
                                   events_colors=events_colors,
                                   events_markers=events_markers)

align_event_times = [trials[align_event_name][trial_id]
                     for trial_id in selected_trials_ids]

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
    trials_ids=selected_trials_ids,
    latentToPlot=latent_to_plot,
    trials_colors_patterns=trials_colors_patterns,
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
    align_event_times=align_event_times,
    marked_events_times=marked_events_times,
    marked_events_colors=marked_events_colors,
    marked_events_markers=marked_events_markers,
    trials_colors=trials_colors,
    trials_annotations=trials_annotations,
    C=estimatedC_np, trials_ids=selected_trials_ids,
    xlabel="Time (msec)")
fig

#%%
# Plot 3D scatter plot of orthonormalized latents
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
    trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
    C=estimatedC_np, trials_ids=selected_trials_ids,
    latentsToPlot=latents_to_3D_plot,
    align_event_times=align_event_times,
    marked_events_times=marked_events_times,
    marked_events_colors=marked_events_colors,
    marked_events_markers=marked_events_markers,
    trials_colors=trials_colors,
    trials_annotations=trials_annotations)
fig

#%%
# Plot embedding
# ^^^^^^^^^^^^^^
embeddingMeans, embeddingVars = model.predictEmbedding(times=trials_times)
embeddingMeans = embeddingMeans.detach().numpy()
embeddingVars = embeddingVars.detach().numpy()
title = "Neuron {:d}".format(cluster_id_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
    times=trials_times.numpy(),
    embeddingsMeans=embeddingMeans[:, :, cluster_id_to_plot_index],
    embeddingsSTDs=np.sqrt(embeddingVars[:, :, cluster_id_to_plot_index]),
    trials_colors_patterns=trials_colors_patterns,
    title=title)
fig

#%%
# Plot intensity functions for one neuron and all trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
title = f"Cluster ID: {clusters_ids[cluster_id_to_plot_index]}, Region: {locs_for_clusters_ids[cluster_id_to_plot]}"
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times,
    cif_values=cif_values,
    neuron_index=cluster_id_to_plot_index,
    spikes_times=spikes_times,
    trials_ids=selected_trials_ids,
    align_event_times=align_event_times,
    marked_events_times=marked_events_times,
    marked_events_colors=marked_events_colors,
    marked_events_markers=marked_events_markers,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors,
    title=title)
fig

#%%
# Plot orthonormalized embedding parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
hovertemplate = "value: %{y}<br>" + \
                "neuron index: %{x}<br>" + \
                "%{text}"
text = [f"cluster_id: {cluster_id}" for cluster_id in selected_clusters_ids]
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

# sphinx_gallery_thumbnail_path = '_static/ibl_logo.png'
