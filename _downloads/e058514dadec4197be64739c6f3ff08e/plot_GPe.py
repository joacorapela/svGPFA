
"""
Basal ganglia recordings from a mouse performing a bandit task
==============================================================

In this notebook we use data recorded from the basal ganglia of a mouse
performing a bandit task from the to estimate an svGPFA model

"""

#%%
# Estimate model
# --------------
# 
# Import required packages
# ^^^^^^^^^^^^^^^^^^^^^^^^

import sys
import time
import warnings
import torch
import pickle
import configparser
import pandas as pd

import gcnu_common.utils.neural_data_analysis
import gcnu_common.stats.pointProcesses.tests
import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import svGPFA.plot.plotUtilsPlotly


#%%
# Get spikes times
# ^^^^^^^^^^^^^^^^
block_types_indices = [0]
region_spikes_times_filename_pattern = "../data/00000000_regionGPe_blockTypeIndices0_spikes_times_epochedaligned__last_center_out.{:s}"
min_nSpikes_perNeuron_perTrial = 1

region_spikes_times_filename = \
    region_spikes_times_filename_pattern.format("pickle")
with open(region_spikes_times_filename, "rb") as f:
    loadRes = pickle.load(f)
spikes_times = loadRes["spikes_times"]
trials_start_times = loadRes["trials_start_times"]
trials_end_times = loadRes["trials_end_times"]


events_times_filename = ("../data/s008_tab_m1113182_LR__20210516_173815__"
                         "probabilistic_switching.df.csv")
events_times = pd.read_csv(events_times_filename)
trials_indices = [r for r in range(len(events_times))
                  if events_times.iloc[r]["block_type_index"]
                  in block_types_indices]
spikes_times, neurons_indices = gcnu_common.utils.neural_data_analysis.\
    removeUnitsWithLessSpikesThanThrInAnyTrial(
        spikes_times=spikes_times,
        min_nSpikes_perNeuron_perTrial=min_nSpikes_perNeuron_perTrial)
spikes_times = [[torch.tensor(spikes_times[r][n])
                 for n in range(len(spikes_times[r]))]
                for r in range(len(spikes_times))]
n_trials = len(spikes_times)
n_neurons = len(spikes_times[0])

#%%
# Check that spikes have been epoched correctly
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# Plot spikes 
# """""""""""
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
# """"""""""""""""""""""""""""""""
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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_latents = 10
em_max_iter_dyn = 200
est_init_number = 40
est_init_config_filename_pattern = "../init/{:08d}_estimation_metaData.ini"
model_save_filename = "../results/basal_ganglia_model.pickle"

#%%
# Get parameters
# ^^^^^^^^^^^^^^
# Details on how to specify svGPFA parameters are provided `here <../params.html>`_

#%%
# Dynamic parameters specification
# """"""""""""""""""""""""""""""""
dynamic_params_spec = {"optim_params": {"em_max_iter": em_max_iter_dyn}}

#%%
# Config file parameters specification
# """"""""""""""""""""""""""""""""""""
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
# Finally, get the parameters from the dynamic and configuration file parameter specifications
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
    n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
    trials_start_times=trials_start_times,
    trials_end_times=trials_end_times,
    dynamic_params_spec=dynamic_params_spec,
    config_file_params_spec=config_file_params_spec)

#%%
# Create kernels, a model and set its initial parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# Build kernels
# """""""""""""
kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
kernels = svGPFA.utils.miscUtils.buildKernels(
    kernels_types=kernels_types, kernels_params=kernels_params0)

#%%
# Create model
# """"""""""""
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
# """"""""""""""""""""""
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
#       estResults = pickle.load(f)
#   lowerBoundHist = estResults["lowerBoundHist"]
#   elapsedTimeHist = estResults["elapsedTimeHist"]
#   model = estResults["model"]

#%%
# Goodness-of-fit analysis
# ------------------------

#%%
# Set goodness-of-fit variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ks_test_gamma = 10
trial_for_gof = 0
cluster_id_for_gof = 1
n_time_steps_IF = 100

cluster_id_for_gof_index = cluster_id_for_gof
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
spikes_times_GOF = spikes_times[trial_for_gof][cluster_id_for_gof_index].numpy()

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
# Imports for and plotting
# ^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import pandas as pd

#%%
# Set plotting parameters
# ^^^^^^^^^^^^^^^^^^^^^^^
latent_to_plot = 0
trial_to_plot = 0
ortho_latents_to_plot = (0, 1, 2)
events_times_filename = "../data/s008_tab_m1113182_LR__20210516_173815__probabilistic_switching.df.csv"
trial_choice_column_name = "choice"
trial_rewarded_column_name = "rewarded"
align_times_column_name = "aligned__last_center_out"
centerIn_times_column_name = "aligned__last_center_in"
centerOut_times_column_name = "aligned__last_center_out"
sideIn_times_column_name = "aligned__side_in_after_last_center_out"
marked_events_colors = ["yellow","magenta","cyan","black"]
fig_filename_prefix = "../figures/basal_ganglia_"

events_times = pd.read_csv(events_times_filename)
trials_choices = events_times.iloc[trials_indices][trial_choice_column_name].to_numpy()

trials_ids = np.array([i for i in trials_indices])
choices_colors_patterns = ["rgba(0,0,255,{:f})", "rgba(255,0,0,{:f})"]
trials_colors_patterns = [choices_colors_patterns[0]
                          if trials_choices[r] == -1
                          else choices_colors_patterns[1]
                          for r in range(n_trials)]
trials_colors = [trial_color_pattern.format(1.0)
                 for trial_color_pattern in trials_colors_patterns]
align_times = events_times.iloc[trials_indices][align_times_column_name].to_numpy()
centerIn_times = events_times.iloc[trials_indices][centerIn_times_column_name].to_numpy()
centerOut_times = events_times.iloc[trials_indices][centerOut_times_column_name].to_numpy()
sideIn_times = events_times.iloc[trials_indices][sideIn_times_column_name].to_numpy()
trialEnd_times = np.append(centerIn_times[1:], np.NAN)
marked_events_times = np.column_stack((centerIn_times, centerOut_times, sideIn_times, trialEnd_times))

trials_choices = events_times.iloc[trials_indices][trial_choice_column_name].to_numpy()
trials_rewarded = events_times.iloc[trials_indices][trial_rewarded_column_name].to_numpy()
trials_annotations = {"choice": trials_choices,
                      "rewarded": trials_rewarded,
                      "choice_prev": np.insert(trials_choices[:-1], 0,
                                               np.NAN),
                      "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
                                                 np.NAN)}
#%%
# Lower bound history
# ^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
    lowerBoundHist=lowerBoundHist)
fig

#%%
# Latent across trials
# ^^^^^^^^^^^^^^^^^^^^
testMuK, testVarK = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
    times=trials_times.numpy(),
    latentsMeans=testMuK,
    latentsSTDs=torch.sqrt(testVarK),
    trials_ids=trials_ids,
    latentToPlot=latent_to_plot,
    trials_colors_patterns=trials_colors_patterns,
    xlabel="Time (msec)")
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
    align_event_times=align_times, marked_events_times=marked_events_times,
    marked_events_colors=marked_events_colors,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors,
    xlabel="Time (msec)")
fig

#%%
# Joint evolution of first three orthonormalized latents
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
    trials_times=trials_times.numpy(), latentsMeans=testMuK_np,
    C=estimatedC_np, latentsToPlot=ortho_latents_to_plot,
    align_event_times=align_times, marked_events_times=marked_events_times,
    marked_events_colors=marked_events_colors,
    trials_ids=trials_ids,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors)
fig

#%%
# Embedding
# ^^^^^^^^^
embeddingMeans, embeddingVars = model.predictEmbedding(times=trials_times)
embeddingMeans = embeddingMeans.detach().numpy()
embeddingVars = embeddingVars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot_index)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
    times=trials_times.numpy(),
    embeddingsMeans=embeddingMeans[:, :, neuron_to_plot_index],
    embeddingsSTDs=np.sqrt(embeddingVars[:, :, neuron_to_plot_index]),
    trials_colors_patterns=trials_colors_patterns,
    title=title)
fig

#%%
# Intensity function
# ^^^^^^^^^^^^^^^^^^
with torch.no_grad():
    cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times, cif_values=cif_values,
    neuron_index=neuron_to_plot_index, spikes_times=spikes_times,
    align_event_times=centerOut_times, marked_events_times=marked_events_times,
    marked_events_colors=marked_events_colors, trials_ids=trials_ids,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors,
)
fig

#%%
# Embedding parameters
# ^^^^^^^^^^^^^^^^^^^^
estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedEmbeddingParams(
    C=estimatedC.numpy(), d=estimatedD.numpy())
fig

#%%
# Kernels parameters
# ^^^^^^^^^^^^^^^^^^
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

# sphinx_gallery_thumbnail_path = '_static/basal_ganglia.png'
