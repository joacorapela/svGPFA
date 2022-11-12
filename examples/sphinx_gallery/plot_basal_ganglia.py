
"""
Basal ganglia recordings from a mouse performing a bandit task
==============================================================

In this notebook we use data recorded from the basal ganglia of a mouse
performing a bandit task from the to estimate an svGPFA model

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
import configparser
import pandas as pd

import gcnu_common.utils.neuralDataAnalysis
import gcnu_common.stats.pointProcesses.tests
import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


#%%
# 1.2 Get spikes times
# ~~~~~~~~~~~~~~~~~~~~
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


events_times_filename = "../data/s008_tab_m1113182_LR__20210516_173815__probabilistic_switching.df.csv"
events_times = pd.read_csv(events_times_filename)
trials_indices = [r for r in range(len(events_times))
                  if events_times.iloc[r]["block_type_index"]
                  in block_types_indices]
spikes_times, neurons_indices = \
    gcnu_common.utils.neuralDataAnalysis.removeUnitsWithLessSpikesThanThrInAnyTrial(
        spikes_times=spikes_times,
        min_nSpikes_perNeuron_perTrial=min_nSpikes_perNeuron_perTrial)
spikes_times = [[torch.tensor(spikes_times[r][n])
                 for n in range(len(spikes_times[r]))]
                for r in range(len(spikes_times))]

#%%
# 1.3 Set estimation hyperparameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_latents = 10
em_max_iter_dyn = 200
est_init_number = 39
n_trials = len(spikes_times)
n_neurons = len(spikes_times[0])
est_init_config_filename_pattern = "../init/{:08d}_estimation_metaData.ini"
model_save_filename = "../results/basal_ganglia_model.pickle"

#%%
# 1.4 Get parameters
# ~~~~~~~~~~~~~~~~~~

#%%
# Dynamic parameters specification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dynamic_params_spec = {"optim_params": {"em_max_iter": em_max_iter_dyn}}

#%%
# Config file parameters specification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
    n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
    trials_start_times=trials_start_times,
    trials_end_times=trials_end_times,
    dynamic_params_spec=dynamic_params_spec,
    config_file_params_spec=config_file_params_spec)

#%%
# 1.5 Create kernels, a model and set its initial parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
# 1.6 Maximize the Lower Bound
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ..
#   est_res_number = 91693124
#   model_save_filename_pattern = "../results/{:08d}_estimatedModel.pickle"
#   
#   model_save_filename = model_save_filename_pattern.format(est_res_number)
#   with open(model_save_filename, "rb") as f:
#       estResults = pickle.load(f)
#   lowerBoundHist = estResults["lowerBoundHist"]
#   elapsedTimeHist = estResults["elapsedTimeHist"]
#   model = estResults["model"]

#%%
# 2 Plotting
# ----------

#%%
# 2.1 Imports for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import plotly.express as px
import svGPFA.plot.plotUtilsPlotly

#%%
# 2.2 Set plotting parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_time_steps_CIF = 100
latent_to_plot = 0
neuron_to_plot = 0
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

trials_labels = np.array([str(i) for i in trials_indices])
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
marked_events = np.column_stack((centerIn_times, centerOut_times, sideIn_times, trialEnd_times))

trials_choices = events_times.iloc[trials_indices][trial_choice_column_name].to_numpy()
trials_rewarded = events_times.iloc[trials_indices][trial_rewarded_column_name].to_numpy()
trials_annotations = {"choice": trials_choices,
                      "rewarded": trials_rewarded,
                      "choice_prev": np.insert(trials_choices[:-1], 0,
                                               np.NAN),
                      "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
                                                 np.NAN)}
trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_CIF)

#%%
# 2.3 Lower bound history
# ^^^^^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
    lowerBoundHist=lowerBoundHist)
fig_filename_pattern = "{:s}_lowerBoundHistVSIterNo.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix, "html"))
fig

#%%
# 2.4 Latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, testVarK = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
    times=trials_times.numpy(),
    latentsMeans=testMuK,
    latentsSTDs=torch.sqrt(testVarK),
    trials_labels=trials_labels,
    latentToPlot=latent_to_plot,
    trials_colors_patterns=trials_colors_patterns,
    xlabel="Time (msec)")
fig_filename_pattern = "{:s}_latent{:d}.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            latent_to_plot, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           latent_to_plot, "html"))
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
    align_event=align_times, marked_events=marked_events,
    marked_events_colors=marked_events_colors,
    trials_labels=trials_labels,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors,
    xlabel="Time (msec)")
fig_filename_pattern = "{:s}_orthonormalized_latent{:d}.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            latent_to_plot, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           latent_to_plot, "html"))
fig

#%%
# 2.7 Joint evolution of first three orthonormalized latents
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
    trials_times=trials_times.numpy(), latentsMeans=testMuK_np,
    C=estimatedC_np, latentsToPlot=ortho_latents_to_plot,
    align_event=align_times, marked_events=marked_events,
    marked_events_colors=marked_events_colors,
    trials_labels=trials_labels,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors)
ortho_latents_to_plot_str = "".join(str(i)+"_" for i in ortho_latents_to_plot)
fig_filename_pattern = "{:s}_orthonormalized_latents{:s}.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            ortho_latents_to_plot_str, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           ortho_latents_to_plot_str, "html"))
fig

#%%
# 2.6 Embedding
# ~~~~~~~~~~~~~
embeddingMeans, embeddingVars = model.predictEmbedding(times=trials_times)
embeddingMeans = embeddingMeans.detach().numpy()
embeddingVars = embeddingVars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
    times=trials_times.numpy(),
    embeddingsMeans=embeddingMeans[:, :, neuron_to_plot],
    embeddingsSTDs=np.sqrt(embeddingVars[:, :, neuron_to_plot]),
    trials_colors_patterns=trials_colors_patterns,
    title=title)
fig_filename_pattern = "{:s}_embedding_neuron{:d}.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            neuron_to_plot, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           neuron_to_plot, "html"))
fig

#%%
# 2.7 Intensity function
# ~~~~~~~~~~~~~~~~~~~~~~
with torch.no_grad():
    cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times, cif_values=cif_values,
    neuron_index=neuron_to_plot, spikes_times=spikes_times,
    align_event=centerOut_times, marked_events=marked_events,
    marked_events_colors=marked_events_colors, trials_labels=trials_labels,
    trials_annotations=trials_annotations,
    trials_colors=trials_colors,
)
fig_filename_pattern = "{:s}_intensity_function_neuron{:d}.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            neuron_to_plot, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           neuron_to_plot, "html"))
fig

#%%
# 2.8 Embedding parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~
estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedEmbeddingParams(
    C=estimatedC.numpy(), d=estimatedD.numpy())
fig_filename_pattern = "{:s}_orthonormalized_embedding_params.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix, "html"))
fig

#%%
# 2.9 Kernels parameters
# ~~~~~~~~~~~~~~~~~~~~~~
kernelsParams = model.getKernelsParams()
kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
    kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
fig_filename_pattern = "{:s}_kernels_params.{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix, "html"))
fig

#%%
# 3 Goodness of fit (GOF)
# -----------------------
trial_GOF = 0
neuron_GOF = 0
cif_values_GOF = cif_values[trial_GOF][neuron_GOF]
trial_times_GOF = trials_times[trial_GOF, :, 0]
spikes_times_GOF = spikes_times[trial_GOF][neuron_to_plot].numpy()

#%%
# 3.1 KS time-rescaling GOF test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ks_test_gamma = 10
if len(spikes_times_GOF) > 0:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF,
                                                                                      cif_times=trial_times_GOF,
                                                                                      cif_values=cif_values_GOF,
                                                                                      gamma=ks_test_gamma)
title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
    trial_GOF, neuron_GOF, len(spikes_times_GOF))
fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig_filename_pattern = \
    "{:s}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:03d}..{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            trial_GOF, neuron_GOF, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           trial_GOF, neuron_GOF, "png"))
fig

#%%
# 3.2 ROC predictive analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
        spikes_times=spikes_times_GOF,
        cif_times=trial_times_GOF,
        cif_values=cif_values_GOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
    fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
fig_filename_pattern = "{:s}_predictive_analysis_trial{:03d}_neuron{:03d}..{:s}"
fig.write_image(fig_filename_pattern.format(fig_filename_prefix,
                                            trial_GOF, neuron_GOF, "png"))
fig.write_html(fig_filename_pattern.format(fig_filename_prefix,
                                           trial_GOF, neuron_GOF, "png"))
fig

#%%
# .. raw:: html
#
#    <h3><font color="red">To run the Python script or Jupyter notebook below,
#    please download them to the <i>examples/sphinx_gallery</i> folder of the
#    repository and execute them from there.</font></h3>

# sphinx_gallery_thumbnail_path = '_static/basal_ganglia.png'
