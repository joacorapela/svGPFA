#!/usr/bin/env python
# coding: utf-8

# # Contents:
# - [1 Estimation](#estimation)
# - [2 Plotting](#plotting)
# - [3 Goodness of fit](#GOF)

# # 1 Estimation <a class="anchor" id="estimation"></a>

# ## 1.1 Import requirements

# In[1]:


import time
import torch
import pickle

import svGPFA.stats.kernels
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import gcnu_common.stats.point_processes.tests


# ## 1.2 Set parameters

# In[5]:


n_latents = 2                                      # number of latents
n_neurons = 100                                    # number of neurons
n_trials = 15                                      # number of trials
prior_cov_reg_param = 1e-5                         # prior covariance regularization parameter
trial_start_time = 0.0                             # trial start time
trial_end_time = 1.0                               # trial end time
em_max_iter = 30                                   # maximum number of EM iterations


# ## 1.3 Load spikes times

# In[3]:


# spikesTimes should be a list of lists
# spikesTimes[r][n] is the list of spikes times of neuron n in trial r
simResFilename = "../../examples/data/32451751_simRes.pickle" # simulation results filename
with open(simResFilename, "rb") as f:
    simRes = pickle.load(f)
spikes_times = simRes["spikes"]


# ## 1.4 Create initial values of parameters

# In[6]:


# refer to this https://joacorapela.github.io/svGPFA/params.html# for a full description
# of different ways of specify svGPFA parameters

# get default params
default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
    n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents)

# over write some default params with dynamic params
dynamic_params = {}
# equal trials lengths
dynamic_params["data_structure_params"] = {
    "trials_start_time": trial_start_time,
    "trials_end_time":   trial_end_time,
}
# set maximum number of EM iterations
# set the prior covariance regularization parameter
dynamic_params["optim_params"] = {
    "em_max_iter": em_max_iter,
    "prior_cov_reg_param": prior_cov_reg_param,
}

# build the svGPFA parameters from their default and dynamic specification
initial_params, quad_params, kernels_types, optim_params = \
    svGPFA.utils.initUtils.getParams(
        n_trials=n_trials, n_neurons=n_neurons,
        default_params=default_params,
        dynamic_params=dynamic_params)
kernels_params0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["kernelsParams0"]
optim_method = optim_params["optim_method"]
prior_cov_reg_param = optim_params["prior_cov_reg_param"]


# ## 1.5 Create a model and set the initial value of parameters

# In[ ]:


# build kernels
kernels = svGPFA.utils.miscUtils.buildKernels(
    kernels_types=kernels_types, kernels_params=kernels_params0)

kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
    conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
    linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
    embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
    kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
    indPointsCovRep=indPointsCovRep)
model.setInitialParamsAndData(
    measurements=spikes_times,
    initialParams=initial_params,
    eLLCalculationParams=quad_params,
    priorCovRegParam=prior_cov_reg_param)


# ## 1.7 Maximize the Lower Bound
# <span style="color:red">(Warning: with the parameters above, this step takes around 5 minutes for 30 em_max_iter)</span>

# In[ ]:


svEM = svGPFA.stats.svEM.SVEM_PyTorch()
tic = time.perf_counter()
lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
    svEM.maximize(model=model, optim_params=optim_params, method=optim_method)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")


# # 2 Plotting <a class="anchor" id="plotting"></a>

# ## 2.1 Imports for plotting

# In[ ]:


import numpy as np
import pandas as pd
import sklearn.metrics
import svGPFA.plot.plotUtilsPlotly


# ## 2.2 Lower bound history

# In[ ]:


fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
fig.show()


# ## 2.3 Set neuron, latent and times to plot

# In[ ]:


neuronToPlot = 0
latentToPlot = 0
sampling_rate = 1000.0 # Hz
trial_times = torch.arange(trial_start_time, trial_end_time, 1.0/sampling_rate)


# ## 2.4 Latents

# In[ ]:


# plot estimated latent across trials
testMuK, testVarK = model.predictLatents(times=trial_times)
indPointsLocs = model.getIndPointsLocs()
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(times=trial_times.numpy(), latentsMeans=testMuK, latentsSTDs=torch.sqrt(testVarK), indPointsLocs=indPointsLocs, latentToPlot=latentToPlot, xlabel="Time (msec)")
fig.show()


# ## 2.5 Embedding

# In[ ]:


embeddingMeans, embeddingVars = model.predictEmbedding(times=trial_times)
embeddingMeans = embeddingMeans.detach().numpy()
embeddingVars = embeddingVars.detach().numpy()
title = "Neuron {:d}".format(neuronToPlot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(times=trial_times.numpy(), embeddingsMeans=embeddingMeans[:,:,neuronToPlot], embeddingsSTDs=np.sqrt(embeddingVars[:,:,neuronToPlot]), title=title)
fig.show()


# ## 2.6 CIFs

# In[ ]:


with torch.no_grad():
    ePosCIFValues = model.computeExpectedPosteriorCIFs(times=trial_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(times=trial_times, cif_values=ePosCIFValues, neuron_index=neuronToPlot)
fig.show()


# ## 2.7 Embedding parameters

# In[ ]:


estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingParams(C=estimatedC.numpy(), d=estimatedD.numpy())
fig.show()


# ## 2.8 Kernels parameters

# In[ ]:


kernelsParams = model.getKernelsParams()
kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
    kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
fig.show()


# # 3 Goodness of fit (GOF) <a class="anchor" id="GOF"></a>

# ## 3.1 Set trial and neuron for GOF assesment

# In[ ]:


trialForGOF = 0
neuronForGOF = 0


# ## 3.2 KS time-rescaling GOF test

# In[ ]:


ksTest_gamma = 20                                 # number of simulations for the KS test numerical correction
with torch.no_grad():
    epmcifValues = model.computeExpectedPosteriorCIFs(times=trial_times)
cifValuesKS = epmcifValues[trialForGOF][neuronForGOF]
spikesTimesKS = spikes_times[trialForGOF][neuronForGOF]
diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = gcnu_common.stats.point_processes.tests.KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=trial_times, cifValues=cifValuesKS, gamma=ksTest_gamma)
title = "Trial {:d}, Neuron {:d}".format(trialForGOF, neuronForGOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig.show()


# ## 3.3 ROC predictive analysis

# In[ ]:


dt = (trial_times[1] - trial_times[0]).item()
pk = cifValuesKS.detach().numpy() * dt
bins = pd.interval_range(start=trial_start_time, end=trial_end_time, periods=len(pk))
cutRes, _ = pd.cut(spikesTimesKS.tolist(), bins=bins, retbins=True)
Y = cutRes.value_counts().values
fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
roc_auc = sklearn.metrics.auc(fpr, tpr)
title = "Trial {:d}, Neuron {:d}".format(trialForGOF, neuronForGOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
fig.show()


# In[ ]:
