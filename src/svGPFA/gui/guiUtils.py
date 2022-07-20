import pdb
import sys
import os
import io
import base64
import math
import pickle
import importlib
from scipy.io import loadmat
import numpy as np
from numpy import array
import torch
import dash_html_components as html
import dash_core_components as dcc
sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svEM
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def guessTrialsLengths(spikesTimes):
    nTrials = len(spikesTimes)
    nNeurons = len(spikesTimes[0])
    guesses = [None for r in range(nTrials)]
    for r in range(nTrials):
        guessedLength = -np.inf
        for n in range(nNeurons):
            maxSpikeTime = torch.max(spikesTimes[r][n])
            if guessedLength<maxSpikeTime:
                guessedLength = maxSpikeTime
        guesses[r] = math.ceil(guessedLength)
    return guesses

def getKernelParams0Div(kernelType, namedKernelParams, latentID):
    if kernelType=="PeriodicKernel":
        aDiv = html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label(
                            id={
                                "type": "kernelsTypeParam0Label",
                                "latent": latentID
                            },
                            children="Kernel {:d} Type".format(latentID+1),
                        ),
                        html.Label(
                            id={
                                "type": "kernelTypeOfParam0",
                                "latent": latentID
                            },
                            children="PeriodicKernel",
                        ),
                    ], style={"display": "inline-block", "width": "30%"},
                ),
                html.Div(
                    children=[
                        html.Label(
                            id={
                                "type": "lengthScaleParam0Label",
                                "latent": latentID
                            },
                            children="Length Scale",
                        ),
                        dcc.Input(
                            id={
                                "type": "lengthScaleParam0",
                                "latent": latentID
                            },
                            type="number",
                            min=0,
                            required=True,
                            value=namedKernelParams["lengthScale"],
                        ),
                    ], style={"display": "inline-block", "width": "30%"}),
                    html.Div(
                        id={
                            "type": "periodParam0Container",
                            "latent": latentID
                        },
                        children=[
                            html.Label(
                                children="Period",
                            ),
                            dcc.Input(
                                id={
                                    "type": "periodParam0",
                                    "latent": latentID
                                },
                                type="number",
                                min=0,
                                required=True,
                                value=namedKernelParams["period"],
                            ),
                        ],
                        style={"display": "inline-block", "width": "30%"}
                    ),
        ])
    elif kernelType=="ExponentialQuadraticKernel":
        aDiv = html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label(
                            id={
                                "type": "kernelTypeParam0Label",
                                "latent": latentID
                            },
                            children="Kernel {:d} Type".format(latentID+1),
                        ),
                        html.Label(
                            id={
                                "type": "kernelTypeOfParam0",
                                "latent": latentID
                            },
                            children="ExponentialQuadraticKernel",
                        )
                ], style={"display": "inline-block", "width": "30%"}),
                html.Div(
                    children=[
                        html.Label(
                            id={
                                "type": "lengthScaleParam0Label",
                                "latent": latentID
                            },
                            children="Length Scale",
                        ),
                        dcc.Input(
                            id={
                                "type": "lengthScaleParam0",
                                "latent": latentID
                            },
                            type="number",
                            min=0,
                            required=True,
                            value=namedKernelParams["lengthScale"],
                        ),
                    ], style={"display": "inline-block", "width": "30%"}),
                html.Div(
                    id={
                        "type": "periodParam0Container",
                        "latent": latentID
                    },
                    children=[
                        html.Label(
                            children="Period"),
                        dcc.Input(
                            id={
                                "type": "periodParam0",
                                "latent": latentID
                            },
                            type="number",
                            min=0,
                            required=True,
                        ),
                    ],
                    style={"display": "none", "width": "30%"}
                ),
            ])
    elif kernelType is None:
        aDiv = html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label(
                            id={
                                "type": "kernelTypeParam0Label",
                                "latent": latentID
                            },
                            children="Kernel {:d} Type".format(latentID+1)),
                        html.Label(
                            id={
                                "type": "kernelTypeOfParam0",
                                "latent": latentID
                            },
                        ),
                    ], style={"display": "inline-block", "width": "30%"}),
                html.Div(
                    children=[
                        html.Label(
                            id={
                                "type": "lengthScaleParam0Label",
                                "latent": latentID
                            },
                            children="Length Scale"),
                        dcc.Input(
                            id={
                                "type": "lengthScaleParam0",
                                "latent": latentID
                            },
                            type="number",
                            min=0,
                            required=True,
                        ),
                    ], style={"display": "inline-block", "width": "30%"}),
                html.Div(
                    id={
                        "type": "periodParam0Container",
                        "latent": latentID
                    },
                    children=[
                        html.Label(
                            children="Period"),
                        dcc.Input(
                            id={
                                "type": "periodParam0",
                                "latent": latentID
                            },
                            type="number",
                            min=0,
                            required=True,
                        ),
                    ],
                    style={"display": "inline-block", "width": "30%"}
                ),
            ])
    else:
        raise RuntimeError("Invalid kernel type: {:s}".format(kernelType))
    return aDiv

class svGPFA_runner:
    def __init__(self, firstIndPoint):
        self._firstIndPoint = firstIndPoint

    def getFirstIndPoint(self):
        return self._firstIndPoint

    def setConditionalDist(self, conditionalDist):
        self._conditionalDist = conditionalDist

    def getConditionalDist(self):
        return self._conditionalDist

    def setLinkFunction(self, linkFunction):
        self._linkFunction = linkFunction

    def getLinkFunction(self):
        return self._linkFunction

    def setEmbeddingType(self, embeddingType):
        self._embeddingType = embeddingType

    def getEmbeddingType(self):
        return self._embeddingType

    def _getKernelsClassNames(self, kernelParams0Components):
        nLatents = len(kernelParams0Components)
        kernelsClassNames = []
        for k in range(nLatents):
            kernelsClassNames.append(kernelParams0Components[k][0]["props"]["children"][1]["props"]["children"])
        return kernelsClassNames

    def _getKernels(self, kernelTypeComponentValues):
        kernels = []
        module = importlib.import_module("stats.kernels")
        for k, kernelClass in enumerate(kernelTypeComponentValues):
            class_ = getattr(module, kernelTypeComponentValues[k])
            kernel = class_(scale=1.0)
            kernels.append(kernel)
        return kernels

    def setKernels(self, kernelTypeComponentValues):
        self._kernels = self._getKernels(kernelTypeComponentValues=kernelTypeComponentValues)

    def getKernels(self):
        return self._kernels

    def setKernelsParams0(self, kernelParams0Children):
        self._kernelsParams0 = [torch.from_numpy(eval(kernelParams0Children[k][0])) for k in range(len(kernelParams0Children))]

    def getKernelsParams0(self):
        return self._kernelsParams0

    def getKenels(self):
        return self._kernels

    def setNIndPointsPerLatent(self, nIndPointsPerLatent):
        self._nIndPointsPerLatent = nIndPointsPerLatent

    def getNIndPointsPerLatent(self):
        return self._nIndPointsPerLatent

    def setTrialsLengths(self, trialsLengths):
        self._trialsLengths = trialsLengths

    def getTrialsLengths(self):
        return self._trialsLengths

    def setSpikesTimes(self, spikesTimes):
        self._spikesTimes = spikesTimes

    def getSpikesTimes(self):
        return self._spikesTimes

    def setSVPosteriorOnIndPointsParams0(self, qMu0Strings, qSVec0Strings, qSDiag0Strings, nTrials):
        nLatents = len(qMu0Strings)
        qMu0 = []
        qSVec0 = []
        qSDiag0 = []
        for k in range(nLatents):
            aQMu0 = torch.from_numpy(eval(qMu0Strings[k]))
            aQMu0Expanded = torch.reshape(aQMu0, (-1, 1)).repeat(nTrials, 1, 1)
            qMu0.append(aQMu0Expanded)
            aQSVec0 = torch.from_numpy(eval(qSVec0Strings[k]))
            aQSVec0Expanded = torch.reshape(aQSVec0, (-1, 1)).repeat(nTrials, 1, 1)
            qSVec0.append(aQSVec0Expanded)
            aQSDiag0 = torch.from_numpy(eval(qSDiag0Strings[k]))
            aQSDiag0Expanded = torch.reshape(aQSDiag0, (-1, 1)).repeat(nTrials, 1, 1)
            qSDiag0.append(aQSDiag0Expanded)
        self._svPosteriosOnIndPointsParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}

    def getSVPosteriorOnIndPointsParams0(self):
        return self._svPosteriosOnIndPointsParams0

    def setEmbeddingParams0(self, nLatents, C0string, d0string):
        C0 = torch.from_numpy(eval(C0string))
        d0 = torch.from_numpy(eval(d0string))
        self._embeddingParams0 = {"C0": C0, "d0": d0}

    def getEmbeddingParams0(self):
        return self._embeddingParams0

    def setNQuad(self, nQuad):
        self._nQuad = nQuad

    def getNQuad(self):
        return self._nQuad

    def setIndPointsLocsKMSRegEpsilon(self, indPointsLocsKMSRegEpsilon):
        self._indPointsLocsKMSRegEpsilon = indPointsLocsKMSRegEpsilon

    def getIndPointsLocsKMSRegEpsilon(self):
        return self._indPointsLocsKMSRegEpsilon

    def setOptimParams(self, optimParams):
        self._optimParams = optimParams

    def getOptimParams(self):
        return self._optimParams

    def run(self, 
            logLock=None, logStreamFN=None, 
            lowerBoundLock=None, lowerBoundStreamFN=None,
            latentsTimes=None, latentsLock=None, latentsStreamFN=None,
           ):
        legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=self.getNQuad(), trialsLengths=self.getTrialsLengths())
        Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=self.getNIndPointsPerLatent(), trialsLengths=self.getTrialsLengths(), firstIndPoint=self.getFirstIndPoint())
        svPosteriorOnIndPointsParams0 = self.getSVPosteriorOnIndPointsParams0()
        svEmbeddingParams0 = self.getEmbeddingParams0()
        kernelsParams0 = self.getKernelsParams0() 
        kmsParams0 = {"kernelsParams0": kernelsParams0, "inducingPointsLocs0": Z0}
        initialParams = {"svPosteriorOnIndPoints": svPosteriorOnIndPointsParams0, "svEmbedding": svEmbeddingParams0, "kernelsMatricesStore": kmsParams0}
        quadParams = {"legQuadPoints": legQuadPoints, "legQuadWeights": legQuadWeights}

        conditionalDist = self.getConditionalDist()
        linkFunction = self.getLinkFunction()
        embeddingType = self.getEmbeddingType()
        kernels = self.getKernels()

        measurements = self.getSpikesTimes()
        optimParams = self.getOptimParams()
        indPointsLocsKMSEpsilon = self.getIndPointsLocsKMSRegEpsilon()


        # maximize lower bound
        svEM = stats.svGPFA.svEM.SVEM()

        model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(conditionalDist=conditionalDist, linkFunction=linkFunction, embeddingType=embeddingType, kernels=kernels)
        lowerBoundHist, elapsedTimeHist = svEM.maximize(model=model,
                                                        measurements=measurements,
                                                        initialParams=initialParams,
                                                        quadParams=quadParams,
                                                        optimParams=optimParams,
                                                        indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon,
                                                        logLock=logLock,
                                                        logStreamFN=logStreamFN,
                                                        lowerBoundLock=lowerBoundLock,
                                                        lowerBoundStreamFN=lowerBoundStreamFN,
                                                        latentsTimes=latentsTimes,
                                                        latentsLock=latentsLock,
                                                        latentsStreamFN=latentsStreamFN)

def getRastergram(trialSpikesTimes, title, xlabel="Time (ms)", ylabel="Neuron"):
    traces = []
    nNeurons = len(trialSpikesTimes)
    for n in range(nNeurons):
        neuronAndTrialSpikeTimes = trialSpikesTimes[n]
        traces.append(
            dict(
                x=neuronAndTrialSpikeTimes,
                y=(n+1)*np.ones(len(neuronAndTrialSpikeTimes)),
                mode="markers",
                name="neuron {:d}".format(n+1),
                marker=dict(symbol = "square"),
                showlegend=False
            )
        )
    layout = dict(
        title=title,
        xaxis=dict(title=xlabel),
        yaxis=dict(
            # range=[0.5, i+2],
            title=ylabel,
            # ticklen= 5,
            # gridwidth= 2,
            # tick0=1
        ),
        autosize=False,
        width=1000,
        height=700
    )
    answer = dict(data=traces, layout=layout)
    return answer

def getKernels(nLatents, config):
    kernels = []
    for k in range(nLatents):
        kernelType = config["kernels"]["kTypeLatent{:d}".format(k+1)]
        if kernelType=="periodic":
            lengthscale = float(config["kernels"]["kLengthscaleLatent{:d}".format(k+1)])
            period = float(config["kernels"]["kPeriodLatent{:d}".format(k+1)])
            kernel = stats.kernels.PeriodicKernel(scale=1.0)
            kernel.setParams(params=torch.Tensor([lengthscale, period]))
        elif kernelType=="exponentialQuadratic":
            lengthscale = float(config["kernels"]["kLengthscaleLatent{:d}".format(k+1)])
            kernel = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernel.setParams(params=torch.Tensor([lengthscale]))
        kernels.append(kernel)
    return kernels

def getSpikesTimes(contents, filename, spikesTimesVar):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    bytesIO = io.BytesIO(decoded)
    extension = os.path.splitext(filename)[1]
    if extension==".npz" or extension==".npy":
        loadRes = np.load(bytesIO, allow_pickle=True)
        spikesTimes = torch.from_numpy(loadRes[spikesTimesVar])
    elif extension==".pickle":
        loadRes = pickle.load(bytesIO)
        spikesTimes = loadRes[spikesTimesVar]
    elif extension==".mat":
        loadRes = loadmat(bytesIO)
        YNonStacked_tmp = loadRes[spikesTimesVar]
        nTrials = YNonStacked_tmp.shape[0]
        nNeurons = YNonStacked_tmp[0,0].shape[0]
        spikesTimes = [[] for n in range(nTrials)]
        for r in range(nTrials):
            spikesTimes[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                spikesTimes[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0])
    return spikesTimes

def getContentsVarsNames(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    bytesIO = io.BytesIO(decoded)
    extension = os.path.splitext(filename)[1]
    if extension==".npz" or extension==".npy":
        loadRes = np.load(bytesIO, allow_pickle=True)
    elif extension==".pickle":
        loadRes = pickle.load(bytesIO)
    elif extension==".mat":
        loadRes = loadmat(bytesIO)
    varNames = loadRes.keys()
    return varNames

