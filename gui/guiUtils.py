import pdb
import sys
import os
import io
import base64
import math
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
    nTrials = spikesTimes.shape[0]
    nNeurons = spikesTimes.shape[1]
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
            kernel = class_()
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

#         model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(conditionalDist=conditionalDist, linkFunction=linkFunction, embeddingType=embeddingType, kernels=kernels)
#         lowerBoundHist, elapsedTimeHist = svEM.maximize(model=model,
#                                                         measurements=measurements,
#                                                         initialParams=initialParams,
#                                                         quadParams=quadParams,
#                                                         optimParams=optimParams,
#                                                         indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon,
#                                                         logLock=logLock,
#                                                         logStreamFN=logStreamFN,
#                                                         lowerBoundLock=lowerBoundLock,
#                                                         lowerBoundStreamFN=lowerBoundStreamFN,
#                                                         latentsTimes=latentsTimes,
#                                                         latentsLock=latentsLock,
#                                                         latentsStreamFN=latentsStreamFN)
#         return

        ppSimulationFilename = os.path.join("../scripts", "data/pointProcessSimulation.mat")
        initDataFilename = os.path.join("../scripts", "data/pointProcessInitialConditions.mat")

        mat = loadmat(initDataFilename)
        nLatents_2 = len(mat['Z0'])
        nTrials_2 = mat['Z0'][0,0].shape[2]
        qMu0_2 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents_2)]
        qSVec0_2 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents_2)]
        qSDiag0_2 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents_2)]
        Z0_2 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents_2)]
        C0_2 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
        b0_2 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
        legQuadPoints_2 = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
        legQuadWeights_2 = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)

        yMat = loadmat(ppSimulationFilename)
        YNonStacked_tmp = yMat['Y']
        nNeurons_2 = YNonStacked_tmp[0,0].shape[0]
        YNonStacked = [[[] for n in range(nNeurons_2)] for r in range(nTrials_2)]
        for r in range(nTrials_2):
            for n in range(nNeurons_2):
                YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

        measurements_2 = YNonStacked

        kernelNames_2 = mat["kernelNames"]
        hprs0_2 = mat["hprs0"]
        indPointsLocsKMSEpsilon_2 = 1e-2

        # create kernels
        kernels_2 = [[None] for k in range(nLatents_2)]
        for k in range(nLatents_2):
            if np.char.equal(kernelNames_2[0,k][0], "PeriodicKernel"):
                kernels_2[k] = stats.kernels.PeriodicKernel()
            elif np.char.equal(kernelNames_2[0,k][0], "rbfKernel"):
                kernels_2[k] = stats.kernels.ExponentialQuadraticKernel()
            else:
                raise ValueError("Invalid kernel name: %s"%(kernelNames_2[k]))

        # create initial parameters
        kernelsParams0_2 = [[None] for k in range(nLatents_2)]
        for k in range(nLatents_2):
            if np.char.equal(kernelNames_2[0,k][0], "PeriodicKernel"):
                kernelsParams0_2[k] = torch.tensor([float(hprs0_2[k,0][0]),
                                                  float(hprs0_2[k,0][1])],
                                                 dtype=torch.double)
            elif np.char.equal(kernelNames_2[0,k][0], "rbfKernel"):
                kernelsParams0_2[k] = torch.tensor([float(hprs0_2[k,0][0])],
                                                 dtype=torch.double)
            else:
                raise ValueError("Invalid kernel name: %s"%(kernelNames_2[k]))

        qUParams0_2 = {"qMu0": qMu0_2, "qSVec0": qSVec0_2, "qSDiag0": qSDiag0_2}
        qHParams0_2 = {"C0": C0_2, "d0": b0_2}
        kmsParams0_2 = {"kernelsParams0": kernelsParams0_2,
                      "inducingPointsLocs0": Z0_2}
        initialParams_2 = {"svPosteriorOnIndPoints": qUParams0_2,
                         "kernelsMatricesStore": kmsParams0_2,
                         "svEmbedding": qHParams0_2}
        quadParams_2 = {"legQuadPoints": legQuadPoints_2,
                      "legQuadWeights": legQuadWeights_2}
        optimParams_2 = {"emMaxIter":50,
                       #
                       "eStepEstimate":True,
                       "eStepMaxIter":100,
                       "eStepTol":1e-3,
                       "eStepLR":1e-3,
                       "eStepLineSearchFn":"strong_wolfe",
                       # "eStepLineSearchFn":"None",
                       "eStepNIterDisplay":1,
                       #
                       "mStepEmbeddingEstimate":True,
                       "mStepEmbeddingMaxIter":100,
                       "mStepEmbeddingTol":1e-3,
                       "mStepEmbeddingLR":1e-3,
                       "mStepEmbeddingLineSearchFn":"strong_wolfe",
                       # "mStepEmbeddingLineSearchFn":"None",
                       "mStepEmbeddingNIterDisplay":1,
                       #
                       "mStepKernelsEstimate":True,
                       "mStepKernelsMaxIter":10,
                       "mStepKernelsTol":1e-3,
                       "mStepKernelsLR":1e-3,
                       "mStepKernelsLineSearchFn":"strong_wolfe",
                       # "mStepKernelsLineSearchFn":"None",
                       "mStepKernelsNIterDisplay":1,
                       "mStepKernelsNIterDisplay":1,
                       #
                       "mStepIndPointsEstimate":True,
                       "mStepIndPointsMaxIter":20,
                       "mStepIndPointsTol":1e-3,
                       "mStepIndPointsLR":1e-4,
                       "mStepIndPointsLineSearchFn":"strong_wolfe",
                       # "mStepIndPointsLineSearchFn":"None",
                       "mStepIndPointsNIterDisplay":1,
                       #
                       "verbose":True
                      }

        conditionalDist_2 = stats.svGPFA.svGPFAModelFactory.PointProcess
        linkFunction_2 = stats.svGPFA.svGPFAModelFactory.ExponentialLink
        embeddingType_2 = stats.svGPFA.svGPFAModelFactory.LinearEmbedding

#         model_2 = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(conditionalDist=conditionalDist_2, linkFunction=linkFunction_2, embeddingType=embeddingType_2, kernels=kernels_2)

#         lowerBoundHist_2, elapsedTimeHist_2 = svEM.maximize(model=model_2, measurements=measurements_2, initialParams=initialParams_2, quadParams=quadParams_2, optimParams=optimParams_2, indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon_2, logLock=logLock, logStreamFN=logStreamFN, lowerBoundLock=lowerBoundLock, lowerBoundStreamFN=lowerBoundStreamFN, latentsTimes=latentsTimes, latentsLock=latentsLock, latentsStreamFN=latentsStreamFN)

#         kmsParams0_3 = {"kernelsParams0": kernelsParams0,
#                         "inducingPointsLocs0": Z0_2}
#         initialParams_3 = {"svPosteriorOnIndPoints": svPosteriorOnIndPointsParams0, "kernelsMatricesStore": kmsParams0, "svEmbedding": qHParams0_2}
        # pdb.set_trace()
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
            kernel = stats.kernels.PeriodicKernel()
            kernel.setParams(params=torch.Tensor([lengthscale, period]))
        elif kernelType=="exponentialQuadratic":
            lengthscale = float(config["kernels"]["kLengthscaleLatent{:d}".format(k+1)])
            kernel = stats.kernels.ExponentialQuadraticKernel()
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
    elif extension==".mat":
        loadRes = loadmat(bytesIO)
        YNonStacked_tmp = loadRes[spikesTimesVar]
        nTrials = YNonStacked_tmp.shape[0]
        nNeurons = YNonStacked_tmp[0,0].shape[0]
        spikesTimes = np.empty(shape=(nTrials, nNeurons), dtype=np.object)
        for r in range(nTrials):
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
    elif extension==".mat":
        loadRes = loadmat(bytesIO)
    varNames = loadRes.keys()
    return varNames

