import pdb
import sys
import io
import os
import base64
import datetime
import configparser
import importlib
import numpy as np
from numpy import array
import torch
from scipy.io import loadmat
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import stats.kernels
import utils.svGPFA.miscUtils

class ComponentIDFactory:
    def __init__(self):
        self._nextID = 0

    def getID(self):
        answer = self._nextID
        self._nextID += 1
        return answer

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
        pdb.set_trace()
        return kernelsClassNames

    def _getKernels(self, kernelTypeComponents):
        kernels = []
        module = importlib.import_module("stats.kernels")
        for k, kernelClass in enumerate(kernelTypeComponents):
            class_ = getattr(module, kernelTypeComponents[k])
            kernel = class_()
            kernels.append(kernel)
        return kernels

    def setKernels(self, kernelTypeComponents):
        self._kernels = self._getKernels(kernelTypeComponents=kernelTypeComponents)

    def _getKernelsParams0(self, kernelParams0Components):
        kernelsParams0 = []
        for k in range(len(kernelParams0Components)):
            pdb.set_trace()
            kernelType = kernelParams0Components[k][0]["props"]["children"][1]["props"]["children"]
            if kernelType=="Periodic":
                lengthScale = float(kernelParams0Components[k][1]["props"]["children"][1]["props"]["value"])
                period = float(kernelParams0Components[k][2]["props"]["children"][1]["props"]["value"])
                kernelParams0 = [lengthScale, period]
            elif kernelType=="Exponential Quadratic":
                lengthScale = float(kernelParams0Components[k][1]["props"]["children"][1]["props"]["value"])
                kernelParams0 = [lengthScale]
            kernelsParams0.append(kernelParams0)
        return kernelsParams0

    def setKernelsParams0(self, kernelParams0Components):
        self._kernelsParams0 = self._getKernelsParams0(kernelParams0Components=kernelParams0Components)

    def getKenels(self):
        return self._kernels

    def setNIndPointsPerLatent(self, nIndPointsPerLatent):
        self._nIndPointsPerLatent = nIndPointsPerLatent

    def getNIndPointsPerLatent(self):
        return _nIndPointsPerLatent

    def setTrialsLengths(self, trialsLengths):
        self._trialsLengths = trialsLengths

    def getTrialsLengths(self):
        return _trialsLengths

    def setSpikesTimes(self, spikesTimes):
        self._spikesTimes = spikesTimes

    def getSpikesTimes(self):
        return self._spikesTimes

    def setSVPosteriorOnIndPointsParams0(self, qMu0Strings, qSVec0Strings, qSDiag0Strings):
        qMu0List = []
        qSVec0List = []
        qSDiag0List = []
        for k in range(len(qMu0Strings)):
            qMu0List.append(eval(qMu0Strings[k]))
            qSVec0List.append(eval(qSVec0Strings[k]))
            qSDiag0List.append(eval(qSDiag0Strings[k]))
        qMu0 = np.concatentate(qMu0List, axis=0)
        qSVec0 = np.concatentate(qSVec0List, axis=0)
        qSDiag0 = np.concatentate(qSDiag0List, axis=0)
        self._svPosteriosOnIndPointsParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}

    def getSVPosteriorOnIndPointsParams0(self):
        return self._svPosteriosOnIndPointsParams0

    def setEmbeddingParams0(self, nLatents, C0string, d0string):
        C0 = eval(C0string)
        d0 = eval(d0string)
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

    def run(self):
        legQuadPoints, legQuadWeights = \
            utils.miscUtils.getLegQuadPointsAndWeights(
                nQuad=self.getNQuad(), 
                trialsLengths=self.getTrialsLengths())
        Z0 = utils.svGPFA.initUtils.getIndPointLocs0(
            nIndPointsPerLatent=self.getNIndPointsPerLatent(),
            trialsLengths=self.getTrialsLengths(),
            firstIndPoint=self.getFirstIndPoint())
        svPosteriorOnIndPointsParams0 = self.getSVPosteriorOnIndPointsParams0()
        svEmbeddingParams0 = self.getEmbeddingParams0()
        kmsParams0 = {"kernelsParams0": self.getKernelsParams0(),
                      "inducingPointsLocs0": Z0}
        initialParams = {"svPosteriorOnIndPoints": svPosteriorOnIndPointsParams0,
                         "svEmbedding": svEmbeddingParams0,
                         "kernelsMatricesStore": kmsParams0}
        quadParams = {"legQuadPoints": legQuadPoints,
                      "legQuadWeights": legQuadWeights}

        model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
            conditionalDist=self.getConditionalDist(),
            linkFunction=self.getLinkFunction(),
            embeddingType=slef.getEmbeddingType(),
            kernels=self.getKernels())

        # maximize lower bound
        svEM = stats.svGPFA.svEM.SVEM()
        lowerBoundHist, elapsedTimeHist = svEM.maximize(
            model=model, measurements=self.getSpikesTimes(),
            initialParams=initialParams, quadParams=quadParams,
            optimParams=self.getOptimParams(),
            indPointsLocsKMSEpsilon=self.getIndPointsLocsKMSRegEpsilon())

        pdb.set_trace()

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
            scale = 1.0
            lengthscale = float(config["kernels"]["kLengthscaleLatent{:d}".format(k+1)])
            period = float(config["kernels"]["kPeriodLatent{:d}".format(k+1)])
            kernel = stats.kernels.PeriodicKernel()
            kernel.setParams(params=torch.Tensor([scale, lengthscale, period]))
        elif kernelType=="exponentialQuadratic":
            scale = 1.0
            lengthscale = float(config["kernels"]["kLengthscaleLatent{:d}".format(k+1)])
            kernel = stats.kernels.ExponentialQuadraticKernel()
            kernel.setParams(params=torch.Tensor([scale, lengthscale]))
        kernels.append(kernel)
    return kernels

def main(argv):
    if(len(argv)!=2):
        print("Usage: {:s} <gui ini>")
        sys.exit(1)
    guiFilename = argv[1]
    guiConfig = configparser.ConfigParser()
    guiConfig.read(guiFilename)
    defaultNLatents = int(guiConfig["latents"]["nLatents"])
    minNLatents = int(guiConfig["latents"]["minNLatents"])
    maxNLatents = int(guiConfig["latents"]["maxNLatents"])
    firstIndPoint = float(guiConfig["indPoints"]["firstIndPoint"])
    defaultKernels = getKernels(nLatents=defaultNLatents, config=guiConfig)

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    runner = svGPFA_runner(firstIndPoint=firstIndPoint)
    idFactory = ComponentIDFactory()

    app.layout = html.Div(children=[
        html.H1(children="Sparse Variational Gaussian Process Factor Analysis"),

        html.Hr(),

        html.H4(children="Model Specification"),
        html.Div(children=[
            html.Div(children=[
                html.Label("Conditional Distribution"),
                dcc.RadioItems(
                    id="conditionalDist",
                    options=[
                        {"label": "Point Process", "value": stats.svGPFA.svGPFAModelFactory.PointProcess},
                        {"label": "Poisson", "value": stats.svGPFA.svGPFAModelFactory.Poisson},
                        {"label": "Gaussian", "value": stats.svGPFA.svGPFAModelFactory.Gaussian},
                    ],
                    value=stats.svGPFA.svGPFAModelFactory.PointProcess,
                ),
            ], style={"display": "inline-block", "background-color": "white", "padding-right": "30px"}),
            html.Div(
                children=[
                    html.Label("Link Function"),
                    dcc.RadioItems(
                        id="linkFunction",
                        options=[
                            {"label": "Exponential", "value": stats.svGPFA.svGPFAModelFactory.ExponentialLink},
                            {"label": "Other", "value": stats.svGPFA.svGPFAModelFactory.NonExponentialLink},
                        ],
                        value=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
                    ),
                ], style={"display": "inline-block", "background-color": "white", "padding-right": "30px"}),
            html.Div(
                children=[
                    html.Label("Embedding Type"),
                    dcc.RadioItems(
                        id="embeddingType",
                        options=[
                            {"label": "Linear", "value": stats.svGPFA.svGPFAModelFactory.LinearEmbedding},
                        ],
                        value=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
                    ),
                ], style={"display": "inline-block", "background-color": "white", "padding-right": "30px"}),
        ], style={"display": "flex", "flex-wrap": "wrap", "width": 800, "padding-bottom": "20px", "background-color": "white"}),
        html.Div(children=[
            html.Label("Number of Latents"),
            html.Div(children=[
                dcc.Slider(
                    id="nLatentsComponent",
                    min=0,
                    max=10,
                    marks={i: str(i) for i in range(1, 11)},
                    value=3,
                )],
                style={"width": "25%"}
            ),
        ], style={"padding-bottom": "20px"}),
        html.Div(id="kernelsTypesContainer", children=[]),
        html.Hr(),
        html.H4(children="Data"),
        html.Div(children=[
            html.Div(children=[
                html.Label("Spikes Variable Name"),
                dcc.Input(id="spikesTimesVar", type="text", value="Y", required=True, size="15"),
            ], style={"width": "200px"}),
            html.Div(children=[
                dcc.Upload(id="uploadSpikes", children=html.Button("Upload Spikes")),
                html.Div(id="spikesInfo"),
            ], style={"width": "400px"}),
        ], style={"width": 800, "background-color": "white"}),
        html.Div(
            id="trialToPlotDiv",
            children=[
                html.Label("Trial to Plot"),
                dcc.Dropdown(
                    id="trialToPlot",
                    options=[
                        {"label": "dummy option", "value": -1},
                    ],
                    value=-1,
                    style={"width": "80%"}
                ),
                dcc.Graph(
                    id="trialRastergram",
                    figure={}
                ),
                html.Hr(),
            ], hidden=True, style={"width": "20%", "columnCount": 1}),
        html.Div(id="kernelParams0Container", children=[]),
        html.Div(id="trialsLengths", children=[html.Div()]),
        html.Div(id="nIndPointsPerLatent", children=[html.Div()]),
        html.Div(id="svPosterioOnIndPointsInitialConditions", children=[html.Div()]),
        html.Div(id="embeddingParams0",
                 children=[
                    html.H4("Initial Conditions for Linear Embedding"),
                    html.Div(children=[
                        html.Label("Mixing Matrix"),
                        dcc.Textarea(
                            id="C0string",
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Offset Vector"),
                        dcc.Textarea(
                            id="d0string",
                        ),
                    ]),
                ], hidden=True),
        html.Hr(),
        html.Div(
            id="optimParams",
            children=[
                html.H4("EM Parameters"),
                html.Div(children=[
                    html.Label("Maximum EM iterations"),
                    dcc.Input(id="maxIter", type="number", value="20", required=True),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Expectation Step"),
                    dcc.Checklist(id="eStepEstimate",
                                  options=[{"label": "estimate",
                                            "value": "eStepEstimate"}],
                                  value=["eStepEstimate"]
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="eStepMaxIter", type="number", value="20", required=True),
                    html.Label("Learning rate"),
                    dcc.Input(id="eStepLR", type="number", value="1e-3", required=True),
                    html.Label("Tolerance"),
                    dcc.Input(id="eStepTol", type="number", value="1e-3", required=True),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "eStepLineSearchStrong_wolfe"},
                            {"label": "none", "value": "eStepLineSearchNone"},
                        ],
                        value="eStepLineSearchStrong_wolfe",
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Embedding Parameters"),
                    dcc.Checklist(id="mStepEmbedding",
                                  options=[{"label": "estimate",
                                            "value": "mStepEmbeddingEstimate"}],
                                  value=["mStepEmbeddingEstimate"]
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="mStepEmbeddingMaxIter", type="number", value="20", required=True),
                    html.Label("Learning rate"),
                    dcc.Input(id="mStepEmbeddingLR", type="number", value="1e-3", required=True),
                    html.Label("Tolerance"),
                    dcc.Input(id="mStepEmbeddingTol", type="number", value="1e-3", required=True),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "mStepEmbeddingLineSearchStrong_wolfe"},
                            {"label": "none", "value": "mStepEmbeddingLineSearchNone"},
                        ],
                        value="mStepEmbeddingLineSearchStrong_wolfe",
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Kernels Parameters"),
                    dcc.Checklist(id="mStepKernelsParams",
                                  options=[{"label": "estimate",
                                            "value": "mStepKernelsParamsEstimate"}],
                                  value=["mStepKernelsParamsEstimate"]
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="mStepKernelsParamsMaxIter", type="number", value="20", required=True),
                    html.Label("Learning rate"),
                    dcc.Input(id="mStepKernelsParamsLR", type="number", value="1e-3", required=True),
                    html.Label("Tolerance"),
                    dcc.Input(id="mStepKernelsParamsTol", type="number", value="1e-3", required=True),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "mStepKernelsParamsLineSearchStrong_wolfe"},
                            {"label": "none", "value": "mStepKernelsParamsLineSearchNone"},
                        ],
                        value="mStepKernelsParamsLineSearchStrong_wolfe",
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Inducing Points Parameters"),
                    dcc.Checklist(id="mStepIndPointsParams",
                                  options=[{"label": "estimate",
                                            "value": "mStepIndPointsParamsEstimate"}],
                                  value=["mStepIndPointsParamsEstimate"]
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="mStepIndPointsParamsMaxIter", type="number", value="20", required=True),
                    html.Label("Learning rate"),
                    dcc.Input(id="mStepIndPointsParamsLR", type="number", value="1e-4", required=True),
                    html.Label("Tolerance"),
                    dcc.Input(id="mStepIndPointsParamsTol", type="number", value="1e-3", required=True),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "mStepIndPointsParamsLineSearchStrong_wolfe"},
                            {"label": "none", "value": "mStepIndPointsParamsLineSearchNone"},
                        ],
                        value="mStepIndPointsParamsLineSearchStrong_wolfe",
                    ),
                ], style={"padding-bottom": "20px"}),
            ]
        ),
        html.Hr(),
        html.H4("Miscellaneous Parameters"),
        html.Div(children=[
            html.Label("Number of quadrature points"),
            dcc.Input(id="nQuad", type="number", value="200", required=True),
        ], style={"padding-bottom": "20px"}),
        html.Div(children=[
            html.Label("Variance added to kernel covariance matrix for inducing points"),
            dcc.Input(id="indPointsLocsKMSRegEpsilon", type="number", value="1e-2", required=True),
        ], style={"padding-bottom": "20px"}),
        html.Hr(),
        html.Button("Estimate", id="doEstimate", n_clicks=0),
        html.Div(id="estimationRes"),
    ])

    @app.callback(
        Output("kernelsTypesContainer", "children"),
        [Input("nLatentsComponent", "value")],
        [State("kernelsTypesContainer", "children")])
    def populateKernels(nLatentsComponentValue, kernelsContainerChildren):
        if nLatentsComponentValue==0:
            raise PreventUpdate
        if nLatentsComponentValue==len(kernelsContainerChildren):
            raise PreventUpdate
        elif nLatentsComponentValue<len(kernelsContainerChildren):
            return kernelsContainerChildren[:nLatentsComponentValue]
        elif len(kernelsContainerChildren)==0:
            newChildren = []
            for k in range(nLatentsComponentValue):
                aDiv = html.Div(children=[
                    html.Div(children=[
                        html.Label("Kernel {:d} Type".format(k+1)),
                        dcc.Dropdown(
                            id={
                                "type": "kernelTypeComponents",
                                "index": idFactory.getID()
                            },
                            options=[
                                {"label": "Exponential Quadratic", "value": "ExponentialQuadraticKernel"},
                                {"label": "Periodic", "value": "PeriodicKernel"},
                            ],
                            value=type(defaultKernels[k]).__name__,
                            style={"width": "45%"}
                        ),
                    ]),
                ], style={"columnCount": 1})
                newChildren.append(aDiv)
            answer = kernelsContainerChildren + newChildren
            return answer
        elif nLatentsComponentValue>len(kernelsContainerChildren):
            newChildren = []
            for k in range(len(kernelsContainerChildren),
                           nLatentsComponentValue):
                aDiv = html.Div(children=[
                    html.Div(children=[
                        html.Label("Kernel {:d} Type".format(k+1)),
                        dcc.Dropdown(
                            id={
                                "type": "kernelTypeComponents",
                                "index": idFactory.getID()
                            },
                            options=[
                                {"label": "Exponential Quadratic", "value": "ExponentialQuadraticKernel"},
                                {"label": "Periodic", "value": "PeriodicKernel"},
                            ],
                            value="ExponentialQuadraticKernel",
                            style={"width": "45%"}
                        ),
                    ]),
                ], style={"columnCount": 1})
                newChildren.append(aDiv)
            answer = kernelsContainerChildren + newChildren
            return answer

    @app.callback(
        Output("kernelParams0Container", "children"),
        [Input({"type": "kernelTypeComponents",  "index": ALL}, "value")],
        [State({"type": "kernelParams0Components",  "index": ALL}, "children"),
         State("kernelParams0Container", "children")])
    def updateKernelsParams0(kernelTypeComponentsValues,
                             kernelParams0ComponentsChildren,
                             kernelParams0ContainerChildren):
        nLatents = len(kernelTypeComponentsValues)
        # pdb.set_trace()
        if nLatents==0:
            # non-relevant event
            raise PreventUpdate
        if len(kernelParams0ComponentsChildren)==0:
            # pdb.set_trace()
            # initialize kernels params with those from defaultKernels
            newChildren = [html.H4("Kernels Parameters")]
            for k in range(nLatents):
                namedKernelsParams = defaultKernels[k].getNamedParams()
                if type(defaultKernels[k]).__name__=="PeriodicKernel":
                    aDiv = html.Div(
                        id={
                            "type": "kernelParams0Components",
                            "index": idFactory.getID()
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Kernel {:d} Type".format(k+1)),
                                    html.Label("Periodic")
                                ], style={"display": "inline-block", "width": "30%"}),
                                html.Div(children=[
                                    html.Label("Length Scale"),
                                    dcc.Input(
                                        type="number",
                                        min=0,
                                        value="{:.2f}".format(namedKernelsParams["lengthScale"]),
                                    ),
                                ], style={"display": "inline-block",
                                      "width": "30%"}),
                                html.Div(children=[
                                    html.Label("Period"),
                                    dcc.Input(
                                        type="number",
                                        min=0,
                                        value="{:.2f}".format(namedKernelsParams["period"]),
                                    ),
                                ], style={"display": "inline-block",
                                          "width": "30%"}),
                            ])
                elif type(defaultKernels[k]).__name__=="ExponentialQuadraticKernel":
                    aDiv = html.Div(
                        id={
                            "type": "kernelParams0Components",
                            "index": idFactory.getID()
                        },
                        children=[
                            html.Div(children=[
                                html.Label("Kernel {:d} Type".format(k+1)),
                                html.Label("Exponential Quadratic")
                            ], style={"display": "inline-block",
                                      "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label("Length Scale"),
                                    dcc.Input(
                                        type="number",
                                        min=0,
                                        value="{:.2f}".format(namedKernelsParams["lengthScale"]),
                                    ),
                                ], style={"display": "inline-block",
                                          "width": "30%"}),
                        ])
                newChildren.append(aDiv)
            newChildren.append(html.Hr())
            return newChildren
        elif nLatents<len(kernelParams0ComponentsChildren):
            # remove kernel params from the end
            # pdb.set_trace()
            newChildren = kernelParams0ContainerChildren[:(1+nLatents)]
            newChildren.append(html.Hr())
            return newChildren
        elif nLatents>len(kernelParams0ComponentsChildren):
            # pdb.set_trace()
            nKernelsParams0ToAdd = nLatents-len(kernelParams0ComponentsChildren)
            newChildren = kernelParams0ContainerChildren[:-1]
            for k in range(nKernelsParams0ToAdd):
                kTypeToAdd = kernelTypeComponentsValues[len(kernelParams0ComponentsChildren)+k]
                if kTypeToAdd=="PeriodicKernel":
                    aDiv = html.Div(
                        id={
                            "type": "kernelParams0Components",
                            "index": idFactory.getID()
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Kernel {:d} Type".format(len(kernelParams0ComponentsChildren)+k+1)),
                                    html.Label("PeriodicKernel")
                                ], style={"display": "inline-block", "width": "30%"}),
                                html.Div(children=[
                                    html.Label("Length Scale"),
                                    dcc.Input(
                                        type="number",
                                        min=0,
                                    ),
                                ], style={"display": "inline-block",
                                      "width": "30%"}),
                                html.Div(children=[
                                    html.Label("Period"),
                                    dcc.Input(
                                        type="number",
                                        min=0,
                                    ),
                                ], style={"display": "inline-block",
                                          "width": "30%"}),
                            ])
                elif kTypeToAdd=="ExponentialQuadraticKernel":
                    aDiv = html.Div(
                        id={
                            "type": "kernelParams0Components",
                            "index": idFactory.getID()
                        },
                        children=[
                            html.Div(children=[
                                html.Label("Kernel {:d} Type".format(len(kernelParams0ComponentsChildren)+k+1)),
                                html.Label("Exponential Quadratic")
                            ], style={"display": "inline-block",
                                      "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label("Length Scale"),
                                    dcc.Input(
                                        type="number",
                                        min=0,
                                    ),
                                ], style={"display": "inline-block",
                                          "width": "30%"}),
                        ])
                newChildren.append(aDiv)
            newChildren.append(html.Hr())
            return newChildren
        elif nLatents==len(kernelParams0ComponentsChildren):
            newChildren = [kernelParams0ContainerChildren[0]]
            for k in range(nLatents):
                kernelTypeLatents = kernelTypeComponentsValues[k]
                kernelTypeLatentsParams = kernelParams0ComponentsChildren[k][0]["props"]["children"][1]["props"]["children"]
                # if False:
                if kernelTypeLatents!=kernelTypeLatentsParams:
                    kTypeToAdd = kernelTypeLatents
                    if kTypeToAdd=="PeriodicKernel":
                        aDiv = html.Div(
                            id={
                                "type": "kernelParams0Components",
                                "index": idFactory.getID()
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Label("Kernel {:d} Type".format(k+1)),
                                        html.Label("PeriodicKernel")
                                    ], style={"display": "inline-block", "width": "30%"}),
                                    html.Div(children=[
                                        html.Label("Length Scale"),
                                        dcc.Input(
                                            type="number",
                                            min=0,
                                        ),
                                    ], style={"display": "inline-block",
                                        "width": "30%"}),
                                    html.Div(children=[
                                        html.Label("Period"),
                                        dcc.Input(
                                            type="number",
                                            min=0,
                                        ),
                                    ], style={"display": "inline-block",
                                            "width": "30%"}),
                                ])
                    elif kTypeToAdd=="ExponentialQuadraticKernel":
                        aDiv = html.Div(
                            id={
                            "type": "kernelParams0Components",
                            "index": idFactory.getID()
                            },
                            children=[
                                html.Div(children=[
                                    html.Label("Kernel {:d} Type".format(k+1)),
                                    html.Label("Exponential Quadratic")
                                ], style={"display": "inline-block",
                                        "width": "30%"}),
                                html.Div(
                                    children=[
                                        html.Label("Length Scale"),
                                        dcc.Input(
                                            type="number",
                                            min=0,
                                        ),
                                    ], style={"display": "inline-block",
                                            "width": "30%"}),
                            ])
                    else:
                        RuntimeError("Invalid kernel type {:s}".format(kTypeToAdd))
                else:
                    aDiv = kernelParams0ContainerChildren[k+1]
                newChildren.append(aDiv)
            newChildren.append(html.Hr())
            return newChildren
            # return kernelParams0ContainerChildren

    @app.callback([Output("spikesInfo", "children"),
                   Output("trialToPlot", "options"),
                   Output("trialToPlot", "value"),
                   Output("trialToPlotDiv", "hidden")],
                  [Input("uploadSpikes", "contents")],
                  [State("uploadSpikes", "filename"),
                   State("uploadSpikes", "last_modified"),
                   State("spikesTimesVar", "value")])
    def loadSpikes(list_of_contents, list_of_names, list_of_dates, spikesTimesVar):
        if list_of_contents is not None:
            content_type, content_string = list_of_contents.split(",")
            decoded = base64.b64decode(content_string)
            bytesIO = io.BytesIO(decoded)
            extension = os.path.splitext(list_of_names)[1]
            if extension==".npz" or extension==".npy":
                loadRes = np.load(bytesIO, allow_pickle=True)
                spikesTimes = loadRes[spikesTimesVar]
            elif extension==".mat":
                loadRes = loadmat(bytesIO)
                YNonStacked_tmp = loadRes[spikesTimesVar]
                nTrials = YNonStacked_tmp.shape[0]
                nNeurons = YNonStacked_tmp[0,0].shape[0]
                spikesTimes = np.empty(shape=(nTrials, nNeurons), dtype=np.object)
                for r in range(nTrials):
                    for n in range(nNeurons):
                        spikesTimes[r][n] = YNonStacked_tmp[r,0][n,0][:,0]
            runner.setSpikesTimes(spikesTimes=spikesTimes)
            nTrials = spikesTimes.shape[0]
            spikesInfoChildren = [
                "Filename: {:s}".format(list_of_names)
            ]
            trialToPlotOptions = [{"label": str(r+1), "value": r} for r in range(nTrials)]
            trialToPlotValue = 0
            trialToPlotDivHidden = False
            return spikesInfoChildren, trialToPlotOptions, trialToPlotValue, trialToPlotDivHidden
        # raise IgnoreCallback("Ignoring callback because list_of_contents is None")
        raise PreventUpdate

    @app.callback(Output("trialRastergram", "figure"),
                  [Input("trialToPlot", "value")],
                  [State("trialRastergram", "figure")])
    def updateRastergram(trialToPlot, oldRastergram):
        if trialToPlot>=0:
            spikesTimes = runner.getSpikesTimes()
            trialSpikesTimes = spikesTimes[trialToPlot,:]
            title="Trial {:d}".format(trialToPlot+1)
            rastergram = getRastergram(trialSpikesTimes=trialSpikesTimes, title=title)
            return rastergram
        raise PreventUpdate

    @app.callback(Output("trialsLengths", "children"),
                  [Input("spikesInfo", "children")])
    def showTrialLengths(spikesInfoChildren):
        if spikesInfoChildren is not None:
            nTrials = runner.getSpikesTimes().shape[0]
            someChildren = [html.H4("Trials lengths")]
            for r in range(nTrials):
                aDiv = html.Div(children=[
                    html.Label("Trial {:d}".format(r+1)),
                    dcc.Input(
                        type="number", placeholder="trial length",
                        min=0,
                    ),
                ])
                someChildren.append(aDiv)
            someChildren.append(html.Hr())
            return someChildren
        # return [None]
        raise PreventUpdate


    @app.callback(Output("nIndPointsPerLatent", "children"),
                  [Input("nLatentsComponent", "value")])
    def showNIndPointsPerLatent(nLatentsComponentValue):
        someChildren = []
        aChildren = html.H4("Number of Inducing Points")
        someChildren.append(aChildren)
        for k in range(nLatentsComponentValue):
            aChildren = html.Div(children=[
                html.Label("Latent {:d}".format(k+1)),
                html.Div(
                    children=[
                    dcc.Slider(
                        id={
                            "type": "nIndPoints",
                            "index": idFactory.getID()
                        },
                        min=5,
                        max=50,
                        marks={i: str(i) for i in range(5, 51, 5)},
                        value=20,
                    )
                ], style={"width": "25%", "height": "50px"}),
            ])
            someChildren.append(aChildren)
        someChildren.append(html.Hr())
        return someChildren

    @app.callback(Output("svPosterioOnIndPointsInitialConditions", "children"),
                  [Input({"type": "nIndPoints", "index": ALL}, "value")])
    def populateSVPosteriorOnIndPointsInitialConditions(values):
        initVar = 0.01
        someChildren = []
        aChildren = html.H4("Initial Conditions for Variational Posterior on Inducing Points")
        someChildren.append(aChildren)
        for k, nIndPoints in enumerate(values):
            qMu0 = np.zeros((nIndPoints, 1))
            qVec0 = np.zeros((nIndPoints, 1))
            qVec0[0] = 1.0
            qDiag0 = initVar*np.ones((nIndPoints, 1))
            aChildren = html.Div(children=[
                html.H6("Latent {:d}".format(k+1)),
                html.Label("Mean"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsMu0",
                        "index": idFactory.getID()
                    },
                    type="text",
                    value=np.array_repr(qMu0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
                html.Label("Spanning Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsVec0",
                        "index": idFactory.getID()
                    },
                    type="text",
                    value=np.array_repr(qVec0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
                html.Label("Diagonal Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsDiag0",
                        "index": idFactory.getID()
                    },
                    type="text",
                    value=np.array_repr(qDiag0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
            ], style={"padding-bottom": "30px"})
            someChildren.append(aChildren)
        return someChildren

    @app.callback([Output("embeddingParams0", "hidden"),
                   Output("C0string", "value"),
                   Output("C0string", "style"),
                   Output("d0string", "value"),
                   Output("d0string", "style")],
                  [Input("nLatentsComponent", "value"),
                   Input("spikesInfo", "children")])
    def populateEmbeddingInitialConditions(nLatentsComponentValue, spikesInfoChildren):
        if spikesInfoChildren is not None:
            nNeurons = runner.getSpikesTimes().shape[1]
            C0 = np.random.uniform(size=(nNeurons, nLatentsComponentValue))
            d0 = np.random.uniform(size=(nNeurons, 1))
            C0style = {"width": nLatentsComponentValue*175, "height": 300}
            d0style={"width": 200, "height": 300}
            # answer = [False, np.array2string(C0), C0style, np.array2string(d0), d0style]
            answer = [False, np.array_repr(C0), C0style, np.array_repr(d0), d0style]
            return answer
        raise PreventUpdate

    @app.callback(
        Output("estimationRes", "children"),
        [Input("doEstimate", "n_clicks")],
        [State("conditionalDist", "value"),
         State("linkFunction", "value"),
         State("embeddingType", "value"),
         State("nLatentsComponent", "value"),
         State("C0string", "value"),
         State("d0string", "value"),
         State({"type": "kernelTypeComponents",  "index": ALL}, "value"),
         State({"type": "kernelParams0Components",  "index": ALL}, "children"),
         State({"type": "nIndPoints",  "index": ALL}, "value"),
         State({"type": "trialsLenghts",  "index": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsMu0",  "index": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsVec0",  "index": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsDiag0",  "index": ALL}, "value"),
         State("nQuad", "value"),
         State("indPointsLocsKMSRegEpsilon", "value"),
        ])
    def estimateSVGPFA(doEstimateNClicks,
                       conditionalDist,
                       linkFunction,
                       embeddingType,
                       nLatentsComponentValue,
                       C0string,
                       d0string,
                       kernelTypeComponents,
                       kernelParams0Components,
                       nIndPoints,
                       trialsLengths,
                       svPosterioOnIndPointsMu0,
                       svPosterioOnIndPointsVec0,
                       svPosterioOnIndPointsDiag0,
                       nQuad,
                       indPointsLocsKMSRegEpsilon,
                      ):
        if doEstimateNClicks>0:
            runner.setConditionalDist(conditionalDist=conditionalDist)
            runner.setLinkFunction(linkFunction=linkFunction)
            runner.setEmbeddingType(embeddingType=embeddingType)
            runner.setKernels(kernelTypeComponents=kernelTypeComponents)
            runner.setKernelsParams0(kernelParams0Components=kernelParams0Components)
            runner.setNIndPointsPerLatent(nIndPointsPerLatent=nIndPoints)
            runner.setEmbeddingParams0(nLatents=nLatentsComponentValue, C0string=C0string, d0string=d0string),
            runner.setTrialsLengths(trialsLengths=trialsLengths)
            runner.setSVPosteriorOnIndPointsParams0(
                qMu0Strings=svPosterioOnIndPointsMu0,
                qSVec0Strings=svPosterioOnIndPointsVec0,
                qSDiag0Strings=svPosterioOnIndPointsDiag0)
            runner.setNQuad(nQuad=nQuad)
            runner.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

            return ["Successful estimation!!!"]

    app.run_server(debug=True)

if __name__ == "__main__":
    main(sys.argv)

