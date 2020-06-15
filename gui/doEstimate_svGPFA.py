import pdb
import sys
import datetime
import configparser
import math
import numpy as np
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
from guiUtils import getContentsVarsNames, getSpikesTimes, getRastergram, getKernels, getKernelParams0Div, guessTrialsLengths, svGPFA_runner

def main(argv):
    if(len(argv)!=2):
        print("Usage: {:s} <gui ini>")
        sys.exit(1)
    guiFilename = argv[1]
    guiConfig = configparser.ConfigParser()
    guiConfig.read(guiFilename)

    condDistString = guiConfig["modelSpecs"]["condDist"]
    if condDistString=="PointProcess":
        condDist0 = stats.svGPFA.svGPFAModelFactory.PointProcess
    elif condDistString=="Poisson":
        condDist0 = stats.svGPFA.svGPFAModelFactory.Poisson
    elif condDistString=="Gaussian":
        condDist0 = stats.svGPFA.svGPFAModelFactory.Gaussian
    else:
        raise RuntimeError("Invalid conditional distribution: {:s}".format(condDistString))

    linkFuncString = guiConfig["modelSpecs"]["linkFunc"]
    if linkFuncString=="ExponentialLink":
        linkFunc0 = stats.svGPFA.svGPFAModelFactory.ExponentialLink
    elif linkFuncString=="NonExponential":
        linkFunc0 = stats.svGPFA.svGPFAModelFactory.NonExponential
    else:
        raise RuntimeError("Invalid link function: {:s}".format(linkFunctionString))

    embeddingString = guiConfig["modelSpecs"]["embedding"]
    if embeddingString=="LinearEmbedding":
        embedding0 = stats.svGPFA.svGPFAModelFactory.LinearEmbedding
    else:
        raise RuntimeError("Invalid embedding: {:s}".format(embeddingString))

    nLatents0 = int(guiConfig["latents"]["nLatents"])
    minNLatents = int(guiConfig["latents"]["minNLatents"])
    maxNLatents = int(guiConfig["latents"]["maxNLatents"])
    firstIndPoint = float(guiConfig["indPoints"]["firstIndPoint"])
    nIndPoints0 = [int(guiConfig["indPoints"]["numberOfIndPointsLatent{:d}".format(k+1)]) for k in range(nLatents0)]
    defaultNIndPoints = int(guiConfig["indPoints"]["defaultNIndPoints"])
    kernels0 = getKernels(nLatents=nLatents0, config=guiConfig)

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
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
                    value=condDist0,
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
                        value=linkFunc0,
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
                        value=embedding0,
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
                    value=nLatents0,
                    marks={i: str(i) for i in range(minNLatents, maxNLatents)},
                )],
                style={"width": "25%"}
            ),
        ], style={"padding-bottom": "20px"}),
        html.Div(id="kernelsTypesContainer", children=[]),
        html.Hr(),
        html.H4(children="Data"),
        html.Div(children=[
            html.Div(children=[
                dcc.Upload(
                    id="uploadSpikes",
                    children=html.Button("Upload Spikes"),
                    style={"display": "inline-block", "width": "30%", "padding-right": "20px"},
                ),
                html.Div(
                    id="spikesInfo",
                    style={"display": "inline-block", "width": "30%"},
                ),
            ], style={"display": "flex", "flex-wrap": "wrap", "width": 1500, "padding-bottom": "20px", "background-color": "white"}),
            html.Div(children=[
                html.Label("Spikes Variable Name"),
                html.Div(children=[
                    dcc.Dropdown(
                        id="spikesTimesVar",
                        disabled=True,
                        # # style={"display": "inline-block", "width": "30%", "padding-right": "20px"}
                        # style={"display": "inline-block", "width": "200px", "padding-right": "20px"}
                        # style={"display": "inline-block", "width": "30%", "padding-right": "20px"}
                        style={"width": "165px", "padding-right": "20px"}
                    ),
                    html.Div(
                        id="nTrialsAndNNeuronsInfoComponent",
                        # style={"display": "inline-block", "width": "30%"},
                        style={"display": "inline-block"},
                    ),
                ], style={"display": "flex", "flex-wrap": "wrap", "width": 800, "padding-bottom": "20px", "background-color": "white"}),
            ], style={"width": "200px"}),
        ], style={"width": 800, "background-color": "white"}),
        html.Div(
            id="nTrialsComponent",
            # style={'display': 'none'},
        ),
        html.Div(
            id="nNeuronsComponent",
            # style={'display': 'none'},
        ),
        html.Hr(),
        html.Div(
            id="trialToPlotParentContainer",
            children=[
                html.Label("Trial to Plot"),
                dcc.Dropdown(
                    id="trialToPlotComponent",
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
            ], hidden=True, style={"width": "20%", "columnCount": 1}),
        html.Div(
            id="kernelParams0ParentContainer",
            children=[
                html.H4("Initial Kernels Parameters"),
                html.Div(id="kernelParams0Container", children=[]),
                html.Div(id="kernelParams0BufferContainer", children=[], hidden=False),
                html.Hr(),
            ],
            hidden=True,
        ),
        html.Div(
            id="embeddingParams0",
            children=[
            html.H4("Initial Conditions for Linear Embedding"),
            html.Div(children=[
                    html.Label("Mixing Matrix"),
                    dcc.Textarea(
                        id="C0string",
                        required=True,
                    ),
                ]),
                html.Div(children=[
                    html.Label("Offset Vector"),
                    dcc.Textarea(
                        id="d0string",
                        required=True,
                    ),
                ]),
                html.Hr(),
            ], hidden=True),
        html.Div(
            id="trialsLengthsParentContainer",
            children=[
                html.H4("Trials Lengths"),
                html.Div(id="trialsLengthsContainer", children=[]),
                html.Hr(),
            ],
            hidden=True,
        ),
        html.Div(
            id="nIndPointsPerLatentParentContainer",
            children=[
                html.H4("Number of Inducing Points"),
                html.Div(id="nIndPointsPerLatentContainer", children=[]),
                html.Hr(),
            ],
            hidden=True),
        html.Div(
            id="svPosteriorOnIndPointsParams0ParentContainer",
            children=[
                html.H4("Initial Conditions for Variational Posterior on Inducing Points"),
                html.Div(id="svPosteriorOnIndPointsParams0Container"),
                html.Hr(),
            ],
            hidden=True),
        html.Div(
            id="optimParams",
            children=[
                html.H4("EM Parameters"),
                html.Div(children=[
                    html.Label("Maximum EM iterations"),
                    dcc.Input(
                        id="emMaxIter",
                        type="number",
                        required=True,
                        value=int(guiConfig["emParams"]["emMaxIter"]),
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Expectation Step"),
                    dcc.Checklist(
                        id="eStepEstimate",
                        options=[{"label": "estimate",
                                  "value": "True"}],
                        value=[guiConfig["emParams"]["eStepEstimate"]],
                    ),
                    html.Label("Maximum iterations"),
                    dcc.Input(
                        id="eStepMaxIter",
                        type="number",
                        required=True,
                        value=int(guiConfig["emParams"]["eStepMaxIter"]),
                    ),
                    html.Label("Learning rate"),
                    dcc.Input(
                        id="eStepLR",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["eStepLR"]),
                    ),
                    html.Label("Tolerance"),
                    dcc.Input(
                        id="eStepTol",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["eStepTol"]),
                    ),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        id="eStepLineSearchFn",
                        options=[
                            {"label": "strong wolfe", "value": "eStepLineSearchStrong_wolfe"},
                        ],
                        value=guiConfig["emParams"]["eStepLineSearchFn"],
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Embedding Parameters"),
                    dcc.Checklist(
                        id="mStepEmbeddingEstimate",
                        options=[{"label": "estimate", "value": "True"}],
                        value=[guiConfig["emParams"]["mStepEmbeddingEstimate"]],
                    ),
                    html.Label("Maximum iterations"),
                    dcc.Input(
                        id="mStepEmbeddingMaxIter",
                        type="number",
                        required=True,
                        value=int(guiConfig["emParams"]["mStepEmbeddingMaxIter"]),
                    ),
                    html.Label("Learning rate"),
                    dcc.Input(
                        id="mStepEmbeddingLR",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["mStepEmbeddingLR"]),
                    ),
                    html.Label("Tolerance"),
                    dcc.Input(
                        id="mStepEmbeddingTol",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["mStepEmbeddingTol"]),
                    ),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        id="mStepEmbeddingLineSearchFn",
                        options=[
                            {"label": "strong wolfe", "value": "mStepEmbeddingLineSearchStrong_wolfe"},
                        ],
                        value=guiConfig["emParams"]["mStepEmbeddingLineSearchFn"],
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Kernels Parameters"),
                    dcc.Checklist(
                        id="mStepKernelsEstimate",
                        options=[{"label": "estimate", "value": "mStepKernelsEstimate"}],
                        value=[guiConfig["emParams"]["mStepKernelsEstimate"]],
                    ),
                    html.Label("Maximum iterations"),
                    dcc.Input(
                        id="mStepKernelsMaxIter",
                        type="number",
                        required=True,
                        value=int(guiConfig["emParams"]["mStepKernelsMaxIter"]),
                    ),
                    html.Label("Learning rate"),
                    dcc.Input(
                        id="mStepKernelsLR",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["mStepKernelsLR"]),
                    ),
                    html.Label("Tolerance"),
                    dcc.Input(
                        id="mStepKernelsTol",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["mStepKernelsTol"]),
                    ),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        id="mStepKernelsLineSearchFn",
                        options=[
                            {"label": "strong wolfe", "value": "mStepKernelsLineSearchStrong_wolfe"},
                        ],
                        value=guiConfig["emParams"]["mStepKernelsLineSearchFn"],
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Inducing Points Parameters"),
                    dcc.Checklist(
                        id="mStepIndPointsEstimate",
                        options=[{"label": "estimate", "value": "mStepIndPointsEstimate"}],
                        value=[guiConfig["emParams"]["mStepIndPointsEstimate"]],
                    ),
                    html.Label("Maximum iterations"),
                    dcc.Input(
                        id="mStepIndPointsMaxIter",
                        type="number",
                        required=True,
                        value=int(guiConfig["emParams"]["mStepIndPointsMaxIter"]),
                    ),
                    html.Label("Learning rate"),
                    dcc.Input(
                        id="mStepIndPointsLR",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["mStepIndPointsLR"]),
                    ),
                    html.Label("Tolerance"),
                    dcc.Input(
                        id="mStepIndPointsTol",
                        type="number",
                        required=True,
                        value=float(guiConfig["emParams"]["mStepIndPointsTol"]),
                    ),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        id="mStepIndPointsLineSearchFn",
                        options=[
                            {"label": "strong wolfe", "value": "mStepIndPointsLineSearchStrong_wolfe"},
                        ],
                        value=guiConfig["emParams"]["mStepIndPointsLineSearchFn"],
                    ),
                ], style={"padding-bottom": "20px"}),
            ]
        ),
        html.Hr(),
        html.H4("Miscellaneous Parameters"),
        html.Div(children=[
            html.Label("Number of quadrature points"),
            dcc.Input(
                id="nQuad",
                type="number",
                required=True,
                value=int(guiConfig["miscParams"]["nQuad"]),
            ),
        ], style={"padding-bottom": "20px"}),
        html.Div(children=[
            html.Label("Variance added to kernel covariance matrix for inducing points"),
            dcc.Input(
                id="indPointsLocsKMSRegEpsilon",
                type="number",
                required=True,
                value=float(guiConfig["miscParams"]["indPointsLocsKMSRegEpsilon"]),
            ),
        ], style={"padding-bottom": "20px"}),
        html.Hr(),
        html.Button("Estimate", id="doEstimate", n_clicks=0),
        html.Div(id="estimationRes"),
    ])

    @app.callback(
        Output("kernelsTypesContainer", "children"),
        [Input("nLatentsComponent", "value")],
        [State("kernelsTypesContainer", "children")])
    def populateKernelsTypes(nLatentsComponentValue, kernelsTypesContainerChildren):
        if nLatentsComponentValue is None:
            raise PreventUpdate
        if nLatentsComponentValue==len(kernelsTypesContainerChildren):
            raise PreventUpdate
        elif nLatentsComponentValue<len(kernelsTypesContainerChildren):
            return kernelsTypesContainerChildren[:nLatentsComponentValue]
        elif len(kernelsTypesContainerChildren)==0:
            newChildren = []
            for k in range(nLatentsComponentValue):
                aDiv = html.Div(children=[
                    html.Div(children=[
                        html.Label("Kernel {:d} Type".format(k+1)),
                        dcc.Dropdown(
                            id={
                                "type": "kernelTypeComponent",
                                "latent": k
                            },
                            options=[
                                {"label": "Exponential Quadratic", "value": "ExponentialQuadraticKernel"},
                                {"label": "Periodic", "value": "PeriodicKernel"},
                            ],
                            value=type(kernels0[k]).__name__,
                            style={"width": "45%"}
                        ),
                    ]),
                ], style={"columnCount": 1})
                newChildren.append(aDiv)
            answer = kernelsTypesContainerChildren + newChildren
            return answer
        elif nLatentsComponentValue>len(kernelsTypesContainerChildren):
            newChildren = []
            for k in range(len(kernelsTypesContainerChildren),
                           nLatentsComponentValue):
                aDiv = html.Div(children=[
                    html.Div(children=[
                        html.Label("Kernel {:d} Type".format(k+1)),
                        dcc.Dropdown(
                            id={
                                "type": "kernelTypeComponent",
                                "latent": k
                            },
                            options=[
                                {"label": "Exponential Quadratic", "value": "ExponentialQuadraticKernel"},
                                {"label": "Periodic", "value": "PeriodicKernel"},
                            ],
                            style={"width": "45%"}
                        ),
                    ]),
                ], style={"columnCount": 1})
                newChildren.append(aDiv)
            answer = kernelsTypesContainerChildren + newChildren
            return answer

    @app.callback(
        [Output("kernelParams0ParentContainer", "hidden"),
         Output("kernelParams0Container", "children")],
        [Input("kernelsTypesContainer", "children")],
        [State({"type": "kernelTypeComponent", "latent": ALL}, "value"),
         State("kernelParams0Container", "children")])
    def addOrRemoveKernelsParams0(kernelTypesContainerChildren,
                                  kernelTypeComponentValues,
                                  kernelParams0ContainerChildren):
        nLatents = len(kernelTypesContainerChildren)
        # pdb.set_trace()
        if nLatents==0:
            # non-relevant event
            raise PreventUpdate
        if len(kernelParams0ContainerChildren)==0:
            # initialize kernels params with those from kernels0
            newChildren = []
            for k in range(nLatents):
                kernelType = type(kernels0[k]).__name__
                ### being convert the values in kernels0[k].getNamedParams() from tensor to number
                namedKernelParamsTmp = kernels0[k].getNamedParams()
                values = [value.item() for value in namedKernelParamsTmp.values()]
                keys = list(namedKernelParamsTmp.keys())
                namedKernelParams = dict(zip(keys, values))
                ### end convert the values in kernels0[k].getNamedParams() from tensor to number
                aDiv = getKernelParams0Div(kernelType=kernelType, namedKernelParams=namedKernelParams, latentID=k)
                newChildren.append(aDiv)
        elif nLatents<len(kernelParams0ContainerChildren):
            # remove kernel params from the end
            newChildren = kernelParams0ContainerChildren[:nLatents]
        elif nLatents>len(kernelParams0ContainerChildren):
            # pdb.set_trace()
            newChildren = []
            nKernelsParams0ToAdd = nLatents-len(kernelParams0ContainerChildren)
            newChildren = kernelParams0ContainerChildren
            namedKernelParams = {"LengthScale": None, "Period": None}
            for k in range(len(kernelParams0ContainerChildren), nLatents):
                kernelType = kernelTypeComponentValues[k]
                aDiv = getKernelParams0Div(kernelType=kernelType, namedKernelParams=namedKernelParams, latentID=k)
                newChildren.append(aDiv)
                # pdb.set_trace()
        kernelParams0ParentContainerHidden = False
        # pdb.set_trace()
        return kernelParams0ParentContainerHidden, newChildren

    @app.callback([Output({"type": "kernelTypeOfParam0", "latent": MATCH}, "children"),
                   Output({"type": "lengthScaleParam0", "latent": MATCH}, "value"),
                   Output({"type": "periodParam0", "latent": MATCH}, "value"),
                   Output({"type": "periodParam0Container", "latent": MATCH}, "style")],
                  [Input({"type": "kernelTypeComponent", "latent": MATCH}, "value")],
                  [State({"type": "kernelTypeOfParam0", "latent": MATCH}, "children"),
                   State({"type": "kernelTypeOfParam0", "latent": MATCH}, "hidden"),],
                 )
    def updateKernelsParams0(kernelTypeComponentValue, kernelTypeOfParam0Children, kernelTypeOfParam0Style):
        if kernelTypeComponentValue is None or kernelTypeComponentValue==kernelTypeOfParam0Children:
            raise PreventUpdate
        if kernelTypeComponentValue=="PeriodicKernel":
            kernelType = "Periodic"
            lengthScaleValue = None
            periodValue = None
            periodContainerStyle = {"display": "inline-block", "width": "30%"}
        elif kernelTypeComponentValue=="ExponentialQuadraticKernel":
            kernelType = "Exponential Quadratic"
            lengthScaleValue = None
            periodValue = None
            periodContainerStyle = {"display": "none"}
        else:
            raise RuntimeError("Invalid kernel type: {:s}".format(kernelTypeComponentValue))
        return kernelType, lengthScaleValue, periodValue, periodContainerStyle

    @app.callback(
        Output("kernelParams0BufferContainer", "children"),
        [Input("kernelParams0Container", "children")])
    def createKernelsParams0Buffer(kernelParams0ContainerChildren):
        # pdb.set_trace()
        nLatents = len(kernelParams0ContainerChildren)
        if nLatents is None:
            raise PreventUpdate

        children = []
        for k in range(nLatents):
            aDiv = html.Div(
                id={
                    "type": "kernelParams0Buffer",
                    "latent": k
                },
                children=["Empty kernelParams0Buffer for latent {:d}".format(k)])
            children.append(aDiv)
        return children

    @app.callback(
        Output({"type": "kernelParams0Buffer", "latent": MATCH}, "children"),
        [Input("kernelParams0BufferContainer", "children"),
         Input({"type": "kernelTypeComponent", "latent": MATCH}, "value"),
         Input({"type": "lengthScaleParam0", "latent": MATCH}, "value"),
         Input({"type": "periodParam0", "latent": MATCH}, "value"),
        ])
    def populateKernelParams0Buffer(kernelParams0ContainerChildren, kernelTypeComponentValue, lengthScaleParam0Value, periodParam0Value):
        # pdb.set_trace()
        if lengthScaleParam0Value is None:
            lengthScaleParam0Value = -1.0
        if periodParam0Value is None:
            periodParam0Value = -1.0
        if kernelTypeComponentValue=="PeriodicKernel":
            stringRep = "{:f},{:f}".format(lengthScaleParam0Value, periodParam0Value)
            params0 = np.fromstring(stringRep, sep=",")
        elif kernelTypeComponentValue=="ExponentialQuadraticKernel":
            stringRep = "{:f}".format(lengthScaleParam0Value)
            params0 = np.fromstring(stringRep, sep=",")
        elif kernelTypeComponentValue is None:
            stringRep = "{:f}".format(-1.0)
            params0 = np.fromstring(stringRep, sep=",")
        else:
            raise RuntimeError("Invalid kernelTypeComponent={:s}".format(kernelTypeComponentValue))
        answer = [np.array_repr(params0)]
        return answer

    @app.callback([Output("spikesInfo", "children"),
                   Output("spikesTimesVar", "options"),
                   Output("spikesTimesVar", "disabled")],
                  [Input("uploadSpikes", "contents")],
                  [State("uploadSpikes", "filename")])
    def loadSpikesTimesVar(contents, filename):
        if contents is not None:
            spikesInfoChildren = [
                "Filename: {:s}".format(filename)
            ]
            varsNames = getContentsVarsNames(contents=contents, filename=filename)
            filteredVarsNames = [varName for varName in varsNames if not varName.startswith("__")]
            spikesTimesVarsOptions = [{"label": varName, "value": varName} for varName in filteredVarsNames]
            return spikesInfoChildren, spikesTimesVarsOptions, False
        raise PreventUpdate

    @app.callback([Output("nTrialsComponent", "children"),
                   Output("nNeuronsComponent", "children"),
                   Output("nTrialsAndNNeuronsInfoComponent", "children"),
                   Output("trialToPlotComponent", "options"),
                   Output("trialToPlotComponent", "value"),
                   Output("trialToPlotParentContainer", "hidden"),
                   Output("trialsLengthsContainer", "children"),
                   Output("trialsLengthsParentContainer", "hidden")],
                  [Input("spikesTimesVar", "value")],
                  [State("uploadSpikes", "contents"),
                   State("uploadSpikes", "filename")])
    def propagateNewSpikesTimesInfo(spikesTimesVar, contents, filename):
        # pdb.set_trace()
        if contents is not None and spikesTimesVar is not None:
            spikesTimes = getSpikesTimes(contents=contents, filename=filename, spikesTimesVar=spikesTimesVar)
            nTrials = spikesTimes.shape[0]
            nNeurons = spikesTimes.shape[1]
            trialsLengthsGuesses = guessTrialsLengths(spikesTimes=spikesTimes)
            nTrialsAndNNeuronsInfo = "trials: {:d}, neurons: {:d}".format(nTrials, nNeurons)
            trialToPlotOptions = [{"label": str(r+1), "value": r} for r in range(nTrials)]
            trialToPlotValue = 0
            trialToPlotDivHidden = False
            #
            trialsLengthsChildren = []
            for r in range(nTrials):
                aDiv = html.Div(children=[
                    html.Label("Trial {:d}".format(r+1)),
                    dcc.Input(
                        id={
                            "type": "trialsLengths",
                            "latent": r,
                        },
                        type="number",
                        placeholder="trial length",
                        min=0,
                        required=True,
                        value = trialsLengthsGuesses[r],
                    ),
                ])
                trialsLengthsChildren.append(aDiv)
            #
            trialsLengthsParentContainerHidden = False
            return nTrials, nNeurons, nTrialsAndNNeuronsInfo, trialToPlotOptions, trialToPlotValue, trialToPlotDivHidden, trialsLengthsChildren, trialsLengthsParentContainerHidden
        raise PreventUpdate

    @app.callback(Output("trialRastergram", "figure"),
                  [Input("trialToPlotComponent", "value")],
                  [State("spikesTimesVar", "value"),
                   State("uploadSpikes", "contents"),
                   State("uploadSpikes", "filename")])
    def updateRastergram(trialToPlot, spikesTimesVar, contents, filename):
        # pdb.set_trace()
        if trialToPlot>=0 and spikesTimesVar is not None:
            spikesTimes = getSpikesTimes(contents=contents, filename=filename, spikesTimesVar=spikesTimesVar)
            trialSpikesTimes = spikesTimes[trialToPlot,:]
            title="Trial {:d}".format(trialToPlot+1)
            rastergram = getRastergram(trialSpikesTimes=trialSpikesTimes, title=title)
            return rastergram
        raise PreventUpdate

    @app.callback([Output("nIndPointsPerLatentContainer", "children"),
                   Output("nIndPointsPerLatentParentContainer", "hidden") ],
                  [Input("nLatentsComponent", "value")],
                  [State("nIndPointsPerLatentContainer", "children")])
    def showNIndPointsPerLatent(nLatents, nIndPointsPerLatentContainerChildren):
        # pdb.set_trace()
        if nLatents is None:
            raise PreventUpdate
        if len(nIndPointsPerLatentContainerChildren)==0:
            someChildren = []
            for k in range(nLatents):
                aChildren = html.Div(children=[
                    html.Label("Latent {:d}".format(k+1)),
                    html.Div(
                        children=[
                        dcc.Slider(
                            id={
                                "type": "nIndPoints",
                                "latent": k
                            },
                            min=5,
                            max=50,
                            marks={i: str(i) for i in range(5, 51, 5)},
                            value=nIndPoints0[k],
                        )
                    ], style={"width": "25%", "height": "50px"}),
                ])
                someChildren.append(aChildren)
        elif len(nIndPointsPerLatentContainerChildren)<nLatents:
            someChildren = nIndPointsPerLatentContainerChildren
            for k in range(len(nIndPointsPerLatentContainerChildren), nLatents):
                aChildren = html.Div(children=[
                    html.Label("Latent {:d}".format(k+1)),
                    html.Div(
                        children=[
                        dcc.Slider(
                            id={
                                "type": "nIndPoints",
                                "latent": k
                            },
                            min=5,
                            max=50,
                            marks={i: str(i) for i in range(5, 51, 5)},
                            value=defaultNIndPoints,
                        )
                    ], style={"width": "25%", "height": "50px"}),
                ])
                someChildren.append(aChildren)
        elif len(nIndPointsPerLatentContainerChildren)>nLatents:
            someChildren = nIndPointsPerLatentContainerChildren[:nLatents]
        nIndPointsPerLatentParentContaineHidden = False
        return someChildren, nIndPointsPerLatentParentContaineHidden

    @app.callback([Output("svPosteriorOnIndPointsParams0Container", "children"),
                   Output("svPosteriorOnIndPointsParams0ParentContainer", "hidden")],
                  [Input({"type": "nIndPoints", "latent": ALL}, "value")])
    def populateSVPosteriorOnIndPointsInitialConditions(values):
        if len(values)==0:
            raise PreventUpdate
        initVar = 0.01
        someChildren = []
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
                        "latent": k
                    },
                    type="text",
                    size="{:d}".format(5*nIndPoints),
                    required=True,
                    value=np.array_repr(qMu0.squeeze()),
                ),
                html.Label("Spanning Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsVec0",
                        "latent": k
                    },
                    type="text",
                    size="{:d}".format(5*nIndPoints),
                    required=True,
                    value=np.array_repr(qVec0.squeeze()),
                ),
                html.Label("Diagonal Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsDiag0",
                        "latent": k
                    },
                    type="text",
                    size="{:d}".format(5*nIndPoints),
                    required=True,
                    value=np.array_repr(qDiag0.squeeze()),
                ),
            ], style={"padding-bottom": "30px"})
            someChildren.append(aChildren)
        svPosteriorOnIndPointsParams0ParentContainerHidden = False
        return someChildren, svPosteriorOnIndPointsParams0ParentContainerHidden

    @app.callback([Output("embeddingParams0", "hidden"),
                   Output("C0string", "value"),
                   Output("C0string", "style"),
                   Output("d0string", "value"),
                   Output("d0string", "style")],
                  [Input("spikesTimesVar", "value"),
                   Input("nLatentsComponent", "value")],
                  [State("uploadSpikes", "contents"),
                   State("uploadSpikes", "filename")])
    def populateEmbeddingInitialConditions(spikesTimesVar, nLatentsComponentValue, contents, filename):
        # pdb.set_trace()
        if spikesTimesVar is not None:
            spikesTimes = getSpikesTimes(contents=contents, filename=filename, spikesTimesVar=spikesTimesVar)
            nNeurons = spikesTimes.shape[1]
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
        [State("spikesTimesVar", "value"),
         State("uploadSpikes", "contents"),
         State("uploadSpikes", "filename"),
         State("conditionalDist", "value"),
         State("linkFunction", "value"),
         State("embeddingType", "value"),
         State("nLatentsComponent", "value"),
         State("C0string", "value"),
         State("d0string", "value"),
         State({"type": "kernelTypeComponent",  "latent": ALL}, "value"),
         State({"type": "kernelParams0Buffer", "latent": ALL}, "children"),
         State({"type": "nIndPoints",  "latent": ALL}, "value"),
         State({"type": "trialsLengths",  "latent": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsMu0",  "latent": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsVec0",  "latent": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsDiag0",  "latent": ALL}, "value"),
         #
         State("emMaxIter", "value"),
         #
         State("eStepEstimate", "value"),
         State("eStepMaxIter", "value"),
         State("eStepLR", "value"),
         State("eStepTol", "value"),
         State("eStepLineSearchFn", "value"),
         #
         State("mStepEmbeddingEstimate", "value"),
         State("mStepEmbeddingMaxIter", "value"),
         State("mStepEmbeddingLR", "value"),
         State("mStepEmbeddingTol", "value"),
         State("mStepEmbeddingLineSearchFn", "value"),
         #
         State("mStepKernelsEstimate", "value"),
         State("mStepKernelsMaxIter", "value"),
         State("mStepKernelsLR", "value"),
         State("mStepKernelsTol", "value"),
         State("mStepKernelsLineSearchFn", "value"),
         #
         State("mStepIndPointsEstimate", "value"),
         State("mStepIndPointsMaxIter", "value"),
         State("mStepIndPointsLR", "value"),
         State("mStepIndPointsTol", "value"),
         State("mStepIndPointsLineSearchFn", "value"),
         #
         State("nQuad", "value"),
         State("indPointsLocsKMSRegEpsilon", "value"),
        ])
    def estimateSVGPFA(doEstimateNClicks,
                       spikesTimesVar,
                       contents,
                       filename,
                       conditionalDist,
                       linkFunction,
                       embeddingType,
                       nLatentsComponentValue,
                       C0string,
                       d0string,
                       kernelTypeComponentValues,
                       kernelParams0Children,
                       nIndPoints,
                       trialsLengths,
                       svPosterioOnIndPointsMu0,
                       svPosterioOnIndPointsVec0,
                       svPosterioOnIndPointsDiag0,
                       #
                       emMaxIter,
                       #
                       eStepEstimate,
                       eStepMaxIter,
                       eStepLR,
                       eStepTol,
                       eStepLineSearchFn,
                       #
                       mStepEmbeddingEstimate,
                       mStepEmbeddingMaxIter,
                       mStepEmbeddingLR,
                       mStepEmbeddingTol,
                       mStepEmbeddingLineSearchFn,
                       #
                       mStepKernelsEstimate,
                       mStepKernelsMaxIter,
                       mStepKernelsLR,
                       mStepKernelsTol,
                       mStepKernelsLineSearchFn,
                       #
                       mStepIndPointsEstimate,
                       mStepIndPointsMaxIter,
                       mStepIndPointsLR,
                       mStepIndPointsTol,
                       mStepIndPointsLineSearchFn,
                       #
                       nQuad,
                       indPointsLocsKMSRegEpsilon,
                      ):
        # pdb.set_trace()
        if doEstimateNClicks>0:
            optimParams = {}
            optimParams["emMaxIter"] = emMaxIter
            #
            optimParams["eStepEstimate"] = eStepEstimate
            optimParams["eStepMaxIter"] = eStepMaxIter
            optimParams["eStepLR"] = eStepLR
            optimParams["eStepTol"] = eStepTol
            optimParams["eStepLineSearchFn"] = eStepLineSearchFn
            optimParams["eStepNIterDisplay"] = 1
            #
            optimParams["mStepEmbeddingEstimate"] = mStepEmbeddingEstimate
            optimParams["mStepEmbeddingMaxIter"] = mStepEmbeddingMaxIter
            optimParams["mStepEmbeddingLR"] = mStepEmbeddingLR
            optimParams["mStepEmbeddingTol"] = mStepEmbeddingTol
            optimParams["mStepEmbeddingLineSearchFn"] = mStepEmbeddingLineSearchFn
            optimParams["mStepEmbeddingNIterDisplay"] = 1
            #
            optimParams["mStepKernelsEstimate"] = mStepKernelsEstimate
            optimParams["mStepKernelsMaxIter"] = mStepKernelsMaxIter
            optimParams["mStepKernelsLR"] = mStepKernelsLR
            optimParams["mStepKernelsTol"] = mStepKernelsTol
            optimParams["mStepKernelsLineSearchFn"] = mStepKernelsLineSearchFn
            optimParams["mStepKernelsNIterDisplay"] = 1
            #
            optimParams["mStepIndPointsEstimate"] = mStepIndPointsEstimate
            optimParams["mStepIndPointsMaxIter"] = mStepIndPointsMaxIter
            optimParams["mStepIndPointsLR"] = mStepIndPointsLR
            optimParams["mStepIndPointsTol"] = mStepIndPointsTol
            optimParams["mStepIndPointsLineSearchFn"] = mStepIndPointsLineSearchFn
            optimParams["mStepIndPointsNIterDisplay"] = 1
            #
            optimParams["verbose"] = True

            nTrials = len(trialsLengths)
            spikesTimes = getSpikesTimes(contents=contents, filename=filename, spikesTimesVar=spikesTimesVar)
            runner = svGPFA_runner(firstIndPoint=firstIndPoint)
            runner.setSpikesTimes(spikesTimes=spikesTimes)
            runner.setConditionalDist(conditionalDist=conditionalDist)
            runner.setLinkFunction(linkFunction=linkFunction)
            runner.setEmbeddingType(embeddingType=embeddingType)
            runner.setKernels(kernelTypeComponentValues=kernelTypeComponentValues)
            runner.setKernelsParams0(kernelParams0Children=kernelParams0Children)
            runner.setNIndPointsPerLatent(nIndPointsPerLatent=nIndPoints)
            runner.setEmbeddingParams0(nLatents=nLatentsComponentValue, C0string=C0string, d0string=d0string),
            runner.setTrialsLengths(trialsLengths=trialsLengths)
            runner.setSVPosteriorOnIndPointsParams0(qMu0Strings=svPosterioOnIndPointsMu0, qSVec0Strings=svPosterioOnIndPointsVec0, qSDiag0Strings=svPosterioOnIndPointsDiag0, nTrials=nTrials)
            runner.setNQuad(nQuad=nQuad)
            runner.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
            runner.setOptimParams(optimParams=optimParams)

            runner.run()

            pdb.set_trace()

            return ["Successful estimation!!!"]

    app.run_server(debug=True)

if __name__ == "__main__":
    main(sys.argv)

