import pdb
import sys
import datetime
import configparser
import importlib
import numpy as np
from numpy import array
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
from guiUtils import getContentsVarsNames, getSpikesTimes, getRastergram

def main(argv):
    if(len(argv)!=2):
        print("Usage: {:s} <gui ini>")
        sys.exit(1)
    guiFilename = argv[1]
    guiConfig = configparser.ConfigParser()
    guiConfig.read(guiFilename)
    minNLatents = int(guiConfig["latents"]["minNLatents"])
    maxNLatents = int(guiConfig["latents"]["maxNLatents"])
    firstIndPoint = float(guiConfig["indPoints"]["firstIndPoint"])

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.config['suppress_callback_exceptions'] = True
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
                        # disabled=True, 
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
            hidden=True),
        html.Div(
            id="trialsLengthsParentContainer",
            children=[
                html.H4("Trials lengths"),
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
            id="embeddingParams0",
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
                html.Hr(),
            ], hidden=True),
        html.Div(
            id="optimParams",
            children=[
                html.H4("EM Parameters"),
                html.Div(children=[
                    html.Label("Maximum EM iterations"),
                    dcc.Input(id="maxIter", type="number"),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Expectation Step"),
                    dcc.Checklist(id="eStepEstimate",
                                  options=[{"label": "estimate",
                                            "value": "eStepEstimate"}],
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="eStepMaxIter", type="number"),
                    html.Label("Learning rate"),
                    dcc.Input(id="eStepLR", type="number"),
                    html.Label("Tolerance"),
                    dcc.Input(id="eStepTol", type="number", required=True),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "eStepLineSearchStrong_wolfe"},
                            {"label": "none", "value": "eStepLineSearchNone"},
                        ],
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Embedding Parameters"),
                    dcc.Checklist(id="mStepEmbedding",
                                  options=[{"label": "estimate",
                                            "value": "mStepEmbeddingEstimate"}],
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="mStepEmbeddingMaxIter", type="number"),
                    html.Label("Learning rate"),
                    dcc.Input(id="mStepEmbeddingLR", type="number"),
                    html.Label("Tolerance"),
                    dcc.Input(id="mStepEmbeddingTol", type="number"),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "mStepEmbeddingLineSearchStrong_wolfe"},
                            {"label": "none", "value": "mStepEmbeddingLineSearchNone"},
                        ],
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Kernels Parameters"),
                    dcc.Checklist(id="mStepKernelsParams",
                                  options=[{"label": "estimate",
                                            "value": "mStepKernelsParamsEstimate"}],
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="mStepKernelsParamsMaxIter", type="number"),
                    html.Label("Learning rate"),
                    dcc.Input(id="mStepKernelsParamsLR", type="number"),
                    html.Label("Tolerance"),
                    dcc.Input(id="mStepKernelsParamsTol", type="number"),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "mStepKernelsParamsLineSearchStrong_wolfe"},
                            {"label": "none", "value": "mStepKernelsParamsLineSearchNone"},
                        ],
                    ),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.H6("Maximization Step on Inducing Points Parameters"),
                    dcc.Checklist(id="mStepIndPointsParams",
                                  options=[{"label": "estimate",
                                            "value": "mStepIndPointsParamsEstimate"}],
                                 ),
                    html.Label("Maximum iterations"),
                    dcc.Input(id="mStepIndPointsParamsMaxIter", type="number"),
                    html.Label("Learning rate"),
                    dcc.Input(id="mStepIndPointsParamsLR", type="number"),
                    html.Label("Tolerance"),
                    dcc.Input(id="mStepIndPointsParamsTol", type="number"),
                    html.Label("Line search"),
                    dcc.RadioItems(
                        options=[
                            {"label": "strong wolfe", "value": "mStepIndPointsParamsLineSearchStrong_wolfe"},
                            {"label": "none", "value": "mStepIndPointsParamsLineSearchNone"},
                        ],
                    ),
                ], style={"padding-bottom": "20px"}),
            ]
        ),
        html.Hr(),
        html.H4("Miscellaneous Parameters"),
        html.Div(children=[
            html.Label("Number of quadrature points"),
            dcc.Input(id="nQuad", type="number"),
        ], style={"padding-bottom": "20px"}),
        html.Div(children=[
            html.Label("Variance added to kernel covariance matrix for inducing points"),
            dcc.Input(id="indPointsLocsKMSRegEpsilon", type="number"),
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

    @app.callback([Output({"type": "kernelTypeOfParam0", "latent": MATCH}, "children"),
                   Output({"type": "lengthScaleParam0", "latent": MATCH}, "value"),
                   Output({"type": "lengthScaleParam0", "latent": MATCH}, "hidden"),
                   Output({"type": "periodParam0", "latent": MATCH}, "value"),
                   Output({"type": "periodParam0", "latent": MATCH}, "hidden"),
                   Output({"type": "periodParam0Label", "latent": MATCH}, "children")],
                  [Input({"type": "kernelTypeComponent", "latent": MATCH}, "value")],
                  [State({"type": "kernelTypeOfParam0", "latent": MATCH}, "value")],
                 )
    def updateKernelsParams0(kernelTypeComponentValue, kernelTypeOfParam0Value):
        # pdb.set_trace()
        if kernelTypeComponentValue is None or kernelTypeComponentValue==kernelTypeOfParam0Value:
            raise PreventUpdate
        if kernelTypeComponentValue=="PeriodicKernel":
            kernelType = "Periodic"
            lengthScaleValue = None
            lengthScaleHidden = False
            periodValue = None
            periodValueHidden = False
            periodLabel = "Period"
        elif kernelTypeComponentValue=="ExponentialQuadraticKernel":
            kernelType = "Exponential Quadratic"
            lengthScaleValue = None
            lengthScaleHidden = False
            periodValue = None
            periodValueHidden = True
            periodLabel = ""
        else:
            raise RuntimeError("Invalid kernel type: {:s}".format(kernelTypeComponentValue))
        return [kernelType, lengthScaleValue, lengthScaleHidden, periodValue, periodValueHidden, periodLabel]

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
        if nLatents<len(kernelParams0ContainerChildren):
            # remove kernel params from the end
            # pdb.set_trace()
            newChildren = kernelParams0ContainerChildren[:nLatents]
            return newChildren
        elif nLatents>len(kernelParams0ContainerChildren):
            # pdb.set_trace()
            nKernelsParams0ToAdd = nLatents-len(kernelParams0ContainerChildren)
            newChildren = kernelParams0ContainerChildren
            for k in range(len(kernelParams0ContainerChildren), nLatents):
                kTypeToAdd = kernelTypeComponentValues[k]
                if kTypeToAdd=="PeriodicKernel":
                    aDiv = html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "kernelTypeParam0Label",
                                            "latent": k
                                        },
                                        children="Kernel {:d} Type".format(k+1)),
                                    html.Label(
                                        id={
                                            "type": "kernelTypeOfParam0",
                                            "latent": k
                                        },
                                        children="Periodic Kernel"
                                    ),
                                ], style={"display": "inline-block", "width": "30%"}
                            ),
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "lengthScaleParam0Label",
                                            "latent": k
                                        },
                                        children="Length Scale"),
                                    dcc.Input(
                                        id={
                                            "type": "lengthScaleParam0",
                                            "latent": k
                                        },
                                        type="number",
                                        min=0,
                                    ),
                                ], style={"display": "inline-block", "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "periodParam0Label",
                                            "latent": k
                                        },
                                        children="Period"),
                                    dcc.Input(
                                        id={
                                            "type": "periodParam0",
                                            "latent": k
                                        },
                                        type="number",
                                        min=0,
                                    ),
                                ], style={"display": "inline-block", "width": "30%"}),
                            ])
                elif kTypeToAdd=="ExponentialQuadraticKernel":
                    aDiv = html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "kernelTypeParam0Label",
                                            "latent": k
                                        },
                                        children="Kernel {:d} Type".format(k+1)),
                                    html.Label(
                                        id={
                                            "type": "kernelTypeOfParam0",
                                            "latent": k
                                        },
                                        children="Exponential Quadratic"
                                    ),
                                ], style={"display": "inline-block", "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "lengthScaleParam0Label",
                                            "latent": k
                                        },
                                        children="Length Scale"),
                                    dcc.Input(
                                        id={
                                            "type": "lengthScaleParam0",
                                            "latent": k
                                        },
                                        type="number",
                                        min=0,
                                    ),
                                ],
                                style={"display": "inline-block", "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "periodParam0Label",
                                            "latent": k
                                        },
                                        children="Period"),
                                        dcc.Input(
                                            id={
                                                "type": "periodParam0",
                                                "latent": k
                                            },
                                            type="number",
                                            min=0,
                                            value=None,
                                        ),
                                    ],
                                hidden=True,
                                style={"display": "inline-block", "width": "30%"}
                                ),
                        ])
                elif kTypeToAdd is None:
                    aDiv = html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "kernelTypeParam0Label",
                                            "latent": k
                                        },
                                        children="Kernel {:d} Type".format(k+1)),
                                    html.Label(
                                        id={
                                            "type": "kernelTypeOfParam0",
                                            "latent": k
                                        },
                                    ),
                                ], style={"display": "inline-block", "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "lengthScaleParam0Label",
                                            "latent": k
                                        },
                                        children="Length Scale"),
                                    dcc.Input(
                                        id={
                                            "type": "lengthScaleParam0",
                                            "latent": k
                                        },
                                        type="number",
                                        min=0,
                                    ),
                                ], style={"display": "inline-block", "width": "30%"}),
                            html.Div(
                                children=[
                                    html.Label(
                                        id={
                                            "type": "periodParam0Label",
                                            "latent": k
                                        },
                                        children="Period"),
                                    dcc.Input(
                                        id={
                                            "type": "periodParam0",
                                            "latent": k
                                        },
                                        type="number",
                                        min=0,
                                        value=None,
                                    ),
                                ],
                                hidden=True,
                                style={"display": "inline-block", "width": "30%"}
                            ),
                        ])
                else:
                    raise RuntimeError("Invalid Kernel type {:s}".format(kTypeToAdd))
                newChildren.append(aDiv)
            kernelParams0ParentContainerHidden = False
            return kernelParams0ParentContainerHidden, newChildren

    @app.callback(
        Output("kernelParams0BufferContainer", "children"),
        [Input("nLatentsComponent", "value")])
    def createKernelsParams0Buffer(nLatentsComponentValue):
        # pdb.set_trace()
        if nLatentsComponentValue is None:
            raise PreventUpdate

        children = []
        for k in range(nLatentsComponentValue):
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
        [Input({"type": "kernelTypeComponent", "latent": MATCH}, "value"),
         Input({"type": "lengthScaleParam0", "latent": MATCH}, "value"),
         Input({"type": "periodParam0", "latent": MATCH}, "value"),
        ])
    def populateKernelParams0Buffer(kernelTypeComponent, lengthScaleParam0, periodParam0):
        # pdb.set_trace()
        if lengthScaleParam0 is None:
            lengthScaleParam0 = -1.0
        if periodParam0 is None:
            periodParam0 = -1.0
        if kernelTypeComponent=="PeriodicKernel":
            stringRep = "{:f},{:f}".format(lengthScaleParam0, periodParam0)
            params0 = np.fromstring(stringRep, sep=",")
        elif kernelTypeComponent=="ExponentialQuadraticKernel":
            stringRep = "{:f}".format(lengthScaleParam0)
            params0 = np.fromstring(stringRep, sep=",")
        elif kernelTypeComponent is None:
            params0 = ""
        else:
            raise RuntimeError("Invalid kernelTypeComponent={:s}".format(kernelTypeComponent))
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
                        type="number", 
                        placeholder="trial length",
                        min=0,
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

#     @app.callback(Output("trialsLengths", "children"),
#                   [Input("spikesInfo", "children")])
#     def showTrialLengths(spikesInfoChildren):
#         # pdb.set_trace()
#         if spikesInfoChildren is not None:
#             nTrials = runner.getSpikesTimes().shape[0]
#             someChildren = [html.H4("Trials lengths")]
#             for r in range(nTrials):
#                 aDiv = html.Div(children=[
#                     html.Label("Trial {:d}".format(r+1)),
#                     dcc.Input(
#                         type="number", placeholder="trial length",
#                         min=0,
#                     ),
#                 ])
#                 someChildren.append(aDiv)
#             someChildren.append(html.Hr())
#             return someChildren
#         # return [None]
#         raise PreventUpdate

    @app.callback([Output("nIndPointsPerLatentContainer", "children"),
                   Output("nIndPointsPerLatentParentContainer", "hidden") ],
                  [Input("nLatentsComponent", "value")])
    def showNIndPointsPerLatent(nLatentsComponentValue):
        # pdb.set_trace()
        if nLatentsComponentValue is None:
            raise PreventUpdate
        someChildren = []
        for k in range(nLatentsComponentValue):
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
                        value=20,
                    )
                ], style={"width": "25%", "height": "50px"}),
            ])
            someChildren.append(aChildren)
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
                    value=np.array_repr(qMu0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
                html.Label("Spanning Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsVec0",
                        "latent": k
                    },
                    type="text",
                    value=np.array_repr(qVec0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
                html.Label("Diagonal Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsDiag0",
                        "latent": k
                    },
                    type="text",
                    value=np.array_repr(qDiag0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
            ], style={"padding-bottom": "30px"})
            someChildren.append(aChildren)
        svPosteriorOnIndPointsParams0ParentContainerHidden = False
        return someChildren, svPosteriorOnIndPointsParams0ParentContainerHidden

#     @app.callback([Output("embeddingParams0", "hidden"),
#                    Output("C0string", "value"),
#                    Output("C0string", "style"),
#                    Output("d0string", "value"),
#                    Output("d0string", "style")],
#                   [Input("nLatentsComponent", "value"),
#                    Input("spikesInfo", "children")])
#     def populateEmbeddingInitialConditions(nLatentsComponentValue, spikesInfoChildren):
#         # pdb.set_trace()
#         if spikesInfoChildren is not None:
#             nNeurons = runner.getSpikesTimes().shape[1]
#             C0 = np.random.uniform(size=(nNeurons, nLatentsComponentValue))
#             d0 = np.random.uniform(size=(nNeurons, 1))
#             C0style = {"width": nLatentsComponentValue*175, "height": 300}
#             d0style={"width": 200, "height": 300}
#             # answer = [False, np.array2string(C0), C0style, np.array2string(d0), d0style]
#             answer = [False, np.array_repr(C0), C0style, np.array_repr(d0), d0style]
#             return answer
#         raise PreventUpdate

    @app.callback(
        Output("estimationRes", "children"),
        [Input("doEstimate", "n_clicks")],
        [State("conditionalDist", "value"),
         State("linkFunction", "value"),
         State("embeddingType", "value"),
         State("nLatentsComponent", "value"),
         State("C0string", "value"),
         State("d0string", "value"),
         State({"type": "kernelTypeComponent",  "latent": ALL}, "value"),
         State({"type": "kernelParams0Components",  "latent": ALL}, "children"),
         State({"type": "nIndPoints",  "latent": ALL}, "value"),
         State({"type": "trialsLenghts",  "latent": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsMu0",  "latent": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsVec0",  "latent": ALL}, "value"),
         State({"type": "svPosterioOnIndPointsDiag0",  "latent": ALL}, "value"),
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
                       kernelTypeComponent,
                       kernelParams0Components,
                       nIndPoints,
                       trialsLengths,
                       svPosterioOnIndPointsMu0,
                       svPosterioOnIndPointsVec0,
                       svPosterioOnIndPointsDiag0,
                       nQuad,
                       indPointsLocsKMSRegEpsilon,
                      ):
        # pdb.set_trace()
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

