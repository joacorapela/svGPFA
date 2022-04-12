import pdb
import sys
import os
import uuid
import datetime
import argparse
import configparser
import math
import multiprocessing as mp
import signal
import numpy as np
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
from plotly.colors import DEFAULT_PLOTLY_COLORS
sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
from guiUtils import getContentsVarsNames, getSpikesTimes, getRastergram, getKernels, getKernelParams0Div, guessTrialsLengths, svGPFA_runner
import utils.networking.multiprocessingUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("ini_filename", help="ini file with configuration parameters")
    parser.add_argument("--debug", help="start GUI with debug functionality", action="store_true", default=False)
    parser.add_argument("--non-local", help="provide GUI access in a specified port (default 8050)", action="store_true")
    args = parser.parse_args()

    if args.debug:
        print("debug on")
    else:
        print("debug off")
    guiFilename = args.ini_filename
    guiConfig = configparser.ConfigParser()
    guiConfig.read(guiFilename)

    dtLatentsTimes = 0.1

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

    def serve_layout():
        session_id = str(uuid.uuid4())
        aDiv = html.Div(children=[
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
                            style={"width": "165px", "padding-right": "20px"}
                        ),
                        html.Div(
                            id="nTrialsAndNNeuronsInfoComponent",
                            style={"display": "inline-block"},
                        ),
                        html.Div(
                            id="nTrialsComponent",
                            style={'display': 'none'},
                        ),
                        html.Div(
                            id="nNeuronsComponent",
                            style={'display': 'none'},
                        ),
                    ], style={"display": "flex", "flex-wrap": "wrap", "width": 800, "padding-bottom": "20px", "background-color": "white"}),
                ], style={"width": "200px"}),
            ], style={"width": 800, "background-color": "white"}),
            html.Hr(),
            html.Div(
                id="trialToPlotRasterParentContainer",
                children=[
                    html.Label("Trial to Plot"),
                    dcc.Dropdown(
                        id="trialToPlotRasterComponent",
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
                    html.Div(id="kernelParams0BufferContainer", children=[], hidden=True),
                    html.Hr(),
                ],
                hidden=True,
            ),
            html.Div(children=[
                # html.H4("EM Parameters"),
                html.Div(children=[
                    html.Label("Number of EM iterations"),
                    dcc.Input(
                        id="emMaxIter",
                        type="number",
                        required=True,
                        value=int(guiConfig["emParams"]["emMaxIter"]),
                    ),
                ], style={"padding-bottom": "20px"}),
            ]),
            html.Button(children="More parameters", id="moreParamsButton", n_clicks=0),
            html.Div(
                id="moreParamsContainer",
                hidden=True,
                children=[
                    html.Div(
                        id="embeddingParams0",
                        children=[
                        html.H4("Initial Conditions for Linear Embedding"),
                        html.Div(children=[
                                html.Label("Mixing Matrix"),
                                dcc.Textarea(
                                    id="C0string",
                                    required=True,
                                    spellCheck=False,
                                ),
                            ]),
                            html.Div(children=[
                                html.Label("Offset Vector"),
                                dcc.Textarea(
                                    id="d0string",
                                    required=True,
                                    spellCheck=False,
                                ),
                            ]),
                            html.Hr(),
                        ], hidden=True),
                    html.Div(
                        id="trialsLengthsParentContainer",
                        children=[
                            html.H4("Trials Durations (sec)"),
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
                                        {"label": "strong wolfe", "value": "strong_wolfe"},
                                        {"label": "none", "value": "noLineSearch"},
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
                                        {"label": "strong wolfe", "value": "strong_wolfe"},
                                        {"label": "none", "value": "noLineSearch"},
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
                                        {"label": "strong wolfe", "value": "strong_wolfe"},
                                        {"label": "none", "value": "noLineSearch"},
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
                                        {"label": "strong wolfe", "value": "strong_wolfe"},
                                        {"label": "none", "value": "noLineSearch"},
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
                ]),
            html.Hr(),
            html.Button("Estimate", id="doEstimate", n_clicks=0),
            html.Button("Cancel Estimation", id="cancelEstimation", n_clicks=0),
            html.Div(id="cancelEstimationDummyDiv", hidden=True),
            html.Div(id='sessionID', children=[session_id], hidden=True),
            html.Div(id='pid', hidden=True),
            html.Div(id='estimationStatus', children=["Not started"]),
            dcc.Interval(
                id='logIntervalComponent',
                interval=1*1000, # in milliseconds
                n_intervals=0,
                disabled=True,
            ),
            dcc.Interval(
                id='estimationProgressGraphsIntervalComponent',
                interval=1*2000, # in milliseconds
                n_intervals=0,
                disabled=True,
            ),
            html.Div(id="estimationRes"),
            html.Div(
                id="estimationProgressContainer",
                style={"display": "none"},
                children=[
                    dcc.Textarea(
                        id='logTextarea',
                        value='',
                        style={"display": "block", 'width': '60%', 'height': 300},
                        spellCheck=False,
                    ),
                    dcc.Graph(id="lowerBoundGraph"),
                    html.Label("Trial to Plot"),
                    dcc.Dropdown(
                        id="trialToPlotLatentsComponent",
                        options=[
                            {"label": "", "value": -1},
                        ],
                        value=-1,
                        style={"width": "30%"}
                    ),
                    html.Label("Latent to Plot"),
                    dcc.Dropdown(
                        id="latentToPlotComponent",
                        options=[
                            {"label": "dummy option", "value": -1},
                        ],
                        value=-1,
                        style={"width": "30%"}
                    ),
                    dcc.Input(
                        id="sampleRateForPlotting",
                        type="number",
                        required=True,
                        value=10,
                        style={"display": "none"}
                    ),
                    dcc.Graph(id="latentsGraph"),
                ],
            ),
            ])
        return aDiv

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.layout = serve_layout

    @app.callback([Output('moreParamsContainer', 'hidden'),
                   Output('moreParamsButton', 'children')],
                  [Input('moreParamsButton', 'n_clicks'),
                  ])
    def toggleHideShowMoreParams(moreParamsButtonNClicks):
        moreParamsContainerHidden = True
        moreParamsContainerChildren = "More Parameters"
        if moreParamsButtonNClicks is not None and moreParamsButtonNClicks%2==1:
           moreParamsContainerHidden = False
           moreParamsContainerChildren = "Less Parameters"

        return moreParamsContainerHidden, moreParamsContainerChildren

    @app.callback([Output('lowerBoundGraph', 'figure'),
                   Output('estimationStatus', 'children')],
                  [Input('estimationProgressGraphsIntervalComponent', 'n_intervals')],
                  [State('emMaxIter', 'value'),
                   State('sessionID', 'children'),
                   State('pid', 'children'),
                   State('lowerBoundGraph', 'figure'),
                  ])
    def update_lowerBoundGraph_live(estimationProgressGraphsIntervalComponent_n_intervals, emMaxIterValue, sessionIDChildren, pidChildren, lowerBoundGraphFigure):
        if estimationProgressGraphsIntervalComponent_n_intervals==0:
            raise PreventUpdate

        lowerBoundLockFN = "/tmp/lockLowerBound{:s}.lock".format(sessionIDChildren[0])
        lowerBoundStreamFN = "/tmp/bufferLowerBound{:s}.npy".format(sessionIDChildren[0])

        if os.path.exists(lowerBoundStreamFN):
            lowerBoundLock = utils.networking.multiprocessingUtils.FileLock(filename=lowerBoundLockFN)
            if not lowerBoundLock.is_locked():
                lowerBoundLock.lock()
                with open(lowerBoundStreamFN, 'rb') as f:
                    lowerBoundArray = np.load(f)
                lowerBoundLock.unlock()
                lowerBoundGraphFigure = dict({
                    "data": [{
                        "type": "scatter", "mode": "lines+markers",
                        "x": np.arange(len(lowerBoundArray)),
                        "y": lowerBoundArray,
                    }],
                    "layout": {
                        "xaxis": {"title": "Iteration", "range": [-1, emMaxIterValue+1]},
                        # "yaxis": {"title": "Lower Bound", "range": [minLowerBound, maxLowerBound]},
                        "yaxis": {"title": "Lower Bound"},
                        "height": 450, # px
                    }
                })
                if (len(lowerBoundArray)-1)<emMaxIterValue:
                    estimationStatusChildren = "In progress (pid={:s}) ...".format(pidChildren)
                else:
                    estimationStatusChildren = "Done"

            answer = (lowerBoundGraphFigure, estimationStatusChildren)
            return answer
        raise PreventUpdate

    @app.callback(Output('latentsGraph', 'figure'),
                  [Input('estimationProgressGraphsIntervalComponent', 'n_intervals')],
                  [State('sessionID', 'children'),
                   State('trialToPlotLatentsComponent', 'value'),
                   State('latentToPlotComponent', 'value'),
                   State('sampleRateForPlotting', 'value'),
                   State({"type": "trialsLengths",  "latent": ALL}, "value"),
                   State('latentsGraph', 'figure'),
                  ])
    def update_latentsGraph_live(estimationProgressGraphsIntervalComponent_n_intervals, sessionIDChildren, trialToPlotLatentComponentValue, latentToPlotValue, sampleRateForPlottingValue, trialsLengths, latentsGraphFigure):
        if estimationProgressGraphsIntervalComponent_n_intervals==0:
            raise PreventUpdate

        latentsLockFN = "/tmp/lockLatents{:s}.lock".format(sessionIDChildren[0])
        latentsStreamFN = "/tmp/bufferLatents{:s}.npz".format(sessionIDChildren[0])

        if os.path.exists(latentsStreamFN):
            latentsLock = utils.networking.multiprocessingUtils.FileLock(filename=latentsLockFN)
            if not latentsLock.is_locked():
                latentsLock.lock()
                with open(latentsStreamFN, 'rb') as f:
                    loadRes = np.load(f)
                    iteration = loadRes["iteration"]
                    times = loadRes["times"]
                    muK = loadRes["muK"]
                    varK = loadRes["varK"]
                latentsLock.unlock()
                latentsGraphFigure = dict({
                    "data": [],
                    "layout": {
                        "title": "Iteration {:d}, Trial {:d}, Latent {:d}".format(iteration, trialToPlotLatentComponentValue+1, latentToPlotValue+1),
                        "xaxis": {"title": "Time (sec)"},
                        "yaxis": {"title": "Latent"},
                        "height": 450, # px
                    }
                })
                times = np.arange(0, trialsLengths[trialToPlotLatentComponentValue-1],
                                  1.0/sampleRateForPlottingValue)
                times_rev = times[::-1]
                cTimes = np.concatenate((times,times_rev))
                meanToPlot = muK[trialToPlotLatentComponentValue-1,:,latentToPlotValue-1]
                varToPlot = varK[trialToPlotLatentComponentValue-1,:,latentToPlotValue-1]
                upperBound = meanToPlot+1.96*np.sqrt(varToPlot)
                lowerBound = meanToPlot-1.96*np.sqrt(varToPlot)
                lowerBound_rev = lowerBound[::-1]
                cBounds = np.concatenate((upperBound, lowerBound_rev))
                transparentFillcolor = DEFAULT_PLOTLY_COLORS[0].replace("rgb", "rgba").replace(")", ", 0.2)")
                lineColor = DEFAULT_PLOTLY_COLORS[0]
                latentsGraphFigure["data"].append(
                    {
                        "type": "scatter",
                        "x": cTimes,
                        "y": cBounds,
                        "fill": "tozerox",
                        "fillcolor": transparentFillcolor,
                        "line":dict(color='rgba(255,255,255,0)'),
                        "showlegend": False,
                    },
                )
                latentsGraphFigure["data"].append(
                    {
                        "type": "scatter",
                        "x": times,
                        "y": meanToPlot,
                        "line": {"color": lineColor},
                        "mode": "lines",
                        "showlegend": False,
                    },
                )
            return latentsGraphFigure
        raise PreventUpdate

    @app.callback(Output('logTextarea', 'value'),
                  [Input('logIntervalComponent', 'n_intervals')],
                  [State('logTextarea', 'value'),
                   State('sessionID', 'children'),
                  ])
    # def update_graph_live(intervalComponentn_intervals, nIterValue, sessionIDChildren, lowerBoundGraphFigure, latentsGraphFigure):
    def update_log_live(logIntervalComponent_n_intervals, logTextareaValue, sessionIDChildren):
        if logIntervalComponent_n_intervals==0:
            raise PreventUpdate

        logLockFN = "/tmp/lockLog{:s}.lock".format(sessionIDChildren[0])
        logStreamFN = "/tmp/bufferLog{:s}.log".format(sessionIDChildren[0])

        logLock = utils.networking.multiprocessingUtils.FileLock(filename=logLockFN)
        if os.path.exists(logStreamFN):

            if not logLock.is_locked():
                logLock.lock()
                with open(logStreamFN, 'r') as f:
                    newLogLines = f.readlines()
                # os.remove(logStreamFN)
                logLock.unlock()
                logTextareaValue = ""
                for logLine in newLogLines:
                    logTextareaValue = logTextareaValue + logLine

            return logTextareaValue
        raise PreventUpdate

    @app.callback(
        [Output("kernelsTypesContainer", "children"),
         Output("latentToPlotComponent", "options"),
         Output("latentToPlotComponent", "value"),
        ],
        [Input("nLatentsComponent", "value")],
        [State("kernelsTypesContainer", "children")])
    def populateKernelsTypes(nLatentsComponentValue, kernelsTypesContainerChildren):
        if nLatentsComponentValue is None:
            raise PreventUpdate
        latentToPlotComponentOptions = [{"label": str(r+1), "value": r} for r in range(nLatentsComponentValue)]
        latentToPlotComponentValue = 0
        if nLatentsComponentValue==len(kernelsTypesContainerChildren):
            raise PreventUpdate
        elif nLatentsComponentValue<len(kernelsTypesContainerChildren):
            kernelsTypesContainerChildren = kernelsTypesContainerChildren[:nLatentsComponentValue]
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
            kernelsTypesContainerChildren = kernelsTypesContainerChildren + newChildren
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
            kernelsTypesContainerChildren = kernelsTypesContainerChildren + newChildren
        return kernelsTypesContainerChildren, latentToPlotComponentOptions, latentToPlotComponentValue

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
                   Output("trialToPlotRasterComponent", "options"),
                   Output("trialToPlotRasterComponent", "value"),
                   Output("trialToPlotRasterParentContainer", "hidden"),
                   Output("trialToPlotLatentsComponent", "options"),
                   Output("trialToPlotLatentsComponent", "value"),
                   Output("trialsLengthsContainer", "children"),
                   Output("trialsLengthsParentContainer", "hidden")],
                  [Input("spikesTimesVar", "value")],
                  [State("uploadSpikes", "contents"),
                   State("uploadSpikes", "filename")])
    def propagateNewSpikesTimesInfo(spikesTimesVar, contents, filename):
        # pdb.set_trace()
        if contents is not None and spikesTimesVar is not None:
            spikesTimes = getSpikesTimes(contents=contents, filename=filename, spikesTimesVar=spikesTimesVar)
            nTrials = len(spikesTimes)
            nNeurons = len(spikesTimes[0])
            trialsLengthsGuesses = guessTrialsLengths(spikesTimes=spikesTimes)
            nTrialsAndNNeuronsInfo = "trials: {:d}, neurons: {:d}".format(nTrials, nNeurons)
            trialToPlotRasterOptions = [{"label": str(r+1), "value": r} for r in range(nTrials)]
            trialToPlotRasterValue = 0
            trialToPlotRasterParentDivHidden = False
            trialToPlotLatentsOptions = [{"label": str(r+1), "value": r} for r in range(nTrials)]
            trialToPlotLatentsValue = 0
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
            return nTrials, nNeurons, nTrialsAndNNeuronsInfo, trialToPlotRasterOptions, trialToPlotRasterValue, trialToPlotRasterParentDivHidden, trialToPlotLatentsOptions, trialToPlotLatentsValue, trialsLengthsChildren, trialsLengthsParentContainerHidden
        raise PreventUpdate

    @app.callback(Output("trialRastergram", "figure"),
                  [Input("trialToPlotRasterComponent", "value")],
                  [State("spikesTimesVar", "value"),
                   State("uploadSpikes", "contents"),
                   State("uploadSpikes", "filename")])
    def updateRastergram(trialToPlotRaster, spikesTimesVar, contents, filename):
        # pdb.set_trace()
        if trialToPlotRaster>=0 and spikesTimesVar is not None:
            spikesTimes = getSpikesTimes(contents=contents, filename=filename, spikesTimesVar=spikesTimesVar)
            trialSpikesTimes = spikesTimes[trialToPlotRaster]
            title="Trial {:d}".format(trialToPlotRaster+1)
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
            qVec0[0] = initVar
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
            nNeurons = len(spikesTimes[0])
            C0 = np.random.uniform(size=(nNeurons, nLatentsComponentValue))
            d0 = np.random.uniform(size=(nNeurons,))
            C0style = {"width": nLatentsComponentValue*175, "height": 300}
            d0style={"width": 200, "height": 300}
            # answer = [False, np.array2string(C0), C0style, np.array2string(d0), d0style]
            answer = [False, np.array_repr(C0), C0style, np.array_repr(d0), d0style]
            return answer
        raise PreventUpdate

    @app.callback(
        [Output("pid", "children"),
         Output("estimationProgressContainer", "style"),
         Output('logIntervalComponent', 'disabled'),
         Output('estimationProgressGraphsIntervalComponent', 'disabled'),
        ],
        [Input("doEstimate", "n_clicks")],
        [State("sessionID", "children"),
         State("spikesTimesVar", "value"),
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
         State('estimationProgressContainer', 'style'),
        ])
    def estimateSVGPFA(doEstimateNClicks,
                       sessionIDChildren,
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
                       #
                       estimationProgressContainerStyle,
                      ):
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

            logLockFN = "/tmp/lockLog{:s}.lock".format(sessionIDChildren[0])
            logStreamFN = "/tmp/bufferLog{:s}.log".format(sessionIDChildren[0])
            logLock = utils.networking.multiprocessingUtils.FileLock(filename=logLockFN)
            lowerBoundLockFN = "/tmp/lockLowerBound{:s}.lock".format(sessionIDChildren[0])
            lowerBoundStreamFN = "/tmp/bufferLowerBound{:s}.npy".format(sessionIDChildren[0])
            lowerBoundLock = utils.networking.multiprocessingUtils.FileLock(filename=lowerBoundLockFN)
            latentsLockFN = "/tmp/lockLatents{:s}.lock".format(sessionIDChildren[0])
            latentsTimes = torch.arange(0, np.max(trialsLengths), dtLatentsTimes)
            latentsLock = utils.networking.multiprocessingUtils.FileLock(filename=latentsLockFN)
            latentsStreamFN = "/tmp/bufferLatents{:s}.npz".format(sessionIDChildren[0])

            keywords = {"logLock": logLock, 
                        "logStreamFN": logStreamFN,
                        "lowerBoundLock": lowerBoundLock, 
                        "lowerBoundStreamFN": lowerBoundStreamFN, 
                        "latentsTimes": latentsTimes,
                        "latentsLock": latentsLock,
                        "latentsStreamFN": latentsStreamFN,
                       }
            # runner.run()
            p = mp.Process(target=runner.run, kwargs=keywords)
            p.start()

            estimationProgressContainerStyle["display"] = "block"
            logIntervalComponentDisabled = False
            estimationProgressGraphsIntervalComponentDisabled = False
            pidChildren = "{:d}".format(p.pid)
            return pidChildren, estimationProgressContainerStyle, logIntervalComponentDisabled, estimationProgressGraphsIntervalComponentDisabled
        raise PreventUpdate

    @app.callback(
        Output("cancelEstimationDummyDiv", "children"),
        [Input("cancelEstimation", "n_clicks")],
        [State("pid", "children")],
    )
    def cancelEstimation(cancelEstimation_n_clicks, pidChildren):
        if cancelEstimation_n_clicks>0:
            pid = int(pidChildren)
            os.kill(pid, signal.SIGKILL)
        raise PreventUpdate

    if args.non_local:
        app.run_server(debug=args.debug, host="0.0.0.0")
    else:
        app.run_server(debug=args.debug)

if __name__ == "__main__":
    main(sys.argv)

