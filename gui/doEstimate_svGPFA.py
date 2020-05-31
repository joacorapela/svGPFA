import pdb
import sys
import io
import os
import base64
import datetime
import numpy as np
from scipy.io import loadmat
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
import stats.kernels

class svGPFA_runner:
    def setSpikesTimes(self, spikesTimes):
        self._spikesTimes = spikesTimes
    def getSpikesTimes(self):
        return self._spikesTimes

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

def main(argv):
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    runner = svGPFA_runner()

    app.layout = html.Div(children=[
        html.H1(children="Sparse Variational Gaussian Process Factor Analysis"),

        html.Hr(),

        html.Div(children=[
            html.Div(children=[
                html.Label("Conditional Distribution"),
                dcc.RadioItems(
                    options=[
                        {"label": "Point Process", "value": stats.svGPFA.svGPFAModelFactory.PointProcess},
                        {"label": "Poisson", "value": stats.svGPFA.svGPFAModelFactory.Poisson},
                        {"label": "Gaussian", "value": stats.svGPFA.svGPFAModelFactory.Gaussian},
                    ],
                    value=stats.svGPFA.svGPFAModelFactory.PointProcess,
                ),
            ], style={"display": "inline-block", "background-color": "white", "padding-right": "30px"}),
            html.Div(children=[
                html.Label("Link Function"),
                dcc.RadioItems(
                    options=[
                        {"label": "Exponential", "value": stats.svGPFA.svGPFAModelFactory.ExponentialLink},
                        {"label": "Other", "value": stats.svGPFA.svGPFAModelFactory.NonExponentialLink},
                    ],
                    value=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
                ),
            ], style={"display": "inline-block", "background-color": "white"}),
        ], style={"display": "flex", "flex-wrap": "wrap", "width": 800, "padding-bottom": "20px", "background-color": "white"}),
        html.Div(children=[
            html.Label("Number of Latents"),
            html.Div(children=[
                dcc.Slider(
                    id="nLatents",
                    min=0,
                    max=10,
                    marks={i: str(i) for i in range(1, 11)},
                    value=3,
                )],
                style={"width": "25%"}
            ),
        ]),
        html.Hr(),
        html.Div(id="kernelsInfo", children=[]),
        html.Hr(),
        html.Div(children=[
            html.Div(children=[
                html.Label("Spikes Variable Name"),
                dcc.Input(id="spikesTimesVar", type="text", value="Y", required=True, size="15"),
            ], style={"width": "200px"}),
            html.Div(children=[
                dcc.Upload(id="uploadSpikes", children=html.Button("Upload Spikes")),
                html.Div(id="spikesInfo"),
            ], style={"width": "400px"}),
        # ], style={"display": "flex", "flex-wrap": "wrap", "align-items": "flex-start", "width": 800, "background-color": "white"}),
        ], style={"width": 800, "background-color": "white"}),
        html.Hr(),
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
        html.Div(id="trialsLengths", children=[html.Div()]),
        html.Div(id="nIndPointsPerLatent", children=[html.Div()]),
        html.Div(id="svPosterioOnIndPointsInitialConditions", children=[html.Div()]),
        html.Div(id="embeddingInitialConditions", children=[html.Div()]),
        html.Hr(),
        html.Div(
            id="optimParams",
            children=[
                html.H4("Optimization Parameters"),
                html.Div(children=[
                    html.Label("Maximum EM iterations"),
                    dcc.Input(id="maxIter", type="number", value="20", required=True),
                ], style={"padding-bottom": "20px"}),
                html.Div(children=[
                    html.Label("Regularization added to kernel covariance matrix for inducing points"),
                    dcc.Input(id="indPointsLocsKMSRegEpsilon", type="number", value="1e-2", required=True),
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
        html.Button(id="estimate", n_clicks=0, children="Estimate"),
    ])

    @app.callback(
        Output("estimate", "n_clicks"),
        [Input("estimate", "n_clicks")],
        [State("kernelsInfo", "children")])
    def estimateSVGPFA(estimateNClicks, kernelsInfoChildren):

    @app.callback(
        Output("kernelsInfo", "children"),
        [Input("nLatents", "value")],
        [State("kernelsInfo", "children")])
    def populateKernels(nLatents, kernelsInfoChildren):
        if nLatents==0:
            raise PreventUpdate
        if nLatents==len(kernelsInfoChildren):
            raise PreventUpdate
        elif nLatents<len(kernelsInfoChildren):
            return kernelsInfoChildren[:nLatents]
        else:
            newChildren = []
            for k in range(len(kernelsInfoChildren), nLatents):
                aDiv = html.Div(children=[
                    html.Div(children=[
                        html.Label("Kernel {:d} Type".format(k+1)),
                        dcc.Dropdown(
                            id="kT
                            options=[
                                {"label": "Exponential Quadratic", "value": "stats.kernels.ExponentialQuadraticKernel"},
                                {"label": "Periodic", "value": "stats.kernels.PeriodicKernel"},
                            ],
                            value="stats.kernels.PeriodicKernel",
                            style={"width": "80%"}
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Length Scale"),
                        dcc.Input(
                            type="number", placeholder="length scale",
                            min=0,
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Period"),
                        dcc.Input(
                            type="number", placeholder="period",
                            min=0,
                        ),
                    ]),
                ], style={"columnCount": 3})
                newChildren.append(aDiv)
            answer = kernelsInfoChildren + newChildren
        return answer

    @app.callback([Output("spikesInfo", "children"),
                   Output("trialToPlot", "options"),
                   Output("trialToPlot", "value"),
                   Output("trialToPlotDiv", "hidden")],
                  [Input("uploadSpikes", "contents")],
                  [State("uploadSpikes", "filename"),
                   State("uploadSpikes", "last_modified"),
                   State("trialToPlot", "options"),
                   State("spikesTimesVar", "value")])
    def loadSpikes(list_of_contents, list_of_names, list_of_dates, oldTrialToPlotOptions, spikesTimesVar):
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
        # return [None, oldTrialToPlotOptions, -1, True]
        raise PreventUpdate

    @app.callback(Output("trialRastergram", "figure"),
                  [Input("trialToPlot", "value")],
                  [State("trialRastergram", "figure")])
    def updateRastergram(trialToPlot, oldRastergram):
        if trialToPlot>=0:
            spikesTimes = runner.getSpikesTimes()
            trialSpikesTimes = spikesTimes[trialToPlot,:]
            title='Trial {:d}'.format(trialToPlot+1)
            rastergram = getRastergram(trialSpikesTimes=trialSpikesTimes, title=title)
            return rastergram
        # return oldRastergram
        raise PreventUpdate

    @app.callback(Output("trialsLengths", "children"),
                  [Input("spikesInfo", "children")])
    def showTrialLengths(spikesInfoChildren):
        if spikesInfoChildren is not None:
            nTrials = runner.getSpikesTimes().shape[0]
            someChildren = []
            for r in range(nTrials):
                aDiv = html.Div(children=[
                    html.Label("Trial {:d} Length".format(r+1)),
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
                  [Input("nLatents", "value")])
    def showNIndPointsPerLatent(aNLatents):
        someChildren = []
        aChildren = html.H4("Number of Inducing Points")
        someChildren.append(aChildren)
        for k in range(aNLatents):
            aChildren = html.Div(children=[
                html.Label("Latent {:d}".format(k+1)),
                html.Div(
                    children=[
                    dcc.Slider(
                        id={
                            "type": "nIndPointsForLatent",
                            "index": k
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
                  [Input({'type': 'nIndPointsForLatent', 'index': ALL}, 'value')])
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
                html.H6("Latent {:d}".format(k)),
                html.Label("Mean"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsMu0",
                        "index": k
                    },
                    type="text",
                    value=np.array2string(qMu0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
                html.Label("Spanning Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsVec0",
                        "index": k
                    },
                    type="text",
                    value=np.array2string(qVec0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
                html.Label("Diagonal Vector of Covariance"),
                dcc.Input(
                    id={
                        "type": "svPosterioOnIndPointsDiag0",
                        "index": k
                    },
                    type="text",
                    value=np.array2string(qDiag0.squeeze()),
                    size="{:d}".format(5*nIndPoints),
                    required=True),
            ], style={"padding-bottom": "30px"})
            someChildren.append(aChildren)
        return someChildren

    @app.callback(Output("embeddingInitialConditions", "children"),
                  [Input("spikesInfo", "children")],
                  [State("nLatents", "value")])
    def populateEmbeddingInitialConditions(spikesInfoChildren, nLatents):
        nNeurons = runner.getSpikesTimes().shape[1]
        C0 = np.random.uniform(size=(nNeurons, nLatents))
        d0 = np.random.uniform(size=(nNeurons, 1))
        someChildren = []
        aChildren = html.H4("Initial Conditions for Linear Embedding")
        someChildren.append(aChildren)
        aDiv = html.Div(children=[
            html.Label("Mixing Matrix"),
            dcc.Textarea(
                id="C0",
                value=np.array2string(C0),
                style={'width': nLatents*150, 'height': 300}),
        ])
        someChildren.append(aDiv)
        aDiv = html.Div(children=[
            html.Label("Offset Vector"),
            dcc.Textarea(
                id="d0",
                value=np.array2string(d0),
                style={'width': 150, 'height': 300}),
        ])
        someChildren.append(aDiv)
        return someChildren

    app.run_server(debug=True)

if __name__ == "__main__":
    main(sys.argv)

