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
                # style={"width": "40%"}
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
            #  style={"width": "40%"}
            ),
        ], style={"display": "inline-block", "background-color": "white"}),
    ], style={"display": "flex", "flex-wrap": "wrap", "width": 800, "padding-bottom": "20px", "background-color": "white"}),
#     html.Label("Embedding Type"),
#     dcc.RadioItems(
#         options=[
#             {"label": "Linear", "value": stats.svGPFA.svGPFAModelFactory.LinearEmbedding},
#         ],
#         value=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
#         style={"width": "40%"}
#     ),
    html.Div(children=[
        html.Label("Number of Latents"),
        html.Div(children=[
            dcc.Slider(
                id="nLatents",
                min=1,
                max=10,
                marks={i: str(i) for i in range(1, 11)},
                value=3,
            )],
            style={"width": "25%"}
        ),
    ]),
    html.Hr(),
    html.Div(id="kernelsInfo", children=[html.Div()]),
    html.Hr(),
    html.Div(children=[
        html.Div(children=[
            html.Label("Spikes Variable Name"),
            dcc.Input(id="spikesTimesVar", type="text", value="spikesTimes", required=True, size="15"),
        # ], style={"width": "200px", "display": "inline-block"}),
        ], style={"width": "200px"}),
        html.Div(children=[
            dcc.Upload(id="uploadSpikes", children=html.Button("Upload Spikes")),
            html.Div(id="spikesInfo"),
        # ], style={"width": "400px", "display": "inline-block"}),
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
#                 figure={
#                     'data': [
#                         {
#                             'x': [1, 2, 3, 4],
#                             'y': [4, 1, 3, 5],
#                             'text': ['a', 'b', 'c', 'd'],
#                             'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
#                             'name': 'Trace 1',
#                             'mode': 'markers',
#                             'marker': {'size': 12}
#                         },
#                         {
#                             'x': [1, 2, 3, 4],
#                             'y': [9, 4, 1, 4],
#                             'text': ['w', 'x', 'y', 'z'],
#                             'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
#                             'name': 'Trace 2',
#                             'mode': 'markers',
#                             'marker': {'size': 12}
#                         }
#                     ],
#                     'layout': {
#                         'clickmode': 'event+select'
#                     }
#                 }
            ),
            html.Hr(),
        ], hidden=True, style={"width": "20%", "columnCount": 1}),
    html.Div(id="trialsLengths", children=[html.Div()]),
    html.Div(id="nIndPointsPerLatent", children=[html.Div()]),
    html.Div(id="initialConditions", children=[html.Div()]),
])

@app.callback(
    Output("kernelsInfo", "children"),
    [Input("nLatents", "value")])
def populateKernels(nLatents):
    someChildren = []
    for k in range(nLatents):
        aDiv = html.Div(children=[
            html.Div(children=[
                html.Label("Kernel {:d} Type".format(k+1)),
                dcc.Dropdown(
                    options=[
                        {"label": "Exponential Quadratic", "value": "stats.kernels.ExponentialQuadraticKernel"},
                        {"label": "Periodic", "value": "stats.kernels.PeriodicKernel"},
                    ],
                    value="stats.kernels.ExponentialQuadraticKernel",
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
        someChildren.append(aDiv)
    return someChildren

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

'''
@app.callback([Output("spikesInfo", "children"),
               Output("trialToPlot", "options"),
               Output("trialToPlot", "value"),
               Output("trialToPlotDiv", "hidden")],
              [Input("uploadSpikes", "contents")],
              [State("uploadSpikes", "filename"),
               State("uploadSpikes", "last_modified")])
def loadSpikes(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(",")
        decoded = base64.b64decode(content_string)
        bytesIO = io.BytesIO(decoded)
        spikesTimes = np.load(bytesIO, allow_pickle=True)["spikesTimes"]
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
    # return [None, None, None, True]
    raise PreventUpdate

@app.callback(Output("spikesInfo", "children"),
              [Input("uploadSpikes", "contents")],
              [State("uploadSpikes", "filename"),
               State("uploadSpikes", "last_modified")])
def loadSpikes(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(",")
        decoded = base64.b64decode(content_string)
        bytesIO = io.BytesIO(decoded)
        spikesTimes = np.load(bytesIO, allow_pickle=True)["spikesTimes"]
        runner.setSpikesTimes(spikesTimes=spikesTimes)
        nTrials = spikesTimes.shape[0]
        spikesInfoChildren = [
            "Filename: {:s}".format(list_of_names)
        ]
        return spikesInfoChildren
    # raise IgnoreCallback("Ignoring callback because list_of_contents is None")
    # return [None]
    raise PreventUpdate
'''

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

@app.callback(Output("initialConditions", "children"),
              [Input({'type': 'nIndPointsForLatent', 'index': ALL}, 'value')])
def populatePosteriorOnIndPointsInitialConditions(values):
    someChildren = []
    aChildren = html.H4("Initial Conditions for Variational Posterior on Inducing Points")
    someChildren.append(aChildren)
    for k, nIndPoints in enumerate(values):
        qMu0 = np.zeros((nIndPoints, 1))
        qMu0[0] = 1.0
        aChildren = html.Div(children=[
            html.Label("Mean of Inducing for Latent {:d}".format(k+1)),
            dcc.Input(
                id={
                    "type": "svPosterioOnIndPointsMean",
                    "index": k
                },
                type="text",
                value=np.array2string(qMu0), required=True),
            html.Label("Covariance Spanning Vector for Latent {:d}".format(k+1)),
            html.Label("Covariance Diagonal Vector for Latent {:d}".format(k+1)),
        ])
        someChildren.append(aChildren)
    return someChildren

if __name__ == "__main__":
    app.run_server(debug=True)
