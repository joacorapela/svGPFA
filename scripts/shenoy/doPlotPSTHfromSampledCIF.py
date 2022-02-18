
import sys
import argparse
import pickle
import scipy.io
import numpy as np
import plotly.graph_objs as go

import shenoyUtils
sys.path.append("../../src")
import stats.neuralDataUtils
import plot.svGPFA.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--spikesFilenamePattern",
                        help="filename pattern for spikes sampled from the sampled CIF",
                        default="results/{:08d}_spikesFromSampledCIF.pickle")
    parser.add_argument("--data_filename", help="data filename",
                        default="/nfs/ghome/live/rapela/dev/work/ucl/gatsby-swc/datasets/george20040123_hnlds.mat")
    parser.add_argument("--neuron_index",
                        help="neuron to plot spikes for all trials", type=int,
                        default=0)
    parser.add_argument("--trials_indices", help="trials indices to analyze",
                        default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        default="timeMoveOnset")
    parser.add_argument("--location", help="location", type=int, default=0)
    parser.add_argument("--psth_bins_start_time", 
                        help="PSTH bins start time (msec)",
                        type=int, default=-2000)
    parser.add_argument("--psth_bins_end_time", 
                        help="PSTH bins end time (msec)",
                        type=int, default=500)
    parser.add_argument("--psth_bin_size", help="PSTH bin size (msec)",
                        type=int, default=50)
    parser.add_argument("--title_pattern", help="title pattern",
                        default="Neuron {:d}")
    parser.add_argument("--target_onset_colour", help="target onset colour",
                        default="violet")
    parser.add_argument("--go_cue_colour", help="go cue colour", default="red")
    parser.add_argument("--move_onset_colour", help="movement onset colour",
                        default="blue")
    parser.add_argument("--move_end_colour", help="movement end colour",
                        default="green")
    parser.add_argument("--vline_width", help="vertical line width", type=int,
                        default=3)
    parser.add_argument("--vline_style", help="vertical line style",
                        default="solid")
    parser.add_argument("--xlabel", help="x-axis label", default="Time (msec)")
    parser.add_argument("--psth_ylabel", help="y-axis label for PSTH",
                        default="mean bin spike rate")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="figures/{:08d}_PSTHfromSampledCIF_neuron{:d}_epoch{:s}.{:s}")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    spikesFilenamePattern = args.spikesFilenamePattern
    data_filename = args.data_filename
    neuron_index = args.neuron_index
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    epoch_event_name = args.epoch_event_name 
    location = args.location
    psth_bins_start_time = args.psth_bins_start_time
    psth_bins_end_time = args.psth_bins_end_time
    psth_bin_size = args.psth_bin_size
    title_pattern = args.title_pattern
    target_onset_colour = args.target_onset_colour
    go_cue_colour = args.go_cue_colour
    move_onset_colour = args.move_onset_colour
    move_end_colour = args.move_end_colour
    vline_width = args.vline_width
    vline_style = args.vline_style
    xlabel = args.xlabel
    psth_ylabel = args.psth_ylabel
    fig_filename_pattern = args.fig_filename_pattern

    mat = scipy.io.loadmat(data_filename)
    epoch_times = [mat["Rb"][trial,location][epoch_event_name].squeeze() for trial in trials_indices]
    spikesFilename = spikesFilenamePattern.format(estResNumber)
    with open(spikesFilename, "rb") as f: loadRes = pickle.load(f)
    spikes_times = loadRes["spikesTimes"]

    spikes_times_numpy = [[spikes_times[r][n].numpy() for n in range(len(spikes_times[r]))] for r in range(len(spikes_times))]
    psth_bin_edges = np.arange(psth_bins_start_time, psth_bins_end_time+psth_bin_size, psth_bin_size)
    psths, psth_mean = stats.neuralDataUtils.computePSTHsAndMeans(spikes_times=spikes_times_numpy, neuron_index=neuron_index, trials_indices=trials_indices, epoch_times=epoch_times, bin_edges=psth_bin_edges)

    title = title_pattern.format(neuron_index)
    psth_bin_centers = np.array([(psth_bin_edges[i]+psth_bin_edges[i+1])/2 \
                                 for i in range(len(psth_bin_edges)-1)])
    fig = plot.svGPFA.plotUtilsPlotly.getPlotMean(
        x=psth_bin_centers, mean=psth_mean,
        xlabel=xlabel, ylabel=psth_ylabel
    )
    for i in range(psths.shape[0]):
        trace = go.Scatter(
            x=psth_bin_centers,
            y=psths[i,:],
            showlegend=True,
            mode="lines",
            name="neuron {:d}".format(i),
            # hoverinfo="skip",
        )
        fig.add_trace(trace)
    png_fig_filename = fig_filename_pattern.format(estResNumber, neuron_index, epoch_event_name, "png")
    fig.write_image(png_fig_filename)
    html_fig_filename = fig_filename_pattern.format(estResNumber, neuron_index, epoch_event_name, "html")
    fig.write_html(html_fig_filename)

    # fig.show()

    # import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
