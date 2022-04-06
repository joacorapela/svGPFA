
import sys
import argparse
import scipy.io
import numpy as np
import plotly.graph_objs as go

import shenoyUtils
sys.path.append("../../src")
import stats.neuralDataUtils
import plot.svGPFA.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", help="data filename",
                        default="../../../../../datasets/george20040123_hnlds.mat")
    parser.add_argument("--neuron_indices",
                        help="indices of neurons to plot spikes for all trials",
                        type=str, default="")
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
    parser.add_argument("--psth_nResamples", help="PSTH number of resamples",
                        type=int, default=500)
    parser.add_argument("--psth_ci_alpha", 
                        help="PSTH confidence interval alpha",
                        type=float, default=.05)
    parser.add_argument("--title_pattern",
                        help="title pattern",
                        default="Neuron {:d}, Location {:d}")
    parser.add_argument("--target_onset_colour", help="target onset colour",
                        default="violet")
    parser.add_argument("--go_cue_colour", help="go cue colour", default="red")
    parser.add_argument("--move_onset_colour", help="movement onset colour",
                        default="blue")
    parser.add_argument("--move_end_colour", help="movement end colour",
                        default="green")
    parser.add_argument("--mean_trace_line_width",
                        help="line width for the mean trace",
                        type=int, default=7)
    parser.add_argument("--psth_line_opacity",
                        help="opacity for psth line",
                        type=float, default=0.3)
    parser.add_argument("--vline_width", help="vertical line width", type=int,
                        default=3)
    parser.add_argument("--vline_style", help="vertical line style",
                        default="solid")
    parser.add_argument("--xlabel", help="x-axis label", default="Time (msec)")
    parser.add_argument("--psth_ylabel", help="y-axis label for PSTH",
                        default="mean bin spike rate")
    parser.add_argument("--fig_filename_pattern",
                        help="figure spikes for one neuron filename pattern",
                        default="figures/PSTH_neuron{:d}_location{:d}_epoch{:s}.{:s}")
    args = parser.parse_args()

    data_filename = args.data_filename
    if len(args.neuron_indices)>0:
        neuron_indices = [int(str) for str in args.neuron_indices[1:-1].split(",")]
    else:
        neuron_indices = None
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    epoch_event_name = args.epoch_event_name 
    location = args.location
    psth_bins_start_time = args.psth_bins_start_time
    psth_bins_end_time = args.psth_bins_end_time
    psth_bin_size = args.psth_bin_size
    psth_nResamples = args.psth_nResamples
    psth_ci_alpha = args.psth_ci_alpha
    title_pattern = args.title_pattern
    target_onset_colour = args.target_onset_colour
    go_cue_colour = args.go_cue_colour
    move_onset_colour = args.move_onset_colour
    move_end_colour = args.move_end_colour
    mean_trace_line_width = args.mean_trace_line_width
    psth_line_opacity = args.psth_line_opacity
    vline_width = args.vline_width
    vline_style = args.vline_style
    xlabel = args.xlabel
    psth_ylabel = args.psth_ylabel
    fig_filename_pattern = args.fig_filename_pattern

    mat = scipy.io.loadmat(data_filename)
    epoch_times = [mat["Rb"][trial,location][epoch_event_name].squeeze() for trial in trials_indices]

    spikes_times = shenoyUtils.getTrialsAndLocationSpikesTimes(mat=mat,
                                                               trials_indices=
                                                                trials_indices,
                                                               location=location)
    spikes_times_numpy = [[spikes_times[r][n].numpy() for n in range(len(spikes_times[r]))] for r in range(len(spikes_times))]
    psth_bin_edges = np.arange(psth_bins_start_time, psth_bins_end_time+psth_bin_size, psth_bin_size)
    if neuron_indices is None:
        neuron_indices = np.arange(0, len(spikes_times[0]))
    for neuron_index in neuron_indices:
        print("Processing neuron {:d}".format(neuron_index))
        psths, psth_mean = stats.neuralDataUtils.computePSTHsAndMeans(spikes_times=spikes_times_numpy, neuron_index=neuron_index, trials_indices=trials_indices, epoch_times=epoch_times, bin_edges=psth_bin_edges)

        title = title_pattern.format(neuron_index, location)
        psth_bin_centers = np.array([(psth_bin_edges[i]+psth_bin_edges[i+1])/2 \
                                    for i in range(len(psth_bin_edges)-1)])
        fig = plot.svGPFA.plotUtilsPlotly.getPlotMean(
            x=psth_bin_centers, mean=psth_mean,
            mean_width=mean_trace_line_width,
            xlabel=xlabel, ylabel=psth_ylabel
        )
        for i in range(psths.shape[0]):
            trace = go.Scatter(
                x=psth_bin_centers,
                y=psths[i,:],
                opacity=psth_line_opacity,
                showlegend=True,
                mode="lines",
                name="neuron {:d}".format(i),
                # hoverinfo="skip",
            )
            fig.add_trace(trace)
        png_fig_filename = fig_filename_pattern.format(neuron_index, location, epoch_event_name, "png")
        fig.write_image(png_fig_filename)
        html_fig_filename = fig_filename_pattern.format(neuron_index, location, epoch_event_name, "html")
        fig.write_html(html_fig_filename)

if __name__=="__main__":
    main(sys.argv)
