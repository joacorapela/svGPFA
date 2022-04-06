
import sys
import argparse
import pickle
import configparser
import numpy as np
import plotly.graph_objs as go

sys.path.append("../src")
import stats.neuralDataUtils
import plot.svGPFA.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", help="simuluation result number",
                        type=int)
    parser.add_argument("--neuron_indices",
                        help="indices of neurons to plot spikes for all trials",
                        type=str, default="")
    parser.add_argument("--trials_indices", help="trials indices to analyze",
                        default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        default="timeMoveOnset")
    parser.add_argument("--psth_bins_start_time",
                        help="PSTH bins start time (sec)",
                        type=float, default=0.0)
    parser.add_argument("--psth_bins_end_time",
                        help="PSTH bins end time (sec)",
                        type=float, default=0.99)
    parser.add_argument("--psth_bin_size", help="PSTH bin size (sec)",
                        type=float, default=0.01)
    parser.add_argument("--title_pattern",
                        help="title pattern",
                        default="Neuron {:d}")
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
    parser.add_argument("--xlabel", help="x-axis label", default="Time (sec)")
    parser.add_argument("--psth_ylabel", help="y-axis label for PSTH",
                        default="mean bin spike rate")
    parser.add_argument("--simres_filename_pattern",
                        help="simulation result filename patter",
                        default="results/{:08d}_simulation_metaData.ini")
    parser.add_argument("--fig_filename_pattern",
                        help="figure spikes for one neuron filename pattern",
                        default="figures/{:08d}_PSTH_neuron{:d}_epoch{:s}.{:s}")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    simRes_filename_pattern = args.simres_filename_pattern

    if len(args.neuron_indices) > 0:
        neuron_indices = \
            [int(str) for str in args.neuron_indices[1:-1].split(",")]
    else:
        neuron_indices = None
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    epoch_event_name = args.epoch_event_name
    psth_bins_start_time = args.psth_bins_start_time
    psth_bins_end_time = args.psth_bins_end_time
    psth_bin_size = args.psth_bin_size
    title_pattern = args.title_pattern
    mean_trace_line_width = args.mean_trace_line_width
    psth_line_opacity = args.psth_line_opacity
    xlabel = args.xlabel
    psth_ylabel = args.psth_ylabel
    fig_filename_pattern = args.fig_filename_pattern

    simRes_config_filename = simRes_filename_pattern.format(simRes_number)
    simRes_config = configparser.ConfigParser()
    simRes_config.read(simRes_config_filename)
    simRes_filename = simRes_config["simulation_results"]["simResFilename"]
    with open(simRes_filename, "rb") as f:
        simRes = pickle.load(f)
    spikes_times = simRes["spikes"]
    nTrials = len(spikes_times)
    epoch_times = [0.0 for r in range(nTrials)]

    psth_bin_edges = np.arange(psth_bins_start_time,
                               psth_bins_end_time+psth_bin_size,
                               psth_bin_size)
    if neuron_indices is None:
        neuron_indices = np.arange(0, len(spikes_times[0]))
    for neuron_index in neuron_indices:
        print("Processing neuron {:d}".format(neuron_index))
        binned_spikes, psth = stats.neuralDataUtils.computeBinnedSpikesAndPSTH(
            spikes_times=spikes_times, neuron_index=neuron_index,
            trials_indices=trials_indices, epoch_times=epoch_times,
            bin_edges=psth_bin_edges, time_unit="sec")

        title = title_pattern.format(neuron_index)
        psth_bin_centers = np.array([(psth_bin_edges[i]+psth_bin_edges[i+1])/2
                                     for i in range(len(psth_bin_edges)-1)])
        fig = plot.svGPFA.plotUtilsPlotly.getPlotMean(
            x=psth_bin_centers, mean=psth,
            mean_width=mean_trace_line_width,
            xlabel=xlabel, ylabel=psth_ylabel
        )
        nTrials = binned_spikes.shape[0]
        for i in range(nTrials):
            trace = go.Scatter(
                x=psth_bin_centers,
                y=binned_spikes[i, :],
                opacity=psth_line_opacity,
                showlegend=True,
                mode="lines",
                name="trial {:d}".format(i),
                # hoverinfo="skip",
            )
            fig.add_trace(trace)
        fig.update_layout(title=title)
        png_fig_filename = fig_filename_pattern.format(simRes_number,
                                                       neuron_index,
                                                       epoch_event_name,
                                                       "png")
        fig.write_image(png_fig_filename)
        html_fig_filename = fig_filename_pattern.format(simRes_number,
                                                        neuron_index,
                                                        epoch_event_name,
                                                        "html")
        fig.write_html(html_fig_filename)


if __name__ == "__main__":
    main(sys.argv)
