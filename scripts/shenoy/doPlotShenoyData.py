
import sys
import argparse
import scipy.io

import shenoyUtils
sys.path.append("../../src")
import plot.svGPFA.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", help="data filename",
                        default="../../../../../datasets/george20040123_hnlds.mat")
    parser.add_argument("--trial_index",
                        help="trial to plot spikes for all neurons", type=int,
                        default=0)
    parser.add_argument("--neuron_index",
                        help="neuron to plot spikes for all trials", type=int,
                        default=0)
    parser.add_argument("--location", help="location", type=int, default=0)
    parser.add_argument("--trials_indices", help="trials indices to analyze",
                        default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--title_oneTrial_pattern", 
                        help="title for one trial pattern",
                        default="Trial {:d}, Location {:d}")
    parser.add_argument("--title_oneNeuron_pattern",
                        help="title for one neuron pattern",
                        default="Neuron {:d}, Location {:d}")
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
    parser.add_argument("--figOneTrial_filename_pattern",
                        help="figure spikes for one trial filename pattern",
                        default="figures/spikesTimes_trial{:d}_location{:d}.{:s}")
    parser.add_argument("--figOneNeuron_filename_pattern",
                        help="figure spikes for one neuron filename pattern",
                        default="figures/spikesTimes_neuron{:d}_location{:d}.{:s}")
    args = parser.parse_args()

    data_filename = args.data_filename
    trial_index = args.trial_index
    neuron_index = args.neuron_index
    location = args.location
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    title_oneTrial_pattern = args.title_oneTrial_pattern
    title_oneNeuron_pattern = args.title_oneNeuron_pattern
    target_onset_colour = args.target_onset_colour
    go_cue_colour = args.go_cue_colour
    move_onset_colour = args.move_onset_colour
    move_end_colour = args.move_end_colour
    vline_width = args.vline_width
    vline_style = args.vline_style
    xlabel = args.xlabel
    figOneTrial_filename_pattern = args.figOneTrial_filename_pattern
    figOneNeuron_filename_pattern = args.figOneNeuron_filename_pattern

    mat = scipy.io.loadmat(data_filename)
#     mat_spike_times = mat["Rb"][trial_index,location]['unit']['spikeTimes']
#     spikes_times = []
#     for j in range(mat_spike_times.shape[1]):
#         spikes_times.append(mat_spike_times[0,j].squeeze())

#     spikes_times = shenoyUtils.getTrialAndLocationSpikesTimes(mat=mat,
#                                                               trial_index=
#                                                                trial_index,
#                                                               location=location)
#     time_fix_target = mat["Rb"][trial_index,location]["timeFixTarget"].squeeze()
#     time_go_cue = mat["Rb"][trial_index,location]["timeGoCue"].squeeze()
#     time_move_onset = mat["Rb"][trial_index,location]["timeMoveOnset"].squeeze()
#     time_move_end = mat["Rb"][trial_index,location]["timeMoveEnd"].squeeze()
#     title_oneTrial = title_oneTrial_pattern.format(trial_index, location)
#     fig = plot.svGPFA.plotUtilsPlotly.getSpikesTimesPlotOneTrial(
#         spikes_times=spikes_times, title=title_oneTrial, xlabel=xlabel)
#     fig.add_vline(x=time_fix_target, line_width=vline_width, line_dash=vline_style, line_color=target_onset_colour)
#     fig.add_vline(x=time_go_cue, line_width=vline_width, line_dash=vline_style, line_color=go_cue_colour)
#     fig.add_vline(x=time_move_onset, line_width=vline_width, line_dash=vline_style, line_color=move_onset_colour)
#     fig.add_vline(x=time_move_end, line_width=vline_width, line_dash=vline_style, line_color=move_end_colour)
#     png_fig_filename = figOneTrial_filename_pattern.format(trial_index, location, "png")
#     fig.write_image(png_fig_filename)
#     html_fig_filename = figOneTrial_filename_pattern.format(trial_index, location, "html")
#     fig.write_html(html_fig_filename)

    spikes_times = shenoyUtils.getTrialsAndLocationSpikesTimes(mat=mat,
                                                               trials_indices=
                                                                trials_indices,
                                                               location=location)
    title_oneNeuron = title_oneNeuron_pattern.format(neuron_index, location)
    fig = plot.svGPFA.plotUtilsPlotly.getSpikesTimesPlotOneNeuron(
        spikes_times=spikes_times, neuron_index=neuron_index, title=title_oneNeuron, xlabel=xlabel)
    png_fig_filename = figOneNeuron_filename_pattern.format(neuron_index, location, "png")
    fig.write_image(png_fig_filename)
    html_fig_filename = figOneNeuron_filename_pattern.format(neuron_index, location, "html")
    fig.write_html(html_fig_filename)

    # fig.show()

    # import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
