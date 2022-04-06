
import sys
import argparse
import pickle
import scipy.io

import shenoyUtils
sys.path.append("../../src")
import plot.svGPFA.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--spikesFilenamePattern",
                        help="filename pattern for spikes sampled from the expected posterior CIF",
                        default="results/{:08d}_spikesFromExpectedPosteriorCIF.pickle")
    parser.add_argument("--data_filename", help="data filename", default="/nfs/ghome/live/rapela/dev/research/gatsby-swc/datasets/george20040123_hnlds.mat")
    parser.add_argument("--trial_index", help="trial index", type=int, default=0)
    parser.add_argument("--location", help="location", type=int, default=0)
    parser.add_argument("--title_pattern", help="title pattern", default="Trial {:d}")
    parser.add_argument("--target_onset_colour", help="target onset colour", default="violet")
    parser.add_argument("--go_cue_colour", help="go cue colour", default="red")
    parser.add_argument("--move_onset_colour", help="movement onset colour", default="blue")
    parser.add_argument("--move_end_colour", help="movement end colour", default="green")
    parser.add_argument("--vline_width", help="vertical line width", type=int, default=3)
    parser.add_argument("--vline_style", help="vertical line style", default="solid")
    parser.add_argument("--xlabel", help="x-axis label", default="Time (msec)")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="figures/{:08d}_spikesTimesFromExpectedPosteriorCIF_trial{:d}.{:s}")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    spikesFilenamePattern = args.spikesFilenamePattern
    data_filename = args.data_filename
    trial_index = args.trial_index
    location = args.location
    title_pattern = args.title_pattern
    target_onset_colour = args.target_onset_colour
    go_cue_colour = args.go_cue_colour
    move_onset_colour = args.move_onset_colour
    move_end_colour = args.move_end_colour
    vline_width = args.vline_width
    vline_style = args.vline_style
    xlabel = args.xlabel
    fig_filename_pattern = args.fig_filename_pattern

    mat = scipy.io.loadmat(data_filename)
#     mat_spike_times = mat["Rb"][trial,location]['unit']['spikeTimes']
#     spikes_times = []
#     for j in range(mat_spike_times.shape[1]):
#         spikes_times.append(mat_spike_times[0,j].squeeze())
#     spikes_times = shenoyUtils.getTrialAndLocationSpikesTimes(mat=mat,
#                                                               trial=trial,
#                                                               location=location)
    spikesFilename = spikesFilenamePattern.format(estResNumber)
    with open(spikesFilename, "rb") as f: loadRes = pickle.load(f)
    spikes_times = loadRes["spikesTimes"][trial_index]
    time_fix_target = mat["Rb"][trial_index,location]["timeFixTarget"].squeeze()
    time_go_cue = mat["Rb"][trial_index,location]["timeGoCue"].squeeze()
    time_move_onset = mat["Rb"][trial_index,location]["timeMoveOnset"].squeeze()
    time_move_end = mat["Rb"][trial_index,location]["timeMoveEnd"].squeeze()
    title = title_pattern.format(trial_index, location)
    fig = plot.svGPFA.plotUtilsPlotly.getSpikesTimesPlotOneTrial(
        spikes_times=spikes_times, title=title, xlabel=xlabel)
    fig.add_vline(x=time_fix_target, line_width=vline_width, line_dash=vline_style, line_color=target_onset_colour)
    fig.add_vline(x=time_go_cue, line_width=vline_width, line_dash=vline_style, line_color=go_cue_colour)
    fig.add_vline(x=time_move_onset, line_width=vline_width, line_dash=vline_style, line_color=move_onset_colour)
    fig.add_vline(x=time_move_end, line_width=vline_width, line_dash=vline_style, line_color=move_end_colour)
    png_fig_filename = fig_filename_pattern.format(estResNumber, trial_index, "png")
    fig.write_image(png_fig_filename)
    html_fig_filename = fig_filename_pattern.format(estResNumber, trial_index, "html")
    fig.write_html(html_fig_filename)

    # fig.show()

    # import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
