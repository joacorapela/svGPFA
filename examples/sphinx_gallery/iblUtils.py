
import numpy as np
import plotly.graph_objs as go

def epoch_neuron_spikes_times(neuron_spikes_times, epoch_times,
                              epoch_start_times, epoch_end_times):
    epoch_start_indices = np.searchsorted(neuron_spikes_times, epoch_start_times)
    epoch_end_indices = np.searchsorted(neuron_spikes_times, epoch_end_times)
    epoch_end_indices[epoch_end_indices==len(neuron_spikes_times)] = len(neuron_spikes_times)-1
    n_trials = len(epoch_start_indices)
    epoched_spikes_times = \
        [(neuron_spikes_times[epoch_start_indices[r]:epoch_end_indices[r]]-epoch_times[r]).tolist()
         for r in range(n_trials)]
    return epoched_spikes_times


def getSpikesTimesPlotOneNeuron(spikes_times,
                                sorting_times,
                                neuron_index, title,
                                trials_ids,
                                feedback_types,
                                behavioral_times_col=[],
                                behavioral_times_labels=[],
                                marked_events_times=None,
                                marked_events_colors=None,
                                marked_events_markers=None,
                                align_event=None,
                                marked_size=10, spikes_symbol="line-ns-open",
                                trials_colors=None, default_trial_color="black",
                                xlabel="Time (sec)", ylabel="Trial",
                                event_line_color="rgba(0, 0, 255, 0.2)",
                                event_line_width=5, spikes_marker_size=9):
    if sorting_times is not None:
        argsort = np.argsort(sorting_times)
        spikes_times = [spikes_times[r] for r in argsort]
        sorting_times = [sorting_times[r] for r in argsort]
        trials_ids = [trials_ids[r] for r in argsort]
        feedback_types = [feedback_types[r] for r in argsort]
        for i, behavioral_times in enumerate(behavioral_times_col):
            sorted_behavioral_times = [behavioral_times[r] for r in argsort]
            behavioral_times_col[i] = sorted_behavioral_times
    n_trials = len(trials_ids)
    fig = go.Figure()
    for r in range(n_trials):
        spikes_times_trial_neuron = spikes_times[r][neuron_index]
        # workaround because if a trial contains only one spike spikes_times[n]
        # does not respond to the len function
        if len(spikes_times_trial_neuron) == 1:
            spikes_times_trial_neuron = [spikes_times_trial_neuron]
        if trials_colors is not None:
            spikes_color = trials_colors[r]
        else:
            spikes_color = default_trial_color
        trial_label = "{:02d}".format(trials_ids[r])
        feedback_type = "{:d}".format(int(feedback_types[r]))
        trace = go.Scatter(
            x=spikes_times_trial_neuron,
            y=r*np.ones(len(spikes_times_trial_neuron)),
            mode="markers",
            marker=dict(size=spikes_marker_size, color=spikes_color,
                        symbol=spikes_symbol),
            name="trial {:s}".format(trial_label),
            legendgroup=f"trial{trial_label}",
            showlegend=False,
            text=[f"Trial {trial_label}<br>Feedback {feedback_type}"]*len(spikes_times_trial_neuron),
            hovertemplate="Time %{x}<br>%{text}",
        )
        fig.add_trace(trace)
        if marked_events_times is not None:
            marked_events_times_centered = marked_events_times[r]-align_event[r]
            n_marked_events = len(marked_events_times[r])
            for i in range(n_marked_events):
                trace_marker = go.Scatter(x=[marked_events_times_centered[i]],
                                          y=[r],
                                          marker=dict(color=marked_events_colors[r][i],
                                                      symbol=marked_events_markers[r][i],
                                                      size=marked_size),
                                          name="trial {:s}".format(trial_label),
                                          text=[trial_label],
                                          hovertemplate="Time %{x}<br>" + "Trial %{text}",
                                          mode="markers",
                                          legendgroup=f"trial{trial_label}",
                                          showlegend=False)
                fig.add_trace(trace_marker)
    for i, behavioral_times in enumerate(behavioral_times_col):
        trace = go.Scatter(x=behavioral_times, y=np.arange(n_trials),
                           name=behavioral_times_labels[i])
        fig.add_trace(trace)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=title)
    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )
    return fig

def subset_trials_ids_data(selected_trials_ids, trials_ids, spikes_times,
                           trials_start_times, trials_end_times):
    indices = np.nonzero(np.in1d(trials_ids, selected_trials_ids))[0]
    spikes_times_subset = [spikes_times[i] for i in indices]
    trials_start_times_subset = [trials_start_times[i] for i in indices]
    trials_end_times_subset = [trials_end_times[i] for i in indices]

    return spikes_times_subset, trials_start_times_subset, trials_end_times_subset

def subset_clusters_ids_data(selected_clusters_ids, clusters_ids,
                             spikes_times):
    indices = np.nonzero(np.in1d(clusters_ids, selected_clusters_ids))[0]
    n_trials = len(spikes_times)
    spikes_times_subset = [[spikes_times[r][i] for i in indices]
                           for r in range(n_trials)]

    return spikes_times_subset

def subset_info_dict(info, ids):
    subset_info = dict()
    for info_key in info.keys():
        if info[info_key].ndim == 1:
            subset_info[info_key] = [info[info_key][i] for i in ids]
        else:
            subset_info[info_key] = info[info_key]
    return subset_info


def buildMarkedEventsInfo(events_times, events_colors, events_markers):
    n_events = len(events_times)
    n_trials = len(events_times[0])
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for r in range(n_trials):
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(n_events):
            trial_marked_events_times.append(events_times[i][r])
            trial_marked_events_colors.append(events_colors[i])
            trial_marked_events_markers.append(events_markers[i])
        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers
