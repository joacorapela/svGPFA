
import numpy as np
import stats.bootstrapTests


def binSpikes(spikes_times, bin_edges, time_unit):
    # spike_times in msec
    bin_width = bin_edges[1]-bin_edges[0]
    binned_spikes, _ = np.histogram(a=spikes_times, bins=bin_edges)
    binned_spikes = binned_spikes.astype(float)
    if time_unit == "sec":
        binned_spikes *= 1.0/bin_width
    elif time_unit == "msec":
        binned_spikes *= 1000.0/bin_width
    else:
        raise ValueError("time_unit should be sec or msec, but not {}".format(time_unit))
    return binned_spikes


def binMultiTrialSpikes(spikes_times, neuron_index, trials_indices,
                        epoch_times, bin_edges, time_unit):
    mt_binned_spikes = np.empty((len(trials_indices), len(bin_edges)-1),
                                dtype=np.double)
    for i, trial_index in enumerate(trials_indices):
        aligned_spikes_times = spikes_times[trial_index][neuron_index] - \
                                epoch_times[i]
        binned_spikes = binSpikes(
            spikes_times=aligned_spikes_times,
            bin_edges=bin_edges, time_unit=time_unit)
        mt_binned_spikes[i, :] = binned_spikes
    return mt_binned_spikes


def computeBinnedSpikesAndPSTH(spikes_times, neuron_index, trials_indices,
                               epoch_times, bin_edges, time_unit):
    binned_spikes = binMultiTrialSpikes(spikes_times=spikes_times,
                                        neuron_index=neuron_index,
                                        trials_indices=trials_indices,
                                        epoch_times=epoch_times,
                                        bin_edges=bin_edges,
                                        time_unit=time_unit)
    psth = np.empty(len(bin_edges)-1, dtype=np.double)
    for j in range(len(bin_edges)-1):
        psth[j] = binned_spikes[:, j].mean()
    return binned_spikes, psth


def computeBinnedSpikesAndPSTHwithCI(spikes_times, neuron_index,
                                     trials_indices, epoch_times,
                                     bin_edges, time_unit,
                                     nResamples, alpha):
    binned_spikes = binMultiTrialSpikes(spikes_times=spikes_times,
                                        neuron_index=neuron_index,
                                        trials_indices=trials_indices,
                                        epoch_times=epoch_times,
                                        bin_edges=bin_edges,
                                        time_unit=time_unit)
    psth = np.empty(len(bin_edges)-1, dtype=np.double)
    psth_ci = np.empty((len(bin_edges)-1, 2), dtype=np.double)
    for j in range(len(bin_edges)-1):
        psth[j] = binned_spikes[:, j].mean()
        bootstrapped_mean = stats.bootstrapTests.bootstrapMean(
            observations=binned_spikes[:, j], nResamples=nResamples)
        psth_ci[j, :] = stats.bootstrapTests.estimatePercentileCI(
            alpha=alpha, bootstrapped_stats=bootstrapped_mean
        )
    return binned_spikes, psth, psth_ci
