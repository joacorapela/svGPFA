
import numpy as np
import stats.bootstrapTests

def computePSTH(spikes_times, bin_edges):
    # spike_times in msec
    bin_width = bin_edges[1]-bin_edges[0]
    psth, _ = np.histogram(a=spikes_times, bins=bin_edges)
    psth = psth.astype(float)
    psth *= 1000.0/bin_width
    return psth

def computePSTHs(spikes_times, neuron_index, trials_indices, epoch_times,
                 bin_edges):
    psths = np.empty((len(trials_indices), len(bin_edges)-1), dtype=np.double)
    for i, trial_index in enumerate(trials_indices):
        psth = computePSTH(
            spikes_times=spikes_times[trial_index][neuron_index]-epoch_times[i],
            bin_edges=bin_edges
        )
        psths[i,:] = psth
    return psths

def computePSTHsAndMeans(spikes_times, neuron_index, trials_indices, epoch_times,
                         bin_edges):
    psths = computePSTHs(spikes_times=spikes_times, neuron_index=neuron_index,
                         trials_indices=trials_indices, epoch_times=epoch_times,
                         bin_edges=bin_edges)
    psth_mean = np.empty(len(bin_edges)-1, dtype=np.double)
    for j in range(len(bin_edges)-1):
        psth_mean[j] = psths[:,j].mean()
#         if psth_mean[j]>0 or psth_ci[j, 0]>0 or psth_ci[j, 1]>0:
#             import pdb; pdb.set_trace()
    return psths, psth_mean

def computePSTHsAndMeansCIs(spikes_times, neuron_index, trials_indices,
                            epoch_times, bin_edges, nResamples, alpha):
    psths = computePSTHs(spikes_times=spikes_times, neuron_index=neuron_index,
                         trials_indices=trials_indices, epoch_times=epoch_times,
                         bin_edges=bin_edges)
    psth_mean = np.empty(len(bin_edges)-1, dtype=np.double)
    psth_ci = np.empty((len(bin_edges)-1, 2), dtype=np.double)
    for j in range(len(bin_edges)-1):
        psth_mean[j] = psths[:,j].mean()
        bootstrapMeans = stats.bootstrapTests.bootstrapMean(
            observations=psths[:,j], nResamples=nResamples)
        psth_ci[j,:] = stats.bootstrapTests.estimatePercentileCI(
            alpha=alpha, bootstrapped_stats=bootstrapMeans
        )
#         if psth_mean[j]>0 or psth_ci[j, 0]>0 or psth_ci[j, 1]>0:
#             import pdb; pdb.set_trace()
    return psths, psth_mean, psth_ci

