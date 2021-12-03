import torch

def clipSpikesTimes(spikes_times, from_time, to_time):
    nTrials = len(spikes_times)
    clipped_spikes_times = [[]] * nTrials
    for r in range(nTrials):
        clipped_spikes_times[r] = clipTrialSpikesTimes(trial_spikes_times=spikes_times[r],
                                                       from_time=from_time,
                                                       to_time=to_time)
    return clipped_spikes_times

def clipTrialSpikesTimes(trial_spikes_times, from_time, to_time):
    nNeurons = len(trial_spikes_times)
    clipped_trial_spikes_times = [[]] * nNeurons
    for n in range(nNeurons):
        clipped_trial_spikes_times[n] = clipNeuronSpikesTimes(
            neuron_spikes_times=trial_spikes_times[n],
            from_time=from_time, to_time=to_time)
    return clipped_trial_spikes_times

def clipNeuronSpikesTimes(neuron_spikes_times, from_time, to_time):
    clipped_neuron_spikes_times = neuron_spikes_times[torch.logical_and(
        from_time <= neuron_spikes_times, neuron_spikes_times < to_time)]
    return clipped_neuron_spikes_times
