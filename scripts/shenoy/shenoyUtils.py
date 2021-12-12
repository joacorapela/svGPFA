import numpy as np
import torch

def  getTrialAndLocationSpikesTimes(mat, trial, location, sRate=1000):
    mat_spike_times = mat["Rb"][trial, location]['unit']['spikeTimes']
    spikes_times = []
    nNeurons = mat_spike_times.shape[1]
    for j in range(nNeurons):
        spikes_times_set = (mat_spike_times[0,j].astype(np.float64)/sRate).squeeze()
        spikes_times.append(torch.from_numpy(spikes_times_set, dtype=torch.double))
    return spikes_times

def getTrialsAndLocationSpikesTimes(mat, trials, location, sRate=1000):
    spikesTimes = [[]]*len(trials)
    for i, trial in enumerate(trials):
        trial_spikes_times = getTrialAndLocationSpikesTimes(mat=mat, trial=trial, 
                                                            location=location,
                                                            sRate=sRate)
        spikesTimes[i] = trial_spikes_times
    return spikesTimes

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

def removeUnits(spikes_times, units_to_remove):
    nTrials = len(spikes_times)
    spikes_times_woUnits = [[]] * nTrials
    for r in range(nTrials):
        spikes_times_woUnits[r] = \
                removeUnitsFromTrial(trial_spikes_times=spikes_times[r],
                                     units_to_remove=units_to_remove)
    return spikes_times_woUnits

def removeUnitsFromTrial(trial_spikes_times, units_to_remove):
    nNeurons = len(trial_spikes_times)
    spikes_times_woUnits = []
    for n in range(nNeurons):
        if n not in units_to_remove:
            spikes_times_woUnits.append(trial_spikes_times[n])
    return spikes_times_woUnits
