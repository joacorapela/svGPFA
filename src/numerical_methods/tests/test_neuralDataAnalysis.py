
import sys
import numpy as np

sys.path.append("../src")
import utils.neuralDataAnalysis

def test_binSpikes():
    spikes_times = np.arange(0.0, 9.99, 0.5)
    bins_edges = np.arange(0.0, 10.01, 1.0)
    binned_spikes = \
        utils.neuralDataAnalysis.binSpikes(spikes_times=spikes_times,
                                           bins_edges=bins_edges,
                                           time_unit="sec")
    for i in range(len(binned_spikes)):
        assert(binned_spikes[i] == 2)

if __name__ == "__main__":
    test_binSpikes()

