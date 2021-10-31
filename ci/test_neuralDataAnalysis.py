
import sys
import torch

sys.path.append("../src")
import utils.neuralDataAnalysis

def test_binSpikes():
    spikes = torch.arange(0.0, 9.99, 0.5)
    bin_edges = torch.arange(0.0, 10.01, 1.0)
    bin_counts, bin_centers = \
        utils.neuralDataAnalysis.binSpikes(spikes=spikes, bin_edges=bin_edges)
    for i in range(len(bin_counts)):
        assert(bin_counts[i]==2)
        assert(bin_centers[i]==i+.5)

if __name__ == "__main__":
    test_binSpikes()

