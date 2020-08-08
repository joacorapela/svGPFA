import pdb
import numpy as np

class PermutationTestResult:
    def __init__(self, t0, t):
        self.t0 = t0
        self.t = t

def permuteDiffPairedMeans(data1, data2, nResamples):
    def computeMeanDiffPairedMeans(data):
        data1 = data[:int(len(data)/2)]
        data2 = data[int(len(data)/2):]
        meanDiff = np.mean(data1-data2)
        return meanDiff

    data0 = np.concatenate((data1, data2))
    t0 = computeMeanDiffPairedMeans(data=data0)
    t = np.empty(nResamples)
    for i in range(nResamples):
        t[i] = computeMeanDiffPairedMeans(data=data0[np.random.randint(low=0, high=len(data0), size=len(data0))])
    # p = float(len(np.nonzero(np.absolute(t)>abs(t0))[0]))/len(t)
    answer = PermutationTestResult(t0=t0, t=t)
    return answer

