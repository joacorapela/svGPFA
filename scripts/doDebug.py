import pdb
import sys
import pickle
sys.path.append("../src")

def main(argv):
    with open("results/52060814_estimatedModel.pickle", "rb") as f:
        maxRes = pickle.load(f)
    model = maxRes["model"]
    pdb.set_trace()
    model.buildKernelsMatrices()
    model.eval()
    print("About to finish ...")
    pdb.set_trace()
    # break ../src/stats/svGPFA/klDivergence.py:41
if __name__=="__main__":
    main(sys.argv)
