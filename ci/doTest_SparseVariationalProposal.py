
import sys
from test_SparseVariationalProposal import test_getMeanAndVarianceAtQuadPoints
from test_SparseVariationalProposal_predict_MultiOutputGP import test_SparseVariationalProposal_predict_MultiOutputGP

def main(argv):
    test_SparseVariationalProposal()
    test_SparseVariationalProposal_predict_MultiOutputGP()

if __name__=="__main__":
    main(sys.argv)
