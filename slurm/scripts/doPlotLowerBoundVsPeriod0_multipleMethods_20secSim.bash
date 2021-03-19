#!/bin/bash

ipython --pdb doPlotMultipleSetsOfLowerBoundsVsCovariate.py -- \
    --pythonDescriptorsFilename ../data/descriptors_20secSim.csv \
    --matlabDescriptorsFilename ../data/descriptors_20secSim.csv \
    --labelsAndEstNumbersFilenames ../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv,../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv,../data/labelsEstNumbers_20secSim_SciPy_trust-ncg.csv,../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv \
    --modelSetsTypes Python,Python,Python,Matlab \
    --modelSetsLegendLabels "PyTorch LBFGS,SciPy L-BFGS-B,SciPy trust-ncg,Matlab minFunc" \
    --pythonModelFilenamePattern ../../scripts/{:s} \
    --matlabModelFilenamePattern ../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat \
    --convergenceTolerance 1e-5 \
    --xlab "Period Parameter Initial Condition" \
    --ylab "Lower Bound" \
    --figFilenamePattern ../figures/20secSim_lowerBoundVsPeriod0_multipleMethods.{:s}
