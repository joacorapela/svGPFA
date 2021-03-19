#!/bin/bash

ipython --pdb doPlotPeriodicKernelParamsVsCovariate.py -- \
    --periodicKernelParamToPlot "Lengthscale" \
    --generativeParam 2.25 \
    --pythonDescriptorsFilename ../data/descriptors_20secSim.csv \
    --matlabDescriptorsFilename ../data/descriptors_20secSim.csv \
    --labelsAndEstNumbersFilenames ../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv,../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv,../data/labelsEstNumbers_20secSim_SciPy_trust-ncg.csv,../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv,../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B_MAP.csv \
    --modelSetsTypes Python,Python,Python,Matlab,Python \
    --modelSetsLegendLabels "PyTorch LBFGS,SciPy L-BFGS-B,SciPy trust-ncg,Matlab LBFGS,SciPy L-BFGS-B MAP" \
    --paramScales 1.0,1.0,1.0,1.0,1.0 \
    --pythonModelFilenamePattern ../../scripts/{:s} \
    --matlabModelFilenamePattern ../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat \
    --xlab "Period Parameter Initial Condition" \
    --ylab "Estimated Lengthscale Parameter" \
    --figFilenamePattern ../figures/20secSim_estimatedLengthscaleVsPeriod0_multipleOptimMethods.{:s}
