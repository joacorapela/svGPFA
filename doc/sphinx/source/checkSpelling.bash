#!/bin/bash

rstFiles="highLevelInterface.rst  index.rst introduction.rst lowLevelInterface.rst params.rst references.rst svGPFA.plot.rst svGPFA.rst svGPFA.stats.rst svGPFA.utils.rst"

for rstFile in $rstFiles; do
    aspell -p ./svGPFA_doc.dic check $rstFile
done
