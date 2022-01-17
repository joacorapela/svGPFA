#!/bin/bash

for i in {0..14}
do
    echo Plotting spikes for trial $i
    python ./doPlotShenoyData.py --trial $i
done
