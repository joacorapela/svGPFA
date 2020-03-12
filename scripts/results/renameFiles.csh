#!/bin/csh

set oldPattern = _spikeTimes_latents
set newPattern = _simulation_spikeTimes

foreach oldName ($*)
    set newName = `echo $oldName | sed "s/$oldPattern/$newPattern/"`
    mv $oldName $newName
end

