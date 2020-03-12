#!/bin/csh

set oldPattern = _simulation
set newPattern = _simulation_

foreach oldName ($*)
    set newName = `echo $oldName | sed "s/$oldPattern/$newPattern/"`
    mv $oldName $newName
end

