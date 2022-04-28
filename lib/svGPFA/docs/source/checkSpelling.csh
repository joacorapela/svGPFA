#!/bin/csh

aspell --personal=./svGPFA_doc.dic check introduction.rst 
aspell --personal=./svGPFA_doc.dic check highLevelInterface.rst 
aspell --personal=./svGPFA_doc.dic check lowLevelInterface.dic
