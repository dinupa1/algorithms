#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

if [ -d "build" ]; then
	rm -rf build
fi


if [ -d "plots" ]; then
	rm -rf plots
fi

mkdir build
mkdir plots

cd build
root -e 'gSystem->CompileMacro("../src/Unfold.cc", "kfgO", "lib_unfold")' -q
cd ..

