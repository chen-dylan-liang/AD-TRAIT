#!/bin/bash
rm -r autodiff_benchmark_results.csv
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j8
./AD
cd ..
