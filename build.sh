#!/bin/bash

set -e

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=$(pwd)/../libtorch ..
cmake --build . --config Release
