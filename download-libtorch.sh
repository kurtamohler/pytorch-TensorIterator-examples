#!/bin/bash
set -e

LIBTORCH_ZIP=libtorch-shared-with-deps-latest.zip
wget https://download.pytorch.org/libtorch/nightly/cpu/$LIBTORCH_ZIP
unzip $LIBTORCH_ZIP
rm -f $LIBTORCH_ZIP
