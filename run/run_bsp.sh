#!/bin/bash

name="BSP$1"

screen -Sdm $name sh -c "source ./set4theano.sh; ./run_bsp_workers.sh '$1'; exec bash"

screen -r $name