#!/bin/bash

screen -Sdm server sh -c "source ./set4theano.sh; ./run_server.sh 'gpu4' 'cop5'; exec bash"
sleep 1 # wait for ompi-server.txt file to be created
screen -Sdm worker0 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu0' 'cop5'; exec bash"
screen -Sdm worker1 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu1' 'cop5'; exec bash"
screen -Sdm worker2 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu2' 'cop5'; exec bash"
screen -Sdm worker3 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu3' 'cop5'; exec bash"





