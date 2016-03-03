#!/bin/bash

screen -Sdm server sh -c "source ./set4theano.sh; ./run_server.sh 'gpu7' 'cop8'; exec bash"
sleep 1 # wait for ompi-server.txt file to be created
screen -Sdm worker0 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu4' 'cop5'; exec bash"
screen -Sdm worker1 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu5' 'cop5'; exec bash"
screen -Sdm worker2 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu4' 'cop8'; exec bash"
screen -Sdm worker3 sh -c "source ./set4theano.sh; ./run_worker.sh 'gpu5' 'cop8'; exec bash"





