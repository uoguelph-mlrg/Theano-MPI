source /opt/sharcnet/testing/set4theano.sh
sqsub -q gpu -f mpi -n 16 -r 7d -o %J.out --gpp=8 --mpp=90g --nompirun ./run_bsp_workers.sh 8