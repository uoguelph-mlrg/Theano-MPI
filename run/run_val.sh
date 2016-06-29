source set4theano.sh
mpirun --mca mpi_warn_on_fork 0 --bind-to none -n 8 python -u ../lib/validate_worker.py